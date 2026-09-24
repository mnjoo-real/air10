"""Run-directory storage layer for post-processing.

This module is purely an I/O convenience layer on top of the existing
:class:`~air_vortex.fields.Fields` / :class:`~air_vortex.config.Config` /
:class:`~air_vortex.grid.Grid` objects -- it does not touch solver physics.
It exists so a simulation run's results can be visualized (figures, videos)
without re-running the solver.

Layout for a run at ``results/<run_id>/``::

    config.yaml           # full Config, see config.config_to_dict
    metadata.json         # run-level metadata: kind, rpm, water_height_mm,
                           # grid shape, created_at, ...
    diagnostics.csv        # one row per SOLVER step (see diagnostics.py)
    fields/
        frame_000000.npz   # one row per OUTPUT SAMPLE (output.save_every_s),
        frame_000001.npz   # NOT one per solver step
        ...
        index.csv           # index, t, step, path -- avoids opening every
                             # frame just to know when it was taken
    figures/
        png/
        pdf/
    videos/

Backward compatibility (see README "Result Visualization"): a run missing
some of these pieces (e.g. no metadata.json, or a frame missing "mu") is
not a hard error -- callers ask for what they need and get a clear
``MissingFieldError`` naming exactly what is absent, so a caller that only
needs a subset (e.g. just diagnostics.csv for Figure 7) can proceed even
if fields/ is incomplete.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import yaml

from .config import Config, config_from_dict, config_to_dict
from .fields import Fields
from .grid import Grid, build_grid


class MissingFieldError(RuntimeError):
    """Raised when requested data is not present in a run directory."""


@dataclass
class RunPaths:
    root: Path

    @property
    def config_path(self) -> Path:
        return self.root / "config.yaml"

    @property
    def metadata_path(self) -> Path:
        return self.root / "metadata.json"

    @property
    def diagnostics_path(self) -> Path:
        return self.root / "diagnostics.csv"

    @property
    def fields_dir(self) -> Path:
        return self.root / "fields"

    @property
    def frames_index_path(self) -> Path:
        return self.fields_dir / "index.csv"

    @property
    def figures_png_dir(self) -> Path:
        return self.root / "figures" / "png"

    @property
    def figures_pdf_dir(self) -> Path:
        return self.root / "figures" / "pdf"

    @property
    def videos_dir(self) -> Path:
        return self.root / "videos"

    def ensure(self) -> "RunPaths":
        for d in (self.root, self.fields_dir, self.figures_png_dir,
                  self.figures_pdf_dir, self.videos_dir):
            d.mkdir(parents=True, exist_ok=True)
        return self


def run_paths(results_root: str | Path, run_id: str) -> RunPaths:
    return RunPaths(Path(results_root) / run_id).ensure()


def save_config(paths: RunPaths, cfg: Config) -> None:
    with open(paths.config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config_to_dict(cfg), f, sort_keys=False)


def save_metadata(paths: RunPaths, metadata: dict) -> None:
    with open(paths.metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)


def load_metadata(paths: RunPaths) -> dict:
    if not paths.metadata_path.exists():
        return {}
    with open(paths.metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)


def grid_metadata(grid: Grid) -> dict:
    """Grid shape/spacing, stored in metadata.json so a run can be
    reloaded (frames + regridding) even without config.yaml."""
    return {
        "Nr": grid.Nr, "Nz": grid.Nz,
        "dr_m": grid.dr, "dz_m": grid.dz,
        "vessel_radius_m": grid.r_v, "z_max_m": grid.z_max,
    }


def grid_from_metadata(meta: dict) -> Grid:
    g = meta["grid"]
    dr, dz = g["dr_m"], g["dz_m"]
    Nr, Nz = g["Nr"], g["Nz"]
    r_f = np.linspace(0.0, Nr * dr, Nr + 1)
    z_f = np.linspace(0.0, Nz * dz, Nz + 1)
    r_c = 0.5 * (r_f[:-1] + r_f[1:])
    z_c = 0.5 * (z_f[:-1] + z_f[1:])
    return Grid(dr=dr, dz=dz, r_v=g["vessel_radius_m"], z_max=g["z_max_m"],
                 Nr=Nr, Nz=Nz, r_c=r_c, z_c=z_c, r_f=r_f, z_f=z_f)


class SnapshotWriter:
    """Writes one compressed .npz per saved frame, plus a lightweight
    index (t, step, path) so readers don't need to open every frame just
    to know when it was taken. Call once per OUTPUT sample, not once per
    solver step -- the caller (e.g. scripts/run_single.py) is responsible
    for the ``output.save_every_s`` decimation, matching the existing
    config schema (README section 29)."""

    def __init__(self, paths: RunPaths):
        self.paths = paths
        self._index_rows: list[tuple[int, float, int, str]] = []
        self._next_index = 0

    @property
    def n_written(self) -> int:
        return self._next_index

    def write(self, fields: Fields, extra: Optional[dict] = None) -> Path:
        idx = self._next_index
        rel_path = f"frame_{idx:06d}.npz"
        out_path = self.paths.fields_dir / rel_path

        data = {
            "t": fields.t, "step": fields.step,
            "u_r": fields.u_r, "u_z": fields.u_z, "u_theta": fields.u_theta,
            "p": fields.p, "phi": fields.phi, "rho": fields.rho,
        }
        if fields.mu is not None and fields.mu.size:
            data["mu"] = fields.mu
        if extra:
            data.update(extra)

        np.savez_compressed(out_path, **data)
        self._index_rows.append((idx, fields.t, fields.step, rel_path))
        self._next_index += 1
        self._flush_index()
        return out_path

    def _flush_index(self) -> None:
        with open(self.paths.frames_index_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["index", "t", "step", "path"])
            writer.writerows(self._index_rows)


@dataclass
class FrameRef:
    """A frame's index-row metadata, without loading its arrays."""
    index: int
    t: float
    step: int
    path: Path

    def load(self) -> dict:
        """Load this frame's arrays. ``np.load`` on an .npz only decompresses
        an array when that key is actually indexed, so callers that only
        need e.g. ``phi`` do not pay for the rest -- this is the "lazy
        loading" the frame reader relies on rather than loading every array
        of every frame up front."""
        with np.load(self.path) as npz:
            return {k: npz[k] for k in npz.files}


def _build_frames_index(paths: RunPaths) -> list[FrameRef]:
    """Fallback for a fields/ directory without index.csv (e.g. hand-copied
    frames): globs the .npz files and opens each just far enough to read
    its scalar t/step -- slower than the index but still robust."""
    refs = []
    for i, p in enumerate(sorted(paths.fields_dir.glob("frame_*.npz"))):
        with np.load(p) as npz:
            t = float(npz["t"]) if "t" in npz.files else float("nan")
            step = int(npz["step"]) if "step" in npz.files else -1
        refs.append(FrameRef(index=i, t=t, step=step, path=p))
    return refs


def list_frames(paths: RunPaths) -> list[FrameRef]:
    if paths.frames_index_path.exists():
        refs = []
        with open(paths.frames_index_path, "r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                refs.append(FrameRef(
                    index=int(row["index"]), t=float(row["t"]), step=int(row["step"]),
                    path=paths.fields_dir / row["path"],
                ))
        return refs
    if paths.fields_dir.exists() and any(paths.fields_dir.glob("frame_*.npz")):
        return _build_frames_index(paths)
    return []


@dataclass
class RunData:
    """Everything post-processing needs about one run, loaded lazily where
    possible (frame arrays are only read on :meth:`load_frame`/`FrameRef.load`)."""
    paths: RunPaths
    cfg: Optional[Config]
    metadata: dict
    grid: Optional[Grid]
    frames: list[FrameRef] = field(default_factory=list)

    @property
    def n_frames(self) -> int:
        return len(self.frames)

    def load_frame(self, i: int) -> dict:
        return self.frames[i].load()

    def require(self, *names: str) -> None:
        """Raise a clear error naming exactly what is missing, instead of a
        bare KeyError/AttributeError deep inside a plotting function
        (README "Result Visualization" backward-compatibility requirement)."""
        missing = []
        if "cfg" in names and self.cfg is None:
            missing.append("config.yaml")
        if "grid" in names and self.grid is None:
            missing.append("config.yaml or metadata.json grid info")
        if "frames" in names and not self.frames:
            missing.append("fields/*.npz")
        if "diagnostics" in names and self.diagnostics_path_missing():
            missing.append("diagnostics.csv")
        if missing:
            raise MissingFieldError(
                f"run '{self.paths.root}' is missing: {', '.join(missing)}"
            )

    def diagnostics_path_missing(self) -> bool:
        return not self.paths.diagnostics_path.exists()

    def load_diagnostics(self):
        import pandas as pd
        if self.diagnostics_path_missing():
            raise MissingFieldError(
                f"run '{self.paths.root}' has no diagnostics.csv"
            )
        return pd.read_csv(self.paths.diagnostics_path)


def load_run(run_dir: str | Path) -> RunData:
    """Load a run directory into a :class:`RunData`. Never raises for
    merely-missing optional pieces (config, metadata, diagnostics, frames)
    -- callers that need a given piece call :meth:`RunData.require` (or a
    loader method) and get a :class:`MissingFieldError` naming exactly
    what's absent, per the backward-compatibility requirement."""
    paths = RunPaths(Path(run_dir))
    metadata = load_metadata(paths)

    cfg: Optional[Config] = None
    if paths.config_path.exists():
        with open(paths.config_path, "r", encoding="utf-8") as f:
            cfg = config_from_dict(yaml.safe_load(f))

    grid: Optional[Grid] = None
    if cfg is not None:
        grid = build_grid(cfg)
    elif "grid" in metadata:
        grid = grid_from_metadata(metadata)

    frames = list_frames(paths)

    return RunData(paths=paths, cfg=cfg, metadata=metadata, grid=grid, frames=frames)
