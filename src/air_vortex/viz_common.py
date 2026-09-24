"""Shared helpers for plotting.py and animation.py: unit conversion,
staggered-to-center interpolation reuse, robust color limits, overlay data
(interface contour, stirrer forcing region, air-core masks), and figure
saving. Kept separate from plotting.py so animation.py can reuse the same
geometry/overlay computations without importing matplotlib.pyplot's
figure-oriented API.

Nothing here touches solver physics; it only reads already-computed fields
(from a live :class:`~air_vortex.fields.Fields` or a loaded .npz frame dict)
and reuses the existing diagnostics/connectivity functions so a figure's
notion of e.g. "vortex depth" or "air-core connected" never diverges from
the solver's own (README "diagnostic consistency" requirement).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence, Union

import numpy as np

from .config import Config
from .connectivity import air_mask, stirbar_target_region, top_connected_air
from .forcing import forcing_mask
from .grid import Grid
from .operators import interp_ur_to_center, interp_uz_to_center

SnapshotLike = Union[Mapping[str, Any], Any]


def field(snap: SnapshotLike, name: str) -> np.ndarray:
    """Read ``name`` off either a dict (loaded .npz frame) or a Fields-like
    object (attribute access) -- lets plotting/animation code accept both
    a live solver snapshot and a saved-and-reloaded one interchangeably."""
    if isinstance(snap, Mapping):
        return snap[name]
    return getattr(snap, name)


def has_field(snap: SnapshotLike, name: str) -> bool:
    if isinstance(snap, Mapping):
        return name in snap
    return getattr(snap, name, None) is not None


def scalar(snap: SnapshotLike, name: str) -> float:
    v = field(snap, name)
    return float(v)


def to_mm(x_m: np.ndarray | float) -> np.ndarray | float:
    return np.asarray(x_m) * 1e3


def meshgrid_mm(grid: Grid) -> tuple[np.ndarray, np.ndarray]:
    """(R, Z) meshgrid in mm, shape (Nr, Nz), matching cell-centered fields."""
    return np.meshgrid(to_mm(grid.r_c), to_mm(grid.z_c), indexing="ij")


def cell_center_velocity(grid: Grid, u_r: np.ndarray, u_z: np.ndarray
                          ) -> tuple[np.ndarray, np.ndarray]:
    """Staggered u_r (Nr+1,Nz) / u_z (Nr,Nz+1) -> cell-centered (Nr,Nz)
    velocity components, for plotting/quivers/streamlines. Thin wrapper
    around the existing operators so there is exactly one place that knows
    how staggered locations map to cell centers."""
    return interp_ur_to_center(u_r), interp_uz_to_center(u_z)


def meridional_speed(u_r_c: np.ndarray, u_z_c: np.ndarray) -> np.ndarray:
    return np.sqrt(u_r_c**2 + u_z_c**2)


def robust_clim(arrays: Sequence[np.ndarray], low: float = 1.0, high: float = 99.0,
                 symmetric: bool = False) -> tuple[float, float]:
    """Percentile-based color limits pooled across one or more arrays (e.g.
    a subsample of animation frames), so a single outlier cell can't wash
    out the whole color range (README Figure 4/Video B requirement)."""
    pooled = np.concatenate([np.asarray(a).ravel() for a in arrays])
    pooled = pooled[np.isfinite(pooled)]
    if pooled.size == 0:
        return (0.0, 1.0)
    lo, hi = np.percentile(pooled, [low, high])
    if symmetric:
        m = max(abs(lo), abs(hi))
        return (-m, m)
    if lo == hi:
        hi = lo + 1e-12
    return (float(lo), float(hi))


def stirrer_region_contour(grid: Grid, cfg: Config) -> np.ndarray:
    """chi(r,z) forcing mask (README section 10.1); callers typically draw
    its 0.5 contour as the "effective stirrer region" outline."""
    return forcing_mask(grid, cfg)


def air_core_masks(phi: np.ndarray, grid: Grid, cfg: Config) -> dict[str, np.ndarray]:
    """All the masks needed to visualize the connectivity criterion
    (README section 22): the full air mask, the top-connected component,
    the stirrer target region B_delta, reusing connectivity.py exactly."""
    a_mask = air_mask(phi)
    return {
        "all_air": a_mask.astype(bool),
        "top_connected_air": top_connected_air(a_mask),
        "stirbar_target": stirbar_target_region(grid, cfg),
    }


def save_figure(fig, paths, stem: str, dpi: int = 300, rasterized_pdf: bool = False) -> dict:
    """Save ``fig`` as both PNG (300 dpi) and PDF under a run's
    figures/{png,pdf}/ directories (README "Figure quality" section).
    ``paths`` is a :class:`~air_vortex.run_io.RunPaths` (or any object with
    ``figures_png_dir``/``figures_pdf_dir``)."""
    png_path = Path(paths.figures_png_dir) / f"{stem}.png"
    pdf_path = Path(paths.figures_pdf_dir) / f"{stem}.pdf"
    png_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    if rasterized_pdf:
        for ax in fig.axes:
            for artist in ax.get_children():
                if hasattr(artist, "set_rasterized"):
                    try:
                        artist.set_rasterized(True)
                    except Exception:
                        pass

    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    return {"png": png_path, "pdf": pdf_path}
