"""Shared helpers for building tiny on-disk run directories in tests."""
import numpy as np

from air_vortex.fields import initialize_still_water
from air_vortex.grid import build_grid
from air_vortex.run_io import SnapshotWriter, grid_metadata, run_paths, save_config, save_metadata

from _config_helpers import small_config


def make_synthetic_snapshot(grid, cfg, t=0.0, step=0):
    """A plain dict with the same keys a loaded .npz frame would have,
    for tests that don't need a real solver run."""
    fields = initialize_still_water(grid, cfg.geometry.water_height_m)
    return {
        "t": t, "step": step,
        "u_r": fields.u_r, "u_z": fields.u_z, "u_theta": fields.u_theta,
        "p": fields.p, "phi": fields.phi,
        "rho": np.full(grid.shape_center, cfg.fluid.water_density),
    }


def make_tiny_run(tmp_path, run_id="tiny_run", n_frames=3, rpm=0.0, kind="production"):
    """Writes a tiny run directory (config.yaml, metadata.json,
    diagnostics.csv, fields/frame_*.npz) using still-water snapshots with a
    slightly deepening synthetic depression, so consumers have something
    to plot/animate without running the real solver."""
    import pandas as pd

    cfg = small_config(rpm=rpm)
    grid = build_grid(cfg)
    paths = run_paths(tmp_path, run_id)
    save_config(paths, cfg)

    writer = SnapshotWriter(paths)
    rows = []
    for i in range(n_frames):
        t = i * 0.01
        fields = initialize_still_water(grid, cfg.geometry.water_height_m)
        # synthetic central depression, deepening with i, purely so the
        # frames differ from each other (no physics implied)
        depression = 0.001 * i
        central = grid.r_c <= cfg.geometry.stirbar_half_length_m
        fields.phi[central, :] += depression
        fields.t = t
        fields.step = i * 10

        writer.write(fields)
        rows.append({"t": t, "step": fields.step, "d": depression,
                      "max_u_theta": 0.0, "max_downward_uz": 0.0,
                      "min_pressure": 0.0, "water_volume": 1.0})

    pd.DataFrame(rows).to_csv(paths.diagnostics_path, index=False)
    save_metadata(paths, {
        "kind": kind, "rpm": rpm,
        "water_height_mm": cfg.geometry.water_height_m * 1e3,
        "grid": grid_metadata(grid),
    })
    return paths
