import numpy as np

from air_vortex.grid import build_grid
from air_vortex.fields import initialize_still_water
from air_vortex.run_io import (
    MissingFieldError,
    SnapshotWriter,
    grid_metadata,
    list_frames,
    load_run,
    run_paths,
    save_config,
    save_metadata,
)
from _config_helpers import small_config


def test_snapshot_roundtrip(tmp_path):
    cfg = small_config(rpm=0.0)
    grid = build_grid(cfg)
    fields = initialize_still_water(grid, cfg.geometry.water_height_m)
    fields.t = 0.123
    fields.step = 7
    fields.u_theta = np.full(grid.shape_center, 3.5)

    paths = run_paths(tmp_path, "roundtrip_run")
    writer = SnapshotWriter(paths)
    out_path = writer.write(fields)

    assert out_path.exists()
    assert writer.n_written == 1

    frames = list_frames(paths)
    assert len(frames) == 1
    assert frames[0].t == 0.123
    assert frames[0].step == 7

    loaded = frames[0].load()
    np.testing.assert_array_equal(loaded["phi"], fields.phi)
    np.testing.assert_array_equal(loaded["u_theta"], fields.u_theta)
    np.testing.assert_array_equal(loaded["u_r"], fields.u_r)
    np.testing.assert_array_equal(loaded["u_z"], fields.u_z)
    np.testing.assert_array_equal(loaded["p"], fields.p)
    np.testing.assert_array_equal(loaded["rho"], fields.rho)


def test_load_run_reconstructs_grid_from_config(tmp_path):
    cfg = small_config(rpm=500.0)
    grid = build_grid(cfg)
    paths = run_paths(tmp_path, "cfg_run")
    save_config(paths, cfg)
    save_metadata(paths, {"kind": "production", "rpm": 500.0, "grid": grid_metadata(grid)})

    run = load_run(paths.root)

    assert run.cfg is not None
    assert run.cfg.stirrer.rpm == 500.0
    assert run.grid is not None
    assert run.grid.Nr == grid.Nr
    assert run.grid.Nz == grid.Nz


def test_load_run_reconstructs_grid_from_metadata_without_config(tmp_path):
    cfg = small_config(rpm=0.0)
    grid = build_grid(cfg)
    paths = run_paths(tmp_path, "meta_only_run")
    save_metadata(paths, {"kind": "production", "grid": grid_metadata(grid)})
    # deliberately no config.yaml

    run = load_run(paths.root)

    assert run.cfg is None
    assert run.grid is not None
    assert run.grid.Nr == grid.Nr
    assert run.grid.dr == grid.dr


def test_require_raises_clear_error_naming_missing_pieces(tmp_path):
    paths = run_paths(tmp_path, "empty_run")
    run = load_run(paths.root)

    try:
        run.require("cfg", "grid", "frames")
        assert False, "expected MissingFieldError"
    except MissingFieldError as e:
        msg = str(e)
        assert "config.yaml" in msg
        assert "fields" in msg
