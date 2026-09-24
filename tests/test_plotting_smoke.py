"""Smoke tests for the result-visualization plot functions: each is called
with small synthetic data and must return a Figure that can be saved,
without raising. These are not visual-regression tests."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from air_vortex.grid import build_grid
from air_vortex.plotting import (
    plot_azimuthal_velocity,
    plot_critical_scaling,
    plot_depth_scaling,
    plot_hydrostatic_validation,
    plot_interface_snapshots,
    plot_meridional_flow,
    plot_pressure_interface,
    plot_regime_map,
    plot_solid_body_validation,
    plot_vortex_depth_time,
)
from _config_helpers import small_config
from _run_helpers import make_synthetic_snapshot


def _assert_valid_figure(fig, tmp_path, name):
    assert fig is not None
    assert len(fig.axes) >= 1
    out = tmp_path / f"{name}.png"
    fig.savefig(out)
    assert out.exists() and out.stat().st_size > 0
    plt.close(fig)


def test_plot_hydrostatic_validation_smoke(tmp_path):
    cfg = small_config(rpm=0.0)
    grid = build_grid(cfg)
    snap = make_synthetic_snapshot(grid, cfg)
    fig = plot_hydrostatic_validation(grid, cfg, snap)
    _assert_valid_figure(fig, tmp_path, "hydrostatic")


def test_plot_solid_body_validation_smoke(tmp_path):
    cfg = small_config(rpm=0.0)
    grid = build_grid(cfg)
    snap = make_synthetic_snapshot(grid, cfg)
    fig = plot_solid_body_validation(grid, cfg, snap, omega=5.0)
    _assert_valid_figure(fig, tmp_path, "solid_body")


def test_plot_interface_snapshots_smoke(tmp_path):
    cfg = small_config(rpm=0.0)
    grid = build_grid(cfg)
    snapshots = [(i * 0.1, make_synthetic_snapshot(grid, cfg, t=i * 0.1)) for i in range(4)]
    fig = plot_interface_snapshots(grid, cfg, snapshots, rpm=900)
    _assert_valid_figure(fig, tmp_path, "interface_snapshots")


def test_plot_pressure_interface_smoke(tmp_path):
    cfg = small_config(rpm=0.0)
    grid = build_grid(cfg)
    snap = make_synthetic_snapshot(grid, cfg)
    fig = plot_pressure_interface(grid, cfg, snap)
    _assert_valid_figure(fig, tmp_path, "pressure_interface")


def test_plot_meridional_flow_smoke(tmp_path):
    cfg = small_config(rpm=0.0)
    grid = build_grid(cfg)
    snap = make_synthetic_snapshot(grid, cfg)
    fig = plot_meridional_flow(grid, cfg, snap)
    _assert_valid_figure(fig, tmp_path, "meridional_flow")


def test_plot_azimuthal_velocity_smoke(tmp_path):
    cfg = small_config(rpm=0.0)
    grid = build_grid(cfg)
    snap = make_synthetic_snapshot(grid, cfg)
    snap["u_theta"] = 3.0 * grid.r_c[:, None] * np.ones(grid.shape_center)
    fig = plot_azimuthal_velocity(grid, cfg, snap)
    _assert_valid_figure(fig, tmp_path, "azimuthal_velocity")


def test_plot_vortex_depth_time_smoke(tmp_path):
    df = pd.DataFrame({"t": np.linspace(0, 1, 10), "d": np.linspace(0, 0.01, 10)})
    fig = plot_vortex_depth_time({"900 rpm": df})
    _assert_valid_figure(fig, tmp_path, "vortex_depth_time")


def test_plot_depth_scaling_smoke(tmp_path):
    N = np.array([500.0, 900.0, 1300.0])
    d_inf = np.array([0.001, 0.003, 0.006])
    fig = plot_depth_scaling(N, d_inf)
    _assert_valid_figure(fig, tmp_path, "depth_scaling")

    fig2 = plot_depth_scaling(N, d_inf, dimensionless=True, R_m=0.015)
    _assert_valid_figure(fig2, tmp_path, "depth_scaling_dimless")


def test_plot_critical_scaling_smoke(tmp_path):
    H = np.array([0.02, 0.03, 0.04, 0.05])
    N_c = np.array([600.0, 750.0, 880.0, 1000.0])
    fig = plot_critical_scaling(H, N_c)
    _assert_valid_figure(fig, tmp_path, "critical_scaling")

    fig2 = plot_critical_scaling(H, N_c, dimensionless=True, R_m=0.015)
    _assert_valid_figure(fig2, tmp_path, "critical_scaling_dimless")


def test_plot_regime_map_smoke(tmp_path):
    df = pd.DataFrame({
        "Fr_Omega": [1.0, 5.0, 10.0, 15.0],
        "H_over_Rm": [2.0, 2.0, 3.0, 3.0],
        "regime": ["no_core", "transient_contact", "stable_core", "stable_core"],
    })
    fig = plot_regime_map(df)
    _assert_valid_figure(fig, tmp_path, "regime_map")
