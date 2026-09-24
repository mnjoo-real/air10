import numpy as np

from air_vortex.connectivity import (
    ConnectivityState,
    is_geometrically_connected,
    stirbar_target_region,
    top_connected_air,
    update_persistence,
)
from air_vortex.diagnostics import find_tip_z, free_surface_height, vortex_depth
from air_vortex.grid import build_grid
from _config_helpers import small_config


def test_top_connected_air_excludes_isolated_island():
    mask = np.zeros((5, 5), dtype=np.uint8)
    mask[2, 3] = 1
    mask[2, 4] = 1  # touches the top column (index -1) -> connected
    mask[0, 0] = 1  # isolated, doesn't touch top, not 4-connected to the above

    result = top_connected_air(mask)

    expected = np.zeros((5, 5), dtype=bool)
    expected[2, 3] = True
    expected[2, 4] = True
    np.testing.assert_array_equal(result, expected)


def test_find_tip_z_and_vortex_depth_on_synthetic_depression():
    cfg = small_config(rpm=0.0)
    grid = build_grid(cfg)
    H = cfg.geometry.water_height_m
    R_m = cfg.geometry.stirbar_half_length_m

    # normal flat interface everywhere...
    phi = (grid.z_c[None, :] - H) * np.ones(grid.shape_center)
    # ...except a synthetic depression in the central (r<=R_m) columns,
    # reaching exactly down to grid.z_f[5].
    central = grid.r_c <= R_m
    assert central.sum() >= 1
    phi[central, :5] = -1.0
    phi[central, 5:] = 1.0

    z_tip = find_tip_z(phi, grid, cfg)
    assert z_tip == grid.z_f[5]

    d = vortex_depth(phi, grid, cfg)
    assert abs(d - (H - grid.z_f[5])) < 1e-12


def test_is_geometrically_connected_true_when_air_reaches_target_region():
    cfg = small_config(rpm=0.0)
    grid = build_grid(cfg)
    H = cfg.geometry.water_height_m

    b_delta = stirbar_target_region(grid, cfg)
    assert np.any(b_delta)

    phi = (grid.z_c[None, :] - H) * np.ones(grid.shape_center)
    central = grid.r_c <= cfg.geometry.stirbar_half_length_m
    phi[central, :] = 1.0  # air all the way down to the bottom, in the central columns

    assert is_geometrically_connected(phi, grid, cfg) is True


def test_is_geometrically_connected_false_when_air_core_is_shallow():
    cfg = small_config(rpm=0.0)
    grid = build_grid(cfg)
    H = cfg.geometry.water_height_m

    phi = (grid.z_c[None, :] - H) * np.ones(grid.shape_center)
    central = grid.r_c <= cfg.geometry.stirbar_half_length_m
    # air only near the top half -- never reaches the stirrer target region
    phi[central, :6] = -1.0
    phi[central, 6:] = 1.0

    assert is_geometrically_connected(phi, grid, cfg) is False


def test_persistence_requires_sustained_contact_and_resets_on_dropout():
    cfg = small_config(rpm=0.0)
    N = 1200.0
    required_s = cfg.air_core.persistence_rotations * 60.0 / N  # = 0.25 s here

    state = ConnectivityState(connected=False)
    state = update_persistence(state, True, 0.0, cfg, N)
    assert state.connected is False

    state = update_persistence(state, True, required_s * 0.5, cfg, N)
    assert state.connected is False

    state = update_persistence(state, True, required_s * 1.1, cfg, N)
    assert state.connected is True

    state = update_persistence(state, False, required_s * 1.2, cfg, N)
    assert state.connected is False
    assert state.t_connected_start is None


def test_find_tip_z_subgrid_interpolation_is_grid_independent():
    """A flat interface at a known off-grid height z0 must be recovered to
    high precision by find_tip_z regardless of grid spacing -- the
    original implementation instead snapped to the nearest grid face,
    making the reported depth jump by ~dx between resolutions."""
    z0 = 0.01237  # 12.37 mm, deliberately not aligned to any grid used below

    for dx in (0.002, 0.001, 0.0005):
        cfg = small_config(rpm=0.0, dr=dx, dz=dx)
        grid = build_grid(cfg)
        # flat interface at z0 everywhere: water (phi<0) below, air above
        phi = grid.z_c[None, :] - z0
        phi = phi * np.ones(grid.shape_center)

        z_tip = find_tip_z(phi, grid, cfg)

        assert abs(z_tip - z0) < 1e-9, f"dx={dx}: z_tip={z_tip} vs z0={z0}"


def test_find_tip_z_matches_free_surface_height_at_axis():
    """For a flat, fully top-connected interface, find_tip_z (restricted to
    r<=R_m) and free_surface_height at the axis column should agree, since
    both use the same interpolation helper on the same data."""
    cfg = small_config(rpm=0.0, dr=0.001, dz=0.001)
    grid = build_grid(cfg)
    z0 = grid.z_c[4] + 0.37 * grid.dz  # arbitrary off-grid height inside the domain
    phi = (grid.z_c[None, :] - z0) * np.ones(grid.shape_center)

    z_tip = find_tip_z(phi, grid, cfg)
    eta = free_surface_height(phi, grid)

    assert abs(z_tip - eta[0]) < 1e-9


def test_volume_consistent_parabola_integrates_to_target_volume():
    from air_vortex.diagnostics import volume_consistent_parabola

    R_v, omega, g, H = 0.045, 94.25, 9.81, 0.05
    target = np.pi * R_v**2 * H
    r = np.linspace(0, R_v, 200_000)
    eta = volume_consistent_parabola(r, R_v, omega, g, target)

    V_numeric = np.trapezoid(2 * np.pi * r * eta, r)
    assert abs(V_numeric - target) / target < 1e-6
