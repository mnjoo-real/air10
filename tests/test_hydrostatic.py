"""Milestone 1 (README section 43): stationary water remains stationary
and conserves volume; pressure matches the hydrostatic profile (Test 1,
README section 31)."""
import numpy as np

from air_vortex.solver import build_solver, water_volume
from _config_helpers import small_config


def test_still_water_stays_still_and_conserves_volume():
    cfg = small_config(rpm=0.0)
    solver = build_solver(cfg, swirl_mode="forced")

    V0 = water_volume(solver.grid, solver.fields.phi)

    for _ in range(20):
        solver.step()

    max_speed = max(
        np.max(np.abs(solver.fields.u_r)),
        np.max(np.abs(solver.fields.u_z)),
    )
    assert max_speed < 1e-3, f"still water developed spurious velocity: {max_speed}"

    V1 = water_volume(solver.grid, solver.fields.phi)
    rel_drift = abs(V1 - V0) / V0
    assert rel_drift < 0.01, f"volume drift too large: {rel_drift:.4f}"


def test_hydrostatic_pressure_profile():
    # A finer grid than the other milestone-1 tests: the smoothed density
    # transition at the interface (README section 9.1) introduces an O(dx)
    # deviation from the sharp-interface hydrostatic formula right near the
    # interface itself; cells well below it converge to the analytic
    # profile as the grid is refined (verified separately), but need
    # adequate resolution for that convergence to have "kicked in" yet.
    cfg = small_config(rpm=0.0, dr=0.0015, dz=0.0015)
    solver = build_solver(cfg, swirl_mode="forced")

    for _ in range(20):
        solver.step()

    grid = solver.grid
    H = cfg.geometry.water_height_m
    rho_w = cfg.fluid.water_density
    g = cfg.fluid.gravity

    i_mid = grid.Nr // 2
    for j in range(grid.Nz):
        z = grid.z_c[j]
        if z >= H - 4 * grid.dz:
            continue  # skip cells inside/near the smoothed interface band
        p_expected = rho_w * g * (H - z)
        p_actual = solver.fields.p[i_mid, j]
        assert abs(p_actual - p_expected) < 0.1 * p_expected + 1.0, (
            f"z={z:.4f}: expected {p_expected:.2f} Pa, got {p_actual:.2f} Pa"
        )
