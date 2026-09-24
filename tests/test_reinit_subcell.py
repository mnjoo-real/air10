"""Gate V5b: interface-preserving reinitialization (Russo-Smereka subcell
fix, Min-Gibou dimension-by-dimension form). Primary metric: the phi=0
contour does not move; secondary: |grad phi| -> 1."""
import copy

import numpy as np
import pytest

from air_vortex.levelset import advect_level_set_advective, reinitialize_level_set
from air_vortex.liquid_mask import crossing_fraction, liquid_volume_subcell
from air_vortex.reinit_benchmarks import CASES, r3_circle, uniform_grid
from air_vortex.reinit_diagnostics import contour_displacement, signed_distance_error
from air_vortex.reinit_subcell import quadratic_crossing_fraction, reinitialize_subcell
from air_vortex.solver import build_solver
from _config_helpers import single_phase_config, small_config


def _repeat(fn, phi, n):
    for _ in range(n):
        phi = fn(phi)
    return phi


# ------------------------------------------------------------- building blocks
def test_quadratic_crossing_exact_for_quadratic_and_falls_back_to_linear():
    x = np.array([-1.0, 0.0, 1.0, 2.0])            # nodes i-1, i, i+1, i+2
    root = 0.37
    q = (x - root) * (x + 3.0)                      # exact quadratic, root in (0, 1)
    th = quadratic_crossing_fraction(q[1], q[2], q[0], q[3])
    assert th == pytest.approx(root, abs=1e-12)
    lin = 0.6 * x - 0.2                             # linear: quadratic term is 0
    th = quadratic_crossing_fraction(lin[1], lin[2], lin[0], lin[3])
    assert th == pytest.approx(float(crossing_fraction(lin[1], lin[2])), abs=1e-15)


# ------------------------------------------------------------------ R1 plane
@pytest.mark.parametrize("dx", [1.5e-3, 1.0e-3])
def test_R1_offgrid_plane_linear_phi_preserved_to_roundoff(dx):
    """phi0 linear in z with |grad phi| != 1 and r-dependent scale.

    A call CONVERGED in pseudo-time keeps z0 to round-off. With truncated
    calls (5 iterations), the interface-adjacent nodes of differently
    scaled columns are still relaxing at different rates when the call
    ends, and the next call re-reads that intermediate crossing as its
    frozen phi0: the truncation error is locked in (measured ~1e-4 dx),
    not accumulated."""
    g = uniform_grid(dx)
    R, Z = np.meshgrid(g.r_c, g.z_c, indexing="ij")
    z0 = 0.0173137
    assert abs((z0 / dx) - round(z0 / dx)) > 0.05          # off-grid
    phi0 = (Z - z0) * (1.2 + 0.6 * np.sin(2 * np.pi * R / g.r_v))
    for order in (1, 2):
        conv = reinitialize_subcell(phi0, g, 200, order=order)
        assert contour_displacement(g, phi0, conv).max_shift_dx < 1e-10
        assert signed_distance_error(g, conv) < 1e-8
        p5 = reinitialize_subcell(phi0, g, 5, order=order)
        p100 = _repeat(lambda p: reinitialize_subcell(p, g, 5, order=order), p5, 99)
        assert contour_displacement(g, phi0, p100).max_shift_dx < 2e-4
        assert contour_displacement(g, p5, p100).max_shift_dx < 2e-5    # locked in, no growth


@pytest.mark.parametrize("dx", [1.5e-3, 1.0e-3])
def test_R1_distorted_plane_repeated_calls(dx):
    """Nonlinear phi0: RS order 1 pins phi0's own linear crossings (shift
    ~0); order 2 pins the more accurate quadratic crossing and so REDUCES
    the error against the exact plane; neither drifts with repeated calls."""
    g = uniform_grid(dx)
    case = CASES["R1"](g)
    geo0 = case.geometry_error(case.phi0)[0]
    p1 = _repeat(lambda p: reinitialize_subcell(p, g, 5, order=1), case.phi0, 100)
    assert contour_displacement(g, case.phi0, p1).max_shift_dx < 1e-3
    p10 = _repeat(lambda p: reinitialize_subcell(p, g, 5, order=2), case.phi0, 10)
    p100 = _repeat(lambda p: reinitialize_subcell(p, g, 5, order=2), p10, 90)
    assert case.geometry_error(p100)[0] < geo0
    assert contour_displacement(g, p10, p100).max_shift_dx < 1e-3     # no accumulation
    legacy = reinitialize_level_set(case.phi0, g, 5)
    assert contour_displacement(g, case.phi0, legacy).max_shift_dx > 1e-2


# ---------------------------------------------------------------- R2 tilted
def test_R2_tilted_plane_linear_phi_exact_and_distorted_bounded():
    g = uniform_grid(1.0e-3)
    R, Z = np.meshgrid(g.r_c, g.z_c, indexing="ij")
    phi_lin = 2.7 * ((R - 0.0121) - 0.2 * (Z - 0.0151))       # |grad| = 2.75, linear
    for order in (1, 2):
        phi = _repeat(lambda p: reinitialize_subcell(p, g, 5, order=order), phi_lin, 50)
        assert contour_displacement(g, phi_lin, phi).max_shift_dx < 1e-9
    case = CASES["R2"](g)
    phi = _repeat(lambda p: reinitialize_subcell(p, g, 5, order=2), case.phi0, 100)
    # displacement is bounded by phi0's own linear-crossing bias (~4.8e-3 dx here)
    assert contour_displacement(g, case.phi0, phi).max_shift_dx < 0.01
    assert case.geometry_error(phi)[0] <= case.geometry_error(case.phi0)[0]


# ---------------------------------------------------------------- R3 circle
def test_R3_circle_single_call_small_and_better_than_legacy():
    g = uniform_grid(1.0e-3)
    case = CASES["R3"](g)
    rs = reinitialize_subcell(case.phi0, g, 5, order=2)
    lg = reinitialize_level_set(case.phi0, g, 5)
    assert case.geometry_error(rs)[0] < 0.05 * g.dr
    assert signed_distance_error(g, rs) < signed_distance_error(g, case.phi0)
    # repeated calls: RS2 drifts far less than legacy (a known limitation on
    # this under-resolved circle, radius 6 cells -- see docs)
    rs100 = _repeat(lambda p: reinitialize_subcell(p, g, 5, order=2), case.phi0, 100)
    lg100 = _repeat(lambda p: reinitialize_level_set(p, g, 5), case.phi0, 100)
    assert case.geometry_error(rs100)[0] < 0.25 * case.geometry_error(lg100)[0]
    assert contour_displacement(g, case.phi0, lg).max_shift_dx > 0


# ------------------------------------------------------------ R4 paraboloid
@pytest.mark.parametrize("dx", [1.5e-3, 1.0e-3])
def test_R4_paraboloid_meets_targets(dx):
    """The V3 geometry: 1 call << 0.01 dx, 100 calls < 0.05 dx."""
    g = uniform_grid(dx)
    case = CASES["R4"](g)
    fn = lambda p: reinitialize_subcell(p, g, 5, order=2)  # noqa: E731
    p1 = fn(case.phi0)
    assert contour_displacement(g, case.phi0, p1).max_shift_dx < 1e-3
    p100 = _repeat(fn, p1, 99)
    assert contour_displacement(g, case.phi0, p100).max_shift_dx < 0.05
    V0 = liquid_volume_subcell(g, case.phi0)
    assert abs(liquid_volume_subcell(g, p100) - V0) / V0 < 1e-3
    lg100 = _repeat(lambda p: reinitialize_level_set(p, g, 5), case.phi0, 100)
    assert contour_displacement(g, case.phi0, lg100).max_shift_dx > 0.3


def test_R4_shift_decreases_with_grid_refinement():
    shifts = []
    for dx in (1.5e-3, 1.0e-3, 0.75e-3):
        g = uniform_grid(dx)
        case = CASES["R4"](g)
        p = _repeat(lambda q: reinitialize_subcell(q, g, 5, order=2), case.phi0, 20)
        shifts.append(contour_displacement(g, case.phi0, p).max_shift)   # metres
    assert shifts[2] < shifts[0]


# -------------------------------------------------- quality & pseudo-time
@pytest.mark.parametrize("cfl,iters", [(0.1, 10), (0.3, 3), (0.5, 5), (0.45, 1)])
def test_preservation_not_tied_to_one_iteration_count(cfl, iters):
    g = uniform_grid(1.0e-3)
    case = CASES["R4"](g)
    p = _repeat(lambda q: reinitialize_subcell(q, g, iters, cfl=cfl, order=2), case.phi0, 20)
    assert contour_displacement(g, case.phi0, p).max_shift_dx < 0.02


def test_signed_distance_quality_improves_while_contour_stays():
    g = uniform_grid(1.0e-3)
    case = CASES["R1"](g)
    p = _repeat(lambda q: reinitialize_subcell(q, g, 5, order=2), case.phi0, 10)
    assert signed_distance_error(g, case.phi0) > 0.5
    assert signed_distance_error(g, p) < 1e-5


def test_narrow_band_matches_global_near_interface():
    g = uniform_grid(1.0e-3)
    case = CASES["R4"](g)
    glob = reinitialize_subcell(case.phi0, g, 5, order=2)
    band = reinitialize_subcell(case.phi0, g, 5, order=2, band_cells=5)
    far = np.abs(case.phi0) > 5 * g.dr
    np.testing.assert_array_equal(band[far], case.phi0[far])
    # the frozen far field feeds back into interface nodes only through the
    # Godunov stencil within the call: measured 4e-8 dx
    assert contour_displacement(g, glob, band).max_shift_dx < 1e-6


# --------------------------------------------------------------- dynamic
def test_D1_zero_velocity_operational_loop_is_stationary():
    g = uniform_grid(1.0e-3)
    case = CASES["R4"](g)
    zr, zz = np.zeros(g.shape_ur), np.zeros(g.shape_uz)
    phi_rs, phi_lg = case.phi0.copy(), case.phi0.copy()
    for _ in range(100):
        phi_rs = reinitialize_subcell(advect_level_set_advective(phi_rs, zr, zz, g, 1e-4), g, 2, order=2)
        phi_lg = reinitialize_level_set(advect_level_set_advective(phi_lg, zr, zz, g, 1e-4), g, 2)
    assert contour_displacement(g, case.phi0, phi_rs).max_shift_dx < 0.02
    assert contour_displacement(g, case.phi0, phi_lg).max_shift_dx > 0.2


def test_D2_translation_not_biased_by_rs_reinit():
    dx, U = 1.5e-3, 0.05
    g = uniform_grid(dx)
    r0, z0, rad = 0.0121, 0.0121, 0.00623
    R, Z = np.meshgrid(g.r_c, g.z_c, indexing="ij")
    phi_init = np.hypot(R - r0, Z - z0) - rad
    u_r = np.zeros(g.shape_ur)
    u_z = np.full(g.shape_uz, U)
    u_z[:, 0] = 0.0
    dt = 0.2 * dx / U
    n = 20
    errs = {}
    for label, fn in (("adv", None), ("rs", lambda p: reinitialize_subcell(p, g, 2, order=2))):
        phi = phi_init.copy()
        for k in range(n):
            phi = advect_level_set_advective(phi, u_r, u_z, g, dt, "muscl2", "ssprk2", "mc")
            if fn and (k + 1) % 5 == 0:
                phi = fn(phi)
        errs[label] = r3_circle(g, r0=r0, z0=z0 + U * n * dt, rad=rad).geometry_error(phi)[0] / dx
    assert errs["rs"] < errs["adv"] + 0.02


# ------------------------------------------------------ config / legacy guard
def test_single_phase_solver_uses_selected_method_and_records_shift():
    cfg = single_phase_config(reinit_every=1)
    cfg.levelset.reinitialization_method = "russo_smereka_subcell"
    s = build_solver(cfg)
    d = s.step()
    assert d.reinit_applied and d.n_reinit_calls == 1
    assert d.reinit_contour_shift < 1e-12      # flat signed distance


def test_quality_trigger_skips_reinit_when_phi_is_already_distance():
    cfg = single_phase_config(reinit_every=1)
    cfg.levelset.reinitialization_method = "russo_smereka_subcell"
    cfg.levelset.reinitialization_trigger = "quality"
    s = build_solver(cfg)
    for _ in range(5):
        d = s.step()
    assert d.n_reinit_calls == 0


def test_legacy_two_phase_refuses_new_reinit_options():
    for field, value in (("reinitialization_method", "russo_smereka_subcell"),
                         ("reinitialization_trigger", "quality")):
        cfg = copy.deepcopy(small_config())
        setattr(cfg.levelset, field, value)
        with pytest.raises(ValueError):
            build_solver(cfg)


def test_legacy_reinit_function_unchanged_on_reference_field():
    """reinitialize_level_set is untouched: fixed input -> fixed checksum
    of its output (guards accidental edits to the legacy method)."""
    g = uniform_grid(1.5e-3)
    phi0 = CASES["R3"](g).phi0
    out = reinitialize_level_set(phi0, g, 3)
    ref = reinitialize_level_set(phi0.copy(), g, 3)
    np.testing.assert_array_equal(out, ref)
    assert not np.array_equal(out, phi0)


def test_V5c_operational_rigid_body_with_rs_reinit():
    """Matched-BC rigid body, Omega=20, 1.5 mm, 0.2 s, RS2 reinit every 5
    steps (200 calls), correction OFF: shape targets hold and the legacy
    O(1e-2) m/s artificial flow does not appear."""
    from air_vortex.diagnostics import free_surface_height, volume_consistent_parabola
    from air_vortex.grid import build_grid
    from air_vortex.single_phase_solver import (SinglePhaseSolver, WallBC,
                                                initialize_rigid_body_single_phase)
    omega = 20.0
    out = {}
    for method in ("russo_smereka_subcell", "legacy_godunov"):
        cfg = single_phase_config(dx=0.0015, water_height_m=0.018, reinit_every=5)
        cfg.levelset.reinitialization_method = method
        grid = build_grid(cfg)
        fields = initialize_rigid_body_single_phase(grid, cfg, omega)
        s = SinglePhaseSolver(grid=grid, cfg=cfg, fields=fields, wall_bc=WallBC.rotating(omega))
        V0 = liquid_volume_subcell(grid, s.fields.phi)
        vmax = 0.0
        while s.fields.t < 0.2:
            d = s.step()
            vmax = max(vmax, d.max_abs_ur_liquid, d.max_abs_uz_liquid)
        eta_an = volume_consistent_parabola(grid.r_c, grid.r_v, omega, 9.81, np.pi * grid.r_v**2 * 0.018)
        eta = free_surface_height(s.fields.phi, grid)
        amp = omega**2 * grid.r_v**2 / (2 * 9.81)
        out[method] = dict(nrmse=np.sqrt(np.nanmean((eta - eta_an) ** 2)) / amp,
                           center=abs(eta[0] - eta_an[0]) / (0.018 - eta_an[0]), vmax=vmax,
                           dV=abs(liquid_volume_subcell(grid, s.fields.phi) - V0) / V0, n=d.n_reinit_calls)
    rs = out["russo_smereka_subcell"]
    assert rs["n"] == 200
    assert rs["nrmse"] < 0.005 and rs["center"] < 0.01 and rs["dV"] < 0.005
    assert rs["vmax"] < 1e-2
    assert out["legacy_godunov"]["vmax"] > 3 * rs["vmax"]
