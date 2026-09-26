"""Gate V4b-P: sub-cell-invariant pinned contact line."""
import numpy as np
import pytest

from air_vortex.config import WallConfig
from air_vortex.contact_angle import (_pinned_ghost, initial_pin_height, reconstruct_pinned,
                                      wall_ghost_column, wall_trace)
from air_vortex.meniscus import solve_meniscus, solve_meniscus_pinned
from air_vortex.pinned_phase import build_phase_solver, depth_for_phase, geometry_sweep_row, phase_of
from air_vortex.reinit_benchmarks import uniform_grid

R_V = 0.008
XI = (0.05, 0.25, 0.45, 0.55, 0.75, 0.95)


# ----------------------------------------------------------------- BVP
def test_pinned_bvp_reproduces_prescribed_theta_solution():
    m = solve_meniscus(60.0, R_V, 0.006)
    p = solve_meniscus_pinned(m.z_wall, R_V, 0.006)
    r = np.linspace(0, R_V, 2001)
    assert np.abs(p.eta(r) - m.eta(r)).max() < 1e-12
    assert p.theta_deg == pytest.approx(60.0, abs=1e-9)
    assert p.P0 == pytest.approx(m.P0, rel=1e-10)


def test_pinned_bvp_theta_is_an_output():
    """Raising the pin at fixed volume makes the equilibrium more wetting."""
    lo = solve_meniscus_pinned(0.0064, R_V, 0.006)
    hi = solve_meniscus_pinned(0.0068, R_V, 0.006)
    assert hi.theta_deg < lo.theta_deg < 90.0
    assert lo.z[-1] == pytest.approx(0.0064, abs=1e-12)


# ------------------------------------------------------ reconstruction
def _line_phi(g, slope, z_pin):
    R, Z = np.meshgrid(g.r_c, g.z_c, indexing="ij")
    return (Z - z_pin - slope * (R - R_V)) / np.sqrt(1 + slope**2)


@pytest.mark.parametrize("fit", ["quadratic", "cubic", "quadratic_weighted"])
def test_constrained_fit_anchors_exactly_and_recovers_plane_at_every_phase(fit):
    """G1: a straight meridional line through (R, z_pin): f(R) = z_pin
    exactly and the slope is recovered to round-off for any sub-cell phase."""
    g = uniform_grid(0.25e-3, R=R_V, Z=0.012)
    for xi in XI:
        z_pin = (24 + xi) * g.dz
        rec = reconstruct_pinned(g, _line_phi(g, 0.577, z_pin), z_pin, fit, 4)
        assert rec.f(R_V) == z_pin
        assert rec.wall_slope == pytest.approx(0.577, abs=1e-10)


def test_ghost_is_continuous_in_z_pin_no_snapping():
    """Moving the pin by 2e-9 m changes the ghost by < 100 x 2e-9 for every
    method: no node is snapped to. (The legacy 4-node window switches at a
    cell centre, but both windows interpolate that node, so the legacy
    ghost is C0 there too -- its phase sensitivity is NOT a jump.)"""
    from air_vortex.meniscus import meniscus_config
    from air_vortex.grid import build_grid
    from air_vortex.single_phase_solver import signed_distance_to_profile
    dx = 0.25e-3
    H, z0 = depth_for_phase(0.45, dx)
    g = build_grid(meniscus_config(dx, None, H=H, contact_model="pinned"))
    phi = signed_distance_to_profile(g, solve_meniscus_pinned(z0, R_V, H).eta)
    zc = g.z_c[np.searchsorted(g.z_c, z0)]           # cell centre: legacy window boundary
    for method in ("reconstruct_ghost", "reconstruct_ghost_distance", "legacy"):
        wc = lambda zp: WallConfig("pinned", None, zp, method, "cubic")  # noqa: E731
        a = wall_ghost_column(g, phi, wc(zc - 1e-9))
        b = wall_ghost_column(g, phi, wc(zc + 1e-9))
        assert np.abs(a - b).max() < 100 * 2e-9      # Lipschitz in z_pin, no jump


def test_initial_pin_height_is_continuous():
    g = uniform_grid(0.25e-3, R=R_V, Z=0.012)
    zs = [(24 + xi) * g.dz for xi in np.linspace(0.3, 0.7, 9)]
    got = [initial_pin_height(g, _line_phi(g, 0.577, z)) for z in zs]
    np.testing.assert_allclose(got, zs, atol=1e-12)


def test_spherical_cap_reconstruction_phase_invariant():
    """G2: spherical cap meeting the wall at 60 deg, pin at varying phase:
    wall slope from the constrained cubic within a narrow band for all xi."""
    g = uniform_grid(0.25e-3, R=R_V, Z=0.012)
    t = np.deg2rad(60.0)
    Rs = R_V / np.cos(t)
    h = np.sqrt(Rs**2 - R_V**2)
    Rm, Zm = np.meshgrid(g.r_c, g.z_c, indexing="ij")
    errs = []
    for xi in XI:
        z_pin = (24 + xi) * g.dz
        phi = Rs - np.hypot(Rm, Zm - (z_pin + h))
        rec = reconstruct_pinned(g, phi, z_pin, "cubic", 4)
        errs.append(abs(rec.wall_slope - 1 / np.tan(t)))
    assert max(errs) < 5e-3 and max(errs) / max(min(errs), 1e-12) < 20


def test_meniscus_wall_curvature_phase_spread_new_vs_legacy():
    """G3: nonlinear BVP meniscus, wall-column curvature error across the
    sub-cell phase. The new ghost lowers the WORST case ~5x; its phase
    spread (max/min ~6) is NOT smaller than legacy's -- recorded, not hidden."""
    dx = 0.25e-3
    new = [geometry_sweep_row(xi, dx, "reconstruct_ghost_distance", "cubic")["kappa_err"] for xi in XI]
    old = [geometry_sweep_row(xi, dx, "legacy")["kappa_err"] for xi in XI]
    assert max(new) < 0.3 * max(old)


def test_phase_helper_puts_pin_at_requested_phase():
    for xi in (0.05, 0.5, 0.95):
        H, zp = depth_for_phase(xi, 0.25e-3)
        assert phase_of(zp, 0.25e-3) == pytest.approx(xi, abs=1e-9)
        s, ref = build_phase_solver(xi, 0.25e-3, "reconstruct_ghost_distance", "cubic")
        assert s.cfg.wall.pinned_contact_height_m == pytest.approx(zp, abs=1e-15)
        assert ref.theta_deg == pytest.approx(60.0, abs=1e-6)


# ------------------------------------------- wall-compatible initial condition


def _wall_column_p_error(s, ref):
    """p_Gamma - (P0 - rho g z) at the vertical crossing of the wall column."""
    from air_vortex.curvature_single_phase import crossing_positions
    from air_vortex.liquid_mask import classify
    from air_vortex.meniscus import G, RHO
    geom = classify(s.fields.phi)
    _, pz, _, _ = s.interface_pressure_bc(geom, s.fields.phi)
    _, _, _, zz = crossing_positions(s.grid, geom)
    j = np.isfinite(pz[-1, :])
    return float(np.max(np.abs(pz[-1, j] - (ref.P0 - RHO * G * zz[-1, j]))))


def test_truncated_initial_phi_has_wall_curvature_error_growing_like_one_over_dx():
    """V4b-P diagnosis: the distance to the profile CUT at r = R gives level
    sets that wrap around (R, z_pin); the wall-column capillary pressure error
    doubles when dx halves (O(1/dx), not a converging error)."""
    e = [_wall_column_p_error(*build_phase_solver(0.25, dx, "reconstruct_ghost", "cubic"))
         for dx in (0.5e-3, 0.25e-3)]
    assert e[1] > 1.6 * e[0] and e[1] > 2.5          # Pa; ~1.8 -> ~3.2 Pa


def test_wall_compatible_initial_phi_removes_the_endpoint_artifact():
    """The xi = 0.25, dx = 0.25 mm case that seeded the V4b-P anomaly."""
    trunc = _wall_column_p_error(*build_phase_solver(0.25, 0.25e-3, "reconstruct_ghost", "cubic"))
    ext = _wall_column_p_error(*build_phase_solver(0.25, 0.25e-3, "reconstruct_ghost", "cubic",
                                                   ic="extended"))
    assert ext < 0.3 * trunc and ext < 1.0          # Pa; 3.2 -> 0.66 Pa


def test_phase_solver_rejects_grids_whose_wall_is_not_at_R():
    """build_grid rounds Nr = R/dr; for non-integer R/dx the wall face moves
    away from the R of the BVP reference (invalid phase/dx comparison)."""
    with pytest.raises(ValueError, match="not an integer"):
        build_phase_solver(0.25, 0.225e-3, "reconstruct_ghost", "cubic")
