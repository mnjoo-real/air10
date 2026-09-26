"""Opt-in single_phase_height research branch (height-function free surface)."""
import numpy as np
import pytest

from air_vortex.height_benchmarks import build_height_flat, build_height_meniscus
from air_vortex.height_interface import (GraphInterface, column_areas, curvature_at_crossings, eta_rate,
                                         geometry_from_eta)
from air_vortex.pinned_phase import R_V


def test_column_update_conserves_volume_algebraically_for_any_velocity():
    """Telescoping column fluxes with zero axis and wall flux: sum_i A_i deta_i/dt = 0
    to round-off for ARBITRARY (not even divergence-free) u_r."""
    s, _ = build_height_meniscus(0.25, 0.5e-3, 60.0)
    g = s.grid
    rng = np.random.default_rng(3)
    A = column_areas(g)
    for _ in range(5):
        u_r = rng.standard_normal((g.Nr + 1, g.Nz)) * 1e-3
        u_r[0], u_r[-1] = 0.0, 0.0
        rate = eta_rate(g, s.eta, s.z_pin, u_r)
        assert abs(np.sum(A * rate)) < 1e-14 * np.sum(A * np.abs(rate))


def test_reconstruction_pins_the_wall_point_and_is_even_at_the_axis():
    s, _ = build_height_meniscus(0.35, 0.25e-3, 60.0)
    gi = s.interface()
    assert gi.value(np.array([R_V]))[0] == pytest.approx(s.z_pin, abs=1e-15)
    _, d1, _ = gi.derivs(np.array([0.0]))
    assert abs(d1[0]) < 1e-9


def test_quadratic_graph_is_reproduced_exactly():
    s = build_height_flat(0.006, 0.25e-3)
    g = s.grid
    A = 30.0
    eta = 0.005 + A * g.r_c**2
    gi = GraphInterface(g, eta, 0.005 + A * R_V**2)
    d1, d2 = gi.node_derivs()
    assert np.allclose(d1, 2 * A * g.r_c, atol=1e-12) and np.allclose(d2, 2 * A, atol=1e-8)


def test_cap_curvature_components_and_sign():
    """Concave-up cap (liquid below): kappa_m = kappa_theta = -1/Rs."""
    s = build_height_flat(0.006, 0.25e-3)
    g = s.grid
    Rs = R_V / np.cos(np.deg2rad(60.0))
    eta_fn = lambda r: 0.004 + Rs - np.sqrt(Rs**2 - np.asarray(r) ** 2)  # noqa: E731
    gi = GraphInterface(g, eta_fn(g.r_c), float(eta_fn(R_V)))
    d1, d2 = gi.node_derivs()
    km, kt = GraphInterface.kappa_parts(g.r_c, d1, d2)
    assert np.abs(km + 1 / Rs).max() < 0.05 and np.abs(kt + 1 / Rs).max() < 0.01     # |kappa| ~ 62 1/m


def test_well_balanced_radial_crossing_curvature_is_linear_in_node_curvature():
    s, _ = build_height_meniscus(0.25, 0.25e-3, 50.0)
    g, gi = s.grid, s.interface()
    geom, rc = geometry_from_eta(g, gi)
    kr, kz = curvature_at_crossings(g, gi, geom, rc)
    d1, d2 = gi.node_derivs()
    kn = np.add(*GraphInterface.kappa_parts(g.r_c, d1, d2))
    fi, fj = np.nonzero(np.isfinite(kr))
    assert fi.size > 0
    w = (rc[fi, fj] - g.r_c[fi - 1]) / g.dr
    assert np.allclose(kr[fi, fj], (1 - w) * kn[fi - 1] + w * kn[fi], rtol=0, atol=1e-10)


def test_derived_phi_never_feeds_back():
    """fields.phi is plotting/compat data: overwriting it must not change the step."""
    a, _ = build_height_meniscus(0.25, 0.5e-3, 60.0)
    b, _ = build_height_meniscus(0.25, 0.5e-3, 60.0)
    b.fields.phi = np.random.default_rng(0).standard_normal(b.fields.phi.shape)
    for _ in range(3):
        a.step(1e-4); b.step(1e-4)
        b.fields.phi = -b.fields.phi
    assert np.array_equal(a.eta, b.eta) and np.array_equal(a.fields.u_r, b.fields.u_r)


def test_hydrostatic_flat_surface_stays_at_round_off():
    s = build_height_flat(0.00617, 0.5e-3)
    V0 = s.volume
    for _ in range(20):
        d = s.step()
    assert max(d.max_abs_ur_liquid, d.max_abs_uz_liquid) < 1e-12
    assert abs(s.volume / V0 - 1) < 1e-14 and np.abs(s.eta - 0.00617).max() < 1e-14


def test_pinned_meniscus_historical_bad_phase_short_run():
    """theta 60, xi 0.25 at 0.5 mm (the V4b-P level-set failure phase, coarse for speed):
    volume exact, interface on the BVP, no growing flow over 0.1 s."""
    s, ref = build_height_meniscus(0.25, 0.5e-3, 60.0)
    V0 = s.volume
    while s.fields.t < 0.1:
        d = s.step()
    assert abs(s.volume / V0 - 1) < 1e-13
    assert np.sqrt(np.mean((s.eta - ref.eta(s.grid.r_c)) ** 2)) / s.grid.dr < 0.01
    assert max(d.max_abs_ur_liquid, d.max_abs_uz_liquid) < 2e-3


def test_model_selector_is_opt_in():
    from air_vortex.config import PhysicsConfig
    assert PhysicsConfig().free_surface_model == "two_phase_diffuse_ls"
    PhysicsConfig(free_surface_model="single_phase_height")


def test_stirrer_forcing_port_spins_up_only_the_bottom_region_and_keeps_volume():
    """Opt-in qualitative forcing (forcing.py term, uncalibrated tau_s): after a few steps
    u_theta > 0 inside the stirrer region and ~0 far above it; volume exact."""
    from air_vortex.height_benchmarks import build_height_production
    s = build_height_production(300.0, dx=1.0e-3)
    V0 = s.volume
    for _ in range(3):
        s.step()
    g, ut = s.grid, s.fields.u_theta
    bar_top = s.cfg.geometry.stirbar_diameter_m
    low = ut[g.r_c < 0.01][:, g.z_c < bar_top]
    high = ut[:, g.z_c > 0.03]
    # the config ramps Omega over 0.3 s, so only the LOCATION of the spin-up is tested here
    assert low.max() > 0.5 * np.abs(ut).max() > 0.0
    assert np.abs(high).max() < 1e-6 * low.max()
    assert abs(s.volume / V0 - 1) < 1e-13


def test_zero_rpm_path_is_unchanged_by_the_forcing_port():
    a, _ = build_height_meniscus(0.25, 0.5e-3, 60.0)
    assert a._chi is None


def _inviscid_lid_swirl(adv):
    from air_vortex.height_benchmarks import build_height_production
    s = build_height_production(0.0, dx=1.0e-3)
    s.cfg.fluid.water_viscosity = 0.0
    s.freeze_interface = s.rigid_lid = True
    s.swirl_advection = adv
    g = s.grid
    r, z = g.r_c[:, None], g.z_c[None, :]
    s.fields.u_theta = np.where(s.geometry.liquid, 5.0 * r * np.exp(-z / 0.01), 0.0)   # drives meridional flow
    L0 = s.torque_budget()["L_z"]
    for _ in range(40):
        s.step(5e-4)
    return s, L0


def test_rigid_lid_conservative_swirl_conserves_angular_momentum_inviscid():
    """nu = 0, no forcing, rigid free-slip lid: the flux form must conserve L_z to round-off,
    which also requires the lid faces to stay no-penetration in the predictor (the velocity
    extension would otherwise overwrite them)."""
    s, L0 = _inviscid_lid_swirl("conservative")
    assert max(s.step(5e-4).max_abs_ur_liquid, 0) > 1e-4          # meridional flow present
    assert abs(s.torque_budget()["L_z"] / L0 - 1) < 1e-12
