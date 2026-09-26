"""V7-T: conservative swirl / meridional momentum transport of the single_phase_height branch."""
import copy
import warnings
from pathlib import Path

import numpy as np
import pytest

from air_vortex.config import PhysicsConfig, load_config
from air_vortex.grid import build_grid
from air_vortex.meridional_transport import advection_terms, mass_fluxes
from air_vortex.swirl_transport import advect_q, angular_momentum_viscous, q_flux_divergence

ROOT = Path(__file__).resolve().parents[1]
R, Z = 0.045, 0.050


def _grid(dx):
    cfg = copy.deepcopy(load_config(ROOT / "configs" / "baseline.yaml"))
    cfg.grid.dr_m = cfg.grid.dz_m = dx
    return build_grid(cfg)


def _flow(g, U=0.3):
    """discretely divergence-free MAC flow from psi = r^2 (1 - r/R)^2 sin^2(pi z / Z) (z < Z)."""
    def psi(r, z):
        return r**2 * (1 - r / R) ** 2 * np.sin(np.pi * np.clip(z, 0, Z) / Z) ** 2
    P = psi(g.r_f[:, None], g.z_f[None, :])
    ur = np.zeros((g.Nr + 1, g.Nz))
    ur[1:] = -(P[1:, 1:] - P[1:, :-1]) / g.dz / g.r_f[1:, None]
    uz = (P[1:, :] - P[:-1, :]) / (g.r_c[:, None] * g.dr)
    a = U / max(np.abs(ur).max(), np.abs(uz).max())
    return a * ur, a * uz


def test_discrete_divergence_of_manufactured_flow_is_round_off():
    g = _grid(1e-3)
    ur, uz = _flow(g)
    div = (g.r_f[1:, None] * ur[1:] - g.r_f[:-1, None] * ur[:-1]) / (g.r_c[:, None] * g.dr) + (uz[:, 1:] - uz[:, :-1]) / g.dz
    assert np.abs(div).max() < 1e-9 * 0.3 / g.dr


def test_upwind1_matches_the_v6v7_inline_conservative_form_bitwise():
    g = _grid(1e-3)
    rng = np.random.default_rng(1)
    q = rng.standard_normal(g.shape_center)
    ur = rng.standard_normal((g.Nr + 1, g.Nz))
    uz = rng.standard_normal((g.Nr, g.Nz + 1))
    qe = np.vstack([q[:1], q, q[-1:]])
    Fr = g.r_f[:, None] * ur * np.where(ur >= 0, qe[:-1], qe[1:])
    Fr[0] = Fr[-1] = 0
    qz = np.hstack([q[:, :1], q, q[:, -1:]])
    Fz = uz * np.where(uz >= 0, qz[:, :-1], qz[:, 1:])
    Fz[:, 0] = 0
    old = (Fr[1:] - Fr[:-1]) / (g.r_c[:, None] * g.dr) + (Fz[:, 1:] - Fz[:, :-1]) / g.dz
    assert np.array_equal(old, q_flux_divergence(g, q, ur, uz, 1))


@pytest.mark.parametrize("order", [1, 2])
def test_swirl_transport_conserves_angular_momentum_to_round_off(order):
    g = _grid(1e-3)
    ur, uz = _flow(g)
    r2, z2 = np.meshgrid(g.r_c, g.z_c, indexing="ij")
    q = np.exp(-((r2 - 0.02) ** 2 + (z2 - 0.02) ** 2) / 0.005**2)
    w = r2 * g.dr * g.dz
    L0 = np.sum(q * w)
    for _ in range(50):
        q = advect_q(g, q, ur, uz, 1e-3, order)
    assert abs(np.sum(q * w) / L0 - 1) < 1e-13


def test_muscl2_converges_at_second_order_and_upwind1_at_first_on_a_smooth_mode():
    """Error against the exact solution (RK4 back-characteristics of the continuous flow, fixed
    amplitude; scripts/validate_swirl_transport.py), dx 1 / 0.5 mm, T = 0.02 s. (A self-
    convergence test with per-grid normalised velocity would advect with O(dx)-different
    amplitudes and wrongly show first order.)"""
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    import validate_swirl_transport as V
    T0 = V.T
    V.T = 0.02
    try:
        e = {o: [V.run2d(dx, "m", o, "vanleer", "mode")["L2"] for dx in (1e-3, 0.5e-3)] for o in (1, 2)}
    finally:
        V.T = T0
    assert 0.85 < np.log2(e[1][0] / e[1][1]) < 1.15
    assert 1.8 < np.log2(e[2][0] / e[2][1]) < 2.3
    assert e[2][1] < 0.1 * e[1][1]


def test_angular_momentum_viscous_operator_telescopes_to_the_wall_torque():
    g = _grid(1e-3)
    rng = np.random.default_rng(2)
    ut = rng.standard_normal(g.shape_center)
    uw = rng.standard_normal(g.Nz)
    ub = np.zeros(g.Nr)
    V = angular_momentum_viscous(g, ut, uw, ub)
    # sum over cells of r_c * (r_c dr dz) * (radial part) = dz * sum_j G_wall (interior stresses cancel)
    lhs = np.sum(g.r_c[:, None] ** 2 * g.dr * g.dz * (V - _z_part(g, ut, ub)))
    G_wall = g.r_f[-1] ** 3 * (uw / g.r_f[-1] - ut[-1] / g.r_c[-1]) / (0.5 * g.dr)
    assert lhs == pytest.approx(np.sum(G_wall) * g.dz, rel=1e-10)


def _z_part(g, ut, ub):
    g_ext = np.concatenate([2.0 * ub[:, None] - ut[:, :1], ut, ut[:, -1:]], axis=1)
    return (g_ext[:, 2:] - 2.0 * g_ext[:, 1:-1] + g_ext[:, :-2]) / g.dz**2


def test_angular_momentum_viscous_operator_vanishes_for_solid_body_rotation():
    g = _grid(1e-3)
    Om = 20.0
    ut = Om * g.r_c[:, None] * np.ones(g.shape_center)
    V = angular_momentum_viscous(g, ut, np.full(g.Nz, Om * g.r_f[-1]), Om * g.r_c)
    assert np.abs(V).max() < 1e-9 * Om / g.dr


@pytest.mark.parametrize("order", [1, 2])
def test_meridional_control_volume_fluxes_are_divergence_free(order):
    """A uniform advected value gives N = 0 away from the Dirichlet ghosts: the r-weighted
    interpolated mass fluxes have zero divergence on every staggered control volume."""
    g = _grid(1e-3)
    ur, uz = _flow(g)
    Nr, Nz = advection_terms(g, np.ones_like(ur), np.ones_like(uz), mass_fluxes(g, ur, uz), order)
    assert np.abs(Nr[2:-2, 2:-2]).max() < 1e-10 * 0.3 / g.dr
    assert np.abs(Nz[2:-2, 2:-2]).max() < 1e-10 * 0.3 / g.dr


def test_meridional_muscl2_is_more_accurate_than_the_advective_operator():
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    from validate_meridional_transport import operator_test
    adv = operator_test(0.5e-3, "advective")
    mus = operator_test(0.5e-3, "conservative_muscl2")
    assert mus["L2_interior"] < 0.2 * adv["L2_interior"]
    assert mus["nu_eff"] < 0.05 * adv["nu_eff"]


def test_height_swirl_default_is_conservative_and_legacy_mode_warns():
    assert PhysicsConfig().height_swirl_advection == "conservative_muscl2"
    assert PhysicsConfig().height_swirl_viscous == "angular_momentum"
    assert PhysicsConfig().height_meridional_advection == "advective"
    assert PhysicsConfig().free_surface_model == "two_phase_diffuse_ls"       # LS defaults untouched
    with pytest.warns(UserWarning):
        PhysicsConfig(free_surface_model="single_phase_height", height_swirl_advection="advective")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        PhysicsConfig(free_surface_model="single_phase_height", height_swirl_advection="advective",
                      height_legacy_swirl_for_validation=True)
        PhysicsConfig(free_surface_model="single_phase_ls", height_swirl_advection="advective")
