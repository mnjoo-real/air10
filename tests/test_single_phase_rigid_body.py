"""Gate V3 (manufactured): rigid-body rotating liquid, sigma = 0, with the
MATCHED co-rotating container BC (WallBC.rotating). Not a production-BC
test: exact solid-body rotation is not an equilibrium against a
stationary no-slip wall (checked explicitly below as a negative control).
"""
import numpy as np
import pytest

from air_vortex.diagnostics import free_surface_height, volume_consistent_parabola
from air_vortex.grid import build_grid
from air_vortex.liquid_mask import liquid_volume_subcell
from air_vortex.single_phase_solver import (
    SinglePhaseSolver, WallBC, initialize_rigid_body_single_phase, laplacian_utheta_dirichlet,
)
from air_vortex.operators import laplacian_utheta
from _config_helpers import single_phase_config

OMEGA = 20.0   # rad/s: 11.7 mm rim-to-center rise on R = 24 mm


def _solver(phi_form, dx=0.0015, wall=None, omega=OMEGA):
    cfg = single_phase_config(dx=dx, water_height_m=0.018)
    grid = build_grid(cfg)
    fields = initialize_rigid_body_single_phase(grid, cfg, omega, phi_form=phi_form)
    return SinglePhaseSolver(grid=grid, cfg=cfg, fields=fields,
                             wall_bc=wall or WallBC.rotating(omega)), cfg, grid


def _eta_analytic(grid, cfg, omega=OMEGA):
    V = np.pi * grid.r_v**2 * cfg.geometry.water_height_m
    return volume_consistent_parabola(grid.r_c, grid.r_v, omega, cfg.fluid.gravity, V)


def test_rotating_wall_operator_matches_legacy_when_stationary():
    cfg = single_phase_config()
    grid = build_grid(cfg)
    u = np.random.default_rng(3).normal(size=grid.shape_center)
    np.testing.assert_array_equal(
        laplacian_utheta_dirichlet(grid, u, np.zeros(grid.Nz), np.zeros(grid.Nr)),
        laplacian_utheta(grid, u))


def test_equilibrium_initialization_is_consistent():
    s, cfg, grid = _solver("signed_distance")
    liq = s.fields.phi < 0
    np.testing.assert_allclose(s.fields.u_theta[liq], (OMEGA * grid.r_c[:, None] * np.ones(grid.shape_center))[liq])
    assert np.all(s.fields.u_r == 0) and np.all(s.fields.u_z == 0)
    eta = free_surface_height(s.fields.phi, grid)
    assert np.nanmax(np.abs(eta - _eta_analytic(grid, cfg))) < 0.05 * grid.dz
    # matched BC: Omega r is in the null space of the rotating-wall vector Laplacian
    lap = laplacian_utheta_dirichlet(grid, OMEGA * grid.r_c[:, None] * np.ones(grid.shape_center),
                                     s.wall_bc.wall_u_theta(grid), s.wall_bc.bottom_u_theta(grid))
    assert np.max(np.abs(lap)) < 1e-9 * OMEGA / grid.dr


def test_stationary_wall_is_not_an_equilibrium_negative_control():
    """Why Gate V3 needs the matched BC: with the production no-slip wall,
    Omega r immediately spins down near the wall/bottom."""
    s, _, grid = _solver("height", wall=WallBC.no_slip())
    ut0 = s.fields.u_theta.copy()
    s.step()
    liq = s.fields.phi < 0
    assert np.max(np.abs(s.fields.u_theta - ut0)[liq]) > 1e-4


def test_one_step_preservation_height_form_is_round_off():
    """phi = z - eta(r) makes p = -rho g phi exactly; the ghost-fluid face
    gradient with a linear crossing is exact for p affine in phi."""
    s, _, grid = _solver("height")
    d = s.step()
    assert d.max_abs_ur_liquid < 1e-12 and d.max_abs_uz_liquid < 1e-12
    assert d.max_divergence_liquid < 1e-10


def test_one_step_preservation_signed_distance():
    s, _, _ = _solver("signed_distance")
    d = s.step()
    assert d.max_abs_ur_liquid < 1e-3 and d.max_abs_uz_liquid < 1e-3
    assert d.max_divergence_liquid < 1e-10


@pytest.mark.parametrize("vc,t_end", [(False, 0.4), (True, 0.1)])
def test_multi_step_rotating_equilibrium_preserved(vc, t_end):
    """Signed-distance phi: the discrete free surface first adjusts to the
    discrete isobar (a small, grid-convergent transient that peaks within
    ~0.03 s), which must then decay -- not grow -- under viscosity."""
    s, cfg, grid = _solver("signed_distance")
    cfg.levelset.volume_correction.enabled = vc
    V0 = liquid_volume_subcell(grid, s.fields.phi)
    eta_an = _eta_analytic(grid, cfg)
    amp = OMEGA**2 * grid.r_v**2 / (2 * cfg.fluid.gravity)
    depression = cfg.geometry.water_height_m - eta_an[0]
    hist = []
    while s.fields.t < t_end:
        d = s.step()
        hist.append(max(d.max_abs_ur_liquid, d.max_abs_uz_liquid))
        assert d.max_divergence_liquid < 1e-10
    eta = free_surface_height(s.fields.phi, grid)
    nrmse = np.sqrt(np.nanmean((eta - eta_an) ** 2)) / amp
    center_rel = abs(eta[0] - eta_an[0]) / depression
    drift = abs(liquid_volume_subcell(grid, s.fields.phi) - V0) / V0
    assert nrmse < 0.05
    assert center_rel < 0.05
    assert drift < 0.005
    assert max(hist) < 1e-2                     # no O(1e-2) meridional flow
    if t_end >= 0.4:
        n = len(hist)
        assert max(hist[3 * n // 4:]) < 0.75 * max(hist)   # decaying, no secular growth
