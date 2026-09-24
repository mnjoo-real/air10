"""Sharp free-surface pressure BC (ghost-fluid Dirichlet at the sub-cell
phi=0 crossing): manufactured solutions in 1D and 2D, projection /
matrix consistency, and the flat hydrostatic off-grid free surface."""
import numpy as np
import pytest

from air_vortex.free_surface_bc import interface_pressure
from air_vortex.liquid_mask import FACE_LIQ_MINUS, FACE_LIQ_PLUS, classify
from air_vortex.pressure_single_phase import (
    build_liquid_pressure_system, liquid_divergence, project_liquid_velocity,
    solve_liquid_pressure, solve_liquid_pressure_from_source, solve_poisson_1d_ghost_fluid,
)
from air_vortex.single_phase_solver import build_single_phase_solver
from _config_helpers import single_phase_config
from _helpers import make_grid

ALPHAS = [0.1, 0.25, 0.5, 0.75, 0.9]
TINY_ALPHAS = [1e-3, 1e-6, 1e-9, 1e-12]


def _observed_order(hs, errs):
    return np.polyfit(np.log(hs), np.log(errs), 1)[0]


# ------------------------------------------------------------------ Test P1
@pytest.mark.parametrize("alpha", ALPHAS + TINY_ALPHAS)
def test_P1_1d_linear_pressure_exact_for_any_crossing(alpha):
    n, h = 20, 0.05
    k_last = 11
    x_gamma = (k_last + 0.5) * h + alpha * h
    p_gamma, slope = 3.0, -7.0
    x, p = solve_poisson_1d_ghost_fluid(n, h, np.zeros(n), x_gamma, p_gamma, flux_left=slope)
    liq = np.isfinite(p)
    exact = p_gamma + slope * (x - x_gamma)
    # THETA_MIN=1e-6 moves the Dirichlet point by <= 1e-6*h (liquid_mask.py)
    tol = 1e-11 if alpha >= 1e-6 else 2e-6 * h * abs(slope)
    assert np.max(np.abs(p[liq] - exact[liq])) < tol


# ------------------------------------------------------------------ Test P2
@pytest.mark.parametrize("x_gamma", [0.6137, 0.8021])
def test_P2_1d_quadratic_second_order(x_gamma):
    p_gamma = 3.0
    hs, errs = [], []
    for n in (10, 20, 40, 80, 160):
        h = 1.0 / n
        x, p = solve_poisson_1d_ghost_fluid(n, h, 2.0 * np.ones(n), x_gamma, p_gamma, 0.0)
        liq = np.isfinite(p)
        exact = x**2 - x_gamma**2 + p_gamma
        hs.append(h)
        errs.append(np.max(np.abs(p[liq] - exact[liq])))
    assert _observed_order(hs, errs) > 1.8, errs


# ------------------------------------------------- 2D manufactured solutions
def _pgamma(geom, value):
    pg_r, pg_z = interface_pressure(geom)
    return np.where(np.isfinite(pg_r), value, np.nan), np.where(np.isfinite(pg_z), value, np.nan)


def test_2d_axial_quadratic_offgrid_interface_converges():
    """p = A(H^2 - z^2) + p0: lap p = -2A, dp/dz(0) = 0, p(H) = p0."""
    A, p0, H = 5.0, 2.0, 0.6137
    hs, errs = [], []
    for n in (10, 20, 40, 80):
        h = 1.0 / n
        g = make_grid(Nr=4, Nz=n, dr=0.25, dz=h)
        phi = g.z_c[None, :] - H + 0 * g.r_c[:, None]
        geom = classify(phi)
        pg_r, pg_z = _pgamma(geom, p0)
        p = solve_liquid_pressure_from_source(g, geom, -2 * A * np.ones(g.shape_center), pg_r, pg_z)
        exact = A * (H**2 - g.z_c[None, :] ** 2) + p0 + 0 * g.r_c[:, None]
        hs.append(h)
        errs.append(np.max(np.abs(p - exact)[geom.liquid]))
    assert _observed_order(hs, errs) > 1.8, errs


def test_2d_radial_air_core_interface_converges():
    """Void core r < r_g (a vertical free surface, like an air column).
    p = r^2 - 2 R^2 ln r + c: lap p = 4, dp/dr(R) = 0, p(r_g) = p0."""
    R, r_g, p0 = 1.0, 0.3371, 1.5

    def exact_fn(r):
        return r**2 - 2 * R**2 * np.log(r) - (r_g**2 - 2 * R**2 * np.log(r_g)) + p0

    hs, errs = [], []
    for n in (10, 20, 40, 80):
        h = R / n
        g = make_grid(Nr=n, Nz=3, dr=h, dz=h)
        phi = r_g - g.r_c[:, None] + 0 * g.z_c[None, :]
        geom = classify(phi)
        assert np.any(geom.face_kind_r == FACE_LIQ_PLUS)
        pg_r, pg_z = _pgamma(geom, p0)
        p = solve_liquid_pressure_from_source(g, geom, 4.0 * np.ones(g.shape_center), pg_r, pg_z,
                                              top_bc="neumann")
        exact = exact_fn(g.r_c)[:, None] + 0 * g.z_c[None, :]
        hs.append(h)
        errs.append(np.max(np.abs(p - exact)[geom.liquid]))
    assert _observed_order(hs, errs) > 1.8, errs


def test_matrix_symmetric_negative_definite_and_projection_consistent():
    """The r-weighted matrix is symmetric; projecting an arbitrary u* with
    the solved p leaves every liquid cell divergence-free (matrix and
    projection share the same interface geometry)."""
    rng = np.random.default_rng(0)
    g = make_grid(Nr=12, Nz=14, dr=0.002, dz=0.002)
    R, Z = np.meshgrid(g.r_c, g.z_c, indexing="ij")
    phi = Z - (0.017 + 0.004 * np.cos(np.pi * R / g.r_v))   # curved, off-grid interface
    geom = classify(phi)
    assert np.any(geom.face_kind_r == FACE_LIQ_MINUS) or np.any(geom.face_kind_r == FACE_LIQ_PLUS)
    pg_r, pg_z = interface_pressure(geom)
    sys_ = build_liquid_pressure_system(g, geom, pg_r, pg_z)
    A = sys_.A.toarray()
    np.testing.assert_allclose(A, A.T, rtol=0, atol=1e-12 * np.abs(A).max())
    assert np.max(np.linalg.eigvalsh(A)) < 0
    u_r = rng.normal(size=g.shape_ur)
    u_z = rng.normal(size=g.shape_uz)
    u_r[0], u_r[-1], u_z[:, 0] = 0, 0, 0
    rho, dt = 998.0, 1e-3
    p = solve_liquid_pressure(g, geom, u_r, u_z, rho, dt, pg_r, pg_z)
    ur2, uz2 = project_liquid_velocity(g, geom, u_r, u_z, p, rho, dt, pg_r, pg_z)
    div = liquid_divergence(g, geom, ur2, uz2)
    assert np.max(np.abs(div)) < 1e-10 * np.max(np.abs(u_r)) / g.dr


# --------------------------------------------- flat hydrostatic, off-grid H
@pytest.mark.parametrize("alpha", ALPHAS + [1e-4, 0.999])
def test_flat_hydrostatic_offgrid_interface_one_step(alpha):
    dx = 0.0015
    k_last = 10
    H = (k_last + 0.5) * dx + alpha * dx
    cfg = single_phase_config(dx=dx, water_height_m=H)
    s = build_single_phase_solver(cfg)
    d = s.step()
    g = s.grid
    liq = s.fields.phi < 0
    exact = cfg.fluid.water_density * cfg.fluid.gravity * (H - g.z_c[None, :]) + 0 * g.r_c[:, None]
    rel = np.max(np.abs(s.fields.p - exact)[liq]) / np.max(exact[liq])
    assert rel < 1e-12
    assert d.max_abs_ur_liquid < 1e-14
    assert d.max_abs_uz_liquid < 1e-14
    assert d.max_divergence_liquid < 1e-10


def test_flat_hydrostatic_long_run_no_secular_growth():
    cfg = single_phase_config()
    s = build_single_phase_solver(cfg)
    g = s.grid
    V0 = s.step().volume_water_subcell
    vmax = []
    for _ in range(500):
        d = s.step()
        vmax.append(max(d.max_abs_ur_liquid, d.max_abs_uz_liquid))
    H = cfg.geometry.water_height_m
    exact = cfg.fluid.water_density * cfg.fluid.gravity * (H - g.z_c[None, :]) + 0 * g.r_c[:, None]
    liq = s.fields.phi < 0
    assert max(vmax) < 1e-13
    assert d.max_divergence_liquid < 1e-10
    assert abs(d.volume_water_subcell - V0) / V0 < 1e-12
    assert np.max(np.abs(s.fields.p - exact)[liq]) < 1e-9
