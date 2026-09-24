import numpy as np

from air_vortex.operators import divergence
from air_vortex.pressure import pressure_projection, solve_pressure_poisson
from _helpers import make_grid


def test_projection_removes_divergence_uniform_density():
    grid = make_grid(Nr=10, Nz=8)
    rho = np.full(grid.shape_center, 1000.0)
    dt = 1e-3

    rng = np.random.default_rng(0)
    u_r_star = rng.normal(size=grid.shape_ur) * 0.1
    u_z_star = rng.normal(size=grid.shape_uz) * 0.1

    # enforce the same boundary conditions the solver applies before solving
    u_r_star[0, :] = 0.0
    u_r_star[-1, :] = 0.0
    u_z_star[:, 0] = 0.0
    u_z_star[:, -1] = u_z_star[:, -2]

    p = solve_pressure_poisson(grid, u_r_star, u_z_star, rho, dt)
    u_r_new, u_z_new = pressure_projection(grid, u_r_star, u_z_star, p, rho, dt)

    div_after = divergence(grid, u_r_new, u_z_new)
    assert np.max(np.abs(div_after)) < 1e-8


def test_projection_removes_divergence_variable_density():
    grid = make_grid(Nr=10, Nz=12)
    rho = np.where(np.arange(grid.Nz)[None, :] < grid.Nz // 2, 1000.0, 1.2) * np.ones(grid.shape_center)
    dt = 1e-4

    rng = np.random.default_rng(1)
    u_r_star = rng.normal(size=grid.shape_ur) * 0.05
    u_z_star = rng.normal(size=grid.shape_uz) * 0.05
    u_r_star[0, :] = 0.0
    u_r_star[-1, :] = 0.0
    u_z_star[:, 0] = 0.0
    u_z_star[:, -1] = u_z_star[:, -2]

    p = solve_pressure_poisson(grid, u_r_star, u_z_star, rho, dt)
    u_r_new, u_z_new = pressure_projection(grid, u_r_star, u_z_star, p, rho, dt)

    div_after = divergence(grid, u_r_new, u_z_new)
    assert np.max(np.abs(div_after)) < 1e-6


def test_cg_method_matches_direct_method():
    """method='cg' (README 'Performance': an opt-in, AMG-preconditioned-if-
    available alternative solver, not the production default) must agree
    with method='direct' to within CG's own convergence tolerance."""
    grid = make_grid(Nr=12, Nz=16)
    rho = np.where(np.arange(grid.Nz)[None, :] < grid.Nz // 2, 1000.0, 1.2) * np.ones(grid.shape_center)
    dt = 1e-4

    rng = np.random.default_rng(2)
    u_r_star = rng.normal(size=grid.shape_ur) * 0.05
    u_z_star = rng.normal(size=grid.shape_uz) * 0.05
    u_r_star[0, :] = 0.0
    u_r_star[-1, :] = 0.0
    u_z_star[:, 0] = 0.0
    u_z_star[:, -1] = u_z_star[:, -2]

    p_direct = solve_pressure_poisson(grid, u_r_star, u_z_star, rho, dt, method="direct")
    p_cg = solve_pressure_poisson(grid, u_r_star, u_z_star, rho, dt, method="cg")

    np.testing.assert_allclose(p_cg, p_direct, rtol=1e-6, atol=1e-6)

    # projecting with the cg-derived pressure must still be divergence-free
    u_r_new, u_z_new = pressure_projection(grid, u_r_star, u_z_star, p_cg, rho, dt)
    div_after = divergence(grid, u_r_new, u_z_new)
    assert np.max(np.abs(div_after)) < 1e-5
