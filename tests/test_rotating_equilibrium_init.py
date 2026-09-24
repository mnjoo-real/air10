import numpy as np

from air_vortex.fields import initialize_rotating_equilibrium
from air_vortex.grid import build_grid
from air_vortex.solver import water_volume
from _config_helpers import small_config


def test_initial_volume_matches_target_exactly():
    cfg = small_config(rpm=0.0)
    grid = build_grid(cfg)
    omega = 10.0

    fields = initialize_rotating_equilibrium(grid, cfg, omega)

    target = np.pi * grid.r_v**2 * cfg.geometry.water_height_m
    actual = water_volume(grid, fields.phi)
    assert abs(actual - target) / target < 1e-10


def test_initial_velocity_is_exact_solid_body_with_zero_meridional_flow():
    cfg = small_config(rpm=0.0)
    grid = build_grid(cfg)
    omega = 7.5

    fields = initialize_rotating_equilibrium(grid, cfg, omega)

    np.testing.assert_array_equal(fields.u_r, 0.0)
    np.testing.assert_array_equal(fields.u_z, 0.0)
    np.testing.assert_allclose(fields.u_theta, omega * grid.r_c[:, None] * np.ones(grid.shape_center))


def test_initial_pressure_satisfies_rotating_hydrostatic_balance():
    """dp/dr = rho*omega^2*r and dp/dz = -rho*g in the water bulk, and
    p=0 at the free surface -- checked via finite differences on the
    constructed field itself, away from the interface transition."""
    cfg = small_config(rpm=0.0)
    grid = build_grid(cfg)
    omega = 8.0
    rho_w = cfg.fluid.water_density
    g = cfg.fluid.gravity

    fields = initialize_rotating_equilibrium(grid, cfg, omega)

    # pick an interior water column, away from the interface and boundaries
    i = grid.Nr // 2
    j_bulk = 1  # a low, definitely-water row

    dpdz = (fields.p[i, j_bulk + 1] - fields.p[i, j_bulk - 1]) / (2 * grid.dz)
    np.testing.assert_allclose(dpdz, -rho_w * g, rtol=1e-8)

    dpdr = (fields.p[i + 1, j_bulk] - fields.p[i - 1, j_bulk]) / (2 * grid.dr)
    expected = rho_w * omega**2 * grid.r_c[i]
    np.testing.assert_allclose(dpdr, expected, rtol=1e-6)


def test_pressure_is_zero_at_the_free_surface():
    """p(r, eta(r)) = 0 by construction -- check at the interface cell
    nearest the surface for a few radii."""
    cfg = small_config(rpm=0.0)
    grid = build_grid(cfg)
    omega = 6.0

    fields = initialize_rotating_equilibrium(grid, cfg, omega)

    # nearest cell center can be up to half a cell from the true surface;
    # the resulting pressure residual should be bounded by roughly one
    # full cell's hydrostatic pressure change, not by an arbitrary constant
    tol = cfg.fluid.water_density * cfg.fluid.gravity * grid.dz
    for i in (0, grid.Nr // 2, grid.Nr - 1):
        col = fields.phi[i, :]
        j = np.argmin(np.abs(col))  # cell nearest the interface
        assert abs(fields.p[i, j]) < tol
