"""Milestone 2 (README section 43): imposed solid-body rotation reproduces
the analytical radial force balance dp/dr = rho * u_theta^2 / r (README
section 8, boxed) that underlies the parabolic free-surface shape
(g d(eta)/dr = u_theta^2/r) checked in Test 2 (README section 31).

The comparison is done with a closed (rigid-lid) top boundary rather than
the production open/free-surface boundary. With the open boundary, the
true discrete solution is genuinely 2D (the Dirichlet p=0 condition high
above the water only makes physical sense once the interface has relaxed
into the matching paraboloid, which takes many gravity-wave periods to
happen by pure time integration and is impractical to reach in a fast
test). Closing the top isolates the radial force balance -- the two extra
physics ingredients (dp/dr vs. u_theta, and pressure/projection coupling)
-- from that separate, slow free-surface relaxation process.
"""
import numpy as np

from air_vortex.operators import center_grad_r
from air_vortex.pressure import pressure_projection, solve_pressure_poisson
from _helpers import make_grid


def test_prescribed_solid_body_rotation_matches_centrifugal_balance():
    grid = make_grid(Nr=8, Nz=10, dr=0.003, dz=0.003)
    rho = np.full(grid.shape_center, 998.0)
    omega = 10.0
    dt = 1.0e-4

    u_r_star = dt * omega**2 * grid.r_f[:, None] * np.ones(grid.shape_ur)
    u_z_star = np.zeros(grid.shape_uz)
    u_r_star[0, :] = 0.0
    u_r_star[-1, :] = 0.0  # no penetration at the wall

    p = solve_pressure_poisson(grid, u_r_star, u_z_star, rho, dt, top_bc="closed")
    u_r_new, _ = pressure_projection(grid, u_r_star, u_z_star, p, rho, dt)

    dpdr = center_grad_r(grid, p)
    expected = 998.0 * omega**2 * grid.r_c

    j_mid = grid.Nz // 2
    # interior points only: the diagnostic center_grad_r itself uses a
    # one-sided (less accurate) stencil right at the axis and outer wall.
    for i in range(1, grid.Nr - 1):
        got = dpdr[i, j_mid]
        exp = expected[i]
        assert abs(got - exp) < 0.02 * abs(exp) + 5.0, (
            f"r={grid.r_c[i]*1e3:.2f}mm: dp/dr={got:.1f}, expected {exp:.1f}"
        )

    # the projected radial velocity should now be (near) zero: the imposed
    # swirl is already balanced by the solved pressure field.
    assert np.max(np.abs(u_r_new)) < 1e-6
