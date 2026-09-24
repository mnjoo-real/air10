import numpy as np

from air_vortex.operators import (
    center_grad_r,
    center_grad_z,
    divergence,
    grad_p_to_ur_faces,
    grad_p_to_uz_faces,
    laplacian_utheta,
)

from _helpers import make_grid


def test_divergence_of_uniform_radial_expansion_is_exact():
    grid = make_grid()
    a = 3.7
    u_r = a * grid.r_f[:, None] * np.ones((grid.Nr + 1, grid.Nz))
    u_z = np.zeros(grid.shape_uz)

    div = divergence(grid, u_r, u_z)

    np.testing.assert_allclose(div, 2 * a, rtol=1e-12)


def test_divergence_of_linear_axial_flow_is_exact():
    grid = make_grid()
    a = 1.3
    u_r = np.zeros(grid.shape_ur)
    u_z = a * grid.z_f[None, :] * np.ones((grid.Nr, grid.Nz + 1))

    div = divergence(grid, u_r, u_z)

    np.testing.assert_allclose(div, a, rtol=1e-12)


def test_grad_p_to_ur_faces_linear_field():
    grid = make_grid()
    C = 2.5
    p = C * grid.r_c[:, None] * np.ones(grid.shape_center)

    dpdr = grad_p_to_ur_faces(grid, p)

    np.testing.assert_allclose(dpdr[1:-1, :], C, rtol=1e-10)


def test_grad_p_to_uz_faces_linear_field():
    grid = make_grid()
    C = -1.7
    p = C * grid.z_c[None, :] * np.ones(grid.shape_center)

    dpdz = grad_p_to_uz_faces(grid, p)

    np.testing.assert_allclose(dpdz[:, 1:-1], C, rtol=1e-10)


def test_center_gradients_linear_fields():
    grid = make_grid()
    f_r = 4.0 * grid.r_c[:, None] * np.ones(grid.shape_center)
    f_z = -2.0 * grid.z_c[None, :] * np.ones(grid.shape_center)

    np.testing.assert_allclose(center_grad_r(grid, f_r)[1:-1, :], 4.0, rtol=1e-10)
    np.testing.assert_allclose(center_grad_z(grid, f_z)[:, 1:-1], -2.0, rtol=1e-10)


def test_center_grad_r_axis_uses_mirror_symmetry():
    """r=0 is a symmetry line: f is implicitly even in r, so the axis row
    of center_grad_r must match the same centered-difference accuracy as
    the interior, not a naive one-sided (2x too large) estimate. A
    quadratic a + b*r^2 is even and exactly reproduced by any correctly
    centered 2nd-order stencil, including at the axis."""
    grid = make_grid(Nr=10, Nz=3)
    a, b = 2.5, 7.0
    f = (a + b * grid.r_c[:, None] ** 2) * np.ones(grid.shape_center)

    dfdr = center_grad_r(grid, f)

    expected = 2 * b * grid.r_c[:, None] * np.ones(grid.shape_center)
    np.testing.assert_allclose(dfdr[:-1, :], expected[:-1, :], rtol=1e-10)


def test_laplacian_utheta_zero_for_solid_body_rotation():
    """Solid-body rotation u_theta = Omega*r is an exact null vector of the
    axisymmetric vector Laplacian (1/r)d/dr(r du/dr) - u/r^2 + d2u/dz2."""
    grid = make_grid(Nr=12, Nz=8)
    Omega = 5.0
    u_theta = Omega * grid.r_c[:, None] * np.ones(grid.shape_center)

    lap = laplacian_utheta(grid, u_theta)

    # interior points only: boundary rows use a Dirichlet ghost (u=0) that
    # intentionally does not match the unbounded solid-body field.
    np.testing.assert_allclose(lap[1:-1, 1:-1], 0.0, atol=1e-8)
