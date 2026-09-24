import numpy as np

from air_vortex.levelset import advect_level_set, reinitialize_level_set
from _helpers import make_grid


def test_advection_shifts_flat_interface_under_uniform_downward_flow():
    grid = make_grid(Nr=6, Nz=20, dr=0.005, dz=0.002)
    H = 0.5 * grid.z_max
    phi0 = grid.z_c[None, :] - H
    phi0 = phi0 * np.ones(grid.shape_center)

    w = -0.05  # uniform downward velocity
    u_r = np.zeros(grid.shape_ur)
    u_z = w * np.ones(grid.shape_uz)
    dt = 0.01

    phi1 = advect_level_set(phi0, u_r, u_z, grid, dt)

    # the interface (phi=0 crossing) should move down by |w|*dt
    expected_shift = w * dt
    tip0 = grid.z_c[np.argmin(np.abs(phi0[0, :]))]
    tip1 = grid.z_c[np.argmin(np.abs(phi1[0, :]))]
    assert abs((tip1 - tip0) - expected_shift) < 1.5 * grid.dz


def test_reinitialization_preserves_interface_sign():
    grid = make_grid(Nr=8, Nz=8, dr=0.005, dz=0.005)
    H = 0.5 * grid.z_max
    phi0 = (grid.z_c[None, :] - H) * np.ones(grid.shape_center)

    # distort the field away from a true signed distance function
    phi_distorted = np.sign(phi0) * np.abs(phi0) ** 1.5

    phi_reinit = reinitialize_level_set(phi_distorted, grid, n_iter=20)

    assert np.all(np.sign(phi_reinit) == np.sign(phi0))

    # gradient magnitude should move closer to 1 than the distorted input
    def grad_mag_error(phi):
        dzp = (phi[:, 1:] - phi[:, :-1]) / grid.dz
        return np.mean(np.abs(np.abs(dzp) - 1.0))

    assert grad_mag_error(phi_reinit) < grad_mag_error(phi_distorted)
