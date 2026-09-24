import numpy as np

from air_vortex.operators import interp_ur_to_center, interp_uz_to_center
from air_vortex.viz_common import cell_center_velocity, robust_clim, to_mm
from _helpers import make_grid


def test_to_mm_unit_conversion():
    assert to_mm(0.045) == 45.0
    np.testing.assert_allclose(to_mm(np.array([0.001, 0.002])), [1.0, 2.0])


def test_cell_center_velocity_matches_operators_directly():
    grid = make_grid(Nr=6, Nz=5)
    u_r = np.random.default_rng(0).normal(size=grid.shape_ur)
    u_z = np.random.default_rng(1).normal(size=grid.shape_uz)

    u_r_c, u_z_c = cell_center_velocity(grid, u_r, u_z)

    np.testing.assert_allclose(u_r_c, interp_ur_to_center(u_r))
    np.testing.assert_allclose(u_z_c, interp_uz_to_center(u_z))


def test_robust_clim_ignores_a_single_outlier():
    rng = np.random.default_rng(2)
    data = rng.normal(size=10000)
    data[0] = 1e6  # single wild outlier

    vmin, vmax = robust_clim([data], low=1, high=99)

    assert vmax < 10  # the outlier must not dominate the range
    assert vmin > -10


def test_robust_clim_pools_multiple_arrays():
    a = np.zeros((5, 5))
    b = np.ones((5, 5)) * 10.0

    vmin, vmax = robust_clim([a, b], low=0, high=100)

    assert vmin <= 0.0
    assert vmax >= 10.0
