import numpy as np

from air_vortex.solver import water_volume
from air_vortex.volume_correction import apply_volume_correction, find_volume_correction_delta
from _helpers import make_grid


def test_local_water_volume_matches_solver_water_volume():
    """The module's internal _water_volume re-implementation (kept separate
    to avoid a circular import with solver.py) must stay numerically
    identical to air_vortex.solver.water_volume."""
    from air_vortex.volume_correction import _water_volume

    grid = make_grid(Nr=10, Nz=12)
    rng = np.random.default_rng(0)
    phi = rng.normal(size=grid.shape_center)

    assert _water_volume(grid, phi) == water_volume(grid, phi)


def test_volume_correction_hits_target_exactly():
    grid = make_grid(Nr=12, Nz=16, dr=0.001, dz=0.001)
    H = grid.z_c[7]
    phi = (grid.z_c[None, :] - H) * np.ones(grid.shape_center)

    V0 = water_volume(grid, phi)
    # perturb phi so its volume no longer matches V0 (simulate drift)
    phi_drifted = phi + 0.3 * grid.dz

    phi_corrected, result = apply_volume_correction(grid, phi_drifted, target_volume=V0)

    assert result.applied
    V_corrected = water_volume(grid, phi_corrected)
    assert abs(V_corrected - V0) / V0 < 1e-8


def test_volume_correction_is_a_uniform_shift_only():
    """The correction must not reshape the interface -- every cell shifts
    by exactly the same delta."""
    grid = make_grid(Nr=8, Nz=10, dr=0.001, dz=0.001)
    rng = np.random.default_rng(1)
    phi = rng.normal(size=grid.shape_center) * 0.01
    target = water_volume(grid, phi) * 1.05  # ask for 5% more water

    phi_corrected, result = apply_volume_correction(grid, phi, target_volume=target)

    np.testing.assert_allclose(phi_corrected - phi, result.delta)


def test_volume_correction_flags_large_shifts():
    grid = make_grid(Nr=10, Nz=10, dr=0.001, dz=0.001)
    H = grid.z_c[5]
    phi = (grid.z_c[None, :] - H) * np.ones(grid.shape_center)
    V0 = water_volume(grid, phi)

    # target a wildly different volume -> requires a large shift
    _, result_small = apply_volume_correction(grid, phi, target_volume=V0 * 1.001)
    assert not result_small.large_correction_warning

    _, result_large = apply_volume_correction(grid, phi, target_volume=V0 * 0.2)
    assert result_large.large_correction_warning


def test_find_volume_correction_delta_monotonic_direction():
    """Increasing delta (raising phi) must decrease water volume -- a
    sanity check on the sign convention (phi<0=water)."""
    grid = make_grid(Nr=8, Nz=10, dr=0.001, dz=0.001)
    H = grid.z_c[5]
    phi = (grid.z_c[None, :] - H) * np.ones(grid.shape_center)

    v_plus = water_volume(grid, phi + 0.002)
    v_minus = water_volume(grid, phi - 0.002)
    assert v_plus < v_minus
