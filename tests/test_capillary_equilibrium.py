import numpy as np

from air_vortex.capillary_equilibrium import (
    capillary_corrected_parabola,
    capillary_corrected_slope,
    capillary_length,
)
from air_vortex.diagnostics import volume_consistent_parabola

R_V, OMEGA, G, RHO, SIGMA = 0.024, 2 * np.pi * 100 / 60, 9.81, 998.0, 0.072
H = 0.018
TARGET = np.pi * R_V**2 * H


def test_capillary_length_matches_known_water_air_value():
    l_c = capillary_length(SIGMA, RHO, G)
    # sqrt(0.072 / (998*9.81)) ~= 2.71 mm, a standard textbook water value
    assert abs(l_c * 1e3 - 2.712) < 0.01


def test_capillary_profile_satisfies_volume_constraint():
    r = np.linspace(1e-4, R_V, 20_000)
    eta = capillary_corrected_parabola(r, R_V, OMEGA, G, RHO, SIGMA, TARGET)
    V = np.trapezoid(2 * np.pi * r * eta, r)
    assert abs(V - TARGET) / TARGET < 1e-3


def test_capillary_profile_has_zero_slope_at_wall():
    slope_wall = capillary_corrected_slope(np.array([R_V]), R_V, OMEGA, G, RHO, SIGMA, TARGET)[0]
    assert abs(slope_wall) < 1e-8


def test_capillary_profile_reduces_to_simple_parabola_when_sigma_zero():
    r = np.linspace(0.0, R_V, 50)
    eta_cap = capillary_corrected_parabola(r, R_V, OMEGA, G, RHO, 0.0, TARGET)
    eta_simple = volume_consistent_parabola(r, R_V, OMEGA, G, TARGET)
    np.testing.assert_allclose(eta_cap, eta_simple)


def test_capillary_correction_is_a_meaningful_but_bounded_deviation():
    """The correction should differ from the simple parabola (surface
    tension is not negligible at this capillary-length/vessel-radius
    ratio -- README "capillary length"), but the deviation should stay a
    small fraction of the total depression amplitude, consistent with the
    small-slope linearization being a reasonable approximation."""
    r = np.linspace(0.0, R_V, 1000)
    eta_cap = capillary_corrected_parabola(r, R_V, OMEGA, G, RHO, SIGMA, TARGET)
    eta_simple = volume_consistent_parabola(r, R_V, OMEGA, G, TARGET)

    max_dev = np.max(np.abs(eta_cap - eta_simple))
    amplitude = OMEGA**2 * R_V**2 / (2 * G)
    assert max_dev > 1e-5  # not negligible
    assert max_dev < amplitude  # but bounded, not a wild divergence


def test_small_slope_assumption_is_reasonable():
    """Quantitative check backing the linearization used by this module:
    the maximum slope (from the sigma=0 parabola, an upper bound since
    capillarity flattens the profile further) should be small enough that
    sqrt(1+eta'^2) ~= 1 to within a few percent."""
    slope_wall_simple = OMEGA**2 * R_V / G
    correction = np.sqrt(1 + slope_wall_simple**2) - 1
    assert correction < 0.05  # < 5% correction to the linearized curvature term
