"""Capillary-corrected rotating-equilibrium free surface (README "capillary-
corrected equilibrium reference"), CASE C: physical surface tension.

The exact (nonlinear) axisymmetric Young-Laplace equilibrium is

    rho*g*eta(r) - 0.5*rho*omega^2*r^2 + sigma*kappa(r) = K   (constant)

with the axisymmetric mean curvature

    kappa(r) = eta'/(r*sqrt(1+eta'^2)) + eta''/(1+eta'^2)^(3/2)

This module solves the SMALL-SLOPE LINEARIZATION of that equation
(kappa ~= eta'' + eta'/r, i.e. sqrt(1+eta'^2) ~= 1), which turns it into a
linear, constant-coefficient-in-the-Bessel-sense ODE with a closed-form
solution in modified Bessel functions -- avoiding a nonlinear BVP shooting
problem for what is meant to be a reference/comparison profile, not
production physics. The linearization is justified quantitatively: for
the validation geometry/RPM used in this project, the maximum interface
slope (from the sigma=0 parabola, an upper bound since capillarity only
flattens the profile further) is ~0.27 (eta'^2 ~ 0.07, a few-percent
correction to sqrt(1+eta'^2)) -- see
tests/test_capillary_equilibrium.py::test_small_slope_assumption_is_reasonable.

Derivation: eta(r) = a + Omega^2 r^2/(2g) + A*I_0(r/l_c), where
l_c = sqrt(sigma/(rho*g)) is the capillary length and I_0/I_1 are modified
Bessel functions of the first kind (order 0/1). A is fixed by the
zero-slope wall condition eta'(R_v)=0 (the same "no contact-angle model"
assumption as README section 4's requirement not to add unmodeled
physics -- this matches the solver's own implicit wall treatment: a
zero-gradient Level Set Neumann condition at r=R_v is, near the interface,
exactly d(phi)/dr=0 => eta'(R_v)=0). The additive constant a is then fixed
by the volume constraint, using the Bessel integral identity
integral_0^R r*I_0(r/l_c) dr = l_c*R*I_1(R/l_c).
"""
from __future__ import annotations

import numpy as np
from scipy.special import iv  # modified Bessel function of the first kind


def capillary_length(sigma: float, rho: float, g: float) -> float:
    return float(np.sqrt(sigma / (rho * g)))


def capillary_corrected_parabola(r: np.ndarray, R_v: float, omega: float, g: float,
                                  rho: float, sigma: float, target_volume: float) -> np.ndarray:
    """eta(r) for the linearized capillary-corrected rotating equilibrium.
    Falls back to the exact sigma=0 volume-consistent parabola if sigma=0
    (the Bessel formula is singular in that limit)."""
    if sigma <= 0:
        from .diagnostics import volume_consistent_parabola
        return volume_consistent_parabola(r, R_v, omega, g, target_volume)

    l_c = capillary_length(sigma, rho, g)
    x_wall = R_v / l_c

    I0_wall = float(iv(0, x_wall))
    I1_wall = float(iv(1, x_wall))

    # A from the zero-slope wall condition
    A = -(omega**2 * R_v * l_c) / (g * I1_wall)

    # a from the volume constraint
    bessel_volume_term = 2.0 * np.pi * A * l_c * R_v * I1_wall
    a = (target_volume - omega**2 * np.pi * R_v**4 / (4.0 * g) - bessel_volume_term) / (np.pi * R_v**2)

    return a + omega**2 * r**2 / (2.0 * g) + A * iv(0, r / l_c)


def capillary_corrected_slope(r: np.ndarray, R_v: float, omega: float, g: float,
                               rho: float, sigma: float, target_volume: float) -> np.ndarray:
    """eta'(r) for the same profile (used to check the small-slope
    assumption and the eta'(R_v)=0 boundary condition)."""
    if sigma <= 0:
        return omega**2 * r / g

    l_c = capillary_length(sigma, rho, g)
    x_wall = R_v / l_c
    I1_wall = float(iv(1, x_wall))
    A = -(omega**2 * R_v * l_c) / (g * I1_wall)
    # d/dr I_0(r/l_c) = (1/l_c) I_1(r/l_c)
    return omega**2 * r / g + A * iv(1, r / l_c) / l_c
