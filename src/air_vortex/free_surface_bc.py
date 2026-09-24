"""Free-surface dynamic boundary condition for the single-phase Level-1A
path (README_rewritten sections 5.5, 6.1, 10).

Level 1A neglects gas inertia and gas viscous stress, so the liquid sees
the atmosphere only through the normal-stress balance at phi=0:

    p_Gamma = p_atm + s_kappa * sigma * kappa         (README 5.5)

with p_atm = 0 gauge. The sign s_kappa (S_KAPPA below) was fixed by the
static capillary benchmark CAP-A (Gate V4), not by reading the notation.
No CSF body force is used anywhere in the single-phase path.

Ghost-fluid Dirichlet gradient
------------------------------
For a face between a liquid center P and a void center Q, with the
interface at fraction theta (liquid_mask.crossing_fraction) of the P->Q
spacing h, the pressure gradient on that face is taken as the one-sided
secant to the interface point

    dp/dn|_face = (p_Gamma - p_P) / (theta * h)       (P -> Q direction)

(Gibou, Fedkiw, Cheng & Kang 2002, J. Comput. Phys. 176, 205). This is
equivalent to a ghost value p_Q^g = p_Gamma + (p_Gamma - p_P)(1-theta)/theta
linearly extrapolated through the interface. :func:`ghost_face_gradient`
is the SINGLE implementation used by both the pressure-matrix assembly and
the projection in pressure_single_phase.py, so the matrix and the
velocity correction cannot see different interface locations.
"""
from __future__ import annotations

import numpy as np

from .liquid_mask import FACE_LIQ_MINUS, FACE_LIQ_PLUS, LiquidGeometry

P_ATM_GAUGE = 0.0


S_KAPPA = +1.0
"""Sign of the capillary jump p_Gamma = p_atm + S_KAPPA * sigma * kappa, with
kappa = div(grad phi/|grad phi|) (curvature_single_phase.py; normal pointing
liquid -> air). Fixed by the CAP-A static sphere (Gate V4): liquid inside a
sphere has kappa = +2/R and must sit at p_atm + 2 sigma/R, so S_KAPPA = +1.
Encoded in tests/test_capillary_pressure.py; not a configuration option."""


def interface_pressure(geom: LiquidGeometry, sigma: float = 0.0,
                        p_atm: float = P_ATM_GAUGE,
                        kappa_r: np.ndarray | None = None,
                        kappa_z: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """p_Gamma at every mixed (liquid/void) face crossing, as two arrays on
    the u_r and u_z face grids (nan on non-mixed faces).

    sigma = 0: p_Gamma = p_atm (the Milestone-1 path, unchanged).
    sigma > 0: p_Gamma = p_atm + S_KAPPA * sigma * kappa, with kappa given
    at the same crossings (curvature_single_phase.interface_curvature, or an
    analytic value for the CAP-A verification)."""
    mixed_r = (geom.face_kind_r == FACE_LIQ_MINUS) | (geom.face_kind_r == FACE_LIQ_PLUS)
    mixed_z = (geom.face_kind_z == FACE_LIQ_MINUS) | (geom.face_kind_z == FACE_LIQ_PLUS)
    if sigma == 0.0:
        return np.where(mixed_r, p_atm, np.nan), np.where(mixed_z, p_atm, np.nan)
    if kappa_r is None or kappa_z is None:
        raise ValueError("sigma > 0 requires interface curvature at the crossings")
    if not (np.all(np.isfinite(kappa_r[mixed_r])) and np.all(np.isfinite(kappa_z[mixed_z]))):
        raise ValueError("non-finite interface curvature on a mixed face")
    pr = np.where(mixed_r, p_atm + S_KAPPA * sigma * np.where(mixed_r, kappa_r, 0.0), np.nan)
    pz = np.where(mixed_z, p_atm + S_KAPPA * sigma * np.where(mixed_z, kappa_z, 0.0), np.nan)
    return pr, pz


def ghost_face_coefficient(theta: np.ndarray, h: float) -> np.ndarray:
    """1/(theta*h): the factor multiplying (p_Gamma - p_P) in the one-sided
    interface gradient."""
    return 1.0 / (theta * h)


def ghost_face_gradient(p_liquid: np.ndarray, p_gamma: np.ndarray, theta: np.ndarray,
                         h: float, liquid_on_minus_side: bool) -> np.ndarray:
    """dp/dx on a mixed face, x increasing with the grid index.

    liquid_on_minus_side=True (FACE_LIQ_MINUS): the interface lies in +x
    from P, gradient = (p_Gamma - p_P)/(theta h).
    liquid_on_minus_side=False (FACE_LIQ_PLUS): interface in -x from P,
    gradient = (p_P - p_Gamma)/(theta h)."""
    c = ghost_face_coefficient(theta, h)
    if liquid_on_minus_side:
        return c * (p_gamma - p_liquid)
    return c * (p_liquid - p_gamma)
