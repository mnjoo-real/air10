"""Liquid / interface / void classification for the single-phase Level-1A
path (README_rewritten sections 9.1, 14.2).

Sign convention (unchanged from the legacy solver):

    phi < 0  : liquid (solved)
    phi >= 0 : void / atmospheric side (NOT solved)
    phi = 0  : free surface

A cell is *liquid* iff ``phi < 0`` at its center, so a cell whose center
sits exactly on the interface is void -- the crossing fraction below is
then exactly 1 when seen from its liquid neighbor, never 0.

Sub-cell geometry
-----------------
Along the grid line joining a liquid center P (phi_P < 0) and a void
center Q (phi_Q >= 0), the free surface is located by the linear zero
crossing of phi:

    theta = phi_P / (phi_P - phi_Q)   in (0, 1]

measured from P, in units of the center-to-center spacing h. The interface
is NEVER snapped to the face between P and Q (theta=0.5) or to a center.

Robustness for theta -> 0
-------------------------
The ghost-fluid pressure coefficient (free_surface_bc.py) scales like
1/theta. Physically nothing is singular: as theta -> 0 the Dirichlet point
approaches the node P, and the discrete equation for P degenerates smoothly
into p_P = p_Gamma. Numerically, however, the face gradient
(p_Gamma - p_P)/(theta*h) becomes a cancellation of two O(p) numbers
divided by theta*h. We therefore use

    theta_eff = max(theta, THETA_MIN),    THETA_MIN = 1e-6.

This is equivalent to moving the Dirichlet point by at most THETA_MIN*h
toward Q, i.e. an interface-position perturbation <= 1e-6*h, which
perturbs p_P by <= 1e-6 * h * |grad p| -- six orders below the O(h)
cell size and far below the O(h^2) truncation error of the scheme for any
grid used in this project. It is NOT a clip that hides a failure: the
floor is only ever active when the interface is within 1e-6 cells of a
node, and tests/test_single_phase_pressure.py measures the error for
theta down to 1e-12 explicitly.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .grid import Grid

THETA_MIN = 1.0e-6

# face kinds
FACE_INACTIVE = 0       # void-void, or a boundary face (Dirichlet velocity)
FACE_LIQUID = 1         # liquid on both sides
FACE_LIQ_MINUS = 2      # liquid on the low-index side, void on the high side
FACE_LIQ_PLUS = 3       # void on the low-index side, liquid on the high side

# cell classes
CELL_VOID = 0
CELL_INTERFACE = 1      # liquid, with at least one void 4-neighbor
CELL_LIQUID = 2         # liquid, all in-domain 4-neighbors liquid


def crossing_fraction(phi_liquid: np.ndarray, phi_void: np.ndarray) -> np.ndarray:
    """theta = phi_l / (phi_l - phi_v), the linear zero crossing measured
    from the liquid point, for phi_l < 0 <= phi_v. Returned *unfloored*."""
    phi_liquid = np.asarray(phi_liquid, dtype=float)
    phi_void = np.asarray(phi_void, dtype=float)
    return phi_liquid / (phi_liquid - phi_void)


@dataclass
class LiquidGeometry:
    """Per-step geometric classification derived from phi alone. Both the
    pressure-matrix assembly and the projection read the SAME instance, so
    they cannot disagree about where the interface is."""
    liquid: np.ndarray        # (Nr, Nz) bool
    cell_class: np.ndarray    # (Nr, Nz) int8, CELL_*
    face_kind_r: np.ndarray   # (Nr+1, Nz) int8, FACE_*
    face_kind_z: np.ndarray   # (Nr, Nz+1) int8, FACE_*
    theta_r: np.ndarray       # (Nr+1, Nz) floored theta on mixed faces, nan elsewhere
    theta_z: np.ndarray       # (Nr, Nz+1)
    theta_r_raw: np.ndarray   # unfloored, for diagnostics
    theta_z_raw: np.ndarray

    @property
    def n_liquid(self) -> int:
        return int(self.liquid.sum())

    @property
    def n_floored(self) -> int:
        """Number of mixed faces where THETA_MIN was active."""
        raw = np.concatenate([self.theta_r_raw[np.isfinite(self.theta_r_raw)],
                               self.theta_z_raw[np.isfinite(self.theta_z_raw)]])
        return int(np.sum(raw < THETA_MIN))

    def ur_face_known(self) -> np.ndarray:
        """u_r faces whose value comes from the liquid projection (at least
        one adjacent liquid cell), plus the two Dirichlet boundary columns."""
        known = self.face_kind_r != FACE_INACTIVE
        known[0, :] = True
        known[-1, :] = True
        return known

    def uz_face_known(self) -> np.ndarray:
        known = self.face_kind_z != FACE_INACTIVE
        known[:, 0] = True
        return known


def classify(phi: np.ndarray) -> LiquidGeometry:
    Nr, Nz = phi.shape
    liquid = phi < 0.0

    kind_r = np.zeros((Nr + 1, Nz), dtype=np.int8)
    theta_r_raw = np.full((Nr + 1, Nz), np.nan)
    lm, lp = liquid[:-1, :], liquid[1:, :]
    pm, pp = phi[:-1, :], phi[1:, :]
    inner = kind_r[1:Nr, :]
    inner[lm & lp] = FACE_LIQUID
    inner[lm & ~lp] = FACE_LIQ_MINUS
    inner[~lm & lp] = FACE_LIQ_PLUS
    th = theta_r_raw[1:Nr, :]
    m = lm & ~lp
    th[m] = crossing_fraction(pm[m], pp[m])
    m = ~lm & lp
    th[m] = crossing_fraction(pp[m], pm[m])

    kind_z = np.zeros((Nr, Nz + 1), dtype=np.int8)
    theta_z_raw = np.full((Nr, Nz + 1), np.nan)
    lm, lp = liquid[:, :-1], liquid[:, 1:]
    pm, pp = phi[:, :-1], phi[:, 1:]
    inner = kind_z[:, 1:Nz]
    inner[lm & lp] = FACE_LIQUID
    inner[lm & ~lp] = FACE_LIQ_MINUS
    inner[~lm & lp] = FACE_LIQ_PLUS
    th = theta_z_raw[:, 1:Nz]
    m = lm & ~lp
    th[m] = crossing_fraction(pm[m], pp[m])
    m = ~lm & lp
    th[m] = crossing_fraction(pp[m], pm[m])

    # cell class: interface if any in-domain neighbor is void
    void_nb = np.zeros_like(liquid)
    void_nb[1:, :] |= ~liquid[:-1, :]
    void_nb[:-1, :] |= ~liquid[1:, :]
    void_nb[:, 1:] |= ~liquid[:, :-1]
    void_nb[:, :-1] |= ~liquid[:, 1:]
    cell_class = np.full((Nr, Nz), CELL_VOID, dtype=np.int8)
    cell_class[liquid] = CELL_LIQUID
    cell_class[liquid & void_nb] = CELL_INTERFACE

    theta_r = np.where(np.isfinite(theta_r_raw), np.maximum(theta_r_raw, THETA_MIN), np.nan)
    theta_z = np.where(np.isfinite(theta_z_raw), np.maximum(theta_z_raw, THETA_MIN), np.nan)

    return LiquidGeometry(liquid=liquid, cell_class=cell_class,
                          face_kind_r=kind_r, face_kind_z=kind_z,
                          theta_r=theta_r, theta_z=theta_z,
                          theta_r_raw=theta_r_raw, theta_z_raw=theta_z_raw)


def liquid_volume_staircase(grid: Grid, phi: np.ndarray) -> float:
    """Identical definition to solver.water_volume (cells with phi<0)."""
    cell_volume = 2.0 * np.pi * grid.r_c[:, None] * grid.dr * grid.dz
    return float(np.sum((phi < 0) * cell_volume))


def liquid_volume_subcell(grid: Grid, phi: np.ndarray) -> float:
    """Sub-cell liquid volume: per-cell liquid fraction
    clip(1/2 - phi/L_n, 0, 1), L_n = |n_r| dr + |n_z| dz (the cell's extent
    along the interface normal). Exact for a planar grid-aligned interface
    with signed-distance phi and far less quantized than the staircase
    count, so volume drift is not dominated by single-cell flips."""
    from .operators import center_grad_r, center_grad_z
    gr = center_grad_r(grid, phi)
    gz = center_grad_z(grid, phi)
    mag = np.sqrt(gr**2 + gz**2) + 1e-300
    L_n = (np.abs(gr) * grid.dr + np.abs(gz) * grid.dz) / mag
    # |grad phi| may differ from 1 (non-signed-distance phi); use the
    # distance estimate phi/|grad phi|
    frac = np.clip(0.5 - (phi / mag) / L_n, 0.0, 1.0)
    cell_volume = 2.0 * np.pi * grid.r_c[:, None] * grid.dr * grid.dz
    return float(np.sum(frac * cell_volume))
