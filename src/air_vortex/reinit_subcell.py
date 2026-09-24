"""Interface-preserving reinitialization with the Russo-Smereka subcell fix
(single_phase_ls; README_rewritten 9.4).

    phi_tau + sgn(phi0) (|grad phi| - 1) = 0,    phi(tau=0) = phi0

Sources (equations taken from the papers, not from memory):

- G. Russo, P. Smereka, "A remark on computing distance functions",
  JCP 163 (2000) 51-67. The first-order 1D subcell fix is quoted verbatim
  in A. du Chene, C. Min, F. Gibou, J. Sci. Comput. 35 (2008) 114, sec. 4:
  at a node whose phi0 changes sign with a neighbor, the one-sided
  difference toward that neighbor uses phi = 0 AT THE INTERFACE POINT
  instead of the neighbor's value.
- The dimension-by-dimension 2D form used here is C. Min, F. Gibou, JCP
  225 (2007) 300 ("a slight improvement of Russo and Smereka"), stated
  equation-by-equation in C. Min, JCP 229 (2010) 2764, sec. 2.2-3:

    D+_x phi_ij = (phi_{i+1,j} - phi_ij)/dx - dx/2 minmod(Dxx phi_ij, Dxx phi_{i+1,j})
    D-_x phi_ij = (phi_ij - phi_{i-1,j})/dx + dx/2 minmod(Dxx phi_ij, Dxx phi_{i-1,j})

  and, if phi0_ij * phi0_{i+1,j} < 0 (interface at distance dx+ from x_i),

    D+_x phi_ij = (0 - phi_ij)/dx+ - dx+/2 minmod(Dxx phi_ij, Dxx phi_{i+1,j})

  (symmetrically for D-_x with dx-). dx+ is the root of the quadratic ENO
  interpolant of phi0 through i-1, i, i+1, i+2:

    dx+ = dx (1/2 + (phi0_i - phi0_{i+1} - sgn(phi0_i - phi0_{i+1}) sqrt(D)) / phi0_xx)
          if |phi0_xx| > eps,  else  dx phi0_i / (phi0_i - phi0_{i+1}),
    phi0_xx = minmod(phi0_{i-1} - 2 phi0_i + phi0_{i+1}, phi0_i - 2 phi0_{i+1} + phi0_{i+2}),
    D = (phi0_xx/2 - phi0_i - phi0_{i+1})^2 - 4 phi0_i phi0_{i+1},   eps = 1e-10.

  Godunov Hamiltonian H_G(a=D+x, b=D-x, c=D+y, d=D-y):
    sgn(phi0) >= 0: sqrt(max((a^-)^2, (b^+)^2) + max((c^-)^2, (d^+)^2))
    sgn(phi0) <  0: sqrt(max((a^+)^2, (b^-)^2) + max((c^+)^2, (d^-)^2))
  sgn is the SHARP sign of phi0 (1, 0, -1) -- no smoothing.
  Local pseudo-step dt_ij = cfl * min(dx+, dx-, dy+, dy-), cfl = 0.45 (2D);
  TVD-RK2 in pseudo-time.

``order=1`` drops the minmod terms and uses the LINEAR crossing -- that is
exactly liquid_mask.crossing_fraction, the same sub-cell geometry the
pressure BC uses. ``order=2`` is the full Min-Gibou scheme above; its
quadratic crossing falls back to the same linear crossing when phi0 is
locally linear, D < 0, or the root leaves the cell.

Why this preserves phi = 0: at every node adjacent to the interface the
stencil sees phi0's own crossing as a fixed Dirichlet point (phi = 0), and
the stationary state satisfies |grad phi| = 1 with THOSE crossing
distances. phi0 (hence sgn, the crossings and dt) is frozen for the whole
call, so the zero set cannot be advected by the iteration.

Boundaries: the axis is a mirror (phi even in r). The wall, bottom and top
use linear extrapolation ghosts, which reproduce the current slope rather
than imposing a 90-degree contact.
"""
from __future__ import annotations

import numpy as np

from .grid import Grid
from .liquid_mask import THETA_MIN, crossing_fraction

_EPS_QUAD = 1e-10


def _minmod(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.where(a * b > 0, np.sign(a) * np.minimum(np.abs(a), np.abs(b)), 0.0)


def _pad(phi: np.ndarray, axis: int, low: str) -> np.ndarray:
    """Two ghost layers along ``axis``: low side 'mirror' (axis) or
    'linear'; high side always 'linear'."""
    f = np.moveaxis(phi, axis, 0)
    if low == "mirror":
        lo = f[[1, 0]]
    else:
        lo = np.stack([3 * f[0] - 2 * f[1], 2 * f[0] - f[1]])
    hi = np.stack([2 * f[-1] - f[-2], 3 * f[-1] - 2 * f[-2]])
    return np.moveaxis(np.concatenate([lo, f, hi], axis=0), 0, axis)


def quadratic_crossing_fraction(a0, a1, a_behind, a_beyond) -> np.ndarray:
    """Fraction (of the node spacing, from the node with value a0 toward
    its neighbor a1, a0*a1 < 0) of the zero of the quadratic ENO
    interpolant (Min 2010 sec. 2.2). Falls back to the linear crossing
    (liquid_mask.crossing_fraction) where the quadratic is degenerate."""
    lin = crossing_fraction(a0, a1)
    pxx = _minmod(a_behind - 2 * a0 + a1, a0 - 2 * a1 + a_beyond)
    disc = (pxx / 2 - a0 - a1) ** 2 - 4 * a0 * a1
    use_q = (np.abs(pxx) > _EPS_QUAD) & (disc >= 0)
    safe_pxx = np.where(use_q, pxx, 1.0)
    diff = a0 - a1
    theta_q = 0.5 + (diff - np.sign(diff) * np.sqrt(np.where(use_q, disc, 0.0))) / safe_pxx
    ok = use_q & (theta_q > 0) & (theta_q <= 1)
    return np.where(ok, theta_q, lin)


def _side_geometry(phi0: np.ndarray, axis: int, low: str, h: float, order: int):
    """Interface masks and subcell distances (d_plus, d_minus) along ``axis``
    at every node, from the frozen phi0."""
    P = _pad(phi0, axis, low)
    n = phi0.shape[axis]

    def sl(k0):
        idx = [slice(None)] * 2
        idx[axis] = slice(k0 + 2, k0 + 2 + n)
        return P[tuple(idx)]

    c, p1, m1, p2, m2 = sl(0), sl(1), sl(-1), sl(2), sl(-2)
    # sign changes only with IN-DOMAIN neighbors
    inside_p = np.ones_like(phi0, dtype=bool)
    inside_m = np.ones_like(phi0, dtype=bool)
    ip = [slice(None)] * 2
    ip[axis] = -1
    inside_p[tuple(ip)] = False
    im = [slice(None)] * 2
    im[axis] = 0
    inside_m[tuple(im)] = False
    if low == "mirror":
        # across the axis the mirror neighbor is the node itself: no crossing
        pass
    mask_p = inside_p & (c * p1 < 0)
    mask_m = inside_m & (c * m1 < 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        if order == 2:
            th_p = quadratic_crossing_fraction(c, p1, m1, p2)
            th_m = quadratic_crossing_fraction(c, m1, p1, m2)
        else:
            th_p = crossing_fraction(c, p1)
            th_m = crossing_fraction(c, m1)
    d_p = np.where(mask_p, np.maximum(th_p, THETA_MIN) * h, h)
    d_m = np.where(mask_m, np.maximum(th_m, THETA_MIN) * h, h)
    return mask_p, mask_m, d_p, d_m


def _one_sided(phi: np.ndarray, axis: int, low: str, h: float, order: int,
               mask_p, mask_m, d_p, d_m):
    P = _pad(phi, axis, low)
    n = phi.shape[axis]

    def sl(k0):
        idx = [slice(None)] * 2
        idx[axis] = slice(k0 + 2, k0 + 2 + n)
        return P[tuple(idx)]

    c, p1, m1 = sl(0), sl(1), sl(-1)
    Dp = (p1 - c) / h
    Dm = (c - m1) / h
    Dp_sub = (0.0 - c) / d_p
    Dm_sub = (c - 0.0) / d_m
    if order == 2:
        p2, m2 = sl(2), sl(-2)
        xx_c = (m1 - 2 * c + p1) / h**2
        xx_p = (c - 2 * p1 + p2) / h**2
        xx_m = (m2 - 2 * m1 + c) / h**2
        mm_p = _minmod(xx_c, xx_p)
        mm_m = _minmod(xx_c, xx_m)
        Dp = Dp - 0.5 * h * mm_p
        Dm = Dm + 0.5 * h * mm_m
        Dp_sub = Dp_sub - 0.5 * d_p * mm_p
        Dm_sub = Dm_sub + 0.5 * d_m * mm_m
    return np.where(mask_p, Dp_sub, Dp), np.where(mask_m, Dm_sub, Dm)


def reinitialize_subcell(phi0: np.ndarray, grid: Grid, n_iter: int, cfl: float = 0.45,
                         order: int = 2, band_cells: float = 0.0) -> np.ndarray:
    """Russo-Smereka / Min-Gibou subcell-fix reinitialization (module
    docstring). ``band_cells`` > 0 updates only nodes with
    |phi0| <= band_cells * min(dr, dz); other nodes are left untouched."""
    if order not in (1, 2):
        raise ValueError("order must be 1 or 2")
    dr, dz = grid.dr, grid.dz
    sgn = np.sign(phi0)
    mp_r, mm_r, dp_r, dm_r = _side_geometry(phi0, 0, "mirror", dr, order)
    mp_z, mm_z, dp_z, dm_z = _side_geometry(phi0, 1, "linear", dz, order)
    dtau = cfl * np.minimum.reduce([dp_r, dm_r, dp_z, dm_z])
    if band_cells > 0:
        dtau = np.where(np.abs(phi0) <= band_cells * min(dr, dz), dtau, 0.0)

    def rhs(phi):
        a, b = _one_sided(phi, 0, "mirror", dr, order, mp_r, mm_r, dp_r, dm_r)
        c, d = _one_sided(phi, 1, "linear", dz, order, mp_z, mm_z, dp_z, dm_z)
        h_pos = np.sqrt(np.maximum(np.minimum(a, 0) ** 2, np.maximum(b, 0) ** 2)
                        + np.maximum(np.minimum(c, 0) ** 2, np.maximum(d, 0) ** 2))
        h_neg = np.sqrt(np.maximum(np.maximum(a, 0) ** 2, np.minimum(b, 0) ** 2)
                        + np.maximum(np.maximum(c, 0) ** 2, np.minimum(d, 0) ** 2))
        H = np.where(sgn >= 0, h_pos, h_neg)
        return -sgn * (H - 1.0)

    phi = phi0.astype(float).copy()
    for _ in range(n_iter):
        phi1 = phi + dtau * rhs(phi)
        phi2 = phi1 + dtau * rhs(phi1)
        phi = 0.5 * (phi + phi2)
    return phi
