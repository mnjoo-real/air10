"""Axisymmetric interface curvature for the single-phase sharp capillary
pressure jump (README_rewritten 5.5, 10; Gate V4).

Convention (verified by tests/test_curvature_single_phase.py):

    n = grad(phi)/|grad(phi)|   points liquid (phi<0) -> air (phi>0)
    kappa = div(n)              = sum of principal curvatures of the 3D
                                  surface of revolution
    sphere, liquid inside:      kappa = +2/R
    vertical cylinder r = R:    kappa = +1/R   (only the hoop term)
    bowl z = a r^2, liquid below: kappa = -4a on the axis

In (r, z), div(n) = d(n_r)/dr + n_r/r + d(n_z)/dz. It is evaluated from phi
derivatives in closed form:

    kappa = (phi_rr phi_z^2 - 2 phi_r phi_z phi_rz + phi_zz phi_r^2) / |grad phi|^3
            + phi_r / (r |grad phi|)

with second-order central differences at cell centers.

Axis. The cell-centered grid has no node on r = 0: the first column sits at
r_c[0] = dr/2. The hoop term phi_r/(r|grad phi|) is therefore always
evaluated at r >= dr/2. The r-derivatives there use the even mirror ghost
phi(-r_c[0]) = phi(r_c[0]) (every axisymmetric scalar is even in r). For
phi = f(z) + b r^2 this gives phi_r(r_c[0]) = 2 b r_c[0] and phi_rr = 2b
EXACTLY, so n_r/r -> d(n_r)/dr near the axis and the discrete curvature in
the first column reproduces the symmetric limit 2 d(n_r)/dr + d(n_z)/dz to
O(dr^2) without a separate formula. This is tested in isolation
(test_axis_curvature_exact_for_even_quadratic).

Wall/top/bottom ghosts use linear extrapolation. The V4 benchmarks keep the
interface away from them; wall contact is V4b.

Interface value. The pressure BC needs kappa at each sub-cell crossing.
Following the ghost-fluid literature (Kang, Fedkiw & Liu 2000), kappa is
interpolated linearly along the grid line between the liquid node P and
the void node Q at the SAME fraction theta that the pressure BC uses
(liquid_mask.classify):

    kappa_Gamma = (1 - theta) kappa_P + theta kappa_Q.

No clipping or smoothing is applied; curvature error is left visible.
"""
from __future__ import annotations

import numpy as np

from .grid import Grid
from .liquid_mask import FACE_LIQ_MINUS, FACE_LIQ_PLUS, LiquidGeometry

_EPS_GRAD = 1e-12


def _pad(phi: np.ndarray, wall_ghost: np.ndarray | None = None) -> np.ndarray:
    """One ghost layer: even mirror at the axis (low r); at the wall the
    contact-geometry ghost (contact_angle.wall_ghost_column; linear
    extrapolation when None); linear extrapolation at bottom and top.
    Corners follow from padding r first, then z."""
    lo_r = phi[:1, :]
    hi_r = (2 * phi[-1:, :] - phi[-2:-1, :]) if wall_ghost is None else wall_ghost[None, :]
    p = np.concatenate([lo_r, phi, hi_r], axis=0)
    lo_z = 2 * p[:, :1] - p[:, 1:2]
    hi_z = 2 * p[:, -1:] - p[:, -2:-1]
    return np.concatenate([lo_z, p, hi_z], axis=1)


def curvature_components_centers(grid: Grid, phi: np.ndarray, wall_cfg=None
                                 ) -> tuple[np.ndarray, np.ndarray]:
    """(kappa_meridional, kappa_azimuthal) at cell centers; their sum is
    div(n). The meridional part is the in-plane curvature of the r-z
    contour, the azimuthal ('hoop') part is n_r / r."""
    dr, dz = grid.dr, grid.dz
    ghost = None
    if wall_cfg is not None:
        from .contact_angle import wall_ghost_column
        ghost = wall_ghost_column(grid, phi, wall_cfg)
    P = _pad(phi, ghost)
    c = P[1:-1, 1:-1]
    e, w = P[2:, 1:-1], P[:-2, 1:-1]
    n, s = P[1:-1, 2:], P[1:-1, :-2]
    ne, nw, se, sw = P[2:, 2:], P[:-2, 2:], P[2:, :-2], P[:-2, :-2]
    pr = (e - w) / (2 * dr)
    pz = (n - s) / (2 * dz)
    prr = (e - 2 * c + w) / dr**2
    pzz = (n - 2 * c + s) / dz**2
    prz = (ne - nw - se + sw) / (4 * dr * dz)
    g2 = pr**2 + pz**2
    g = np.sqrt(g2) + _EPS_GRAD
    meridional = (prr * pz**2 - 2 * pr * pz * prz + pzz * pr**2) / g**3
    hoop = pr / (grid.r_c[:, None] * g)
    return meridional, hoop


def curvature_centers(grid: Grid, phi: np.ndarray, wall_cfg=None) -> np.ndarray:
    """kappa = div(grad phi / |grad phi|) (axisymmetric) at cell centers."""
    m, h = curvature_components_centers(grid, phi, wall_cfg)
    return m + h


def curvature_at_crossings(grid: Grid, geom: LiquidGeometry, kappa_c: np.ndarray
                            ) -> tuple[np.ndarray, np.ndarray]:
    """kappa at every mixed-face crossing on the u_r and u_z face grids
    (nan elsewhere), interpolated along the grid line at the pressure-BC
    theta."""
    Nr, Nz = grid.Nr, grid.Nz
    kr = np.full((Nr + 1, Nz), np.nan)
    kz = np.full((Nr, Nz + 1), np.nan)

    kind = geom.face_kind_r[1:Nr, :]
    th = geom.theta_r_raw[1:Nr, :]
    km, kp = kappa_c[:-1, :], kappa_c[1:, :]
    out = kr[1:Nr, :]
    m = kind == FACE_LIQ_MINUS          # liquid at minus side
    out[m] = ((1 - th) * km + th * kp)[m]
    m = kind == FACE_LIQ_PLUS
    out[m] = ((1 - th) * kp + th * km)[m]

    kind = geom.face_kind_z[:, 1:Nz]
    th = geom.theta_z_raw[:, 1:Nz]
    km, kp = kappa_c[:, :-1], kappa_c[:, 1:]
    out = kz[:, 1:Nz]
    m = kind == FACE_LIQ_MINUS
    out[m] = ((1 - th) * km + th * kp)[m]
    m = kind == FACE_LIQ_PLUS
    out[m] = ((1 - th) * kp + th * km)[m]
    return kr, kz


def interface_curvature(grid: Grid, geom: LiquidGeometry, phi: np.ndarray, wall_cfg=None
                         ) -> tuple[np.ndarray, np.ndarray]:
    return curvature_at_crossings(grid, geom, curvature_centers(grid, phi, wall_cfg))


def principal_resolution(grid: Grid, geom: LiquidGeometry, phi: np.ndarray, wall_cfg=None,
                         kappa_floor: float = 1e-12) -> dict:
    """Resolution of the two principal curvatures at the interface crossings:
    N_m = R_m/dx, N_theta = R_theta/dx with R = 1/|kappa_component|
    (|kappa| floored, so flat directions give a very large N). The minimum
    over the interface is what a resolution criterion must use, not
    1/|kappa_total|."""
    km_c, kh_c = curvature_components_centers(grid, phi, wall_cfg)
    dx = min(grid.dr, grid.dz)
    out = {}
    for name, kc in (("meridional", km_c), ("azimuthal", kh_c)):
        kr, kz = curvature_at_crossings(grid, geom, kc)
        k = np.concatenate([kr[np.isfinite(kr)], kz[np.isfinite(kz)]])
        n_cells = 1.0 / np.maximum(np.abs(k), kappa_floor) / dx
        out[name] = {"min_cells": float(n_cells.min()), "median_cells": float(np.median(n_cells))}
    return out


def crossing_positions(grid: Grid, geom: LiquidGeometry) -> tuple[np.ndarray, np.ndarray,
                                                                   np.ndarray, np.ndarray]:
    """(r, z) of every mixed-face crossing, as arrays on the u_r and u_z
    face grids (nan elsewhere) -- for evaluating analytic references at
    exactly the points where p_Gamma is imposed."""
    Nr, Nz = grid.Nr, grid.Nz
    rr = np.full((Nr + 1, Nz), np.nan)
    zr = np.full((Nr + 1, Nz), np.nan)
    th = geom.theta_r_raw[1:Nr, :]
    kind = geom.face_kind_r[1:Nr, :]
    rm, rp = grid.r_c[:-1, None], grid.r_c[1:, None]
    pos = np.where(kind == FACE_LIQ_MINUS, rm + th * grid.dr,
                   np.where(kind == FACE_LIQ_PLUS, rp - th * grid.dr, np.nan))
    rr[1:Nr, :] = pos
    zr[1:Nr, :] = np.where(np.isfinite(pos), grid.z_c[None, :], np.nan)

    rz = np.full((Nr, Nz + 1), np.nan)
    zz = np.full((Nr, Nz + 1), np.nan)
    th = geom.theta_z_raw[:, 1:Nz]
    kind = geom.face_kind_z[:, 1:Nz]
    zm, zp = grid.z_c[None, :-1], grid.z_c[None, 1:]
    pos = np.where(kind == FACE_LIQ_MINUS, zm + th * grid.dz,
                   np.where(kind == FACE_LIQ_PLUS, zp - th * grid.dz, np.nan))
    zz[:, 1:Nz] = pos
    rz[:, 1:Nz] = np.where(np.isfinite(pos), grid.r_c[:, None], np.nan)
    return rr, zr, rz, zz
