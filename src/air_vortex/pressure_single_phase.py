"""Liquid-only constant-density pressure Poisson solve and projection for
the single-phase Level-1A path (README_rewritten section 6).

    lap(p) = (rho_w / dt) div(u*)        in liquid cells (phi < 0)
    p      = p_Gamma                     at the sub-cell phi=0 crossing
    dp/dn  = 0                           at axis, wall, bottom
                                         (u* . n = 0 there already)

    u^{n+1} = u* - (dt / rho_w) grad p   on every face adjacent to liquid

This module is deliberately separate from pressure.py (the legacy
variable-density solver), which is left untouched: no rho(phi), no
face-averaged 1/rho, no rectangular-domain top p=0 row. Void cells carry
no pressure unknown at all.

Discretization
--------------
Finite-volume form of the MAC divergence (operators.divergence) with each
cell's equation multiplied by its (r_c dr dz) weight, which makes the
matrix symmetric (and negative definite once at least one Dirichlet
interface face is present):

    radial face at r_f:  coefficient r_f dz / dr
    axial face:          coefficient r_c dr / dz

On a liquid-liquid face the gradient is the usual centered difference; on
a liquid-void face it is free_surface_bc.ghost_face_gradient, with the
SAME LiquidGeometry theta used in the matrix and in the projection. The
projected velocity is therefore discretely divergence-free in every
liquid cell to solver precision.

Faces with void on both sides are left untouched here; they are filled by
velocity_extension.py for Level Set transport only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy import ndimage

from .free_surface_bc import ghost_face_coefficient, ghost_face_gradient
from .grid import Grid
from .liquid_mask import (
    CELL_INTERFACE,
    FACE_LIQ_MINUS,
    FACE_LIQ_PLUS,
    FACE_LIQUID,
    LiquidGeometry,
    THETA_MIN,
    crossing_fraction,
)
from .operators import divergence

TopBC = Literal["error", "neumann"]


class LiquidReachedTopError(RuntimeError):
    """The liquid occupies the top row of the domain. Level 1A has no
    rectangular-domain atmospheric boundary (README_rewritten section 7):
    production runs need enough void headspace that this cannot happen."""


@dataclass
class PressureSystem:
    A: sp.csr_matrix          # (n_liq, n_liq), symmetric negative definite
    b_dirichlet: np.ndarray   # (n_liq,) LHS contribution c*p_Gamma of the interface faces
    index: np.ndarray         # (Nr, Nz) int, -1 for void
    weight: np.ndarray        # (Nr, Nz) r_c dr dz


def _check_solvable(geom: LiquidGeometry, top_bc: TopBC) -> None:
    liquid = geom.liquid
    if top_bc == "error" and np.any(liquid[:, -1]):
        raise LiquidReachedTopError(
            "single_phase_ls: liquid reached the top row of the computational domain; "
            "increase geometry.air_height_m (README_rewritten section 7, 'Free surface').")
    labels, n = ndimage.label(liquid)
    if n == 0:
        raise RuntimeError("single_phase_ls: no liquid cells (phi < 0 nowhere).")
    has_dirichlet = np.zeros(n + 1, dtype=bool)
    has_dirichlet[np.unique(labels[geom.cell_class == CELL_INTERFACE])] = True
    missing = [k for k in range(1, n + 1) if not has_dirichlet[k]]
    if missing:
        raise RuntimeError(
            f"single_phase_ls: {len(missing)} liquid component(s) have no free-surface "
            "Dirichlet face -- the pure-Neumann pressure problem is singular.")


def build_liquid_pressure_system(grid: Grid, geom: LiquidGeometry,
                                  p_gamma_r: np.ndarray, p_gamma_z: np.ndarray,
                                  top_bc: TopBC = "error") -> PressureSystem:
    _check_solvable(geom, top_bc)
    Nr, Nz = grid.Nr, grid.Nz
    dr, dz = grid.dr, grid.dz
    index = np.full((Nr, Nz), -1, dtype=np.int64)
    index[geom.liquid] = np.arange(geom.n_liquid)
    n = geom.n_liquid

    rows, cols, vals = [], [], []
    diag = np.zeros(n)
    b = np.zeros(n)

    # ---- radial faces, interior i_f = 1..Nr-1 between cells i_f-1 (minus) and i_f (plus)
    r_face = grid.r_f[1:Nr, None] * np.ones((1, Nz))
    c_r = r_face * dz / dr
    kind = geom.face_kind_r[1:Nr, :]
    im = index[:-1, :]
    ip = index[1:, :]

    m = kind == FACE_LIQUID
    c = c_r[m]
    a, bb = im[m], ip[m]
    np.add.at(diag, a, -c)
    np.add.at(diag, bb, -c)
    rows += [a, bb]
    cols += [bb, a]
    vals += [c, c]

    for k_side, idx_arr in ((FACE_LIQ_MINUS, im), (FACE_LIQ_PLUS, ip)):
        m = kind == k_side
        th = geom.theta_r[1:Nr, :][m]
        pg = p_gamma_r[1:Nr, :][m]
        c = c_r[m] * dr * ghost_face_coefficient(th, dr)   # = r_f dz / (theta dr)
        np.add.at(diag, idx_arr[m], -c)
        np.add.at(b, idx_arr[m], c * pg)

    # ---- axial faces, interior j_f = 1..Nz-1
    c_z_full = grid.r_c[:, None] * dr / dz * np.ones((1, Nz - 1))
    kind = geom.face_kind_z[:, 1:Nz]
    jm = index[:, :-1]
    jp = index[:, 1:]

    m = kind == FACE_LIQUID
    c = c_z_full[m]
    a, bb = jm[m], jp[m]
    np.add.at(diag, a, -c)
    np.add.at(diag, bb, -c)
    rows += [a, bb]
    cols += [bb, a]
    vals += [c, c]

    for k_side, idx_arr in ((FACE_LIQ_MINUS, jm), (FACE_LIQ_PLUS, jp)):
        m = kind == k_side
        th = geom.theta_z[:, 1:Nz][m]
        pg = p_gamma_z[:, 1:Nz][m]
        c = c_z_full[m] * dz * ghost_face_coefficient(th, dz)  # = r_c dr / (theta dz)
        np.add.at(diag, idx_arr[m], -c)
        np.add.at(b, idx_arr[m], c * pg)

    rows.append(np.arange(n))
    cols.append(np.arange(n))
    vals.append(diag)
    A = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(n, n))
    weight = grid.r_c[:, None] * dr * dz * np.ones((1, Nz))
    return PressureSystem(A=A, b_dirichlet=b, index=index, weight=weight)


def solve_liquid_pressure_from_source(grid: Grid, geom: LiquidGeometry, source: np.ndarray,
                                       p_gamma_r: np.ndarray, p_gamma_z: np.ndarray,
                                       top_bc: TopBC = "error", void_value: float = 0.0
                                       ) -> np.ndarray:
    """Solve lap(p) = source in the liquid (cell-centered source, same units
    as the discrete Laplacian of p) with the free-surface Dirichlet BC.
    Returns the full (Nr, Nz) field, void cells set to ``void_value``."""
    sys_ = build_liquid_pressure_system(grid, geom, p_gamma_r, p_gamma_z, top_bc)
    liq = geom.liquid
    rhs = (source * sys_.weight)[liq] - sys_.b_dirichlet
    # -A is SPD; the direct solve is exact up to round-off (README 14.5:
    # direct sparse solve for validation grids).
    p_liq = spla.spsolve((-sys_.A).tocsc(), -rhs)
    p = np.full((grid.Nr, grid.Nz), void_value, dtype=float)
    p[liq] = p_liq
    return p


def solve_liquid_pressure(grid: Grid, geom: LiquidGeometry, u_star_r: np.ndarray,
                           u_star_z: np.ndarray, rho: float, dt: float,
                           p_gamma_r: np.ndarray, p_gamma_z: np.ndarray,
                           top_bc: TopBC = "error") -> np.ndarray:
    """lap(p) = (rho/dt) div(u*) on liquid cells (README_rewritten 6)."""
    source = (rho / dt) * divergence(grid, u_star_r, u_star_z)
    return solve_liquid_pressure_from_source(grid, geom, source, p_gamma_r, p_gamma_z, top_bc)


def pressure_face_gradients(grid: Grid, geom: LiquidGeometry, p: np.ndarray,
                             p_gamma_r: np.ndarray, p_gamma_z: np.ndarray
                             ) -> tuple[np.ndarray, np.ndarray]:
    """Discrete grad p on every liquid-adjacent face (0 elsewhere), using
    exactly the face rules of :func:`build_liquid_pressure_system`."""
    Nr, Nz = grid.Nr, grid.Nz
    gr = np.zeros((Nr + 1, Nz))
    kind = geom.face_kind_r
    inner_kind = kind[1:Nr, :]
    g_in = gr[1:Nr, :]
    m = inner_kind == FACE_LIQUID
    g_in[m] = ((p[1:, :] - p[:-1, :]) / grid.dr)[m]
    m = inner_kind == FACE_LIQ_MINUS
    g_in[m] = ghost_face_gradient(p[:-1, :][m], p_gamma_r[1:Nr, :][m],
                                  geom.theta_r[1:Nr, :][m], grid.dr, True)
    m = inner_kind == FACE_LIQ_PLUS
    g_in[m] = ghost_face_gradient(p[1:, :][m], p_gamma_r[1:Nr, :][m],
                                  geom.theta_r[1:Nr, :][m], grid.dr, False)

    gz = np.zeros((Nr, Nz + 1))
    inner_kind = geom.face_kind_z[:, 1:Nz]
    g_in = gz[:, 1:Nz]
    m = inner_kind == FACE_LIQUID
    g_in[m] = ((p[:, 1:] - p[:, :-1]) / grid.dz)[m]
    m = inner_kind == FACE_LIQ_MINUS
    g_in[m] = ghost_face_gradient(p[:, :-1][m], p_gamma_z[:, 1:Nz][m],
                                  geom.theta_z[:, 1:Nz][m], grid.dz, True)
    m = inner_kind == FACE_LIQ_PLUS
    g_in[m] = ghost_face_gradient(p[:, 1:][m], p_gamma_z[:, 1:Nz][m],
                                  geom.theta_z[:, 1:Nz][m], grid.dz, False)
    return gr, gz


def project_liquid_velocity(grid: Grid, geom: LiquidGeometry, u_star_r: np.ndarray,
                             u_star_z: np.ndarray, p: np.ndarray, rho: float, dt: float,
                             p_gamma_r: np.ndarray, p_gamma_z: np.ndarray
                             ) -> tuple[np.ndarray, np.ndarray]:
    """u = u* - (dt/rho) grad p on liquid-adjacent faces; boundary faces
    (axis, wall, bottom) are Dirichlet 0; void-void faces are returned
    unchanged from u* (to be overwritten by velocity extension)."""
    gr, gz = pressure_face_gradients(grid, geom, p, p_gamma_r, p_gamma_z)
    u_r = u_star_r - (dt / rho) * gr
    u_z = u_star_z - (dt / rho) * gz
    u_r[0, :] = 0.0
    u_r[-1, :] = 0.0
    u_z[:, 0] = 0.0
    return u_r, u_z


def liquid_divergence(grid: Grid, geom: LiquidGeometry, u_r: np.ndarray, u_z: np.ndarray
                       ) -> np.ndarray:
    """div(u) on liquid cells, 0 on void cells (the extension band is not
    divergence-free and is not physical flow)."""
    return np.where(geom.liquid, divergence(grid, u_r, u_z), 0.0)


# ---------------------------------------------------------------------------
# 1D Cartesian reference implementation (verification only, README task
# "Test P1/P2"): same crossing / ghost-coefficient helpers as the 2D solver.
# ---------------------------------------------------------------------------


def solve_poisson_1d_ghost_fluid(n: int, h: float, f: np.ndarray, x_gamma: float,
                                  p_gamma: float, flux_left: float) -> tuple[np.ndarray, np.ndarray]:
    """p'' = f on [0, x_gamma), cells centered at x_k = (k+1/2) h, liquid
    where phi = x - x_gamma < 0, Neumann p'(0) = flux_left at the left
    boundary face, Dirichlet p(x_gamma) = p_gamma at the sub-cell crossing.
    Returns (x_c, p) with nan in void cells."""
    x_c = (np.arange(n) + 0.5) * h
    phi = x_c - x_gamma
    liquid = phi < 0
    m = int(liquid.sum())
    if m == 0 or m == n:
        raise ValueError("interface must lie strictly inside the 1D domain")
    theta = max(float(crossing_fraction(phi[m - 1], phi[m])), THETA_MIN)

    A = np.zeros((m, m))
    b = np.asarray(f[:m], dtype=float) * h  # equation multiplied by h
    for k in range(m):
        # west face
        if k == 0:
            b[k] -= -flux_left   # (G_e - G_w): G_w = flux_left known
        else:
            A[k, k - 1] += 1.0 / h
            A[k, k] -= 1.0 / h
        # east face
        if k == m - 1:
            c = ghost_face_coefficient(theta, h)
            A[k, k] -= c
            b[k] -= c * p_gamma
        else:
            A[k, k + 1] += 1.0 / h
            A[k, k] -= 1.0 / h
    p = np.full(n, np.nan)
    p[:m] = np.linalg.solve(A, b)
    return x_c, p
