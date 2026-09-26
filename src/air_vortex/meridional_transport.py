"""Opt-in conservative (flux-form) advection of the meridional velocity on the MAC grid
(single_phase_height, V7-T). Replaces u.grad(u_r), u.grad(u_z) of the predictor by

    N_r = (1/r) d(r u_r u_r)/dr + d(u_z u_r)/dz,    N_z = (1/r) d(r u_r u_z)/dr + d(u_z u_z)/dz

(equal to u.grad(u_r), u.grad(u_z) when div u = 0; the -u_theta^2/r term stays separate),
integrated over the staggered control volumes:

u_r CV at (r_f[i], z_c[j]), r in [r_c[i-1], r_c[i]]:
    radial mass flux at r_c[k]:  M = (r_f[k] u_r[k] + r_f[k+1] u_r[k+1]) / 2
    axial mass flux at z_f[j]:   W = (r_c[i-1] u_z[i-1] + r_c[i] u_z[i]) / (2 r_f[i])
u_z CV at (r_c[i], z_f[j]), z in [z_c[j-1], z_c[j]]:
    radial mass flux at r_f[i]:  M = r_f[i] (u_r[i, j-1] + u_r[i, j]) / 2
    axial mass flux at z_c[k]:   W = (u_z[k] + u_z[k+1]) / 2

With these r-weighted averages the discrete divergence of every CV is the average of the two
adjacent cell divergences, so it vanishes exactly for a projected field: a uniform advected
value gives N = 0 (no geometric momentum source). Transported face values: upwind (order 1)
or limited MUSCL (order 2) via swirl_transport.face_values. Ghosts: u_r odd at the axis and
the wall (u_r = 0 there), odd at the no-slip bottom, even at the top; u_z even at the axis,
odd at the no-slip side wall and bottom, even at the top. The advected field is updated with
the mass fluxes of the step-start velocity held fixed (SSPRK2 for order 2).
"""
from __future__ import annotations

import numpy as np

from .swirl_transport import face_values

MERIDIONAL_ADVECTION_MODES = ("advective", "conservative_upwind1", "conservative_muscl2")


def _pad(a, axis, lo, hi, on_node_lo, on_node_hi):
    """Two ghosts per side along ``axis``; lo/hi = +1 even or -1 odd mirror. ``on_node``: the
    boundary coincides with the first/last node (mirror about that node) instead of a face."""
    a0 = np.moveaxis(a, axis, 0)
    if on_node_lo:
        g_lo = lo * a0[2:0:-1]
    else:
        g_lo = lo * a0[1::-1]
    if on_node_hi:
        g_hi = hi * a0[-2:-4:-1]
    else:
        g_hi = hi * a0[:-3:-1]
    return np.moveaxis(np.concatenate([g_lo, a0, g_hi], axis=0), 0, axis)


def mass_fluxes(grid, u_r, u_z):
    rf, rc = grid.r_f, grid.r_c
    m = rf[:, None] * u_r                                     # r u_r on u_r faces (Nr+1, Nz)
    M_ur = 0.5 * (m[:-1] + m[1:])                             # at cell centres (Nr, Nz)
    W_ur = np.zeros((grid.Nr + 1, grid.Nz + 1))               # at (r_f[i], z_f[j])
    W_ur[1:-1] = (rc[:-1, None] * u_z[:-1] + rc[1:, None] * u_z[1:]) / (2.0 * rf[1:-1, None])
    M_uz = np.zeros((grid.Nr + 1, grid.Nz + 1))               # at (r_f[i], z_f[j])
    M_uz[:, 1:-1] = 0.5 * (m[:, :-1] + m[:, 1:])
    W_uz = 0.5 * (u_z[:, :-1] + u_z[:, 1:])                   # at cell centres (Nr, Nz)
    return M_ur, W_ur, M_uz, W_uz


def advection_terms(grid, a_r, a_z, fluxes, order=2, limiter="vanleer"):
    """N_r on u_r faces (Nr+1, Nz) and N_z on u_z faces (Nr, Nz+1) for advected fields
    a_r (u_r layout) and a_z (u_z layout) with the given mass fluxes. Boundary rows
    (axis/wall u_r, bottom/top u_z) are returned as zero (Dirichlet in the predictor)."""
    M_ur, W_ur, M_uz, W_uz = fluxes
    dr, dz = grid.dr, grid.dz
    # ---- u_r: radial fluxes at cell centres = interior "faces" between u_r nodes
    p = _pad(a_r, 0, -1.0, -1.0, True, True)
    fr = face_values(p, np.concatenate([M_ur[:1], M_ur, M_ur[-1:]], axis=0), order, limiter, 0)[1:-1]
    Fr = M_ur * fr                                            # (Nr, Nz)
    p = _pad(a_r, 1, -1.0, 1.0, False, False)
    fz = face_values(p, W_ur, order, limiter, 1)              # (Nr+1, Nz+1)
    Fz = W_ur * fz
    N_r = np.zeros_like(a_r)
    N_r[1:-1] = (Fr[1:] - Fr[:-1]) / (grid.r_f[1:-1, None] * dr) + (Fz[1:-1, 1:] - Fz[1:-1, :-1]) / dz
    # ---- u_z: axial fluxes at cell centres = interior "faces" between u_z nodes
    p = _pad(a_z, 1, -1.0, 1.0, True, True)
    fz = face_values(p, np.concatenate([W_uz[:, :1], W_uz, W_uz[:, -1:]], axis=1), order, limiter, 1)[:, 1:-1]
    Gz = W_uz * fz                                            # (Nr, Nz)
    p = _pad(a_z, 0, 1.0, -1.0, False, False)
    fr = face_values(p, M_uz, order, limiter, 0)              # (Nr+1, Nz+1)
    Gr = M_uz * fr
    N_z = np.zeros_like(a_z)
    N_z[:, 1:-1] = (Gr[1:, 1:-1] - Gr[:-1, 1:-1]) / (grid.r_c[:, None] * dr) + (Gz[:, 1:] - Gz[:, :-1]) / dz
    return N_r, N_z


def advect_meridional(grid, u_r, u_z, dt, order=2, limiter="vanleer"):
    """Advected (u_r, u_z) after dt with the step-start mass fluxes frozen: forward Euler for
    order 1, SSPRK2 for order 2. Returns the equivalent advection rates (N_r, N_z) such that
    u_adv = u - dt N (so the predictor can add the remaining terms unchanged)."""
    fl = mass_fluxes(grid, u_r, u_z)
    Nr1, Nz1 = advection_terms(grid, u_r, u_z, fl, order, limiter)
    if order == 1:
        return Nr1, Nz1
    r1, z1 = u_r - dt * Nr1, u_z - dt * Nz1
    Nr2, Nz2 = advection_terms(grid, r1, z1, fl, order, limiter)
    return 0.5 * (Nr1 + Nr2), 0.5 * (Nz1 + Nz2)
