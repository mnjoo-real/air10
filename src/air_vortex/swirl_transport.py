"""Conservative transport of the specific angular momentum q = r u_theta (single_phase_height).

Finite-volume form on the MAC grid, per unit cell volume (2 pi r_c dr dz):

    dq/dt = -D(q),  D(q) = (F_r[i+1] - F_r[i]) / (r_c dr) + (F_z[j+1] - F_z[j]) / dz,
    F_r = r_f u_r q_face,  F_z = u_z q_face,

with u_r, u_z the MAC face velocities, i.e. the SAME face mass fluxes (r_f u_r, u_z) that
define the discrete divergence used by the projection. Interior fluxes telescope, so
sum_ij q_ij r_c dr dz changes only through boundary fluxes (axis r_f = 0, wall F_r = 0,
bottom F_z = 0, and whatever face velocity the caller leaves at the top/interface).

Face values q_face (upwind with the sign of the face velocity):
  order 1: the upwind cell value (identical to the V6/V7 "conservative" option);
  order 2: MUSCL, q_upwind +- slope/2 with a limited slope.
Ghost cells are even mirrors at every boundary (q is even about the axis; the wall, bottom
and top fluxes are zero or set by the caller, so the ghosts only shape the adjacent slopes).

Limiters (order 2): "vanleer" (default; TVD, smooth, phi <= 2), "minmod", "mc", and "none"
(unlimited centred slope = Fromm's scheme; validation only, not TVD).
TVD bound for forward Euler with phi <= 2: sum of directional Courant numbers <= 1/2; SSPRK2
keeps that bound, so the MUSCL2 path uses SWIRL_MUSCL_CFL for (|u_r|/dr + |u_z|/dz) dt.
"""
from __future__ import annotations

import numpy as np

LIMITERS = ("vanleer", "minmod", "mc", "none")
SWIRL_ADVECTION_MODES = ("advective", "conservative_upwind1", "conservative_muscl2")
SWIRL_VISCOUS_MODES = ("vector_laplacian", "angular_momentum")
SWIRL_MUSCL_CFL = 0.45


def limited_slope(dL: np.ndarray, dR: np.ndarray, limiter: str) -> np.ndarray:
    if limiter == "none":
        return 0.5 * (dL + dR)
    same = dL * dR > 0.0
    if limiter == "vanleer":
        den = np.where(same, dL + dR, 1.0)
        return np.where(same, 2.0 * dL * dR / den, 0.0)
    if limiter == "minmod":
        return np.where(same, np.sign(dL) * np.minimum(np.abs(dL), np.abs(dR)), 0.0)
    if limiter == "mc":
        m = np.minimum(np.minimum(2.0 * np.abs(dL), 2.0 * np.abs(dR)), 0.5 * np.abs(dL + dR))
        return np.where(same, np.sign(dL) * m, 0.0)
    raise ValueError(f"unknown limiter {limiter!r}; expected one of {LIMITERS}")


def face_values(qp: np.ndarray, vel: np.ndarray, order: int, limiter: str, axis: int) -> np.ndarray:
    """Upwind face values at the N+1 faces along ``axis`` from ``qp`` = q padded with TWO ghost
    cells at each end along that axis (length N+4); ``vel`` holds the face velocities."""
    qp = np.moveaxis(qp, axis, 0)
    v = np.moveaxis(vel, axis, 0)
    qL, qR = qp[1:-2], qp[2:-1]                  # cells left / right of faces 0..N
    if order == 2:
        d = np.diff(qp, axis=0)
        s = limited_slope(d[:-1], d[1:], limiter)  # slopes of padded cells 1..N+2
        qL = qL + 0.5 * s[:-1]
        qR = qR - 0.5 * s[1:]
    elif order != 1:
        raise ValueError("order must be 1 or 2")
    return np.moveaxis(np.where(v >= 0, qL, qR), 0, axis)


def _pad_even(q: np.ndarray, axis: int) -> np.ndarray:
    q0 = np.moveaxis(q, axis, 0)
    out = np.concatenate([q0[1::-1], q0, q0[:-3:-1]], axis=0)
    return np.moveaxis(out, 0, axis)


def q_flux_divergence(grid, q: np.ndarray, u_r: np.ndarray, u_z: np.ndarray,
                      order: int = 1, limiter: str = "vanleer") -> np.ndarray:
    """D(q) = (1/r) d(r u_r q)/dr + d(u_z q)/dz in conservative finite-volume form (see module
    docstring). Axis and wall radial fluxes and the bottom axial flux are zero."""
    qf_r = face_values(_pad_even(q, 0), u_r, order, limiter, 0)
    Fr = grid.r_f[:, None] * u_r * qf_r
    Fr[0, :] = 0.0
    Fr[-1, :] = 0.0
    qf_z = face_values(_pad_even(q, 1), u_z, order, limiter, 1)
    Fz = u_z * qf_z
    Fz[:, 0] = 0.0
    return (Fr[1:, :] - Fr[:-1, :]) / (grid.r_c[:, None] * grid.dr) + (Fz[:, 1:] - Fz[:, :-1]) / grid.dz


def advect_q(grid, q, u_r, u_z, dt, order=1, limiter="vanleer"):
    """One step of dq/dt = -D(q) with frozen face velocities: forward Euler for order 1
    (the V6/V7 option), SSPRK2 (Shu-Osher) for order 2. Each stage is conservative, so is
    their convex combination."""
    if order == 1:
        return q - dt * q_flux_divergence(grid, q, u_r, u_z, 1)
    q1 = q - dt * q_flux_divergence(grid, q, u_r, u_z, 2, limiter)
    return 0.5 * q + 0.5 * (q1 - dt * q_flux_divergence(grid, q1, u_r, u_z, 2, limiter))


def angular_momentum_viscous(grid, u_theta: np.ndarray, u_wall: np.ndarray, u_bottom: np.ndarray) -> np.ndarray:
    """nu-free viscous operator for u_theta in angular-momentum flux form:
        (1/r^2) d/dr( r^3 d(u/r)/dr ) + d^2 u / dz^2,
    radial stress flux G = r_f^3 (omega_i - omega_{i-1}) / dr with omega = u / r_c, G = 0 on the
    axis and G_wall = R^3 (u_wall / R - omega_N) / (dr / 2) (the same one-sided wall shear as
    SinglePhaseHeightSolver.torque_budget); z: Dirichlet bottom by ghost 2 u_b - u, even top.
    Interior stresses telescope in sum r_c^2 dr dz (.), so viscosity exchanges angular momentum
    only with the walls. Solid-body rotation u = Omega r gives exactly zero radial part."""
    rc = grid.r_c[:, None]
    om = u_theta / rc
    G = np.zeros((grid.Nr + 1, u_theta.shape[1]))
    G[1:-1] = grid.r_f[1:-1, None] ** 3 * (om[1:] - om[:-1]) / grid.dr
    R = grid.r_f[-1]
    G[-1] = R**3 * (np.asarray(u_wall) / R - om[-1]) / (0.5 * grid.dr)
    r_part = (G[1:] - G[:-1]) / (rc**2 * grid.dr)
    g_ext = np.concatenate([2.0 * np.asarray(u_bottom)[:, None] - u_theta[:, :1], u_theta, u_theta[:, -1:]], axis=1)
    z_part = (g_ext[:, 2:] - 2.0 * g_ext[:, 1:-1] + g_ext[:, :-2]) / grid.dz**2
    return r_part + z_part
