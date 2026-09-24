"""Finite-difference operators in axisymmetric cylindrical coordinates
(README sections 7.1, 7.3, 8).

All operators act on the staggered grid defined in :mod:`air_vortex.grid`:
``u_r`` on radial faces (Nr+1, Nz), ``u_z`` on axial faces (Nr, Nz+1),
everything else at cell centers (Nr, Nz).
"""
from __future__ import annotations

import numpy as np

from .grid import Grid


def divergence(grid: Grid, u_r: np.ndarray, u_z: np.ndarray) -> np.ndarray:
    """(1/r) d(r u_r)/dr + d(u_z)/dz, evaluated at cell centers (README 7.1)."""
    r_f = grid.r_f
    flux_r = r_f[:, None] * u_r  # (Nr+1, Nz)
    d_rur_dr = (flux_r[1:, :] - flux_r[:-1, :]) / grid.dr
    div = d_rur_dr / grid.r_c[:, None] + (u_z[:, 1:] - u_z[:, :-1]) / grid.dz
    return div


def grad_p_to_ur_faces(grid: Grid, p: np.ndarray) -> np.ndarray:
    """dp/dr on interior radial faces; shape (Nr+1, Nz). Faces 0 and Nr are
    left at 0 because u_r is prescribed (Dirichlet no-penetration) there."""
    dpdr = np.zeros(grid.shape_ur)
    dpdr[1:-1, :] = (p[1:, :] - p[:-1, :]) / grid.dr
    return dpdr


def grad_p_to_uz_faces(grid: Grid, p: np.ndarray) -> np.ndarray:
    """dp/dz on interior axial faces; shape (Nr, Nz+1). Face 0 (bottom) is
    left at 0 (no-penetration); face Nz (top) uses a one-sided value so the
    open-boundary pressure (p=0) can still drive flow there."""
    dpdz = np.zeros(grid.shape_uz)
    dpdz[:, 1:-1] = (p[:, 1:] - p[:, :-1]) / grid.dz
    dpdz[:, -1] = (0.0 - p[:, -1]) / (0.5 * grid.dz)
    return dpdz


def center_grad_r(grid: Grid, f: np.ndarray) -> np.ndarray:
    """Centered d/dr of a cell-centered field, at cell centers (Nr, Nz).

    Axis (r=0) is a symmetry line, not a physical boundary: f is implicitly
    extended as an even function of r (f(-r) = f(r), true for any genuinely
    axisymmetric scalar field -- phi, p, r*n_r, ...), so the ghost value one
    cell to the left of r_c[0] equals f[0] itself, at position -r_c[0]. That
    ghost sits 2*dr away from r_c[1] (the same spacing as every interior
    centered-difference stencil), giving (f[1]-f[0])/(2*dr) -- NOT
    (f[1]-f[0])/dr, which was this function's bug for a long time: it
    silently doubled every near-axis radial gradient (and, through
    interface_curvature's extra 1/r division, produced a curvature spike of
    O(1000) at the axis -- see README "Performance"/validation notes and
    tests/test_operators.py::test_center_grad_r_axis_uses_mirror_symmetry).

    The outer wall (r=R_v) IS a physical boundary (no symmetry to exploit),
    so a one-sided difference there is correct as-is.
    """
    out = np.empty_like(f)
    out[1:-1, :] = (f[2:, :] - f[:-2, :]) / (2 * grid.dr)
    out[0, :] = (f[1, :] - f[0, :]) / (2 * grid.dr)
    out[-1, :] = (f[-1, :] - f[-2, :]) / grid.dr
    return out


def center_grad_z(grid: Grid, f: np.ndarray) -> np.ndarray:
    """Centered d/dz of a cell-centered field, at cell centers (Nr, Nz)."""
    out = np.empty_like(f)
    out[:, 1:-1] = (f[:, 2:] - f[:, :-2]) / (2 * grid.dz)
    out[:, 0] = (f[:, 1] - f[:, 0]) / grid.dz
    out[:, -1] = (f[:, -1] - f[:, -2]) / grid.dz
    return out


def interp_center_to_ur(f_center: np.ndarray) -> np.ndarray:
    """Average a cell-centered field onto interior radial faces; the two
    boundary faces (r=0 and r=R_v) are filled with the nearest cell value."""
    Nr, Nz = f_center.shape
    out = np.empty((Nr + 1, Nz))
    out[1:-1, :] = 0.5 * (f_center[1:, :] + f_center[:-1, :])
    out[0, :] = f_center[0, :]
    out[-1, :] = f_center[-1, :]
    return out


def interp_center_to_uz(f_center: np.ndarray) -> np.ndarray:
    """Average a cell-centered field onto interior axial faces; the two
    boundary faces (bottom, top) are filled with the nearest cell value."""
    Nr, Nz = f_center.shape
    out = np.empty((Nr, Nz + 1))
    out[:, 1:-1] = 0.5 * (f_center[:, 1:] + f_center[:, :-1])
    out[:, 0] = f_center[:, 0]
    out[:, -1] = f_center[:, -1]
    return out


def interp_ur_to_center(f_ur: np.ndarray) -> np.ndarray:
    return 0.5 * (f_ur[1:, :] + f_ur[:-1, :])


def interp_uz_to_center(f_uz: np.ndarray) -> np.ndarray:
    return 0.5 * (f_uz[:, 1:] + f_uz[:, :-1])


def u_theta_on_ur_faces(grid: Grid, u_theta: np.ndarray) -> np.ndarray:
    """u_theta interpolated to radial faces, used for the centrifugal term
    -u_theta^2/r in the radial momentum equation (README 7.3)."""
    return interp_center_to_ur(u_theta)


def uz_at_ur_locations(u_z: np.ndarray) -> np.ndarray:
    """4-point corner average of u_z onto the u_r grid. Only the interior
    faces (excluding axis/wall) are meaningful; u_r is Dirichlet there."""
    Nr = u_z.shape[0]
    Nz = u_z.shape[1] - 1
    out = np.zeros((Nr + 1, Nz))
    out[1:-1, :] = 0.25 * (u_z[:-1, :-1] + u_z[:-1, 1:] + u_z[1:, :-1] + u_z[1:, 1:])
    return out


def ur_at_uz_locations(u_r: np.ndarray) -> np.ndarray:
    """4-point corner average of u_r onto the u_z grid. Only the interior
    faces (excluding bottom/top) are meaningful; u_z is Dirichlet at the
    bottom and set by the open-boundary BC at the top."""
    Nr = u_r.shape[0] - 1
    Nz = u_r.shape[1]
    out = np.zeros((Nr, Nz + 1))
    out[:, 1:-1] = 0.25 * (u_r[:-1, :-1] + u_r[:-1, 1:] + u_r[1:, :-1] + u_r[1:, 1:])
    return out


def upwind_derivative(f: np.ndarray, vel: np.ndarray, h: float, axis: int) -> np.ndarray:
    """First-order upwind d f/d(axis) using edge-clamped one-sided
    differences at the domain boundary."""
    fp = np.roll(f, -1, axis=axis)
    fm = np.roll(f, 1, axis=axis)
    if axis == 0:
        fp[-1, :] = f[-1, :]
        fm[0, :] = f[0, :]
    else:
        fp[:, -1] = f[:, -1]
        fm[:, 0] = f[:, 0]

    d_fwd = (fp - f) / h
    d_bwd = (f - fm) / h
    return np.where(vel >= 0, d_bwd, d_fwd)


# ---------------------------------------------------------------------------
# Viscous (vector Laplacian) terms.
#
# For a Newtonian fluid with *constant* mu, the divergence of the viscous
# stress tensor for axisymmetric flow reduces exactly to mu times the vector
# Laplacian:
#
#   (div tau)_r     = mu * [ (1/r) d/dr(r du_r/dr)     - u_r/r^2     + d2 u_r/dz2     ]
#   (div tau)_theta = mu * [ (1/r) d/dr(r du_theta/dr) - u_theta/r^2 + d2 u_theta/dz2 ]
#   (div tau)_z     = mu * [ (1/r) d/dr(r du_z/dr)                  + d2 u_z/dz2     ]
#
# Here mu varies across the water/air interface, so these formulas are used
# with a *locally interpolated* mu multiplying the constant-coefficient
# operator rather than the fully conservative div(mu(grad u + grad u^T))
# form. This is a standard first-pass simplification (README section 38
# explicitly expects the initial solver to start simple) and is adequate
# while the interfacial viscous stress itself is not yet a target quantity.
# ---------------------------------------------------------------------------


def _pad_axis(f: np.ndarray, axis: int, mode_low: str, mode_high: str) -> np.ndarray:
    """Add one ghost layer on each side of ``axis``.

    mode: 'even' mirrors the boundary value (zero-gradient / Neumann),
    'odd' mirrors with a sign flip (zero-value / Dirichlet)."""
    low = np.take(f, [0], axis=axis)
    high = np.take(f, [f.shape[axis] - 1], axis=axis)
    if mode_low == "odd":
        low = -low
    if mode_high == "odd":
        high = -high
    return np.concatenate([low, f, high], axis=axis)


def _extended_coord(coord: np.ndarray) -> np.ndarray:
    left = 2 * coord[0] - coord[1]
    right = 2 * coord[-1] - coord[-2]
    return np.concatenate(([left], coord, [right]))


def _radial_flux_laplacian(f: np.ndarray, r_coord: np.ndarray, dr: float,
                            mode_low: str, mode_high: str) -> np.ndarray:
    """(1/r) d/dr(r df/dr) at every point of ``f`` (shape (M, N)), where
    ``r_coord`` gives the radial location of each row of ``f``.

    If ``r_coord[0] == 0`` (the u_r face grid, whose first face sits exactly
    on the axis), that row is computed with a safe placeholder radius: the
    row is Dirichlet-zero there by construction and always overwritten by
    the boundary condition afterwards, so only the divide-by-zero warning
    is being avoided, not a real value."""
    f_ext = _pad_axis(f, axis=0, mode_low=mode_low, mode_high=mode_high)
    r_ext = _extended_coord(r_coord)
    r_mid = 0.5 * (r_ext[:-1] + r_ext[1:])  # length M+1

    flux_plus = r_mid[1:, None] * (f_ext[2:, :] - f_ext[1:-1, :])
    flux_minus = r_mid[:-1, None] * (f_ext[1:-1, :] - f_ext[:-2, :])
    r_safe = np.where(r_coord == 0.0, dr, r_coord)
    return (flux_plus - flux_minus) / (r_safe[:, None] * dr**2)


def _axial_second_derivative(f: np.ndarray, dz: float, mode_low: str, mode_high: str) -> np.ndarray:
    f_ext = _pad_axis(f, axis=1, mode_low=mode_low, mode_high=mode_high)
    return (f_ext[:, 2:] - 2.0 * f_ext[:, 1:-1] + f_ext[:, :-2]) / dz**2


def laplacian_ur(grid: Grid, u_r: np.ndarray) -> np.ndarray:
    """Vector Laplacian for the r-component, on the u_r face grid.

    r-direction: Dirichlet (odd) at both the axis and the wall, since
    u_r=0 is imposed there. z-direction: Neumann (even) at bottom (no-slip
    tangential derivative not needed here) and top (open, zero-gradient)."""
    r_part = _radial_flux_laplacian(u_r, grid.r_f, grid.dr, "odd", "odd")
    z_part = _axial_second_derivative(u_r, grid.dz, "odd", "even")
    # r_f[0] = 0 exactly (the axis); u_r is Dirichlet-zero there and this
    # row is always overwritten by the boundary condition, so substitute a
    # safe placeholder to avoid a 0/0 divide-by-zero warning.
    r_f_safe = np.where(grid.r_f == 0.0, grid.dr, grid.r_f)
    return r_part - u_r / r_f_safe[:, None] ** 2 + z_part


def laplacian_uz(grid: Grid, u_z: np.ndarray) -> np.ndarray:
    """Vector Laplacian for the z-component, on the u_z face grid.

    r-direction: Neumann (even) at the axis (symmetry), Dirichlet (odd) at
    the wall (no-slip). z-direction: values only needed at interior faces;
    boundary rows are overwritten by BCs afterwards."""
    r_part = _radial_flux_laplacian(u_z, grid.r_c, grid.dr, "even", "odd")
    z_part = _axial_second_derivative(u_z, grid.dz, "odd", "even")
    return r_part + z_part


def laplacian_utheta(grid: Grid, u_theta: np.ndarray) -> np.ndarray:
    """Vector Laplacian for the theta-component, at cell centers.

    Both radial ends use a Dirichlet (odd) ghost: u_theta=0 at the axis (by
    symmetry of an azimuthal field) and at the wall (no-slip). z-direction:
    Dirichlet (odd) at the bottom (no-slip), Neumann (even) at the open top."""
    r_part = _radial_flux_laplacian(u_theta, grid.r_c, grid.dr, "odd", "odd")
    z_part = _axial_second_derivative(u_theta, grid.dz, "odd", "even")
    return r_part - u_theta / grid.r_c[:, None] ** 2 + z_part
