"""Side-wall contact geometry for the Level Set (Gate V4b-S).

Convention (verified in tests/test_contact_angle.py)
----------------------------------------------------
phi < 0 liquid, n = grad(phi)/|grad(phi)| points liquid -> air, the side
wall's outward normal is n_w = +e_r, and theta is measured THROUGH THE LIQUID.
For a graph z = eta(r) with liquid below, the liquid wedge at the contact
point lies between the wall direction -e_z and the inward interface tangent
t = (-1, -eta')/sqrt(1+eta'^2), so

    cos(theta) = eta'(R) / sqrt(1 + eta'(R)^2),  i.e.  eta'(R) = cot(theta),
    n . n_w = -eta' / sqrt(1 + eta'^2) = -cos(theta).

theta < 90 deg: wetting, meniscus rises toward the wall. theta = 90 deg: flat.

Ghost construction
------------------
One ghost column at r = R + h/2 (h = dr), built so that BOTH the angle and
the curvature stencil stay consistent:

1. Base ghost: cubic extrapolation of the column through the four wall-
   adjacent cells, ghost_0 = 4 phi_1 - 6 phi_2 + 4 phi_3 - phi_4. It is
   O(h^4) accurate for smooth phi, so the last cell's phi_rr is O(h^2)
   accurate. This is the angle-free ``extrapolate`` model.
   The earlier linear extrapolation 2 phi_1 - phi_2 forced phi_rr = 0 in
   the last column and dropped the meridional curvature there: 30-45 %
   wall curvature error on a spherical cap, non-convergent (docs sec. 9).

2. ``static_angle`` adds one uniform wall-slope correction,
   ghost = ghost_0 + h * dg. dg is evaluated only AT THE CONTACT POINT z_cl,
   where the wall trace of the cubic, phi_w(z), crosses zero:

       dg = [-cot(theta) |phi_z,w| - g_w](z_cl),   g_w = d(cubic)/dr at r = R.

   So only the zero level set is forced to n . e_r = -cos(theta).
   Imposing theta on EVERY level set, the textbook extension, was tried
   first. It is inconsistent with a curved interface: neighbouring level
   sets of a distance function meet the wall at theta + O(h kappa). That
   gives an O(h^2) ghost error and hence an O(1) curvature error that did
   not converge (6 % at theta = 60 deg on every grid).
   phi_z,w and g_w are evaluated at z_cl with a local 4-point cubic in z,
   so when the interior already meets the wall at theta, dg = O(h^3) and
   the correction vanishes with refinement. Several contact points each
   correct their nearest stretch of wall.

Curvature is still kappa = div(n) (curvature_single_phase.py) and is
never overwritten.

This one helper supplies the wall ghost for every consumer that needs
contact geometry. Reinitialization currently uses its own linear-extrapolation
wall ghost; whether that preserves the contact angle is measured, not assumed
(docs sec. 9).
"""
from __future__ import annotations

import numpy as np

from .grid import Grid


def _phi_z_centered(phi: np.ndarray, dz: float) -> np.ndarray:
    out = np.empty_like(phi)
    out[:, 1:-1] = (phi[:, 2:] - phi[:, :-2]) / (2 * dz)
    out[:, 0] = (phi[:, 1] - phi[:, 0]) / dz
    out[:, -1] = (phi[:, -1] - phi[:, -2]) / dz
    return out


# Lagrange weights for the cubic through x = -1/2, -3/2, -5/2, -7/2 (units of h,
# origin at the wall face): value and d/dx at x = 0, and value at x = +1/2.
_X = np.array([-0.5, -1.5, -2.5, -3.5])


def _lagrange_weights(x0: float, deriv: bool = False) -> np.ndarray:
    w = np.zeros(4)
    for i in range(4):
        e = np.zeros(4)
        e[i] = 1.0
        c = np.polyfit(_X, e, 3)
        w[i] = np.polyval(np.polyder(c), x0) if deriv else np.polyval(c, x0)
    return w


_W_WALL = _lagrange_weights(0.0)
_W_SLOPE = _lagrange_weights(0.0, deriv=True)
_W_GHOST = _lagrange_weights(0.5)          # = (4, -6, 4, -1)


def wall_trace(grid: Grid, phi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(phi_w(z), dphi/dr_w(z)) on the wall face from the cubic through the
    four wall-adjacent cells."""
    cols = phi[-1:-5:-1, :]                  # phi_1..phi_4 (nearest the wall first)
    return _W_WALL @ cols, (_W_SLOPE @ cols) / grid.dr


def wall_contact_points(grid: Grid, phi: np.ndarray) -> np.ndarray:
    """Heights where the wall trace phi_w(z) changes sign (linear in z)."""
    pw, _ = wall_trace(grid, phi)
    j = np.nonzero(np.sign(pw[:-1]) * np.sign(pw[1:]) < 0)[0]
    t = pw[j] / (pw[j] - pw[j + 1])
    return grid.z_c[j] + t * grid.dz


def wall_ghost_column(grid: Grid, phi: np.ndarray, wall_cfg=None) -> np.ndarray:
    """phi in the ghost column r = R + dr/2, shape (Nz,)."""
    if phi.shape[0] < 4:
        raise ValueError("contact geometry needs at least 4 radial cells")
    ghost0 = _W_GHOST @ phi[-1:-5:-1, :]
    if wall_cfg is None or wall_cfg.contact_model == "extrapolate":
        return ghost0
    if wall_cfg.contact_model == "pinned":
        return _pinned_ghost(grid, phi, wall_cfg.pinned_contact_height_m)
    if wall_cfg.contact_model != "static_angle":
        raise ValueError(wall_cfg.contact_model)
    zc = wall_contact_points(grid, phi)
    if zc.size == 0:
        return ghost0
    pw, gw = wall_trace(grid, phi)
    cot = 1.0 / np.tan(np.deg2rad(wall_cfg.contact_angle_deg))
    dg_pts = np.empty(zc.size)
    for k, z in enumerate(zc):
        # local cubic in z through 4 nodes around z_cl: phi_z,w and g_w at z_cl
        # to O(h^3), so dg = O(h^3) when the interior already meets theta
        j0 = int(np.clip(np.searchsorted(grid.z_c, z) - 2, 0, grid.Nz - 4))
        zz = grid.z_c[j0:j0 + 4] - z
        cp = np.polyfit(zz, pw[j0:j0 + 4], 3)
        cg = np.polyfit(zz, gw[j0:j0 + 4], 3)
        dg_pts[k] = -cot * abs(cp[2]) - cg[3]
    nearest = np.argmin(np.abs(grid.z_c[:, None] - zc[None, :]), axis=1)
    return ghost0 + grid.dr * dg_pts[nearest]


def _pinned_ghost(grid: Grid, phi: np.ndarray, z_pin: float | None) -> np.ndarray:
    """CL-P ghost: the wall trace pw(z) of the cubic is shifted by the
    constant pw(z_pin) (local 4-point cubic in z), so the zero level meets
    the wall exactly at z_pin; the angle is whatever the interior implies.
    Ghost = quadratic through phi_2 (x=-3/2), phi_1 (x=-1/2) and the pinned
    wall value (x=0), evaluated at x=+1/2: phi_2/3 - 2 phi_1 + 8/3 phi_w."""
    if z_pin is None:
        raise ValueError("pinned contact model needs wall.pinned_contact_height_m")
    pw, _ = wall_trace(grid, phi)
    j0 = int(np.clip(np.searchsorted(grid.z_c, z_pin) - 2, 0, grid.Nz - 4))
    c = np.polyfit(grid.z_c[j0:j0 + 4] - z_pin, pw[j0:j0 + 4], 3)
    phi_w = pw - c[3]
    return phi[-2, :] / 3.0 - 2.0 * phi[-1, :] + 8.0 / 3.0 * phi_w


def contact_point(grid: Grid, phi: np.ndarray, n_fit: int = 3) -> tuple[float, float]:
    """(z_contact, theta_deg) measured from the interface GEOMETRY, not from
    the imposed ghost: fit a quadratic eta(r) to the free-surface heights of
    the last ``n_fit`` columns (sub-cell phi=0 crossings), extrapolate it to
    r = R, and take theta = arccot(eta'(R)). Valid while the interface is a
    graph near the wall."""
    from .diagnostics import free_surface_height
    eta = free_surface_height(phi, grid)
    r = grid.r_c[-n_fit:]
    e = eta[-n_fit:]
    if not np.all(np.isfinite(e)):
        return float("nan"), float("nan")
    c = np.polyfit(r - grid.r_v, e, 2)
    slope = c[1]
    return float(c[2]), float(np.rad2deg(np.arctan2(1.0, slope)))
