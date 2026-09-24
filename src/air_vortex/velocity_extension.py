"""Narrow-band extension of the LIQUID velocity into the void side, for
Level Set transport only (README_rewritten sections 5.4, 9.3).

The extended values are NOT air velocity. Level 1A solves no gas
dynamics; the extension exists only so that the kinematic condition
dphi/dt + u_ext . grad(phi) = 0 has a velocity at the void-side stencil
points around phi=0.

Method
------
Constant extension along the interface normal, i.e. a discrete solution of

    n . grad(q) = 0     in the void band,   q = liquid value on known points

using the first-order upwind (fast-marching-style) discretization: an
unknown point takes the weighted average of its already-known 4-neighbors,
with each neighbor's weight equal to the component of n pointing from that
neighbor toward the point (neighbors "behind" it along n), divided by the
spacing. Points are filled layer by layer outward from the liquid, up to
``n_layers`` layers; everything farther out is set to 0.

If no known neighbor lies upwind along n (degenerate normal), the plain
mean of the known neighbors is used. Both rules are convex combinations of
known values, which gives by construction:

    zero field      -> exactly zero extension
    constant field  -> exactly constant extension
    known points    -> never modified

Swirl component
---------------
For u_theta the extended quantity is the angular velocity omega = u_theta/r,
not u_theta itself: zero tangential stress on the free surface for the
swirl reads tau_{theta n} = mu r n.grad(u_theta / r) = 0 (the r-weighted
rate of strain of an azimuthal flow), so constant-along-n extension of
omega is the discrete stress-free condition and leaves solid-body rotation
u_theta = Omega r exactly unchanged. Constant extension of u_theta would
impose a spurious shear on a rigidly rotating free surface.

u_r and u_z use n.grad(u) = 0, the usual first approximation of the
tangential-stress-free condition for the meridional velocity.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .grid import Grid
from .liquid_mask import LiquidGeometry
from .operators import center_grad_r, center_grad_z, interp_center_to_ur, interp_center_to_uz

_W_EPS = 1e-12


def _shift(a: np.ndarray, di: int, dj: int, fill) -> np.ndarray:
    """out[i, j] = a[i - di, j - dj] (value of the neighbor at offset
    (-di, -dj)), padded with ``fill`` outside the array."""
    out = np.full_like(a, fill)
    Ni, Nj = a.shape
    src_i = slice(max(0, -di), Ni - max(0, di))
    dst_i = slice(max(0, di), Ni - max(0, -di))
    src_j = slice(max(0, -dj), Nj - max(0, dj))
    dst_j = slice(max(0, dj), Nj - max(0, -dj))
    out[dst_i, dst_j] = a[src_i, src_j]
    return out


def extend_field(q: np.ndarray, known: np.ndarray, n_r: np.ndarray, n_z: np.ndarray,
                 h_r: float, h_z: float, n_layers: int, fill_value: float = 0.0
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Extend ``q`` from the ``known`` mask outward by ``n_layers`` layers.

    ``n_r``, ``n_z``: unit interface normal (pointing into the void, i.e.
    grad(phi)/|grad(phi)|) at the same locations as ``q``.

    Returns (q_ext, band) where ``band`` marks the points that were filled.
    Points neither known nor in the band are set to ``fill_value``."""
    q_ext = np.where(known, q, fill_value).astype(float)
    have = known.copy()
    band = np.zeros_like(known)

    # neighbor offsets (di, dj) and the upwind weight: a neighbor at
    # (i-1, j) lies "behind" (i, j) along +r, so it is upwind when n_r > 0.
    offsets = (
        (1, 0, lambda: np.maximum(n_r, 0.0) / h_r),
        (-1, 0, lambda: np.maximum(-n_r, 0.0) / h_r),
        (0, 1, lambda: np.maximum(n_z, 0.0) / h_z),
        (0, -1, lambda: np.maximum(-n_z, 0.0) / h_z),
    )
    weights = [(di, dj, wfn()) for di, dj, wfn in offsets]

    for _ in range(n_layers):
        num = np.zeros_like(q_ext)
        den = np.zeros_like(q_ext)
        cnt = np.zeros_like(q_ext)
        acc = np.zeros_like(q_ext)
        for di, dj, w in weights:
            nb_have = _shift(have, di, dj, False)
            nb_q = _shift(q_ext, di, dj, 0.0)
            wk = np.where(nb_have, w, 0.0)
            num += wk * nb_q
            den += wk
            cnt += nb_have
            acc += np.where(nb_have, nb_q, 0.0)
        cand = (~have) & (cnt > 0)
        if not np.any(cand):
            break
        upwind_ok = den > _W_EPS
        new_val = np.where(upwind_ok, num / np.where(upwind_ok, den, 1.0),
                           acc / np.maximum(cnt, 1))
        q_ext[cand] = new_val[cand]
        have |= cand
        band |= cand
    return q_ext, band


def center_normals(grid: Grid, phi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gr = center_grad_r(grid, phi)
    gz = center_grad_z(grid, phi)
    mag = np.sqrt(gr**2 + gz**2)
    safe = np.where(mag > 0, mag, 1.0)
    return np.where(mag > 0, gr / safe, 0.0), np.where(mag > 0, gz / safe, 1.0)


@dataclass
class ExtensionInfo:
    band_ur: np.ndarray
    band_uz: np.ndarray
    band_center: np.ndarray


def extend_velocity(grid: Grid, geom: LiquidGeometry, phi: np.ndarray,
                    u_r: np.ndarray, u_z: np.ndarray, u_theta: np.ndarray,
                    n_layers: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, ExtensionInfo]:
    """Extend (u_r, u_z, u_theta) from liquid-controlled locations into a
    ``n_layers``-wide void band. Liquid-controlled values are returned
    bit-identical."""
    n_r_c, n_z_c = center_normals(grid, phi)

    def unit_on(nr, nz):
        mag = np.sqrt(nr**2 + nz**2)
        safe = np.where(mag > 0, mag, 1.0)
        return nr / safe, nz / safe

    nr_ur, nz_ur = unit_on(interp_center_to_ur(n_r_c), interp_center_to_ur(n_z_c))
    nr_uz, nz_uz = unit_on(interp_center_to_uz(n_r_c), interp_center_to_uz(n_z_c))

    ur_ext, band_ur = extend_field(u_r, geom.ur_face_known(), nr_ur, nz_ur,
                                   grid.dr, grid.dz, n_layers)
    uz_ext, band_uz = extend_field(u_z, geom.uz_face_known(), nr_uz, nz_uz,
                                   grid.dr, grid.dz, n_layers)
    omega = u_theta / grid.r_c[:, None]
    om_ext, band_c = extend_field(omega, geom.liquid, n_r_c, n_z_c,
                                  grid.dr, grid.dz, n_layers)
    ut_ext = np.where(geom.liquid, u_theta, om_ext * grid.r_c[:, None])
    # Dirichlet boundary faces stay exactly as given
    ur_ext[0, :] = 0.0
    ur_ext[-1, :] = 0.0
    uz_ext[:, 0] = 0.0
    return ur_ext, uz_ext, ut_ext, ExtensionInfo(band_ur, band_uz, band_c)


# ---------------------------------------------------------------------------
# Minimum safe extension width when the curvature feeds the pressure BC
# (sigma > 0). Derived from the stencils actually used; counted in the same
# 4-neighbour BFS layers that extend_field fills (Manhattan distance from
# the liquid):
#
#   CURVATURE_EVAL_LAYER = 1   kappa enters p_Gamma through the void-side
#                              node Q of every mixed face (layer 1), via
#                              curvature_single_phase.curvature_at_crossings.
#   CURVATURE_STENCIL_REACH = 2  curvature_centers uses a 3x3 stencil; its
#                              diagonal corners are Manhattan distance 2.
#   TRANSPORT_REACH = 2        advect_level_set_advective (MUSCL2 + SSPRK2):
#                              stage 2 at node j reads stage-1 phi at j-2..j+2
#                              (limited slopes); stage-1 phi at m depends on
#                              velocity on the faces of m. The velocity face on
#                              the far side of node layer k is extension layer k.
#
# A phi value used by kappa must never have been advected with the zero
# velocity beyond the band edge, so every face within
#   1 + 2 + 2 = 5 layers
# must be an extended (smooth) value. Fewer layers put the band-edge
# |grad phi| kink inside the curvature stencil: 3 layers gave the V4 CAP-B
# instability (E_sd -> 0.8, U -> 0.5 m/s by 1.25 s). The default
# extension_layers_capillary = 6 is this minimum plus one safety layer.
# ---------------------------------------------------------------------------
CURVATURE_EVAL_LAYER = 1
CURVATURE_STENCIL_REACH = 2
TRANSPORT_REACH = 2


def minimum_capillary_extension_layers() -> int:
    return CURVATURE_EVAL_LAYER + CURVATURE_STENCIL_REACH + TRANSPORT_REACH
