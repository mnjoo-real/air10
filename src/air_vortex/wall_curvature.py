"""Wall-near geometric (graph) curvature (V4b-P redesign, docs sec. 9.10).

Near the side wall the free surface is single-valued, z = eta(r). In a band of
the last ``band`` radial columns the capillary curvature at every interface
crossing used by the ghost-fluid pressure BC is computed from a constrained
polynomial fit of the column interface heights, evaluated AT THAT CROSSING's
r. This replaces the bulk path (level-set div(n) at cell centres, then linear
interpolation to the crossing) only inside the band; the bulk path is
untouched outside it.

Column heights: the phi = 0 root along each column from a local cubic
(4 nodes around the sign change), so the height error is O(dx^4) and the
curvature error O(dx^2) instead of the O(kappa dx^2) height / O(kappa)
curvature error of the linear crossing.

Stencils are INTERPOLATING by default (window = order + 1 points, the wall
constraint replacing one column in the last 1.5 cells): for order 2 this is the
standard 3-point height-function curvature. A least-squares window
(window > order + 1) is a Savitzky-Golay filter whose second-derivative response
to the grid-scale mode has the WRONG sign (5 points: +4/7 instead of -4), i.e.
an anti-restoring capillary operator; it is kept only as a documented negative
option (V4b-P redesign, Jacobian scan).

Wall constraint (shared algorithm, only the constraint row differs):
  pinned:        eta(R) = z_pin
  static_angle:  eta'(R) = cot(theta)          (theta through the liquid)
  extrapolate:   none

Curvature of the graph with liquid below and n pointing liquid -> air
(kappa = div n, the convention of curvature_single_phase):
  kappa_m = -eta'' / (1 + eta'^2)^(3/2),   kappa_theta = -eta' / (r sqrt(1 + eta'^2)).
"""
from __future__ import annotations

import numpy as np

from .grid import Grid
from .liquid_mask import LiquidGeometry

WALL_CURVATURE_METHODS = ("level_set", "graph")
WALL_CURVATURE_FITS = ("local", "global")


def column_heights(grid: Grid, phi: np.ndarray, cols) -> np.ndarray:
    """Topmost air-to-liquid phi = 0 root in each column (nan if none), from
    the cubic through the 4 nodes around the sign change (linear when fewer
    nodes exist), refined by a few Newton steps."""
    out = np.full(len(cols), np.nan)
    z = grid.z_c
    for k, i in enumerate(cols):
        col = phi[i, :]
        for j in range(grid.Nz - 1, 0, -1):
            if col[j] >= 0.0 and col[j - 1] < 0.0:
                zl = z[j - 1] - col[j - 1] * grid.dz / (col[j] - col[j - 1]) if col[j] != col[j - 1] else z[j - 1]
                j0 = j - 2
                if j0 >= 0 and j + 1 < grid.Nz:
                    zz = z[j0:j0 + 4]
                    c = np.polyfit(zz - zl, col[j0:j0 + 4], 3)
                    dc = np.polyder(c)
                    s = 0.0
                    for _ in range(6):
                        s -= np.polyval(c, s) / np.polyval(dc, s)
                    if abs(s) <= grid.dz:
                        zl = zl + s
                out[k] = zl
                break
    return out


def _fit(x, y, order, cons):
    """Least squares a (eta = sum a_k x^k, k = 0..order) subject to the
    linear equality constraints cons = [(row, value)]; KKT solve."""
    A = np.vander(x, order + 1, increasing=True)
    if not cons:
        return np.linalg.lstsq(A, y, rcond=None)[0]
    C = np.array([c for c, _ in cons]); d = np.array([v for _, v in cons])
    n, m = order + 1, len(cons)
    K = np.zeros((n + m, n + m))
    K[:n, :n] = 2 * A.T @ A
    K[:n, n:] = C.T
    K[n:, :n] = C
    rhs = np.r_[2 * A.T @ y, d]
    return np.linalg.solve(K, rhs)[:n]


def _wall_constraint(wall_cfg, grid, r0, order, z_ref):
    """Constraint row in the local coordinate x = (r - r0)/dx, or None."""
    xR = (grid.r_v - r0) / grid.dr
    if wall_cfg.contact_model == "pinned":
        return (np.array([xR**k for k in range(order + 1)]), wall_cfg.pinned_contact_height_m - z_ref)
    if wall_cfg.contact_model == "static_angle":
        cot = 1.0 / np.tan(np.deg2rad(wall_cfg.contact_angle_deg))
        return (np.array([k * xR ** (k - 1) if k > 0 else 0.0 for k in range(order + 1)]) / grid.dr, cot)
    return None


def graph_kappa_at(grid, wall_cfg, heights, cols, r_eval):
    """(kappa_m, kappa_theta) of the fitted graph at radii r_eval."""
    order = wall_cfg.wall_curvature_order
    W = wall_cfg.wall_curvature_window
    r_cols = grid.r_c[cols]
    ok = np.isfinite(heights)
    km = np.full(len(r_eval), np.nan); kt = np.full(len(r_eval), np.nan)
    z_ref = float(np.nanmean(heights)) if ok.any() else 0.0
    if wall_cfg.wall_curvature_fit == "global":
        r0 = grid.r_v
        cons = [c for c in [_wall_constraint(wall_cfg, grid, r0, order, z_ref)] if c is not None]
        if ok.sum() + len(cons) < order + 1:
            return km, kt
        a = _fit((r_cols[ok] - r0) / grid.dr, heights[ok] - z_ref, order, cons)
        for q, r in enumerate(r_eval):
            x = (r - r0) / grid.dr
            e1 = sum(k * a[k] * x ** (k - 1) for k in range(1, order + 1)) / grid.dr
            e2 = sum(k * (k - 1) * a[k] * x ** (k - 2) for k in range(2, order + 1)) / grid.dr**2
            qn = np.sqrt(1 + e1**2)
            km[q], kt[q] = -e2 / qn**3, -e1 / (r * qn)
        return km, kt
    if cols[0] == 0:
        # band reaches the axis: mirror the first columns (eta(-r) = eta(r))
        nm = min(W, len(cols))
        r_cols = np.r_[-r_cols[:nm][::-1], r_cols]
        heights = np.r_[heights[:nm][::-1], heights]
        cols = np.r_[-np.ones(nm, int), cols]
        ok = np.isfinite(heights)
    for q, r in enumerate(r_eval):
        # the wall constraint replaces one column in the last 1.5 cells
        c = _wall_constraint(wall_cfg, grid, r, order, z_ref) if r > grid.r_c[-2] else None
        cons = [] if c is None else [c]
        need = max(W - len(cons), order + 1 - len(cons))
        cand = np.nonzero(ok)[0]
        near = cand[np.argsort(np.abs(r_cols[cand] - r), kind="stable")[:need]]
        if len(near) + len(cons) < order + 1:
            continue
        a = _fit((r_cols[near] - r) / grid.dr, heights[near] - z_ref, order, cons)
        e1 = a[1] / grid.dr
        e2 = 2 * a[2] / grid.dr**2
        qn = np.sqrt(1 + e1**2)
        km[q], kt[q] = -e2 / qn**3, -e1 / (r * qn)
    return km, kt


def graph_band_curvature(grid: Grid, geom: LiquidGeometry, phi: np.ndarray, wall_cfg,
                         kr: np.ndarray, kz: np.ndarray):
    """Replace kappa at the crossings of the last ``wall_curvature_band``
    columns by the graph curvature (optionally blended linearly over
    ``wall_curvature_blend`` columns at the band's inner edge)."""
    from .curvature_single_phase import crossing_positions
    Nr = grid.Nr
    band, blend = min(wall_cfg.wall_curvature_band, Nr), wall_cfg.wall_curvature_blend
    if band == Nr:
        blend = 0                          # whole radius: no transition
    support = wall_cfg.wall_curvature_window if wall_cfg.wall_curvature_fit == "local" else 0
    c0 = max(0, Nr - band - blend - support)
    cols = np.arange(c0, Nr)
    h = column_heights(grid, phi, cols)
    r_in = grid.r_f[Nr - band] - blend * grid.dr - (grid.dr if band == Nr else 0.0)  # weight 0 here
    r_full = grid.r_f[Nr - band]                          # weight 1 from here to the wall
    rr, _, rz, _ = crossing_positions(grid, geom)
    kr, kz = kr.copy(), kz.copy()
    for K, R in ((kz, rz), (kr, rr)):
        m = np.isfinite(K) & (R > r_in)
        if not m.any():
            continue
        r_eval = R[m]
        km, kt = graph_kappa_at(grid, wall_cfg, h, cols, r_eval)
        kg = km + kt
        w = np.ones_like(r_eval) if blend == 0 else np.clip((r_eval - r_in) / (r_full - r_in), 0.0, 1.0)
        good = np.isfinite(kg)
        vals = K[m]
        vals[good] = w[good] * kg[good] + (1 - w[good]) * vals[good]
        K[m] = vals
    return kr, kz
