"""Height-function (graph) free-surface geometry for the opt-in
``single_phase_height`` model (research branch, docs sec. 10).

The interface is the single-valued graph z = eta(r) with ONE authoritative
height per radial cell centre, eta_i = eta(r_c[i]). There is no level-set
state: no off-contour degrees of freedom exist. Everything below (liquid
mask, ghost-fluid crossing fractions, curvature, column fluxes) is computed
from eta and the pinned wall point (R, z_pin) only.

Reconstruction. Nodes x = (-r_c[2], -r_c[1], -r_c[0], r_c[0..N-1], R) with
values (eta_2, eta_1, eta_0, eta_0..eta_{N-1}, z_pin): even mirror at the axis
(eta_r(0) = 0) and the pinned wall value at r = R (never snapped to a cell).
On each interval [x_k, x_{k+1}] the interpolant is the cubic through the FIXED
nodes x_{k-1..k+2} (the last interval, which has no node beyond R, uses
x_{k-2..k+1}). The stencil depends only on the interval, never on the data or
on the sub-cell phase of z_pin. The curvature at a column node is the mean of
its two adjacent interval cubics (both interpolate the node).

Scope. Valid only while the interface is a single-valued graph (flat surface,
vortex depression, first air-core contact). ``graph_validity`` reports the
slope and the principal-curvature resolution. Overturning, pinch-off and
multi-valued interfaces are OUTSIDE this model.

Curvature (liquid below, n = liquid -> air, kappa = div n, p_Gamma = p_atm + sigma kappa):
  kappa_m = -eta'' / (1 + eta'^2)^(3/2),   kappa_theta = -eta' / (r sqrt(1 + eta'^2)).
"""
from __future__ import annotations

import numpy as np

from .grid import Grid
from .liquid_mask import (CELL_INTERFACE, CELL_LIQUID, CELL_VOID, FACE_LIQ_MINUS, FACE_LIQ_PLUS,
                          FACE_LIQUID, THETA_MIN, LiquidGeometry)

N_MIRROR = 2


class GraphInterface:
    """Piecewise-cubic graph through the column heights and the pinned wall point.

    ``crossing="linear"`` (default, WELL-BALANCED): radial crossings (on u_r faces)
    are located with the LINEAR interpolant of the two adjacent column heights and
    carry the LINEAR interpolation of the two node curvatures. Then a node
    equilibrium sigma kappa_i + rho g eta_i = P0 implies sigma kappa + rho g z = P0 at
    every crossing, so a discrete static equilibrium with zero flow exists (with a
    cubic crossing the pressure BC is overdetermined: more crossings than height
    DOFs, O(dx^2) mismatch, steady spurious current). ``crossing="cubic"``: locate and
    evaluate radial crossings on the cubic (diagnostic comparison)."""

    def __init__(self, grid: Grid, eta: np.ndarray, z_pin: float, crossing: str = "linear"):
        if crossing not in ("linear", "cubic"):
            raise ValueError(crossing)
        self.crossing = crossing
        if len(eta) < 4:
            raise ValueError("height interface needs at least 4 radial columns")
        self.grid, self.eta, self.z_pin = grid, np.asarray(eta, float), float(z_pin)
        m = N_MIRROR
        self.x = np.r_[-grid.r_c[m::-1], grid.r_c, grid.r_v]
        self.y = np.r_[self.eta[m::-1], self.eta, z_pin]
        self.nx = len(self.x)

    # ---- local cubic on the interval that contains r
    def _stencil(self, k):
        """First node index of the 4-node stencil for interval [x_k, x_{k+1}]."""
        return np.clip(k - 1, 0, self.nx - 4)

    def _interval(self, r):
        k = np.searchsorted(self.x, r, side="right") - 1
        return np.clip(k, 0, self.nx - 2)

    def _coeffs(self, r, k):
        """Taylor coefficients (value, 1st, 2nd derivative) at r of the cubic of interval k."""
        r = np.atleast_1d(np.asarray(r, float)); k = np.atleast_1d(k)
        s = self._stencil(k)
        idx = s[:, None] + np.arange(4)[None, :]
        X = self.x[idx] - r[:, None]
        V = np.stack([np.ones_like(X), X, X**2, X**3], axis=-1)          # (M, 4 nodes, 4 powers)
        a = np.linalg.solve(V, self.y[idx][..., None])[..., 0]           # a0 + a1 d + a2 d^2 + a3 d^3
        return a[:, 0], a[:, 1], 2 * a[:, 2]

    def value(self, r):
        return self._coeffs(r, self._interval(r))[0]

    def derivs(self, r):
        """(eta, eta', eta'') at arbitrary r (interval cubic)."""
        return self._coeffs(r, self._interval(r))

    def node_derivs(self):
        """(eta', eta'') at the column centres: mean of the two adjacent interval cubics."""
        g = self.grid
        kn = np.searchsorted(self.x, g.r_c) - 0          # node index of r_c[i] in x
        _, d1l, d2l = self._coeffs(g.r_c, kn - 1)
        _, d1r, d2r = self._coeffs(g.r_c, kn)
        return 0.5 * (d1l + d1r), 0.5 * (d2l + d2r)

    @staticmethod
    def kappa_parts(r, d1, d2):
        q = np.sqrt(1 + d1**2)
        return -d2 / q**3, -d1 / (np.asarray(r) * q)

    def radial_crossing(self, r_lo, r_hi, z, i_lo=None):
        """r in [r_lo, r_hi] with eta(r) = z (a sign change is guaranteed by the caller).
        Linear mode: the root of the linear interpolant between columns i_lo, i_lo + 1."""
        if self.crossing == "linear":
            e0, e1 = self.eta[i_lo], self.eta[i_lo + 1]
            return np.asarray(r_lo) + (np.asarray(z) - e0) / (e1 - e0) * self.grid.dr
        a, b = np.array(r_lo, float), np.array(r_hi, float)
        fa = self.value(a) - z
        for _ in range(60):
            m = 0.5 * (a + b)
            fm = self.value(m) - z
            left = np.sign(fm) == np.sign(fa)
            a = np.where(left, m, a); fa = np.where(left, fm, fa)
            b = np.where(left, b, m)
        return 0.5 * (a + b)


def geometry_from_eta(grid: Grid, gi: GraphInterface):
    """(LiquidGeometry, radial-crossing r array on the u_r face grid) from eta alone."""
    Nr, Nz = grid.Nr, grid.Nz
    eta = gi.eta
    liquid = grid.z_c[None, :] < eta[:, None]
    kind_z = np.zeros((Nr, Nz + 1), dtype=np.int8)
    theta_z_raw = np.full((Nr, Nz + 1), np.nan)
    lm, lp = liquid[:, :-1], liquid[:, 1:]
    kind_z[:, 1:Nz][lm & lp] = FACE_LIQUID
    kind_z[:, 1:Nz][lm & ~lp] = FACE_LIQ_MINUS                  # liquid below (graph)
    kind_z[:, 1:Nz][~lm & lp] = FACE_LIQ_PLUS                   # impossible for a graph
    i, j = np.nonzero(lm & ~lp)
    theta_z_raw[i, j + 1] = (eta[i] - grid.z_c[j]) / grid.dz
    kind_r = np.zeros((Nr + 1, Nz), dtype=np.int8)
    theta_r_raw = np.full((Nr + 1, Nz), np.nan)
    r_cross = np.full((Nr + 1, Nz), np.nan)
    lm, lp = liquid[:-1, :], liquid[1:, :]
    kind_r[1:Nr][lm & lp] = FACE_LIQUID
    kind_r[1:Nr][lm & ~lp] = FACE_LIQ_MINUS
    kind_r[1:Nr][~lm & lp] = FACE_LIQ_PLUS
    for kind in (FACE_LIQ_MINUS, FACE_LIQ_PLUS):
        fi, fj = np.nonzero(kind_r == kind)
        if fi.size == 0:
            continue
        rc = gi.radial_crossing(grid.r_c[fi - 1], grid.r_c[fi], grid.z_c[fj], fi - 1)
        r_cross[fi, fj] = rc
        r_liq = grid.r_c[fi - 1] if kind == FACE_LIQ_MINUS else grid.r_c[fi]
        theta_r_raw[fi, fj] = np.abs(rc - r_liq) / grid.dr
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
    geom = LiquidGeometry(liquid=liquid, cell_class=cell_class, face_kind_r=kind_r, face_kind_z=kind_z,
                          theta_r=theta_r, theta_z=theta_z, theta_r_raw=theta_r_raw, theta_z_raw=theta_z_raw)
    return geom, r_cross


def curvature_at_crossings(grid: Grid, gi: GraphInterface, geom: LiquidGeometry, r_cross: np.ndarray):
    """(kappa_r, kappa_z) on the face grids (nan on non-mixed faces), from the graph,
    evaluated AT the crossings the pressure BC uses."""
    kr = np.full((grid.Nr + 1, grid.Nz), np.nan)
    kz = np.full((grid.Nr, grid.Nz + 1), np.nan)
    d1n, d2n = gi.node_derivs()
    kmn, ktn = GraphInterface.kappa_parts(grid.r_c, d1n, d2n)
    mz = (geom.face_kind_z == FACE_LIQ_MINUS) | (geom.face_kind_z == FACE_LIQ_PLUS)
    i, j = np.nonzero(mz)
    kz[i, j] = kmn[i] + ktn[i]
    mr = np.isfinite(r_cross)
    if mr.any():
        rr = r_cross[mr]
        if gi.crossing == "linear":
            fi = np.nonzero(mr)[0]                                   # u_r face index i: columns i-1, i
            w = (rr - grid.r_c[fi - 1]) / grid.dr
            kn = kmn + ktn
            kr[mr] = (1 - w) * kn[fi - 1] + w * kn[fi]
        else:
            _, d1, d2 = gi.derivs(rr)
            km, kt = GraphInterface.kappa_parts(rr, d1, d2)
            kr[mr] = km + kt
    return kr, kz


def column_areas(grid: Grid) -> np.ndarray:
    return np.pi * (grid.r_f[1:] ** 2 - grid.r_f[:-1] ** 2)


def column_volume(grid: Grid, eta: np.ndarray, z_b: float = 0.0) -> float:
    """The authoritative discrete liquid volume sum_i A_i (eta_i - z_b)."""
    return float(np.sum(column_areas(grid) * (eta - z_b)))


def radial_fluxes(grid: Grid, gi: GraphInterface, u_r: np.ndarray) -> np.ndarray:
    """F_{i-1/2} = 2 pi r_f[i] int_0^{eta(r_f[i])} u_r(r_f[i], z) dz on every u_r face
    column (length Nr + 1). The staggered u_r is piecewise constant in z per cell;
    the top partial cell counts with its liquid fraction. Axis and wall: 0."""
    Nr = grid.Nr
    F = np.zeros(Nr + 1)
    ef = gi.value(grid.r_f[1:Nr])
    zf = grid.z_f
    for k, i in enumerate(range(1, Nr)):
        h = ef[k]
        frac = np.clip((h - zf[:-1]) / grid.dz, 0.0, 1.0)          # liquid fraction of each cell
        F[i] = 2 * np.pi * grid.r_f[i] * grid.dz * float(np.sum(frac * u_r[i, :]))
    return F


def eta_rate(grid: Grid, eta: np.ndarray, z_pin: float, u_r: np.ndarray, crossing: str = "linear") -> np.ndarray:
    """d eta_i / dt = (F_{i-1/2} - F_{i+1/2}) / A_i (exactly volume conservative)."""
    F = radial_fluxes(grid, GraphInterface(grid, eta, z_pin, crossing), u_r)
    return (F[:-1] - F[1:]) / column_areas(grid)


def graph_validity(grid: Grid, gi: GraphInterface) -> dict:
    """Scope diagnostic: max |eta_r|, minimum principal-curvature resolution
    N_m = R_m/dx and N_theta = R_theta/dx over the columns."""
    d1, d2 = gi.node_derivs()
    km, kt = GraphInterface.kappa_parts(grid.r_c, d1, d2)
    tiny = 1e-300
    return {"max_slope": float(np.abs(d1).max()),
            "min_N_m": float((1.0 / np.maximum(np.abs(km), tiny)).min() / grid.dr),
            "min_N_theta": float((1.0 / np.maximum(np.abs(kt), tiny)).min() / grid.dr)}


def derived_psi(grid: Grid, eta: np.ndarray) -> np.ndarray:
    """psi = z - eta(r_c): DERIVED compatibility/plotting field (NOT a signed distance,
    never evolved, never used for curvature or eta evolution)."""
    return grid.z_c[None, :] - np.asarray(eta)[:, None]
