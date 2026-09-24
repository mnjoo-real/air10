"""Zero-contour preservation metrics for reinitialization (README_rewritten
9.4) and the reinitialization dispatcher used by the single-phase path.

Primary metric: displacement of the phi = 0 contour, measured with the
SAME sub-cell linear crossings the pressure BC uses
(liquid_mask.classify): every grid line (vertical and radial) that crosses
the interface before reinitialization is searched for its crossing after,
and the positional difference along that line is recorded. Secondary
metric: E_sd = RMS(|grad phi| - 1) in the band |phi| <= band * dx.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .grid import Grid
from .liquid_mask import FACE_LIQ_MINUS, FACE_LIQ_PLUS, LiquidGeometry, classify
from .operators import center_grad_r, center_grad_z

REINIT_METHODS = ("legacy_godunov", "russo_smereka_subcell")


def interface_crossings(grid: Grid, geom: LiquidGeometry) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Sub-cell crossing positions per grid line: (z-crossings per column
    i, r-crossings per row j)."""
    zc = []
    for i in range(grid.Nr):
        k = geom.face_kind_z[i, 1:grid.Nz]
        th = geom.theta_z_raw[i, 1:grid.Nz]
        pos = []
        for jf in np.nonzero(k == FACE_LIQ_MINUS)[0]:
            pos.append(grid.z_c[jf] + th[jf] * grid.dz)       # liquid below
        for jf in np.nonzero(k == FACE_LIQ_PLUS)[0]:
            pos.append(grid.z_c[jf + 1] - th[jf] * grid.dz)   # liquid above
        zc.append(np.sort(np.array(pos)))
    rc = []
    for j in range(grid.Nz):
        k = geom.face_kind_r[1:grid.Nr, j]
        th = geom.theta_r_raw[1:grid.Nr, j]
        pos = []
        for f in np.nonzero(k == FACE_LIQ_MINUS)[0]:
            pos.append(grid.r_c[f] + th[f] * grid.dr)
        for f in np.nonzero(k == FACE_LIQ_PLUS)[0]:
            pos.append(grid.r_c[f + 1] - th[f] * grid.dr)
        rc.append(np.sort(np.array(pos)))
    return zc, rc


@dataclass
class ContourDisplacement:
    """Displacement of the phi = 0 contour.

    ``max_shift`` / ``rms_shift`` are NORMAL displacements: each along-line
    crossing shift s is multiplied by |n . e_line| (n = unit normal from
    grad phi_old at the crossing). Along a grid line nearly tangent to a
    curved interface, a normal motion delta moves the crossing by
    delta / |n . e_line|, which is unbounded at tangency -- the raw
    along-line numbers are kept as ``max_shift_along_line`` for reference."""
    max_shift: float
    rms_shift: float
    mean_signed_z: float     # normal-projected, along vertical lines (+ = up)
    mean_signed_r: float     # normal-projected, along radial lines (+ = outward)
    n_crossings: int
    n_lost: int              # old crossings with no new crossing within 3 cells
    dx: float
    max_shift_along_line: float = 0.0

    @property
    def max_shift_dx(self) -> float:
        return self.max_shift / self.dx

    @property
    def rms_shift_dx(self) -> float:
        return self.rms_shift / self.dx


def contour_displacement(grid: Grid, phi_old: np.ndarray, phi_new: np.ndarray,
                         exclude_boundary_cells: int = 0) -> ContourDisplacement:
    """Line-by-line displacement of the phi = 0 crossings from phi_old to
    phi_new, projected on the interface normal. ``exclude_boundary_cells``
    skips lines within that many cells of the wall/top/bottom (not the
    axis)."""
    zo, ro = interface_crossings(grid, classify(phi_old))
    zn, rn = interface_crossings(grid, classify(phi_new))
    gr = center_grad_r(grid, phi_old)
    gz = center_grad_z(grid, phi_old)
    mag = np.hypot(gr, gz) + 1e-300
    n_r, n_z = np.abs(gr) / mag, np.abs(gz) / mag

    def normal_component(line_is_z: bool, k: int, x: float) -> float:
        # interpolate |n . e_line| linearly along the line at position x
        if line_is_z:
            return float(np.interp(x, grid.z_c, n_z[k, :]))
        return float(np.interp(x, grid.r_c, n_r[:, k]))

    e = exclude_boundary_cells
    shifts_z, shifts_r, along, lost = [], [], [], 0
    search = 3.0
    for is_z, lines_o, lines_n, h, out, n_lines in ((True, zo, zn, grid.dz, shifts_z, grid.Nr),
                                                    (False, ro, rn, grid.dr, shifts_r, grid.Nz)):
        for k in range(n_lines):
            if e and (k >= n_lines - e or (not is_z and k < e)):
                continue
            o, nw = lines_o[k], lines_n[k]
            for x in o:
                if nw.size == 0:
                    lost += 1
                    continue
                d = nw - x
                m = d[np.argmin(np.abs(d))]
                if abs(m) > search * h:
                    lost += 1
                    continue
                along.append(abs(m))
                out.append(m * normal_component(is_z, k, x))
    allv = np.abs(np.array(shifts_z + shifts_r))
    dx = min(grid.dr, grid.dz)
    return ContourDisplacement(
        max_shift=float(allv.max()) if allv.size else 0.0,
        rms_shift=float(np.sqrt(np.mean(allv**2))) if allv.size else 0.0,
        mean_signed_z=float(np.mean(shifts_z)) if shifts_z else 0.0,
        mean_signed_r=float(np.mean(shifts_r)) if shifts_r else 0.0,
        n_crossings=int(allv.size), n_lost=lost, dx=dx,
        max_shift_along_line=float(max(along)) if along else 0.0)


def signed_distance_error(grid: Grid, phi: np.ndarray, band_cells: float = 3.0,
                          exclude_boundary: bool = True) -> float:
    """E_sd = RMS(|grad phi| - 1) over |phi| <= band_cells*dx (centered
    differences; one-sided boundary rows excluded by default)."""
    g = np.sqrt(center_grad_r(grid, phi) ** 2 + center_grad_z(grid, phi) ** 2)
    band = np.abs(phi) <= band_cells * min(grid.dr, grid.dz)
    if exclude_boundary:
        band[-1, :] = False
        band[:, 0] = False
        band[:, -1] = False
    if not np.any(band):
        return 0.0
    return float(np.sqrt(np.mean((g[band] - 1.0) ** 2)))


def reinitialize(phi: np.ndarray, grid: Grid, levelset_cfg) -> np.ndarray:
    """Dispatch on levelset.reinitialization_method (single-phase path)."""
    method = levelset_cfg.reinitialization_method
    if method == "legacy_godunov":
        from .levelset import reinitialize_level_set
        return reinitialize_level_set(phi, grid, levelset_cfg.reinitialize_iterations)
    if method == "russo_smereka_subcell":
        from .reinit_subcell import reinitialize_subcell
        return reinitialize_subcell(phi, grid, levelset_cfg.reinitialize_iterations,
                                    cfl=levelset_cfg.reinitialization_cfl,
                                    order=levelset_cfg.reinitialization_order,
                                    band_cells=levelset_cfg.reinitialization_band_cells)
    raise ValueError(f"unknown reinitialization_method {method!r}")
