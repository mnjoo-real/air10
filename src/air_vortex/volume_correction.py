"""Optional global water-volume correction (README "Level Set is non-
conservative" / "optional global volume correction").

Level Set advection -- even the conservative MUSCL2 flux-form scheme, since
it is only conservative up to the pressure projection's residual divergence
(machine precision, but not exactly zero) -- can still drift the total
water volume over a long run. This module implements an OPTIONAL, OFF-BY-
DEFAULT correction: after advection (and periodic reinitialization), shift
phi by a small uniform constant so the water volume exactly matches a
target (normally the initial volume), found by 1D root-finding.

This deliberately does NOT reshape the interface locally -- only a single
uniform additive shift to the whole field, so it cannot introduce spurious
local curvature (README explicit requirement: "interface shape를 억지로
변형시키는 local correction은 하지 말 것").
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from .grid import Grid


@dataclass
class VolumeCorrectionResult:
    applied: bool
    delta: float
    volume_before: float
    volume_after: float
    target_volume: float
    large_correction_warning: bool


def _water_volume(grid: Grid, phi: np.ndarray) -> float:
    # Local re-implementation (not importing air_vortex.solver.water_volume)
    # to avoid a circular import: solver.py imports from levelset.py, which
    # would need to import this module for the solver's correction hook.
    # Must stay numerically identical to solver.water_volume -- see
    # tests/test_volume_correction.py::test_matches_solver_water_volume.
    water = (phi < 0).astype(float)
    cell_volume = 2.0 * np.pi * grid.r_c[:, None] * grid.dr * grid.dz
    return float(np.sum(water * cell_volume))


def find_volume_correction_delta(grid: Grid, phi: np.ndarray, target_volume: float,
                                  search_margin_factor: float = 2.0,
                                  volume_fn=None) -> float:
    """Find delta such that water_volume(grid, phi + delta) == target_volume.

    phi < 0 = water: ADDING a positive delta raises phi everywhere, pushing
    cells from water (phi<0) toward air (phi>=0), so volume is a strictly
    decreasing function of delta -- monotonic and safe for brentq.

    ``volume_fn(grid, phi)`` defaults to the staircase count (legacy
    behavior, unchanged). The single-phase path passes the continuous
    sub-cell volume (liquid_mask.liquid_volume_subcell): with the
    staircase count the root lies anywhere on a zero plateau up to ~a
    cell wide, so the "correction" would itself move the interface."""
    vol = _water_volume if volume_fn is None else volume_fn

    def f(delta: float) -> float:
        return vol(grid, phi + delta) - target_volume

    span = search_margin_factor * grid.z_max
    lo, hi = -span, span  # f(lo) > 0 (very negative shift -> more water), f(hi) < 0
    f_lo, f_hi = f(lo), f(hi)
    if f_lo * f_hi > 0:
        raise RuntimeError(
            f"volume correction: target volume {target_volume:.6e} m^3 is not "
            f"reachable by a uniform phi shift within +/-{span:.4g} m "
            f"(f(lo)={f_lo:.3e}, f(hi)={f_hi:.3e}); the interface may already "
            f"be far from any reasonable state."
        )
    return brentq(f, lo, hi, xtol=1e-12, rtol=1e-12)


def apply_volume_correction(grid: Grid, phi: np.ndarray, target_volume: float,
                             large_correction_threshold_m: float | None = None,
                             volume_fn=None
                             ) -> tuple[np.ndarray, VolumeCorrectionResult]:
    """Apply the uniform phi shift found by :func:`find_volume_correction_delta`.
    ``large_correction_threshold_m`` (default: one grid cell) flags -- via
    the returned result, not a raised exception -- corrections large enough
    to suggest the underlying advection error is not "small drift" anymore."""
    if large_correction_threshold_m is None:
        large_correction_threshold_m = min(grid.dr, grid.dz)

    vol = _water_volume if volume_fn is None else volume_fn
    v_before = vol(grid, phi)
    delta = find_volume_correction_delta(grid, phi, target_volume, volume_fn=volume_fn)
    phi_corrected = phi + delta
    v_after = vol(grid, phi_corrected)

    result = VolumeCorrectionResult(
        applied=True, delta=delta, volume_before=v_before, volume_after=v_after,
        target_volume=target_volume,
        large_correction_warning=abs(delta) > large_correction_threshold_m,
    )
    return phi_corrected, result
