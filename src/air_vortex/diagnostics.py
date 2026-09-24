"""Vortex-depth and air-core-geometry diagnostics (README sections 21, 23, 24)."""
from __future__ import annotations

import numpy as np

from .config import Config
from .connectivity import air_mask, top_connected_air
from .grid import Grid


def _interpolate_zero_crossing(z_low: float, phi_low: float, z_high: float, phi_high: float) -> float:
    """Linear interpolation of the z where phi crosses 0, given
    phi_low < 0 <= phi_high at z_low < z_high (water below, air above --
    README section 6/9 sign convention). Shared by :func:`find_tip_z` and
    :func:`free_surface_height` so both read the interface the same way."""
    if phi_high == phi_low:
        return 0.5 * (z_low + z_high)
    frac = -phi_low / (phi_high - phi_low)
    return z_low + frac * (z_high - z_low)


def find_tip_z(phi: np.ndarray, grid: Grid, cfg: Config) -> float:
    """z_tip(t): sub-grid-interpolated height of the lowest boundary of the
    top-connected air region, restricted to the central column r <= R_m to
    avoid the wall meniscus (README section 21).

    Structure (as requested when this was fixed -- README "Performance"
    validation notes on grid-quantized vortex depth): first find the
    top-connected air component (connectivity.top_connected_air, same
    definition used everywhere else), then its lowest row within the
    central columns, then interpolate that specific boundary's phi=0
    crossing -- rather than returning the raw grid face/cell location.
    Previously returned ``grid.z_f[j_min]`` directly, which made every
    downstream vortex-depth reading jump in exact multiples of the grid
    spacing between resolutions instead of converging smoothly
    (scripts/run_grid_convergence.py first exposed this)."""
    R_m = cfg.geometry.stirbar_half_length_m
    a_top = top_connected_air(air_mask(phi))

    central_mask = grid.r_c <= R_m
    central_idx = np.nonzero(central_mask)[0]
    sub = a_top[central_mask, :]
    if not np.any(sub):
        return grid.z_max  # no depression: tip at the undisturbed surface region top

    i_local, j_idx = np.nonzero(sub)
    j_min = int(j_idx.min())

    if j_min == 0:
        # air already reaches the bottom-most cell: no water cell below to
        # interpolate against, fall back to the bottom face (as before).
        return float(grid.z_f[0])

    # Average the interpolated crossing over every central column that
    # ties for the minimum row, rather than picking an arbitrary one.
    tied_i_global = central_idx[i_local[j_idx == j_min]]
    z_low, z_high = grid.z_c[j_min - 1], grid.z_c[j_min]
    crossings = [
        _interpolate_zero_crossing(z_low, phi[i, j_min - 1], z_high, phi[i, j_min])
        for i in tied_i_global
    ]
    return float(np.mean(crossings))


def vortex_depth(phi: np.ndarray, grid: Grid, cfg: Config) -> float:
    """d(t) = H - z_tip(t) (README section 21, boxed)."""
    z_tip = find_tip_z(phi, grid, cfg)
    return cfg.geometry.water_height_m - z_tip


def air_core_radius_profile(phi: np.ndarray, grid: Grid) -> np.ndarray:
    """r_air(z): outer radius of the central top-connected air column at
    every height (README section 23). Returns NaN where no air is present."""
    a_top = top_connected_air(air_mask(phi))
    r_air = np.full(grid.Nz, np.nan)

    for j in range(grid.Nz):
        row = a_top[:, j]
        if not row[0]:
            continue  # air column must start at the axis to count as "central"
        i_last = 0
        for i in range(grid.Nr):
            if row[i]:
                i_last = i
            else:
                break
        r_air[j] = grid.r_f[i_last + 1]

    return r_air


def free_surface_height(phi: np.ndarray, grid: Grid) -> np.ndarray:
    """eta(r): height of the (outermost/topmost) water-air interface in
    each radial column, by linear interpolation of the phi=0 crossing
    (README section 6/9 sign convention: phi<0 water, phi>0 air).

    Scans each column from the top down and takes the first air-to-water
    crossing. This is the *free surface* (used for the solid-body-rotation
    paraboloid check, README section 8/31 Test 2), which is deliberately a
    different quantity from :func:`find_tip_z` -- that one restricts to the
    top-connected component within r<=R_m specifically to track the vortex
    core once an air channel has formed. Returns NaN for an all-water or
    all-air column (no crossing found)."""
    Nr, Nz = grid.Nr, grid.Nz
    eta = np.full(Nr, np.nan)
    for i in range(Nr):
        col = phi[i, :]
        for j in range(Nz - 1, 0, -1):
            if col[j] >= 0.0 and col[j - 1] < 0.0:
                eta[i] = _interpolate_zero_crossing(grid.z_c[j - 1], col[j - 1], grid.z_c[j], col[j])
                break
    return eta


def scalar_diagnostics(fields, grid: Grid, cfg: Config) -> dict:
    """Per-step scalar diagnostics for logging (README section 24). Only
    quantities computable from a single instant (fields, grid, cfg) live
    here; cross-step state (stable air core / persistence, statistically
    steady) is tracked separately by the run loop (scripts/run_single.py)
    since it needs history, not just the current frame."""
    from .connectivity import is_geometrically_connected
    from .operators import divergence
    from .solver import water_volume

    div = divergence(grid, fields.u_r, fields.u_z)
    single_phase = cfg.physics.free_surface_model == "single_phase_ls"
    if single_phase:
        # the void extension band is not physical flow and is not
        # divergence-free; only liquid cells are constrained
        div = np.where(fields.phi < 0, div, 0.0)

    row = {
        "t": fields.t,
        "step": fields.step,
        "d": vortex_depth(fields.phi, grid, cfg),
        "max_u_theta": float(np.max(np.abs(fields.u_theta))),
        "max_downward_uz": float(-np.min(fields.u_z)) if fields.u_z.size else 0.0,
        "min_pressure": float(np.min(fields.p)),
        "max_divergence": float(np.max(np.abs(div))),
        "water_volume": water_volume(grid, fields.phi),
        "air_core_connected": is_geometrically_connected(fields.phi, grid, cfg),
    }
    if single_phase:
        from .liquid_mask import liquid_volume_subcell
        row["water_volume_subcell"] = liquid_volume_subcell(grid, fields.phi)
        row["free_surface_model"] = "single_phase_ls"
    return row


def volume_consistent_parabola_constant(R_v: float, omega: float, g: float, target_volume: float) -> float:
    """C in eta(r) = C + Omega^2 r^2 / (2g), chosen so the analytic
    paraboloid encloses exactly ``target_volume`` (README "solid-body
    analytical free surface의 volume-consistent form"):

        integral_0^{R_v} 2*pi*r*eta(r) dr = target_volume
        => pi*C*R_v^2 + pi*Omega^2*R_v^4/(4g) = target_volume
        => C = target_volume/(pi*R_v^2) - Omega^2*R_v^2/(4g)

    For target_volume = pi*R_v^2*H (a flat cylinder of depth H -- the
    initial condition, README section 13), this reduces to the closed form
    C = H - Omega^2*R_v^2/(4g). Passing the actual initial water_volume
    (rather than assuming a perfect flat cylinder) accounts for any
    geometry constraints if the domain isn't a simple cylinder."""
    return target_volume / (np.pi * R_v**2) - omega**2 * R_v**2 / (4.0 * g)


def volume_consistent_parabola(r: np.ndarray, R_v: float, omega: float, g: float,
                                target_volume: float) -> np.ndarray:
    """eta(r) = C + Omega^2 r^2/(2g), with C from
    :func:`volume_consistent_parabola_constant`."""
    C = volume_consistent_parabola_constant(R_v, omega, g, target_volume)
    return C + omega**2 * r**2 / (2.0 * g)
