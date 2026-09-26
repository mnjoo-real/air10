"""Grid-phase sweep machinery for the pinned contact line (Gate V4b-P).

A vertical translation of the whole liquid by delta (adding a uniform layer
of depth delta) is an EXACT symmetry of the gravity-capillary meniscus in a
flat-bottomed cylinder: the interface translates by delta with an
identical shape. Choosing delta = (xi - xi0) dz therefore puts the pin at
any sub-cell phase xi = frac(z_pin / dz) while the continuum problem is
unchanged. Cell centres sit at xi = 0.5.
"""
from __future__ import annotations

import copy

import numpy as np

from .config import WallConfig
from .curvature_single_phase import crossing_positions, curvature_at_crossings, curvature_components_centers
from .liquid_mask import classify
from .meniscus import G, RHO, SIGMA, meniscus_config, solve_meniscus, solve_meniscus_pinned
from .single_phase_solver import signed_distance_to_profile

R_V, H0 = 0.008, 0.006


def phase_of(z: float, dz: float) -> float:
    return float((z / dz) % 1.0)


def depth_for_phase(xi: float, dx: float, theta_ref: float = 60.0) -> tuple[float, float]:
    """(H, z_pin): the reference meniscus (prescribed-theta BVP at H0,
    translated) with its wall height at sub-cell phase xi."""
    ref = solve_meniscus(theta_ref, R_V, H0)
    shift = ((xi - phase_of(ref.z_wall, dx)) % 1.0) * dx
    return H0 + shift, ref.z_wall + shift


def extended_profile(ref, r):
    """BVP profile; beyond the wall, its quadratic Taylor continuation (for
    the exact ghost reference)."""
    r = np.asarray(r)
    zR = ref.z_wall
    s1 = np.tan(ref.psi[-1])
    # eta'' from the meridional curvature at the wall: kappa_m = -eta''/(1+eta'^2)^1.5
    kappa_total = ref.kappa_at_height(zR)
    kappa_theta = -np.sin(ref.psi[-1]) / R_V
    s2 = -(kappa_total - kappa_theta) * (1 + s1**2) ** 1.5
    inside = ref.eta(np.clip(r, 0, R_V))
    out = zR + s1 * (r - R_V) + 0.5 * s2 * (r - R_V) ** 2
    return np.where(r <= R_V, inside, out)


def wall_compatible_sdf(grid, ref, margin_cells: int = 4):
    """phi = signed distance to the pinned BVP profile CONTINUED beyond the
    wall (extended_profile), sampled over |r| <= R + margin.

    single_phase_solver.signed_distance_to_profile samples the curve only on
    |r| <= R. For a tilted meniscus the level sets phi = +-O(dx) near the wall
    are then distances to the endpoint (R, z_pin), i.e. circles of radius
    O(dx), and the level-set curvature of the wall column is O(1/dx) wrong
    (V4b-P diagnosis: ~+45 1/m at dx = 0.25 mm, theta = 60, doubling with
    each halving of dx). Without reinitialization that error is never
    removed, so a wall-equilibrium test must start from this field."""
    rs = np.linspace(-grid.r_v - margin_cells * grid.dr, grid.r_v + margin_cells * grid.dr, 40001)
    zs = extended_profile(ref, np.abs(rs))
    out = np.empty(grid.shape_center)
    for i in range(grid.Nr):
        d2 = (grid.r_c[i] - rs[None, :]) ** 2 + (grid.z_c[:, None] - zs[None, :]) ** 2
        out[i] = np.sign(grid.z_c - ref.eta(grid.r_c[i])) * np.sqrt(d2.min(axis=1))
    return out


def geometry_sweep_row(xi, dx, method, fit="quadratic", theta_ref=60.0):
    """Static geometry metrics at one phase for one pinned method."""
    H, z_pin = depth_for_phase(xi, dx, theta_ref)
    cfg = meniscus_config(dx, None, H=H, contact_model="pinned")
    cfg.wall = WallConfig("pinned", None, z_pin, method if method != "legacy" else "legacy", fit)
    from .grid import build_grid
    g = build_grid(cfg)
    ref = solve_meniscus_pinned(z_pin, R_V, H)
    phi = signed_distance_to_profile(g, ref.eta)
    # exact ghost: signed distance to the extended profile at r = R + dr/2
    rg = R_V + 0.5 * dx
    rs = np.linspace(-R_V - 3 * dx, R_V + 3 * dx, 40001)
    zs = extended_profile(ref, np.abs(rs))
    d2 = (rg - rs[None, :]) ** 2 + (g.z_c[:, None] - zs[None, :]) ** 2
    ghost_exact = np.sign(g.z_c - extended_profile(ref, rg)) * np.sqrt(d2.min(axis=1))
    from .contact_angle import reconstruct_pinned, wall_ghost_column
    ghost = wall_ghost_column(g, phi, cfg.wall)
    near = np.abs(ghost_exact) < 2 * dx
    km_c, kt_c = curvature_components_centers(g, phi, cfg.wall)
    geom = classify(phi)
    km_r, km_z = curvature_at_crossings(g, geom, km_c)
    kt_r, kt_z = curvature_at_crossings(g, geom, kt_c)
    rr, zr, rz, zz = crossing_positions(g, geom)
    # wall-column crossings (vertical, column N-1; radial, face N-2|N-1)
    sel_z, sel_r = np.isfinite(km_z[-1, :]), np.isfinite(km_r[-2, :])
    km = np.r_[km_z[-1, sel_z], km_r[-2, sel_r]]
    kt = np.r_[kt_z[-1, sel_z], kt_r[-2, sel_r]]
    rx = np.r_[rz[-1, sel_z], rr[-2, sel_r]]
    zx = np.r_[zz[-1, sel_z], zr[-2, sel_r]]
    if method == "reconstruct_direct":
        rec = reconstruct_pinned(g, phi, z_pin, fit, cfg.wall.pinned_fit_columns)
        km, kt = rec.kappa_parts(rx)
    # exact components at the crossing: psi at that r from the BVP
    psi = np.interp(rx, ref.r, ref.psi)
    kt_ex = -np.sin(psi) / rx
    k_ex = ref.kappa_at_height(zx)
    km_ex = k_ex - kt_ex
    scale = max(np.abs(ref.kappa_at_height(ref.z)).max(), 1.0 / R_V)
    slope_err = np.nan
    if method != "legacy":
        rec = reconstruct_pinned(g, phi, z_pin, fit, cfg.wall.pinned_fit_columns)
        slope_err = rec.wall_slope - np.tan(ref.psi[-1])
    return {"xi": xi, "dx_mm": dx * 1e3, "method": method, "fit": fit,
            "ghost_err_dx": float(np.abs(ghost - ghost_exact)[near].max() / dx),
            "wall_slope_err": float(slope_err),
            "kappa_m_err": float(np.abs(km - km_ex).max() / scale),
            "kappa_theta_err": float(np.abs(kt - kt_ex).max() / scale),
            "kappa_err": float(np.abs(km + kt - k_ex).max() / scale)}


def build_phase_solver(xi, dx, method, fit="quadratic", theta_ref=60.0, ic="truncated"):
    """CFD solver at phase xi: phi = signed distance to the PINNED BVP
    equilibrium, p = P0 - rho g z, pinned wall with the continuous z_pin.

    ic="truncated" (historical default, reproduces the first V4b-P sweeps):
    distance to the profile cut at r = R. ic="extended": wall_compatible_sdf,
    the compatible initial condition for wall-equilibrium tests."""
    from .fields import allocate_fields
    from .grid import build_grid
    from .single_phase_solver import SinglePhaseSolver
    H, z_pin = depth_for_phase(xi, dx, theta_ref)
    cfg = meniscus_config(dx, None, H=H, contact_model="pinned")
    cfg.wall = WallConfig("pinned", None, z_pin, method, fit)
    ref = solve_meniscus_pinned(z_pin, R_V, H)
    g = build_grid(cfg)
    f = allocate_fields(g)
    if ic == "extended":
        f.phi = wall_compatible_sdf(g, ref)
    elif ic == "truncated":
        f.phi = signed_distance_to_profile(g, ref.eta)
    else:
        raise ValueError(ic)
    if abs(g.Nr * g.dr - g.r_v) > 1e-9 * g.r_v:
        raise ValueError(f"R/dx = {g.r_v / g.dr:.4f} is not an integer: the wall face would sit at "
                         f"{g.Nr * g.dr:.6g} m instead of R = {g.r_v} m (grid.build_grid rounds Nr)")
    z2d = g.z_c[None, :] * np.ones(g.shape_center)
    f.p = np.where(f.phi < 0, ref.P0 - RHO * G * z2d, 0.0)
    return SinglePhaseSolver(grid=g, cfg=cfg, fields=f), ref
