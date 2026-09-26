"""Benchmark builders for the opt-in single_phase_height research branch."""
from __future__ import annotations

import numpy as np

from .config import WallConfig
from .fields import allocate_fields
from .grid import build_grid
from .height_solver import SinglePhaseHeightSolver
from .meniscus import G, RHO, meniscus_config, solve_meniscus_pinned
from .pinned_phase import R_V, depth_for_phase


def _height_cfg(dx, H, z_pin, sigma=True):
    cfg = meniscus_config(dx, None, H=H, contact_model="pinned")
    cfg.physics.free_surface_model = "single_phase_height"
    cfg.wall = WallConfig("pinned", None, z_pin)
    if not sigma:
        cfg.fluid.surface_tension = 0.0
    return cfg


def build_height_meniscus(xi: float, dx: float, theta_ref: float = 60.0):
    """Pinned Young-Laplace meniscus (the V4b-P geometry), pin at sub-cell phase xi.
    eta_i = BVP height at r_c[i]; p = P0 - rho g z in the liquid; u = 0."""
    H, z_pin = depth_for_phase(xi, dx, theta_ref)
    cfg = _height_cfg(dx, H, z_pin)
    g = build_grid(cfg)
    if abs(g.Nr * g.dr - g.r_v) > 1e-9 * g.r_v:
        raise ValueError("R/dx must be an integer (the wall face must sit at R)")
    ref = solve_meniscus_pinned(z_pin, R_V, H)
    f = allocate_fields(g)
    eta = ref.eta(g.r_c)
    z2d = g.z_c[None, :] * np.ones(g.shape_center)
    f.p = np.where(g.z_c[None, :] < eta[:, None], ref.P0 - RHO * G * z2d, 0.0)
    return SinglePhaseHeightSolver(grid=g, cfg=cfg, fields=f, eta=eta, z_pin=z_pin), ref


def build_height_flat(H: float, dx: float, sigma: bool = False):
    """Flat free surface at height H (off-grid allowed), pinned at H, u = 0, hydrostatic p."""
    cfg = _height_cfg(dx, H, H, sigma)
    g = build_grid(cfg)
    f = allocate_fields(g)
    eta = np.full(g.Nr, H)
    z2d = g.z_c[None, :] * np.ones(g.shape_center)
    f.p = np.where(z2d < H, RHO * G * (H - z2d), 0.0)
    return SinglePhaseHeightSolver(grid=g, cfg=cfg, fields=f, eta=eta, z_pin=H)


def solve_meniscus_pinned_rotating(z_pin: float, R: float, H: float, omega: float, tol: float = 1e-10):
    """Independent reference for the integrated test: axisymmetric equilibrium of a
    liquid in solid-body rotation Omega with gravity and surface tension, pinned at
    eta(R) = z_pin, volume pi R^2 H. Interface pressure balance
        P0 + rho Omega^2 r^2 / 2 - rho g z = sigma kappa
    (Young-Laplace ODEs in arc length as in meniscus.solve_meniscus_pinned, with the
    centrifugal term added). Returns a MeniscusReference-like object (eta, P0, psi)."""
    from scipy.integrate import solve_bvp
    from .meniscus import MeniscusReference, SIGMA
    V = np.pi * R**2 * H

    def f(tau, y, p):
        S, P0 = p
        r, z, psi, _ = y
        a = (RHO * G * z - 0.5 * RHO * omega**2 * r**2 - P0) / SIGMA
        small = r < 1e-9 * R
        hoop = np.where(small, 0.0, np.sin(psi) / np.where(small, 1.0, r))
        dpsi = np.where(small, 0.5 * a, a - hoop)
        return np.vstack([S * np.cos(psi), S * np.sin(psi), S * dpsi, S * 2 * np.pi * r * z * np.cos(psi)])

    def bc(ya, yb, p):
        return np.array([ya[0], ya[2], ya[3], yb[0] - R, yb[1] - z_pin, yb[3] - V])

    tau = np.linspace(0, 1, 400)
    y0 = np.vstack([tau * R, H + (z_pin - H) * tau**2, 2 * (z_pin - H) / R * tau, tau**2 * V])
    sol = solve_bvp(f, bc, tau, y0, p=np.array([R, RHO * G * H]), tol=tol, max_nodes=200000)
    if not sol.success:
        raise RuntimeError(f"rotating pinned meniscus BVP failed: {sol.message}")
    y = sol.sol(np.linspace(0, 1, 4001))
    ref = MeniscusReference(float(np.rad2deg(np.pi / 2 - y[2][-1])), R, V, float(sol.p[1]), y[0], y[1], y[2],
                            float(np.max(np.abs(sol.rms_residuals))))
    ref.omega = omega
    return ref


def build_height_rotating(xi: float, dx: float, omega: float, theta_ref: float = 60.0):
    """Integrated gravity + capillarity + rotation equilibrium: pin from the static
    theta_ref meniscus at phase xi, reference from solve_meniscus_pinned_rotating,
    u_theta = Omega r, matched rotating wall and bottom, p = P0 + rho Omega^2 r^2/2 - rho g z."""
    from .single_phase_solver import WallBC
    H, z_pin = depth_for_phase(xi, dx, theta_ref)
    cfg = _height_cfg(dx, H, z_pin)
    g = build_grid(cfg)
    if abs(g.Nr * g.dr - g.r_v) > 1e-9 * g.r_v:
        raise ValueError("R/dx must be an integer")
    ref = solve_meniscus_pinned_rotating(z_pin, R_V, H, omega)
    f = allocate_fields(g)
    eta = ref.eta(g.r_c)
    r2, z2 = g.r_c[:, None] * np.ones(g.shape_center), g.z_c[None, :] * np.ones(g.shape_center)
    liq = z2 < eta[:, None]
    f.u_theta = np.where(liq, omega * r2, 0.0)
    f.p = np.where(liq, ref.P0 + 0.5 * RHO * omega**2 * r2**2 - RHO * G * z2, 0.0)
    s = SinglePhaseHeightSolver(grid=g, cfg=cfg, fields=f, eta=eta, z_pin=z_pin, wall_bc=WallBC.rotating(omega))
    return s, ref


def build_height_production(rpm: float, dx: float = 0.5e-3, config_path=None):
    """QUALITATIVE forcing sanity: baseline vessel/stirrer geometry (placeholder values from
    configs/baseline.yaml, NOT measurements), flat free surface pinned at its initial height,
    uncalibrated tau_s, stationary no-slip wall, bottom-localized stirrer forcing."""
    import copy
    from pathlib import Path
    from .config import load_config
    root = Path(__file__).resolve().parents[2]
    cfg = copy.deepcopy(load_config(config_path or root / "configs" / "baseline.yaml"))
    cfg.physics.free_surface_model = "single_phase_height"
    cfg.grid.dr_m = cfg.grid.dz_m = dx
    cfg.stirrer.rpm = rpm
    H = cfg.geometry.water_height_m
    cfg.wall = WallConfig("pinned", None, H)
    g = build_grid(cfg)
    f = allocate_fields(g)
    eta = np.full(g.Nr, H)
    z2d = g.z_c[None, :] * np.ones(g.shape_center)
    f.p = np.where(z2d < H, RHO * G * (H - z2d), 0.0)
    return SinglePhaseHeightSolver(grid=g, cfg=cfg, fields=f, eta=eta, z_pin=H)
