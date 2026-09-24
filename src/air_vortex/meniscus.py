"""Static gravity-capillary meniscus in a cylinder (Gate V4b-S).

Independent reference (no CFD discretization involved)
------------------------------------------------------
Nonlinear axisymmetric Young-Laplace equation, arc-length form. psi is the
tangent angle above horizontal, liquid below, p_atm = 0:

    sigma * kappa = p_liquid(interface) = P0 - rho g z,
    kappa = div(n) = -(dpsi/ds + sin(psi)/r)

    dr/ds = cos psi,  dz/ds = sin psi,
    dpsi/ds = (rho g z - P0)/sigma - sin(psi)/r    [axis: half of the first term]
    dV/ds   = 2 pi r z cos psi                     (volume under the graph)

    r(0) = 0, psi(0) = 0, V(0) = 0,
    r(S) = R, psi(S) = pi/2 - theta   (eta'(R) = cot theta; theta through liquid),
    V(S) = V_target

The unknowns are the arc length S and P0, solved with scipy.integrate.solve_bvp
on tau = s/S. The small-slope Bessel solution (:func:`small_slope_meniscus`)
is a secondary cross-check near theta = 90 deg.

CFD harness
-----------
:func:`build_meniscus_solver` initializes phi as the signed distance to the
BVP profile and p = P0 - rho g z in the liquid. It uses the PRODUCTION
no-slip wall and the chosen wall contact model.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.integrate import solve_bvp
from scipy.interpolate import interp1d
from scipy.special import iv

from .config import Config, WallConfig, load_config

ROOT = Path(__file__).resolve().parents[2]
RHO, SIGMA, G = 998.0, 0.072, 9.81


@dataclass
class MeniscusReference:
    theta_deg: float
    R: float
    volume: float
    P0: float
    r: np.ndarray
    z: np.ndarray
    psi: np.ndarray
    residual_max: float

    def eta(self, r):
        return interp1d(self.r, self.z, kind="cubic")(np.clip(r, self.r[0], self.r[-1]))

    @property
    def z_wall(self) -> float:
        return float(self.z[-1])

    def kappa_at_height(self, z):
        """Exact curvature on the interface at height z (Young-Laplace)."""
        return (self.P0 - RHO * G * np.asarray(z)) / SIGMA


def solve_meniscus(theta_deg: float, R: float, H: float, rho=RHO, sigma=SIGMA, g=G,
                   tol=1e-10) -> MeniscusReference:
    V = np.pi * R**2 * H
    th = np.deg2rad(theta_deg)

    def f(tau, y, p):
        S, P0 = p
        r, z, psi, _ = y
        a = (rho * g * z - P0) / sigma
        small = r < 1e-9 * R
        hoop = np.where(small, 0.0, np.sin(psi) / np.where(small, 1.0, r))
        dpsi = np.where(small, 0.5 * a, a - hoop)
        return np.vstack([S * np.cos(psi), S * np.sin(psi), S * dpsi,
                          S * 2 * np.pi * r * z * np.cos(psi)])

    def bc(ya, yb, p):
        return np.array([ya[0], ya[2], ya[3], yb[0] - R, yb[2] - (np.pi / 2 - th), yb[3] - V])

    tau = np.linspace(0, 1, 400)
    y0 = np.vstack([tau * R, np.full_like(tau, H), tau * (np.pi / 2 - th), tau**2 * V])
    sol = solve_bvp(f, bc, tau, y0, p=np.array([R, rho * g * H]), tol=tol, max_nodes=200000)
    if not sol.success:
        raise RuntimeError(f"meniscus BVP failed: {sol.message}")
    tt = np.linspace(0, 1, 4001)
    y = sol.sol(tt)
    return MeniscusReference(theta_deg, R, V, float(sol.p[1]), y[0], y[1], y[2],
                             float(np.max(np.abs(sol.rms_residuals))))


def small_slope_meniscus(theta_deg: float, R: float, H: float, rho=RHO, sigma=SIGMA, g=G):
    """Linearized reference: eta = P0/(rho g) + A I0(r/lc), eta'(R) = cot theta,
    volume pi R^2 H. Returns (eta_fn, P0)."""
    lc = np.sqrt(sigma / (rho * g))
    A = lc / np.tan(np.deg2rad(theta_deg)) / iv(1, R / lc)
    P0 = rho * g * (H - 2 * A * lc * iv(1, R / lc) / R)
    return (lambda r: P0 / (rho * g) + A * iv(0, np.asarray(r) / lc)), P0


def meniscus_config(dx: float, theta_deg: float | None, R: float = 0.008, H: float = 0.006,
                    air: float = 0.006, contact_model: str = "static_angle") -> Config:
    cfg = copy.deepcopy(load_config(ROOT / "configs" / "single_phase_hydrostatic.yaml"))
    cfg.geometry.vessel_radius_m = R
    cfg.geometry.water_height_m = H
    cfg.geometry.air_height_m = air
    cfg.grid.dr_m = cfg.grid.dz_m = dx
    cfg.fluid.surface_tension = SIGMA
    cfg.fluid.gravity = G
    cfg.time.dt_max_s = 1.0
    cfg.wall = (WallConfig("static_angle", theta_deg) if contact_model == "static_angle"
                else WallConfig(contact_model, None))
    return cfg


def build_meniscus_solver(dx: float, theta_deg: float, contact_model: str = "static_angle",
                          start: str = "equilibrium", **kw):
    """start='equilibrium': phi = signed distance to the BVP meniscus.
    start='flat': flat surface at the same volume (a transient that needs
    contact-line motion; used only to demonstrate the no-slip limitation)."""
    from .fields import allocate_fields
    from .grid import build_grid
    from .single_phase_solver import SinglePhaseSolver, signed_distance_to_profile

    cfg = meniscus_config(dx, theta_deg, contact_model=contact_model, **kw)
    ref = solve_meniscus(theta_deg, cfg.geometry.vessel_radius_m, cfg.geometry.water_height_m)
    grid = build_grid(cfg)
    fields = allocate_fields(grid)
    if start == "equilibrium":
        fields.phi = signed_distance_to_profile(grid, ref.eta)
        z2d = grid.z_c[None, :] * np.ones(grid.shape_center)
        fields.p = np.where(fields.phi < 0, ref.P0 - RHO * G * z2d, 0.0)
    elif start == "flat":
        fields.phi = grid.z_c[None, :] - cfg.geometry.water_height_m + 0 * grid.r_c[:, None]
        z2d = grid.z_c[None, :] * np.ones(grid.shape_center)
        fields.p = np.where(fields.phi < 0, RHO * G * (cfg.geometry.water_height_m - z2d), 0.0)
    else:
        raise ValueError(start)
    return SinglePhaseSolver(grid=grid, cfg=cfg, fields=fields), ref
