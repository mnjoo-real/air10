"""DIAGNOSTIC ONLY (V4b-P, Phase 4): manufactured interface kinematics, no
pressure, no surface tension.

Interface: paraboloid eta(r) = z0 + A r^2 with wall slope eta'(R) = cot(theta)
(theta = 90: flat), sub-cell phase xi of eta(R); phi = exact signed distance
(dense sampling, continued past the wall).
Velocity: discretely divergence-free MAC field from corner stream function
psi = U0 r^2 (R^2 - r^2)^2 f(z), f(z) = z (1 + z/Z); zero at the wall, so the
contact point does not move. Exact kinematics at column r_i:
  eta_t = u_z(r_i, eta_i) - u_r(r_i, eta_i) eta'(r_i)   (analytic field)
Measured rates (one step, heights via cubic root unless noted):
  LS          production: extension of the liquid faces, MUSCL2-MC + SSPRK2, dt_small
  LS_dtprod   same with the production capillary dt (time-integration error)
  LS_full     exact face velocities everywhere (no extension)       -> extension effect
  LS_up1      extension, first-order upwind + Euler                  -> reconstruction effect
  LS_lin      production, heights by linear crossing                 -> extraction effect
  Q / Qv      mixed-face flux / ring area (radial faces owned by liquid / void side)
Errors: ||rate - exact|| / ||exact|| over all columns and over the last 8.

Usage: python scripts/diag_kinematics_manufactured.py > out.jsonl
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

from air_vortex.diagnostics import free_surface_height  # noqa: E402
from air_vortex.levelset import advect_level_set_advective  # noqa: E402
from air_vortex.liquid_mask import classify  # noqa: E402
from air_vortex.pinned_phase import R_V, build_phase_solver  # noqa: E402
from air_vortex.velocity_extension import extend_velocity  # noqa: E402
from air_vortex.wall_curvature import column_heights  # noqa: E402
from diag_kinematic_consistency import col_flux  # noqa: E402

U0 = 1.0


def setup(theta, xi, dx):
    s, _ = build_phase_solver(xi, dx, "reconstruct_ghost", "cubic", theta_ref=60.0, ic="extended")
    g = s.grid
    A = (1.0 / np.tan(np.deg2rad(theta))) / (2 * R_V)
    zw = s.cfg.wall.pinned_contact_height_m                   # a height at phase xi (theta-60 geometry)
    z0 = zw - A * R_V**2
    eta = lambda r: z0 + A * np.asarray(r) ** 2
    rs = np.linspace(-R_V - 4 * g.dr, R_V + 4 * g.dr, 60001)
    zs = eta(rs)
    phi = np.empty(g.shape_center)
    for i in range(g.Nr):
        d2 = (g.r_c[i] - rs[None, :]) ** 2 + (g.z_c[:, None] - zs[None, :]) ** 2
        phi[i] = np.sign(g.z_c - eta(g.r_c[i])) * np.sqrt(d2.min(axis=1))
    Z = g.z_f[-1]
    fz = lambda z: z * (1 + z / Z)
    dfz = lambda z: 1 + 2 * z / Z
    psi = U0 * (g.r_f**2 * (R_V**2 - g.r_f**2) ** 2)[:, None] * fz(g.z_f)[None, :]
    ur = -(psi[:, 1:] - psi[:, :-1]) / (np.where(g.r_f == 0, 1, g.r_f)[:, None] * g.dz); ur[0] = 0
    uz = (psi[1:, :] - psi[:-1, :]) / (g.r_c[:, None] * g.dr)
    scale = 1e-3 / max(np.abs(ur).max(), np.abs(uz).max())
    ur, uz = ur * scale, uz * scale
    r, e = g.r_c, eta(g.r_c)
    ur_a = -U0 * scale * r * (R_V**2 - r**2) ** 2 * dfz(e)
    uz_a = U0 * scale * (2 * (R_V**2 - r**2) ** 2 - 4 * r**2 * (R_V**2 - r**2)) * fz(e)
    exact = uz_a - ur_a * 2 * A * r
    return s, g, phi, ur, uz, exact


def rates(s, g, phi, ur, uz):
    geom = classify(phi)
    ls, nl = s.cfg.levelset, s.cfg.physics.extension_layers_capillary
    ut = np.zeros(g.shape_center)
    liq_r, liq_z = geom.ur_face_known(), geom.uz_face_known()
    cols = np.arange(g.Nr)
    h0 = column_heights(g, phi, cols)
    h0l = free_surface_height(phi, g)
    dt_prod = s.stable_timestep(ur * 0, uz * 0)
    out = {}

    def adv(a, b, scheme, integ, dt, ext=True):
        if ext:
            a, b, _, _ = extend_velocity(g, geom, phi, np.where(liq_r, a, 0.0), np.where(liq_z, b, 0.0), ut, nl)
        return advect_level_set_advective(phi, a, b, g, dt, scheme=scheme, time_integrator=integ,
                                          limiter=ls.limiter)
    dts = 1e-6
    ph = adv(ur, uz, ls.advection_scheme, ls.time_integrator, dts)
    out["LS"] = (column_heights(g, ph, cols) - h0) / dts
    out["LS_lin"] = (free_surface_height(ph, g) - h0l) / dts
    ph = adv(ur, uz, ls.advection_scheme, ls.time_integrator, dt_prod)
    out["LS_dtprod"] = (column_heights(g, ph, cols) - h0) / dt_prod
    ph = adv(ur, uz, ls.advection_scheme, ls.time_integrator, dts, ext=False)
    out["LS_full"] = (column_heights(g, ph, cols) - h0) / dts
    ph = adv(ur, uz, "upwind1", "euler", dts)
    out["LS_up1"] = (column_heights(g, ph, cols) - h0) / dts
    w = 2 * np.pi * g.r_c * g.dr
    out["Q"] = col_flux(g, geom, ur, uz, "liquid") / w
    out["Qv"] = col_flux(g, geom, ur, uz, "void") / w
    return out


def main():
    for dx_mm in (0.5, 0.25, 0.125):
        for theta in (90.0, 75.0, 60.0, 50.0):
            for xi in (0.05, 0.25, 0.45, 0.65, 0.85):
                s, g, phi, ur, uz, exact = setup(theta, xi, dx_mm * 1e-3)
                rr = rates(s, g, phi, ur, uz)
                n = np.linalg.norm(exact); n8 = np.linalg.norm(exact[-8:])
                row = {"dx_mm": dx_mm, "theta": theta, "xi": xi}
                for k, v in rr.items():
                    row[f"{k}_err"] = float(np.linalg.norm(v - exact) / n)
                    row[f"{k}_err8"] = float(np.linalg.norm((v - exact)[-8:]) / n8)
                row["LS_vs_Q"] = float(np.linalg.norm(rr["LS"] - rr["Q"]) / np.linalg.norm(rr["LS"]))
                row["LS_vs_Q8"] = float(np.linalg.norm((rr["LS"] - rr["Q"])[-8:]) / np.linalg.norm(rr["LS"][-8:]))
                print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
