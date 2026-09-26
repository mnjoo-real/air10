"""V6/V7 production stirrer-forcing convergence (height branch; tau_s FIXED, uncalibrated;
baseline vessel/bar geometry = numerical-validation placeholder, NOT measured).

Usage: python scripts/validate_forcing_convergence.py MODE RPM DX_MM T_END [DT_FACTOR] [SWIRL] [key=value ...]
  SWIRL  advective | conservative (= conservative_upwind1) | conservative_upwind1 | conservative_muscl2
         (omitted -> the config default, physics.height_swirl_advection)
  keys   limiter=vanleer|mc|minmod|none   visc=vector_laplacian|angular_momentum
         merid=advective|conservative_upwind1|conservative_muscl2
         bottom=footprint  (DIAGNOSTIC hybrid: volume forcing + moving bottom under the bar; V7-T only)
         stirrer=wall_touching_volume|tapered_volume|moving_footprint  taper=ELL_Z_MM  (V7-S models M0/M1/M2)
         snap=PATH.npz  (u_theta, u_r, u_z at cell centres + liquid mask at SNAP_TIMES, for
         cross-grid profile and boundary-layer analysis)
  MODE swirl  frozen flat interface, meridional velocity zeroed each step (forcing + swirl viscosity)
       fixed  frozen flat interface (atmospheric-pressure outlet), full meridional momentum
       lid    rigid free-slip lid (no penetration), full meridional momentum
       free   full free surface (production path)
Logs one JSON line every 0.05 s: angular-momentum budget (L_z, T_stir, T_side, T_bottom,
window residual dL - int(sum T) dt), max u_theta, U_mer, u_theta at 4 fixed physical probes (bilinear), depression,
min pressure, volume, max slope, N_m, N_theta, N_curv class. Free mode stops (flag) when N_curv < 8
once the surface is deformed (d > 1 dx), rather than integrating an unresolved collapse.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from air_vortex.height_benchmarks import build_height_production  # noqa: E402
from air_vortex.height_interface import graph_validity  # noqa: E402

PROBES = ((0.010, 0.0035), (0.020, 0.010), (0.005, 0.025), (0.030, 0.045),   # (r, z) in m; V6/V7 set
          (0.005, 0.040), (0.010, 0.030), (0.003, 0.045))                        # V7-T upper-axis probes
SNAP_TIMES = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0)
if "SNAP_TIMES" in __import__("os").environ:      # e.g. SNAP_TIMES=0.1,0.2,0.3 (fine-grid discriminator)
    SNAP_TIMES = tuple(float(x) for x in __import__("os").environ["SNAP_TIMES"].split(","))


def bilinear(g, q, r, z):
    i = np.clip(np.searchsorted(g.r_c, r) - 1, 0, g.Nr - 2)
    j = np.clip(np.searchsorted(g.z_c, z) - 1, 0, g.Nz - 2)
    a = (r - g.r_c[i]) / g.dr; b = (z - g.z_c[j]) / g.dz
    return float((1 - a) * (1 - b) * q[i, j] + a * (1 - b) * q[i + 1, j] + (1 - a) * b * q[i, j + 1] + a * b * q[i + 1, j + 1])


class FootprintBottomBC:
    """DIAGNOSTIC ONLY (V7-T fine-grid discriminator): stationary no-slip side wall, but the
    bottom under the stirrer footprint moves with the forced fluid, u_b = chi_r(r) Omega(t) r,
    chi_r = (1 - tanh((r - R_m)/eps)) / 2. Removes the artificial forcing-wall layer of thickness
    ~sqrt(nu tau_s) between the relaxation-forced fluid and a no-slip bottom. Not a physical
    stirrer model."""

    def __init__(self, solver):
        self.s = solver

    def wall_u_theta(self, grid):
        return np.zeros(grid.Nz)

    def bottom_u_theta(self, grid):
        cfg = self.s.cfg
        R_m, eps = cfg.geometry.stirbar_half_length_m, cfg.stirrer.forcing_smoothing_m
        chi_r = 0.5 * (1.0 - np.tanh((grid.r_c - R_m) / eps))
        return chi_r * cfg.stirrer.omega(self.s.fields.t) * grid.r_c


def main(mode, rpm, dx_mm, T, dt_factor=1.0, swirl_adv=None, limiter=None, visc=None, snap=None, merid=None,
         bottom=None, stirrer=None, taper=None):
    from air_vortex.height_solver import SinglePhaseHeightSolver
    s0 = build_height_production(rpm, dx=dx_mm * 1e-3)
    if swirl_adv == "advective":
        s0.cfg.physics.height_legacy_swirl_for_validation = True
    if stirrer is not None:                      # V7-S model candidates (M0 / M1 / M2)
        import dataclasses
        s0.cfg.stirrer = dataclasses.replace(s0.cfg.stirrer, model=stirrer,
                                             bottom_taper_m=float(taper) * 1e-3 if taper else 0.0)
    s = SinglePhaseHeightSolver(grid=s0.grid, cfg=s0.cfg, fields=s0.fields, eta=s0.eta, z_pin=s0.z_pin,
                                swirl_advection=swirl_adv, swirl_limiter=limiter, swirl_viscous=visc,
                                meridional_advection=merid)
    if bottom == "footprint":
        s.wall_bc = FootprintBottomBC(s)
    s.freeze_interface = mode in ("swirl", "fixed", "lid")
    s.rigid_lid = mode == "lid"
    s.swirl_only = mode == "swirl"
    g, cfg = s.grid, s.cfg
    H = cfg.geometry.water_height_m
    V0 = s.volume
    print(json.dumps({"mode": mode, "rpm": rpm, "dx_mm": dx_mm, "dt_factor": dt_factor, "swirl_advection": s.swirl_advection,
                      "swirl_limiter": s.swirl_limiter, "swirl_viscous": s.swirl_viscous,
                      "meridional_advection": s.meridional_advection,
                      "bottom_bc": bottom or "no_slip", "stirrer_model": s.cfg.stirrer.model,
                      "bottom_taper_mm": s.cfg.stirrer.bottom_taper_m * 1e3,
                      "tau_s": cfg.stirrer.forcing_tau_s, "tau_s_calibrated": False,
                      "geometry": "baseline.yaml placeholder (numerical validation only)",
                      "R": g.r_v, "H": H, "Nr": g.Nr, "Nz": g.Nz}), flush=True)
    nxt, Lprev, tprev = 0.0, None, None
    L0, cumT, cumTs = s.torque_budget()["L_z"], 0.0, 0.0
    snaps, snap_i = {}, 0
    intT = 0.0                                   # int (T_stir + T_side + T_bottom) dt since last log
    b0 = s.torque_budget()
    while s.fields.t < T:
        dt = dt_factor * s.stable_timestep(s.fields.u_r, s.fields.u_z)
        d = s.step(dt)
        b = s.torque_budget()
        Tsum = b["T_stir"] + b["T_side"] + b["T_bottom"]
        inc = 0.5 * dt * (Tsum + (b0["T_stir"] + b0["T_side"] + b0["T_bottom"]))
        intT += inc
        cumT += inc
        cumTs += 0.5 * dt * (b["T_stir"] + b0["T_stir"])
        b0 = b
        if snap and snap_i < len(SNAP_TIMES) and d.t >= SNAP_TIMES[snap_i] - 1e-9:
            from air_vortex.operators import interp_ur_to_center, interp_uz_to_center
            f = s.fields
            k = f"t{SNAP_TIMES[snap_i]:.1f}"
            snaps[k + "_ut"] = f.u_theta.copy()
            snaps[k + "_ur"] = interp_ur_to_center(f.u_r)
            snaps[k + "_uz"] = interp_uz_to_center(f.u_z)
            snaps[k + "_liq"] = s.geometry.liquid.copy()
            snaps[k + "_eta"] = s.eta.copy()
            snaps[k + "_time"] = np.array(d.t)
            np.savez_compressed(snap, r_c=g.r_c, z_c=g.z_c, **snaps)
            snap_i += 1
        if d.t < nxt:
            continue
        v = graph_validity(g, s.interface())
        liq = s.geometry.liquid
        f = s.fields
        ut = np.where(liq, f.u_theta, 0.0)
        Ncurv = min(v["min_N_m"], v["min_N_theta"])
        depr = float(H - s.eta[0])
        row = {"t": d.t, "dt": dt, **b,
               "dLdt": None if Lprev is None else (b["L_z"] - Lprev) / (d.t - tprev),
               "u_theta_max": float(np.abs(ut).max()),
               "U_mer": max(d.max_abs_ur_liquid, d.max_abs_uz_liquid),
               "probes": [bilinear(g, ut, r, z) for r, z in PROBES],
               "depression_mm": depr * 1e3, "eta_range_mm": float((s.eta.max() - s.eta.min()) * 1e3),
               "p_min": float(np.where(liq, f.p, np.inf).min()), "dV": s.volume / V0 - 1,
               "max_slope": v["max_slope"], "N_m": v["min_N_m"], "N_theta": v["min_N_theta"],
               "N_curv_class": "unresolved" if Ncurv < 8 else ("marginal" if Ncurv < 12 else "resolved")}
        row["cum_residual_rel"] = (b["L_z"] - L0 - cumT) / max(abs(cumTs), 1e-30)
        if Lprev is not None:
            dL = b["L_z"] - Lprev
            row["window_dL"] = dL
            row["window_intT"] = intT
            row["balance_residual_rel"] = (dL - intT) / max(abs(intT), abs(dL), 1e-30)
        intT = 0.0
        if mode == "lid":
            row["lid_flux_rel"] = float(getattr(s, "lid_flux", 0.0) / (np.pi * g.r_v**2 * max(row["U_mer"], 1e-30)))
        print(json.dumps(row), flush=True)
        Lprev, tprev = b["L_z"], d.t
        nxt += 0.05
        if mode == "free" and Ncurv < 8 and abs(depr) > g.dz:
            print(json.dumps({"t": d.t, "stop": "N_curv < 8: interface curvature unresolved", "N_curv": Ncurv}),
                  flush=True)
            return


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if "=" not in x]
    kw = dict(x.split("=", 1) for x in sys.argv[1:] if "=" in x)
    main(a[0], float(a[1]), float(a[2]), float(a[3]), float(a[4]) if len(a) > 4 else 1.0,
         a[5] if len(a) > 5 else None, kw.get("limiter"), kw.get("visc"), kw.get("snap"),
         kw.get("merid"), kw.get("bottom"), kw.get("stirrer"), kw.get("taper"))
