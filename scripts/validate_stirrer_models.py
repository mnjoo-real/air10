"""V7-S gate 1: static geometry of the stirrer model candidates (operator-only, no flow).

M0 wall_touching_volume, M1 tapered_volume (ell_z = 1.0 / 1.75 / 3.5 mm = eps_f / D_m/4 / D_m/2),
M2 moving_footprint. On dx = 1 / 0.5 / 0.25 / 0.125 mm against a 20x-finer midpoint quadrature:
  M0/M1: V_chi = int chi dV, I_chi = int chi r^2 dV, T0 = rho Omega I_chi / tau_s (u_theta = 0, 300 rpm),
         chi at the bottom cell centre (max over r), and chi_z(z) sampled at z = 0 (analytic).
  M2:    wall-velocity moments M1_w = int_0^R w(r) r dA (dA = 2 pi r dr) and M3_w = int w r^3 dA-type
         moment int 2 pi w r^3 dr, which set the torque a moving bottom exerts for a given shear scale.
tau_s = 0.005 s fixed (uncalibrated). Prints JSON lines.
"""
from __future__ import annotations

import copy
import dataclasses
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from air_vortex.config import load_config  # noqa: E402
from air_vortex.forcing import bottom_taper, footprint_weight, forcing_mask  # noqa: E402
from air_vortex.grid import build_grid  # noqa: E402

MODELS = (("M0", "wall_touching_volume", 0.0), ("M1_1.0", "tapered_volume", 1.0e-3),
          ("M1_1.75", "tapered_volume", 1.75e-3), ("M1_3.5", "tapered_volume", 3.5e-3),
          ("M2", "moving_footprint", 0.0))


def cfg_for(dx, model, ell):
    cfg = copy.deepcopy(load_config(ROOT / "configs" / "baseline.yaml"))
    cfg.grid.dr_m = cfg.grid.dz_m = dx
    cfg.stirrer.rpm = 300.0
    cfg.stirrer = dataclasses.replace(cfg.stirrer, model=model, bottom_taper_m=ell)
    return cfg


def reference(cfg, h=0.0125e-3):
    R_m, D_m, eps = cfg.geometry.stirbar_half_length_m, cfg.geometry.stirbar_diameter_m, cfg.stirrer.forcing_smoothing_m
    Rv = cfg.geometry.vessel_radius_m
    r = np.arange(h / 2, Rv, h)
    if cfg.stirrer.model == "moving_footprint":
        w = footprint_weight(r, cfg)
        return {"M1_w": float(np.sum(2 * np.pi * w * r * r * h)), "M3_w": float(np.sum(2 * np.pi * w * r**3 * h))}
    z = np.arange(h / 2, D_m + 12 * eps, h)
    cr = 0.5 * (1 - np.tanh((r - R_m) / eps))
    cz = 0.5 * (1 - np.tanh((z - D_m) / eps))
    if cfg.stirrer.model == "tapered_volume":
        cz = cz * bottom_taper(z, cfg.stirrer.bottom_taper_m)
    V = float(np.sum(cr * 2 * np.pi * r * h) * np.sum(cz * h))
    I = float(np.sum(cr * r**2 * 2 * np.pi * r * h) * np.sum(cz * h))
    rho, Om, tau = cfg.fluid.water_density, cfg.stirrer.omega_target, cfg.stirrer.forcing_tau_s
    return {"V_chi": V, "I_chi": I, "T0": rho * Om / tau * I}


def gate(dx, lab, model, ell):
    cfg = cfg_for(dx, model, ell)
    g = build_grid(cfg)
    ref = reference(cfg)
    out = {"model": lab, "stirrer_model": model, "ell_z_mm": ell * 1e3, "dx_mm": dx * 1e3,
           "tau_s": cfg.stirrer.forcing_tau_s, "tau_s_calibrated": False}
    if model == "moving_footprint":
        w = footprint_weight(g.r_c, cfg)
        out["M1_w"] = float(np.sum(2 * np.pi * w * g.r_c**2 * g.dr))
        out["M3_w"] = float(np.sum(2 * np.pi * w * g.r_c**3 * g.dr))
        out["cells_across_edge"] = 2.2 * cfg.stirrer.forcing_smoothing_m / dx
        for k in ("M1_w", "M3_w"):
            out[k + "_rel_err"] = (out[k] - ref[k]) / ref[k]
        return out
    chi = forcing_mask(g, cfg)
    dV = 2 * np.pi * g.r_c[:, None] * g.dr * g.dz * np.ones(g.shape_center)
    r2 = g.r_c[:, None] * np.ones(g.shape_center)
    rho, Om, tau = cfg.fluid.water_density, cfg.stirrer.omega_target, cfg.stirrer.forcing_tau_s
    out["V_chi"] = float(np.sum(chi * dV))
    out["I_chi"] = float(np.sum(chi * r2**2 * dV))
    out["T0"] = float(np.sum(rho * r2 * chi * Om * r2 / tau * dV))
    for k in ("V_chi", "I_chi", "T0"):
        out[k + "_rel_err"] = (out[k] - ref[k]) / ref[k]
    out["chi_bottom_cell_max"] = float(chi[:, 0].max())
    out["chi_z_at_z0_analytic"] = float(0.5 * (1 - np.tanh((0 - cfg.geometry.stirbar_diameter_m) / cfg.stirrer.forcing_smoothing_m))
                                        * (bottom_taper(0.0, ell) if model == "tapered_volume" else 1.0))
    out["cells_across_taper"] = ell / dx if ell else None
    return out


if __name__ == "__main__":
    for lab, model, ell in MODELS:
        for dx in (1.0e-3, 0.5e-3, 0.25e-3, 0.125e-3):
            print(json.dumps(gate(dx, lab, model, ell)), flush=True)
