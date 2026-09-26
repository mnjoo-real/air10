"""V6/V7 step 1: is the production stirrer-forcing OPERATOR grid-independent?

forcing.forcing_mask: chi(r, z) = 1/2 (1 - tanh((r - R_m)/eps)) * 1/2 (1 - tanh((z - D_m)/eps)),
evaluated at cell centres (a smooth kernel with PHYSICAL width, not a cell mask).
For the fixed baseline geometry (placeholder numerical-validation values) and FIXED tau_s:
  V_chi = int chi dV,  I_chi = int chi r^2 dV,  (dV = 2 pi r dr dz)
  T0 = int rho r f_theta dV with u_theta = 0  (= rho Omega int chi r^2 dV / tau_s)
  P  = int rho u_theta f_theta dV for u_theta = alpha r (alpha = Omega/2) and a smooth
       non-solid-body profile
on dx = 1.0 / 0.5 / 0.25 / 0.125 mm, against a 20x-finer midpoint reference.
Also the forcing-only single-step test: Delta u_theta vs dt chi (Omega r - u)/tau_s
(the predictor term in isolation) and the integrated angular-momentum change.
Prints JSON lines.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from air_vortex.config import load_config  # noqa: E402
from air_vortex.forcing import forcing_mask, stirrer_forcing  # noqa: E402
from air_vortex.grid import build_grid  # noqa: E402

RPM = 300.0


def cfg_for(dx):
    cfg = copy.deepcopy(load_config(ROOT / "configs" / "baseline.yaml"))
    cfg.grid.dr_m = cfg.grid.dz_m = dx
    cfg.stirrer.rpm = RPM
    return cfg


def moments(dx):
    cfg = cfg_for(dx)
    g = build_grid(cfg)
    rho = cfg.fluid.water_density
    chi = forcing_mask(g, cfg)
    dV = 2 * np.pi * g.r_c[:, None] * g.dr * g.dz * np.ones(g.shape_center)
    Om = cfg.stirrer.omega_target
    r2 = g.r_c[:, None] * np.ones(g.shape_center)
    z2 = g.z_c[None, :] * np.ones(g.shape_center)
    out = {"dx_mm": dx * 1e3, "tau_s": cfg.stirrer.forcing_tau_s, "Omega": Om,
           "R_m": cfg.geometry.stirbar_half_length_m, "D_m": cfg.geometry.stirbar_diameter_m,
           "eps": cfg.stirrer.forcing_smoothing_m, "R_vessel": g.r_v, "Nr": g.Nr, "Nz": g.Nz}
    out["V_chi"] = float(np.sum(chi * dV))
    out["I_chi"] = float(np.sum(chi * r2**2 * dV))
    f0 = stirrer_forcing(np.zeros(g.shape_center), Om, g, cfg, chi)
    out["T0"] = float(np.sum(rho * r2 * f0 * dV))
    for lab, ut in (("solid_half", 0.5 * Om * r2),
                    ("smooth", 0.3 * Om * r2 * np.exp(-((r2 - 0.01) / 0.008) ** 2) * (1 + z2 / 0.02))):
        f = stirrer_forcing(ut, Om, g, cfg, chi)
        out[f"T_{lab}"] = float(np.sum(rho * r2 * f * dV))
        out[f"P_{lab}"] = float(np.sum(rho * ut * f * dV))
    # cells across the forcing edges (10-90 % width of the tanh = 2.2 eps) and the forcing layer
    out["cells_across_edge"] = 2.2 * cfg.stirrer.forcing_smoothing_m / dx
    out["cells_across_layer_height"] = cfg.geometry.stirbar_diameter_m / dx
    return out


def reference(dx_fine=0.0125e-3):
    """20x-finer midpoint quadrature over the region where chi is non-negligible."""
    cfg = cfg_for(1e-3)
    R_m, D_m, eps = (cfg.geometry.stirbar_half_length_m, cfg.geometry.stirbar_diameter_m,
                     cfg.stirrer.forcing_smoothing_m)
    rho, Om, tau = cfg.fluid.water_density, cfg.stirrer.omega_target, cfg.stirrer.forcing_tau_s
    Rv = cfg.geometry.vessel_radius_m
    r = np.arange(dx_fine / 2, Rv, dx_fine)
    z = np.arange(dx_fine / 2, D_m + 12 * eps, dx_fine)
    cr = 0.5 * (1 - np.tanh((r - R_m) / eps)); cz = 0.5 * (1 - np.tanh((z - D_m) / eps))
    wr = 2 * np.pi * r * dx_fine; wz = np.full_like(z, dx_fine)
    V = float((cr * wr).sum() * (cz * wz).sum())
    I = float((cr * r**2 * wr).sum() * (cz * wz).sum())
    return {"V_chi": V, "I_chi": I, "T0": rho * Om / tau * I}


def forcing_step(dx, dt=1e-4):
    """Forcing term in isolation (the predictor adds dt * f to u_theta)."""
    cfg = cfg_for(dx)
    g = build_grid(cfg)
    chi = forcing_mask(g, cfg)
    Om = cfg.stirrer.omega_target
    r2 = g.r_c[:, None] * np.ones(g.shape_center)
    dV = 2 * np.pi * r2 * g.dr * g.dz
    rows = {}
    for lab, ut in (("zero", 0 * r2), ("solid_alpha", 0.4 * Om * r2),
                    ("smooth", 0.3 * Om * r2 * np.exp(-((r2 - 0.01) / 0.008) ** 2))):
        du = dt * stirrer_forcing(ut, Om, g, cfg, chi)
        expect = dt * chi * (Om * r2 - ut) / cfg.stirrer.forcing_tau_s
        dL = float(np.sum(cfg.fluid.water_density * r2 * du * dV))
        rows[lab] = {"max_abs_diff": float(np.abs(du - expect).max()), "dLz": dL}
    return rows


if __name__ == "__main__":
    ref = reference()
    print(json.dumps({"reference": ref}), flush=True)
    for dx in (1.0e-3, 0.5e-3, 0.25e-3, 0.125e-3):
        m = moments(dx)
        for k in ("V_chi", "I_chi", "T0"):
            m[f"{k}_rel_err"] = (m[k] - ref[k]) / ref[k]
        m["forcing_step"] = forcing_step(dx)
        print(json.dumps(m), flush=True)
