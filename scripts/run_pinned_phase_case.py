"""One dynamic grid-phase case for the pinned contact line (Gate V4b-P);
prints a single JSON line. Driven in parallel by
scripts/validate_pinned_phase.py.

Usage: python scripts/run_pinned_phase_case.py METHOD FIT NCOLS XI DX_MM T_END [skip] [ic=extended]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from air_vortex.contact_angle import initial_pin_height  # noqa: E402
from air_vortex.curvature_single_phase import crossing_positions  # noqa: E402
from air_vortex.diagnostics import free_surface_height  # noqa: E402
from air_vortex.liquid_mask import classify, liquid_volume_subcell  # noqa: E402
from air_vortex.meniscus import G, RHO, SIGMA  # noqa: E402
from air_vortex.pinned_phase import build_phase_solver, phase_of  # noqa: E402

CHECK = (0.1, 0.3, 1.0, 3.0, 5.0)


def main():
    method, fit, ncols, xi, dx_mm, t_end = sys.argv[1:7]
    extra = sys.argv[7:]
    skip = "skip" in extra
    ic = next((a.split("=", 1)[1] for a in extra if a.startswith("ic=")), "truncated")
    ncols, xi, dx, t_end = int(ncols), float(xi), float(dx_mm) * 1e-3, float(t_end)
    s, ref = build_phase_solver(xi, dx, method, fit, ic=ic)
    s.cfg.wall.pinned_fit_columns = ncols
    s.cfg.wall.pinned_skip_wall_column = skip
    g = s.grid
    z_pin = s.cfg.wall.pinned_contact_height_m
    V0 = liquid_volume_subcell(g, s.fields.phi)
    cell_vol = 2 * np.pi * g.r_c[:, None] * g.dr * g.dz

    def pressure_residual():
        geom = classify(s.fields.phi)
        pr, pz, _, _ = s.interface_pressure_bc(geom, s.fields.phi)
        _, zr, _, zz = crossing_positions(g, geom)
        pg = np.r_[pr[np.isfinite(pr)], pz[np.isfinite(pz)]]
        zx = np.r_[zr[np.isfinite(pr)], zz[np.isfinite(pz)]]
        return float(np.sqrt(np.mean((pg - (ref.P0 - RHO * G * zx)) ** 2)) / ref.P0)

    out = {"method": method, "fit": fit, "ncols": ncols, "skip_wall_column": skip, "ic": ic, "xi": xi, "xi_actual": phase_of(z_pin, dx),
           "dx_mm": dx * 1e3, "theta_eq": ref.theta_deg, "p_resid_t0": pressure_residual()}
    win_max, next_i, pz_max, t_prev, div_max = 0.0, 0, 0.0, 0.0, 0.0
    windows = {}
    while s.fields.t < t_end - 1e-12:
        d = s.step()
        u = max(d.max_abs_ur_liquid, d.max_abs_uz_liquid)
        # window [0.9 t_c, t_c] maxima at the checkpoints
        for tc in CHECK:
            if 0.9 * tc < d.t <= tc + 0.5 * d.dt:
                windows[tc] = max(windows.get(tc, 0.0), u)
        liq = s.fields.phi < 0
        pz_lin = float(np.sum(RHO * 0.5 * (s.fields.u_z[:, 1:] + s.fields.u_z[:, :-1]) * cell_vol * liq))
        pz_max = max(pz_max, abs(pz_lin))
        div_max = max(div_max, d.max_divergence_liquid)
        if not np.isfinite(u) or u > 5.0:
            out["blowup_t"] = d.t
            break
    eta = free_surface_height(s.fields.phi, g)
    e = eta - ref.eta(g.r_c)
    wall = g.r_c > g.r_v - 4 * dx
    out.update({
        "T": s.fields.t, **{f"U_{tc}": windows.get(tc) for tc in CHECK},
        "rmse_dx": float(np.sqrt(np.nanmean(e**2)) / dx),
        "wall_rmse_dx": float(np.sqrt(np.nanmean(e[wall] ** 2)) / dx),
        "contact_err_dx": (initial_pin_height(g, s.fields.phi, ncols) - z_pin) / dx,
        "volume_drift": liquid_volume_subcell(g, s.fields.phi) / V0 - 1,
        "p_resid_end": pressure_residual(), "max_liquid_Pz": pz_max, "max_div": div_max,
        "theta_eq_final_deg": float(np.rad2deg(np.arctan2(1.0, np.polyfit(
            (g.r_c[-3:] - g.r_v) / dx, eta[-3:], 2)[1] / dx))),
    })
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
