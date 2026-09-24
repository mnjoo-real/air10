"""Gate V5b/V5c: interface-preserving reinitialization (single_phase_ls).

Writes results/validation_reinit/metadata.json and summary.md with

  A  static preservation R1-R4 x {legacy, RS order 1, RS order 2} x grids x
     {1, 10, 50, 100} repeated calls (volume correction never involved)
  B  grid convergence of the contour displacement (order only where monotone)
  S  pseudo-time sensitivity: cfl {0.1,0.2,0.3,0.5} x iterations {1,3,5,10}
  L  narrow band vs whole domain
  D1 zero-velocity operational loop (advect -> reinit), 200 steps
  D2 prescribed vertical translation of a circle: advection only vs + legacy
     reinit vs + RS reinit, error against the exact translated circle
  V3 operational rigid-body equilibrium (Omega=20, 1.0 s) with RS reinit ON,
     compared to reinit OFF and legacy reinit

Usage:  python scripts/validate_reinitialization.py [--quick] [--skip-v3]
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from air_vortex.config import load_config  # noqa: E402
from air_vortex.diagnostics import free_surface_height, volume_consistent_parabola  # noqa: E402
from air_vortex.grid import build_grid  # noqa: E402
from air_vortex.levelset import advect_level_set_advective, reinitialize_level_set  # noqa: E402
from air_vortex.liquid_mask import liquid_volume_subcell  # noqa: E402
from air_vortex.reinit_benchmarks import CASES, r3_circle, uniform_grid  # noqa: E402
from air_vortex.reinit_diagnostics import contour_displacement, signed_distance_error  # noqa: E402
from air_vortex.reinit_subcell import reinitialize_subcell  # noqa: E402
from air_vortex.single_phase_solver import (  # noqa: E402
    SinglePhaseSolver, WallBC, initialize_rigid_body_single_phase)

ITER = 5          # iterations per static call (sensitivity section varies it)
CALLS = (1, 10, 50, 100)


def methods(grid, iters=ITER, cfl=0.45, band=0.0):
    return {
        "legacy_godunov": lambda p: reinitialize_level_set(p, grid, iters),
        "rs_order1": lambda p: reinitialize_subcell(p, grid, iters, cfl=cfl, order=1, band_cells=band),
        "rs_order2": lambda p: reinitialize_subcell(p, grid, iters, cfl=cfl, order=2, band_cells=band),
    }


def static_series(grid, case, fn, calls=CALLS):
    phi = case.phi0.copy()
    V0 = liquid_volume_subcell(grid, case.phi0)
    rows = []
    for n in range(1, max(calls) + 1):
        phi = fn(phi)
        if n in calls:
            cd = contour_displacement(grid, case.phi0, phi)
            gmax, grms = case.geometry_error(phi)
            rows.append({"calls": n, "max_shift_dx": cd.max_shift_dx, "rms_shift_dx": cd.rms_shift_dx,
                         "mean_signed_z_dx": cd.mean_signed_z / cd.dx,
                         "mean_signed_r_dx": cd.mean_signed_r / cd.dx,
                         "max_shift_along_line_dx": cd.max_shift_along_line / cd.dx,
                         "n_lost": cd.n_lost, "geo_err_max_dx": gmax / cd.dx,
                         "geo_err_rms_dx": grms / cd.dx,
                         "volume_change": (liquid_volume_subcell(grid, phi) - V0) / V0,
                         "E_sd": signed_distance_error(grid, phi)})
    return rows


def section_a(grids):
    out = []
    for key, build in CASES.items():
        for dx in grids:
            g = uniform_grid(dx)
            case = build(g)
            e_sd0 = signed_distance_error(g, case.phi0)
            geo0 = case.geometry_error(case.phi0)[0] / dx
            for name, fn in methods(g).items():
                t0 = time.perf_counter()
                rows = static_series(g, case, fn)
                out.append({"case": key, "dx_mm": dx * 1e3, "method": name, "E_sd_before": e_sd0,
                            "geo_err_phi0_dx": geo0, "series": rows,
                            "wall_s": time.perf_counter() - t0})
                print(f"  A {key} dx={dx*1e3:.2f} {name}: 1-call {rows[0]['max_shift_dx']:.2e}dx "
                      f"100-call {rows[-1]['max_shift_dx']:.2e}dx", flush=True)
    return out


def _order(dxs, errs):
    errs = np.asarray(errs)
    if len(errs) < 2 or np.any(errs <= 0) or np.any(np.diff(errs) >= 0):
        return None   # not monotone: no order reported
    return float(np.polyfit(np.log(dxs), np.log(errs), 1)[0])


def section_b(a_rows):
    """Order of the absolute (metres) normal displacement vs dx."""
    out = []
    for key in CASES:
        for calls_idx, calls in ((0, 1), (3, 100)):
            rec = {"case": key, "calls": calls}
            for m in ("legacy_godunov", "rs_order2", "rs_order1"):
                rows = [r for r in a_rows if r["case"] == key and r["method"] == m]
                dxs = [r["dx_mm"] for r in rows]
                abs_shift = [r["series"][calls_idx]["max_shift_dx"] * r["dx_mm"] for r in rows]
                abs_geo = [r["series"][calls_idx]["geo_err_max_dx"] * r["dx_mm"] for r in rows]
                rec[m] = {"dx_mm": dxs, "max_shift_mm": abs_shift, "order_shift": _order(dxs, abs_shift),
                          "geo_err_mm": abs_geo, "order_geo": _order(dxs, abs_geo)}
            out.append(rec)
    return out


def section_s(dx=1.0e-3):
    out = []
    g = uniform_grid(dx)
    for key in ("R3", "R4"):
        case = CASES[key](g)
        for cfl in (0.1, 0.2, 0.3, 0.5):
            for iters in (1, 3, 5, 10):
                fn = lambda p: reinitialize_subcell(p, g, iters, cfl=cfl, order=2)  # noqa: E731
                rows = static_series(g, case, fn, calls=(1, 100))
                out.append({"case": key, "cfl": cfl, "iters": iters,
                            "shift_1": rows[0]["max_shift_dx"], "shift_100": rows[1]["max_shift_dx"],
                            "E_sd_1": rows[0]["E_sd"], "E_sd_100": rows[1]["E_sd"]})
    return out


def section_l(dx=1.0e-3):
    out = []
    g = uniform_grid(dx)
    for key in ("R3", "R4"):
        case = CASES[key](g)
        for band in (0.0, 5.0):
            t0 = time.perf_counter()
            rows = static_series(g, case, methods(g, band=band)["rs_order2"], calls=(1, 100))
            out.append({"case": key, "band_cells": band, "shift_1": rows[0]["max_shift_dx"],
                        "shift_100": rows[1]["max_shift_dx"], "E_sd_100": rows[1]["E_sd"],
                        "wall_s": time.perf_counter() - t0})
    return out


def section_d1(dx=1.0e-3, steps=200):
    out = []
    g = uniform_grid(dx)
    zr, zz = np.zeros(g.shape_ur), np.zeros(g.shape_uz)
    for key in ("R3", "R4"):
        case = CASES[key](g)
        for name, fn in methods(g, iters=2).items():
            phi = case.phi0.copy()
            for _ in range(steps):
                phi = advect_level_set_advective(phi, zr, zz, g, 1e-4)
                phi = fn(phi)
            cd = contour_displacement(g, case.phi0, phi)
            out.append({"case": key, "method": name, "steps": steps,
                        "max_shift_dx": cd.max_shift_dx, "geo_err_max_dx": case.geometry_error(phi)[0] / dx,
                        "volume_change": (liquid_volume_subcell(g, phi) - liquid_volume_subcell(g, case.phi0))
                        / liquid_volume_subcell(g, case.phi0)})
    return out


def section_d2(dxs, U=0.05, T=0.1, reinit_every=5, iters=2):
    """Circle (torus) of radius 6.23 mm translated upward at U; exact
    solution is the translated circle. Signed-distance initial phi."""
    out = []
    for dx in dxs:
        g = uniform_grid(dx)
        r0, z0, rad = 0.0121, 0.0121, 0.00623
        R, Z = np.meshgrid(g.r_c, g.z_c, indexing="ij")
        phi_init = np.hypot(R - r0, Z - z0) - rad
        u_r = np.zeros(g.shape_ur)
        u_z = np.full(g.shape_uz, U)
        u_z[:, 0] = 0.0
        dt = 0.2 * dx / U
        nsteps = int(round(T / dt))
        rec = {"dx_mm": dx * 1e3, "steps": nsteps}
        for label, fn in (("advection_only", None),
                          ("legacy_reinit", lambda p: reinitialize_level_set(p, g, iters)),
                          ("rs2_reinit", lambda p: reinitialize_subcell(p, g, iters, order=2))):
            phi = phi_init.copy()
            for k in range(nsteps):
                phi = advect_level_set_advective(phi, u_r, u_z, g, dt, "muscl2", "ssprk2", "mc")
                if fn is not None and (k + 1) % reinit_every == 0:
                    phi = fn(phi)
            zc = z0 + U * nsteps * dt
            exact = r3_circle(g, r0=r0, z0=zc, rad=rad)
            gmax, grms = exact.geometry_error(phi)
            V = liquid_volume_subcell(g, phi)
            V0 = liquid_volume_subcell(g, phi_init)
            rec[label] = {"geo_err_max_dx": gmax / dx, "geo_err_rms_dx": grms / dx,
                          "volume_change": (V - V0) / V0}
        out.append(rec)
        print(f"  D2 dx={dx*1e3}: " + ", ".join(f"{k}={v['geo_err_max_dx']:.3f}dx"
                                               for k, v in rec.items() if isinstance(v, dict)), flush=True)
    return out


def run_v3(dx, method, every, iters, t_end, omega=20.0, vc=False, trigger="periodic",
           order=2, q_threshold=0.05):
    base = load_config(ROOT / "configs" / "single_phase_rigid_body.yaml")
    cfg = copy.deepcopy(base)
    cfg.grid.dr_m = cfg.grid.dz_m = dx
    ls = cfg.levelset
    ls.volume_correction.enabled = vc
    ls.reinitialize_every = every
    ls.reinitialize_iterations = iters
    ls.reinitialization_method = method
    ls.reinitialization_order = order
    ls.reinitialization_trigger = trigger
    ls.reinitialization_quality_threshold = q_threshold
    grid = build_grid(cfg)
    fields = initialize_rigid_body_single_phase(grid, cfg, omega, phi_form="signed_distance")
    s = SinglePhaseSolver(grid=grid, cfg=cfg, fields=fields, wall_bc=WallBC.rotating(omega))
    g_acc = cfg.fluid.gravity
    eta_an = volume_consistent_parabola(grid.r_c, grid.r_v, omega, g_acc,
                                        np.pi * grid.r_v**2 * cfg.geometry.water_height_m)
    amp = omega**2 * grid.r_v**2 / (2 * g_acc)
    dep = cfg.geometry.water_height_m - eta_an[0]
    V0 = liquid_volume_subcell(grid, s.fields.phi)
    phi_start = s.fields.phi.copy()
    hist, mx_ur, mx_uz = [], 0.0, 0.0
    t0 = time.perf_counter()
    while s.fields.t < t_end:
        d = s.step()
        mx_ur = max(mx_ur, d.max_abs_ur_liquid)
        mx_uz = max(mx_uz, d.max_abs_uz_liquid)
        hist.append(max(d.max_abs_ur_liquid, d.max_abs_uz_liquid))
    eta = free_surface_height(s.fields.phi, grid)
    n = len(hist)
    return {"dx_mm": dx * 1e3, "method": method if every > 0 else "off", "every": every, "iters": iters,
            "trigger": trigger, "volume_correction": vc,
            "eta_nrmse": float(np.sqrt(np.nanmean((eta - eta_an) ** 2)) / amp),
            "center_err_rel": float(abs(eta[0] - eta_an[0]) / dep),
            "max_ur": mx_ur, "max_uz": mx_uz,
            "late_U_mer": float(max(hist[3 * n // 4:])), "peak_U_mer": float(max(hist)),
            "volume_drift": (liquid_volume_subcell(grid, s.fields.phi) - V0) / V0,
            "n_reinit": d.n_reinit_calls,
            "cum_reinit_shift_dx": d.cumulative_reinit_contour_shift / dx,
            "net_contour_shift_dx": contour_displacement(grid, phi_start, s.fields.phi).max_shift_dx,
            "wall_s": time.perf_counter() - t0}


def section_v3(grids, t_end):
    out = []
    for dx in grids:
        for method, every, iters in (("off", 0, 2), ("russo_smereka_subcell", 5, 2),
                                     ("legacy_godunov", 5, 2)):
            r = run_v3(dx, "legacy_godunov" if method == "off" else method, every, iters, t_end)
            out.append(r)
            print(f"  V3 dx={dx*1e3} {r['method']}: nrmse={r['eta_nrmse']:.2e} center={r['center_err_rel']:.2e} "
                  f"late={r['late_U_mer']:.2e} dV={r['volume_drift']:.2e} n={r['n_reinit']}", flush=True)
    dx = 1.0e-3
    extras = [dict(method="russo_smereka_subcell", every=20, iters=5),
              dict(method="russo_smereka_subcell", every=5, iters=2, trigger="quality", q_threshold=0.01),
              dict(method="russo_smereka_subcell", every=5, iters=2, order=1),
              dict(method="russo_smereka_subcell", every=5, iters=2, vc=True)]
    for kw in extras:
        r = run_v3(dx, t_end=t_end, **kw)
        r["variant"] = {k: v for k, v in kw.items() if k != "method"}
        out.append(r)
        print(f"  V3 extra {r['variant']}: nrmse={r['eta_nrmse']:.2e} late={r['late_U_mer']:.2e} "
              f"n={r['n_reinit']}", flush=True)
    return out


def _fmt(v):
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.2e}"
    return str(v)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--skip-v3", action="store_true")
    ap.add_argument("--t-end", type=float, default=1.0)
    ap.add_argument("--results-root", default=str(ROOT / "results"))
    args = ap.parse_args()
    grids = (1.5e-3, 1.0e-3) if args.quick else (1.5e-3, 1.0e-3, 0.75e-3, 0.5e-3)
    v3_grids = (1.5e-3,) if args.quick else (1.5e-3, 1.0e-3, 0.75e-3)
    out_dir = Path(args.results_root) / "validation_reinit"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("A static", flush=True)
    a = section_a(grids)
    b = section_b(a)
    print("S sensitivity", flush=True)
    s = section_s()
    print("L band", flush=True)
    l_ = section_l()
    print("D1", flush=True)
    d1 = section_d1()
    print("D2", flush=True)
    d2 = section_d2(grids[:3])
    v3 = [] if args.skip_v3 else section_v3(v3_grids, 0.2 if args.quick else args.t_end)
    meta = {"kind": "reinit_validation", "A": a, "B": b, "S": s, "L": l_, "D1": d1, "D2": d2, "V3": v3}
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    md = ["# Reinitialization validation (V5b / V5c)\n\n",
          "Shifts are NORMAL displacements of phi=0 relative to phi0's own linear crossings; "
          "geo = distance of the crossings from the EXACT interface. All in units of dx.\n\n",
          "## A static preservation\n\n| geometry | method | dx mm | 1-call shift | 10 | 50 | 100-call shift | "
          "100-call RMS | geo err(phi0) | geo err 100 | volume change 100 | E_sd before | E_sd after 100 |\n",
          "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"]
    for r in a:
        se = r["series"]
        md.append(f"| {r['case']} | {r['method']} | {r['dx_mm']:.2f} | {se[0]['max_shift_dx']:.2e} | "
                  f"{se[1]['max_shift_dx']:.2e} | {se[2]['max_shift_dx']:.2e} | {se[3]['max_shift_dx']:.2e} | "
                  f"{se[3]['rms_shift_dx']:.2e} | {r['geo_err_phi0_dx']:.2e} | {se[3]['geo_err_max_dx']:.2e} | "
                  f"{se[3]['volume_change']:+.2e} | {r['E_sd_before']:.2e} | {se[3]['E_sd']:.2e} |\n")
    md.append("\n## B convergence of |shift| in mm (order only when monotone)\n\n"
              "| case | calls | legacy mm | order | RS2 mm | order | RS2 geo mm | order |\n|---|---:|---|---:|---|---:|---|---:|\n")
    for r in b:
        lg, rs = r["legacy_godunov"], r["rs_order2"]
        md.append(f"| {r['case']} | {r['calls']} | {', '.join(f'{x:.1e}' for x in lg['max_shift_mm'])} | "
                  f"{_fmt(lg['order_shift'])} | {', '.join(f'{x:.1e}' for x in rs['max_shift_mm'])} | "
                  f"{_fmt(rs['order_shift'])} | {', '.join(f'{x:.1e}' for x in rs['geo_err_mm'])} | "
                  f"{_fmt(rs['order_geo'])} |\n")
    md.append("\n## S sensitivity (dx=1.0 mm, RS2)\n\n| case | cfl | iters | shift 1 | shift 100 | E_sd 1 | E_sd 100 |\n"
              "|---|---:|---:|---:|---:|---:|---:|\n")
    md += [f"| {r['case']} | {r['cfl']} | {r['iters']} | {r['shift_1']:.2e} | {r['shift_100']:.2e} | "
           f"{r['E_sd_1']:.2e} | {r['E_sd_100']:.2e} |\n" for r in s]
    md.append("\n## L band vs global (dx=1.0 mm, RS2)\n\n| case | band cells | shift 1 | shift 100 | E_sd 100 | wall s |\n"
              "|---|---:|---:|---:|---:|---:|\n")
    md += [f"| {r['case']} | {r['band_cells']} | {r['shift_1']:.2e} | {r['shift_100']:.2e} | {r['E_sd_100']:.2e} | "
           f"{r['wall_s']:.1f} |\n" for r in l_]
    md.append("\n## D1 zero velocity, advect + reinit every step (2 iters), 200 steps, dx=1.0 mm\n\n"
              "| case | method | shift | geo err | volume change |\n|---|---|---:|---:|---:|\n")
    md += [f"| {r['case']} | {r['method']} | {r['max_shift_dx']:.2e} | {r['geo_err_max_dx']:.2e} | "
           f"{r['volume_change']:+.2e} |\n" for r in d1]
    md.append("\n## D2 translating circle (geo err max, dx units; volume change)\n\n"
              "| dx mm | steps | advection only | legacy reinit | RS2 reinit | dV adv | dV legacy | dV RS2 |\n"
              "|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    md += [f"| {r['dx_mm']:.2f} | {r['steps']} | {r['advection_only']['geo_err_max_dx']:.3e} | "
           f"{r['legacy_reinit']['geo_err_max_dx']:.3e} | {r['rs2_reinit']['geo_err_max_dx']:.3e} | "
           f"{r['advection_only']['volume_change']:+.2e} | {r['legacy_reinit']['volume_change']:+.2e} | "
           f"{r['rs2_reinit']['volume_change']:+.2e} |\n" for r in d2]
    if v3:
        md.append("\n## V3 operational rigid body (Omega=20, correction OFF unless noted)\n\n"
                  "| dx mm | reinit | variant | eta NRMSE | center err | max ur | max uz | late U_mer | peak U_mer | "
                  "volume drift | n reinit | cum reinit shift (dx) | net contour shift (dx) |\n"
                  "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in v3:
            md.append(f"| {r['dx_mm']:.2f} | {r['method']} | {r.get('variant', '')} | {r['eta_nrmse']:.2e} | "
                      f"{r['center_err_rel']:.2e} | {r['max_ur']:.2e} | {r['max_uz']:.2e} | {r['late_U_mer']:.2e} | "
                      f"{r['peak_U_mer']:.2e} | {r['volume_drift']:+.2e} | {r['n_reinit']} | "
                      f"{r['cum_reinit_shift_dx']:.2e} | {r['net_contour_shift_dx']:.2e} |\n")
    (out_dir / "summary.md").write_text("".join(md), encoding="utf-8")
    print("".join(md))
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
