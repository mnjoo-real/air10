"""Level-1A Milestone-1 verification report (README_rewritten Gates V1-V3,
sigma = 0, no stirring). Writes

    results/validation_single_phase_m1/metadata.json   (kind="single_phase_milestone1")
    results/validation_single_phase_m1/summary.md

Sections:
  E1  1D interface-Dirichlet manufactured tests (P1 linear, P2 quadratic)
  E2  2D axisymmetric manufactured tests (axial off-grid surface, radial air core)
  C   hydrostatic flat free surface, long run          (Gate V2)
  D   rigid-body rotating liquid, MATCHED co-rotating BC (Gate V3), three
      grids, volume correction OFF and ON, phi = signed distance (and the
      phi = z - eta(r) form, which is exact by construction, for reference)
  R   reinitialization-drift diagnostic (same rigid-body state, reinit ON)

Usage:
    python scripts/validate_single_phase_milestone1.py [--t-end 1.0] [--quick]
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
from air_vortex.free_surface_bc import interface_pressure  # noqa: E402
from air_vortex.grid import Grid, build_grid  # noqa: E402
from air_vortex.liquid_mask import classify, liquid_volume_staircase, liquid_volume_subcell  # noqa: E402
from air_vortex.pressure_single_phase import (  # noqa: E402
    solve_liquid_pressure_from_source, solve_poisson_1d_ghost_fluid)
from air_vortex.single_phase_solver import (  # noqa: E402
    SinglePhaseSolver, WallBC, build_single_phase_solver, initialize_rigid_body_single_phase)


def _grid(Nr, Nz, dr, dz) -> Grid:
    r_f = np.linspace(0.0, Nr * dr, Nr + 1)
    z_f = np.linspace(0.0, Nz * dz, Nz + 1)
    return Grid(dr=dr, dz=dz, r_v=r_f[-1], z_max=z_f[-1], Nr=Nr, Nz=Nz,
                r_c=0.5 * (r_f[:-1] + r_f[1:]), z_c=0.5 * (z_f[:-1] + z_f[1:]), r_f=r_f, z_f=z_f)


def _order(hs, errs):
    return float(np.polyfit(np.log(hs), np.log(errs), 1)[0])


def section_e1() -> dict:
    out = {"P1_linear": [], "P2_quadratic": []}
    n, h = 20, 0.05
    for alpha in (0.1, 0.25, 0.5, 0.75, 0.9, 1e-3, 1e-6, 1e-9, 1e-12):
        xg = 11.5 * h + alpha * h
        x, p = solve_poisson_1d_ghost_fluid(n, h, np.zeros(n), xg, 3.0, -7.0)
        liq = np.isfinite(p)
        err = float(np.max(np.abs(p[liq] - (3.0 - 7.0 * (x[liq] - xg)))))
        out["P1_linear"].append({"h": h, "alpha": alpha, "max_err": err})
    for xg in (0.6137, 0.8021):
        hs, errs = [], []
        for n in (10, 20, 40, 80, 160):
            h = 1.0 / n
            x, p = solve_poisson_1d_ghost_fluid(n, h, 2.0 * np.ones(n), xg, 3.0, 0.0)
            liq = np.isfinite(p)
            m = int(liq.sum())
            alpha = (xg - x[m - 1]) / h
            err = float(np.max(np.abs(p[liq] - (x[liq] ** 2 - xg**2 + 3.0))))
            hs.append(h)
            errs.append(err)
            out["P2_quadratic"].append({"x_gamma": xg, "h": h, "alpha": alpha, "max_err": err})
        out.setdefault("P2_order", {})[str(xg)] = _order(hs, errs)
    return out


def section_e2() -> dict:
    out = {"axial": [], "radial": []}
    A, p0, H = 5.0, 2.0, 0.6137
    hs, errs = [], []
    for n in (10, 20, 40, 80):
        h = 1.0 / n
        g = _grid(4, n, 0.25, h)
        geom = classify(g.z_c[None, :] - H + 0 * g.r_c[:, None])
        pr, pz = interface_pressure(geom)
        pr, pz = np.where(np.isfinite(pr), p0, np.nan), np.where(np.isfinite(pz), p0, np.nan)
        p = solve_liquid_pressure_from_source(g, geom, -2 * A * np.ones(g.shape_center), pr, pz)
        exact = A * (H**2 - g.z_c[None, :] ** 2) + p0 + 0 * g.r_c[:, None]
        err = float(np.max(np.abs(p - exact)[geom.liquid]))
        alpha = float(np.nanmax(geom.theta_z_raw))
        hs.append(h)
        errs.append(err)
        out["axial"].append({"h": h, "alpha": alpha, "max_err": err})
    out["axial_order"] = _order(hs, errs)

    R, rg, p0 = 1.0, 0.3371, 1.5
    hs, errs = [], []
    for n in (10, 20, 40, 80):
        h = R / n
        g = _grid(n, 3, h, h)
        geom = classify(rg - g.r_c[:, None] + 0 * g.z_c[None, :])
        pr, pz = interface_pressure(geom)
        pr, pz = np.where(np.isfinite(pr), p0, np.nan), np.where(np.isfinite(pz), p0, np.nan)
        p = solve_liquid_pressure_from_source(g, geom, 4.0 * np.ones(g.shape_center), pr, pz,
                                              top_bc="neumann")
        ex = g.r_c**2 - 2 * R**2 * np.log(g.r_c) - (rg**2 - 2 * R**2 * np.log(rg)) + p0
        err = float(np.max(np.abs(p - ex[:, None])[geom.liquid]))
        alpha = float(np.nanmax(geom.theta_r_raw))
        hs.append(h)
        errs.append(err)
        out["radial"].append({"h": h, "alpha": alpha, "max_err": err})
    out["radial_order"] = _order(hs, errs)
    return out


def section_c(t_end: float) -> dict:
    cfg = load_config(ROOT / "configs" / "single_phase_hydrostatic.yaml")
    s = build_single_phase_solver(cfg)
    g = s.grid
    H = cfg.geometry.water_height_m
    V0 = liquid_volume_subcell(g, s.fields.phi)
    V0s = liquid_volume_staircase(g, s.fields.phi)
    exact = cfg.fluid.water_density * cfg.fluid.gravity * (H - g.z_c[None, :]) + 0 * g.r_c[:, None]
    mx_ur = mx_uz = mx_div = mx_perr = 0.0
    t0 = time.perf_counter()
    while s.fields.t < t_end:
        d = s.step()
        liq = s.fields.phi < 0
        mx_ur = max(mx_ur, d.max_abs_ur_liquid)
        mx_uz = max(mx_uz, d.max_abs_uz_liquid)
        mx_div = max(mx_div, d.max_divergence_liquid)
        mx_perr = max(mx_perr, float(np.max(np.abs(s.fields.p - exact)[liq])))
    eta = free_surface_height(s.fields.phi, g)
    return {
        "dx_mm": g.dr * 1e3, "H_mm": H * 1e3, "t_end_s": s.fields.t, "steps": s.fields.step,
        "alpha_interface": float(np.nanmax(classify(s.fields.phi).theta_z_raw)),
        "max_ur": mx_ur, "max_uz": mx_uz, "max_div": mx_div,
        "max_pressure_err_pa": mx_perr,
        "max_pressure_err_rel": mx_perr / float(np.max(exact)),
        "volume_drift_subcell": (liquid_volume_subcell(g, s.fields.phi) - V0) / V0,
        "volume_drift_staircase": (liquid_volume_staircase(g, s.fields.phi) - V0s) / V0s,
        "max_eta_err_m": float(np.nanmax(np.abs(eta - H))),
        "wall_time_s": time.perf_counter() - t0,
    }


def run_rigid(dx: float, omega: float, t_end: float, phi_form: str, vc: bool,
              reinit_every: int = 0) -> dict:
    base = load_config(ROOT / "configs" / "single_phase_rigid_body.yaml")
    cfg = copy.deepcopy(base)
    cfg.grid.dr_m = cfg.grid.dz_m = dx
    cfg.levelset.volume_correction.enabled = vc
    cfg.levelset.reinitialize_every = reinit_every
    grid = build_grid(cfg)
    fields = initialize_rigid_body_single_phase(grid, cfg, omega, phi_form=phi_form)
    s = SinglePhaseSolver(grid=grid, cfg=cfg, fields=fields, wall_bc=WallBC.rotating(omega))
    g_acc = cfg.fluid.gravity
    Vt = np.pi * grid.r_v**2 * cfg.geometry.water_height_m
    eta_an = volume_consistent_parabola(grid.r_c, grid.r_v, omega, g_acc, Vt)
    amp = omega**2 * grid.r_v**2 / (2 * g_acc)
    dep = cfg.geometry.water_height_m - eta_an[0]
    V0 = liquid_volume_subcell(grid, s.fields.phi)
    rho = cfg.fluid.water_density
    C = eta_an[0]
    p_an = rho * (0.5 * omega**2 * grid.r_c[:, None] ** 2 - g_acc * (grid.z_c[None, :] - C))
    hist, mx_div, mx_perr, reinit_shift, reinit_dv = [], 0.0, 0.0, 0.0, 0.0
    t0 = time.perf_counter()
    while s.fields.t < t_end:
        d = s.step()
        hist.append(max(d.max_abs_ur_liquid, d.max_abs_uz_liquid))
        mx_div = max(mx_div, d.max_divergence_liquid)
        liq = s.fields.phi < 0
        mx_perr = max(mx_perr, float(np.max(np.abs(s.fields.p - p_an)[liq])))
        if d.reinit_applied:
            reinit_shift = max(reinit_shift, d.reinit_max_eta_shift)
            reinit_dv += d.reinit_volume_change
    eta = free_surface_height(s.fields.phi, grid)
    n = len(hist)
    return {
        "dx_mm": dx * 1e3, "omega": omega, "phi_form": phi_form, "volume_correction": vc,
        "reinitialize_every": reinit_every, "t_end_s": s.fields.t, "steps": s.fields.step,
        "eta_nrmse": float(np.sqrt(np.nanmean((eta - eta_an) ** 2)) / amp),
        "center_err_rel_depression": float(abs(eta[0] - eta_an[0]) / dep),
        "max_ur": float(max(d_ for d_ in hist)), "max_vel_first_quarter": float(max(hist[: n // 4])),
        "max_vel_last_quarter": float(max(hist[3 * n // 4:])),
        "max_ur_final_step": d.max_abs_ur_liquid, "max_uz_final_step": d.max_abs_uz_liquid,
        "max_div": mx_div, "max_pressure_err_pa": mx_perr,
        "volume_drift_subcell": (liquid_volume_subcell(grid, s.fields.phi) - V0) / V0,
        "reinit_max_eta_shift_m": reinit_shift, "reinit_cum_volume_change_m3": reinit_dv,
        "cumulative_volume_correction_m": s.cumulative_volume_correction,
        "wall_time_s": time.perf_counter() - t0,
    }


def _md_table(rows, cols, fmt):
    head = "| " + " | ".join(cols) + " |\n|" + "|".join("---:" for _ in cols) + "|\n"
    body = "".join("| " + " | ".join(fmt.get(c, "{}").format(r[c]) for c in cols) + " |\n" for r in rows)
    return head + body


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--t-end", type=float, default=1.0)
    ap.add_argument("--omega", type=float, default=20.0)
    ap.add_argument("--results-root", default=str(ROOT / "results"))
    ap.add_argument("--quick", action="store_true", help="t_end=0.2, coarse grids only")
    args = ap.parse_args()
    t_end = 0.2 if args.quick else args.t_end
    grids = (1.5e-3, 1.0e-3) if args.quick else (1.5e-3, 1.0e-3, 0.75e-3)

    out_dir = Path(args.results_root) / "validation_single_phase_m1"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("E1 1D manufactured ...")
    e1 = section_e1()
    print("E2 2D manufactured ...")
    e2 = section_e2()
    print(f"C  hydrostatic, t_end={t_end} ...")
    c = section_c(t_end)
    print(f"D  rigid body, omega={args.omega}, t_end={t_end} ...")
    d_rows = []
    for dx in grids:
        for vc in (False, True):
            d_rows.append(run_rigid(dx, args.omega, t_end, "signed_distance", vc))
            print(f"   dx={dx*1e3:.2f}mm vc={vc}: nrmse={d_rows[-1]['eta_nrmse']:.2e} "
                  f"maxvel={d_rows[-1]['max_ur']:.2e}")
    d_rows.append(run_rigid(grids[0], args.omega, t_end, "height", False))
    print("R  reinit diagnostic ...")
    r_rows = [run_rigid(dx, args.omega, t_end, "signed_distance", False, reinit_every=5)
              for dx in grids]

    meta = {"kind": "single_phase_milestone1", "omega": args.omega, "t_end_s": t_end,
            "E1": e1, "E2": e2, "C_hydrostatic": c, "D_rigid_body": d_rows, "R_reinit": r_rows}
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    g = "{:.3e}"
    md = [f"# Level-1A Milestone 1 verification (sigma=0, t_end={t_end}s, Omega={args.omega} rad/s)\n"]
    md.append("## E1 1D P1 (linear, exact for every alpha)\n")
    md.append(_md_table(e1["P1_linear"], ["h", "alpha", "max_err"], {"alpha": "{:.0e}", "max_err": g}))
    md.append("\n## E1 1D P2 (quadratic)\n")
    md.append(_md_table(e1["P2_quadratic"], ["x_gamma", "h", "alpha", "max_err"],
                        {"h": "{:.4f}", "alpha": "{:.3f}", "max_err": g}))
    md.append(f"\nobserved order: {e1['P2_order']}\n")
    md.append("\n## E2 2D axial / radial\n")
    md.append(_md_table(e2["axial"], ["h", "alpha", "max_err"], {"h": "{:.4f}", "alpha": "{:.3f}", "max_err": g}))
    md.append(f"\naxial order {e2['axial_order']:.2f}\n\n")
    md.append(_md_table(e2["radial"], ["h", "alpha", "max_err"], {"h": "{:.4f}", "alpha": "{:.3f}", "max_err": g}))
    md.append(f"\nradial order {e2['radial_order']:.2f}\n")
    md.append("\n## C hydrostatic\n")
    md.append("\n".join(f"- {k}: {v}" for k, v in c.items()) + "\n")
    cols = ["dx_mm", "phi_form", "volume_correction", "eta_nrmse", "center_err_rel_depression",
            "max_ur", "max_vel_last_quarter", "max_div", "volume_drift_subcell", "wall_time_s"]
    fmt = {k: g for k in cols}
    fmt.update({"dx_mm": "{:.2f}", "phi_form": "{}", "volume_correction": "{}", "wall_time_s": "{:.1f}"})
    md.append("\n## D rigid body, matched co-rotating BC\n(max_ur column = max over run of max(|u_r|,|u_z|) in liquid)\n\n")
    md.append(_md_table(d_rows, cols, fmt))
    md.append("\n## R reinitialization ON (every 5 steps) -- diagnostic\n\n")
    md.append(_md_table(r_rows, cols + ["reinit_max_eta_shift_m"], {**fmt, "reinit_max_eta_shift_m": g}))
    (out_dir / "summary.md").write_text("".join(md), encoding="utf-8")
    print("".join(md))
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
