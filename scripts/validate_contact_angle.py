"""Gate V4b-S: static wall contact geometry (single_phase_ls, sigma > 0).

Sections (volume correction OFF, reinit OFF unless stated):
  A  geometric contact angle: cone (CA1) and spherical cap (CA2) vs input theta
  B  static gravity-capillary meniscus vs the nonlinear Young-Laplace BVP,
     theta = 60/75/90/105/120 deg, three grids, static_angle and pinned wall models
  C  wall-region convergence (from B)
  D  contact-line policy evidence: long static runs (drift), and a FLAT start
     with a prescribed 60-degree angle (needs contact-line motion; no-slip wall)
  R  RS2 reinitialization near the contact line (quality-triggered, forced periodic)

Writes results/validation_contact_angle/*.csv, figures and summary.md
(Table E, the long-horizon isolated drop, comes from
scripts/validate_capillary_long_horizon.py).

Usage: python scripts/validate_contact_angle.py [--quick]
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from air_vortex.config import WallConfig  # noqa: E402
from air_vortex.contact_angle import contact_point, wall_contact_points  # noqa: E402
from air_vortex.curvature_single_phase import (  # noqa: E402
    crossing_positions, interface_curvature, principal_resolution)
from air_vortex.diagnostics import free_surface_height  # noqa: E402
from air_vortex.liquid_mask import classify, liquid_volume_subcell  # noqa: E402
from air_vortex.meniscus import G, RHO, SIGMA, build_meniscus_solver  # noqa: E402
from air_vortex.reinit_benchmarks import uniform_grid  # noqa: E402
from air_vortex.reinit_diagnostics import contour_displacement  # noqa: E402

R_V = 0.008
INK, MUTED, GRIDC, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]


def _write_csv(path, rows):
    if not rows:
        return
    keys = list(rows[0])
    for r in rows[1:]:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def _style(ax):
    ax.set_facecolor(SURF)
    ax.grid(True, color=GRIDC, lw=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRIDC)
    ax.tick_params(labelsize=8, colors=MUTED)


def _crossings_kappa(g, phi, wc):
    geom = classify(phi)
    kr, kz = interface_curvature(g, geom, phi, wc)
    rr, zr, rz, zz = crossing_positions(g, geom)
    m1, m2 = np.isfinite(kr), np.isfinite(kz)
    return np.r_[kr[m1], kz[m2]], np.r_[rr[m1], rz[m2]], np.r_[zr[m1], zz[m2]]


# ------------------------------------------------------------------ A
def section_a(dx=0.25e-3):
    rows = []
    g = uniform_grid(dx, R=R_V, Z=0.012)
    Rm, Zm = np.meshgrid(g.r_c, g.z_c, indexing="ij")
    for th in (45.0, 60.0, 90.0, 120.0, 135.0):
        t = np.deg2rad(th)
        phi = (Zm - 0.0063 - (Rm - R_V) / np.tan(t)) * np.sin(t)
        k, r, _ = _crossings_kappa(g, phi, WallConfig("static_angle", th))
        wall = r > R_V - 3 * dx
        ex = -np.cos(t) / r
        _, thm = contact_point(g, phi)
        rows.append({"geometry": "cone (CA1)", "theta_in": th, "theta_measured": thm,
                     "angle_err_deg": thm - th,
                     "wall_kappa_err_rel_to_1overR": float(np.abs(k[wall] - ex[wall]).max() * R_V),
                     "bulk_kappa_err_rel_to_1overR": float(np.abs(k[~wall & (r > R_V / 2)] - ex[~wall & (r > R_V / 2)]).max() * R_V)})
    for th in (60.0, 75.0, 105.0, 120.0):
        t = np.deg2rad(th)
        Rs = R_V / abs(np.cos(t))
        h = np.sqrt(Rs**2 - R_V**2)
        phi, kex = ((Rs - np.hypot(Rm, Zm - (0.0063 + h)), -2 / Rs) if th < 90
                    else (np.hypot(Rm, Zm - (0.0063 - h)) - Rs, 2 / Rs))
        k, r, _ = _crossings_kappa(g, phi, WallConfig("static_angle", th))
        wall = r > R_V - 3 * dx
        _, thm = contact_point(g, phi)
        rows.append({"geometry": "spherical cap (CA2)", "theta_in": th, "theta_measured": thm,
                     "angle_err_deg": thm - th,
                     "wall_kappa_err_rel": float(np.abs(k[wall] / kex - 1).max()),
                     "bulk_kappa_err_rel": float(np.abs(k[~wall] / kex - 1).max())})
    return rows


# ------------------------------------------------------------------ B
def meniscus_case(dx, th, model, T, reinit=None, log_reinit=None):
    s, ref = build_meniscus_solver(dx, th, contact_model=model)
    g = s.grid
    if reinit:
        ls = s.cfg.levelset
        ls.reinitialization_method = "russo_smereka_subcell"
        ls.reinitialize_every = reinit["every"]
        ls.reinitialize_iterations = reinit.get("iters", 2)
        ls.reinitialization_trigger = reinit.get("trigger", "periodic")
        ls.reinitialization_quality_threshold = reinit.get("threshold", 0.05)
        if log_reinit is not None:
            def cb(before, after, step):
                zb, tb = contact_point(g, before)
                za, ta = contact_point(g, after)
                kb = _kappa_err(g, before, s.cfg.wall, ref)
                ka = _kappa_err(g, after, s.cfg.wall, ref)
                log_reinit.append({"model": model, "trigger": reinit.get("trigger", "periodic"), "step": step,
                                   "theta_before": tb, "theta_after": ta, "theta_change_deg": ta - tb,
                                   "z_contact_change_dx": (za - zb) / g.dz,
                                   "kappa_wall_err_before": kb[1], "kappa_wall_err_after": ka[1],
                                   "kappa_err_before": kb[0], "kappa_err_after": ka[0],
                                   "shift_dx": contour_displacement(g, before, after).max_shift_dx})
            s.reinit_callback = cb
    V0 = liquid_volume_subcell(g, s.fields.phi)
    ts, us, zcl = [], [], []
    t0 = time.perf_counter()
    k = 0
    while s.fields.t < T:
        d = s.step()
        k += 1
        ts.append(d.t)
        us.append(max(d.max_abs_ur_liquid, d.max_abs_uz_liquid))
        if k % 20 == 0:
            zc = wall_contact_points(g, s.fields.phi)
            zcl.append((d.t, float(zc[0]) if zc.size == 1 else np.nan))
    eta = free_surface_height(s.fields.phi, g)
    e = eta - ref.eta(g.r_c)
    wall = g.r_c > R_V - 4 * dx
    z_meas, th_meas = contact_point(g, s.fields.phi)
    kerr_glob, kerr_wall, pres_resid = _kappa_err(g, s.fields.phi, s.cfg.wall, ref, with_p=True)
    liq = s.fields.phi < 0
    z2d = g.z_c[None, :] * np.ones(g.shape_center)
    p_bulk_err = float(np.abs(s.fields.p - (ref.P0 - RHO * G * z2d))[liq].max() / ref.P0)
    pr = principal_resolution(g, classify(s.fields.phi), s.fields.phi, s.cfg.wall)
    us = np.array(us)
    m = len(us)
    return {"theta": th, "dx_mm": dx * 1e3, "model": model, "T": s.fields.t, "steps": m,
            "rmse_dx": float(np.sqrt(np.mean(e**2)) / dx), "rmse_um": float(np.sqrt(np.mean(e**2)) * 1e6),
            "wall_rmse_dx": float(np.sqrt(np.mean(e[wall] ** 2)) / dx),
            "wall_height_err_dx": (z_meas - ref.z_wall) / dx, "theta_measured": th_meas,
            "theta_err_deg": th_meas - th,
            "U_peak": float(us.max()), "U_late": float(us[3 * m // 4:].max()),
            "volume_drift": liquid_volume_subcell(g, s.fields.phi) / V0 - 1,
            "kappa_err_rel_global": kerr_glob, "kappa_err_rel_wall": kerr_wall,
            "pGamma_residual_rel": pres_resid, "p_bulk_err_rel": p_bulk_err,
            "N_meridional_min": pr["meridional"]["min_cells"], "N_azimuthal_min": pr["azimuthal"]["min_cells"],
            "n_reinit": d.n_reinit_calls, "wall_s": time.perf_counter() - t0,
            "_series": {"t": ts, "U": us.tolist(), "zcl": zcl, "z_wall_ref": ref.z_wall},
            "_profile": {"r": g.r_c.tolist(), "eta": eta.tolist()}}


def _kappa_err(g, phi, wall_cfg, ref, with_p=False):
    """Curvature at the crossings vs the exact Young-Laplace value
    (P0 - rho g z)/sigma at the SAME height; normalised by max|kappa_exact|."""
    k, r, z = _crossings_kappa(g, phi, wall_cfg)
    kex = ref.kappa_at_height(z)
    # floor at 1/R_v: a flat (theta = 90) meniscus has kappa_exact = 0 everywhere
    scale = max(np.abs(ref.kappa_at_height(ref.z)).max(), 1.0 / R_V)
    wall = r > R_V - 3 * g.dr
    rel = np.abs(k - kex) / scale
    out = (float(np.sqrt(np.mean(rel**2))), float(rel[wall].max()) if np.any(wall) else float("nan"))
    if with_p:
        # normal-stress residual: imposed sigma*kappa vs the hydrostatic liquid pressure there
        resid = np.abs(SIGMA * k - (ref.P0 - RHO * G * z)) / ref.P0
        return out[0], out[1], float(np.sqrt(np.mean(resid**2)))
    return out


def section_b(grids, T):
    rows = []
    for dx in grids:
        for th in (60.0, 75.0, 90.0, 105.0, 120.0):
            for model in ("static_angle", "pinned"):
                r = meniscus_case(dx, th, model, T)
                rows.append(r)
                print(f"  B th={th} dx={dx*1e3} {model}: rmse={r['rmse_dx']:.2e}dx wallH={r['wall_height_err_dx']:+.2e}dx "
                      f"Ulate={r['U_late']:.1e} dV={r['volume_drift']:+.1e}", flush=True)
    return rows


# ------------------------------------------------------------------ D
def section_d(long_T, flat_T):
    long_rows = []
    for model in ("pinned", "static_angle"):
        for dx, T in ((0.5e-3, long_T), (0.25e-3, long_T / 2)):
            r = meniscus_case(dx, 60.0, model, T)
            long_rows.append(r)
            print(f"  D long {model} dx={dx*1e3}: Ulate={r['U_late']:.1e} wallH={r['wall_height_err_dx']:+.2e}dx", flush=True)
    flat = []
    for dx in (0.5e-3, 0.25e-3):
        s, ref = build_meniscus_solver(dx, 60.0, contact_model="static_angle", start="flat")
        g = s.grid
        z0 = wall_contact_points(g, s.fields.phi)[0]
        hist = []
        k = 0
        while s.fields.t < flat_T:
            d = s.step()
            k += 1
            if k % 20 == 0:
                zc = wall_contact_points(g, s.fields.phi)
                hist.append((d.t, float(zc[0]) if zc.size == 1 else np.nan,
                             max(d.max_abs_ur_liquid, d.max_abs_uz_liquid)))
        flat.append({"dx_mm": dx * 1e3, "z_wall_equilibrium_rise_mm": (ref.z_wall - z0) * 1e3,
                     "contact_rise_at_end_mm": (hist[-1][1] - z0) * 1e3, "T": s.fields.t,
                     "U_max": max(h[2] for h in hist), "_hist": hist})
        print(f"  D flat-start dx={dx*1e3}: rise {flat[-1]['contact_rise_at_end_mm']:.3f} mm of "
              f"{flat[-1]['z_wall_equilibrium_rise_mm']:.3f} mm", flush=True)
    return long_rows, flat


# ------------------------------------------------------------------ R
def section_r(T):
    log, rows = [], []
    for model in ("static_angle", "pinned"):
        for mode in ({"every": 1, "trigger": "quality", "threshold": 0.01},
                     {"every": 50, "iters": 2}):
            r = meniscus_case(0.25e-3, 60.0, model, T, reinit=mode, log_reinit=log)
            r["reinit_mode"] = mode.get("trigger", "periodic") + f"/{mode['every']}"
            rows.append(r)
            print(f"  R {model} {r['reinit_mode']}: n={r['n_reinit']} theta_err={r['theta_err_deg']:+.2f} "
                  f"wallH={r['wall_height_err_dx']:+.2e}dx Ulate={r['U_late']:.1e}", flush=True)
    return rows, log


# ------------------------------------------------------------------ figures
def figures(out, a_rows, b_rows, long_rows, flat):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from air_vortex.meniscus import solve_meniscus

    # meniscus profiles: BVP lines + CFD markers (finest static_angle grid)
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=150)
    fig.patch.set_facecolor(SURF)
    _style(ax)
    fine = max(r["dx_mm"] for r in b_rows)
    fine = min(r["dx_mm"] for r in b_rows)
    for c, th in zip(SERIES, (60.0, 75.0, 90.0, 105.0, 120.0)):
        ref = solve_meniscus(th, R_V, 0.006)
        ax.plot(ref.r * 1e3, ref.z * 1e3, color=c, lw=2)
        row = next(r for r in b_rows if r["theta"] == th and r["dx_mm"] == fine and r["model"] == "static_angle")
        ax.plot(np.array(row["_profile"]["r"]) * 1e3, np.array(row["_profile"]["eta"]) * 1e3, "o", ms=3.5,
                mfc=SURF, mec=c, mew=1.2)
        ax.annotate(f"θ = {th:.0f}°", (R_V * 1e3, ref.z_wall * 1e3), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=8, color=INK)
    ax.set_xlim(0, R_V * 1e3 * 1.18)
    ax.set_xlabel("r (mm)", fontsize=9, color=MUTED)
    ax.set_ylabel("free-surface height (mm)", fontsize=9, color=MUTED)
    ax.set_title(f"Static meniscus: Young-Laplace BVP (lines) vs CFD after {b_rows[0]['T']:.2f} s "
                 f"(markers, dx = {fine} mm)", loc="left", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(out / "meniscus_profiles.png")
    plt.close(fig)

    # wall curvature error vs dx (theta = 60)
    fig, ax = plt.subplots(figsize=(6.4, 3.8), dpi=150)
    fig.patch.set_facecolor(SURF)
    _style(ax)
    for c, model in zip(SERIES, ("static_angle", "pinned")):
        rr = sorted([r for r in b_rows if r["theta"] == 60.0 and r["model"] == model], key=lambda r: r["dx_mm"])
        ax.loglog([r["dx_mm"] for r in rr], [r["kappa_err_rel_wall"] for r in rr], "-o", color=c, lw=2, ms=5,
                  label=f"{model}: wall region")
        ax.loglog([r["dx_mm"] for r in rr], [r["kappa_err_rel_global"] for r in rr], "--", color=c, lw=2,
                  label=f"{model}: all crossings (RMS)")
    ax.set_xlabel("dx (mm)", fontsize=9, color=MUTED)
    ax.set_ylabel("|kappa - kappa_YL| / max|kappa_YL|", fontsize=9, color=MUTED)
    ax.set_title("Meniscus curvature error, theta = 60 deg", loc="left", fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "wall_curvature_error.png")
    plt.close(fig)

    # contact-point drift + spurious velocity, long runs
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), dpi=150)
    fig.patch.set_facecolor(SURF)
    for ax in axes:
        _style(ax)
    for c, r in zip(SERIES, long_rows):
        lab = f"{r['model']}, dx = {r['dx_mm']} mm"
        zc = np.array(r["_series"]["zcl"])
        axes[0].plot(zc[:, 0], (zc[:, 1] - r["_series"]["z_wall_ref"]) / (r["dx_mm"] * 1e-3), color=c, lw=2, label=lab)
        axes[1].semilogy(r["_series"]["t"], np.maximum(r["_series"]["U"], 1e-18), color=c, lw=1.5, label=lab)
    axes[0].set_ylabel("contact height - BVP (dx)", fontsize=9, color=MUTED)
    axes[1].set_ylabel("max |u| in liquid (m/s)", fontsize=9, color=MUTED)
    for ax in axes:
        ax.set_xlabel("time (s)", fontsize=9, color=MUTED)
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle("Static meniscus, theta = 60 deg, no-slip wall, reinit OFF", x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "contact_point_drift.png")
    plt.close(fig)

    # contact angle geometry
    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=150)
    fig.patch.set_facecolor(SURF)
    _style(ax)
    for c, geom in zip(SERIES, ("cone (CA1)", "spherical cap (CA2)")):
        rr = [r for r in a_rows if r["geometry"] == geom]
        ax.plot([r["theta_in"] for r in rr], [abs(r["angle_err_deg"]) + 1e-16 for r in rr], "o-", color=c, lw=2,
                label=geom)
    ax.set_yscale("log")
    ax.set_xlabel("input contact angle (deg)", fontsize=9, color=MUTED)
    ax.set_ylabel("|measured - input| (deg)", fontsize=9, color=MUTED)
    ax.set_title("Contact angle measured from interface geometry, dx = 0.25 mm", loc="left", fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "contact_angle_geometry.png")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--results-root", default=str(ROOT / "results"))
    args = ap.parse_args()
    out = Path(args.results_root) / "validation_contact_angle"
    out.mkdir(parents=True, exist_ok=True)
    grids = (0.5e-3, 0.25e-3) if args.quick else (0.5e-3, 0.25e-3, 0.125e-3)
    T = 0.1 if args.quick else 0.3

    print("A geometry", flush=True)
    a = section_a()
    _write_csv(out / "static_angle_geometry.csv", a)
    print("B meniscus", flush=True)
    b = section_b(grids, T)
    _write_csv(out / "meniscus_bvp_comparison.csv", [{k: v for k, v in r.items() if not k.startswith("_")} for r in b])
    conv = [{k: r[k] for k in ("model", "dx_mm", "rmse_dx", "wall_rmse_dx", "theta_err_deg", "kappa_err_rel_global",
                               "kappa_err_rel_wall", "U_late")} for r in b if r["theta"] == 60.0]
    _write_csv(out / "wall_grid_convergence.csv", conv)
    print("D policy", flush=True)
    long_rows, flat = section_d(0.5 if args.quick else 3.0, 0.2 if args.quick else 0.5)
    _write_csv(out / "wall_long_horizon.csv",
               [{k: v for k, v in r.items() if not k.startswith("_")} for r in long_rows])
    _write_csv(out / "flat_start_fixed_angle.csv", [{k: v for k, v in r.items() if not k.startswith("_")} for r in flat])
    print("R reinit", flush=True)
    r_rows, r_log = section_r(0.1 if args.quick else 0.5)
    _write_csv(out / "reinit_contact_angle.csv", r_log or [{"note": "no reinit events"}])
    figures(out, a, b, long_rows, flat)

    f = lambda v: f"{v:.2e}" if isinstance(v, float) else str(v)  # noqa: E731
    md = ["# V4b-S static wall contact geometry\n\n## A geometric contact angle (dx = 0.25 mm)\n\n"
          "| geometry | theta in | theta measured | angle error (deg) | wall kappa error | bulk kappa error |\n"
          "|---|---:|---:|---:|---:|---:|\n"]
    for r in a:
        wk = r.get("wall_kappa_err_rel", r.get("wall_kappa_err_rel_to_1overR"))
        bk = r.get("bulk_kappa_err_rel", r.get("bulk_kappa_err_rel_to_1overR"))
        md.append(f"| {r['geometry']} | {r['theta_in']:.0f} | {r['theta_measured']:.4f} | {r['angle_err_deg']:+.1e} | "
                  f"{wk:.1e} | {bk:.1e} |\n")
    md.append(f"\n## B static meniscus vs Young-Laplace BVP (T = {T} s)\n\n| theta | dx mm | model | RMSE (dx) | "
              "wall RMSE (dx) | wall-height err (dx) | theta err (deg) | U late | volume drift | kappa err RMS | "
              "kappa err wall | p_Gamma residual | N_m min | N_theta min |\n"
              "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for r in b:
        md.append(f"| {r['theta']:.0f} | {r['dx_mm']} | {r['model']} | {r['rmse_dx']:.1e} | {r['wall_rmse_dx']:.1e} | "
                  f"{r['wall_height_err_dx']:+.1e} | {r['theta_err_deg']:+.2f} | {r['U_late']:.1e} | "
                  f"{r['volume_drift']:+.1e} | {r['kappa_err_rel_global']:.1e} | {r['kappa_err_rel_wall']:.1e} | "
                  f"{r['pGamma_residual_rel']:.1e} | {r['N_meridional_min']:.1f} | {r['N_azimuthal_min']:.1f} |\n")
    md.append("\n## D long static runs (theta = 60) and flat start with a fixed 60-degree angle\n\n"
              "| model | dx mm | T | U peak | U late | wall-height err (dx) | volume drift |\n|---|---:|---:|---:|---:|---:|---:|\n")
    for r in long_rows:
        md.append(f"| {r['model']} | {r['dx_mm']} | {r['T']:.2f} | {r['U_peak']:.1e} | {r['U_late']:.1e} | "
                  f"{r['wall_height_err_dx']:+.2e} | {r['volume_drift']:+.1e} |\n")
    md.append("\n| dx mm | contact rise after T (mm) | equilibrium rise (mm) | T | U max |\n|---:|---:|---:|---:|---:|\n")
    for r in flat:
        md.append(f"| {r['dx_mm']} | {r['contact_rise_at_end_mm']:.4f} | {r['z_wall_equilibrium_rise_mm']:.4f} | "
                  f"{r['T']:.2f} | {r['U_max']:.1e} |\n")
    md.append("\n## R reinit near the contact line (theta = 60, dx = 0.25 mm)\n\n| model | mode | n reinit | "
              "theta err | wall-height err (dx) | U late |\n|---|---|---:|---:|---:|---:|\n")
    for r in r_rows:
        md.append(f"| {r['model']} | {r['reinit_mode']} | {r['n_reinit']} | {r['theta_err_deg']:+.2f} | "
                  f"{r['wall_height_err_dx']:+.2e} | {r['U_late']:.1e} |\n")
    if r_log:
        md.append("\nper-event (first 6):\n\n| model | step | theta change (deg) | contact change (dx) | "
                  "kappa wall err before -> after | shift (dx) |\n|---|---:|---:|---:|---|---:|\n")
        for e in r_log[:6]:
            md.append(f"| {e['model']} | {e['step']} | {e['theta_change_deg']:+.3f} | {e['z_contact_change_dx']:+.3f} | "
                      f"{e['kappa_wall_err_before']:.1e} -> {e['kappa_wall_err_after']:.1e} | {e['shift_dx']:.1e} |\n")
    (out / "summary.md").write_text("".join(md), encoding="utf-8")
    print("".join(md))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
