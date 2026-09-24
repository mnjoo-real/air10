"""Gate V4: sharp-interface surface tension (single_phase_ls).

Pressure-jump error and curvature error are measured SEPARATELY:

  A  curvature operator alone (no flow): K0 plane, K1 sphere, K2 cylinder,
     axis quadratic, R/dx = 4..24 at fixed physical radius
  B  CAP-A: static sphere, gravity 0, EXACT kappa = 2/R supplied to the BC
  C  CAP-B: same sphere, kappa computed numerically from phi, 1.0 s
  D  reinitialization interaction (CAP-B, R/dx = 12): OFF, RS2 quality-
     triggered, RS2 sparse periodic; every reinit event logged
  E  capillary time-step sensitivity (CAP-B, R/dx = 12)
  F  U_spurious(t), analytic vs numerical curvature (figure)

Writes results/validation_surface_tension/{metadata.json, summary.md,
u_spurious_vs_time.png}. Volume correction is OFF throughout.

Usage: python scripts/validate_surface_tension.py [--quick]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from air_vortex.capillary_benchmarks import build_sphere_solver  # noqa: E402
from air_vortex.curvature_single_phase import (  # noqa: E402
    crossing_positions, curvature_centers, interface_curvature)
from air_vortex.liquid_mask import classify, liquid_volume_subcell  # noqa: E402
from air_vortex.reinit_benchmarks import uniform_grid  # noqa: E402
from air_vortex.reinit_diagnostics import contour_displacement, signed_distance_error  # noqa: E402

RS = 0.006          # physical sphere radius for all CAP runs
SIGMA = 0.072
DP = 2 * SIGMA / RS
N_CELLS = (4, 6, 8, 12, 16, 24)


def _order(xs, errs):
    errs = np.asarray(errs, float)
    if np.any(errs <= 0) or np.any(np.diff(errs) >= 0):
        return None
    return float(np.polyfit(np.log(xs), np.log(errs), 1)[0])


def kappa_at_crossings(grid, phi):
    geom = classify(phi)
    kr, kz = interface_curvature(grid, geom, phi)
    rr, zr, rz, zz = crossing_positions(grid, geom)
    k = np.concatenate([kr[np.isfinite(kr)], kz[np.isfinite(kz)]])
    r = np.concatenate([rr[np.isfinite(kr)], rz[np.isfinite(kz)]])
    return k, r


def sphere_kappa_error(grid, phi, radius=RS):
    k, r = kappa_at_crossings(grid, phi)
    e = k * radius / 2 - 1
    return {"mean": float(np.mean(e)), "rms": float(np.sqrt(np.mean(e**2))),
            "max": float(np.max(np.abs(e))),
            "axis_max": float(np.max(np.abs(e[r < grid.dr]))) if np.any(r < grid.dr) else None,
            "offaxis_max": float(np.max(np.abs(e[r >= grid.dr])))}


# ------------------------------------------------------------------ A
def section_a():
    out = {"K0": [], "K1": [], "K2": [], "axis_quadratic": None}
    g = uniform_grid(1e-3)
    for z0 in (0.0113, 0.01537, 0.0159):
        k, _ = kappa_at_crossings(g, g.z_c[None, :] - z0 + 0 * g.r_c[:, None])
        out["K0"].append({"z0_over_dx": z0 / g.dz, "max_abs_kappa": float(np.abs(k).max())})
    R, Z = np.meshgrid(g.r_c, g.z_c, indexing="ij")
    a = 30.0
    kc = curvature_centers(g, Z - 0.01 - a * R**2)
    gr = -2 * a * g.r_c[:, None]
    exact = -2 * a / (1 + gr**2) ** 1.5 + gr / (g.r_c[:, None] * np.sqrt(1 + gr**2))
    err = np.abs(kc - exact)[:-1, 1:-1]        # wall column uses the extrapolation ghost
    out["axis_quadratic"] = {"max_abs_err_interior": float(err.max()),
                             "max_abs_err_first_column": float(err[0].max()),
                             "kappa_first_column": float(kc[0, 10]), "minus_4a": -4 * a}
    for n in N_CELLS:
        dx = RS / n
        g = uniform_grid(dx, R=2 * RS, Z=4 * RS)
        R, Z = np.meshgrid(g.r_c, g.z_c, indexing="ij")
        z0 = 0.5 * g.z_max + 0.37 * g.dz
        out["K1"].append({"R_over_dx": n, "dx_mm": dx * 1e3,
                          **sphere_kappa_error(g, np.hypot(R, Z - z0) - RS)})
        k, _ = kappa_at_crossings(g, R - RS * 1.0)
        e = k * RS - 1
        out["K2"].append({"R_over_dx": n, "mean": float(np.mean(e)),
                          "rms": float(np.sqrt(np.mean(e**2))), "max": float(np.abs(e).max())})
    for key in ("K1", "K2"):
        rows = out[key]
        out[key + "_order_rms"] = _order([r["R_over_dx"] ** -1.0 for r in rows], [r["rms"] for r in rows])
        out[key + "_order_max"] = _order([r["R_over_dx"] ** -1.0 for r in rows], [r["max"] for r in rows])
    return out


# ------------------------------------------------------------- runners
def run_sphere(n, curvature, t_end, record=False, reinit=None, dt_factor=1.0, n_steps=None,
               extension_layers_capillary=None):
    kw = {"capillary_dt_factor": dt_factor}
    if reinit:
        kw["reinit_every"] = reinit["every"]
    s, _ = build_sphere_solver(RS / n, RS, curvature=curvature, **kw)
    if extension_layers_capillary:
        s.cfg.physics.extension_layers_capillary = extension_layers_capillary
    events = []
    if reinit:
        ls = s.cfg.levelset
        ls.reinitialization_method = "russo_smereka_subcell"
        ls.reinitialize_iterations = reinit.get("iters", 2)
        ls.reinitialization_trigger = reinit.get("trigger", "periodic")
        ls.reinitialization_quality_threshold = reinit.get("threshold", 0.05)

        def cb(before, after, step):
            kb, ka = sphere_kappa_error(s.grid, before), sphere_kappa_error(s.grid, after)
            pgb = SIGMA * kappa_at_crossings(s.grid, before)[0]
            pga = SIGMA * kappa_at_crossings(s.grid, after)[0]
            events.append({"step": step, "shift_dx": contour_displacement(s.grid, before, after).max_shift_dx,
                           "E_sd_before": signed_distance_error(s.grid, before),
                           "E_sd_after": signed_distance_error(s.grid, after),
                           "kappa_rms_before": kb["rms"], "kappa_rms_after": ka["rms"],
                           "kappa_max_before": kb["max"], "kappa_max_after": ka["max"],
                           "pGamma_std_before": float(np.std(pgb)), "pGamma_std_after": float(np.std(pga))})
        s.reinit_callback = cb
    g = s.grid
    phi0 = s.fields.phi.copy()
    V0 = liquid_volume_subcell(g, phi0)
    k0 = sphere_kappa_error(g, phi0)
    ts, us, ps = [], [], []
    t0 = time.perf_counter()
    k = 0
    while (n_steps is None and s.fields.t < t_end) or (n_steps is not None and k < n_steps):
        d = s.step()
        k += 1
        u = max(d.max_abs_ur_liquid, d.max_abs_uz_liquid)
        ts.append(d.t)
        us.append(u)
        ps.append(d.p_gamma_std)
    liq = s.fields.phi < 0
    us = np.array(us)
    m = len(us)
    res = {"R_over_dx": n, "dx_mm": RS / n * 1e3, "curvature": curvature, "steps": m, "dt": d.dt,
           "t_end": s.fields.t, "kappa_rms_t0": k0["rms"], "kappa_max_t0": k0["max"],
           "kappa_axis_t0": k0["axis_max"],
           **{f"kappa_{a}_end": b for a, b in sphere_kappa_error(g, s.fields.phi).items()},
           "dp_measured": float(s.fields.p[liq].mean()), "dp_expected": DP,
           "dp_rel_err": float(s.fields.p[liq].mean() / DP - 1),
           "p_liquid_spread": float(np.ptp(s.fields.p[liq])),
           "pGamma_std_rel_mean": float(np.mean(ps) / DP),
           "U_first": float(us[0]), "U_peak": float(us.max()), "U_late": float(us[3 * m // 4:].max()),
           "max_div": d.max_divergence_liquid,
           "volume_drift": liquid_volume_subcell(g, s.fields.phi) / V0 - 1,
           "E_sd_end": signed_distance_error(g, s.fields.phi),
           "interface_shift_dx": contour_displacement(g, phi0, s.fields.phi).max_shift_dx,
           "n_reinit": d.n_reinit_calls, "wall_s": time.perf_counter() - t0}
    if reinit:
        res["reinit_events"] = events
    if record:
        res["series"] = {"t": ts, "U": us.tolist()}
    return res


def plot_series(path, analytic, numerical, n):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, muted, grid_c = "#0b0b0b", "#52514e", "#e4e3df"
    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    floor = 1e-18
    for res, color, label in ((numerical, "#eb6834", "numerical curvature (CAP-B)"),
                              (analytic, "#2a78d6", "exact curvature (CAP-A)")):
        t = np.array(res["series"]["t"])
        u = np.maximum(np.array(res["series"]["U"]), floor)
        ax.plot(t, u, color=color, lw=2, label=label)
        ax.annotate(label.split(" (")[0], (t[-1], u[-1]), xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=8, color=muted)
    ax.set_yscale("log")
    ax.set_xlabel("time (s)", color=muted, fontsize=9)
    ax.set_ylabel("max |u| in liquid (m/s)", color=muted, fontsize=9)
    ax.set_title(f"Static drop, R = 6 mm, R/dx = {n}, sigma = 0.072 N/m, gravity 0",
                 loc="left", fontsize=10, color=ink)
    ax.grid(True, which="major", color=grid_c, lw=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(grid_c)
    ax.tick_params(colors=muted, labelsize=8)
    ax.legend(frameon=False, fontsize=8, loc="center right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--t-end", type=float, default=1.0)
    ap.add_argument("--results-root", default=str(ROOT / "results"))
    args = ap.parse_args()
    T = 0.2 if args.quick else args.t_end
    cells = (8, 12) if args.quick else N_CELLS
    out_dir = Path(args.results_root) / "validation_surface_tension"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("A curvature", flush=True)
    a = section_a()

    print("B CAP-A", flush=True)
    b = []
    for n in (8, 12, 16, 24):
        for steps in (1, 10, 100):
            b.append(run_sphere(n, "analytic", None, n_steps=steps))
    b.append(run_sphere(12, "analytic", T, record=True))
    cap_a_series = b[-1]

    print("C CAP-B", flush=True)
    c = []
    for n in cells:
        r = run_sphere(n, "numerical", T, record=(n == 12))
        c.append(r)
        print(f"  R/dx={n}: krms={r['kappa_rms_end']:.1e} dp={r['dp_rel_err']:+.1e} "
              f"Ulate={r['U_late']:.1e} dV={r['volume_drift']:+.1e}", flush=True)
    cap_b_series = next(r for r in c if r["R_over_dx"] == 12)
    c_3layer = run_sphere(8, "numerical", T, extension_layers_capillary=3)

    print("D reinit", flush=True)
    d = [dict(run_sphere(12, "numerical", T), mode="OFF"),
         dict(run_sphere(12, "numerical", T, reinit={"every": 1, "trigger": "quality", "threshold": 0.01}),
              mode="RS2 quality-triggered (E_sd > 0.01, checked every step)"),
         dict(run_sphere(12, "numerical", T, reinit={"every": 50, "iters": 2}),
              mode="RS2 periodic every 50 steps")]

    print("E dt", flush=True)
    e = [dict(run_sphere(12, "numerical", min(T, 0.5), dt_factor=f), dt_factor=f) for f in (1.0, 0.5, 0.25)]

    plot_series(out_dir / "u_spurious_vs_time.png", cap_a_series, cap_b_series, 12)
    meta = {"kind": "surface_tension_v4", "A": a, "B": b, "C": c, "C_3layer": c_3layer, "D": d, "E": e}
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2, default=float), encoding="utf-8")

    f = lambda v: "n/a" if v is None else f"{v:.2e}"  # noqa: E731
    md = [f"# V4 surface tension (sigma={SIGMA}, R={RS*1e3} mm, T={T} s, volume correction OFF)\n\n",
          "## A curvature (relative errors at the phi=0 crossings)\n\n",
          "| geometry | R/dx | mean error | RMS rel error | max rel error | axis max | off-axis max |\n",
          "|---|---:|---:|---:|---:|---:|---:|\n"]
    for r in a["K1"]:
        md.append(f"| sphere | {r['R_over_dx']} | {r['mean']:+.2e} | {r['rms']:.2e} | {r['max']:.2e} | "
                  f"{f(r['axis_max'])} | {r['offaxis_max']:.2e} |\n")
    for r in a["K2"]:
        md.append(f"| cylinder | {r['R_over_dx']} | {r['mean']:+.2e} | {r['rms']:.2e} | {r['max']:.2e} | - | - |\n")
    md.append(f"\nobserved order vs dx: sphere RMS {f(a['K1_order_rms'])}, max {f(a['K1_order_max'])}; "
              f"cylinder RMS {f(a['K2_order_rms'])}, max {f(a['K2_order_max'])}\n\n")
    md.append(f"K0 plane max |kappa|: {[r['max_abs_kappa'] for r in a['K0']]}\n\n")
    md.append(f"axis quadratic phi = z - a r^2: {a['axis_quadratic']}\n\n")
    md.append("## B CAP-A (exact kappa)\n\n| R/dx | steps | expected dp | measured dp | rel error | "
              "p spread in liquid | U | max div |\n|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for r in b:
        md.append(f"| {r['R_over_dx']} | {r['steps']} | {r['dp_expected']:.4f} | {r['dp_measured']:.6f} | "
                  f"{r['dp_rel_err']:+.1e} | {r['p_liquid_spread']:.1e} | {r['U_late']:.1e} | {r['max_div']:.1e} |\n")
    md.append("\n## C CAP-B (numerical kappa)\n\n| R/dx | kappa RMS t0 | kappa RMS end | kappa max end | "
              "dp error | std(p_Gamma)/dp | U first | U peak | U late | volume drift | E_sd end |\n"
              "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for r in c + [dict(c_3layer, R_over_dx="8 (3-layer ext.)")]:
        md.append(f"| {r['R_over_dx']} | {r['kappa_rms_t0']:.2e} | {r['kappa_rms_end']:.2e} | "
                  f"{r['kappa_max_end']:.2e} | {r['dp_rel_err']:+.2e} | {r['pGamma_std_rel_mean']:.1e} | "
                  f"{r['U_first']:.1e} | {r['U_peak']:.1e} | {r['U_late']:.1e} | {r['volume_drift']:+.1e} | "
                  f"{r['E_sd_end']:.1e} |\n")
    md.append("\n## D reinit interaction (R/dx = 12)\n\n| mode | n reinit | interface shift (dx) | "
              "kappa RMS end | kappa max end | U late | volume drift |\n|---|---:|---:|---:|---:|---:|---:|\n")
    for r in d:
        md.append(f"| {r['mode']} | {r['n_reinit']} | {r['interface_shift_dx']:.2e} | {r['kappa_rms_end']:.2e} | "
                  f"{r['kappa_max_end']:.2e} | {r['U_late']:.1e} | {r['volume_drift']:+.1e} |\n")
    for r in d:
        ev = r.get("reinit_events") or []
        if ev:
            md.append(f"\nreinit events for '{r['mode']}' (first 5 of {len(ev)}):\n\n| step | shift dx | E_sd before->after | "
                      "kappa RMS before->after | kappa max before->after | std p_Gamma before->after |\n"
                      "|---:|---:|---|---|---|---|\n")
            for x in ev[:5]:
                md.append(f"| {x['step']} | {x['shift_dx']:.1e} | {x['E_sd_before']:.1e} -> {x['E_sd_after']:.1e} | "
                          f"{x['kappa_rms_before']:.1e} -> {x['kappa_rms_after']:.1e} | "
                          f"{x['kappa_max_before']:.1e} -> {x['kappa_max_after']:.1e} | "
                          f"{x['pGamma_std_before']:.2e} -> {x['pGamma_std_after']:.2e} |\n")
    md.append("\n## E timestep sensitivity (R/dx = 12)\n\n| dt factor | dt | dp error | U late | volume drift | "
              "kappa RMS end |\n|---:|---:|---:|---:|---:|---:|\n")
    for r in e:
        md.append(f"| {r['dt_factor']} | {r['dt']:.2e} | {r['dp_rel_err']:+.2e} | {r['U_late']:.2e} | "
                  f"{r['volume_drift']:+.2e} | {r['kappa_rms_end']:.2e} |\n")
    md.append("\n![U_spurious vs time](u_spurious_vs_time.png)\n")
    (out_dir / "summary.md").write_text("".join(md), encoding="utf-8")
    print("".join(md))
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
