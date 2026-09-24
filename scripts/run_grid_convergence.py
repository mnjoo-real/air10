"""Grid convergence check (README section 15 / "grid convergence 재평가"),
using WINDOWED/time-averaged metrics rather than a single instantaneous
snapshot -- an instant-in-time comparison is confounded by which phase of
the persistent free-surface oscillation (README section 47.3) each
resolution happens to land on at exactly t_end, which showed up as
non-monotonic d_final/volume_drift in the previous version of this script.

For each dx, runs to t_end and reports, over the window [window_start, t_end]:
    mean depth       = mean_t d(t)
    RMS oscillation   = sqrt(mean_t (d(t) - mean_depth)^2)
    mean-interface RMSE (vs. the finest grid's own time-averaged eta(r))
    volume drift (final, still a single instantaneous but well-defined value)
instantaneous d_final is kept as an auxiliary column only, not the metric
grid convergence is judged on.

Usage:
    python scripts/run_grid_convergence.py --config configs/validation_solid_body.yaml \
        --swirl-mode prescribed --t-end 0.6 --window-start 0.2 --dx-mm 3.0 1.5 0.75
"""
from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from air_vortex.config import load_config  # noqa: E402
from air_vortex.diagnostics import free_surface_height, vortex_depth  # noqa: E402
from air_vortex.operators import divergence  # noqa: E402
from air_vortex.solver import build_solver, water_volume  # noqa: E402


def run_case(cfg, swirl_mode: str, t_end: float, window_start: float, dx_m: float,
             scheme: str, integrator: str, limiter: str):
    cfg = copy.deepcopy(cfg)
    cfg.grid.dr_m = dx_m
    cfg.grid.dz_m = dx_m
    cfg.levelset.advection_scheme = scheme
    cfg.levelset.time_integrator = integrator
    cfg.levelset.limiter = limiter
    solver = build_solver(cfg, swirl_mode=swirl_mode)
    grid = solver.grid
    V0 = water_volume(grid, solver.fields.phi)

    d_hist, eta_window = [], []
    max_div = 0.0
    t0 = time.perf_counter()
    while solver.fields.t < t_end:
        solver.step()
        d_hist.append(vortex_depth(solver.fields.phi, grid, cfg))
        div = divergence(grid, solver.fields.u_r, solver.fields.u_z)
        max_div = max(max_div, float(np.max(np.abs(div))))
        if solver.fields.t >= window_start:
            eta_window.append(free_surface_height(solver.fields.phi, grid))
    wall_time = time.perf_counter() - t0

    Vf = water_volume(grid, solver.fields.phi)
    d_final = vortex_depth(solver.fields.phi, grid, cfg)

    d_arr = np.array(d_hist)
    tail_frac = max(1, int(len(d_arr) * (1 - window_start / t_end)))
    d_window = d_arr[-tail_frac:]
    mean_depth = float(np.mean(d_window))
    rms_osc = float(np.sqrt(np.mean((d_window - mean_depth) ** 2)))

    eta_mean = np.nanmean(np.array(eta_window), axis=0)

    summary = {
        "dx_mm": dx_m * 1e3, "Nr": grid.Nr, "Nz": grid.Nz, "cells": grid.Nr * grid.Nz,
        "steps": solver.fields.step, "wall_time": wall_time,
        "d_final_mm": d_final * 1e3,  # auxiliary only, not the convergence judgment
        "mean_depth_mm": mean_depth * 1e3,
        "rms_oscillation_mm": rms_osc * 1e3,
        "volume_drift": abs(Vf - V0) / V0,
        "max_divergence": max_div,
    }
    return summary, eta_mean, grid


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/validation_solid_body.yaml")
    parser.add_argument("--swirl-mode", choices=["forced", "prescribed"], default="prescribed")
    parser.add_argument("--t-end", type=float, default=0.6)
    parser.add_argument("--window-start", type=float, default=0.2)
    parser.add_argument("--dx-mm", type=float, nargs="+", default=[3.0, 1.5, 0.75],
                         help="Coarse-to-fine dx list in mm (default 3.0/1.5/0.75mm, i.e. "
                              "2x/1x/0.5x the validation geometry's baseline dx=1.5mm -- "
                              "see README section 15 for why 1.0/0.5/0.25mm on the *physical* "
                              "baseline geometry is too slow to run here; this uses the "
                              "small validation geometry instead, scaled the same 2x/1x/0.5x way.")
    parser.add_argument("--out-dir", default="results/convergence")
    parser.add_argument("--scheme", choices=["upwind1", "muscl2"], default="muscl2")
    parser.add_argument("--integrator", choices=["euler", "ssprk2"], default="ssprk2")
    parser.add_argument("--limiter", choices=["mc", "minmod"], default="mc")
    args = parser.parse_args()

    base_cfg = load_config(args.config)
    dx_list_m = sorted((x * 1e-3 for x in args.dx_mm), reverse=True)  # coarse -> fine

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, interfaces, grids = [], {}, {}
    for dx_m in dx_list_m:
        print(f"Running dx={dx_m*1e3:.4f} mm ({args.scheme}+{args.integrator}) ...")
        summary, eta_mean, grid = run_case(base_cfg, args.swirl_mode, args.t_end, args.window_start,
                                            dx_m, args.scheme, args.integrator, args.limiter)
        rows.append(summary)
        interfaces[dx_m] = eta_mean
        grids[dx_m] = grid
        print(f"  cells={summary['cells']} steps={summary['steps']} "
              f"wall_time={summary['wall_time']:.2f}s mean_depth={summary['mean_depth_mm']:.4f}mm "
              f"rms_osc={summary['rms_oscillation_mm']:.4f}mm "
              f"volume_drift={summary['volume_drift']*100:.3f}% max_div={summary['max_divergence']:.3e}")

    # mean-interface RMSE of each grid's time-averaged eta vs. the finest
    # grid's, interpolated onto each coarser grid's own r_c points
    fine_dx = dx_list_m[-1]
    fine_grid, fine_eta = grids[fine_dx], interfaces[fine_dx]
    fine_valid = np.isfinite(fine_eta)
    for row, dx_m in zip(rows, dx_list_m):
        grid = grids[dx_m]
        eta = interfaces[dx_m]
        valid = np.isfinite(eta)
        if dx_m == fine_dx or not np.any(valid):
            row["mean_interface_rmse_vs_finest_mm"] = 0.0 if dx_m == fine_dx else np.nan
            continue
        eta_fine_interp = np.interp(grid.r_c[valid], fine_grid.r_c[fine_valid], fine_eta[fine_valid])
        row["mean_interface_rmse_vs_finest_mm"] = float(np.sqrt(np.mean((eta[valid] - eta_fine_interp) ** 2))) * 1e3

    df = pd.DataFrame(rows).sort_values("dx_mm", ascending=False).reset_index(drop=True)

    if len(df) >= 2:
        fine = df.iloc[-1]
        medium = df.iloc[-2]
        rel_change_depth = abs(fine["mean_depth_mm"] - medium["mean_depth_mm"]) / max(abs(fine["mean_depth_mm"]), 1e-12)
        rel_change_volume = abs(fine["volume_drift"] - medium["volume_drift"]) / max(abs(fine["volume_drift"]), 1e-12)
        print(f"\nmedium ({medium['dx_mm']}mm) vs fine ({fine['dx_mm']}mm):")
        print(f"  relative_change(mean_depth)   = {rel_change_depth:.4f}")
        print(f"  relative_change(volume_drift) = {rel_change_volume:.4f}")

    csv_path = out_dir / "grid_convergence_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved {csv_path}")
    print(df.to_string(index=False))

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for dx_m, eta in interfaces.items():
        grid = grids[dx_m]
        valid = np.isfinite(eta)
        ax.plot(grid.r_c[valid] * 1e3, eta[valid] * 1e3, "o-", ms=3, label=f"dx={dx_m*1e3:.3f}mm")
    ax.set_xlabel("r [mm]")
    ax.set_ylabel(f"time-averaged eta(r) [mm], window=[{args.window_start},{args.t_end}]s")
    ax.set_title("grid convergence: time-averaged free-surface shape")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig_path = out_dir / "grid_convergence.png"
    fig.savefig(fig_path, dpi=200)
    print(f"Saved {fig_path}")


if __name__ == "__main__":
    main()
