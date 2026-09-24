"""Long-duration solid-body-rotation free-surface validation (README
section 31 Test 2), comparing four Level Set scheme/correction combinations
(README "Level Set advection accuracy" Case A-D) so the effect of advection
accuracy and of volume correction can be told apart -- neither is reported
in isolation, and volume-correction-ON results are never used to claim the
underlying advection accuracy improved.

    Case A: upwind1 + euler,  volume correction OFF  (original scheme)
    Case B: muscl2  + ssprk2, volume correction OFF  (accuracy improvement only)
    Case C: upwind1 + euler,  volume correction ON
    Case D: muscl2  + ssprk2, volume correction ON

The analytic comparison target is the VOLUME-CONSISTENT parabola (README
"solid-body analytical free surface의 volume-consistent form"): the vertical
offset constant is chosen so the analytic paraboloid encloses exactly the
INITIAL water volume, not fit to whatever the simulation's own (possibly
volume-drifted) center height happens to be -- so a case with real volume
drift is not let off the hook by silently comparing itself to a shifted
target.

Usage:
    python scripts/validate_solid_body_long.py --config configs/validation_solid_body.yaml \\
        --t-end 0.4 --window-start 0.15
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
from air_vortex.diagnostics import free_surface_height, volume_consistent_parabola  # noqa: E402
from air_vortex.operators import divergence  # noqa: E402
from air_vortex.plotting import plot_center_depression_convergence, plot_freesurface_timeaverage  # noqa: E402
from air_vortex.run_io import SnapshotWriter, grid_metadata, run_paths, save_config, save_metadata  # noqa: E402
from air_vortex.solver import build_solver, water_volume  # noqa: E402
from air_vortex.viz_common import save_figure  # noqa: E402

CASES = {
    "A": dict(scheme="upwind1", integrator="euler", correction=False,
              label="upwind1+Euler, correction OFF"),
    "B": dict(scheme="muscl2", integrator="ssprk2", correction=False,
              label="MUSCL2+SSPRK2, correction OFF"),
    "C": dict(scheme="upwind1", integrator="euler", correction=True,
              label="upwind1+Euler, correction ON"),
    "D": dict(scheme="muscl2", integrator="ssprk2", correction=True,
              label="MUSCL2+SSPRK2, correction ON"),
}


def run_case(base_cfg, case_key: str, t_end: float, window_start: float):
    spec = CASES[case_key]
    cfg = copy.deepcopy(base_cfg)
    cfg.levelset.advection_scheme = spec["scheme"]
    cfg.levelset.time_integrator = spec["integrator"]
    cfg.levelset.volume_correction.enabled = spec["correction"]

    solver = build_solver(cfg, swirl_mode="prescribed")
    grid = solver.grid
    g = cfg.fluid.gravity
    R_v = grid.r_v
    V0 = water_volume(grid, solver.fields.phi)
    omega_final = cfg.stirrer.omega(t_end)

    t_hist, eta0_hist, vol_err_hist, rmse_hist, div_hist, maxu_hist = [], [], [], [], [], []
    eta_window_samples = []

    t0 = time.perf_counter()
    while solver.fields.t < t_end:
        solver.step()
        t = solver.fields.t
        phi = solver.fields.phi

        eta = free_surface_height(phi, grid)
        eta_analytic = volume_consistent_parabola(grid.r_c, R_v, omega_final, g, V0)
        valid = np.isfinite(eta)

        t_hist.append(t)
        eta0_hist.append(eta[0] if np.isfinite(eta[0]) else np.nan)
        Vt = water_volume(grid, phi)
        vol_err_hist.append((Vt - V0) / V0)
        if np.any(valid):
            rmse_hist.append(float(np.sqrt(np.mean((eta[valid] - eta_analytic[valid]) ** 2))))
        else:
            rmse_hist.append(np.nan)
        div = divergence(grid, solver.fields.u_r, solver.fields.u_z)
        div_hist.append(float(np.max(np.abs(div))))
        maxu_hist.append(max(float(np.max(np.abs(solver.fields.u_r))),
                              float(np.max(np.abs(solver.fields.u_z)))))

        if t >= window_start:
            eta_window_samples.append(eta)

    wall_time = time.perf_counter() - t0

    eta_window = np.array(eta_window_samples)
    eta_mean = np.nanmean(eta_window, axis=0)
    eta_std = np.nanstd(eta_window, axis=0)
    eta_analytic_final = volume_consistent_parabola(grid.r_c, R_v, omega_final, g, V0)
    valid = np.isfinite(eta_mean)

    err = eta_mean[valid] - eta_analytic_final[valid]
    rmse = float(np.sqrt(np.mean(err**2)))
    max_err = float(np.max(np.abs(err)))
    amplitude = omega_final**2 * R_v**2 / (2.0 * g)
    nrmse = rmse / amplitude if amplitude > 0 else float("nan")

    C_analytic = eta_analytic_final[0]
    center_depression_error = float(abs((cfg.geometry.water_height_m - eta_mean[0])
                                         - (cfg.geometry.water_height_m - C_analytic)))

    summary = {
        "case": case_key, "label": spec["label"],
        "scheme": spec["scheme"], "integrator": spec["integrator"], "correction": spec["correction"],
        "rmse_mm": rmse * 1e3, "nrmse": nrmse, "max_error_mm": max_err * 1e3,
        "center_depression_error_mm": center_depression_error * 1e3,
        "volume_drift_final": vol_err_hist[-1],
        "max_volume_drift": float(np.max(np.abs(vol_err_hist))),
        "max_divergence": max(div_hist),
        "max_velocity": max(maxu_hist),
        "t_end_s": t_end, "wall_time_s": wall_time, "steps": solver.fields.step,
        "cumulative_volume_correction": solver.cumulative_volume_correction,
    }

    series = pd.DataFrame({
        "t": t_hist, "eta0_mm": np.array(eta0_hist) * 1e3,
        "volume_error": vol_err_hist, "interface_rmse_mm": np.array(rmse_hist) * 1e3,
        "max_divergence": div_hist, "max_velocity": maxu_hist,
    })

    return summary, series, eta_mean, eta_std, grid, omega_final, V0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/validation_solid_body.yaml")
    parser.add_argument("--t-end", type=float, default=0.4)
    parser.add_argument("--window-start", type=float, default=0.15)
    parser.add_argument("--cases", nargs="+", default=list(CASES.keys()), choices=list(CASES.keys()))
    parser.add_argument("--run-id", default="validation_solid_body_long")
    parser.add_argument("--results-root", default="results")
    args = parser.parse_args()

    base_cfg = load_config(args.config)
    paths = run_paths(args.results_root, args.run_id)
    save_config(paths, base_cfg)

    all_summaries = []
    all_series = {}
    fig_ts, axes_ts = plt.subplots(1, 3, figsize=(15, 4.5))

    for case_key in args.cases:
        print(f"\n=== Case {case_key}: {CASES[case_key]['label']} ===")
        summary, series, eta_mean, eta_std, grid, omega_final, V0 = run_case(
            base_cfg, case_key, args.t_end, args.window_start)
        all_summaries.append(summary)
        all_series[case_key] = series

        print(f"  steps={summary['steps']} wall_time={summary['wall_time_s']:.2f}s")
        print(f"  RMSE={summary['rmse_mm']:.4f}mm  NRMSE={summary['nrmse']:.4f}  "
              f"max_err={summary['max_error_mm']:.4f}mm")
        print(f"  volume_drift_final={summary['volume_drift_final']*100:.3f}%  "
              f"max_divergence={summary['max_divergence']:.3e}")

        series.to_csv(paths.root / f"case_{case_key}_timeseries.csv", index=False)

        fig1 = plot_freesurface_timeaverage(
            grid, eta_mean, eta_std, omega_final, base_cfg.fluid.gravity,
            eta0_ref=volume_consistent_parabola(np.array([0.0]), grid.r_v, omega_final,
                                                 base_cfg.fluid.gravity, V0)[0])
        save_figure(fig1, paths, f"case_{case_key}_freesurface_timeaverage")
        plt.close(fig1)

        axes_ts[0].plot(series["t"], series["eta0_mm"], label=case_key)
        axes_ts[1].plot(series["t"], series["volume_error"] * 100, label=case_key)
        axes_ts[2].plot(series["t"], series["interface_rmse_mm"], label=case_key)

    axes_ts[0].set_xlabel("t [s]"); axes_ts[0].set_ylabel("eta(0) [mm]")
    axes_ts[0].set_title("center depression vs. time"); axes_ts[0].legend(fontsize=8)
    axes_ts[1].set_xlabel("t [s]"); axes_ts[1].set_ylabel("volume error [%]")
    axes_ts[1].set_title("volume_error(t)"); axes_ts[1].legend(fontsize=8)
    axes_ts[2].set_xlabel("t [s]"); axes_ts[2].set_ylabel("interface RMSE [mm]")
    axes_ts[2].set_title("interface_RMSE(t)"); axes_ts[2].legend(fontsize=8)
    for ax in axes_ts:
        ax.grid(True, alpha=0.3)
    fig_ts.tight_layout()
    save_figure(fig_ts, paths, "case_comparison_timeseries")

    df = pd.DataFrame(all_summaries)
    csv_path = paths.root / "case_comparison_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved {csv_path}")
    print(df.to_string(index=False))

    save_metadata(paths, {
        "kind": "solid_body_rotation_long_case_comparison",
        "swirl_mode": "prescribed", "rpm": base_cfg.stirrer.rpm,
        "t_end_s": args.t_end, "window_start_s": args.window_start,
        "cases_run": args.cases,
        "grid": grid_metadata(grid),
        "case_summaries": all_summaries,
    })


if __name__ == "__main__":
    main()
