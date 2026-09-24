"""Flat-start spin-up TRANSIENT characterization (README "Flat-start
transient는 별도 physics test로 재정의") -- explicitly NOT an equilibrium-
preservation verification test (that's
scripts/validate_exact_rotating_equilibrium.py). This one starts from
flat, still water and a sudden-to-gradual spin-up ramp, and characterizes
the resulting free-surface oscillation: dominant frequency, damping rate
(or "not resolvable"), and sensitivity to the startup ramp time -- the
physics question of whether the persistent oscillation found in the
exact-equilibrium test (README section 47.3) is a startup-excited mode
that a slower ramp would avoid, or something the ramp doesn't control.

Usage:
    python scripts/validate_spinup_transient.py --config configs/validation_solid_body.yaml \\
        --t-end 0.8 --ramp-times 0.05 0.2 0.5 1.0
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
from air_vortex.diagnostics import free_surface_height  # noqa: E402
from air_vortex.oscillation_analysis import damping_from_peaks, dominant_frequency  # noqa: E402
from air_vortex.solver import build_solver, water_volume  # noqa: E402
from air_vortex.viz_common import save_figure  # noqa: E402
from air_vortex.run_io import run_paths, save_metadata  # noqa: E402


def run_case(base_cfg, ramp_time_s: float, t_end: float):
    cfg = copy.deepcopy(base_cfg)
    cfg.stirrer.ramp_time_s = ramp_time_s
    cfg.levelset.advection_scheme = "muscl2"
    cfg.levelset.time_integrator = "ssprk2"
    # Volume correction ON here is deliberate and different from the
    # Case A-D comparison: this script's job is isolating the ramp-time ->
    # oscillation relationship, and uncorrected volume drift otherwise
    # contaminates the "time-averaged depression" / "oscillation
    # amplitude" metrics with a monotonic upward trend unrelated to
    # startup-mode physics (found when this script was first run without
    # it -- see README "startup ramp sensitivity" notes).
    cfg.levelset.volume_correction.enabled = True

    solver = build_solver(cfg, swirl_mode="prescribed")
    grid = solver.grid
    V0 = water_volume(grid, solver.fields.phi)
    H = cfg.geometry.water_height_m

    t_hist, eta0_hist = [], []
    t0 = time.perf_counter()
    while solver.fields.t < t_end:
        solver.step()
        eta = free_surface_height(solver.fields.phi, grid)
        t_hist.append(solver.fields.t)
        eta0_hist.append(H - eta[0] if np.isfinite(eta[0]) else np.nan)  # center DEPRESSION, not height
    wall_time = time.perf_counter() - t0

    t_arr = np.array(t_hist)
    d_arr = np.array(eta0_hist)

    # Exclude the initial rise/transient (dominated by the ramp itself,
    # not the free oscillation) from the frequency/damping analysis --
    # otherwise the FFT and peak-picking mix the ramp-driven rise in with
    # whatever periodic content follows it. Start the analysis window
    # after the ramp has clearly completed (ramp_time_s is when Omega(t)
    # reaches ~63% of target; use 3x that as "fully ramped").
    analysis_start = min(3.0 * ramp_time_s, 0.5 * t_end)
    mask = t_arr >= analysis_start
    t_post, d_post = t_arr[mask], d_arr[mask]

    freq_result = dominant_frequency(t_post, d_post)
    damp_result = damping_from_peaks(t_post, d_post)

    tail = d_post  # post-transient window, for amplitude/mean stats too
    summary = {
        "ramp_time_s": ramp_time_s,
        "analysis_start_s": analysis_start,
        "max_overshoot_mm": float(np.nanmax(d_arr)) * 1e3,
        "oscillation_amplitude_mm": float(np.nanmax(tail) - np.nanmin(tail)) * 1e3,
        "time_averaged_depression_mm": float(np.nanmean(tail)) * 1e3,
        "f_dominant_hz": freq_result.f_dominant,
        "period_s": freq_result.period,
        "damping_resolvable": damp_result.resolvable,
        "lambda_1_per_s": damp_result.lambda_,
        "tau_d_s": damp_result.tau_d,
        "damping_fit_r2": damp_result.fit_quality_r2,
        "wall_time_s": wall_time, "steps": solver.fields.step,
    }
    return summary, t_arr, d_arr


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/validation_solid_body.yaml")
    parser.add_argument("--t-end", type=float, default=0.8)
    parser.add_argument("--ramp-times", type=float, nargs="+", default=[0.05, 0.2, 0.5, 1.0])
    parser.add_argument("--run-id", default="validation_spinup_transient")
    parser.add_argument("--results-root", default="results")
    args = parser.parse_args()

    base_cfg = load_config(args.config)
    paths = run_paths(args.results_root, args.run_id)

    rows = []
    fig, ax = plt.subplots(figsize=(7, 5))
    for ramp in args.ramp_times:
        print(f"Running ramp_time_s={ramp} ...")
        summary, t_arr, d_arr = run_case(base_cfg, ramp, args.t_end)
        rows.append(summary)
        ax.plot(t_arr, d_arr * 1e3, label=f"t_ramp={ramp}s")

        freq_str = f"{summary['f_dominant_hz']:.2f} Hz" if summary["f_dominant_hz"] else "n/a"
        damp_str = (f"tau_d={summary['tau_d_s']:.3f}s (R2={summary['damping_fit_r2']:.2f})"
                    if summary["damping_resolvable"] else "not resolvable")
        print(f"  max_overshoot={summary['max_overshoot_mm']:.3f}mm  "
              f"osc_amp={summary['oscillation_amplitude_mm']:.3f}mm  "
              f"mean_depression={summary['time_averaged_depression_mm']:.3f}mm  "
              f"f_dominant={freq_str}  damping={damp_str}")

        pd.DataFrame({"t": t_arr, "depression_mm": d_arr * 1e3}).to_csv(
            paths.root / f"ramp_{ramp}_timeseries.csv", index=False)

    ax.set_xlabel("t [s]")
    ax.set_ylabel("center depression [mm]")
    ax.set_title("startup ramp sensitivity")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_figure(fig, paths, "ramp_sensitivity")

    df = pd.DataFrame(rows)
    csv_path = paths.root / "ramp_sensitivity_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved {csv_path}")
    print(df.to_string(index=False))

    save_metadata(paths, {"kind": "spinup_transient_ramp_sensitivity", "t_end_s": args.t_end,
                           "ramp_times_s": args.ramp_times, "summaries": rows})


if __name__ == "__main__":
    main()
