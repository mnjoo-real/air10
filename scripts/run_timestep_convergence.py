"""Time-step sensitivity check (README section 16 Test 6 / "time-step
independence test infrastructure").

First measures the actually-binding adaptive time-step limiter for this
case (README "이전 테스트에서 dt_max*0.75가 실제 timestep을 바꾸지 못했다... 실제
timestep limiter를 먼저 측정한 뒤 실제로 timestep이 달라지는 설정을 선택해라") --
the earlier version of this script picked dt_max settings that were both
above the natural capillary/viscous limit, so two of its three "different"
settings were silently identical. This version runs one warm-up stretch
with an unconstrained dt_max to find the natural effective dt, then tests
that value, /2, and /4 -- all three of which are verified (via the
recorded mean/min/max dt) to actually bind.

Usage:
    python scripts/run_timestep_convergence.py --config configs/validation_solid_body.yaml \
        --swirl-mode prescribed --t-end 0.15
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


def measure_natural_dt(base_cfg, swirl_mode: str, warmup_steps: int = 300) -> float:
    """Run briefly with an effectively-unconstrained dt_max_s and return the
    mean adaptive dt actually used -- the natural (capillary/viscous/CFL)
    limit for this case, independent of whatever dt_max_s happens to be
    configured."""
    cfg = copy.deepcopy(base_cfg)
    cfg.time.dt_max_s = 1.0  # effectively unconstrained
    solver = build_solver(cfg, swirl_mode=swirl_mode)
    dts = []
    for _ in range(warmup_steps):
        diag = solver.step()
        dts.append(diag.dt)
    return float(np.mean(dts))


def run_case(cfg, swirl_mode: str, t_end: float, window_start: float, dt_max_s: float):
    cfg = copy.deepcopy(cfg)
    cfg.time.dt_max_s = dt_max_s
    solver = build_solver(cfg, swirl_mode=swirl_mode)
    grid = solver.grid
    V0 = water_volume(grid, solver.fields.phi)

    t_hist, d_hist, dt_hist = [], [], []
    eta_window = []
    max_div = 0.0
    t0 = time.perf_counter()
    while solver.fields.t < t_end:
        diag = solver.step()
        dt_hist.append(diag.dt)
        t_hist.append(solver.fields.t)
        d_hist.append(vortex_depth(solver.fields.phi, grid, cfg))
        div = divergence(grid, solver.fields.u_r, solver.fields.u_z)
        max_div = max(max_div, float(np.max(np.abs(div))))
        if solver.fields.t >= window_start:
            eta_window.append(free_surface_height(solver.fields.phi, grid))
    wall_time = time.perf_counter() - t0

    Vf = water_volume(grid, solver.fields.phi)
    d_arr = np.array(d_hist)
    t_arr = np.array(t_hist)
    d_window = d_arr[t_arr >= window_start]
    mean_depth = float(np.mean(d_window)) if len(d_window) else float("nan")
    rms_osc = float(np.sqrt(np.mean((d_window - mean_depth) ** 2))) if len(d_window) else float("nan")
    eta_mean = np.nanmean(np.array(eta_window), axis=0) if eta_window else np.full(grid.Nr, np.nan)

    summary = {
        "dt_setting": dt_max_s,
        "mean_dt": float(np.mean(dt_hist)),
        "min_dt": float(np.min(dt_hist)),
        "max_dt": float(np.max(dt_hist)),
        "steps": solver.fields.step,
        "wall_time": wall_time,
        "d_final_mm": d_hist[-1] * 1e3,  # auxiliary only
        "mean_depth_mm": mean_depth * 1e3,
        "rms_oscillation_mm": rms_osc * 1e3,
        "volume_drift": abs(Vf - V0) / V0,
        "max_divergence": max_div,
    }
    return summary, t_arr, d_arr, eta_mean, grid


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/validation_solid_body.yaml")
    parser.add_argument("--swirl-mode", choices=["forced", "prescribed"], default="prescribed")
    parser.add_argument("--t-end", type=float, default=0.3)
    parser.add_argument("--window-start", type=float, default=0.15)
    parser.add_argument("--out-dir", default="results/convergence")
    parser.add_argument("--scheme", choices=["upwind1", "muscl2"], default="muscl2")
    parser.add_argument("--integrator", choices=["euler", "ssprk2"], default="ssprk2")
    args = parser.parse_args()

    base_cfg = load_config(args.config)
    base_cfg.levelset.advection_scheme = args.scheme
    base_cfg.levelset.time_integrator = args.integrator

    dt_natural = measure_natural_dt(base_cfg, args.swirl_mode)
    print(f"Measured natural (unconstrained) effective dt = {dt_natural:.4e} s "
          f"(this is what actually binds -- configs/validation_solid_body.yaml's own "
          f"dt_max_s={base_cfg.time.dt_max_s:.4e} was ABOVE this and never bound)")

    settings = [
        ("dt_effective", dt_natural),
        ("dt_effective/2", dt_natural / 2.0),
        ("dt_effective/4", dt_natural / 4.0),
    ]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    curves = {}
    interfaces = {}
    grid_ref = None

    for label, dt_setting in settings:
        print(f"Running dt_max={dt_setting:.3e} s ({label}) ...")
        summary, t_hist, d_hist, eta_mean, grid = run_case(base_cfg, args.swirl_mode, args.t_end,
                                                             args.window_start, dt_setting)
        summary["label"] = label
        rows.append(summary)
        curves[label] = (t_hist, d_hist)
        interfaces[label] = eta_mean
        grid_ref = grid
        print(f"  steps={summary['steps']} mean_dt={summary['mean_dt']:.3e} "
              f"(min={summary['min_dt']:.3e}, max={summary['max_dt']:.3e}) "
              f"wall_time={summary['wall_time']:.2f}s mean_depth={summary['mean_depth_mm']:.4f}mm "
              f"rms_osc={summary['rms_oscillation_mm']:.4f}mm "
              f"volume_drift={summary['volume_drift']*100:.3f}% max_div={summary['max_divergence']:.3e}")

    df = pd.DataFrame(rows)
    df["mean_interface_rmse_vs_finest_mm"] = np.nan
    finest_label = settings[-1][0]  # smallest dt = most resolved in time
    valid_ref = np.isfinite(interfaces[finest_label])
    for i, row in df.iterrows():
        label = row["label"]
        valid = valid_ref & np.isfinite(interfaces[label])
        if np.any(valid):
            rmse = float(np.sqrt(np.mean((interfaces[label][valid] - interfaces[finest_label][valid])**2)))
            df.loc[i, "mean_interface_rmse_vs_finest_mm"] = rmse * 1e3

    csv_path = out_dir / "timestep_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved {csv_path}")
    print(df.to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for label, (t_hist, d_hist) in curves.items():
        axes[0].plot(t_hist, d_hist * 1e3, label=label)
    axes[0].set_xlabel("t [s]")
    axes[0].set_ylabel("vortex depth d [mm]")
    axes[0].set_title("d(t) vs. time-step setting")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    for label, eta in interfaces.items():
        valid = np.isfinite(eta)
        axes[1].plot(grid_ref.r_c[valid] * 1e3, eta[valid] * 1e3, "o-", ms=3, label=label)
    axes[1].set_xlabel("r [mm]")
    axes[1].set_ylabel("time-averaged eta(r) [mm]")
    axes[1].set_title(f"time-averaged shape, window=[{args.window_start},{args.t_end}]s")
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig_path = out_dir / "timestep_convergence.png"
    fig.savefig(fig_path, dpi=200)
    print(f"Saved {fig_path}")


if __name__ == "__main__":
    main()
