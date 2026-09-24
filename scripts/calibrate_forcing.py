"""Stir-bar forcing calibration framework (README section 10.2, "stir-bar
forcing calibration framework").

Finds the effective forcing timescale tau_s in

    f_theta = chi(r,z) * (Omega*r - u_theta) / tau_s

that reproduces one MEASURED vortex depth at one (actual RPM, water
height) condition, via a bracketed root search (vortex depth is monotonic
decreasing in tau_s -- README section 10.2: "smaller tau_s -> stronger
coupling -> deeper vortex").

This script never invents a measured depth -- ``--measured-depth-mm`` and
``--actual-rpm`` must be supplied by the caller from a real experiment.
Per README section 10.2 ("do not recalibrate tau_s for each RPM"), the
result from ONE calibration point is meant to be reused unchanged at other
RPM/H conditions, not refit -- this script only ever calibrates against a
single supplied point per invocation.

Usage:
    python scripts/calibrate_forcing.py \\
        --config configs/calibration.yaml \\
        --measured-depth-mm 12.4 --actual-rpm 800
"""
from __future__ import annotations

import argparse
import copy
import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

from air_vortex.config import load_config  # noqa: E402
from air_vortex.diagnostics import vortex_depth  # noqa: E402
from air_vortex.solver import build_solver  # noqa: E402
from air_vortex.steady_state import SteadyStateConfig, SteadyStateDetector  # noqa: E402


def simulate_depth(base_cfg, tau_s: float, actual_rpm: float, water_height_m: float | None,
                    t_max: float, steady_cfg: SteadyStateConfig) -> tuple[float, dict]:
    cfg = copy.deepcopy(base_cfg)
    cfg.stirrer.forcing_tau_s = tau_s
    cfg.stirrer.actual_rpm = actual_rpm
    if water_height_m is not None:
        cfg.geometry.water_height_m = water_height_m

    solver = build_solver(cfg, swirl_mode="forced")
    detector = SteadyStateDetector(steady_cfg)
    d_hist = []
    while solver.fields.t < t_max:
        solver.step()
        d = vortex_depth(solver.fields.phi, solver.grid, cfg)
        d_hist.append(d)
        if detector.update(solver.fields.t, d):
            break

    tail = max(1, len(d_hist) // 5)
    d_mean = float(np.mean(d_hist[-tail:]))
    return d_mean, {"steady": detector.is_steady, "steps": solver.fields.step, "t_final": solver.fields.t}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/calibration.yaml")
    parser.add_argument("--measured-depth-mm", type=float, required=True,
                         help="Real, experimentally measured vortex depth at --actual-rpm. Never fabricated by this script.")
    parser.add_argument("--actual-rpm", type=float, required=True,
                         help="Real, measured stir-bar RPM for the calibration condition (README section 33).")
    parser.add_argument("--water-height-mm", type=float, default=None)
    parser.add_argument("--tau-min-s", type=float, default=None)
    parser.add_argument("--tau-max-s", type=float, default=None)
    parser.add_argument("--tolerance-depth-mm", type=float, default=None)
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument("--t-max-s", type=float, default=0.3,
                         help="Per-trial simulated-time cap if steady state isn't reached first.")
    parser.add_argument("--out-dir", default="results/calibration")
    args = parser.parse_args()

    base_cfg = load_config(args.config)
    with open(args.config, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    cal = raw.get("calibration", {})

    tau_min = args.tau_min_s if args.tau_min_s is not None else cal.get("tau_min_s", 0.0005)
    tau_max = args.tau_max_s if args.tau_max_s is not None else cal.get("tau_max_s", 0.05)
    tol_mm = args.tolerance_depth_mm if args.tolerance_depth_mm is not None else cal.get("tolerance_depth_mm", 0.1)
    max_iter = args.max_iterations if args.max_iterations is not None else cal.get("max_iterations", 20)

    measured_depth_m = args.measured_depth_mm / 1e3
    water_height_m = args.water_height_mm / 1e3 if args.water_height_mm is not None else None
    steady_cfg = base_cfg.steady_state

    history = []

    def residual(tau_s: float) -> float:
        d, info = simulate_depth(base_cfg, tau_s, args.actual_rpm, water_height_m, args.t_max_s, steady_cfg)
        r = d - measured_depth_m
        history.append({"tau_s": tau_s, "depth_mm": d * 1e3, "residual_mm": r * 1e3, **info})
        print(f"  tau_s={tau_s:.5f}s -> depth={d*1e3:.4f}mm (target {args.measured_depth_mm:.4f}mm), "
              f"residual={r*1e3:+.4f}mm, steady={info['steady']}, steps={info['steps']}")
        return r

    print(f"Calibrating tau_s against measured depth={args.measured_depth_mm}mm "
          f"at actual_rpm={args.actual_rpm}, bracket=[{tau_min}, {tau_max}]s")
    print("Evaluating bracket endpoints...")
    r_min = residual(tau_min)
    r_max = residual(tau_max)

    if r_min == 0.0:
        tau_best = tau_min
    elif r_max == 0.0:
        tau_best = tau_max
    elif (r_min > 0) == (r_max > 0):
        print(f"\nERROR: bracket [{tau_min}, {tau_max}] does not straddle the target depth "
              f"(residuals {r_min*1e3:+.4f}mm and {r_max*1e3:+.4f}mm have the same sign). "
              f"Widen --tau-min-s/--tau-max-s and retry. NOT writing a calibrated_forcing.yaml.")
        sys.exit(1)
    else:
        lo, hi = tau_min, tau_max
        r_lo = r_min
        tau_best = None
        for i in range(max_iter):
            mid = math_geomean(lo, hi)
            r_mid = residual(mid)
            if abs(r_mid * 1e3) < tol_mm:
                tau_best = mid
                break
            if (r_mid > 0) == (r_lo > 0):
                lo, r_lo = mid, r_mid
            else:
                hi = mid
        if tau_best is None:
            tau_best = mid  # best available after max_iterations

    d_final, info_final = simulate_depth(base_cfg, tau_best, args.actual_rpm, water_height_m, args.t_max_s, steady_cfg)
    residual_final_mm = (d_final - measured_depth_m) * 1e3

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nBest-fit tau_s = {tau_best:.5f} s")
    print(f"  simulation depth = {d_final*1e3:.4f} mm, residual = {residual_final_mm:+.4f} mm "
          f"(tolerance {tol_mm} mm), steady={info_final['steady']}")

    result = {
        "tau_s": tau_best,
        "calibration_rpm": args.actual_rpm,
        "water_height_m": water_height_m if water_height_m is not None else base_cfg.geometry.water_height_m,
        "measured_depth_mm": args.measured_depth_mm,
        "simulation_depth_mm": d_final * 1e3,
        "residual_mm": residual_final_mm,
        "reached_steady_state": info_final["steady"],
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "config_source": str(args.config),
    }
    try:
        import subprocess
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent,
                                              stderr=subprocess.DEVNULL).decode().strip()
        result["git_commit"] = git_commit
    except Exception:
        pass

    yaml_path = out_dir / "calibrated_forcing.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(result, f, sort_keys=False)
    print(f"Saved {yaml_path}")

    hist_taus = [h["tau_s"] for h in history]
    hist_depths = [h["depth_mm"] for h in history]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    order = np.argsort(hist_taus)
    ax.plot(np.array(hist_taus)[order], np.array(hist_depths)[order], "o-", label="search history")
    ax.axhline(args.measured_depth_mm, color="tab:red", ls="--", label="measured depth")
    ax.axvline(tau_best, color="tab:green", ls=":", label=f"best-fit tau_s={tau_best:.4f}s")
    ax.set_xlabel(r"$\tau_s$ [s]")
    ax.set_ylabel("simulated vortex depth [mm]")
    ax.set_title(f"forcing calibration, RPM={args.actual_rpm}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig_path = out_dir / "calibration_search.png"
    fig.savefig(fig_path, dpi=200)
    print(f"Saved {fig_path}")


def math_geomean(a: float, b: float) -> float:
    """Geometric-mean midpoint: tau_s spans potentially orders of
    magnitude, so bisecting in log-space converges far faster than a
    linear midpoint."""
    import math
    return math.sqrt(a * b)


if __name__ == "__main__":
    main()
