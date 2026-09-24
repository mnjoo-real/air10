"""Aggregate a set of pilot run directories (each produced by
scripts/run_single.py) into results/pilot/pilot_summary.csv (README
"physical quantities to record for each pilot run").

Usage:
    python scripts/collect_pilot_summary.py --runs results/run_H050_RPM0500 \\
        results/run_H050_RPM0900 results/run_H050_RPM1200 results/run_H050_RPM1500
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from air_vortex.run_io import load_run  # noqa: E402


def summarize_run(run_dir: str, tail_fraction: float = 0.2) -> dict:
    run = load_run(run_dir)
    meta = run.metadata
    df = run.load_diagnostics()

    tail = df.tail(max(1, int(len(df) * tail_fraction)))

    return {
        "rpm": meta.get("rpm"),
        "actual_rpm": meta.get("actual_rpm"),
        "rpm_used": meta.get("rpm_used"),
        "rpm_source": meta.get("rpm_source"),
        "water_height_mm": meta.get("water_height_mm"),
        "d_mean_mm": float(tail["d"].mean()) * 1e3,
        "d_std_mm": float(tail["d"].std()) * 1e3,
        "max_u_theta": float(df["max_u_theta"].max()),
        "min_u_z": float(-df["max_downward_uz"].max()) if "max_downward_uz" in df else np.nan,
        "min_pressure_pa": float(df["min_pressure"].min()),
        "max_divergence": float(df["max_divergence"].max()) if "max_divergence" in df else np.nan,
        "air_core_connected": bool(df["air_core_connected"].iloc[-1]) if "air_core_connected" in df else None,
        "stable_air_core": bool(df["stable_air_core"].iloc[-1]) if "stable_air_core" in df else None,
        "statistically_steady": bool(df["statistically_steady"].iloc[-1]) if "statistically_steady" in df else None,
        "steady_time_s": meta.get("steady_time_s"),
        "termination_reason": meta.get("termination_reason"),
        "volume_drift": float(abs(df["water_volume"].iloc[-1] - df["water_volume"].iloc[0])
                               / df["water_volume"].iloc[0]) if "water_volume" in df else np.nan,
        "wall_time_s": meta.get("wall_time_s"),
        "steps": meta.get("n_solver_steps"),
        "forcing_calibrated": meta.get("forcing_calibrated"),
        "run_dir": str(run_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--out", default="results/pilot/pilot_summary.csv")
    args = parser.parse_args()

    rows = [summarize_run(r) for r in args.runs]
    df = pd.DataFrame(rows).sort_values("rpm_used")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(df.to_string(index=False))
    print(f"\nSaved {out_path}")

    if not df["forcing_calibrated"].all():
        print("\nNOTE: at least one run used forcing_calibrated=False. This "
              "summary reflects UNCALIBRATED simulations and must not be read "
              "as a validated physical prediction (README safeguards).")


if __name__ == "__main__":
    main()
