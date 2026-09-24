"""Sanity-check trends across a pilot RPM sweep (README "pilot result
automatic sanity check" / "d vs N^2 pilot plot").

Reports whether, as RPM increases, vortex depth / swirl / downward jet
increase and central pressure drops -- WITHOUT forcing these trends to
pass; a broken trend is reported as broken, with the raw values, not
hidden (README safeguard: "이 경향을 강제로 PASS시키지 말고 실제 결과를 보고한다").

Also plots d_mean vs RPM^2 (and, given R_m, the dimensionless d/R_m vs
Fr_Omega version), explicitly labeled "pilot / preliminary" -- 4 points is
not enough to claim a universal scaling law (README safeguard).

Usage:
    python scripts/analyze_pilot.py --summary results/pilot/pilot_summary.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from air_vortex.config import load_config  # noqa: E402
from air_vortex.plotting import plot_depth_scaling  # noqa: E402
from air_vortex.viz_common import save_figure  # noqa: E402
from air_vortex.run_io import RunPaths  # noqa: E402


def is_monotonic_increasing(x: np.ndarray, y: np.ndarray) -> bool:
    order = np.argsort(x)
    y_sorted = np.asarray(y)[order]
    return bool(np.all(np.diff(y_sorted) >= 0))


def is_monotonic_decreasing(x: np.ndarray, y: np.ndarray) -> bool:
    order = np.argsort(x)
    y_sorted = np.asarray(y)[order]
    return bool(np.all(np.diff(y_sorted) <= 0))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", default="results/pilot/pilot_summary.csv")
    parser.add_argument("--config", default="configs/baseline.yaml",
                         help="Used only for R_m/g in the dimensionless plot.")
    parser.add_argument("--out-dir", default="results/pilot")
    args = parser.parse_args()

    df = pd.read_csv(args.summary)
    if df.empty:
        print(f"{args.summary} has no rows; nothing to analyze.")
        return

    rpm = df["rpm_used"].to_numpy(dtype=float)
    d_mean = df["d_mean_mm"].to_numpy(dtype=float)
    max_u_theta = df["max_u_theta"].to_numpy(dtype=float)
    min_pressure = df["min_pressure_pa"].to_numpy(dtype=float)
    min_u_z = df["min_u_z"].to_numpy(dtype=float) if "min_u_z" in df else None

    trends = {
        "monotonic_depth": is_monotonic_increasing(rpm, d_mean),
        "monotonic_swirl": is_monotonic_increasing(rpm, max_u_theta),
        "monotonic_pressure_drop": is_monotonic_decreasing(rpm, min_pressure),
    }
    if min_u_z is not None:
        trends["monotonic_downward_jet"] = is_monotonic_decreasing(rpm, min_u_z)  # more negative = stronger

    print("=== pilot trend sanity check (RPM increasing) ===")
    print(f"N runs: {len(df)}")
    if not df["forcing_calibrated"].all():
        print("NOTE: forcing_calibrated=False for at least one run -- these are "
              "UNCALIBRATED simulations, not validated physical predictions.\n")

    for name, ok in trends.items():
        print(f"{name}: {ok}")
        if not ok:
            print("  raw values (sorted by rpm_used):")
            print(df.sort_values("rpm_used")[["rpm_used", "d_mean_mm", "max_u_theta",
                                               "min_pressure_pa"] +
                                              (["min_u_z"] if min_u_z is not None else [])
                                              ].to_string(index=False))

    out_paths = RunPaths(Path(args.out_dir)).ensure()

    if len(df) >= 2:
        fig = plot_depth_scaling(rpm, d_mean / 1e3)
        fig.axes[0].set_title(fig.axes[0].get_title() + "  [pilot / preliminary]")
        saved = save_figure(fig, out_paths, "pilot_depth_vs_n_squared")
        print(f"\nSaved {saved['png']}\nSaved {saved['pdf']}")

        try:
            cfg = load_config(args.config)
            fig2 = plot_depth_scaling(rpm, d_mean / 1e3, dimensionless=True,
                                       R_m=cfg.geometry.stirbar_half_length_m, g=cfg.fluid.gravity)
            fig2.axes[0].set_title(fig2.axes[0].get_title() + "  [pilot / preliminary]")
            saved = save_figure(fig2, out_paths, "pilot_depth_dimensionless")
            print(f"Saved {saved['png']}\nSaved {saved['pdf']}")
        except Exception as e:
            print(f"(skipped dimensionless plot: {e})")
    else:
        print(f"\nOnly {len(df)} run(s); need >= 2 for a scaling plot.")

    print("\nCAUTION: with only", len(df), "pilot points, do not read the fit "
          "in these plots as a validated/universal scaling law -- it is a "
          "preliminary trend only (README safeguard).")


if __name__ == "__main__":
    main()
