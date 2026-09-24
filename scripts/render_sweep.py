"""Render sweep-scaling figures (7-10) from saved sweep results, without
re-running the solver.

Two input shapes are supported under --results (auto-detected):

1. A depth_sweep.csv (as scripts/run_depth_sweep.py produces: columns
   water_height_mm, N_c_rpm) -> Figure 9 (critical-RPM scaling).
2. A directory of run_io run directories (each with diagnostics.csv +
   metadata.json giving its rpm), e.g. one run per swept RPM -> Figure 7
   (d(t) per run) and Figure 8 (d_infinity vs N^2, using the mean of the
   last fraction of each run's "d" column as a d_infinity estimate).

Usage:
    python scripts/render_sweep.py --results results/rpm_sweep
    python scripts/render_sweep.py --results results/depth_sweep
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
from air_vortex.plotting import plot_critical_scaling, plot_depth_scaling, plot_vortex_depth_time  # noqa: E402
from air_vortex.run_io import RunPaths, load_run  # noqa: E402
from air_vortex.viz_common import save_figure  # noqa: E402


def _find_run_dirs(results_dir: Path) -> list[Path]:
    return sorted(d for d in results_dir.iterdir()
                  if d.is_dir() and (d / "diagnostics.csv").exists())


def _render_from_run_dirs(run_dirs: list[Path], out_paths: RunPaths, R_m: float | None,
                           g: float, tail_fraction: float = 0.2) -> None:
    runs = {}
    N_list, d_inf_list = [], []

    for d in run_dirs:
        run = load_run(d)
        try:
            df = run.load_diagnostics()
        except Exception:
            continue
        rpm = run.metadata.get("rpm")
        label = f"{rpm:.0f} rpm" if rpm is not None else d.name
        runs[label] = df

        if rpm is not None and len(df):
            tail = df.tail(max(1, int(len(df) * tail_fraction)))
            N_list.append(rpm)
            d_inf_list.append(tail["d"].mean())

    if runs:
        fig7 = plot_vortex_depth_time(runs)
        saved = save_figure(fig7, out_paths, "fig07_vortex_depth_time")
        print(f"Saved {saved['png']}\nSaved {saved['pdf']}")

    if len(N_list) >= 2:
        N_arr, d_arr = np.array(N_list), np.array(d_inf_list)
        fig8 = plot_depth_scaling(N_arr, d_arr)
        saved = save_figure(fig8, out_paths, "fig08_depth_scaling")
        print(f"Saved {saved['png']}\nSaved {saved['pdf']}")

        if R_m is not None:
            fig8b = plot_depth_scaling(N_arr, d_arr, dimensionless=True, R_m=R_m, g=g)
            saved = save_figure(fig8b, out_paths, "fig08_depth_scaling_dimensionless")
            print(f"Saved {saved['png']}\nSaved {saved['pdf']}")
    elif N_list:
        print("Only one RPM run found; skipping Figure 8 (needs >= 2 points for a scaling plot).")


def _render_from_depth_sweep_csv(csv_path: Path, out_paths: RunPaths,
                                  R_m: float | None, g: float, z_bar_top: float | None) -> None:
    df = pd.read_csv(csv_path).dropna(subset=["N_c_rpm"])
    if df.empty:
        print(f"{csv_path} has no rows with a resolved N_c_rpm; nothing to plot.")
        return

    H_m = df["water_height_mm"].to_numpy() / 1e3
    N_c = df["N_c_rpm"].to_numpy()
    H_eff = H_m - z_bar_top if z_bar_top is not None else None

    fig9 = plot_critical_scaling(H_m, N_c, H_eff=H_eff)
    saved = save_figure(fig9, out_paths, "fig09_critical_scaling")
    print(f"Saved {saved['png']}\nSaved {saved['pdf']}")

    if R_m is not None:
        fig9b = plot_critical_scaling(H_m, N_c, H_eff=H_eff, dimensionless=True, R_m=R_m, g=g)
        saved = save_figure(fig9b, out_paths, "fig09_critical_dimensionless")
        print(f"Saved {saved['png']}\nSaved {saved['pdf']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True)
    parser.add_argument("--config", default="configs/baseline.yaml",
                         help="Used only for R_m, g, z_bar_top in the dimensionless plots.")
    args = parser.parse_args()

    results_dir = Path(args.results)
    out_paths = RunPaths(results_dir).ensure()

    cfg = None
    try:
        cfg = load_config(args.config)
    except Exception:
        pass
    R_m = cfg.geometry.stirbar_half_length_m if cfg else None
    g = cfg.fluid.gravity if cfg else 9.81
    z_bar_top = cfg.geometry.stirbar_top_z_m if cfg else None

    depth_sweep_csv = results_dir / "depth_sweep.csv"
    if depth_sweep_csv.exists():
        _render_from_depth_sweep_csv(depth_sweep_csv, out_paths, R_m, g, z_bar_top)
        return

    run_dirs = _find_run_dirs(results_dir)
    if run_dirs:
        _render_from_run_dirs(run_dirs, out_paths, R_m, g)
        return

    print(f"Nothing recognized under {results_dir}: expected a depth_sweep.csv "
          f"or a directory of run_io run subdirectories (each with diagnostics.csv).")


if __name__ == "__main__":
    main()
