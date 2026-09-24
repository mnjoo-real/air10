"""Build Plot B from a depth-sweep summary CSV (README section 25)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from air_vortex.config import load_config  # noqa: E402
from air_vortex.plotting import plot_b_critical_n_squared_vs_h_eff  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depth-sweep-csv", default="results/summary/depth_sweep.csv")
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--out", default="results/figures/plot_b_depth_sweep.png")
    args = parser.parse_args()

    cfg = load_config(args.config)
    df = pd.read_csv(args.depth_sweep_csv).dropna(subset=["N_c_rpm"])

    H_eff_mm = df["water_height_mm"] - cfg.geometry.stirbar_top_z_m * 1e3
    N_c = df["N_c_rpm"].to_numpy()

    fig, ax = plt.subplots()
    plot_b_critical_n_squared_vs_h_eff(H_eff_mm.to_numpy() / 1e3, N_c, ax=ax)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
