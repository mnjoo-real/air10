"""Water-depth sweep: N_c(H) (README sections 19, 20, 26 Study B)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from air_vortex.config import load_config  # noqa: E402
from air_vortex.sweep import depth_sweep  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-config", default="configs/sweep_depth.yaml")
    parser.add_argument("--out", default="results/summary/depth_sweep.csv")
    args = parser.parse_args()

    with open(args.sweep_config, "r", encoding="utf-8") as f:
        sweep_cfg = yaml.safe_load(f)

    base_cfg = load_config(sweep_cfg["base_config"])
    water_heights = sweep_cfg["water_heights_m"]
    rpm_coarse = sweep_cfg["rpm_coarse"]
    bisection = sweep_cfg.get("bisection", {})

    results = depth_sweep(
        base_cfg, water_heights, rpm_coarse,
        target_resolution_rpm=bisection.get("target_resolution_rpm", 5.0),
        max_iterations=bisection.get("max_iterations", 8),
    )

    rows = [{"water_height_mm": H * 1e3, "N_c_rpm": N_c} for H, N_c in results.items()]
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
