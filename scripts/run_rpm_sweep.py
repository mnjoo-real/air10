"""RPM sweep at fixed water depth, with bisection to N_c (README sections 20, 26 Study A)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from air_vortex.config import load_config  # noqa: E402
from air_vortex.sweep import bracket_transition, bisect_critical_rpm  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-config", default="configs/sweep_rpm.yaml")
    parser.add_argument("--out", default="results/summary/rpm_sweep.csv")
    args = parser.parse_args()

    with open(args.sweep_config, "r", encoding="utf-8") as f:
        sweep_cfg = yaml.safe_load(f)

    base_cfg = load_config(sweep_cfg["base_config"])
    H = sweep_cfg["water_height_m"]
    rpm_coarse = sweep_cfg["rpm_coarse"]
    bisection = sweep_cfg.get("bisection", {})

    bracket = bracket_transition(base_cfg, rpm_coarse, H)
    if bracket is None:
        print(f"No air-core formation found for H={H*1e3:.0f} mm "
              f"within RPM range {rpm_coarse}.")
        return

    low, high = bracket
    N_c = bisect_critical_rpm(
        base_cfg, low, high, H,
        target_resolution_rpm=bisection.get("target_resolution_rpm", 5.0),
        max_iterations=bisection.get("max_iterations", 8),
    )

    print(f"H={H*1e3:.0f} mm: N_c ~= {N_c:.0f} rpm (bracket {low:.0f}-{high:.0f} rpm)")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"water_height_mm": H * 1e3, "N_c_rpm": N_c,
                    "bracket_low": low, "bracket_high": high}]).to_csv(out_path, index=False)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
