"""Phase 1 deliverable: Level 0 reduced vortex model (README section 4) plus
the dimensionless parameter table (README section 17-18).

This is NOT a CFD solution - it only estimates a useful RPM range and
provides initial guesses before running the Level 1 solver. `a` and `beta`
are unknown effective parameters (README section 4, "Important limitation"),
so several values are reported to show the resulting sensitivity.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from air_vortex.config import load_config  # noqa: E402


def level0_critical_rpm(g: float, H_eff: float, beta: float, a: float) -> float:
    """N_c ~ (60/2pi) * sqrt(g*H_eff) / (beta*a) (README section 4, boxed)."""
    omega_c = math.sqrt(g * H_eff) / (beta * a)
    return 60.0 / (2.0 * math.pi) * omega_c


def dimensionless_numbers(cfg, rpm: float) -> dict:
    """README section 17: Re_Omega, Fr_Omega, We_Omega, Bo, H*, C, AR."""
    Omega = 2.0 * math.pi * rpm / 60.0
    R_m = cfg.geometry.stirbar_half_length_m
    rho, mu, sigma, g = (cfg.fluid.water_density, cfg.fluid.water_viscosity,
                         cfg.fluid.surface_tension, cfg.fluid.gravity)

    return {
        "Re_Omega": rho * Omega * R_m**2 / mu,
        "Fr_Omega": Omega**2 * R_m / g,
        "We_Omega": rho * Omega**2 * R_m**3 / sigma,
        "Bo": rho * g * R_m**2 / sigma,
        "H_star": cfg.geometry.water_height_m / R_m,
        "C": cfg.geometry.confinement_ratio,
        "AR": cfg.geometry.bar_aspect_ratio,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--betas", type=float, nargs="+", default=[0.3, 0.5, 0.7, 1.0])
    args = parser.parse_args()

    cfg = load_config(args.config)
    H_eff = cfg.geometry.effective_depth_m
    R_m = cfg.geometry.stirbar_half_length_m
    g = cfg.fluid.gravity

    print(f"Baseline geometry: R_v={cfg.geometry.vessel_radius_m*1e3:.1f} mm, "
          f"H={cfg.geometry.water_height_m*1e3:.1f} mm, "
          f"R_m={R_m*1e3:.1f} mm, H_eff={H_eff*1e3:.1f} mm\n")

    print("Level 0 estimated N_c (a = R_m):")
    print(f"{'beta':>6} | {'N_c [rpm]':>12}")
    for beta in args.betas:
        N_c = level0_critical_rpm(g, H_eff, beta, a=R_m)
        print(f"{beta:6.2f} | {N_c:12.1f}")

    print("\nDimensionless numbers at the baseline RPM "
          f"({cfg.stirrer.rpm:.0f} rpm):")
    for name, value in dimensionless_numbers(cfg, cfg.stirrer.rpm).items():
        print(f"  {name:10s} = {value:.4g}")

    print("\nRPM -> Omega [rad/s] conversion (README section 18):")
    for rpm in (80, 300, 500, 700, 900, 1100, 1300, 1500):
        omega = 2.0 * math.pi * rpm / 60.0
        print(f"  N={rpm:5.0f} rpm -> Omega={omega:7.2f} rad/s, "
              f"U_tip={omega*R_m:.3f} m/s")


if __name__ == "__main__":
    main()
