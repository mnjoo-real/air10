"""Exact rotating-equilibrium PRESERVATION test (README "Exact Equilibrium
Preservation Test") -- a numerical-verification test distinct from the
flat-start spin-up transient (scripts/validate_spinup_transient.py).

Instead of starting from flat water and waiting to see if the solver finds
its way to the solid-body-rotation equilibrium (confounded by a real,
weakly-damped free-surface oscillation -- see README section 47.3), this
test starts EXACTLY AT the analytical equilibrium (fields.
initialize_rotating_equilibrium: solid-body swirl, the matching
volume-consistent parabola free surface, and the matching rotating-
hydrostatic pressure field) and checks whether the solver keeps it there.

CASE P (surface_tension forced to 0 for this run, regardless of the config
file) is the only case with an exact closed-form equilibrium; see
capillary_equilibrium.py for the CASE C (physical sigma) numerical
reference, used by validate_spinup_transient.py instead -- never mixed
with this exact-parabola comparison (README section 1's explicit
requirement).

Usage:
    python scripts/validate_exact_rotating_equilibrium.py \\
        --config configs/validation_solid_body.yaml --omega-rpm 100 --t-end 0.4
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
from air_vortex.diagnostics import free_surface_height, volume_consistent_parabola  # noqa: E402
from air_vortex.fields import initialize_rotating_equilibrium  # noqa: E402
from air_vortex.operators import divergence, interp_ur_to_center, interp_uz_to_center  # noqa: E402
from air_vortex.solver import Solver, water_volume  # noqa: E402
from air_vortex.viz_common import save_figure  # noqa: E402
from air_vortex.run_io import run_paths, save_config, save_metadata  # noqa: E402


def meridional_kinetic_energy(grid, rho, u_r, u_z) -> float:
    u_r_c = interp_ur_to_center(u_r)
    u_z_c = interp_uz_to_center(u_z)
    speed_sq = u_r_c**2 + u_z_c**2
    cell_volume = 2.0 * np.pi * grid.r_c[:, None] * grid.dr * grid.dz
    return float(np.sum(0.5 * rho * speed_sq * cell_volume))


def run_case(base_cfg, omega: float, t_end: float, correction: bool):
    cfg = copy.deepcopy(base_cfg)
    cfg.fluid.surface_tension = 0.0  # CASE P: exact parabola has no closed form with sigma!=0
    cfg.levelset.advection_scheme = "muscl2"
    cfg.levelset.time_integrator = "ssprk2"
    cfg.levelset.volume_correction.enabled = correction

    from air_vortex.grid import build_grid
    grid = build_grid(cfg)
    fields = initialize_rotating_equilibrium(grid, cfg, omega)
    solver = Solver(grid=grid, cfg=cfg, fields=fields, swirl_mode="prescribed", pressure_method="direct")

    V0 = water_volume(grid, fields.phi)
    eta_analytic = volume_consistent_parabola(grid.r_c, grid.r_v, omega, cfg.fluid.gravity, V0)
    p_analytic = fields.p.copy()  # the exact equilibrium pressure field, fixed reference

    rows = []
    t0 = time.perf_counter()
    # record t=0 state too, before any stepping
    rows.append(dict(t=0.0, volume_drift=0.0, eta_rmse_mm=0.0, center_error_mm=0.0,
                      max_ur=0.0, max_uz=0.0, max_div=0.0, pressure_rmse=0.0,
                      meridional_ke=0.0))

    while solver.fields.t < t_end:
        solver.step()
        t = solver.fields.t
        phi, u_r, u_z, p, rho = (solver.fields.phi, solver.fields.u_r, solver.fields.u_z,
                                  solver.fields.p, solver.fields.rho)

        Vt = water_volume(grid, phi)
        eta = free_surface_height(phi, grid)
        valid = np.isfinite(eta)
        eta_err = eta[valid] - eta_analytic[valid]
        eta_rmse = float(np.sqrt(np.mean(eta_err**2))) if np.any(valid) else np.nan
        center_err = abs(eta[0] - eta_analytic[0]) if np.isfinite(eta[0]) else np.nan

        div = divergence(grid, u_r, u_z)
        water_mask = phi < 0
        p_err = (p - p_analytic)[water_mask]
        p_rmse = float(np.sqrt(np.mean(p_err**2))) if p_err.size else np.nan

        rows.append(dict(
            t=t, volume_drift=(Vt - V0) / V0,
            eta_rmse_mm=eta_rmse * 1e3, center_error_mm=center_err * 1e3,
            max_ur=float(np.max(np.abs(u_r))), max_uz=float(np.max(np.abs(u_z))),
            max_div=float(np.max(np.abs(div))), pressure_rmse=p_rmse,
            meridional_ke=meridional_kinetic_energy(grid, rho, u_r, u_z),
        ))

    wall_time = time.perf_counter() - t0
    series = pd.DataFrame(rows)

    amplitude = omega**2 * grid.r_v**2 / (2.0 * cfg.fluid.gravity)
    final = series.iloc[-1]
    summary = {
        "correction": correction,
        "volume_drift_final": final["volume_drift"],
        "max_volume_drift": float(series["volume_drift"].abs().max()),
        "eta_nrmse": final["eta_rmse_mm"] / 1e3 / amplitude if amplitude > 0 else float("nan"),
        "eta_rmse_mm": final["eta_rmse_mm"],
        "center_error_mm": final["center_error_mm"],
        "center_error_relative": (final["center_error_mm"] / 1e3) / max(abs(eta_analytic[0]), 1e-9),
        "max_divergence": float(series["max_div"].max()),
        "max_ur": float(series["max_ur"].max()),
        "max_uz": float(series["max_uz"].max()),
        "max_pressure_rmse": float(series["pressure_rmse"].max()),
        "wall_time_s": wall_time, "steps": solver.fields.step,
        "cumulative_volume_correction": solver.cumulative_volume_correction,
    }
    return summary, series, grid


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/validation_solid_body.yaml")
    parser.add_argument("--omega-rpm", type=float, default=100.0,
                         help="RPM used only to compute omega=2*pi*rpm/60 -- this test does not use "
                              "the stirrer-forcing model at all (prescribed solid-body swirl throughout).")
    parser.add_argument("--t-end", type=float, default=0.4)
    parser.add_argument("--run-id", default="validation_exact_equilibrium")
    parser.add_argument("--results-root", default="results")
    args = parser.parse_args()

    base_cfg = load_config(args.config)
    omega = 2.0 * np.pi * args.omega_rpm / 60.0

    paths = run_paths(args.results_root, args.run_id)
    save_config(paths, base_cfg)

    all_summaries = {}
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for correction, label in [(False, "A: correction OFF"), (True, "B: correction ON")]:
        print(f"\n=== {label} ===")
        summary, series, grid = run_case(base_cfg, omega, args.t_end, correction)
        all_summaries[label] = summary
        series.to_csv(paths.root / f"{'B' if correction else 'A'}_timeseries.csv", index=False)

        print(f"  volume_drift_final={summary['volume_drift_final']*100:.4f}%  "
              f"eta_NRMSE={summary['eta_nrmse']:.4f}  center_error={summary['center_error_mm']:.4f}mm")
        print(f"  max_div={summary['max_divergence']:.3e}  max|u_r|={summary['max_ur']:.3e}  "
              f"max|u_z|={summary['max_uz']:.3e}")

        axes[0, 0].plot(series["t"], series["eta_rmse_mm"], label=label)
        axes[0, 1].plot(series["t"], series["volume_drift"] * 100, label=label)
        axes[0, 2].plot(series["t"], series["center_error_mm"], label=label)
        axes[1, 0].plot(series["t"], series["max_ur"], label=label)
        axes[1, 1].plot(series["t"], series["max_uz"], label=label)
        axes[1, 2].plot(series["t"], series["meridional_ke"], label=label)

    titles = [["interface RMSE [mm]", "volume drift [%]", "center depression error [mm]"],
              ["max |u_r| [m/s]", "max |u_z| [m/s]", "meridional KE [J]"]]
    for i in range(2):
        for j in range(3):
            axes[i, j].set_xlabel("t [s]")
            axes[i, j].set_ylabel(titles[i][j])
            axes[i, j].set_title(titles[i][j])
            axes[i, j].legend(fontsize=7)
            axes[i, j].grid(True, alpha=0.3)
    fig.suptitle(f"Exact rotating equilibrium preservation, sigma=0, Omega={omega:.3f} rad/s")
    fig.tight_layout()
    save_figure(fig, paths, "exact_equilibrium_preservation")

    df = pd.DataFrame(all_summaries).T
    csv_path = paths.root / "exact_equilibrium_summary.csv"
    df.to_csv(csv_path)
    print(f"\nSaved {csv_path}")
    print(df.to_string())

    save_metadata(paths, {
        "kind": "exact_rotating_equilibrium_preservation",
        "sigma_used": 0.0, "omega_rad_s": omega, "omega_rpm": args.omega_rpm,
        "t_end_s": args.t_end, "summaries": all_summaries,
    })


if __name__ == "__main__":
    main()
