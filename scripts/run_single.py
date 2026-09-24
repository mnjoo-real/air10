"""Run a single (RPM, H) case and save diagnostics + field snapshots into a
results/<run_id>/ run directory (README sections 24, 30; "Result
Visualization" run-directory layout; "steady-state detector").

The solver's physics loop itself is unchanged -- this script only decides
*when* to persist a snapshot (every ``output.save_every_s``, not every
solver step), tracks cross-step state the solver itself doesn't need to
know about (air-core persistence, statistically-steady detection), and
decides when to stop: ``t_end`` reached, statistically steady (if
``steady_state.enabled``), or a numerical failure (NaN/Inf in any field).
"""
from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from tqdm import tqdm  # noqa: E402

from air_vortex.config import load_config, with_overrides  # noqa: E402
from air_vortex.connectivity import ConnectivityState, is_geometrically_connected, update_persistence  # noqa: E402
from air_vortex.diagnostics import scalar_diagnostics  # noqa: E402
from air_vortex.run_io import SnapshotWriter, grid_metadata, run_paths, save_config, save_metadata  # noqa: E402
from air_vortex.solver import build_solver  # noqa: E402
from air_vortex.steady_state import SteadyStateDetector  # noqa: E402


def _fields_finite(fields) -> bool:
    return all(np.all(np.isfinite(a)) for a in
               (fields.u_r, fields.u_z, fields.u_theta, fields.p, fields.phi))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--rpm", type=float, default=None)
    parser.add_argument("--actual-rpm", type=float, default=None,
                         help="Measured stir-bar RPM (README section 33); overrides "
                              "the config's actual_rpm if given. Physics uses this "
                              "over the display/set --rpm when available.")
    parser.add_argument("--water-height-mm", type=float, default=None)
    parser.add_argument("--t-end", type=float, default=None)
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--run-id", default=None,
                         help="Run directory name under --results-root. "
                              "Default: run_H<water_height_mm>_RPM<rpm>.")
    parser.add_argument("--swirl-mode", choices=["forced", "prescribed"], default="forced",
                         help="'forced' is the production stirrer-forcing model "
                              "(README section 10). 'prescribed' is VALIDATION-ONLY and "
                              "legacy two_phase_diffuse_ls only: it pins u_theta to "
                              "solid-body rotation over the whole domain (README Test 2) "
                              "and is refused together with --kind production.")
    parser.add_argument("--kind", default="production",
                         choices=["production", "hydrostatic", "solid_body_rotation"],
                         help="Tag stored in metadata.json so render_validation.py "
                              "knows which validation figure(s) apply.")
    parser.add_argument("--pressure-method", choices=["direct", "cg"], default="direct")
    args = parser.parse_args()
    if args.swirl_mode == "prescribed" and args.kind == "production":
        parser.error("--swirl-mode prescribed is validation-only (README_rewritten 21.3); "
                     "use --kind solid_body_rotation for that legacy diagnostic.")

    cfg = load_config(args.config)
    overrides = {}
    if args.rpm is not None:
        overrides["rpm"] = args.rpm
    if args.water_height_mm is not None:
        overrides["water_height_m"] = args.water_height_mm / 1000.0
    if overrides:
        cfg = with_overrides(cfg, **overrides)
    if args.actual_rpm is not None:
        cfg.stirrer.actual_rpm = args.actual_rpm

    if cfg.stirrer.rpm_source == "setpoint_fallback":
        print(f"WARNING: no actual_rpm available -- using the display/set RPM "
              f"({cfg.stirrer.rpm:.1f}) as the simulation's angular velocity. "
              f"Magnetic slip/step-out can make this differ from the true "
              f"stir-bar RPM at high load (README section 33).")

    t_end = args.t_end if args.t_end is not None else cfg.time.t_end_s

    run_id = args.run_id or f"run_H{cfg.geometry.water_height_m*1e3:03.0f}_RPM{cfg.stirrer.rpm_used:04.0f}"
    paths = run_paths(args.results_root, run_id)

    solver = build_solver(cfg, swirl_mode=args.swirl_mode, pressure_method=args.pressure_method)
    save_config(paths, cfg)

    writer = SnapshotWriter(paths)
    rows = []
    next_save_t = 0.0

    connectivity_state = ConnectivityState(connected=False)
    steady_detector = SteadyStateDetector(cfg.steady_state)
    termination_reason = "reached_t_end"
    wall_start = datetime.datetime.now()

    with tqdm(total=t_end, unit="s") as pbar:
        while solver.fields.t < t_end:
            t_before = solver.fields.t
            solver.step()
            pbar.update(solver.fields.t - t_before)

            if not _fields_finite(solver.fields):
                termination_reason = "numerical_failure"
                print(f"\nNumerical failure (NaN/Inf) detected at t={solver.fields.t:.6g}s, "
                      f"step={solver.fields.step}. Stopping.")
                break

            row = scalar_diagnostics(solver.fields, solver.grid, cfg)

            connected_now = row["air_core_connected"]
            connectivity_state = update_persistence(
                connectivity_state, connected_now, solver.fields.t, cfg, cfg.stirrer.rpm_used)
            row["stable_air_core"] = connectivity_state.connected

            just_steady = steady_detector.update(solver.fields.t, row["d"])
            row["statistically_steady"] = steady_detector.is_steady
            rows.append(row)

            if solver.fields.t >= next_save_t:
                writer.write(solver.fields)
                next_save_t += cfg.output.save_every_s

            if just_steady:
                termination_reason = "statistically_steady"
                print(f"\nStatistically steady at t={solver.fields.t:.6g}s "
                      f"(step {solver.fields.step}). Stopping early.")
                break

    wall_time_s = (datetime.datetime.now() - wall_start).total_seconds()

    df = pd.DataFrame(rows)
    df.to_csv(paths.diagnostics_path, index=False)

    save_metadata(paths, {
        "kind": args.kind,
        "swirl_mode": args.swirl_mode,
        "free_surface_model": cfg.physics.free_surface_model,
        "pressure_method": args.pressure_method,
        "rpm": cfg.stirrer.rpm,
        "actual_rpm": cfg.stirrer.actual_rpm,
        "rpm_used": cfg.stirrer.rpm_used,
        "rpm_source": cfg.stirrer.rpm_source,
        "water_height_mm": cfg.geometry.water_height_m * 1e3,
        "t_end_s": t_end,
        "t_final_s": solver.fields.t,
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "grid": grid_metadata(solver.grid),
        "n_solver_steps": solver.fields.step,
        "n_saved_frames": writer.n_written,
        "termination_reason": termination_reason,
        "steady_time_s": steady_detector.steady_time_s,
        "wall_time_s": wall_time_s,
        "forcing_calibrated": cfg.stirrer.calibrated,
    })

    print(f"Run directory: {paths.root}")
    print(f"  diagnostics:        {paths.diagnostics_path}")
    print(f"  frames:             {writer.n_written} snapshots in {paths.fields_dir}")
    print(f"  termination_reason: {termination_reason}")
    print(f"  wall_time_s:        {wall_time_s:.2f}")


if __name__ == "__main__":
    main()
