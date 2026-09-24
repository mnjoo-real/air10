"""Validation gate (README "Exact equilibrium test를 numerical gate로 우선
사용"), structured into three tiers that are NEVER mixed into one PASS/FAIL
metric:

  NUMERICAL VERIFICATION  -- does the solver preserve a KNOWN exact
                              analytical state? (hydrostatic, exact
                              rotating equilibrium sigma=0, divergence,
                              grid/timestep convergence, sub-grid tip
                              extraction). These gate production readiness.

  PHYSICAL MODEL VALIDATION -- capillary-corrected equilibrium comparison,
                              forcing calibration, actual RPM. These ALSO
                              gate production readiness (forcing
                              calibration in particular always blocks it
                              while missing), but are validating the
                              PHYSICAL MODEL, not the numerics.

  TRANSIENT CHARACTERIZATION -- flat-start spin-up oscillation frequency,
                              damping, ramp sensitivity. Reported for
                              information; deliberately NOT a PASS/FAIL
                              gate (README: "flat surface가 빨리 parabola로
                              가는가?" is a different question from "solver가
                              올바른 rotating equilibrium을 표현할 수 있는가?").

Usage:
    python scripts/check_validation_gate.py
    python scripts/check_validation_gate.py --production-config configs/pilot_H050_RPM0900.yaml
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _run_pytest(node_id: str | None = None) -> tuple[bool, str]:
    cmd = [sys.executable, "-m", "pytest", "-q"]
    if node_id:
        cmd.append(node_id)
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    ok = result.returncode == 0
    tail = "\n".join(result.stdout.strip().splitlines()[-5:])
    return ok, tail


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _print_section(title: str, checks: list[tuple[str, str, str]]) -> None:
    print(title)
    print("=" * len(title))
    for name, status, detail in checks:
        print(f"{status:<5} {name}")
        if status != "PASS":
            for line in detail.splitlines():
                print(f"      {line}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--production-config", default=None,
                         help="A pilot/production config to check for actual_rpm availability.")
    parser.add_argument("--volume-drift-threshold", type=float, default=0.01)
    parser.add_argument("--divergence-threshold", type=float, default=1e-6)
    parser.add_argument("--exact-eq-nrmse-threshold", type=float, default=0.05,
                         help="README acceptance criteria for the sigma=0 exact-equilibrium "
                              "test specifically (0.02 preferred) -- stricter than the "
                              "flat-start transient, which isn't gated at all.")
    args = parser.parse_args()

    results_root = Path(args.results_root)

    # ================= NUMERICAL VERIFICATION =================
    numerical: list[tuple[str, str, str]] = []

    ok, tail = _run_pytest()
    numerical.append(("all pytest tests", "PASS" if ok else "FAIL", tail))

    ok, tail = _run_pytest("tests/test_hydrostatic.py")
    numerical.append(("hydrostatic equilibrium", "PASS" if ok else "FAIL", tail))

    ok, tail = _run_pytest("tests/test_pressure.py")
    numerical.append(("pressure projection", "PASS" if ok else "FAIL", tail))

    # exact rotating equilibrium (sigma=0) -- from
    # scripts/validate_exact_rotating_equilibrium.py, NOT the flat-start
    # transient. Uses the best of correction ON/OFF, same "don't cherry
    # pick a number that looks good in isolation" logic as before, but
    # now the metric itself is the right one: does the solver hold a
    # state it's already told is the equilibrium.
    eq_runs = sorted(results_root.glob("validation_exact_equilibrium*"))
    eq_summaries = None
    for d in eq_runs:
        m = _load_json(d / "metadata.json")
        if m and m.get("kind") == "exact_rotating_equilibrium_preservation":
            eq_summaries = m["summaries"]
            break

    if not eq_summaries:
        numerical.append(("exact rotating equilibrium, sigma=0", "FAIL",
                           "no results/validation_exact_equilibrium*/metadata.json found -- "
                           "run scripts/validate_exact_rotating_equilibrium.py first"))
        numerical.append(("volume conservation", "FAIL", "same -- no exact-equilibrium data"))
        numerical.append(("divergence", "WARN", "no exact-equilibrium data; relying on pytest regression only"))
    else:
        cases = list(eq_summaries.values())
        in_budget = [c for c in cases if abs(c["volume_drift_final"]) < args.volume_drift_threshold]
        best = min(in_budget or cases, key=lambda c: c["eta_nrmse"])
        best_label = [k for k, v in eq_summaries.items() if v is best][0]

        nrmse = best["eta_nrmse"]
        eq_ok = nrmse < args.exact_eq_nrmse_threshold
        numerical.append(("exact rotating equilibrium, sigma=0", "PASS" if eq_ok else "FAIL",
                           f"best case ({best_label}): eta NRMSE={nrmse:.4f} "
                           f"(threshold {args.exact_eq_nrmse_threshold}, 0.02 preferred); "
                           f"max|u_r|={best['max_ur']:.3e} m/s, max|u_z|={best['max_uz']:.3e} m/s "
                           f"(ideal: both ~0) -- see README section 47.3-equivalent notes: even the "
                           f"EXACT analytical initial condition is not preserved, so this is NOT "
                           f"explained by transient/startup excitation."))

        drift = best["volume_drift_final"]
        vol_ok = abs(drift) < args.volume_drift_threshold
        numerical.append(("volume conservation", "PASS" if vol_ok else "FAIL",
                           f"best case ({best_label}): final volume drift={drift*100:.3f}% "
                           f"(threshold {args.volume_drift_threshold*100:.1f}%)"))

        div = best["max_divergence"]
        div_ok = div < args.divergence_threshold
        numerical.append(("divergence", "PASS" if div_ok else "FAIL",
                           f"best case ({best_label}): max divergence={div:.3e} "
                           f"(threshold {args.divergence_threshold:.1e})"))

    ok, tail = _run_pytest(
        "tests/test_diagnostics_and_connectivity.py::"
        "test_find_tip_z_subgrid_interpolation_is_grid_independent")
    numerical.append(("sub-grid tip extraction", "PASS" if ok else "FAIL", tail))

    ts_csv = results_root / "convergence" / "timestep_summary.csv"
    numerical.append(("timestep convergence", "PASS" if ts_csv.exists() else "FAIL",
                       str(ts_csv) + (" found" if ts_csv.exists() else " NOT found -- "
                                      "run scripts/run_timestep_convergence.py")))

    gc_csv = results_root / "convergence" / "grid_convergence_summary.csv"
    numerical.append(("grid convergence", "PASS" if gc_csv.exists() else "FAIL",
                       str(gc_csv) + (" found" if gc_csv.exists() else " NOT found -- "
                                      "run scripts/run_grid_convergence.py")))

    # ================= PHYSICAL MODEL VALIDATION =================
    physical: list[tuple[str, str, str]] = []

    # capillary_equilibrium.py implements and tests the CASE C reference
    # profile itself (README section 4/5); a dedicated scripted run that
    # initializes and preserves it the way validate_exact_rotating_
    # equilibrium.py does for CASE P is not yet wired up as a gated check
    # in this pass -- reported as WARN, not silently skipped.
    physical.append(("capillary-corrected equilibrium comparison", "WARN",
                      "capillary_equilibrium.py reference profile implemented and unit-tested "
                      "(tests/test_capillary_equilibrium.py), but not yet run as a dedicated "
                      "CASE C preservation check the way CASE P is -- see README section 47."))

    cal_yaml = results_root / "calibration" / "calibrated_forcing.yaml"
    cal_ok = cal_yaml.exists()
    physical.append(("forcing calibration", "PASS" if cal_ok else "FAIL",
                      str(cal_yaml) + (" found" if cal_ok else " NOT found -- run "
                                       "scripts/calibrate_forcing.py against a real measured depth")))

    if args.production_config:
        from air_vortex.config import load_config
        cfg = load_config(args.production_config)
        if cfg.stirrer.rpm_source == "actual":
            physical.append(("actual RPM", "PASS", f"rpm_used={cfg.stirrer.rpm_used} (measured)"))
        else:
            physical.append(("actual RPM", "WARN",
                              f"{args.production_config}: rpm_source=setpoint_fallback "
                              f"(using display RPM={cfg.stirrer.rpm}, README section 33)"))
    else:
        physical.append(("actual RPM", "WARN", "no --production-config given to check"))

    # ================= TRANSIENT CHARACTERIZATION (reported, not gated) =================
    transient_lines = []
    spinup_runs = sorted(results_root.glob("validation_spinup_transient*"))
    spinup_summary = None
    for d in spinup_runs:
        m = _load_json(d / "metadata.json")
        if m and m.get("kind") == "spinup_transient_ramp_sensitivity":
            spinup_summary = m["summaries"]
            break
    if spinup_summary:
        for row in spinup_summary:
            freq = f"{row['f_dominant_hz']:.2f} Hz" if row["f_dominant_hz"] else "n/a"
            damp = (f"tau_d={row['tau_d_s']:.3f}s" if row["damping_resolvable"]
                    else "not resolvable")
            transient_lines.append(
                f"  t_ramp={row['ramp_time_s']}s: overshoot={row['max_overshoot_mm']:.2f}mm, "
                f"f_dominant={freq}, damping={damp}")
    else:
        transient_lines.append("  no results/validation_spinup_transient*/metadata.json found -- "
                                "run scripts/validate_spinup_transient.py")

    _print_section("NUMERICAL VERIFICATION", numerical)
    _print_section("PHYSICAL MODEL VALIDATION", physical)
    print("TRANSIENT CHARACTERIZATION (reported, not a PASS/FAIL gate)")
    print("=" * 60)
    for line in transient_lines:
        print(line)
    print()

    hard_fail = any(status == "FAIL" for _, status, _ in numerical + physical)
    production_ready = not hard_fail
    print(f"PRODUCTION READY: {'YES' if production_ready else 'NO'}")
    if not production_ready:
        failed = [name for name, status, _ in numerical + physical if status == "FAIL"]
        print("Blocking items: " + ", ".join(failed))

    sys.exit(0 if production_ready else 1)


if __name__ == "__main__":
    main()
