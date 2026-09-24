"""Performance profiling and before/after regression harness for the
solver's numerical kernels (primarily the pressure Poisson solve).

Usage:
    # profile where time actually goes (cProfile, sorted by cumulative time)
    python scripts/benchmark_solver.py --profile --steps 20

    # plain wall-clock timing (for comparing before/after an optimization)
    python scripts/benchmark_solver.py --steps 50

    # save the final field state as a regression golden reference
    python scripts/benchmark_solver.py --steps 50 --save-golden tests/fixtures/regression_golden.npz

    # compare a fresh run against a previously saved golden reference
    python scripts/benchmark_solver.py --steps 50 --compare-golden tests/fixtures/regression_golden.npz
"""
from __future__ import annotations

import argparse
import cProfile
import pstats
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from air_vortex.config import load_config  # noqa: E402
from air_vortex.solver import build_solver  # noqa: E402

FIELD_KEYS = ("u_r", "u_z", "u_theta", "p", "phi", "rho")


def run_steps(cfg, n_steps: int):
    solver = build_solver(cfg, swirl_mode="forced")
    for _ in range(n_steps):
        solver.step()
    return solver


def save_golden(solver, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = {k: getattr(solver.fields, k) for k in FIELD_KEYS}
    data["t"] = solver.fields.t
    data["step"] = solver.fields.step
    np.savez(out_path, **data)
    print(f"Saved golden reference to {out_path}")


def compare_golden(solver, golden_path: Path, rtol: float, atol: float) -> bool:
    golden = np.load(golden_path)
    ok = True
    print(f"{'field':<10} {'max_abs_diff':>14} {'max_rel_diff':>14} {'status':>8}")
    for key in FIELD_KEYS:
        actual = getattr(solver.fields, key)
        expected = golden[key]
        if actual.shape != expected.shape:
            print(f"{key:<10} SHAPE MISMATCH: {actual.shape} vs {expected.shape}")
            ok = False
            continue
        abs_diff = np.abs(actual - expected)
        rel_diff = abs_diff / (np.abs(expected) + atol)
        passed = np.allclose(actual, expected, rtol=rtol, atol=atol)
        ok &= passed
        print(f"{key:<10} {abs_diff.max():14.3e} {rel_diff.max():14.3e} "
              f"{'OK' if passed else 'FAIL':>8}")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/benchmark.yaml")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--profile", action="store_true", help="run under cProfile instead of plain timing")
    parser.add_argument("--profile-top", type=int, default=25)
    parser.add_argument("--save-golden", default=None)
    parser.add_argument("--compare-golden", default=None)
    parser.add_argument("--rtol", type=float, default=1e-6)
    parser.add_argument("--atol", type=float, default=1e-10)
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.profile:
        profiler = cProfile.Profile()
        profiler.enable()
        solver = run_steps(cfg, args.steps)
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats("cumulative")
        stats.print_stats(args.profile_top)
    else:
        t0 = time.perf_counter()
        solver = run_steps(cfg, args.steps)
        elapsed = time.perf_counter() - t0
        print(f"grid: {solver.grid.Nr} x {solver.grid.Nz} = {solver.grid.Nr*solver.grid.Nz} cells")
        print(f"{args.steps} steps in {elapsed:.3f} s ({elapsed/args.steps*1000:.2f} ms/step)")
        print(f"final t = {solver.fields.t:.6g} s")

    if args.save_golden:
        save_golden(solver, Path(args.save_golden))

    if args.compare_golden:
        ok = compare_golden(solver, Path(args.compare_golden), args.rtol, args.atol)
        print("REGRESSION: " + ("PASS" if ok else "FAIL"))
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
