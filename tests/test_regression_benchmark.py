"""Quantitative regression guard for the pressure-solver / numerical-kernel
performance work (README "Performance"): reruns the fixed benchmark
scenario (configs/benchmark.yaml, 50 steps, swirl_mode='forced') and checks
the resulting fields against a golden reference captured before that
optimization work started (scripts/benchmark_solver.py --save-golden).

If this test ever needs its golden reference regenerated (i.e. a solver
change *intentionally* changes the physical result, not just how it's
computed), regenerate it explicitly:

    python scripts/benchmark_solver.py --steps 50 \
        --save-golden tests/fixtures/regression_golden.npz

and explain why in the commit -- silently regenerating it would defeat the
point of a regression guard.
"""
from pathlib import Path

import numpy as np
import pytest

from air_vortex.config import load_config
from air_vortex.solver import build_solver

GOLDEN_PATH = Path(__file__).parent / "fixtures" / "regression_golden.npz"
FIELD_KEYS = ("u_r", "u_z", "u_theta", "p", "phi", "rho")
N_STEPS = 50


@pytest.mark.skipif(not GOLDEN_PATH.exists(), reason="golden reference fixture not found")
def test_benchmark_scenario_matches_golden_reference():
    cfg = load_config(str(Path(__file__).parent.parent / "configs" / "benchmark.yaml"))
    solver = build_solver(cfg, swirl_mode="forced")
    for _ in range(N_STEPS):
        solver.step()

    golden = np.load(GOLDEN_PATH)

    for key in FIELD_KEYS:
        actual = getattr(solver.fields, key)
        expected = golden[key]
        assert actual.shape == expected.shape, f"{key}: shape changed"
        # loose enough to tolerate a different (still exact) direct-solve
        # elimination order (see pressure.py permc_spec), tight enough to
        # catch a genuine change in the physics/discretization
        np.testing.assert_allclose(
            actual, expected, rtol=1e-6, atol=1e-8,
            err_msg=f"{key} drifted from the golden reference",
        )
