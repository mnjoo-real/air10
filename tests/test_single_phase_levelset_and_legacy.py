"""Level Set transport in the single-phase path, reinitialization drift
recording, and the legacy two_phase_diffuse_ls path being unchanged."""
import copy
from pathlib import Path

import numpy as np

from air_vortex.config import PhysicsConfig, load_config
from air_vortex.levelset import advect_level_set_advective, levelset_rhs_muscl2
from air_vortex.levelset import levelset_rhs_muscl2_advective
from air_vortex.single_phase_solver import build_single_phase_solver
from air_vortex.solver import Solver, build_solver
from _config_helpers import single_phase_config
from _helpers import make_grid

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "tests" / "fixtures" / "regression_golden.npz"


def test_levelset_zero_flow_invariance():
    g = make_grid(Nr=10, Nz=12, dr=0.002, dz=0.002)
    R, Z = np.meshgrid(g.r_c, g.z_c, indexing="ij")
    phi = Z - 0.011 - 0.003 * np.cos(R / g.r_v * np.pi)
    for integ in ("euler", "ssprk2"):
        out = advect_level_set_advective(phi, np.zeros(g.shape_ur), np.zeros(g.shape_uz), g, 1e-3,
                                         scheme="muscl2", time_integrator=integ)
        np.testing.assert_array_equal(out, phi)


def test_advective_form_equals_flux_form_for_solenoidal_velocity():
    g = make_grid(Nr=10, Nz=12, dr=0.002, dz=0.002)
    phi = g.z_c[None, :] - 0.0113 + 0 * g.r_c[:, None]
    u_z = np.full(g.shape_uz, 0.05)
    u_z[:, 0] = 0.05   # uniform axial flow: div u = 0 except none at boundaries
    u_r = np.zeros(g.shape_ur)
    a = levelset_rhs_muscl2_advective(phi, u_r, u_z, g)
    b = levelset_rhs_muscl2(phi, u_r, u_z, g)
    np.testing.assert_allclose(a, b, atol=1e-15)


def test_single_phase_hydrostatic_phi_does_not_move():
    s = build_single_phase_solver(single_phase_config())
    phi0 = s.fields.phi.copy()
    for _ in range(100):
        s.step()
    np.testing.assert_allclose(s.fields.phi, phi0, rtol=0, atol=1e-15)


def test_reinit_displacement_is_recorded_and_zero_for_flat_distance():
    s = build_single_phase_solver(single_phase_config(reinit_every=1))
    d = s.step()
    assert d.reinit_applied
    assert d.reinit_max_eta_shift < 1e-12
    assert abs(d.reinit_volume_change) < 1e-15


def test_legacy_model_explicit_selection_matches_golden():
    """two_phase_diffuse_ls selected EXPLICITLY must reproduce the
    pre-redesign golden fixture (same tolerance as the regression test)."""
    cfg = load_config(ROOT / "configs" / "benchmark.yaml")
    cfg = copy.deepcopy(cfg)
    cfg.physics = PhysicsConfig(free_surface_model="two_phase_diffuse_ls")
    solver = build_solver(cfg, swirl_mode="forced")
    assert type(solver) is Solver
    for _ in range(50):
        solver.step()
    golden = np.load(GOLDEN)
    for key in ("u_r", "u_z", "u_theta", "p", "phi", "rho"):
        np.testing.assert_allclose(getattr(solver.fields, key), golden[key], rtol=1e-6, atol=1e-8)
