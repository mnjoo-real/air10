"""V7-S: stirrer-bottom model candidates (M0 wall-touching volume, M1 tapered volume,
M2 moving footprint). The default (M0) must stay bit-identical."""
import copy
import dataclasses
from pathlib import Path

import numpy as np
import pytest

from air_vortex.config import load_config
from air_vortex.forcing import bottom_taper, footprint_weight, forcing_mask
from air_vortex.grid import build_grid

ROOT = Path(__file__).resolve().parents[1]


def _cfg(model="wall_touching_volume", ell=0.0, dx=1e-3):
    cfg = copy.deepcopy(load_config(ROOT / "configs" / "baseline.yaml"))
    cfg.grid.dr_m = cfg.grid.dz_m = dx
    cfg.stirrer = dataclasses.replace(cfg.stirrer, model=model, bottom_taper_m=ell)
    return cfg


def test_default_model_is_wall_touching_and_mask_is_bit_identical_to_legacy_formula():
    cfg = _cfg()
    assert load_config(ROOT / "configs" / "baseline.yaml").stirrer.model == "wall_touching_volume"
    g = build_grid(cfg)
    R_m, D_m, eps = cfg.geometry.stirbar_half_length_m, cfg.geometry.stirbar_diameter_m, cfg.stirrer.forcing_smoothing_m
    legacy = (0.5 * (1 - np.tanh((g.r_c[:, None] - R_m) / eps))) * (0.5 * (1 - np.tanh((g.z_c[None, :] - D_m) / eps)))
    assert np.array_equal(forcing_mask(g, cfg), legacy)


def test_bottom_taper_is_c2_and_zero_at_the_bottom():
    ell = 1.75e-3
    z = np.linspace(0, 2 * ell, 2001)
    S = bottom_taper(z, ell)
    assert S[0] == 0.0 and np.all(S[z >= ell] == 1.0) and np.all(np.diff(S) >= 0)
    h = 1e-9
    for z0 in (0.0, ell):                           # first and second derivatives vanish at both ends
        d1 = (bottom_taper(z0 + h, ell) - bottom_taper(max(z0 - h, 0), ell)) / (h if z0 == 0 else 2 * h)
        assert abs(d1) < 1e-4 / ell


def test_tapered_volume_vanishes_at_the_wall_and_needs_a_physical_length():
    cfg = _cfg("tapered_volume", 1.0e-3, dx=0.25e-3)
    g = build_grid(cfg)
    chi = forcing_mask(g, cfg)
    assert chi[:, 0].max() < 0.02                   # first cell centre z = dx/2 = ell/8
    with pytest.raises(ValueError):
        _cfg("tapered_volume", 0.0)
    with pytest.raises(ValueError):
        _cfg("not_a_model")


def test_moving_footprint_has_no_volume_mask_and_a_smooth_fixed_width_edge():
    cfg = _cfg("moving_footprint")
    with pytest.raises(ValueError):
        forcing_mask(build_grid(cfg), cfg)
    r = np.linspace(0, 0.045, 4501)
    w = footprint_weight(r, cfg)
    assert w[0] == pytest.approx(1.0, abs=1e-12) and w[-1] < 1e-12
    assert np.abs(np.diff(w) / np.diff(r)).max() < 0.51 / cfg.stirrer.forcing_smoothing_m   # |w'| <= 1/(2 eps)


def test_moving_footprint_solver_replaces_the_bottom_and_splits_the_torque():
    from air_vortex.height_benchmarks import build_height_production
    from air_vortex.height_solver import SinglePhaseHeightSolver
    s0 = build_height_production(300.0, dx=1e-3)
    s0.cfg.stirrer = dataclasses.replace(s0.cfg.stirrer, model="moving_footprint")
    s = SinglePhaseHeightSolver(grid=s0.grid, cfg=s0.cfg, fields=s0.fields, eta=s0.eta, z_pin=s0.z_pin)
    assert s._chi is None and s.wall_bc.kind == "moving_footprint"
    s.rigid_lid = s.freeze_interface = True
    for _ in range(20):
        s.step()
    b = s.torque_budget()
    assert b["T_stir"] == 0.0 and b["T_bottom"] > 0.0                     # the moving bottom drives the flow
    assert b["T_bottom_under"] + b["T_bottom_outside"] == pytest.approx(b["T_bottom"], rel=1e-12)
    ub = s.wall_bc.bottom_u_theta(s.grid)
    assert ub[-1] < 1e-9 and ub[0] > 0.0
