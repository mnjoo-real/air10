"""Architecture switch (README_rewritten 19.2): model selection, guards,
and separation of the two runtime paths."""
import copy
import warnings
from pathlib import Path

import pytest
import yaml

from air_vortex.config import PhysicsConfig, config_from_dict, config_to_dict, load_config
from air_vortex.single_phase_solver import SinglePhaseSolver
from air_vortex.solver import Solver, ValidationOnlyWarning, build_solver
from _config_helpers import single_phase_config, small_config

ROOT = Path(__file__).resolve().parents[1]


def test_default_model_is_legacy_during_transition():
    assert small_config().physics.free_surface_model == "two_phase_diffuse_ls"
    cfg = load_config(ROOT / "configs" / "benchmark.yaml")  # no physics section
    assert cfg.physics.free_surface_model == "two_phase_diffuse_ls"


def test_yaml_physics_section_selects_single_phase(tmp_path):
    raw = yaml.safe_load(open(ROOT / "configs" / "benchmark.yaml", encoding="utf-8"))
    raw["physics"] = {"free_surface_model": "single_phase_ls"}
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    assert load_config(p).physics.free_surface_model == "single_phase_ls"


def test_single_phase_config_files_select_single_phase():
    for name in ("single_phase_hydrostatic.yaml", "single_phase_rigid_body.yaml"):
        cfg = load_config(ROOT / "configs" / name)
        assert cfg.physics.free_surface_model == "single_phase_ls"
        assert cfg.fluid.surface_tension == 0.0


def test_unknown_model_rejected():
    with pytest.raises(ValueError):
        PhysicsConfig(free_surface_model="vof")


def test_config_roundtrip_keeps_model():
    cfg = single_phase_config()
    again = config_from_dict(config_to_dict(cfg))
    assert again.physics == cfg.physics


def test_build_solver_dispatches_by_model():
    assert type(build_solver(small_config())) is Solver
    assert isinstance(build_solver(single_phase_config()), SinglePhaseSolver)


def test_single_phase_refuses_prescribed_swirl():
    with pytest.raises(ValueError):
        build_solver(single_phase_config(), swirl_mode="prescribed")


def test_legacy_prescribed_swirl_warns_validation_only():
    with pytest.warns(ValidationOnlyWarning):
        build_solver(small_config(), swirl_mode="prescribed")
    with warnings.catch_warnings():
        warnings.simplefilter("error", ValidationOnlyWarning)
        build_solver(small_config(), swirl_mode="forced")


def test_single_phase_surface_tension_is_sharp_jump_not_csf():
    """Milestone D (Gate V4) lifted the sigma=0 restriction: sigma > 0 is
    accepted (applied as the interface pressure jump), sigma < 0 rejected."""
    cfg = single_phase_config()
    cfg.fluid.surface_tension = 0.072
    assert isinstance(build_solver(cfg), SinglePhaseSolver)
    cfg.fluid.surface_tension = -0.01
    with pytest.raises(ValueError):
        build_solver(cfg)


def test_single_phase_refuses_stirrer_forcing_until_milestone_e():
    cfg = copy.deepcopy(single_phase_config())
    cfg.stirrer.rpm = 500.0
    with pytest.raises(NotImplementedError):
        build_solver(cfg)


def test_single_phase_path_imports_no_two_phase_material_code():
    """No rho(phi)/mu(phi), CSF, or variable-density Poisson in Level 1A."""
    import air_vortex.single_phase_solver as sps
    import air_vortex.pressure_single_phase as psp
    for mod in (sps, psp):
        src = Path(mod.__file__).read_text(encoding="utf-8")
        for forbidden in ("material_properties", "surface_tension_force",
                          "from .pressure import", "smoothed_heaviside"):
            assert forbidden not in src, f"{mod.__name__} uses {forbidden}"
