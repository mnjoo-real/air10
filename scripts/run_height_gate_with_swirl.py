"""Run a validate_height_branch.py gate with a given single_phase_height swirl transport
(V7-T section 26 regression): python scripts/run_height_gate_with_swirl.py SWIRL[+am] <gate args...>
("+am" also selects the angular-momentum viscous form)"""
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from air_vortex.height_solver import SinglePhaseHeightSolver  # noqa: E402

MODE, _, VISC = sys.argv[1].partition("+")
_orig = SinglePhaseHeightSolver.__post_init__


def _post_init(self):
    if self.swirl_advection is None:
        self.swirl_advection = MODE
    if VISC == "am" and self.swirl_viscous is None:
        self.swirl_viscous = "angular_momentum"
    if MODE == "advective":
        self.cfg.physics.height_legacy_swirl_for_validation = True
    _orig(self)


SinglePhaseHeightSolver.__post_init__ = _post_init
sys.argv = [str(ROOT / "scripts" / "validate_height_branch.py")] + sys.argv[2:]
runpy.run_path(sys.argv[0], run_name="__main__")
