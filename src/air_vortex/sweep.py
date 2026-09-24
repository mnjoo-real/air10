"""RPM / water-depth parameter sweeps and critical-RPM bisection
(README sections 19, 20, 26 Study A/B)."""
from __future__ import annotations

from dataclasses import dataclass

from .config import Config, with_overrides
from .connectivity import ConnectivityState, is_geometrically_connected, update_persistence
from .solver import Solver, build_solver


@dataclass
class CaseResult:
    rpm: float
    water_height_m: float
    formed: bool
    d_infinity: float
    t_end: float


def run_case(cfg: Config, t_end: float | None = None) -> CaseResult:
    """Run a single (RPM, H) case to completion and report whether a
    persistent air core formed (README section 22)."""
    from .diagnostics import vortex_depth

    solver = build_solver(cfg, swirl_mode="forced")
    state = ConnectivityState(connected=False)
    t_end = cfg.time.t_end_s if t_end is None else t_end

    d_last = 0.0
    while solver.fields.t < t_end:
        solver.step()
        connected_now = is_geometrically_connected(solver.fields.phi, solver.grid, cfg)
        state = update_persistence(state, connected_now, solver.fields.t, cfg, cfg.stirrer.rpm)
        d_last = vortex_depth(solver.fields.phi, solver.grid, cfg)
        if state.connected:
            break

    return CaseResult(
        rpm=cfg.stirrer.rpm,
        water_height_m=cfg.geometry.water_height_m,
        formed=state.connected,
        d_infinity=d_last,
        t_end=solver.fields.t,
    )


def bracket_transition(base_cfg: Config, rpm_coarse: list[float], water_height_m: float,
                        t_end: float | None = None) -> tuple[float, float] | None:
    """Find (N_no_core, N_core) from a coarse RPM list (README section 20)."""
    prev_rpm = None
    prev_formed = False
    for rpm in sorted(rpm_coarse):
        cfg = with_overrides(base_cfg, rpm=rpm, water_height_m=water_height_m)
        result = run_case(cfg, t_end=t_end)
        if result.formed and not prev_formed and prev_rpm is not None:
            return (prev_rpm, rpm)
        if result.formed and prev_rpm is None:
            return (rpm, rpm)  # already forms at the lowest tested RPM
        prev_rpm, prev_formed = rpm, result.formed
    return None  # never formed in the tested range


def bisect_critical_rpm(base_cfg: Config, low: float, high: float, water_height_m: float,
                         target_resolution_rpm: float = 5.0, max_iterations: int = 8,
                         t_end: float | None = None) -> float:
    """Bisect the (no-core, core) RPM interval to the target resolution
    (README section 20)."""
    for _ in range(max_iterations):
        if (high - low) <= target_resolution_rpm:
            break
        mid = round((low + high) / 2.0)
        cfg = with_overrides(base_cfg, rpm=mid, water_height_m=water_height_m)
        result = run_case(cfg, t_end=t_end)
        if result.formed:
            high = mid
        else:
            low = mid
    return high


def find_critical_rpm(base_cfg: Config, rpm_coarse: list[float], water_height_m: float,
                       target_resolution_rpm: float = 5.0, max_iterations: int = 8,
                       t_end: float | None = None) -> float | None:
    bracket = bracket_transition(base_cfg, rpm_coarse, water_height_m, t_end=t_end)
    if bracket is None:
        return None
    low, high = bracket
    if low == high:
        return float(high)
    return bisect_critical_rpm(base_cfg, low, high, water_height_m,
                                target_resolution_rpm, max_iterations, t_end=t_end)


def depth_sweep(base_cfg: Config, water_heights_m: list[float], rpm_coarse: list[float],
                 target_resolution_rpm: float = 5.0, max_iterations: int = 8,
                 t_end: float | None = None) -> dict[float, float | None]:
    """N_c(H) (README sections 19, 20, Study B)."""
    return {
        H: find_critical_rpm(base_cfg, rpm_coarse, H, target_resolution_rpm,
                              max_iterations, t_end=t_end)
        for H in water_heights_m
    }
