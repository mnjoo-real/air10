"""Objective air-core connectivity criterion (README section 22)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .config import Config
from .grid import Grid


@dataclass
class ConnectivityState:
    connected: bool
    t_connected_start: float | None = None


def air_mask(phi: np.ndarray) -> np.ndarray:
    """A(r,z) = 1 where phi>0 (air), 0 otherwise (README section 22)."""
    return (phi > 0).astype(np.uint8)


def top_connected_air(mask: np.ndarray) -> np.ndarray:
    """Boolean mask of the connected air component touching the top row
    (z=Z_max), i.e. A_top (README section 22)."""
    labeled, _ = ndimage.label(mask)
    top_labels = set(np.unique(labeled[:, -1]))
    top_labels.discard(0)
    if not top_labels:
        return np.zeros_like(mask, dtype=bool)
    return np.isin(labeled, list(top_labels))


def stirbar_target_region(grid: Grid, cfg: Config) -> np.ndarray:
    """Boolean mask for B_delta: the effective stir-bar region expanded by
    a few grid cells (README section 22)."""
    R_m = cfg.geometry.stirbar_half_length_m
    D_m = cfg.geometry.stirbar_diameter_m
    delta_r = cfg.air_core.contact_tolerance_cells * grid.dr
    delta_z = cfg.air_core.contact_tolerance_cells * grid.dz

    r = grid.r_c[:, None]
    z = grid.z_c[None, :]
    return (r <= R_m + delta_r) & (z <= D_m + delta_z)


def is_geometrically_connected(phi: np.ndarray, grid: Grid, cfg: Config) -> bool:
    """A_top ∩ B_delta != empty (README section 22, boxed criterion)."""
    a_top = top_connected_air(air_mask(phi))
    b_delta = stirbar_target_region(grid, cfg)
    return bool(np.any(a_top & b_delta))


def update_persistence(state: ConnectivityState, connected_now: bool, t: float,
                        cfg: Config, N: float) -> ConnectivityState:
    """Track persistence per README section 22 ("Persistence criterion"):
    require geometric connection to hold continuously for
    ``t_connected >= 300/N`` seconds (persistence_rotations rotation periods)."""
    required = cfg.air_core.persistence_rotations * 60.0 / max(N, 1e-9)

    if not connected_now:
        return ConnectivityState(connected=False, t_connected_start=None)

    # Keep the start of the current contact streak fixed for as long as
    # contact continues -- gate on whether a streak is already in progress
    # (t_connected_start is not None), not on whether it has already
    # persisted long enough (state.connected), which would keep resetting
    # the clock back to "now" on every call and never accumulate duration.
    start = state.t_connected_start if state.t_connected_start is not None else t
    persisted = (t - start) >= required
    return ConnectivityState(connected=persisted, t_connected_start=start)
