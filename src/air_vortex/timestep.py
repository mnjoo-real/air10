"""Adaptive time-step selection (README section 16)."""
from __future__ import annotations

import numpy as np

from .config import Config
from .grid import Grid

_VELOCITY_EPS = 1e-6


def compute_stable_timestep(grid: Grid, u_r: np.ndarray, u_z: np.ndarray,
                             rho_min: float, mu_max: float, cfg: Config) -> float:
    dr, dz = grid.dr, grid.dz
    dx = min(dr, dz)

    # convective (CFL) constraint
    dt_adv = cfg.time.cfl * min(
        dr / (np.max(np.abs(u_r)) + _VELOCITY_EPS),
        dz / (np.max(np.abs(u_z)) + _VELOCITY_EPS),
    )

    # viscous constraint for explicit diffusion
    C_nu = 0.2
    dt_visc = C_nu * rho_min * dx**2 / max(mu_max, 1e-12)

    # capillary constraint
    C_sigma = 0.5
    sigma = cfg.fluid.surface_tension
    if sigma > 0:
        dt_cap = C_sigma * np.sqrt(rho_min * dx**3 / sigma)
    else:
        dt_cap = np.inf

    # forcing time scale must be resolved
    dt_forcing = 0.2 * cfg.stirrer.forcing_tau_s

    dt = min(dt_adv, dt_visc, dt_cap, dt_forcing, cfg.time.dt_max_s)
    return float(dt)
