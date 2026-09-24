"""Effective stir-bar forcing (README section 10)."""
from __future__ import annotations

import numpy as np

from .config import Config
from .grid import Grid


def forcing_mask(grid: Grid, cfg: Config) -> np.ndarray:
    """chi(r,z) in [0,1], smooth top-hat over the swept stirrer volume
    (README section 10.1)."""
    R_m = cfg.geometry.stirbar_half_length_m
    D_m = cfg.geometry.stirbar_diameter_m
    eps_f = cfg.stirrer.forcing_smoothing_m

    r = grid.r_c[:, None]
    z = grid.z_c[None, :]

    chi_r = 0.5 * (1.0 - np.tanh((r - R_m) / eps_f))
    chi_z = 0.5 * (1.0 - np.tanh((z - D_m) / eps_f))
    return chi_r * chi_z


def stirrer_forcing(u_theta: np.ndarray, omega: float, grid: Grid, cfg: Config,
                     chi: np.ndarray) -> np.ndarray:
    """f_theta = chi * (u_theta,target - u_theta) / tau_s (README section 10.2)."""
    u_target = omega * grid.r_c[:, None]
    tau_s = cfg.stirrer.forcing_tau_s
    return chi * (u_target - u_theta) / tau_s
