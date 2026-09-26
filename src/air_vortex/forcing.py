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

    model = getattr(cfg.stirrer, "model", "wall_touching_volume")
    if model == "moving_footprint":
        raise ValueError("stirrer.model='moving_footprint' has no volume forcing (use MovingFootprintWallBC)")
    chi_r = 0.5 * (1.0 - np.tanh((r - R_m) / eps_f))
    chi_z = 0.5 * (1.0 - np.tanh((z - D_m) / eps_f))
    if model == "tapered_volume":
        chi_z = chi_z * bottom_taper(z, cfg.stirrer.bottom_taper_m)
    return chi_r * chi_z


def bottom_taper(z, ell):
    """C2 smoothstep S(z / ell) = xi^3 (10 - 15 xi + 6 xi^2), 0 at the bottom (with zero first and
    second derivatives), 1 for z >= ell. ell is a PHYSICAL length (V7-S model M1)."""
    xi = np.clip(np.asarray(z, float) / ell, 0.0, 1.0)
    return xi**3 * (10.0 - 15.0 * xi + 6.0 * xi**2)


def footprint_weight(r, cfg: Config):
    """w(r) = (1 - tanh((r - R_m)/eps))/2 for the moving-footprint model (V7-S M2): a smooth
    transition of fixed physical width eps = stirrer.forcing_smoothing_m, never a cell count."""
    R_m = cfg.geometry.stirbar_half_length_m
    eps_f = cfg.stirrer.forcing_smoothing_m
    return 0.5 * (1.0 - np.tanh((np.asarray(r, float) - R_m) / eps_f))


class MovingFootprintWallBC:
    """V7-S model M2 (single_phase_height only): an effective axisymmetric moving surface that
    REPLACES the stationary no-slip bottom under the stir bar, u_theta(r, z=0) = Omega(t) r w(r);
    the side wall stays stationary. This is not a rotating glass bottom and not the real
    (non-axisymmetric) bar: an effective stir-bar/fluid coupling boundary. No volume forcing is
    applied with it (no double counting)."""
    kind = "moving_footprint"

    def __init__(self, cfg: Config, time_fn):
        self.cfg = cfg
        self.time_fn = time_fn

    def wall_u_theta(self, grid: Grid) -> np.ndarray:
        return np.zeros(grid.Nz)

    def bottom_u_theta(self, grid: Grid) -> np.ndarray:
        return self.cfg.stirrer.omega(self.time_fn()) * grid.r_c * footprint_weight(grid.r_c, self.cfg)


def stirrer_forcing(u_theta: np.ndarray, omega: float, grid: Grid, cfg: Config,
                     chi: np.ndarray) -> np.ndarray:
    """f_theta = chi * (u_theta,target - u_theta) / tau_s (README section 10.2)."""
    u_target = omega * grid.r_c[:, None]
    tau_s = cfg.stirrer.forcing_tau_s
    return chi * (u_target - u_theta) / tau_s
