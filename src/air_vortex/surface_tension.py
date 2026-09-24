"""Continuum Surface Force (CSF) surface tension (README section 9.2)."""
from __future__ import annotations

import numpy as np

from .config import Config
from .grid import Grid
from .operators import center_grad_r, center_grad_z, interp_center_to_ur, interp_center_to_uz
from .properties import interface_epsilon, smoothed_delta

_EPS_N = 1e-12


def interface_normal(grid: Grid, phi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """n = grad(phi) / (|grad(phi)| + eps_n) (README section 9.2)."""
    dphidr = center_grad_r(grid, phi)
    dphidz = center_grad_z(grid, phi)
    mag = np.sqrt(dphidr**2 + dphidz**2) + _EPS_N
    return dphidr / mag, dphidz / mag


def interface_curvature(grid: Grid, n_r: np.ndarray, n_z: np.ndarray) -> np.ndarray:
    """kappa = div(n), with the cylindrical divergence of a cell-centered
    vector field approximated by centered differences."""
    r_c = grid.r_c[:, None]
    d_r_nr = center_grad_r(grid, r_c * n_r) / r_c
    d_z_nz = center_grad_z(grid, n_z)
    return d_r_nr + d_z_nz


def surface_tension_force(grid: Grid, phi: np.ndarray, cfg: Config
                           ) -> tuple[np.ndarray, np.ndarray]:
    """F_sigma = sigma * kappa * delta_eps(phi) * n, returned on the u_r and
    u_z staggered faces so it can be added directly to the momentum
    predictor (README section 9.2, boxed equation)."""
    n_r, n_z = interface_normal(grid, phi)
    kappa = interface_curvature(grid, n_r, n_z)
    eps = interface_epsilon(grid, cfg)
    delta = smoothed_delta(phi, eps)

    f_center_r = cfg.fluid.surface_tension * kappa * delta * n_r
    f_center_z = cfg.fluid.surface_tension * kappa * delta * n_z

    f_r = interp_center_to_ur(f_center_r)
    f_z = interp_center_to_uz(f_center_z)
    return f_r, f_z
