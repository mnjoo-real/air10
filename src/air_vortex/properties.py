"""Smoothed material properties from the Level Set field (README section 9.1)."""
from __future__ import annotations

import numpy as np

from .config import Config
from .grid import Grid


def smoothed_heaviside(phi: np.ndarray, eps: float) -> np.ndarray:
    """H_eps(phi): 0 in water (phi<0), 1 in air (phi>0), smooth transition
    of half-width ``eps`` across the interface."""
    h = np.empty_like(phi)
    inside = np.abs(phi) <= eps
    h[phi > eps] = 1.0
    h[phi < -eps] = 0.0
    x = phi[inside] / eps
    h[inside] = 0.5 * (1.0 + x + np.sin(np.pi * x) / np.pi)
    return h


def smoothed_delta(phi: np.ndarray, eps: float) -> np.ndarray:
    """delta_eps(phi), the derivative of smoothed_heaviside w.r.t. phi."""
    d = np.zeros_like(phi)
    inside = np.abs(phi) <= eps
    x = phi[inside] / eps
    d[inside] = (1.0 + np.cos(np.pi * x)) / (2.0 * eps)
    return d


def interface_epsilon(grid: Grid, cfg: Config) -> float:
    dx = 0.5 * (grid.dr + grid.dz)
    return cfg.levelset.interface_width_cells * dx


def material_properties(phi: np.ndarray, grid: Grid, cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    """rho(phi), mu(phi) per README section 9.1."""
    eps = interface_epsilon(grid, cfg)
    h = smoothed_heaviside(phi, eps)
    rho_w, rho_a = cfg.fluid.water_density, cfg.fluid.air_density
    mu_w, mu_a = cfg.fluid.water_viscosity, cfg.fluid.air_viscosity
    rho = rho_w + (rho_a - rho_w) * h
    mu = mu_w + (mu_a - mu_w) * h
    return rho, mu
