"""Field allocation for the staggered grid (README section 14.1)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .grid import Grid


@dataclass
class Fields:
    u_r: np.ndarray
    u_z: np.ndarray
    u_theta: np.ndarray
    p: np.ndarray
    phi: np.ndarray
    rho: np.ndarray
    mu: np.ndarray
    t: float = 0.0
    step: int = 0


def allocate_fields(grid: Grid) -> Fields:
    return Fields(
        u_r=np.zeros(grid.shape_ur),
        u_z=np.zeros(grid.shape_uz),
        u_theta=np.zeros(grid.shape_center),
        p=np.zeros(grid.shape_center),
        phi=np.zeros(grid.shape_center),
        rho=np.zeros(grid.shape_center),
        mu=np.zeros(grid.shape_center),
    )


def initialize_still_water(grid: Grid, water_height: float) -> Fields:
    """Initial condition from README section 13: u=0, phi = z - H."""
    fields = allocate_fields(grid)
    z2d = np.broadcast_to(grid.z_c[None, :], grid.shape_center)
    fields.phi = z2d - water_height
    return fields


def initialize_rotating_equilibrium(grid: Grid, cfg, omega: float,
                                     target_volume: float | None = None) -> Fields:
    """Exact analytical rotating hydrostatic equilibrium (README "Exact
    Equilibrium Preservation Test"): solid-body swirl everywhere, a free
    surface exactly matching the (surface-tension-free) volume-consistent
    parabola, and a pressure field satisfying

        dp/dr = rho*omega^2*r,  dp/dz = -rho*g,  p=0 (gauge) at the surface

    i.e. p(r,z) = 0.5*rho*omega^2*r^2 - rho*g*(z - C) in the water, 0 in
    the air, where C = eta(0) (so p=0 exactly at z=eta(r) for every r --
    verified: substituting eta(r)=C+omega^2 r^2/(2g) into the formula
    above gives p=0 there for any r).

    This is the CASE P (sigma=0) construction: ignores surface tension
    regardless of cfg.fluid.surface_tension -- the caller is responsible
    for also zeroing cfg.fluid.surface_tension on the Config actually used
    to run the solver, so the initial condition and the equations being
    solved agree (README section 1: "이 두 case를 절대로 섞지 마라"). There is no
    equivalent closed form for the capillary-corrected (CASE C) case --
    that reference is built numerically, see capillary_equilibrium.py.
    """
    from .diagnostics import volume_consistent_parabola

    if target_volume is None:
        target_volume = np.pi * grid.r_v**2 * cfg.geometry.water_height_m

    fields = allocate_fields(grid)
    fields.u_theta = omega * grid.r_c[:, None] * np.ones(grid.shape_center)
    # u_r, u_z stay exactly zero (already zero from allocate_fields)

    g = cfg.fluid.gravity
    rho_w = cfg.fluid.water_density
    eta_r = volume_consistent_parabola(grid.r_c, grid.r_v, omega, g, target_volume)
    C = float(eta_r[0])

    fields.phi = grid.z_c[None, :] - eta_r[:, None]

    r2d = grid.r_c[:, None] * np.ones(grid.shape_center)
    z2d = grid.z_c[None, :] * np.ones(grid.shape_center)
    p_water = 0.5 * rho_w * omega**2 * r2d**2 - rho_w * g * (z2d - C)
    fields.p = np.where(fields.phi < 0.0, p_water, 0.0)

    return fields
