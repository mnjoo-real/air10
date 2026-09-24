"""Static capillary benchmarks for Gate V4 (single_phase_ls).

CAP-A: isolated liquid sphere, gravity 0, u = 0, the EXACT curvature 2/R
       supplied to the pressure BC -- verifies the sharp pressure-jump
       machinery (Dirichlet at the sub-cell crossing, projection) alone.
CAP-B: the same sphere with curvature computed numerically from phi --
       whatever spurious current remains is attributable to curvature error.

The sphere is centered on the axis and kept >= 3R/2 away from the wall,
bottom and top: no contact line (that is V4b).
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np

from .config import Config, load_config
from .grid import build_grid
from .single_phase_solver import SinglePhaseSolver

ROOT = Path(__file__).resolve().parents[2]
SIGMA_WATER = 0.072


def sphere_config(dx: float, sphere_radius: float, sigma: float = SIGMA_WATER,
                  reinit_every: int = 0, capillary_dt_factor: float = 1.0,
                  dt_max: float = 1.0) -> Config:
    """Domain R_v = 2 R_s, height 4 R_s; gravity 0; no stirring. dt_max is
    large so that the capillary limit (or viscous/CFL) actually binds."""
    cfg = copy.deepcopy(load_config(ROOT / "configs" / "single_phase_hydrostatic.yaml"))
    cfg.geometry.vessel_radius_m = 2.0 * sphere_radius
    cfg.geometry.water_height_m = 2.0 * sphere_radius   # unused: phi is set explicitly
    cfg.geometry.air_height_m = 2.0 * sphere_radius
    cfg.grid.dr_m = cfg.grid.dz_m = dx
    cfg.fluid.gravity = 0.0
    cfg.fluid.surface_tension = sigma
    cfg.time.dt_max_s = dt_max
    cfg.levelset.reinitialize_every = reinit_every
    cfg.physics.capillary_dt_factor = capillary_dt_factor
    return cfg


def sphere_phi(grid, sphere_radius: float) -> tuple[np.ndarray, float]:
    """Signed distance to a sphere centred on the axis at mid-height,
    liquid inside. The centre is shifted off the cell centers so the poles
    are sub-cell."""
    z0 = 0.5 * grid.z_max + 0.37 * grid.dz
    R, Z = np.meshgrid(grid.r_c, grid.z_c, indexing="ij")
    return np.hypot(R, Z - z0) - sphere_radius, z0


def analytic_sphere_curvature(sphere_radius: float):
    """curvature_fn for CAP-A: kappa = 2/R at every crossing."""
    def fn(grid, geom, phi):
        from .free_surface_bc import interface_pressure
        pr, pz = interface_pressure(geom)            # nan off the mixed faces
        k = 2.0 / sphere_radius
        return np.where(np.isfinite(pr), k, np.nan), np.where(np.isfinite(pz), k, np.nan)
    return fn


def build_sphere_solver(dx: float, sphere_radius: float, curvature: str = "numerical",
                        **cfg_kwargs) -> tuple[SinglePhaseSolver, float]:
    cfg = sphere_config(dx, sphere_radius, **cfg_kwargs)
    grid = build_grid(cfg)
    from .fields import allocate_fields
    fields = allocate_fields(grid)
    fields.phi, z0 = sphere_phi(grid, sphere_radius)
    fields.p = np.where(fields.phi < 0, 2.0 * cfg.fluid.surface_tension / sphere_radius, 0.0)
    fn = analytic_sphere_curvature(sphere_radius) if curvature == "analytic" else None
    if curvature not in ("analytic", "numerical"):
        raise ValueError(curvature)
    return SinglePhaseSolver(grid=grid, cfg=cfg, fields=fields, curvature_fn=fn), z0
