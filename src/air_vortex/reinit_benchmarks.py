"""Manufactured static reinitialization benchmarks R1-R4 (README_rewritten
9.4 / Gate V5b). Each case supplies a deliberately non-signed-distance
phi0 whose zero set is known exactly, plus an exact-geometry error
function, so two different questions can be answered separately:

  displacement : did reinitialization move phi=0 relative to phi0's own
                 sub-cell (linear) crossings?          (reinit_diagnostics)
  geometry err : how far are the crossings from the EXACT interface?

The distinction matters: phi0's linear crossing is itself biased by
O(h^2 phi''/phi') when phi0 is nonlinear across the cell, so a scheme that
places the interface MORE accurately (quadratic crossing) shows a nonzero
'displacement' while reducing the geometry error.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .grid import Grid
from .liquid_mask import classify
from .reinit_diagnostics import interface_crossings


def uniform_grid(dx: float, R: float = 0.024, Z: float = 0.030) -> Grid:
    Nr, Nz = round(R / dx), round(Z / dx)
    r_f = np.linspace(0.0, Nr * dx, Nr + 1)
    z_f = np.linspace(0.0, Nz * dx, Nz + 1)
    return Grid(dr=dx, dz=dx, r_v=r_f[-1], z_max=z_f[-1], Nr=Nr, Nz=Nz,
                r_c=0.5 * (r_f[:-1] + r_f[1:]), z_c=0.5 * (z_f[:-1] + z_f[1:]), r_f=r_f, z_f=z_f)


@dataclass
class Case:
    name: str
    phi0: np.ndarray
    geometry_error: Callable[[np.ndarray], tuple[float, float]]
    """phi -> (max, rms) distance of phi's sub-cell crossings from the exact
    interface, measured along the grid lines (vertical and radial)."""


def _crossing_points(grid: Grid, phi: np.ndarray) -> np.ndarray:
    zc, rc = interface_crossings(grid, classify(phi))
    pts = [(grid.r_c[i], z) for i, zs in enumerate(zc) for z in zs]
    pts += [(r, grid.z_c[j]) for j, rs in enumerate(rc) for r in rs]
    return np.array(pts).reshape(-1, 2)


def _err_from_distance(grid, dist_fn):
    def f(phi):
        p = _crossing_points(grid, phi)
        d = np.abs(dist_fn(p[:, 0], p[:, 1]))
        return float(d.max()), float(np.sqrt(np.mean(d**2)))
    return f


def r1_plane(grid: Grid, z0: float = 0.0173137) -> Case:
    """Horizontal off-grid plane z = z0; phi0 = (z-z0) * g(r) * h(z-z0) with
    g, h > 0 and h nonlinear, so |grad phi0| != 1 and phi0 is not linear."""
    R, Z = np.meshgrid(grid.r_c, grid.z_c, indexing="ij")
    phi0 = (Z - z0) * (1.2 + 0.6 * np.sin(2 * np.pi * R / grid.r_v)) * (1 + 0.8 * np.tanh((Z - z0) / 0.004))
    return Case("R1_plane", phi0, _err_from_distance(grid, lambda r, z: z - z0))


def r2_tilted(grid: Grid, r0: float = 0.0121, zc: float = 0.0151, slope: float = 0.2) -> Case:
    """Tilted straight line r - r0 = slope (z - zc), away from axis/wall;
    phi0 = f(s) * g(z) with s the exact signed distance, f(0)=0, f'(0)!=1."""
    R, Z = np.meshgrid(grid.r_c, grid.z_c, indexing="ij")
    n = np.hypot(1.0, slope)
    s = ((R - r0) - slope * (Z - zc)) / n
    phi0 = s * (0.6 + 0.9 * np.tanh(s / 0.005) ** 2) * (1.0 + 0.3 * np.sin(2 * np.pi * Z / grid.z_max))
    return Case("R2_tilted", phi0, _err_from_distance(grid, lambda r, z: ((r - r0) - slope * (z - zc)) / n))


def r3_circle(grid: Grid, r0: float = 0.0121, z0: float = 0.0151, rad: float = 0.00623) -> Case:
    """Circle (torus in 3D) in the r-z plane, liquid inside:
    phi0 = f(d) * (1 + 0.3 sin(angle)), d = exact signed distance."""
    R, Z = np.meshgrid(grid.r_c, grid.z_c, indexing="ij")
    d = np.hypot(R - r0, Z - z0) - rad
    a = 0.002
    phi0 = d * (0.4 + 1.2 * d**2 / (d**2 + a**2)) * (1.0 + 0.3 * np.sin(np.arctan2(Z - z0, R - r0)))
    return Case("R3_circle", phi0, _err_from_distance(grid, lambda r, z: np.hypot(r - r0, z - z0) - rad))


def r4_paraboloid(grid: Grid, omega: float = 20.0, H: float = 0.018, g: float = 9.81) -> Case:
    """Rigid-body free surface eta = C + Omega^2 r^2/(2g) (V3 geometry),
    phi0 = signed distance (as in the operational V3 gate)."""
    from .diagnostics import volume_consistent_parabola_constant
    from .single_phase_solver import signed_distance_to_profile
    C = volume_consistent_parabola_constant(grid.r_v, omega, g, np.pi * grid.r_v**2 * H)
    eta = lambda r: C + omega**2 * r**2 / (2 * g)  # noqa: E731
    phi0 = signed_distance_to_profile(grid, eta)

    def err(phi):
        # vertical distance to the surface is what eta NRMSE measures; the
        # normal distance is smaller by 1/sqrt(1+eta'^2)
        p = _crossing_points(grid, phi)
        slope = omega**2 * p[:, 0] / g
        d = np.abs(p[:, 1] - eta(p[:, 0])) / np.sqrt(1 + slope**2)
        return float(d.max()), float(np.sqrt(np.mean(d**2)))
    return Case("R4_paraboloid", phi0, err)


CASES = {"R1": r1_plane, "R2": r2_tilted, "R3": r3_circle, "R4": r4_paraboloid}
