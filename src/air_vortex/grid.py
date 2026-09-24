"""Axisymmetric staggered (MAC-type) cylindrical grid (README sections 6, 14).

Layout for an ``(Nr, Nz)`` cell-centered grid:

- scalars (``p``, ``phi``, ``u_theta``, ``rho``, ``mu``) live at cell centers
  ``(r_c[i], z_c[j])``, shape ``(Nr, Nz)``.
- ``u_r`` lives on radial faces ``(r_f[i], z_c[j])``, shape ``(Nr+1, Nz)``.
- ``u_z`` lives on axial faces ``(r_c[i], z_f[j])``, shape ``(Nr, Nz+1)``.

The first radial face ``r_f[0] = 0`` sits exactly on the vessel axis, and the
first cell center ``r_c[0] = dr/2`` is offset by half a cell, which keeps
``1/r`` finite everywhere it is evaluated.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import Config


@dataclass
class Grid:
    dr: float
    dz: float
    r_v: float
    z_max: float
    Nr: int
    Nz: int

    r_c: np.ndarray  # (Nr,)
    z_c: np.ndarray  # (Nz,)
    r_f: np.ndarray  # (Nr+1,)
    z_f: np.ndarray  # (Nz+1,)

    @property
    def shape_center(self) -> tuple[int, int]:
        return (self.Nr, self.Nz)

    @property
    def shape_ur(self) -> tuple[int, int]:
        return (self.Nr + 1, self.Nz)

    @property
    def shape_uz(self) -> tuple[int, int]:
        return (self.Nr, self.Nz + 1)

    def r_c_2d(self) -> np.ndarray:
        """Broadcast r_c to a (Nr, Nz) grid for vectorized centered ops."""
        return np.broadcast_to(self.r_c[:, None], self.shape_center)

    def r_f_2d(self) -> np.ndarray:
        return np.broadcast_to(self.r_f[:, None], self.shape_ur)


def build_grid(cfg: Config) -> Grid:
    dr = cfg.grid.dr_m
    dz = cfg.grid.dz_m
    r_v = cfg.geometry.vessel_radius_m
    z_max = cfg.geometry.z_max_m

    Nr = max(1, round(r_v / dr))
    Nz = max(1, round(z_max / dz))

    r_f = np.linspace(0.0, Nr * dr, Nr + 1)
    z_f = np.linspace(0.0, Nz * dz, Nz + 1)

    r_c = 0.5 * (r_f[:-1] + r_f[1:])
    z_c = 0.5 * (z_f[:-1] + z_f[1:])

    return Grid(
        dr=dr, dz=dz, r_v=r_v, z_max=z_max, Nr=Nr, Nz=Nz,
        r_c=r_c, z_c=z_c, r_f=r_f, z_f=z_f,
    )
