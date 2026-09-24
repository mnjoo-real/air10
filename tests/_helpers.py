"""Shared test fixtures (not collected by pytest as a test module)."""
import numpy as np

from air_vortex.grid import Grid


def make_grid(Nr=8, Nz=6, dr=0.005, dz=0.005) -> Grid:
    r_f = np.linspace(0.0, Nr * dr, Nr + 1)
    z_f = np.linspace(0.0, Nz * dz, Nz + 1)
    r_c = 0.5 * (r_f[:-1] + r_f[1:])
    z_c = 0.5 * (z_f[:-1] + z_f[1:])
    return Grid(dr=dr, dz=dz, r_v=r_f[-1], z_max=z_f[-1], Nr=Nr, Nz=Nz,
                r_c=r_c, z_c=z_c, r_f=r_f, z_f=z_f)
