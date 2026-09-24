"""Boundary conditions (README section 12).

Axis r=0:      u_r=0, u_theta=0, du_z/dr=0, dp/dr=0
Wall r=R_v:    no slip, u_r=u_theta=u_z=0
Bottom z=0:    no slip, u_r=u_theta=u_z=0 (stirrer enters via volumetric forcing)
Top:           open, p=0 (gauge), zero-gradient velocity
"""
from __future__ import annotations

import numpy as np

from .fields import Fields
from .grid import Grid


def apply_velocity_bc(grid: Grid, fields: Fields) -> None:
    """Re-enforces the *hard* (Dirichlet no-penetration/no-slip) velocity
    constraints after the pressure projection. The top boundary is
    deliberately NOT touched here: it is open, and pressure.pressure_projection
    already computes a divergence-consistent u_z there from the open-boundary
    pressure gradient. Overwriting it with a zero-gradient copy (as this
    function used to do) discarded that correct value and reintroduced an
    O(1) divergence residual concentrated entirely in the top row -- caught
    by scripts/run_grid_convergence.py showing max|div| growing with grid
    refinement instead of staying near machine precision. See README
    "Performance"/validation notes.
    """
    # Axis (r=0): no radial flow through the axis.
    fields.u_r[0, :] = 0.0
    # Outer wall (r=R_v): no slip.
    fields.u_r[-1, :] = 0.0

    # Bottom (z=0): no slip / no penetration.
    fields.u_z[:, 0] = 0.0

    # u_theta at cell centers: axis value pinned to 0, wall/bottom no-slip
    # are enforced approximately via a ghost-less one-sided decay so the
    # centered stencils near boundaries stay well behaved.
    fields.u_theta[0, :] = -fields.u_theta[1, :] * 0.0  # axis: u_theta(0)=0 handled by grid offset (r_c[0]=dr/2 > 0)


def apply_pressure_bc(grid: Grid, p: np.ndarray) -> np.ndarray:
    """Return a copy of p with the top row forced to gauge pressure 0."""
    p = p.copy()
    p[:, -1] = 0.0
    return p


def enforce_wall_no_slip(u_theta: np.ndarray) -> None:
    """u_theta = 0 at the outer wall and bottom (README section 12)."""
    u_theta[-1, :] = 0.0
    u_theta[:, 0] = 0.0
