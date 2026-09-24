"""Variable-density pressure Poisson solve (README section 14.1, step 4).

    div( (1/rho) grad p^{n+1} ) = (1/dt) div(u*)

Discretized with a 5-point stencil on the cylindrical cell-centered grid.
Boundary conditions (README section 12):

- axis (r=0) and outer wall (r=R_v): homogeneous Neumann (no radial flux)
- bottom (z=0): homogeneous Neumann (no penetration)
- top (open boundary): Dirichlet p=0 (gauge), enforced with a one-sided
  ghost value at half a cell beyond the last row.
"""
from __future__ import annotations

from typing import Literal

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .grid import Grid
from .operators import divergence, grad_p_to_ur_faces, grad_p_to_uz_faces


def face_inv_rho_ur(rho: np.ndarray) -> np.ndarray:
    """1/rho arithmetically averaged onto the full u_r face grid (Nr+1, Nz).

    Boundary faces (axis, wall) are left at 0 -- they are never used, since
    u_r is Dirichlet there. Used identically by the pressure matrix and by
    the projection correction so the two stay consistent at density jumps."""
    Nr, Nz = rho.shape
    inv_rho = 1.0 / rho
    out = np.zeros((Nr + 1, Nz))
    out[1:-1, :] = 0.5 * (inv_rho[1:, :] + inv_rho[:-1, :])
    return out


def face_inv_rho_uz(rho: np.ndarray) -> np.ndarray:
    """1/rho arithmetically averaged onto the full u_z face grid (Nr, Nz+1).

    The bottom face is left at 0 (u_z Dirichlet there). The top face uses
    the last cell's own 1/rho, matching the one-sided ghost gradient in
    :func:`grad_p_to_uz_faces`."""
    Nr, Nz = rho.shape
    inv_rho = 1.0 / rho
    out = np.zeros((Nr, Nz + 1))
    out[:, 1:-1] = 0.5 * (inv_rho[:, 1:] + inv_rho[:, :-1])
    out[:, -1] = inv_rho[:, -1]
    return out


def _pressure_matrix_coefficients(grid: Grid, rho: np.ndarray,
                                   top_bc: Literal["open", "closed"]):
    """The 5-point stencil coefficients (west, east, south, north, diag) as
    full (Nr, Nz) arrays, vectorized -- no Python-level loop over cells.
    Exactly the same arithmetic as the original per-cell loop (see git
    history), just computed for every cell at once via numpy slicing.
    Row-major flattening (``.ravel()`` on a C-contiguous (Nr, Nz) array)
    matches the ``idx(i, j) = i*Nz + j`` numbering used throughout, so the
    resulting arrays drop directly into offset-diagonal sparse construction."""
    Nr, Nz = grid.Nr, grid.Nz
    dr, dz = grid.dr, grid.dz
    r_f = grid.r_f
    r_c = grid.r_c[:, None]  # (Nr, 1), broadcasts over columns

    inv_rho_ur = face_inv_rho_ur(rho)  # faces i=0..Nr, shape (Nr+1, Nz)
    inv_rho_uz = face_inv_rho_uz(rho)  # faces j=0..Nz, shape (Nr, Nz+1)

    c_west = np.zeros((Nr, Nz))
    c_west[1:, :] = r_f[1:Nr, None] * inv_rho_ur[1:Nr, :] / (r_c[1:, :] * dr * dr)

    c_east = np.zeros((Nr, Nz))
    c_east[:-1, :] = r_f[1:Nr, None] * inv_rho_ur[1:Nr, :] / (r_c[:-1, :] * dr * dr)

    c_south = np.zeros((Nr, Nz))
    c_south[:, 1:] = inv_rho_uz[:, 1:Nz] / (dz * dz)

    c_north = np.zeros((Nr, Nz))
    c_north[:, :-1] = inv_rho_uz[:, 1:Nz] / (dz * dz)

    c_top_ghost = np.zeros((Nr, Nz))
    if top_bc == "open":
        # Dirichlet p=0 at a ghost half-cell beyond j=Nz-1; contributes to
        # the diagonal only (no off-diagonal neighbor).
        c_top_ghost[:, -1] = inv_rho_uz[:, Nz] / (0.5 * dz * dz)
    # top_bc == "closed": homogeneous Neumann, no term added

    diag = -(c_west + c_east + c_south + c_north + c_top_ghost)
    return c_west, c_east, c_south, c_north, diag


def build_pressure_matrix(grid: Grid, rho: np.ndarray,
                           top_bc: Literal["open", "closed"] = "open") -> sp.csr_matrix:
    """``top_bc="open"`` is the production boundary condition (README
    section 12): Dirichlet p=0 above the free surface. ``top_bc="closed"``
    treats the top as a rigid, impermeable lid (homogeneous Neumann, like
    the other three walls) with p pinned to 0 at cell (0,0) to remove the
    resulting null space; it exists for isolating the radial force balance
    in tests, decoupled from the open-boundary/free-surface machinery.

    Built via ``scipy.sparse.diags`` from five (Nr, Nz) coefficient arrays
    (row-major-flattened, offsets 0/±1/±Nz) instead of a per-cell Python
    loop -- see :func:`_pressure_matrix_coefficients`. This is purely a
    performance change: the coefficients and their placement in the matrix
    are identical to the original loop-based assembly (verified by the
    regression benchmark, scripts/benchmark_solver.py)."""
    Nr, Nz = grid.Nr, grid.Nz
    N = Nr * Nz

    c_west, c_east, c_south, c_north, diag = _pressure_matrix_coefficients(grid, rho, top_bc)

    # offset -1 (south, A[k,k-1]) and +1 (north, A[k,k+1]) each skip one
    # element at a row-block boundary; those positions are guaranteed 0
    # (c_south[:,0]==0, c_north[:,-1]==0), so including them costs a few
    # explicit stored zeros, not incorrect entries.
    A = sp.diags(
        diagonals=[
            c_west.ravel()[Nz:],
            c_south.ravel()[1:],
            diag.ravel(),
            c_north.ravel()[:-1],
            c_east.ravel()[:-Nz],
        ],
        offsets=[-Nz, -1, 0, 1, Nz],
        shape=(N, N),
        format="csr",
    )

    if top_bc == "closed":
        A = A.tolil()
        A[0, :] = 0.0
        A[0, 0] = 1.0
        A = A.tocsr()

    return A


def pressure_rhs(grid: Grid, u_r: np.ndarray, u_z: np.ndarray, dt: float) -> np.ndarray:
    div = divergence(grid, u_r, u_z)
    return (div / dt).reshape(-1)


def _build_spd_system(grid: Grid, rho: np.ndarray, u_star_r: np.ndarray,
                       u_star_z: np.ndarray, dt: float):
    """Symmetric positive-definite form of the same linear system solved by
    :func:`build_pressure_matrix` / :func:`pressure_rhs` (``top_bc="open"``
    only -- the only mode the production solver uses).

    Multiplying equation i by the (positive) cell weight ``r_c[i]``
    symmetrizes the matrix: the original (r_f[i]/r_c[i])-type coefficients
    become identical to their transpose partner (the algebra is exactly the
    row-scaling used in a finite-volume formulation, where each equation is
    naturally weighted by the cell's cylindrical volume). Negating both
    sides turns the negative-(semi)definite operator into an SPD one, so
    :func:`scipy.sparse.linalg.cg` applies. This changes only *how* the
    same linear system is solved, not the physical equation -- verified
    against the direct solve by the regression benchmark
    (scripts/benchmark_solver.py)."""
    c_west, c_east, c_south, c_north, diag = _pressure_matrix_coefficients(grid, rho, "open")
    Nr, Nz = grid.Nr, grid.Nz
    N = Nr * Nz
    weight = np.broadcast_to(grid.r_c[:, None], (Nr, Nz))

    def w(c):
        return -(c * weight).ravel()

    A = sp.diags(
        diagonals=[w(c_west)[Nz:], w(c_south)[1:], w(diag), w(c_north)[:-1], w(c_east)[:-Nz]],
        offsets=[-Nz, -1, 0, 1, Nz],
        shape=(N, N),
        format="csr",
    )
    b = -(weight.ravel() * pressure_rhs(grid, u_star_r, u_star_z, dt))
    return A, b


def _jacobi_preconditioner(A: sp.spmatrix) -> spla.LinearOperator:
    inv_diag = 1.0 / A.diagonal()
    return spla.LinearOperator(A.shape, matvec=lambda x: inv_diag * x)


def _amg_preconditioner(A: sp.spmatrix) -> spla.LinearOperator | None:
    """Algebraic-multigrid preconditioner via pyamg, or None if pyamg isn't
    installed. AMG is the only preconditioner tested (Jacobi, ILU, AMG;
    see README "Performance") that converges in a small, near-mesh-
    independent number of iterations for this matrix -- the 1/r cylindrical
    weighting gives it a wide dynamic range of row magnitudes that defeats
    simple diagonal/ILU preconditioning (Jacobi needed ~260 iterations at
    2880 cells; ILU with default drop tolerance failed to converge at all;
    AMG converged in ~20 iterations at both 2880 and 14400 cells)."""
    try:
        import pyamg
    except ImportError:
        return None
    ml = pyamg.smoothed_aggregation_solver(A.tocsr())
    return ml.aspreconditioner()


def solve_pressure_poisson(grid: Grid, u_star_r: np.ndarray, u_star_z: np.ndarray,
                            rho: np.ndarray, dt: float,
                            top_bc: Literal["open", "closed"] = "open",
                            method: Literal["direct", "cg"] = "direct",
                            p0: np.ndarray | None = None,
                            rtol: float = 1e-10, atol: float = 1e-10, maxiter: int = 500
                            ) -> np.ndarray:
    """Solve for p^{n+1} given a provisional velocity field (README 14.1
    step 4).

    ``method="direct"`` (default, and the only one used by the production
    solver): sparse LU via ``spsolve`` with an MMD_AT_PLUS_A column
    ordering, exact up to floating-point round-off -- what every existing
    test validates against.

    ``method="cg"``: the symmetrized system (:func:`_build_spd_system`,
    ``top_bc="open"`` only), solved with preconditioned conjugate gradient,
    warm-started from ``p0`` (typically the previous step's pressure
    field). Uses an AMG preconditioner if pyamg is installed (recommended;
    see :func:`_amg_preconditioner`), else a much weaker Jacobi fallback.
    Falls back to the direct solve -- rather than returning an unconverged
    result -- if CG has not reached tolerance within ``maxiter``
    iterations, so switching methods can never silently degrade accuracy.

    At the problem sizes exercised so far (thousands to ~15000 cells,
    README "Performance"), ``method="direct"`` measured as fast as or
    faster than ``method="cg"`` even with AMG (2D sparse LU is not
    prohibitively expensive at this scale, and AMG setup has to be
    amortized over enough solves to pay for itself). ``cg`` is provided for
    future, much finer grids (README section 15's convergence study) where
    that balance may flip -- it is not currently the recommended default."""
    if method == "direct":
        A = build_pressure_matrix(grid, rho, top_bc=top_bc)
        b = pressure_rhs(grid, u_star_r, u_star_z, dt)
        if top_bc == "closed":
            b = b.copy()
            b[0] = 0.0
        # MMD_AT_PLUS_A (minimum-degree ordering on A+A^T) measured ~25-30%
        # faster than SuperLU's default COLAMD for this matrix's structure
        # (scripts/benchmark_solver.py --profile), with no change to the
        # linear system being solved -- still an exact direct solve.
        p_flat = spla.spsolve(A.tocsc(), b, permc_spec="MMD_AT_PLUS_A")
        return p_flat.reshape(grid.Nr, grid.Nz)

    if method != "cg":
        raise ValueError(f"unknown pressure solve method: {method!r}")
    if top_bc != "open":
        raise ValueError("method='cg' only supports top_bc='open' (the production boundary condition)")

    A, b = _build_spd_system(grid, rho, u_star_r, u_star_z, dt)
    M = _amg_preconditioner(A) or _jacobi_preconditioner(A)
    x0 = p0.reshape(-1) if p0 is not None else None

    p_flat, info = spla.cg(A, b, x0=x0, rtol=rtol, atol=atol, maxiter=maxiter, M=M)
    if info != 0:
        # did not converge within maxiter -- fall back to the direct solve
        # so a slow-converging step degrades performance, never correctness
        return solve_pressure_poisson(grid, u_star_r, u_star_z, rho, dt,
                                       top_bc=top_bc, method="direct")
    return p_flat.reshape(grid.Nr, grid.Nz)


def pressure_projection(grid: Grid, u_star_r: np.ndarray, u_star_z: np.ndarray,
                         p: np.ndarray, rho: np.ndarray, dt: float
                         ) -> tuple[np.ndarray, np.ndarray]:
    """u^{n+1} = u* - dt * (1/rho) grad p^{n+1} (README 14.1 step 5).

    Uses the same face-averaged 1/rho as :func:`build_pressure_matrix` so
    the corrected velocity is exactly divergence-free w.r.t. the equation
    that was actually solved, even across a density jump."""
    inv_rho_ur = face_inv_rho_ur(rho)
    inv_rho_uz = face_inv_rho_uz(rho)

    dpdr = grad_p_to_ur_faces(grid, p)
    dpdz = grad_p_to_uz_faces(grid, p)

    u_r = u_star_r - dt * inv_rho_ur * dpdr
    u_z = u_star_z - dt * inv_rho_uz * dpdz

    # boundary faces carry no pressure-gradient correction (Dirichlet velocity BC)
    u_r[0, :] = 0.0
    u_r[-1, :] = 0.0
    u_z[:, 0] = 0.0

    return u_r, u_z
