"""Level Set advection and reinitialization (README sections 9, 9.3).

Two interchangeable advection schemes, selected via
``config.levelset.advection_scheme`` / ``time_integrator`` / ``limiter``
(README "Level Set advection accuracy"):

- ``upwind1`` + ``euler`` (the original scheme, kept byte-for-byte as
  :func:`advect_level_set` -- this is what every existing regression
  fixture was generated with, and remains the default so nothing that
  worked before silently changes behavior).
- ``muscl2`` + ``ssprk2``: a conservative, flux-form MUSCL-TVD
  reconstruction (MC or minmod limiter) with SSP-RK2 time integration,
  intended to reduce the numerical diffusion of the first-order scheme
  (README "Performance" validation notes: ~6.5% water-volume drift over a
  0.4s solid-body-rotation run was traced to this).

Both schemes are exposed through :func:`advect_level_set_configurable`,
which dispatches by name; :func:`air_vortex.solver.Solver.step` reads its
scheme/integrator/limiter choice from ``cfg.levelset``.
"""
from __future__ import annotations

import numpy as np

from .grid import Grid
from .operators import _pad_axis, interp_ur_to_center, interp_uz_to_center, upwind_derivative


def advect_level_set(phi: np.ndarray, u_r: np.ndarray, u_z: np.ndarray,
                      grid: Grid, dt: float) -> np.ndarray:
    """phi^{n+1} = phi^n - dt*(u_r dphi/dr + u_z dphi/dz) (README section 9).

    First-order upwind, explicit Euler. This is the original scheme; kept
    unchanged (not just "equivalent") so the existing regression golden
    fixture (tests/fixtures/regression_golden.npz) and every test written
    against it stay valid without regeneration."""
    u_r_c = interp_ur_to_center(u_r)
    u_z_c = interp_uz_to_center(u_z)

    dphidr = upwind_derivative(phi, u_r_c, grid.dr, axis=0)
    dphidz = upwind_derivative(phi, u_z_c, grid.dz, axis=1)

    return phi - dt * (u_r_c * dphidr + u_z_c * dphidz)


def levelset_rhs_upwind1(phi: np.ndarray, u_r: np.ndarray, u_z: np.ndarray, grid: Grid) -> np.ndarray:
    """L(phi) = -(u_r dphi/dr + u_z dphi/dz), the same spatial operator as
    :func:`advect_level_set` but exposed as a standalone RHS so it can be
    driven by any time integrator (used for the upwind1+ssprk2 combination,
    and as the euler-equivalent RHS for upwind1+euler)."""
    u_r_c = interp_ur_to_center(u_r)
    u_z_c = interp_uz_to_center(u_z)
    dphidr = upwind_derivative(phi, u_r_c, grid.dr, axis=0)
    dphidz = upwind_derivative(phi, u_z_c, grid.dz, axis=1)
    return -(u_r_c * dphidr + u_z_c * dphidz)


# ---------------------------------------------------------------------------
# MUSCL-TVD reconstruction (2nd order, limited) and SSP-RK2 time integration.
# ---------------------------------------------------------------------------


def minmod_limiter(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Standard 2-argument minmod, applied to undivided differences a, b
    (e.g. phi[i]-phi[i-1] and phi[i+1]-phi[i]): 0 if they disagree in sign
    (a local extremum -- don't extrapolate through it), else the smaller
    in magnitude, with their common sign."""
    same_sign = (a * b) > 0
    return np.where(same_sign, np.sign(a) * np.minimum(np.abs(a), np.abs(b)), 0.0)


def mc_limiter(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Monotonized-central limiter: minmod3(2a, (a+b)/2, 2b) (van Leer /
    LeVeque). Less compressive than minmod (steeper reconstructed slopes
    where the data is smooth), while remaining TVD."""
    s = 0.5 * (a + b)
    same_sign = (a * b) > 0
    lim = np.minimum(np.minimum(np.abs(2 * a), np.abs(2 * b)), np.abs(s))
    return np.where(same_sign, np.sign(s) * lim, 0.0)


_LIMITERS = {"mc": mc_limiter, "minmod": minmod_limiter}


def _limited_slope(phi: np.ndarray, axis: int, limiter_fn) -> np.ndarray:
    """Limited slope at every cell along ``axis`` (undivided-difference
    units: this is d(phi)/d(index), not d(phi)/dx -- reconstruction below
    uses it as a half-cell-width extrapolation directly, which is
    equivalent). Ghost cells at both ends use zero-gradient ('even')
    padding: correct for a genuine physical boundary (outer wall, bottom,
    open top), and also correct at the axis, where an axisymmetric scalar
    field is implicitly even in r (same convention already established for
    operators.center_grad_r) -- the resulting zero-slope-at-axis matches
    the true symmetry condition dphi/dr=0 there."""
    phi_ext = _pad_axis(phi, axis=axis, mode_low="even", mode_high="even")
    if axis == 0:
        a = phi_ext[1:-1, :] - phi_ext[:-2, :]
        b = phi_ext[2:, :] - phi_ext[1:-1, :]
    else:
        a = phi_ext[:, 1:-1] - phi_ext[:, :-2]
        b = phi_ext[:, 2:] - phi_ext[:, 1:-1]
    return limiter_fn(a, b)


def muscl_flux_r(phi: np.ndarray, u_r: np.ndarray, grid: Grid, limiter_fn) -> np.ndarray:
    """Conservative radial flux F_r = u_r * phi_face on the u_r grid
    (Nr+1, Nz), phi_face MUSCL-reconstructed and upwind-selected by the
    sign of u_r at each face. Boundary faces (axis, wall) are exactly 0:
    u_r is Dirichlet-zero there (README section 12), so no reconstruction
    is needed or performed."""
    Nr, Nz = phi.shape
    slope = _limited_slope(phi, axis=0, limiter_fn=limiter_fn)
    phi_right_extrap = phi + 0.5 * slope  # value extrapolated to this cell's +r face
    phi_left_extrap = phi - 0.5 * slope   # value extrapolated to this cell's -r face

    F = np.zeros((Nr + 1, Nz))
    u_int = u_r[1:Nr, :]
    phi_face = np.where(u_int >= 0, phi_right_extrap[:-1, :], phi_left_extrap[1:, :])
    F[1:Nr, :] = u_int * phi_face
    return F


def muscl_flux_z(phi: np.ndarray, u_z: np.ndarray, grid: Grid, limiter_fn) -> np.ndarray:
    """Conservative axial flux F_z = u_z * phi_face on the u_z grid
    (Nr, Nz+1). Bottom face is exactly 0 (u_z Dirichlet-zero). The open top
    face uses a one-sided (zero-gradient / extrapolated-outflow) value from
    the last cell's own MUSCL reconstruction, matching how the pressure
    solve's open-boundary gradient already treats that face."""
    Nr, Nz = phi.shape
    slope = _limited_slope(phi, axis=1, limiter_fn=limiter_fn)
    phi_right_extrap = phi + 0.5 * slope  # toward +z face
    phi_left_extrap = phi - 0.5 * slope   # toward -z face

    F = np.zeros((Nr, Nz + 1))
    u_int = u_z[:, 1:Nz]
    phi_face = np.where(u_int >= 0, phi_right_extrap[:, :-1], phi_left_extrap[:, 1:])
    F[:, 1:Nz] = u_int * phi_face
    F[:, Nz] = u_z[:, Nz] * phi_right_extrap[:, -1]
    return F


def levelset_rhs_muscl2(phi: np.ndarray, u_r: np.ndarray, u_z: np.ndarray, grid: Grid,
                         limiter: str = "mc") -> np.ndarray:
    """L(phi) = -div(u * phi) in cylindrical flux form (same r-weighted
    flux-difference structure as operators.divergence), using MUSCL-TVD
    reconstructed face values. Equivalent to the non-conservative transport
    equation up to the divergence of u (README section 9), which is at
    machine precision after the pressure projection (README "Performance"
    validation notes) -- so this is effectively exact conservative
    advection of phi, not an approximation of a different equation."""
    limiter_fn = _LIMITERS[limiter]
    F_r = muscl_flux_r(phi, u_r, grid, limiter_fn)
    F_z = muscl_flux_z(phi, u_z, grid, limiter_fn)

    r_f = grid.r_f
    flux_r_weighted = r_f[:, None] * F_r
    d_flux_r = (flux_r_weighted[1:, :] - flux_r_weighted[:-1, :]) / grid.dr
    div_flux = d_flux_r / grid.r_c[:, None] + (F_z[:, 1:] - F_z[:, :-1]) / grid.dz
    return -div_flux


def _ssprk2_step(phi: np.ndarray, rhs_fn, dt: float) -> np.ndarray:
    """Shu-Osher SSP-RK2: phi1 = phi + dt*L(phi); phi^{n+1} = phi/2 +
    (phi1 + dt*L(phi1))/2. Strong-stability-preserving (TVD) for a TVD
    spatial operator L, which MUSCL-TVD is by construction."""
    L0 = rhs_fn(phi)
    phi1 = phi + dt * L0
    L1 = rhs_fn(phi1)
    return 0.5 * phi + 0.5 * (phi1 + dt * L1)


def advect_level_set_configurable(phi: np.ndarray, u_r: np.ndarray, u_z: np.ndarray,
                                   grid: Grid, dt: float,
                                   scheme: str = "upwind1",
                                   time_integrator: str = "euler",
                                   limiter: str = "mc") -> np.ndarray:
    """Dispatches to the requested (scheme, time_integrator) combination.
    velocity is held fixed across any RK sub-stages (README "Level Set
    advection accuracy": only the phi transport sub-step uses SSP-RK2, not
    the full Navier-Stokes system) -- u_r/u_z are exactly the already-
    projected, divergence-free field passed in by the caller.

    ``scheme="upwind1", time_integrator="euler"`` reproduces
    :func:`advect_level_set` exactly (verified by
    tests/test_levelset_muscl.py), so switching the config back to that
    combination is a true no-op relative to the original behavior."""
    if scheme == "upwind1":
        rhs_fn = lambda p: levelset_rhs_upwind1(p, u_r, u_z, grid)  # noqa: E731
    elif scheme == "muscl2":
        rhs_fn = lambda p: levelset_rhs_muscl2(p, u_r, u_z, grid, limiter=limiter)  # noqa: E731
    else:
        raise ValueError(f"unknown advection_scheme: {scheme!r}")

    if time_integrator == "euler":
        return phi + dt * rhs_fn(phi)
    elif time_integrator == "ssprk2":
        return _ssprk2_step(phi, rhs_fn, dt)
    else:
        raise ValueError(f"unknown time_integrator: {time_integrator!r}")


def _grad_magnitude(phi: np.ndarray, grid: Grid) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Godunov upwind |grad phi| for the reinitialization equation, plus the
    forward/backward one-sided derivatives used to build it."""
    dr, dz = grid.dr, grid.dz

    dr_p = (np.roll(phi, -1, axis=0) - phi) / dr
    dr_m = (phi - np.roll(phi, 1, axis=0)) / dr
    dr_p[-1, :] = 0.0
    dr_m[0, :] = 0.0

    dz_p = (np.roll(phi, -1, axis=1) - phi) / dz
    dz_m = (phi - np.roll(phi, 1, axis=1)) / dz
    dz_p[:, -1] = 0.0
    dz_m[:, 0] = 0.0

    return dr_p, dr_m, dz_p, dz_m  # type: ignore[return-value]


def reinitialize_level_set(phi0: np.ndarray, grid: Grid, n_iter: int,
                            d_tau: float | None = None) -> np.ndarray:
    """Solve d(phi)/d(tau) = S(phi0) * (1 - |grad phi|) for a few pseudo-time
    steps (README section 9.3), using a Godunov upwind scheme so the sign
    function stays stable near the interface."""
    dx = min(grid.dr, grid.dz)
    if d_tau is None:
        d_tau = 0.5 * dx

    eps = dx
    S = phi0 / np.sqrt(phi0**2 + eps**2)

    phi = phi0.copy()
    for _ in range(n_iter):
        dr_p, dr_m, dz_p, dz_m = _grad_magnitude(phi, grid)

        pos = S > 0
        neg = ~pos

        grad_sq = np.empty_like(phi)

        a = np.maximum(dr_m, 0.0) ** 2
        b = np.minimum(dr_p, 0.0) ** 2
        c = np.maximum(dz_m, 0.0) ** 2
        d = np.minimum(dz_p, 0.0) ** 2
        grad_sq[pos] = np.maximum(a, b)[pos] + np.maximum(c, d)[pos]

        a2 = np.minimum(dr_m, 0.0) ** 2
        b2 = np.maximum(dr_p, 0.0) ** 2
        c2 = np.minimum(dz_m, 0.0) ** 2
        d2 = np.maximum(dz_p, 0.0) ** 2
        grad_sq[neg] = np.maximum(a2, b2)[neg] + np.maximum(c2, d2)[neg]

        grad_mag = np.sqrt(grad_sq)
        phi = phi + d_tau * S * (1.0 - grad_mag)

    return phi


# ---------------------------------------------------------------------------
# Advective-form transport for the single-phase Level-1A path. New code
# only -- nothing above is changed, so the legacy two-phase path keeps its
# exact behavior.
# ---------------------------------------------------------------------------


def levelset_rhs_muscl2_advective(phi: np.ndarray, u_r: np.ndarray, u_z: np.ndarray, grid: Grid,
                                   limiter: str = "mc") -> np.ndarray:
    """-u . grad(phi) = -div(u phi) + phi div(u), MUSCL-reconstructed.

    The flux form alone (:func:`levelset_rhs_muscl2`) equals the transport
    equation only when div(u)=0. In the single-phase path the velocity
    used for transport is the *extended* liquid velocity, which is
    divergence-free in the liquid but not in the void extension band, so
    the phi*div(u) correction is required there. It vanishes on the
    interface itself (phi=0) and in the projected liquid (div u ~ 0)."""
    from .operators import divergence
    return levelset_rhs_muscl2(phi, u_r, u_z, grid, limiter=limiter) + phi * divergence(grid, u_r, u_z)


def advect_level_set_advective(phi: np.ndarray, u_r: np.ndarray, u_z: np.ndarray,
                                grid: Grid, dt: float, scheme: str = "muscl2",
                                time_integrator: str = "ssprk2", limiter: str = "mc") -> np.ndarray:
    """Transport with a (possibly non-solenoidal) extended velocity.
    ``upwind1`` is already in advective form and is dispatched unchanged."""
    if scheme == "upwind1":
        rhs_fn = lambda p: levelset_rhs_upwind1(p, u_r, u_z, grid)  # noqa: E731
    elif scheme == "muscl2":
        rhs_fn = lambda p: levelset_rhs_muscl2_advective(p, u_r, u_z, grid, limiter=limiter)  # noqa: E731
    else:
        raise ValueError(f"unknown advection_scheme: {scheme!r}")
    if time_integrator == "euler":
        return phi + dt * rhs_fn(phi)
    if time_integrator == "ssprk2":
        return _ssprk2_step(phi, rhs_fn, dt)
    raise ValueError(f"unknown time_integrator: {time_integrator!r}")
