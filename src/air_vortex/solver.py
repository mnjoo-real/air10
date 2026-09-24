"""Main projection-method time integrator (README section 14.1 / 30).

This module holds the LEGACY Level-1B full two-phase diffuse Level-Set
solver (``physics.free_surface_model: two_phase_diffuse_ls``). It is kept
runnable and bit-for-bit unchanged for historical diagnostics and the
regression golden fixture. The Level-1A single-phase production path lives
in :mod:`air_vortex.single_phase_solver`; :func:`build_solver` is the one
place where the two are selected (README_rewritten section 19.2).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal

import numpy as np

from .boundary import apply_velocity_bc, enforce_wall_no_slip
from .config import Config
from .fields import Fields, initialize_still_water
from .forcing import forcing_mask, stirrer_forcing
from .grid import Grid, build_grid
from .levelset import advect_level_set_configurable, reinitialize_level_set
from .operators import (
    interp_center_to_ur,
    interp_center_to_uz,
    interp_ur_to_center,
    interp_uz_to_center,
    laplacian_ur,
    laplacian_utheta,
    laplacian_uz,
    ur_at_uz_locations,
    u_theta_on_ur_faces,
    upwind_derivative,
    uz_at_ur_locations,
)
from .pressure import pressure_projection, solve_pressure_poisson
from .properties import material_properties
from .surface_tension import surface_tension_force
from .timestep import compute_stable_timestep
from .volume_correction import apply_volume_correction

SwirlMode = Literal["forced", "prescribed"]
"""``"forced"``: the production local bottom stirrer relaxation forcing
(forcing.py). ``"prescribed"``: VALIDATION-ONLY -- overwrites u_theta with
Omega*r over the WHOLE domain (water and air) every step. Session-7
diagnostics showed this construction is not a clean equilibrium (rotating
air vs. flat top p=0; non-integrable masked forcing), so it must never be
used for production runs; :func:`build_solver` warns when it is selected."""


class ValidationOnlyWarning(UserWarning):
    """A validation-only construction was selected."""


@dataclass
class StepDiagnostics:
    dt: float
    t: float
    max_u_theta: float
    max_downward_uz: float
    min_pressure: float
    volume_water: float
    volume_correction_applied: bool = False
    volume_before_correction: float | None = None
    volume_after_correction: float | None = None
    volume_correction_delta: float | None = None
    cumulative_volume_correction: float = 0.0


@dataclass
class Solver:
    grid: Grid
    cfg: Config
    fields: Fields
    swirl_mode: SwirlMode = "forced"
    pressure_method: Literal["direct", "cg"] = "direct"
    """"direct" (default) is the recommended, validated method at the grid
    sizes used so far (README "Performance"). "cg" is available for future,
    much finer grids where an iterative solve may start to win; see
    pressure.solve_pressure_poisson."""
    _initial_volume: float | None = field(default=None, repr=False)
    _cumulative_volume_correction: float = field(default=0.0, repr=False)

    @property
    def cumulative_volume_correction(self) -> float:
        return self._cumulative_volume_correction

    def step(self, dt: float | None = None) -> StepDiagnostics:
        f = self.fields
        cfg = self.cfg
        grid = self.grid

        # Step 1: material properties (README 14.1 step 1)
        rho, mu = material_properties(f.phi, grid, cfg)

        # Step 2: interface geometry / surface tension (README 14.1 step 2)
        f_sigma_r, f_sigma_z = surface_tension_force(grid, f.phi, cfg)

        if dt is None:
            dt = compute_stable_timestep(grid, f.u_r, f.u_z, rho.min(), mu.max(), cfg)

        omega = cfg.stirrer.omega(f.t)
        chi = forcing_mask(grid, cfg)

        rho_ur = interp_center_to_ur(rho)
        rho_uz = interp_center_to_uz(rho)
        mu_ur = interp_center_to_ur(mu)
        mu_uz = interp_center_to_uz(mu)

        # --- Step 3: predictor velocity u* (README 14.1 step 3) ---

        # radial momentum
        w_at_ur = uz_at_ur_locations(f.u_z)
        adv_r = f.u_r * upwind_derivative(f.u_r, f.u_r, grid.dr, axis=0) \
            + w_at_ur * upwind_derivative(f.u_r, w_at_ur, grid.dz, axis=1)
        u_theta_ur = u_theta_on_ur_faces(grid, f.u_theta)
        # r_f[0] = 0 exactly (the axis); u_r is forced to 0 there regardless
        # (line below), so a safe placeholder avoids a 0-divide warning.
        r_f_safe = np.where(grid.r_f == 0.0, grid.dr, grid.r_f)
        centrifugal = u_theta_ur**2 / r_f_safe[:, None]
        visc_r = mu_ur * laplacian_ur(grid, f.u_r)

        u_r_star = f.u_r + dt * (
            -adv_r + centrifugal + visc_r / rho_ur + f_sigma_r / rho_ur
        )

        # axial momentum
        u_at_uz = ur_at_uz_locations(f.u_r)
        adv_z = u_at_uz * upwind_derivative(f.u_z, u_at_uz, grid.dr, axis=0) \
            + f.u_z * upwind_derivative(f.u_z, f.u_z, grid.dz, axis=1)
        visc_z = mu_uz * laplacian_uz(grid, f.u_z)

        u_z_star = f.u_z + dt * (
            -adv_z - cfg.fluid.gravity + visc_z / rho_uz + f_sigma_z / rho_uz
        )

        # azimuthal momentum
        if self.swirl_mode == "prescribed":
            u_theta_new = omega * grid.r_c[:, None] * np.ones_like(f.u_theta)
        else:
            u_r_c = interp_ur_to_center(f.u_r)
            u_z_c = interp_uz_to_center(f.u_z)
            adv_theta = u_r_c * upwind_derivative(f.u_theta, u_r_c, grid.dr, axis=0) \
                + u_z_c * upwind_derivative(f.u_theta, u_z_c, grid.dz, axis=1)
            extra = u_r_c * f.u_theta / grid.r_c[:, None]
            visc_theta = mu * laplacian_utheta(grid, f.u_theta)
            f_stir = stirrer_forcing(f.u_theta, omega, grid, cfg, chi)

            u_theta_new = f.u_theta + dt * (
                -adv_theta - extra + visc_theta / rho + f_stir
            )
            enforce_wall_no_slip(u_theta_new)

        # The Poisson RHS must see the boundary-enforced predictor velocity,
        # otherwise the solve has no way to know that no-penetration at the
        # bottom/wall/axis blocks the predictor's tendency to keep falling
        # under gravity -- without this the projection cannot recover
        # hydrostatic equilibrium (README section 12, Test 1).
        u_r_star[0, :] = 0.0
        u_r_star[-1, :] = 0.0
        u_z_star[:, 0] = 0.0
        u_z_star[:, -1] = u_z_star[:, -2]

        # --- Step 4/5: pressure Poisson + projection (README 14.1 steps 4-5) ---
        p_new = solve_pressure_poisson(grid, u_r_star, u_z_star, rho, dt,
                                        method=self.pressure_method, p0=f.p)
        u_r_new, u_z_new = pressure_projection(grid, u_r_star, u_z_star, p_new, rho, dt)

        f.u_r, f.u_z, f.u_theta, f.p = u_r_new, u_z_new, u_theta_new, p_new
        apply_velocity_bc(grid, f)

        # --- Step 6/7: Level Set advect + periodic reinit (README 14.1 steps 6-7) ---
        f.phi = advect_level_set_configurable(
            f.phi, f.u_r, f.u_z, grid, dt,
            scheme=cfg.levelset.advection_scheme,
            time_integrator=cfg.levelset.time_integrator,
            limiter=cfg.levelset.limiter,
        )
        if f.step % cfg.levelset.reinitialize_every == 0:
            f.phi = reinitialize_level_set(f.phi, grid, cfg.levelset.reinitialize_iterations)

        f.rho, f.mu = rho, mu
        f.t += dt
        f.step += 1

        # --- optional global volume correction (README "optional global
        # volume correction"), default OFF -- a uniform phi shift only,
        # applied every `every_n_steps` steps against the volume at t=0 ---
        vc_cfg = cfg.levelset.volume_correction
        vc_applied = False
        vc_before = vc_after = vc_delta = None
        if vc_cfg.enabled:
            if self._initial_volume is None:
                self._initial_volume = water_volume(grid, f.phi)
            if f.step % max(1, vc_cfg.every_n_steps) == 0:
                f.phi, vc_result = apply_volume_correction(grid, f.phi, self._initial_volume)
                vc_applied = True
                vc_before, vc_after, vc_delta = (
                    vc_result.volume_before, vc_result.volume_after, vc_result.delta)
                self._cumulative_volume_correction += vc_delta
                if vc_result.large_correction_warning:
                    print(f"WARNING: volume correction at step {f.step} required a shift of "
                          f"{vc_delta:.3e} m (> one grid cell) -- the underlying advection "
                          f"error is no longer 'small drift'.")

        # --- Step 8: diagnostics (README 14.1 step 8) ---
        volume = water_volume(grid, f.phi)
        diag = StepDiagnostics(
            dt=dt,
            t=f.t,
            max_u_theta=float(np.max(np.abs(f.u_theta))),
            max_downward_uz=float(-np.min(f.u_z)) if f.u_z.size else 0.0,
            min_pressure=float(np.min(f.p)),
            volume_water=volume,
            volume_correction_applied=vc_applied,
            volume_before_correction=vc_before,
            volume_after_correction=vc_after,
            volume_correction_delta=vc_delta,
            cumulative_volume_correction=self._cumulative_volume_correction,
        )
        return diag

    def run(self, t_end: float | None = None,
             on_step: Callable[[StepDiagnostics], None] | None = None) -> None:
        t_end = self.cfg.time.t_end_s if t_end is None else t_end
        while self.fields.t < t_end:
            diag = self.step()
            if on_step is not None:
                on_step(diag)


def water_volume(grid: Grid, phi: np.ndarray) -> float:
    """Axisymmetric water volume: integral of [phi<0] * 2*pi*r dr dz."""
    water = (phi < 0).astype(float)
    cell_volume = 2.0 * np.pi * grid.r_c[:, None] * grid.dr * grid.dz
    return float(np.sum(water * cell_volume))


def build_solver(cfg: Config, swirl_mode: SwirlMode = "forced",
                  pressure_method: Literal["direct", "cg"] = "direct"):
    """Architecture switch (README_rewritten section 19.2): returns the
    legacy two-phase :class:`Solver` for ``two_phase_diffuse_ls`` (the
    transition default) or a
    :class:`~air_vortex.single_phase_solver.SinglePhaseSolver` for
    ``single_phase_ls``."""
    model = cfg.physics.free_surface_model
    if model == "single_phase_ls":
        if swirl_mode != "forced":
            raise ValueError(
                "single_phase_ls has no domain-wide prescribed swirl (README_rewritten 8.2); "
                "swirl_mode='prescribed' is a legacy validation-only construction.")
        if pressure_method != "direct":
            raise ValueError("single_phase_ls currently supports pressure_method='direct' only.")
        from .single_phase_solver import build_single_phase_solver
        return build_single_phase_solver(cfg)
    if model != "two_phase_diffuse_ls":
        raise ValueError(f"unknown physics.free_surface_model: {model!r}")
    ls = cfg.levelset
    if ls.reinitialization_method != "legacy_godunov" or ls.reinitialization_trigger != "periodic":
        raise ValueError(
            "two_phase_diffuse_ls runs the legacy Godunov reinitialization only (kept bit-for-bit "
            "unchanged); russo_smereka_subcell / quality trigger are single_phase_ls options.")
    if swirl_mode == "prescribed":
        import warnings
        warnings.warn(
            "swirl_mode='prescribed' is VALIDATION-ONLY (domain-wide Omega*r, not an equilibrium "
            "construction; README_rewritten 21.3). Production uses swirl_mode='forced'.",
            ValidationOnlyWarning, stacklevel=2)
    grid = build_grid(cfg)
    fields = initialize_still_water(grid, cfg.geometry.water_height_m)
    return Solver(grid=grid, cfg=cfg, fields=fields, swirl_mode=swirl_mode,
                  pressure_method=pressure_method)
