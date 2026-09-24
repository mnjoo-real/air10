"""Level-1A time integrator: axisymmetric single-phase water with a moving
free surface (README_rewritten sections 5, 6, 19.4).

Per step:

    phi^n
      -> liquid / interface geometry          (liquid_mask.classify)
      -> extend u^n into the void band         (velocity_extension)
      -> liquid predictor u*                   (constant rho_w, mu_w; no CSF)
      -> interface p_Gamma = p_atm             (free_surface_bc; sigma=0 only)
      -> liquid-only pressure solve            (pressure_single_phase)
      -> projection on liquid-adjacent faces
      -> extend u^{n+1} into the void band
      -> Level Set advection, advective MUSCL2 (levelset.advect_level_set_advective)
      -> optional reinitialization (displacement recorded)
      -> optional global volume correction (OFF by default)

What this path does NOT use (README_rewritten section 6.2 / task item 4):
rho(phi), mu(phi), an air momentum equation, a variable-density Poisson
equation, CSF surface tension, or domain-wide prescribed swirl. None of
properties.py, surface_tension.py or pressure.py is imported here.

Surface tension (Gate V4) enters ONLY as the sharp interface pressure
p_Gamma = p_atm + S_KAPPA sigma kappa at each sub-cell crossing, with kappa
from curvature_single_phase.py evaluated on phi^n.

Stirrer forcing is intentionally NOT connected yet (README_rewritten
Milestone E: only after the numerical gates pass). A config with a nonzero
stirrer RPM is refused rather than silently run without forcing.

Wall boundary conditions
------------------------
Production (default) is :meth:`WallBC.no_slip`: u_r = u_z = u_theta = 0 on
the side wall and bottom. :meth:`WallBC.rotating` exists ONLY for the
manufactured rigid-body verification (README_rewritten Gate V3): exact
solid-body rotation u_theta = Omega r is not an equilibrium against a
stationary no-slip wall, so that test uses a container rotating with the
liquid (u_theta = Omega r on wall and bottom; u_r = u_z = 0 unchanged).
It is a constructor argument, not a YAML key, so a production config
cannot select it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal

import numpy as np

from .config import Config
from .diagnostics import free_surface_height
from .fields import Fields, initialize_still_water
from .curvature_single_phase import interface_curvature
from .free_surface_bc import interface_pressure
from .grid import Grid, build_grid
from .levelset import advect_level_set_advective
from .reinit_diagnostics import contour_displacement, reinitialize, signed_distance_error
from .liquid_mask import LiquidGeometry, classify, liquid_volume_staircase, liquid_volume_subcell
from .operators import (
    _extended_coord,
    interp_ur_to_center,
    interp_uz_to_center,
    laplacian_ur,
    laplacian_uz,
    u_theta_on_ur_faces,
    upwind_derivative,
    ur_at_uz_locations,
    uz_at_ur_locations,
)
from .pressure_single_phase import liquid_divergence, project_liquid_velocity, solve_liquid_pressure
from .timestep import compute_stable_timestep
from .velocity_extension import extend_velocity, minimum_capillary_extension_layers
from .volume_correction import apply_volume_correction

_GRAVITY_WAVE_CFL = 0.5


@dataclass(frozen=True)
class WallBC:
    """Tangential (swirl) velocity of the side wall and bottom."""
    kind: Literal["no_slip", "rotating"] = "no_slip"
    omega: float = 0.0

    @classmethod
    def no_slip(cls) -> "WallBC":
        """PRODUCTION beaker: stationary wall and bottom."""
        return cls("no_slip", 0.0)

    @classmethod
    def rotating(cls, omega: float) -> "WallBC":
        """VALIDATION ONLY (Gate V3): container co-rotating at ``omega``."""
        return cls("rotating", float(omega))

    def wall_u_theta(self, grid: Grid) -> np.ndarray:
        return np.full(grid.Nz, self.omega * grid.r_f[-1])

    def bottom_u_theta(self, grid: Grid) -> np.ndarray:
        return self.omega * grid.r_c


def laplacian_utheta_dirichlet(grid: Grid, u_theta: np.ndarray, u_wall: np.ndarray,
                                u_bottom: np.ndarray) -> np.ndarray:
    """Vector Laplacian of u_theta (same stencil as operators.laplacian_utheta)
    with Dirichlet wall value ``u_wall`` (Nz,) at r=R_v and bottom value
    ``u_bottom`` (Nr,) at z=0, imposed by ghost = 2*u_b - u_interior. With
    both zero this is bit-for-bit the legacy no-slip operator. Axis: odd
    (u_theta=0 by symmetry). Top: even."""
    dr, dz = grid.dr, grid.dz
    f_ext = np.concatenate([-u_theta[:1, :], u_theta, 2.0 * u_wall[None, :] - u_theta[-1:, :]], axis=0)
    r_ext = _extended_coord(grid.r_c)
    r_mid = 0.5 * (r_ext[:-1] + r_ext[1:])
    flux_plus = r_mid[1:, None] * (f_ext[2:, :] - f_ext[1:-1, :])
    flux_minus = r_mid[:-1, None] * (f_ext[1:-1, :] - f_ext[:-2, :])
    r_part = (flux_plus - flux_minus) / (grid.r_c[:, None] * dr**2)

    g_ext = np.concatenate([2.0 * u_bottom[:, None] - u_theta[:, :1], u_theta, u_theta[:, -1:]], axis=1)
    z_part = (g_ext[:, 2:] - 2.0 * g_ext[:, 1:-1] + g_ext[:, :-2]) / dz**2
    return r_part - u_theta / grid.r_c[:, None] ** 2 + z_part


@dataclass
class SinglePhaseStepDiagnostics:
    dt: float
    t: float
    max_u_theta: float
    max_downward_uz: float
    min_pressure: float
    volume_water: float              # staircase count (same definition as legacy water_volume)
    volume_water_subcell: float
    max_divergence_liquid: float
    max_abs_ur_liquid: float
    max_abs_uz_liquid: float
    n_liquid_cells: int
    min_theta: float
    n_theta_floored: int
    p_gamma_std: float = 0.0          # spread of the imposed interface pressure
    p_gamma_range: float = 0.0
    reinit_applied: bool = False
    reinit_max_eta_shift: float = 0.0
    reinit_volume_change: float = 0.0
    reinit_contour_shift: float = 0.0          # max phi=0 displacement of this call (m)
    n_reinit_calls: int = 0
    cumulative_reinit_contour_shift: float = 0.0   # sum of per-call max shifts (m)
    volume_correction_applied: bool = False
    volume_correction_delta: float | None = None
    cumulative_volume_correction: float = 0.0


@dataclass
class SinglePhaseSolver:
    grid: Grid
    cfg: Config
    fields: Fields
    wall_bc: WallBC = field(default_factory=WallBC.no_slip)
    curvature_fn: Callable | None = field(default=None, repr=False)
    """VALIDATION ONLY (CAP-A): (grid, geom, phi) -> (kappa_r, kappa_z) at
    the crossings, replacing the numerical curvature. Constructor argument,
    never a YAML key."""
    reinit_callback: Callable | None = field(default=None, repr=False)
    """Diagnostic hook called as f(phi_before, phi_after, step) at each
    reinitialization event."""
    geometry: LiquidGeometry | None = field(default=None, repr=False)
    _initial_volume: float | None = field(default=None, repr=False)
    _cumulative_volume_correction: float = field(default=0.0, repr=False)
    _n_reinit: int = field(default=0, repr=False)
    _cumulative_reinit_contour: float = field(default=0.0, repr=False)

    def __post_init__(self) -> None:
        cfg = self.cfg
        if cfg.physics.free_surface_model != "single_phase_ls":
            raise ValueError("SinglePhaseSolver requires physics.free_surface_model='single_phase_ls'")
        if cfg.fluid.surface_tension < 0.0:
            raise ValueError("fluid.surface_tension must be >= 0")
        if cfg.stirrer.rpm_used != 0.0:
            raise NotImplementedError(
                "single_phase_ls: the local stirrer forcing is not connected until the "
                "Milestone-1 numerical gates pass (README_rewritten Milestone E). "
                "Use stirrer.rpm: 0 (validation) or the legacy two_phase_diffuse_ls model.")
        self._check_extension_width()
        if cfg.wall.contact_model == "pinned" and cfg.wall.pinned_contact_height_m is None:
            from .contact_angle import wall_contact_points
            zc = wall_contact_points(self.grid, self.fields.phi)
            if zc.size != 1:
                raise ValueError(f"pinned contact model: expected one wall contact point, found {zc.size}")
            cfg.wall.pinned_contact_height_m = float(zc[0])
        f = self.fields
        f.rho = np.full(self.grid.shape_center, cfg.fluid.water_density)
        f.mu = np.full(self.grid.shape_center, cfg.fluid.water_viscosity)
        self.geometry = classify(f.phi)
        f.u_r, f.u_z, f.u_theta, _ = extend_velocity(
            self.grid, self.geometry, f.phi, f.u_r, f.u_z,
            np.where(self.geometry.liquid, f.u_theta, 0.0),
            cfg.physics.extension_layers_capillary if cfg.fluid.surface_tension > 0
            else cfg.physics.extension_layers)
        f.p = np.where(self.geometry.liquid, f.p, 0.0)

    def _check_extension_width(self) -> None:
        ph = self.cfg.physics
        if (self.cfg.fluid.surface_tension > 0
                and ph.extension_layers_capillary < minimum_capillary_extension_layers()
                and not ph.allow_unsafe_extension_for_diagnostics):
            raise ValueError(
                f"physics.extension_layers_capillary = {ph.extension_layers_capillary} is below the "
                f"stencil-derived minimum {minimum_capillary_extension_layers()} for sigma > 0 "
                "(velocity_extension.minimum_capillary_extension_layers).")

    @property
    def cumulative_volume_correction(self) -> float:
        return self._cumulative_volume_correction

    def stable_timestep(self, u_r: np.ndarray, u_z: np.ndarray) -> float:
        cfg, grid = self.cfg, self.grid
        dt = compute_stable_timestep(grid, u_r, u_z, cfg.fluid.water_density,
                                     cfg.fluid.water_viscosity, cfg)
        # explicit free-surface kinematics: resolve the shortest gravity wave
        dx = min(grid.dr, grid.dz)
        dt_g = _GRAVITY_WAVE_CFL * np.sqrt(dx / cfg.fluid.gravity) if cfg.fluid.gravity > 0 else np.inf
        sigma = cfg.fluid.surface_tension
        dt_s = (cfg.physics.capillary_dt_factor
                * np.sqrt(cfg.fluid.water_density * dx**3 / (4.0 * np.pi * sigma))) if sigma > 0 else np.inf
        return float(min(dt, dt_g, dt_s))

    def interface_pressure_bc(self, geom: LiquidGeometry, phi: np.ndarray):
        """(p_Gamma on u_r faces, p_Gamma on u_z faces, kappa_r, kappa_z)."""
        sigma = self.cfg.fluid.surface_tension
        if sigma == 0.0:
            pr, pz = interface_pressure(geom)
            return pr, pz, None, None
        if self.curvature_fn is not None:
            kr, kz = self.curvature_fn(self.grid, geom, phi)
        else:
            kr, kz = interface_curvature(self.grid, geom, phi, self.cfg.wall)
        pr, pz = interface_pressure(geom, sigma=sigma, kappa_r=kr, kappa_z=kz)
        return pr, pz, kr, kz

    def step(self, dt: float | None = None) -> SinglePhaseStepDiagnostics:
        f, cfg, grid = self.fields, self.cfg, self.grid
        rho = cfg.fluid.water_density
        nu = cfg.fluid.water_viscosity / rho
        self._check_extension_width()
        n_layers = (cfg.physics.extension_layers_capillary if cfg.fluid.surface_tension > 0
                    else cfg.physics.extension_layers)

        geom = classify(f.phi)
        pg_r, pg_z, _, _ = self.interface_pressure_bc(geom, f.phi)
        u_r, u_z, u_th, _ = extend_velocity(grid, geom, f.phi, f.u_r, f.u_z, f.u_theta, n_layers)

        if dt is None:
            dt = self.stable_timestep(u_r, u_z)

        # ---- predictor (same operators as the legacy solver, constant properties)
        w_at_ur = uz_at_ur_locations(u_z)
        adv_r = u_r * upwind_derivative(u_r, u_r, grid.dr, axis=0) \
            + w_at_ur * upwind_derivative(u_r, w_at_ur, grid.dz, axis=1)
        r_f_safe = np.where(grid.r_f == 0.0, grid.dr, grid.r_f)
        centrifugal = u_theta_on_ur_faces(grid, u_th) ** 2 / r_f_safe[:, None]
        u_r_star = u_r + dt * (-adv_r + centrifugal + nu * laplacian_ur(grid, u_r))

        u_at_uz = ur_at_uz_locations(u_r)
        adv_z = u_at_uz * upwind_derivative(u_z, u_at_uz, grid.dr, axis=0) \
            + u_z * upwind_derivative(u_z, u_z, grid.dz, axis=1)
        u_z_star = u_z + dt * (-adv_z - cfg.fluid.gravity + nu * laplacian_uz(grid, u_z))

        u_r_c = interp_ur_to_center(u_r)
        u_z_c = interp_uz_to_center(u_z)
        adv_t = u_r_c * upwind_derivative(u_th, u_r_c, grid.dr, axis=0) \
            + u_z_c * upwind_derivative(u_th, u_z_c, grid.dz, axis=1)
        extra = u_r_c * u_th / grid.r_c[:, None]
        visc_t = nu * laplacian_utheta_dirichlet(grid, u_th, self.wall_bc.wall_u_theta(grid),
                                                 self.wall_bc.bottom_u_theta(grid))
        u_theta_new = np.where(geom.liquid, u_th + dt * (-adv_t - extra + visc_t), 0.0)

        u_r_star[0, :] = 0.0
        u_r_star[-1, :] = 0.0
        u_z_star[:, 0] = 0.0

        # ---- liquid-only pressure + projection, one shared geometry
        p = solve_liquid_pressure(grid, geom, u_r_star, u_z_star, rho, dt, pg_r, pg_z)
        u_r_new, u_z_new = project_liquid_velocity(grid, geom, u_r_star, u_z_star, p, rho, dt,
                                                   pg_r, pg_z)
        div_liq = liquid_divergence(grid, geom, u_r_new, u_z_new)

        # ---- kinematics: extended liquid velocity transports phi
        u_r_new, u_z_new, u_theta_new, _ = extend_velocity(grid, geom, f.phi, u_r_new, u_z_new,
                                                           u_theta_new, n_layers)
        phi_new = advect_level_set_advective(
            f.phi, u_r_new, u_z_new, grid, dt,
            scheme=cfg.levelset.advection_scheme,
            time_integrator=cfg.levelset.time_integrator,
            limiter=cfg.levelset.limiter)

        reinit_applied = False
        reinit_shift = 0.0
        reinit_dv = 0.0
        reinit_contour = 0.0
        ls = cfg.levelset
        every = ls.reinitialize_every
        if every > 0 and (f.step + 1) % every == 0:
            do_it = True
            if ls.reinitialization_trigger == "quality":
                do_it = signed_distance_error(grid, phi_new) > ls.reinitialization_quality_threshold
            if do_it:
                eta_before = free_surface_height(phi_new, grid)
                v_before = liquid_volume_subcell(grid, phi_new)
                phi_before = phi_new
                phi_new = reinitialize(phi_new, grid, ls)
                if self.reinit_callback is not None:
                    self.reinit_callback(phi_before, phi_new, f.step + 1)
                eta_after = free_surface_height(phi_new, grid)
                ok = np.isfinite(eta_before) & np.isfinite(eta_after)
                reinit_shift = float(np.max(np.abs(eta_after[ok] - eta_before[ok]))) if np.any(ok) else 0.0
                reinit_dv = liquid_volume_subcell(grid, phi_new) - v_before
                reinit_contour = contour_displacement(grid, phi_before, phi_new).max_shift
                reinit_applied = True
                self._n_reinit += 1
                self._cumulative_reinit_contour += reinit_contour

        vc_cfg = cfg.levelset.volume_correction
        vc_applied, vc_delta = False, None
        if vc_cfg.enabled:
            if self._initial_volume is None:
                self._initial_volume = liquid_volume_subcell(grid, f.phi)
            if (f.step + 1) % max(1, vc_cfg.every_n_steps) == 0:
                phi_new, res = apply_volume_correction(grid, phi_new, self._initial_volume,
                                                       volume_fn=liquid_volume_subcell)
                vc_applied, vc_delta = True, res.delta
                self._cumulative_volume_correction += res.delta

        f.u_r, f.u_z, f.u_theta, f.p, f.phi = u_r_new, u_z_new, u_theta_new, p, phi_new
        f.t += dt
        f.step += 1
        self.geometry = geom

        pg_all = np.concatenate([pg_r[np.isfinite(pg_r)], pg_z[np.isfinite(pg_z)]])
        liq_ur = geom.ur_face_known()
        liq_uz = geom.uz_face_known()
        return SinglePhaseStepDiagnostics(
            dt=dt, t=f.t,
            max_u_theta=float(np.max(np.abs(np.where(geom.liquid, f.u_theta, 0.0)))),
            max_downward_uz=float(-np.min(np.where(liq_uz, f.u_z, 0.0))),
            min_pressure=float(np.min(p[geom.liquid])),
            volume_water=liquid_volume_staircase(grid, f.phi),
            volume_water_subcell=liquid_volume_subcell(grid, f.phi),
            max_divergence_liquid=float(np.max(np.abs(div_liq))),
            max_abs_ur_liquid=float(np.max(np.abs(np.where(liq_ur, f.u_r, 0.0)))),
            max_abs_uz_liquid=float(np.max(np.abs(np.where(liq_uz, f.u_z, 0.0)))),
            n_liquid_cells=geom.n_liquid,
            min_theta=float(np.nanmin(np.concatenate([geom.theta_r_raw.ravel(),
                                                      geom.theta_z_raw.ravel()]))),
            n_theta_floored=geom.n_floored,
            p_gamma_std=float(np.std(pg_all)) if pg_all.size else 0.0,
            p_gamma_range=float(np.ptp(pg_all)) if pg_all.size else 0.0,
            reinit_applied=reinit_applied, reinit_max_eta_shift=reinit_shift,
            reinit_volume_change=reinit_dv,
            reinit_contour_shift=reinit_contour, n_reinit_calls=self._n_reinit,
            cumulative_reinit_contour_shift=self._cumulative_reinit_contour,
            volume_correction_applied=vc_applied, volume_correction_delta=vc_delta,
            cumulative_volume_correction=self._cumulative_volume_correction,
        )

    def run(self, t_end: float | None = None,
            on_step: Callable[[SinglePhaseStepDiagnostics], None] | None = None) -> None:
        t_end = self.cfg.time.t_end_s if t_end is None else t_end
        while self.fields.t < t_end:
            diag = self.step()
            if on_step is not None:
                on_step(diag)


def build_single_phase_solver(cfg: Config, wall_bc: WallBC | None = None) -> SinglePhaseSolver:
    grid = build_grid(cfg)
    fields = initialize_still_water(grid, cfg.geometry.water_height_m)
    return SinglePhaseSolver(grid=grid, cfg=cfg, fields=fields,
                             wall_bc=wall_bc or WallBC.no_slip())


def signed_distance_to_profile(grid: Grid, eta_fn: Callable[[np.ndarray], np.ndarray],
                                n_samples: int = 20001) -> np.ndarray:
    """phi(r,z) = signed Euclidean distance to the surface z = eta(|r|)
    (negative below = liquid), by dense sampling of the meridional curve
    mirrored through the axis. Sampling spacing 2R/(n_samples-1) gives a
    distance error O(spacing^2 * curvature), negligible at the default."""
    rs = np.linspace(-grid.r_v, grid.r_v, n_samples)
    zs = eta_fn(np.abs(rs))
    out = np.empty(grid.shape_center)
    for i in range(grid.Nr):
        d2 = (grid.r_c[i] - rs[None, :]) ** 2 + (grid.z_c[:, None] - zs[None, :]) ** 2
        out[i] = np.sign(grid.z_c - eta_fn(np.array(grid.r_c[i]))) * np.sqrt(d2.min(axis=1))
    return out


def initialize_rigid_body_single_phase(grid: Grid, cfg: Config, omega: float,
                                        phi_form: Literal["signed_distance", "height"] = "signed_distance",
                                        target_volume: float | None = None) -> Fields:
    """Manufactured rigid-body state for Gate V3 (sigma=0): u_theta = Omega r,
    u_r = u_z = 0, eta(r) = C + Omega^2 r^2/(2g) with C from the liquid
    volume, p = rho(Omega^2 r^2/2 - g(z - C)) in the liquid.

    ``phi_form="height"``: phi = z - eta(r). Then p = -rho g phi exactly, and
    the ghost-fluid face gradient with a linear phi crossing is exact for
    any p affine in phi -- so this form tests only round-off.
    ``phi_form="signed_distance"`` (default): phi is the true signed
    distance (the state reinitialization aims for), p is NOT affine in phi,
    and radial crossing errors O(h^2 kappa) are genuinely exercised."""
    from .diagnostics import volume_consistent_parabola_constant
    from .fields import initialize_rotating_equilibrium

    if target_volume is None:
        target_volume = np.pi * grid.r_v**2 * cfg.geometry.water_height_m
    fields = initialize_rotating_equilibrium(grid, cfg, omega, target_volume=target_volume)
    if phi_form == "signed_distance":
        g = cfg.fluid.gravity
        C = volume_consistent_parabola_constant(grid.r_v, omega, g, target_volume)
        fields.phi = signed_distance_to_profile(grid, lambda r: C + omega**2 * r**2 / (2.0 * g))
        rho_w = cfg.fluid.water_density
        r2d = grid.r_c[:, None] * np.ones(grid.shape_center)
        z2d = grid.z_c[None, :] * np.ones(grid.shape_center)
        p_w = 0.5 * rho_w * omega**2 * r2d**2 - rho_w * g * (z2d - C)
        fields.p = np.where(fields.phi < 0.0, p_w, 0.0)
    elif phi_form != "height":
        raise ValueError(f"unknown phi_form {phi_form!r}")
    return fields
