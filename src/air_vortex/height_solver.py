"""Opt-in research model ``physics.free_surface_model = "single_phase_height"``:
single-phase water with a HEIGHT-FUNCTION free surface z = eta(r, t).

Differences from single_phase_ls (which is untouched):
  * the authoritative interface is eta_i (radial cell centres); there is no
    evolved level set, so no off-contour degrees of freedom exist;
  * eta moves by the exactly conservative column finite-volume update
    d(A_i eta_i)/dt = F_{i-1/2} - F_{i+1/2}, F = 2 pi r_f int_0^eta u_r dz (axis
    and wall fluxes are zero, so sum_i A_i eta_i is conserved to round-off);
  * liquid mask, ghost-fluid crossing fractions and capillary curvature come
    from the graph reconstruction (height_interface.GraphInterface), with the
    pinned wall point eta(R) = z_pin imposed in the reconstruction.
Reused unchanged: MAC grid, momentum predictor, liquid pressure solve,
projection, velocity extension (for momentum stencils only, with the graph
normal of the DERIVED psi = z - eta), stable timestep.
fields.phi is set to the derived psi after each step for plotting/compat only.

Scope: single-valued graph interfaces (see height_interface.graph_validity).
No reinitialization, no volume correction. Stirrer forcing (forcing.py, uncalibrated tau_s) is
available for QUALITATIVE sanity runs only."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import Config
from .fields import Fields
from .free_surface_bc import interface_pressure
from .grid import Grid
from .height_interface import (GraphInterface, column_volume, curvature_at_crossings, derived_psi,
                               eta_rate, geometry_from_eta, graph_validity)
from .liquid_mask import LiquidGeometry
from .operators import (interp_ur_to_center, interp_uz_to_center, laplacian_ur, laplacian_uz,
                        u_theta_on_ur_faces, upwind_derivative, ur_at_uz_locations, uz_at_ur_locations)
from .pressure_single_phase import liquid_divergence, project_liquid_velocity, solve_liquid_pressure
from .single_phase_solver import (WallBC, _GRAVITY_WAVE_CFL,
                                  laplacian_utheta_dirichlet)
from .swirl_transport import SWIRL_MUSCL_CFL
from .timestep import compute_stable_timestep
from .velocity_extension import extend_velocity


@dataclass
class HeightStepDiagnostics:
    dt: float
    t: float
    max_abs_ur_liquid: float
    max_abs_uz_liquid: float
    max_divergence_liquid: float
    volume: float
    max_slope: float


@dataclass
class SinglePhaseHeightSolver:
    grid: Grid
    cfg: Config
    fields: Fields
    eta: np.ndarray
    z_pin: float
    wall_bc: WallBC = field(default_factory=WallBC.no_slip)
    curvature_fn: object = field(default=None, repr=False)
    """VALIDATION ONLY: (grid, gi, geom, r_cross) -> (kappa_r, kappa_z)."""
    geometry: LiquidGeometry | None = field(default=None, repr=False)
    crossing: str = "linear"
    freeze_interface: bool = False
    """DIAGNOSTIC (V6/V7 rigid tests): eta is not updated; the flat surface stays an
    atmospheric-pressure boundary (a fixed pressure outlet, NOT a no-penetration lid)."""
    rigid_lid: bool = False
    """DIAGNOSTIC (implies freeze_interface): the flat surface faces are no-penetration,
    free-slip walls (u_z = 0 there before and after projection). One axis-column surface face
    keeps the atmospheric Dirichlet value only to fix the pressure constant (its flux ~ 0)."""
    swirl_only: bool = False
    """DIAGNOSTIC: meridional velocity zeroed after every step (forcing + swirl viscosity only)."""
    swirl_advection: str | None = None
    """None -> cfg.physics.height_swirl_advection. One of swirl_transport.SWIRL_ADVECTION_MODES:
    "conservative_upwind1" / "conservative_muscl2" (flux form for q = r u_theta with the MAC face
    mass fluxes, swirl_transport.py) or "advective" (LEGACY first-order upwind u.grad(u_theta)
    + u_r u_theta / r shared with the other solvers; not angular-momentum conservative, kept for
    reproducing V6/V7 diagnostics). "conservative" is accepted as an alias of conservative_upwind1."""
    meridional_advection: str | None = None
    """None -> cfg.physics.height_meridional_advection: "advective" (current first-order upwind
    u.grad u of the shared predictor) or the flux-form "conservative_upwind1" /
    "conservative_muscl2" of meridional_transport.py (V7-T, opt-in)."""
    swirl_limiter: str | None = None
    """None -> cfg.physics.height_swirl_limiter (MUSCL2 only)."""
    swirl_viscous: str | None = None
    """None -> cfg.physics.height_swirl_viscous: "vector_laplacian" (legacy stencil) or
    "angular_momentum" (flux form, interior stresses telescope; swirl_transport.py)."""

    def __post_init__(self) -> None:
        cfg = self.cfg
        if cfg.physics.free_surface_model != "single_phase_height":
            raise ValueError("SinglePhaseHeightSolver requires physics.free_surface_model='single_phase_height'")
        # stirrer forcing: the production relaxation term of forcing.py (QUALITATIVE use only in
        # this research branch; tau_s is NOT calibrated). rpm = 0 leaves every path unchanged.
        from .swirl_transport import LIMITERS, SWIRL_ADVECTION_MODES, SWIRL_VISCOUS_MODES
        ph = cfg.physics
        if self.swirl_advection is None:
            self.swirl_advection = ph.height_swirl_advection
        if self.swirl_limiter is None:
            self.swirl_limiter = ph.height_swirl_limiter
        if self.meridional_advection is None:
            self.meridional_advection = ph.height_meridional_advection
        from .meridional_transport import MERIDIONAL_ADVECTION_MODES
        if self.meridional_advection not in MERIDIONAL_ADVECTION_MODES:
            raise ValueError(f"meridional_advection must be one of {MERIDIONAL_ADVECTION_MODES}")
        if self.swirl_viscous is None:
            self.swirl_viscous = ph.height_swirl_viscous
        if self.swirl_advection == "conservative":
            self.swirl_advection = "conservative_upwind1"
        if self.swirl_advection not in SWIRL_ADVECTION_MODES:
            raise ValueError(f"swirl_advection must be one of {SWIRL_ADVECTION_MODES}")
        if self.swirl_limiter not in LIMITERS or self.swirl_viscous not in SWIRL_VISCOUS_MODES:
            raise ValueError("invalid swirl_limiter / swirl_viscous")
        if self.swirl_advection == "advective" and not ph.height_legacy_swirl_for_validation:
            import warnings
            warnings.warn("single_phase_height with the legacy non-conservative 'advective' swirl transport "
                          "(diagnostic reproduction only)", UserWarning, stacklevel=2)
        self._chi = None
        if cfg.stirrer.model == "moving_footprint":
            # V7-S M2: the stirrer acts only through the effective moving bottom (no volume forcing)
            from .forcing import MovingFootprintWallBC
            if getattr(self.wall_bc, "kind", None) != "no_slip":
                raise ValueError("moving_footprint replaces the stationary no-slip bottom; other wall_bc not allowed")
            self.wall_bc = MovingFootprintWallBC(cfg, lambda: self.fields.t)
        elif cfg.stirrer.rpm_used != 0.0:
            from .forcing import forcing_mask
            self._chi = forcing_mask(self.grid, cfg)
        self.eta = np.asarray(self.eta, float).copy()
        f = self.fields
        f.rho = np.full(self.grid.shape_center, cfg.fluid.water_density)
        f.mu = np.full(self.grid.shape_center, cfg.fluid.water_viscosity)
        gi = self.interface()
        self.geometry, _ = geometry_from_eta(self.grid, gi)
        f.phi = derived_psi(self.grid, self.eta)
        f.p = np.where(self.geometry.liquid, f.p, 0.0)

    # ------------------------------------------------------------------ geometry
    def interface(self, eta=None) -> GraphInterface:
        return GraphInterface(self.grid, self.eta if eta is None else eta, self.z_pin, self.crossing)

    @property
    def volume(self) -> float:
        return column_volume(self.grid, self.eta)

    def n_layers(self) -> int:
        ph = self.cfg.physics
        return ph.extension_layers_capillary if self.cfg.fluid.surface_tension > 0 else ph.extension_layers

    def interface_pressure_bc(self, gi, geom, r_cross):
        sigma = self.cfg.fluid.surface_tension
        if sigma == 0.0:
            pr, pz = interface_pressure(geom)
            return pr, pz
        if self.curvature_fn is not None:
            kr, kz = self.curvature_fn(self.grid, gi, geom, r_cross)
        else:
            kr, kz = curvature_at_crossings(self.grid, gi, geom, r_cross)
        return interface_pressure(geom, sigma=sigma, kappa_r=kr, kappa_z=kz)

    def torque_budget(self) -> dict:
        """Angular-momentum accounting (z-component, liquid only, SI):
        L_z = int rho r u_theta dV; T_stir = int rho r f_theta dV (forcing.py term at the
        current state); T_side = wall shear torque 2 pi R^2 int mu R d(u/r)/dr dz (from the
        wall-adjacent cells, u_wall = Omega_wall R); T_bottom = int r mu du/dz 2 pi r dr at
        z = 0. Viscous torques are the torques EXERTED ON the liquid (negative when the
        liquid spins faster than the walls)."""
        g, f, cfg = self.grid, self.fields, self.cfg
        rho, mu = cfg.fluid.water_density, cfg.fluid.water_viscosity
        liq = self.geometry.liquid if self.geometry is not None else (g.z_c[None, :] < self.eta[:, None])
        r2 = g.r_c[:, None] * np.ones(g.shape_center)
        dV = 2 * np.pi * r2 * g.dr * g.dz
        ut = np.where(liq, f.u_theta, 0.0)
        out = {"L_z": float(np.sum(rho * r2 * ut * dV))}
        T_stir = 0.0
        if self._chi is not None:
            from .forcing import stirrer_forcing
            fs = stirrer_forcing(f.u_theta, cfg.stirrer.omega(f.t), g, cfg, self._chi)
            T_stir = float(np.sum(rho * r2 * np.where(liq, fs, 0.0) * dV))
        out["T_stir"] = T_stir
        R = g.r_v
        uw = self.wall_bc.wall_u_theta(g)
        uw = uw if np.ndim(uw) else np.full(g.Nz, float(uw))
        om_c, om_w = ut[-1, :] / g.r_c[-1], uw / R
        tau_side = mu * R * (om_w - om_c) / (0.5 * g.dr)
        out["T_side"] = float(np.sum(np.where(liq[-1], tau_side * R * 2 * np.pi * R * g.dz, 0.0)))
        ub = self.wall_bc.bottom_u_theta(g)
        ub = ub if np.ndim(ub) else np.full(g.Nr, float(ub))
        tau_b = mu * (ub - ut[:, 0]) / (0.5 * g.dz)
        tb = tau_b * g.r_c * 2 * np.pi * g.r_c * g.dr
        out["T_bottom"] = float(np.sum(tb))
        under = g.r_c < cfg.geometry.stirbar_half_length_m          # V7-S split (diagnostic only)
        out["T_bottom_under"] = float(np.sum(tb[under]))
        out["T_bottom_outside"] = float(np.sum(tb[~under]))
        return out

    def _frozen_pressure(self, grid, geom, u_r_star, u_z_star, rho, dt, pg_r, pg_z):
        """Same system as solve_liquid_pressure, but the sparse LU is reused while the matrix is
        unchanged (frozen-interface / rigid-lid diagnostics only; the free-surface path always
        uses solve_liquid_pressure). Agrees with spsolve to round-off."""
        import scipy.sparse.linalg as spla
        from .operators import divergence
        from .pressure_single_phase import build_liquid_pressure_system
        sys_ = build_liquid_pressure_system(grid, geom, pg_r, pg_z, "error")
        A = (-sys_.A).tocsc()
        key = (A.shape, hash(A.indptr.tobytes()), hash(A.indices.tobytes()), hash(A.data.tobytes()))
        if getattr(self, "_lu_key", None) != key:
            self._lu, self._lu_key = spla.splu(A), key
        liq = geom.liquid
        rhs = ((rho / dt) * divergence(grid, u_r_star, u_z_star) * sys_.weight)[liq] - sys_.b_dirichlet
        p = np.zeros((grid.Nr, grid.Nz))
        p[liq] = self._lu.solve(-rhs)
        return p

    def stable_timestep(self, u_r, u_z) -> float:
        cfg, grid = self.cfg, self.grid
        dt = compute_stable_timestep(grid, u_r, u_z, cfg.fluid.water_density, cfg.fluid.water_viscosity, cfg)
        dx = min(grid.dr, grid.dz)
        dt_g = _GRAVITY_WAVE_CFL * np.sqrt(dx / cfg.fluid.gravity) if cfg.fluid.gravity > 0 else np.inf
        sigma = cfg.fluid.surface_tension
        dt_s = (cfg.physics.capillary_dt_factor * np.sqrt(cfg.fluid.water_density * dx**3 / (4.0 * np.pi * sigma))
                if sigma > 0 else np.inf)
        if "conservative_muscl2" in (self.swirl_advection, self.meridional_advection):
            # TVD bound of the limited MUSCL fluxes (swirl_transport module docstring)
            rate = np.max(np.abs(u_r)) / grid.dr + np.max(np.abs(u_z)) / grid.dz
            dt = min(dt, SWIRL_MUSCL_CFL / max(rate, 1e-30))
        return float(min(dt, dt_g, dt_s))

    # ------------------------------------------------------------------ step
    def step(self, dt: float | None = None) -> HeightStepDiagnostics:
        f, cfg, grid = self.fields, self.cfg, self.grid
        rho = cfg.fluid.water_density
        nu = cfg.fluid.water_viscosity / rho
        nl = self.n_layers()
        gi = self.interface()
        geom, r_cross = geometry_from_eta(grid, gi)
        psi = derived_psi(grid, self.eta)                    # graph normal for the extension only
        pg_r, pg_z = self.interface_pressure_bc(gi, geom, r_cross)
        lid = None
        if self.rigid_lid:
            from .liquid_mask import FACE_INACTIVE, FACE_LIQ_MINUS
            lid = geom.face_kind_z == FACE_LIQ_MINUS
            keep = np.zeros_like(lid); keep[0] = lid[0]          # axis-column face: pressure reference
            geom.face_kind_z[lid & ~keep] = FACE_INACTIVE
            geom.theta_z[lid & ~keep] = np.nan
            pg_z = np.where(lid & ~keep, np.nan, pg_z)
        u_r, u_z, u_th, _ = extend_velocity(grid, geom, psi, f.u_r, f.u_z, f.u_theta, nl)
        if lid is not None:
            # lid faces are FACE_INACTIVE, so the extension overwrote the no-penetration value
            u_z[lid] = 0.0
        if dt is None:
            dt = self.stable_timestep(u_r, u_z)

        # ---- predictor (identical operators to SinglePhaseSolver)
        if self.meridional_advection == "advective":
            w_at_ur = uz_at_ur_locations(u_z)
            adv_r = u_r * upwind_derivative(u_r, u_r, grid.dr, axis=0) \
                + w_at_ur * upwind_derivative(u_r, w_at_ur, grid.dz, axis=1)
            u_at_uz = ur_at_uz_locations(u_r)
            adv_z = u_at_uz * upwind_derivative(u_z, u_at_uz, grid.dr, axis=0) \
                + u_z * upwind_derivative(u_z, u_z, grid.dz, axis=1)
        else:
            from .meridional_transport import advect_meridional
            adv_r, adv_z = advect_meridional(grid, u_r, u_z, dt,
                                             1 if self.meridional_advection == "conservative_upwind1" else 2,
                                             self.swirl_limiter)
        r_f_safe = np.where(grid.r_f == 0.0, grid.dr, grid.r_f)
        centrifugal = u_theta_on_ur_faces(grid, u_th) ** 2 / r_f_safe[:, None]
        u_r_star = u_r + dt * (-adv_r + centrifugal + nu * laplacian_ur(grid, u_r))
        u_z_star = u_z + dt * (-adv_z - cfg.fluid.gravity + nu * laplacian_uz(grid, u_z))
        u_r_c = interp_ur_to_center(u_r)
        u_z_c = interp_uz_to_center(u_z)
        if self.swirl_advection == "advective":
            adv_t = u_r_c * upwind_derivative(u_th, u_r_c, grid.dr, axis=0) \
                + u_z_c * upwind_derivative(u_th, u_z_c, grid.dz, axis=1)
            extra = u_r_c * u_th / grid.r_c[:, None]
        else:
            # dq/dt + (1/r) d(r u_r q)/dr + d(u_z q)/dz = 0, q = r u_theta, MAC face mass fluxes
            from .swirl_transport import q_flux_divergence, advect_q
            q = u_th * grid.r_c[:, None]
            if self.swirl_advection == "conservative_upwind1":
                adv_t = q_flux_divergence(grid, q, u_r, u_z, 1) / grid.r_c[:, None]
            else:
                q_new = advect_q(grid, q, u_r, u_z, dt, 2, self.swirl_limiter)
                adv_t = (q - q_new) / (dt * grid.r_c[:, None])
            extra = 0.0
        if self.swirl_viscous == "angular_momentum":
            from .swirl_transport import angular_momentum_viscous
            visc_t = nu * angular_momentum_viscous(grid, u_th, self.wall_bc.wall_u_theta(grid),
                                                   self.wall_bc.bottom_u_theta(grid))
        else:
            visc_t = nu * laplacian_utheta_dirichlet(grid, u_th, self.wall_bc.wall_u_theta(grid),
                                                     self.wall_bc.bottom_u_theta(grid))
        rhs_t = -adv_t - extra + visc_t
        if self._chi is not None:
            from .forcing import stirrer_forcing
            rhs_t = rhs_t + stirrer_forcing(u_th, cfg.stirrer.omega(f.t), grid, cfg, self._chi)
        u_theta_new = np.where(geom.liquid, u_th + dt * rhs_t, 0.0)
        u_r_star[0, :] = 0.0
        u_r_star[-1, :] = 0.0
        u_z_star[:, 0] = 0.0
        if lid is not None:
            u_z_star[lid] = 0.0

        # ---- liquid-only pressure + projection (same geometry instance)
        if self.freeze_interface or self.rigid_lid:
            p = self._frozen_pressure(grid, geom, u_r_star, u_z_star, rho, dt, pg_r, pg_z)
        else:
            p = solve_liquid_pressure(grid, geom, u_r_star, u_z_star, rho, dt, pg_r, pg_z)
        u_r_new, u_z_new = project_liquid_velocity(grid, geom, u_r_star, u_z_star, p, rho, dt, pg_r, pg_z)
        div_liq = liquid_divergence(grid, geom, u_r_new, u_z_new)
        u_r_new, u_z_new, u_theta_new, _ = extend_velocity(grid, geom, psi, u_r_new, u_z_new, u_theta_new, nl)
        if lid is not None:
            self.lid_flux = float(np.sum(u_z_new[lid] * 2 * np.pi * grid.r_c[np.nonzero(lid)[0]] * grid.dr))
            u_z_new[lid] = 0.0

        # ---- conservative height update, SSPRK2 with the projected velocity
        if self.swirl_only:
            u_r_new = np.zeros_like(u_r_new)
            u_z_new = np.zeros_like(u_z_new)
        if self.freeze_interface or self.rigid_lid:
            e2 = self.eta
        else:
            e1 = self.eta + dt * eta_rate(grid, self.eta, self.z_pin, u_r_new, self.crossing)
            e2 = 0.5 * self.eta + 0.5 * (e1 + dt * eta_rate(grid, e1, self.z_pin, u_r_new, self.crossing))

        f.u_r, f.u_z, f.u_theta, f.p = u_r_new, u_z_new, u_theta_new, p
        self.eta = e2
        f.phi = derived_psi(grid, self.eta)
        f.t += dt
        f.step += 1
        self.geometry = geom
        liq_ur, liq_uz = geom.ur_face_known(), geom.uz_face_known()
        return HeightStepDiagnostics(
            dt=dt, t=f.t,
            max_abs_ur_liquid=float(np.max(np.abs(np.where(liq_ur, f.u_r, 0.0)))),
            max_abs_uz_liquid=float(np.max(np.abs(np.where(liq_uz, f.u_z, 0.0)))),
            max_divergence_liquid=float(np.max(np.abs(div_liq))),
            volume=self.volume, max_slope=graph_validity(grid, self.interface())["max_slope"])
