"""Configuration schema and YAML loading (README section 29)."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .steady_state import SteadyStateConfig


@dataclass
class GeometryConfig:
    vessel_radius_m: float
    vessel_height_m: float
    water_height_m: float
    air_height_m: float
    stirbar_length_m: float
    stirbar_diameter_m: float
    stirbar_center_z_m: float

    @property
    def stirbar_half_length_m(self) -> float:
        """R_m = L_m / 2 (README section 1.3)."""
        return self.stirbar_length_m / 2.0

    @property
    def stirbar_top_z_m(self) -> float:
        """z_bar,top (README section 1.3)."""
        return self.stirbar_diameter_m

    @property
    def z_max_m(self) -> float:
        return self.water_height_m + self.air_height_m

    @property
    def effective_depth_m(self) -> float:
        """H_eff = H - z_bar,top (README section 3, Q3)."""
        return self.water_height_m - self.stirbar_top_z_m

    @property
    def confinement_ratio(self) -> float:
        """C = R_m / R_v (README section 17)."""
        return self.stirbar_half_length_m / self.vessel_radius_m

    @property
    def bar_aspect_ratio(self) -> float:
        """AR = L_m / D_m (README section 17)."""
        return self.stirbar_length_m / self.stirbar_diameter_m


@dataclass
class FluidConfig:
    water_density: float
    water_viscosity: float
    air_density: float
    air_viscosity: float
    surface_tension: float
    gravity: float = 9.81


STIRRER_MODELS = ("wall_touching_volume", "tapered_volume", "moving_footprint")


@dataclass
class StirrerConfig:
    rpm: float
    ramp_time_s: float
    forcing_tau_s: float
    forcing_smoothing_m: float
    actual_rpm: float | None = None
    """Measured stir-bar RPM (README section 33), distinct from the
    stirrer's *display/set* RPM (``rpm``) -- magnetic slip/step-out can
    make these differ at high load. When present, this is what the
    simulation physics actually uses (see ``rpm_used``); ``rpm`` alone
    stays available as the setpoint for reference/labeling."""
    model: str = "wall_touching_volume"
    """Effective stirrer representation (V7-S, docs sec. 10.z). "wall_touching_volume" (default,
    legacy: chi(r, z) of forcing.forcing_mask reaches the no-slip bottom); "tapered_volume" (M1,
    chi_z additionally multiplied by a C2 smoothstep S(z / bottom_taper_m), zero at the bottom);
    "moving_footprint" (M2, single_phase_height only: NO volume forcing; the bottom tangential
    velocity under the bar is Omega(t) r w(r), w = (1 - tanh((r - R_m)/forcing_smoothing_m))/2).
    Numerical model candidates, not calibrated physics."""
    bottom_taper_m: float = 0.0
    """tapered_volume only: physical taper length ell_z (> 0; never a number of cells)."""
    calibrated: bool = False
    """Whether ``forcing_tau_s`` came from an actual calibration run
    (scripts/calibrate_forcing.py) against a measured vortex depth, rather
    than being an unvalidated initial guess (README section 10.2). Read by
    scripts/check_validation_gate.py; a pilot/production sweep should not
    be treated as a validated physical prediction while this is False."""

    @property
    def rpm_used(self) -> float:
        """The RPM value the physics actually uses: actual_rpm if known,
        else the setpoint rpm (README section 33 priority rule)."""
        return self.actual_rpm if self.actual_rpm is not None else self.rpm

    @property
    def rpm_source(self) -> str:
        return "actual" if self.actual_rpm is not None else "setpoint_fallback"

    @property
    def omega_target(self) -> float:
        """Omega_m = 2*pi*N/60 (README section 4 / 18), using rpm_used."""
        return 2.0 * math.pi * self.rpm_used / 60.0

    def omega(self, t: float) -> float:
        """Smooth ramp Omega(t) = Omega_target * (1 - exp(-t/t_r)) (README section 13)."""
        if self.ramp_time_s <= 0:
            return self.omega_target
        return self.omega_target * (1.0 - math.exp(-t / self.ramp_time_s))

    def __post_init__(self) -> None:
        if self.model not in STIRRER_MODELS:
            raise ValueError(f"stirrer.model must be one of {STIRRER_MODELS}, got {self.model!r}")
        if self.model == "tapered_volume" and not self.bottom_taper_m > 0:
            raise ValueError("stirrer.bottom_taper_m must be > 0 (a physical length) for tapered_volume")


@dataclass
class GridConfig:
    dr_m: float
    dz_m: float


@dataclass
class TimeConfig:
    t_end_s: float
    cfl: float
    dt_max_s: float


@dataclass
class VolumeCorrectionConfig:
    enabled: bool = False
    """Default OFF (README "optional global volume correction"): Level Set
    advection is non-conservative, so an explicit correction can mask
    genuine accuracy problems if enabled by default -- Case A-D comparisons
    (README "Level Set advection accuracy") must always be run with this
    OFF as well as ON, never ON-only."""
    target: str = "initial"
    """Only "initial" (the water volume at t=0) is currently implemented."""
    every_n_steps: int = 10
    tolerance_relative: float = 1e-4


@dataclass
class LevelSetConfig:
    interface_width_cells: float
    reinitialize_every: int
    reinitialize_iterations: int
    advection_scheme: str = "upwind1"
    """"upwind1" (default, byte-identical to the original solver) or
    "muscl2" (2nd-order MUSCL-TVD, README "Level Set advection accuracy")."""
    time_integrator: str = "euler"
    """"euler" (default) or "ssprk2" (SSP-RK2, only meaningful paired with
    advection_scheme="muscl2" -- upwind1+ssprk2 is implemented but not the
    combination this work is about)."""
    limiter: str = "mc"
    """"mc" (default, monotonized-central) or "minmod", used only when
    advection_scheme="muscl2"."""
    volume_correction: VolumeCorrectionConfig = field(default_factory=VolumeCorrectionConfig)
    reinitialization_method: str = "legacy_godunov"
    """"legacy_godunov" (default; levelset.reinitialize_level_set, the only
    method the two_phase_diffuse_ls solver accepts) or
    "russo_smereka_subcell" (reinit_subcell.py, single_phase_ls only)."""
    reinitialization_order: int = 2
    """russo_smereka_subcell: 1 = first-order RS with the linear crossing,
    2 = Min-Gibou ENO2 with quadratic crossing."""
    reinitialization_cfl: float = 0.45
    """russo_smereka_subcell pseudo-time CFL (Min 2010: 0.45 in 2D)."""
    reinitialization_band_cells: float = 0.0
    """russo_smereka_subcell: 0 = whole domain, > 0 = only |phi0| <= N*dx."""
    reinitialization_trigger: str = "periodic"
    """"periodic" (every reinitialize_every steps) or "quality"
    (DIAGNOSTIC: reinitialize only when E_sd > reinitialization_quality_threshold,
    checked every reinitialize_every steps). single_phase_ls only."""
    reinitialization_quality_threshold: float = 0.05

    def __post_init__(self) -> None:
        if self.reinitialization_method not in ("legacy_godunov", "russo_smereka_subcell"):
            raise ValueError(f"unknown levelset.reinitialization_method "
                             f"{self.reinitialization_method!r}")
        if self.reinitialization_trigger not in ("periodic", "quality"):
            raise ValueError(f"unknown levelset.reinitialization_trigger "
                             f"{self.reinitialization_trigger!r}")


CONTACT_MODELS = ("extrapolate", "static_angle", "pinned")


@dataclass
class WallConfig:
    """Free-surface / side-wall contact geometry (single_phase_ls with
    sigma > 0; Gate V4b, docs/level1a_single_phase.md sec. 9).

    ``contact_model``:

    - ``"extrapolate"`` (default): phi's wall ghost is linearly
      extrapolated; the contact ANGLE is free and follows the interior
      interface. With the no-slip wall the contact LINE is kinematically
      pinned up to O(dr/2) numerical slip (model CL-P, pinned). This is
      exactly the V4 behaviour.
    - ``"pinned"``: model CL-P. The wall trace of phi is shifted so its zero
      stays at ``pinned_contact_height_m`` (captured from the initial phi
      when None); the angle is free. This matches the no-slip wall at the
      macroscopic level.
    - ``"static_angle"``: the wall ghost enforces n . e_r = -cos(theta)
      (theta measured THROUGH THE LIQUID). This is valid for STATIC
      equilibria. It is NOT a moving-contact-line model with the no-slip
      wall (model CL-A; see V4b-M in the docs).

    ``contact_angle_deg`` has no default on purpose: the water/glass value
    for the experiment is unmeasured, and no config may claim one."""
    contact_model: str = "extrapolate"
    contact_angle_deg: float | None = None
    pinned_contact_height_m: float | None = None
    pinned_method: str = "reconstruct_ghost"
    """pinned only. "reconstruct_ghost": constrained interface reconstruction
    -> ghost phi (V4b-P). "reconstruct_direct": same ghost plus direct
    reconstruction curvature at the wall-column crossings (diagnostic
    hybrid). "legacy": the V4b cubic-window ghost (diagnostic comparison
    only; grid-phase dependent)."""
    pinned_fit: str = "quadratic"
    """"quadratic" | "quadratic_weighted" | "cubic" (constrained LSQ, f(R) = z_pin)."""
    pinned_fit_columns: int = 4
    pinned_skip_wall_column: bool = False
    wall_curvature: str = "level_set"
    """Capillary curvature at the crossings of the last ``wall_curvature_band``
    columns. "level_set" (default): the bulk div(n) path everywhere (bitwise
    the pre-existing behaviour). "graph": wall_curvature.graph_band_curvature,
    a constrained fit of the column interface heights evaluated at each
    crossing (V4b-P redesign)."""
    wall_curvature_band: int = 6
    wall_curvature_order: int = 2
    wall_curvature_fit: str = "local"
    """"local": a moving fit of the ``wall_curvature_window`` columns nearest
    each crossing; "global": one fit over the whole band."""
    wall_curvature_window: int = 3
    """Points per local fit. order + 1 = interpolating height-function stencil
    (required for stability); larger = least-squares smoothing, which is
    anti-restoring at the grid scale (diagnostic negative option only)."""
    wall_curvature_blend: int = 0
    """0: hard switch at the band edge; n > 0: linear blend over n columns."""

    def __post_init__(self) -> None:
        if self.contact_model not in CONTACT_MODELS:
            raise ValueError(f"wall.contact_model must be one of {CONTACT_MODELS}")
        if self.pinned_method not in ("reconstruct_ghost", "reconstruct_ghost_distance",
                                      "reconstruct_direct", "legacy"):
            raise ValueError(f"unknown wall.pinned_method {self.pinned_method!r}")
        if self.pinned_fit not in ("quadratic", "quadratic_weighted", "cubic"):
            raise ValueError(f"unknown wall.pinned_fit {self.pinned_fit!r}")
        if self.contact_model == "static_angle":
            if self.contact_angle_deg is None or not (0.0 < self.contact_angle_deg < 180.0):
                raise ValueError("wall.contact_model='static_angle' needs 0 < contact_angle_deg < 180")
        if self.wall_curvature not in ("level_set", "graph"):
            raise ValueError(f"unknown wall.wall_curvature {self.wall_curvature!r}")
        if self.wall_curvature_fit not in ("local", "global"):
            raise ValueError(f"unknown wall.wall_curvature_fit {self.wall_curvature_fit!r}")
        if self.wall_curvature_order not in (2, 3, 4) or self.wall_curvature_band < 1                 or self.wall_curvature_blend < 0 or self.wall_curvature_window < self.wall_curvature_order + 1:
            raise ValueError("invalid wall.wall_curvature_{order,band,blend,window}")


FREE_SURFACE_MODELS = ("two_phase_diffuse_ls", "single_phase_ls", "single_phase_height")


@dataclass
class PhysicsConfig:
    """Architecture selector (README section 19.2).

    ``free_surface_model``:

    - ``"two_phase_diffuse_ls"`` -- Level 1B / legacy: water AND air solved
      together with rho(phi), mu(phi), variable-density projection and CSF
      surface tension (:class:`air_vortex.solver.Solver`). Still the
      default during the transition so every pre-existing config, test and
      golden fixture keeps running the exact same code path.
    - ``"single_phase_ls"`` -- Level 1A: water-only momentum/pressure solve
      with constant rho_w, mu_w; the Level Set is geometry only; the
      atmosphere enters as a sharp Dirichlet pressure at the sub-cell phi=0
      crossing (:class:`air_vortex.single_phase_solver.SinglePhaseSolver`).
    - ``"single_phase_height"`` -- opt-in RESEARCH branch: same water-only
      solve, but the free surface is a height function z = eta(r) moved by an
      exactly conservative column flux (:class:`air_vortex.height_solver.
      SinglePhaseHeightSolver`). Single-valued interfaces only.

    A config file without a ``physics`` section gets the legacy default."""
    free_surface_model: str = "two_phase_diffuse_ls"
    extension_layers: int = 3
    """single_phase_ls only: width (in cells) of the narrow void band into
    which liquid velocity is extended for Level Set transport
    (velocity_extension.py). Not physical air flow."""
    extension_layers_capillary: int = 6
    """single_phase_ls with sigma > 0: extension band width used INSTEAD of
    extension_layers. The extended velocity drops to zero at the band edge;
    the resulting |grad phi| kink must stay outside the reach of the
    curvature stencil (+-1 cell around interface-adjacent nodes) plus the
    MUSCL stencil (+-2 cells). With 3 layers the static-drop CAP-B test
    became unstable after ~1 s (E_sd -> 0.8, U -> 0.5 m/s); with 6 or 12 it
    stayed at E_sd ~ 3e-3 (Gate V4, docs sec. 8). The sigma = 0 path keeps
    extension_layers so its results are unchanged."""
    allow_unsafe_extension_for_diagnostics: bool = False
    """Permit extension_layers_capillary below the derived minimum
    (velocity_extension.minimum_capillary_extension_layers). Diagnostics
    only; never in a production config."""
    capillary_dt_factor: float = 1.0
    """single_phase_ls only: dt <= factor * sqrt(rho_w dx^3 / (4 pi sigma))
    (Brackbill-Kothe-Zemach 1992 capillary-wave limit with the gas density
    set to zero). Used only when sigma > 0."""
    height_swirl_advection: str = "conservative_muscl2"
    """single_phase_height only: swirl transport (swirl_transport.SWIRL_ADVECTION_MODES).
    Default (V7-T): second-order limited flux form, validated (manufactured order 2.0, L_z to
    round-off, height gates H3/H4/V4b-H/integrated unchanged). "conservative_upwind1" is the
    first-order flux form (the V6/V7 "conservative" diagnostic).
    "advective" is the LEGACY non-conservative form (angular-momentum budget errors of 10-80 %
    in the V6/V7 rigid-lid runs, docs sec. 10); selecting it warns unless
    height_legacy_swirl_for_validation is set. Ignored by the other models."""
    height_meridional_advection: str = "advective"
    """single_phase_height only: meridional momentum advection, "advective" (first-order upwind
    u.grad u shared with the other solvers) or the opt-in flux forms "conservative_upwind1" /
    "conservative_muscl2" (meridional_transport.py, V7-T)."""
    height_swirl_limiter: str = "vanleer"
    """single_phase_height, conservative_muscl2 only: slope limiter (swirl_transport.LIMITERS)."""
    height_swirl_viscous: str = "angular_momentum"
    """single_phase_height only: "angular_momentum" (default, V7-T: flux form of
    (1/r^2) d/dr(r^3 d(u/r)/dr); viscous stresses exchange angular momentum only with the walls,
    budget closes to ~1e-5) or "vector_laplacian" (legacy stencil shared with the LS path)."""
    height_legacy_swirl_for_validation: bool = False
    """Silences the warning for height_swirl_advection = "advective" (reproducing old diagnostics)."""

    def __post_init__(self) -> None:
        if (self.free_surface_model == "single_phase_height" and self.height_swirl_advection == "advective"
                and not self.height_legacy_swirl_for_validation):
            import warnings
            warnings.warn("physics.height_swirl_advection='advective' is the legacy NON-conservative swirl "
                          "transport (V6/V7: 10-80 % angular-momentum error); use it only to reproduce old "
                          "diagnostics (set height_legacy_swirl_for_validation: true)", UserWarning, stacklevel=2)
        if self.free_surface_model not in FREE_SURFACE_MODELS:
            raise ValueError(
                f"physics.free_surface_model must be one of {FREE_SURFACE_MODELS}, "
                f"got {self.free_surface_model!r}")
        if self.extension_layers < 1:
            raise ValueError("physics.extension_layers must be >= 1")
        if self.extension_layers_capillary < 1:
            raise ValueError("physics.extension_layers_capillary must be >= 1")
        if self.capillary_dt_factor <= 0:
            raise ValueError("physics.capillary_dt_factor must be > 0")


@dataclass
class AirCoreConfig:
    contact_tolerance_cells: int
    persistence_rotations: float


@dataclass
class OutputConfig:
    save_every_s: float


@dataclass
class Config:
    geometry: GeometryConfig
    fluid: FluidConfig
    stirrer: StirrerConfig
    grid: GridConfig
    time: TimeConfig
    levelset: LevelSetConfig
    air_core: AirCoreConfig
    output: OutputConfig
    steady_state: SteadyStateConfig = field(default_factory=SteadyStateConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    wall: WallConfig = field(default_factory=WallConfig)
    source_path: Path | None = field(default=None, repr=False)


def _build_levelset_config(raw_levelset: dict) -> LevelSetConfig:
    """LevelSetConfig has a nested VolumeCorrectionConfig dataclass field,
    which a plain ``LevelSetConfig(**raw_levelset)`` can't construct from a
    raw dict -- convert it explicitly. Missing/absent -> defaults (OFF),
    so existing configs without a volume_correction section are unaffected."""
    raw_levelset = dict(raw_levelset)
    vc_raw = raw_levelset.pop("volume_correction", None)
    volume_correction = VolumeCorrectionConfig(**vc_raw) if vc_raw else VolumeCorrectionConfig()
    return LevelSetConfig(volume_correction=volume_correction, **raw_levelset)


def load_config(path: str | Path) -> Config:
    """Load a YAML config file matching the schema in README section 29
    (plus the optional ``steady_state`` section, README "steady-state
    detector"; a config file without it gets the SteadyStateConfig
    defaults, so existing configs keep working unchanged)."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    steady_state = SteadyStateConfig(**raw["steady_state"]) if "steady_state" in raw else SteadyStateConfig()

    return Config(
        geometry=GeometryConfig(**raw["geometry"]),
        fluid=FluidConfig(**raw["fluid"]),
        stirrer=StirrerConfig(**raw["stirrer"]),
        grid=GridConfig(**raw["grid"]),
        time=TimeConfig(**raw["time"]),
        levelset=_build_levelset_config(raw["levelset"]),
        air_core=AirCoreConfig(**raw["air_core"]),
        output=OutputConfig(**raw["output"]),
        steady_state=steady_state,
        physics=PhysicsConfig(**raw["physics"]) if "physics" in raw else PhysicsConfig(),
        wall=WallConfig(**raw["wall"]) if "wall" in raw else WallConfig(),
        source_path=path,
    )


def config_to_dict(cfg: Config) -> dict:
    """Inverse of :func:`load_config`: a plain dict matching the section 29
    YAML schema, suitable for ``yaml.safe_dump`` (used by the run-storage
    layer to save ``config.yaml`` alongside a run's results)."""
    import dataclasses

    def section(obj) -> dict:
        return {k: v for k, v in dataclasses.asdict(obj).items()}

    return {
        "geometry": section(cfg.geometry),
        "fluid": section(cfg.fluid),
        "stirrer": section(cfg.stirrer),
        "grid": section(cfg.grid),
        "time": section(cfg.time),
        "levelset": section(cfg.levelset),
        "air_core": section(cfg.air_core),
        "output": section(cfg.output),
        "steady_state": section(cfg.steady_state),
        "physics": section(cfg.physics),
        "wall": section(cfg.wall),
    }


def config_from_dict(raw: dict) -> Config:
    """Inverse of :func:`config_to_dict`, for reloading a saved run's
    ``config.yaml`` without needing the original source file path."""
    steady_state = SteadyStateConfig(**raw["steady_state"]) if "steady_state" in raw else SteadyStateConfig()
    return Config(
        geometry=GeometryConfig(**raw["geometry"]),
        fluid=FluidConfig(**raw["fluid"]),
        stirrer=StirrerConfig(**raw["stirrer"]),
        grid=GridConfig(**raw["grid"]),
        time=TimeConfig(**raw["time"]),
        levelset=_build_levelset_config(raw["levelset"]),
        air_core=AirCoreConfig(**raw["air_core"]),
        output=OutputConfig(**raw["output"]),
        steady_state=steady_state,
        physics=PhysicsConfig(**raw["physics"]) if "physics" in raw else PhysicsConfig(),
        wall=WallConfig(**raw["wall"]) if "wall" in raw else WallConfig(),
        source_path=None,
    )


def with_overrides(base: Config, **overrides) -> Config:
    """Return a copy of ``base`` with a few leaf values overridden.

    Supported keys: ``rpm``, ``water_height_m``, ``water_viscosity``,
    ``stirbar_length_m``. Used by the RPM / depth sweep scripts so a single
    baseline config can drive many runs without editing YAML files.
    """
    import copy

    cfg = copy.deepcopy(base)
    if "rpm" in overrides:
        cfg.stirrer.rpm = overrides["rpm"]
    if "water_height_m" in overrides:
        cfg.geometry.water_height_m = overrides["water_height_m"]
    if "water_viscosity" in overrides:
        cfg.fluid.water_viscosity = overrides["water_viscosity"]
    if "stirbar_length_m" in overrides:
        cfg.geometry.stirbar_length_m = overrides["stirbar_length_m"]
    return cfg
