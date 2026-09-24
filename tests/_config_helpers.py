"""Small, fast-to-run configs for the solver-level milestone tests."""
from air_vortex.config import (
    AirCoreConfig,
    Config,
    FluidConfig,
    GeometryConfig,
    GridConfig,
    LevelSetConfig,
    OutputConfig,
    StirrerConfig,
    TimeConfig,
)


def small_config(rpm: float = 0.0, dr=0.003, dz=0.003) -> Config:
    geometry = GeometryConfig(
        vessel_radius_m=0.024,
        vessel_height_m=0.040,
        water_height_m=0.018,
        air_height_m=0.012,
        stirbar_length_m=0.012,
        stirbar_diameter_m=0.004,
        stirbar_center_z_m=0.002,
    )
    fluid = FluidConfig(
        water_density=998.0,
        water_viscosity=1.0e-3,
        air_density=1.2,
        air_viscosity=1.8e-5,
        surface_tension=0.072,
        gravity=9.81,
    )
    stirrer = StirrerConfig(
        rpm=rpm,
        ramp_time_s=0.05,
        forcing_tau_s=0.01,
        forcing_smoothing_m=0.001,
    )
    grid = GridConfig(dr_m=dr, dz_m=dz)
    time = TimeConfig(t_end_s=1.0, cfl=0.3, dt_max_s=2.0e-4)
    levelset = LevelSetConfig(interface_width_cells=1.5, reinitialize_every=5,
                               reinitialize_iterations=2)
    air_core = AirCoreConfig(contact_tolerance_cells=2, persistence_rotations=5)
    output = OutputConfig(save_every_s=0.05)

    return Config(geometry=geometry, fluid=fluid, stirrer=stirrer, grid=grid,
                  time=time, levelset=levelset, air_core=air_core, output=output)


def single_phase_config(dx=0.0015, water_height_m=0.0173, reinit_every=0) -> Config:
    """Level-1A (single_phase_ls) validation config: sigma=0, no stirring,
    MUSCL2+SSPRK2 transport, reinitialization OFF by default. The default
    water height 17.3 mm is deliberately off-grid for dx=1.5 mm."""
    from air_vortex.config import PhysicsConfig

    cfg = small_config(rpm=0.0, dr=dx, dz=dx)
    cfg.physics = PhysicsConfig(free_surface_model="single_phase_ls")
    cfg.fluid.surface_tension = 0.0
    cfg.geometry.water_height_m = water_height_m
    cfg.levelset.advection_scheme = "muscl2"
    cfg.levelset.time_integrator = "ssprk2"
    cfg.levelset.reinitialize_every = reinit_every
    return cfg
