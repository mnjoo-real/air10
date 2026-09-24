"""Gate V4 part 2: the sharp capillary pressure jump. CAP-A (exact
curvature) isolates the pressure BC; CAP-B (numerical curvature) measures
the spurious-current floor."""
import numpy as np
import pytest

import air_vortex.free_surface_bc as fsb
from air_vortex.capillary_benchmarks import SIGMA_WATER, build_sphere_solver
from air_vortex.liquid_mask import liquid_volume_subcell
from air_vortex.single_phase_solver import build_single_phase_solver
from _config_helpers import single_phase_config

RS = 0.006
DP = 2 * SIGMA_WATER / RS


def _run(s, n_steps=None, t_end=None):
    umax = []
    while (n_steps is not None and len(umax) < n_steps) or (t_end is not None and s.fields.t < t_end):
        d = s.step()
        umax.append(max(d.max_abs_ur_liquid, d.max_abs_uz_liquid))
    return d, np.array(umax)


def test_pressure_jump_sign_is_fixed_by_cap_a():
    """Liquid sphere (phi<0 inside, n outward, kappa = +2/R) must sit at
    p_atm + 2 sigma/R: S_KAPPA = +1, and the opposite sign demonstrably
    gives the wrong (negative) Laplace pressure."""
    assert fsb.S_KAPPA == 1.0
    s, _ = build_sphere_solver(RS / 8, RS, curvature="analytic")
    _run(s, n_steps=2)
    liq = s.fields.phi < 0
    assert s.fields.p[liq].mean() == pytest.approx(DP, rel=1e-12)
    try:
        fsb.S_KAPPA = -1.0
        s, _ = build_sphere_solver(RS / 8, RS, curvature="analytic")
        _run(s, n_steps=2)
        assert s.fields.p[s.fields.phi < 0].mean() == pytest.approx(-DP, rel=1e-12)
    finally:
        fsb.S_KAPPA = 1.0


@pytest.mark.parametrize("n", [8, 12, 24])
def test_cap_a_exact_curvature_static_to_roundoff(n):
    s, _ = build_sphere_solver(RS / n, RS, curvature="analytic")
    d, u = _run(s, n_steps=100)
    p = s.fields.p[s.fields.phi < 0]
    assert abs(p.mean() / DP - 1) < 1e-12
    assert np.ptp(p) < 1e-10 * DP
    assert u.max() < 1e-13
    assert d.max_divergence_liquid < 1e-10


def test_cap_b_numerical_curvature_short_time():
    s, _ = build_sphere_solver(RS / 12, RS)
    V0 = liquid_volume_subcell(s.grid, s.fields.phi)
    d, u = _run(s, t_end=0.1)
    p = s.fields.p[s.fields.phi < 0]
    assert abs(p.mean() / DP - 1) < 0.02
    assert u.max() < 1e-3
    assert abs(liquid_volume_subcell(s.grid, s.fields.phi) / V0 - 1) < 1e-3


def test_cap_b_long_time_bounded_with_capillary_extension_band():
    """Default extension_layers_capillary = 6: bounded over ~10 capillary
    times with reinit OFF. Negative control: the sigma=0 width (3 layers)
    lets the band-edge |grad phi| kink reach the curvature stencil and the
    spurious current grows (measured 5e-1 m/s by t = 1.25 s)."""
    s, _ = build_sphere_solver(RS / 8, RS)
    assert s.cfg.physics.extension_layers_capillary == 6
    _, u = _run(s, t_end=1.25)
    assert u[len(u) // 2:].max() < 2e-3
    s3, _ = build_sphere_solver(RS / 8, RS)
    s3.cfg.physics.extension_layers_capillary = 3
    s3.cfg.physics.allow_unsafe_extension_for_diagnostics = True
    _, u3 = _run(s3, t_end=1.25)
    assert u3.max() > 10 * u.max()


def test_capillary_timestep_limit_binds():
    s, _ = build_sphere_solver(RS / 12, RS)
    dx = s.grid.dr
    expected = np.sqrt(998.0 * dx**3 / (4 * np.pi * SIGMA_WATER))
    assert s.stable_timestep(s.fields.u_r, s.fields.u_z) == pytest.approx(expected, rel=1e-12)
    s.cfg.physics.capillary_dt_factor = 0.25
    assert s.stable_timestep(s.fields.u_r, s.fields.u_z) == pytest.approx(0.25 * expected, rel=1e-12)


def test_flat_hydrostatic_with_surface_tension_on():
    """kappa = 0 on a flat off-grid surface: sigma > 0 must not change the
    hydrostatic pressure (p_Gamma = p_atm exactly)."""
    cfg = single_phase_config()
    cfg.fluid.surface_tension = SIGMA_WATER
    s = build_single_phase_solver(cfg)
    d, u = _run(s, n_steps=100)
    g, H = s.grid, cfg.geometry.water_height_m
    exact = 998.0 * 9.81 * (H - g.z_c[None, :]) + 0 * g.r_c[:, None]
    liq = s.fields.phi < 0
    assert np.max(np.abs(s.fields.p - exact)[liq]) < 1e-9
    assert u.max() < 1e-13
