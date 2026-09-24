"""Gate V4b-S: static wall contact geometry, the Young-Laplace meniscus
reference, the pinned contact-line model, and the extension-halo guard."""
import copy

import numpy as np
import pytest

from air_vortex.capillary_benchmarks import build_sphere_solver
from air_vortex.config import WallConfig, config_from_dict, config_to_dict
from air_vortex.contact_angle import contact_point, wall_contact_points, wall_ghost_column, wall_trace
from air_vortex.curvature_single_phase import (crossing_positions, interface_curvature,
                                               principal_resolution)
from air_vortex.liquid_mask import classify
from air_vortex.meniscus import build_meniscus_solver, small_slope_meniscus, solve_meniscus
from air_vortex.reinit_benchmarks import uniform_grid
from air_vortex.velocity_extension import minimum_capillary_extension_layers

R_V = 0.008


def _cone(g, theta, zc=0.0063):
    t = np.deg2rad(theta)
    Rm, Zm = np.meshgrid(g.r_c, g.z_c, indexing="ij")
    return (Zm - zc - (Rm - g.r_v) / np.tan(t)) * np.sin(t)


def _cap(g, theta, z_wall=0.0063):
    t = np.deg2rad(theta)
    Rm, Zm = np.meshgrid(g.r_c, g.z_c, indexing="ij")
    Rs = g.r_v / abs(np.cos(t))
    h = np.sqrt(Rs**2 - g.r_v**2)
    if theta < 90:
        return Rs - np.hypot(Rm, Zm - (z_wall + h)), -2 / Rs
    return np.hypot(Rm, Zm - (z_wall - h)) - Rs, 2 / Rs


def _kappa(g, phi, wc):
    geom = classify(phi)
    kr, kz = interface_curvature(g, geom, phi, wc)
    rr, _, rz, _ = crossing_positions(g, geom)
    return np.r_[kr[np.isfinite(kr)], kz[np.isfinite(kz)]], np.r_[rr[np.isfinite(kr)], rz[np.isfinite(kz)]]


# ------------------------------------------------------------- convention
@pytest.mark.parametrize("theta", [45.0, 60.0, 90.0, 120.0, 135.0])
def test_convention_n_dot_wall_normal_is_minus_cos_theta(theta):
    """theta through the liquid: eta'(R) = cot(theta) and n.e_r = -cos(theta);
    wetting (theta < 90) rises toward the wall."""
    g = uniform_grid(0.25e-3, R=R_V, Z=0.012)
    phi = _cone(g, theta)
    i, j = g.Nr - 2, g.Nz // 2
    gr = (phi[i + 1, j] - phi[i - 1, j]) / (2 * g.dr)
    gz = (phi[i, j + 1] - phi[i, j - 1]) / (2 * g.dz)
    assert gr / np.hypot(gr, gz) == pytest.approx(-np.cos(np.deg2rad(theta)), abs=1e-12)
    z_cl, th_meas = contact_point(g, phi)
    assert th_meas == pytest.approx(theta, abs=1e-9)
    eta_axis_side = g.z_c[np.argmin(np.abs(phi[g.Nr // 2, :]))]
    if theta < 90:
        assert z_cl > eta_axis_side           # rises toward the wall
    elif theta > 90:
        assert z_cl < eta_axis_side


# ------------------------------------------------------------------ CA0/CA1
def test_CA0_ninety_degrees_flat_is_exact():
    g = uniform_grid(0.5e-3, R=R_V, Z=0.012)
    phi = g.z_c[None, :] - 0.00631 + 0 * g.r_c[:, None]
    wc = WallConfig("static_angle", 90.0)
    np.testing.assert_allclose(wall_ghost_column(g, phi, wc), phi[-1], atol=1e-15)
    k, _ = _kappa(g, phi, wc)
    assert np.abs(k).max() < 1e-9
    z_cl, th = contact_point(g, phi)
    assert z_cl == pytest.approx(0.00631, abs=1e-12) and th == pytest.approx(90.0, abs=1e-9)


@pytest.mark.parametrize("theta", [45.0, 60.0, 120.0, 135.0])
def test_CA1_cone_curvature_is_hoop_term_and_converges(theta):
    """A straight meridional line is a CONE in axisymmetry: exact kappa is
    n_r/r = -cos(theta)/r (meridional part 0), measured away from the apex."""
    errs = []
    for dx in (0.5e-3, 0.25e-3, 0.125e-3):
        g = uniform_grid(dx, R=R_V, Z=0.012)
        k, r = _kappa(g, _cone(g, theta), WallConfig("static_angle", theta))
        sel = r > R_V / 2
        errs.append(np.abs(k[sel] + np.cos(np.deg2rad(theta)) / r[sel]).max() * R_V)
    assert errs[-1] < 1e-3 and errs[-1] < errs[0]


def test_static_angle_ghost_enforces_theta_at_contact_point():
    """A flat interface with a prescribed 60-degree angle: at z_cl the wall
    slope implied by the ghost equals -cot(theta)|phi_z| (to round-off)."""
    g = uniform_grid(0.5e-3, R=R_V, Z=0.012)
    phi = g.z_c[None, :] - 0.00631 + 0 * g.r_c[:, None]
    wc = WallConfig("static_angle", 60.0)
    ghost = wall_ghost_column(g, phi, wc)
    j = np.argmin(np.abs(g.z_c - 0.00631))
    _, g_w0 = wall_trace(g, phi)
    slope = g_w0[j] + (ghost[j] - (4 * phi[-1, j] - 6 * phi[-2, j] + 4 * phi[-3, j] - phi[-4, j])) / g.dr
    assert slope == pytest.approx(-1.0 / np.tan(np.deg2rad(60.0)), rel=1e-9)


# ------------------------------------------------------------------ CA2 cap
@pytest.mark.parametrize("theta", [60.0, 75.0, 105.0, 120.0])
def test_CA2_spherical_cap_wall_curvature(theta):
    for dx in (0.5e-3, 0.25e-3, 0.125e-3):
        g = uniform_grid(dx, R=R_V, Z=0.012)
        phi, kex = _cap(g, theta)
        for wc in (WallConfig("static_angle", theta), WallConfig("extrapolate")):
            k, r = _kappa(g, phi, wc)
            wall = r > R_V - 3 * dx
            assert np.abs(k[wall] / kex - 1).max() < 3e-4      # was 6-45 % before the fix
            assert np.abs(k[~wall] / kex - 1).max() < 5e-4


def test_linear_wall_ghost_would_drop_meridional_curvature():
    """Documents the defect fixed in V4b: a linear-extrapolation ghost
    forces phi_rr = 0 in the wall column (30-45 % wall curvature error)."""
    from air_vortex import curvature_single_phase as cs
    g = uniform_grid(0.25e-3, R=R_V, Z=0.012)
    phi, kex = _cap(g, 60.0)
    km, kh = cs.curvature_components_centers(g, phi, None)   # None -> linear ghost
    k_lin = cs.curvature_at_crossings(g, classify(phi), km + kh)
    rr, _, rz, _ = crossing_positions(g, classify(phi))
    k = np.r_[k_lin[0][np.isfinite(k_lin[0])], k_lin[1][np.isfinite(k_lin[1])]]
    r = np.r_[rr[np.isfinite(k_lin[0])], rz[np.isfinite(k_lin[1])]]
    assert np.abs(k[r > R_V - 1.5 * g.dr] / kex - 1).max() > 0.1


# ------------------------------------------------------------ BVP reference
def test_meniscus_bvp_reference_properties():
    m60, m120, m90 = (solve_meniscus(t, R_V, 0.006) for t in (60.0, 120.0, 90.0))
    assert m60.residual_max < 1e-8
    r = np.linspace(0, R_V, 2001)
    assert np.trapezoid(2 * np.pi * r * m60.eta(r), r) == pytest.approx(m60.volume, rel=1e-6)
    np.testing.assert_allclose(m60.eta(r) - 0.006, -(m120.eta(r) - 0.006), atol=1e-9)
    assert np.ptp(m90.z) < 1e-12
    assert m60.z_wall > m60.z[0]                                  # wetting rises
    eta_lin, _ = small_slope_meniscus(88.0, R_V, 0.006)
    m88 = solve_meniscus(88.0, R_V, 0.006)
    rise = m88.z_wall - m88.z[0]
    assert np.abs(m88.eta(r) - eta_lin(r)).max() < 1e-3 * rise
    # Young-Laplace holds on the solved profile: -(dpsi/ds + sin(psi)/r) = (P0 - rho g z)/sigma
    s_arc = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(m60.r), np.diff(m60.z)))])
    k_geom = -(np.gradient(m60.psi, s_arc) + np.sin(m60.psi) / np.maximum(m60.r, 1e-12))
    sl = slice(50, -50)
    k_yl = m60.kappa_at_height(m60.z)
    assert np.abs(k_geom[sl] - k_yl[sl]).max() < 1e-3 * np.abs(k_yl).max()


# ------------------------------------------------------- wall models (CFD)
def test_pinned_static_meniscus_holds_contact_point():
    s, ref = build_meniscus_solver(0.5e-3, 60.0, contact_model="pinned")
    z0 = s.cfg.wall.pinned_contact_height_m
    assert z0 == pytest.approx(ref.z_wall, abs=0.02 * s.grid.dr)
    umax = 0.0
    while s.fields.t < 0.1:
        d = s.step()
        umax = max(umax, d.max_abs_ur_liquid, d.max_abs_uz_liquid)
    z_cl = wall_contact_points(s.grid, s.fields.phi)
    assert z_cl.size == 1 and abs(z_cl[0] - z0) < 0.05 * s.grid.dr
    assert umax < 5e-3


def test_extrapolate_wall_is_dynamically_unstable_negative_control():
    """Angle-free (extrapolated) wall ghost with capillarity: round-off on a
    FLAT surface grows by many orders within 0.2 s -- not a usable wall model."""
    s, _ = build_meniscus_solver(0.5e-3, 90.0, contact_model="extrapolate")
    while s.fields.t < 0.2:
        d = s.step()
    assert max(d.max_abs_ur_liquid, d.max_abs_uz_liquid) > 1e-8


def test_flat_ninety_degree_meniscus_stays_at_roundoff_for_constrained_models():
    for model in ("static_angle", "pinned"):
        s, _ = build_meniscus_solver(0.5e-3, 90.0, contact_model=model)
        while s.fields.t < 0.2:
            d = s.step()
        assert max(d.max_abs_ur_liquid, d.max_abs_uz_liquid) < 1e-12


# ----------------------------------------------------------- config, halo
def test_wall_config_has_no_default_angle_and_roundtrips():
    with pytest.raises(ValueError):
        WallConfig("static_angle", None)
    with pytest.raises(ValueError):
        WallConfig("static_angle", 180.0)
    with pytest.raises(ValueError):
        WallConfig("sliding")
    assert WallConfig().contact_angle_deg is None
    s, _ = build_meniscus_solver(0.5e-3, 75.0)
    again = config_from_dict(config_to_dict(s.cfg))
    assert again.wall == s.cfg.wall


def test_extension_halo_minimum_is_enforced():
    assert minimum_capillary_extension_layers() == 5
    s, _ = build_sphere_solver(0.0005, 0.006)
    s.cfg.physics.extension_layers_capillary = 4
    with pytest.raises(ValueError):
        s.step()
    s.cfg.physics.allow_unsafe_extension_for_diagnostics = True
    s.step()


def test_extension_width_6_and_12_equivalent_for_isolated_drop():
    out = []
    for layers in (6, 12):
        s, _ = build_sphere_solver(0.006 / 12, 0.006)
        s.cfg.physics.extension_layers_capillary = layers
        for _ in range(150):
            s.step()
        out.append(s.fields.p[s.fields.phi < 0].mean())
    assert out[0] == pytest.approx(out[1], rel=1e-9)


def test_principal_curvature_resolution_counts():
    g = uniform_grid(0.5e-3, R=0.012, Z=0.024)
    Rm, Zm = np.meshgrid(g.r_c, g.z_c, indexing="ij")
    phi = np.hypot(Rm, Zm - 0.01213) - 0.006
    res = principal_resolution(g, classify(phi), phi)
    for comp in ("meridional", "azimuthal"):
        assert res[comp]["median_cells"] == pytest.approx(12.0, rel=0.02)
    cyl = Rm - 0.006
    res = principal_resolution(g, classify(cyl), cyl)
    assert res["azimuthal"]["median_cells"] == pytest.approx(12.0, rel=0.05)
    assert res["meridional"]["min_cells"] > 1e6          # flat meridian
