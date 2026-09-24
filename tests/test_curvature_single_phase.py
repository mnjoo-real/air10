"""Gate V4 part 1: the axisymmetric curvature operator, tested alone
(no flow, no pressure solve)."""
import numpy as np
import pytest

from air_vortex.curvature_single_phase import (
    crossing_positions, curvature_at_crossings, curvature_centers, interface_curvature)
from air_vortex.liquid_mask import classify
from air_vortex.reinit_benchmarks import uniform_grid


def _kappa_at_crossings(g, phi):
    geom = classify(phi)
    kr, kz = interface_curvature(g, geom, phi)
    rr, _, rz, _ = crossing_positions(g, geom)
    return (np.concatenate([kr[np.isfinite(kr)], kz[np.isfinite(kz)]]),
            np.concatenate([rr[np.isfinite(kr)], rz[np.isfinite(kz)]]))


def _sphere(n, R=0.006):
    g = uniform_grid(R / n, R=2 * R, Z=4 * R)
    Rm, Zm = np.meshgrid(g.r_c, g.z_c, indexing="ij")
    return g, np.hypot(Rm, Zm - (0.5 * g.z_max + 0.37 * g.dz)) - R


@pytest.mark.parametrize("z0", [0.0113, 0.01537, 0.0159])
def test_K0_offgrid_plane_has_zero_curvature(z0):
    g = uniform_grid(1e-3)
    k, _ = _kappa_at_crossings(g, g.z_c[None, :] - z0 + 0 * g.r_c[:, None])
    assert np.max(np.abs(k)) < 1e-9


def test_normal_points_liquid_to_air_and_sphere_curvature_positive():
    g, phi = _sphere(12)
    k, _ = _kappa_at_crossings(g, phi)
    assert np.all(k > 0)                      # liquid inside: kappa = +2/R
    np.testing.assert_allclose(np.mean(k), 2 / 0.006, rtol=1e-3)


def test_axis_curvature_exact_for_even_quadratic():
    """phi = z - c - a r^2: central differences with the even mirror ghost
    are exact for this field, so the first column (r = dr/2, nearest the
    axis) must match the analytic div(n), whose axis limit is -4a."""
    g = uniform_grid(1e-3)
    R, Z = np.meshgrid(g.r_c, g.z_c, indexing="ij")
    a = 30.0
    kc = curvature_centers(g, Z - 0.01 - a * R**2)
    gr = -2 * a * g.r_c[:, None]
    exact = -2 * a / (1 + gr**2) ** 1.5 + gr / (g.r_c[:, None] * np.sqrt(1 + gr**2))
    # wall column excluded: it uses the linear-extrapolation ghost
    np.testing.assert_allclose(kc[:-1, 1:-1], np.broadcast_to(exact[:-1], kc[:-1, 1:-1].shape),
                               rtol=1e-8, atol=1e-8)
    assert kc[0, 10] == pytest.approx(-4 * a, rel=2e-3)


def test_K2_vertical_cylinder_isolates_hoop_term():
    for n in (8, 16, 24):
        g = uniform_grid(0.006 / n, R=0.012, Z=0.024)
        R, _ = np.meshgrid(g.r_c, g.z_c, indexing="ij")
        k, _ = _kappa_at_crossings(g, R - 0.006)
        assert np.max(np.abs(k * 0.006 - 1)) < 0.4 / n**2 + 1e-12


def test_K1_sphere_accuracy_and_convergence():
    rms, mx, axis = [], [], []
    for n in (6, 8, 12, 16, 24):
        g, phi = _sphere(n)
        k, r = _kappa_at_crossings(g, phi)
        e = k * 0.006 / 2 - 1
        rms.append(np.sqrt(np.mean(e**2)))
        mx.append(np.max(np.abs(e)))
        axis.append(np.max(np.abs(e[r < g.dr])))
    assert all(np.diff(rms) < 0) and all(np.diff(mx) < 0)
    order = np.polyfit(np.log([1 / 6, 1 / 8, 1 / 12, 1 / 16, 1 / 24]), np.log(rms), 1)[0]
    assert order > 1.7
    assert rms[2] < 5e-3 and mx[2] < 5e-3          # R/dx = 12
    assert axis[2] < 5e-3                          # pole (first column) error at R/dx = 12


def test_crossing_interpolation_uses_pressure_bc_theta():
    """A field linear along each grid line is reproduced exactly at the
    crossing: kappa_Gamma = (1 - theta) kappa_P + theta kappa_Q."""
    g, phi = _sphere(8)
    geom = classify(phi)
    kr, kz = curvature_at_crossings(g, geom, g.z_c[None, :] + 2 * g.r_c[:, None])
    rr, zr, rz, zz = crossing_positions(g, geom)
    m = np.isfinite(kr)
    np.testing.assert_allclose(kr[m], zr[m] + 2 * rr[m], atol=1e-15)
    m = np.isfinite(kz)
    np.testing.assert_allclose(kz[m], zz[m] + 2 * rz[m], atol=1e-15)
