"""Narrow-band liquid-velocity extension (for Level Set transport only)."""
import numpy as np
import pytest

from air_vortex.liquid_mask import classify
from air_vortex.velocity_extension import extend_field, extend_velocity
from _helpers import make_grid

LAYERS = 3


def _curved_setup():
    g = make_grid(Nr=16, Nz=20, dr=0.0015, dz=0.0015)
    R, Z = np.meshgrid(g.r_c, g.z_c, indexing="ij")
    phi = Z - (0.013 + 0.006 * (R / g.r_v) ** 2)     # paraboloid, off-grid
    return g, phi, classify(phi)


def test_A_zero_velocity_extends_to_exactly_zero():
    g, phi, geom = _curved_setup()
    ur, uz, ut, info = extend_velocity(g, geom, phi, np.zeros(g.shape_ur), np.zeros(g.shape_uz),
                                       np.zeros(g.shape_center), LAYERS)
    assert np.all(ur == 0) and np.all(uz == 0) and np.all(ut == 0)
    assert info.band_uz.any() and info.band_ur.any() and info.band_center.any()


def test_B_constant_velocity_extends_to_exactly_constant():
    g, phi, geom = _curved_setup()
    ur = np.full(g.shape_ur, 0.37)
    ur[0], ur[-1] = 0.0, 0.0
    uz = np.full(g.shape_uz, -0.21)
    uz[:, 0] = 0.0
    ur_e, uz_e, _, info = extend_velocity(g, geom, phi, ur, uz, np.zeros(g.shape_center), LAYERS)
    # a convex combination of equal values: exact up to 1 ulp of round-off
    np.testing.assert_allclose(uz_e[info.band_uz], -0.21, rtol=1e-15, atol=0)
    q = np.full((10, 10), 2.5)
    known = np.zeros((10, 10), bool)
    known[:, :4] = True
    n_r, n_z = np.zeros((10, 10)), np.ones((10, 10))
    q_e, band = extend_field(q, known, n_r, n_z, 1.0, 1.0, LAYERS)
    np.testing.assert_allclose(q_e[band], 2.5, rtol=1e-15, atol=0)
    assert band[:, 4:7].all() and not band[:, 7:].any()
    np.testing.assert_array_equal(q_e[:, 7:], 0.0)   # beyond the band


def test_C_linear_velocity_extension_error_is_measured_and_bounded():
    """Constant-along-normal extension is exact only for fields constant
    along n; for a linear field u = a r + b z the error is O(|grad u . n|
    * distance). Measured and bounded by (layers * h * |grad u|)."""
    g, phi, geom = _curved_setup()
    a, b = 3.0, -2.0
    uz = a * g.r_c[:, None] + b * g.z_f[None, :]
    uz[:, 0] = 0.0
    _, uz_e, _, info = extend_velocity(g, geom, phi, np.zeros(g.shape_ur), uz,
                                       np.zeros(g.shape_center), LAYERS)
    exact = a * g.r_c[:, None] + b * g.z_f[None, :]
    err = np.max(np.abs(uz_e - exact)[info.band_uz])
    bound = (LAYERS + 1) * g.dz * np.hypot(a, b)
    assert 0 < err < bound
    # a field constant along the (vertical) normal of a FLAT interface is exact
    phi_flat = g.z_c[None, :] - 0.0137 + 0 * g.r_c[:, None]
    geom_flat = classify(phi_flat)
    uz_r = a * g.r_c[:, None] + 0 * g.z_f[None, :]
    uz_r[:, 0] = 0.0
    _, uz_e, _, info = extend_velocity(g, geom_flat, phi_flat, np.zeros(g.shape_ur), uz_r,
                                       np.zeros(g.shape_center), LAYERS)
    np.testing.assert_allclose(uz_e[info.band_uz], uz_r[info.band_uz], rtol=0, atol=1e-15)


def test_D_extension_never_alters_liquid_controlled_values():
    rng = np.random.default_rng(1)
    g, phi, geom = _curved_setup()
    ur = rng.normal(size=g.shape_ur)
    uz = rng.normal(size=g.shape_uz)
    ut = rng.normal(size=g.shape_center)
    ur[0], ur[-1], uz[:, 0] = 0, 0, 0
    ur_e, uz_e, ut_e, _ = extend_velocity(g, geom, phi, ur, uz, ut, LAYERS)
    kr, kz = geom.ur_face_known(), geom.uz_face_known()
    np.testing.assert_array_equal(ur_e[kr], ur[kr])
    np.testing.assert_array_equal(uz_e[kz], uz[kz])
    np.testing.assert_array_equal(ut_e[geom.liquid], ut[geom.liquid])


@pytest.mark.parametrize("omega", [5.0, 20.0])
def test_swirl_extension_preserves_solid_body_rotation(omega):
    """Extension of omega = u_theta/r (stress-free swirl) keeps u_theta =
    Omega r exact in the band, on a curved interface."""
    g, phi, geom = _curved_setup()
    ut = omega * g.r_c[:, None] * np.ones(g.shape_center)
    _, _, ut_e, info = extend_velocity(g, geom, phi, np.zeros(g.shape_ur), np.zeros(g.shape_uz),
                                       np.where(geom.liquid, ut, 0.0), LAYERS)
    np.testing.assert_allclose(ut_e[info.band_center], ut[info.band_center], rtol=1e-14)
