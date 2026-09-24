"""Liquid / interface / void classification and sub-cell phi=0 crossings."""
import numpy as np
import pytest

from air_vortex.liquid_mask import (
    CELL_INTERFACE, CELL_LIQUID, CELL_VOID, FACE_INACTIVE, FACE_LIQ_MINUS, FACE_LIQ_PLUS,
    FACE_LIQUID, THETA_MIN, classify, crossing_fraction, liquid_volume_staircase,
    liquid_volume_subcell,
)
from _helpers import make_grid


def test_flat_interface_classification():
    g = make_grid(Nr=4, Nz=6, dr=1.0, dz=1.0)
    phi = g.z_c[None, :] - 2.3 + 0 * g.r_c[:, None]   # liquid rows j=0,1 (z=0.5,1.5)
    geom = classify(phi)
    assert geom.n_liquid == 8
    assert np.all(geom.cell_class[:, 0] == CELL_LIQUID)
    assert np.all(geom.cell_class[:, 1] == CELL_INTERFACE)
    assert np.all(geom.cell_class[:, 2:] == CELL_VOID)
    assert np.all(geom.face_kind_z[:, 1] == FACE_LIQUID)
    assert np.all(geom.face_kind_z[:, 2] == FACE_LIQ_MINUS)
    assert np.all(geom.face_kind_z[:, 3:] == FACE_INACTIVE)
    assert np.all(geom.face_kind_z[:, 0] == FACE_INACTIVE)   # bottom boundary
    # interface at z=2.3 lies 0.8 of the way from z=1.5 to z=2.5 -- not snapped
    np.testing.assert_allclose(geom.theta_z[:, 2], 0.8, rtol=0, atol=1e-14)
    assert np.all(geom.face_kind_r[1:-1, :2] == FACE_LIQUID)
    assert np.all(geom.face_kind_r[0, :] == FACE_INACTIVE)


def test_vertical_interface_both_orientations():
    g = make_grid(Nr=6, Nz=3, dr=1.0, dz=1.0)
    core = 2.2 - g.r_c[:, None] + 0 * g.z_c[None, :]      # void r<2.2, liquid outside
    geom = classify(core)
    assert np.all(geom.face_kind_r[2, :] == FACE_LIQ_PLUS)
    # crossing measured from the LIQUID node r=2.5 toward the void node r=1.5
    np.testing.assert_allclose(geom.theta_r[2, :], 0.3, atol=1e-14)


@pytest.mark.parametrize("zeta", [0.1, 0.25, 0.5, 0.75, 0.9])
def test_subcell_crossing_is_linear_zero(zeta):
    phi_l, phi_v = -zeta, 1.0 - zeta
    assert crossing_fraction(phi_l, phi_v) == pytest.approx(zeta, abs=1e-15)


def test_center_on_interface_is_void_and_theta_is_one():
    g = make_grid(Nr=2, Nz=4, dr=1.0, dz=1.0)
    phi = g.z_c[None, :] - 1.5 + 0 * g.r_c[:, None]   # center j=1 exactly on phi=0
    geom = classify(phi)
    assert not geom.liquid[0, 1]
    np.testing.assert_allclose(geom.theta_z[:, 1], 1.0)


def test_theta_floor_only_below_theta_min():
    g = make_grid(Nr=2, Nz=4, dr=1.0, dz=1.0)
    for eps in (1e-3, 1e-7, 1e-12):
        phi = g.z_c[None, :] - (1.5 + eps) + 0 * g.r_c[:, None]   # liquid node at z=1.5, eps below
        geom = classify(phi)
        raw = geom.theta_z_raw[0, 2]
        assert raw == pytest.approx(eps, rel=1e-6)
        assert geom.theta_z[0, 2] == max(raw, THETA_MIN)
        assert geom.n_floored == (2 if raw < THETA_MIN else 0)


def test_subcell_volume_exact_for_flat_signed_distance():
    g = make_grid(Nr=5, Nz=10, dr=0.002, dz=0.002)
    for H in (0.0071, 0.0100, 0.0133):
        phi = g.z_c[None, :] - H + 0 * g.r_c[:, None]
        exact = np.pi * g.r_v**2 * H
        assert liquid_volume_subcell(g, phi) == pytest.approx(exact, rel=1e-12)
        assert abs(liquid_volume_staircase(g, phi) - exact) / exact > 1e-3 or H == 0.0100
