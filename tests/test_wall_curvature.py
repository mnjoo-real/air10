"""Wall-near graph curvature (wall_curvature.py, V4b-P redesign)."""
import copy

import numpy as np
import pytest

from air_vortex.config import WallConfig
from air_vortex.pinned_phase import build_phase_solver
from air_vortex.wall_curvature import column_heights, graph_kappa_at

XI = (0.05, 0.35, 0.65, 0.95)


def _graph_cfg(wall, order=3, window=None):
    w = copy.deepcopy(wall)
    w.wall_curvature, w.wall_curvature_order = "graph", order
    w.wall_curvature_window = order + 1 if window is None else window
    return w


def _case(theta, xi, dx=0.25e-3):
    s, ref = build_phase_solver(xi, dx, "reconstruct_ghost", "cubic", theta_ref=theta, ic="extended")
    return s.grid, s.fields.phi, s.cfg.wall, ref


def test_config_default_is_the_bulk_level_set_path():
    assert WallConfig().wall_curvature == "level_set"
    with pytest.raises(ValueError):
        WallConfig("pinned", None, 0.01, wall_curvature="nope")
    with pytest.raises(ValueError):                    # window must reach order + 1 points
        WallConfig("pinned", None, 0.01, wall_curvature="graph", wall_curvature_order=3,
                   wall_curvature_window=3)


def test_flat_interface_has_zero_graph_curvature():
    g, phi, wall, _ = _case(90.0, 0.35)
    cols = np.arange(g.Nr - 10, g.Nr)
    km, kt = graph_kappa_at(g, _graph_cfg(wall), column_heights(g, phi, cols), cols, g.r_c[cols])
    assert np.abs(km).max() < 0.1 and np.abs(kt).max() < 0.01          # 1/m


def test_column_heights_are_phase_independent_and_accurate():
    """Cubic root along each column: O(dx^4) heights on the pinned meniscus."""
    for xi in XI:
        g, phi, _, ref = _case(60.0, xi)
        cols = np.arange(g.Nr - 8, g.Nr)
        err = column_heights(g, phi, cols) - ref.eta(g.r_c[cols])
        assert np.abs(err).max() < 2e-3 * g.dz


def test_graph_curvature_principal_components_on_meniscus_all_phases():
    """kappa_m, kappa_theta and the total, separately, at the last 8 column
    centres of the theta = 60 pinned meniscus (exact: Young-Laplace BVP)."""
    worst = []
    for xi in XI:
        g, phi, wall, ref = _case(60.0, xi)
        cols = np.arange(g.Nr - 12, g.Nr)
        r = g.r_c[-8:]
        km, kt = graph_kappa_at(g, _graph_cfg(wall), column_heights(g, phi, cols), cols, r)
        psi = np.interp(r, ref.r, ref.psi)
        kt_ex = -np.sin(psi) / r
        km_ex = ref.kappa_at_height(ref.eta(r)) - kt_ex
        worst.append((np.abs(km - km_ex).max(), np.abs(kt - kt_ex).max()))
    w = np.array(worst)
    assert w[:, 0].max() < 3.0 and w[:, 1].max() < 0.05             # 1/m (|kappa| ~ 235 at the wall)
    assert w[:, 0].std() < 0.5                                       # phase independence


def test_interpolating_stencil_is_restoring_and_lsq_window_is_not():
    """Raising one column must raise its own curvature (restoring capillary
    pressure). A least-squares window (Savitzky-Golay) answers a grid-scale
    sawtooth with the wrong sign."""
    g, phi, wall, _ = _case(90.0, 0.35)
    cols = np.arange(g.Nr - 14, g.Nr)
    h = column_heights(g, phi, cols)
    saw = 1e-6 * (-1.0) ** np.arange(len(cols))
    r = g.r_c[cols[4:-4]]
    for order, window, restoring in ((3, 4, True), (2, 3, True), (2, 5, False)):
        w = _graph_cfg(wall, order, window)
        k0 = sum(graph_kappa_at(g, w, h, cols, r))
        k1 = sum(graph_kappa_at(g, w, h + saw, cols, r))
        response = (k1 - k0) * np.sign(saw[4:-4])
        assert (np.all(response > 0) if restoring else np.all(response < 0))
