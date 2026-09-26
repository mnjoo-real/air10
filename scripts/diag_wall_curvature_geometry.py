"""Geometry-only accuracy of wall-near curvature (V4b-P redesign, TABLE A).

For every interface crossing used by the pressure BC in the last EVAL columns,
compare kappa_m, kappa_theta and the total against the exact reference, for
  LS                  the bulk path: div(n) components at cell centres, interpolated
                      to the crossing (curvature_single_phase)
  G<fit><order>[SA]   wall_curvature.graph_kappa_at (local/global, order 2/3),
                      pinned constraint, or static-angle constraint with SA
Geometries: pinned Young-Laplace menisci theta = 50..90 (wall-compatible IC) and
a zero-gravity spherical cap (theta = 60 at the wall; kappa_m = kappa_theta = -1/Rs).
Errors in 1/m and relative to max|kappa_ref| over the profile.

Usage: python scripts/diag_wall_curvature_geometry.py DX_MM > out.jsonl
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from air_vortex.config import WallConfig  # noqa: E402
from air_vortex.curvature_single_phase import (crossing_positions, curvature_at_crossings,  # noqa: E402
                                               curvature_components_centers)
from air_vortex.liquid_mask import classify  # noqa: E402
from air_vortex.pinned_phase import R_V, build_phase_solver  # noqa: E402
from air_vortex.wall_curvature import column_heights, graph_kappa_at  # noqa: E402

EVAL = 8
XI = (0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95)
METHODS = ["LS", "Glocal2", "Glocal3", "Glocal2SA", "Glocal3SA", "Glocal2W5"]


def cap_case(xi, dx):
    """Zero-gravity spherical cap meeting the wall at 60 deg, pin at phase xi."""
    s, _ = build_phase_solver(xi, dx, "reconstruct_ghost", "cubic", theta_ref=60.0, ic="extended")
    g = s.grid
    th = np.deg2rad(60.0)
    Rs = R_V / np.cos(th)
    z_pin = s.cfg.wall.pinned_contact_height_m
    zb = z_pin - (Rs - np.sqrt(Rs**2 - R_V**2))
    eta = lambda r: zb + Rs - np.sqrt(Rs**2 - np.minimum(np.asarray(r) ** 2, Rs**2 * 0.999))
    rs = np.linspace(-R_V - 4 * g.dr, R_V + 4 * g.dr, 40001)
    zs = eta(np.abs(rs))
    phi = np.empty(g.shape_center)
    for i in range(g.Nr):
        d2 = (g.r_c[i] - rs[None, :]) ** 2 + (g.z_c[:, None] - zs[None, :]) ** 2
        phi[i] = np.sign(g.z_c - eta(g.r_c[i])) * np.sqrt(d2.min(axis=1))
    ex = lambda r: (np.full_like(r, -1 / Rs), np.full_like(r, -1 / Rs))
    return g, phi, s.cfg.wall, ex, 1 / Rs, 60.0


def meniscus_case(theta, xi, dx):
    s, ref = build_phase_solver(xi, dx, "reconstruct_ghost", "cubic", theta_ref=theta, ic="extended")
    g = s.grid

    def ex(r):
        psi = np.interp(r, ref.r, ref.psi)
        kt = -np.sin(psi) / r
        return ref.kappa_at_height(ref.eta(r)) - kt, kt
    scale = max(np.abs(ref.kappa_at_height(ref.z)).max(), 1.0 / R_V)
    return g, s.fields.phi, s.cfg.wall, ex, scale, float(ref.theta_deg)


def evaluate(g, phi, wall, ex, scale, theta_eq, method):
    geom = classify(phi)
    rr, _, rz, _ = crossing_positions(g, geom)
    rmin = g.r_f[g.Nr - EVAL]
    if method == "LS":
        km_c, kt_c = curvature_components_centers(g, phi, wall)
        kmr, kmz = curvature_at_crossings(g, geom, km_c)
        ktr, ktz = curvature_at_crossings(g, geom, kt_c)
        r = np.r_[rz[np.isfinite(kmz) & (rz > rmin)], rr[np.isfinite(kmr) & (rr > rmin)]]
        km = np.r_[kmz[np.isfinite(kmz) & (rz > rmin)], kmr[np.isfinite(kmr) & (rr > rmin)]]
        kt = np.r_[ktz[np.isfinite(kmz) & (rz > rmin)], ktr[np.isfinite(kmr) & (rr > rmin)]]
    else:
        fit = "local" if "local" in method else "global"
        base = method.split("W")[0] if "W5" in method else method
        order = int(base.rstrip("SA")[-1])
        w = copy.deepcopy(wall)
        if method.endswith("SA"):
            w = WallConfig("static_angle", theta_eq)
        w.wall_curvature, w.wall_curvature_fit, w.wall_curvature_order = "graph", fit, order
        w.wall_curvature_band = EVAL
        w.wall_curvature_window = 5 if method.endswith("W5") else order + 1
        cols = np.arange(g.Nr - EVAL - w.wall_curvature_window, g.Nr)
        h = column_heights(g, phi, cols)
        mz = np.isfinite(rz) & (rz > rmin)
        mr = np.isfinite(rr) & (rr > rmin)
        r = np.r_[rz[mz], rr[mr]]
        # keep only real crossings (mixed faces)
        from air_vortex.liquid_mask import FACE_LIQ_MINUS, FACE_LIQ_PLUS
        mixz = (geom.face_kind_z == FACE_LIQ_MINUS) | (geom.face_kind_z == FACE_LIQ_PLUS)
        mixr = (geom.face_kind_r == FACE_LIQ_MINUS) | (geom.face_kind_r == FACE_LIQ_PLUS)
        r = np.r_[rz[mz & mixz], rr[mr & mixr]]
        km, kt = graph_kappa_at(g, w, h, cols, r)
    kme, kte = ex(r)
    e = lambda a, b: float(np.nanmax(np.abs(a - b)))
    return {"n": int(len(r)), "km_err": e(km, kme), "kt_err": e(kt, kte), "k_err": e(km + kt, kme + kte),
            "k_err_rel": e(km + kt, kme + kte) / scale, "km_err_rel": e(km, kme) / scale,
            "kt_err_rel": e(kt, kte) / scale}


def main(dx_mm):
    dx = dx_mm * 1e-3
    for geo in ["cap60", 50, 60, 70, 80, 90]:
        for xi in XI:
            case = cap_case(xi, dx) if geo == "cap60" else meniscus_case(geo, xi, dx)
            for m in METHODS:
                row = {"geometry": geo, "xi": xi, "dx_mm": dx_mm, "method": m, **evaluate(*case, m)}
                print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 0.25)
