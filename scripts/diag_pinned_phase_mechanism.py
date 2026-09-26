"""DIAGNOSTIC ONLY (Gate V4b-P, xi = 0.25 / dx = 0.25 mm anomaly).

Mechanism isolation for the pinned contact line; nothing here is used in
production. Exact/BVP injections go through the solver's validation hook
(curvature_fn) or a runtime patch of contact_angle.wall_ghost_column in this
process only.

Modes (one JSON line per record on stdout):
  ts     VARIANT THETA XI DX_MM T_END [LOG_DT]
         U(t) time series with Umax location and interface error at the
         axis / wall.  VARIANT:
           A   production reconstruct_ghost (cubic, 4 columns)
           B   A + exact BVP curvature at the wall-adjacent crossings
               (column N-1 vertical crossings, face N-2|N-1 radial crossings)
           B4  A + exact BVP curvature at all crossings of the 4 fit columns
           C   A + exact BVP ghost column (static signed distance to the
               extended BVP profile), numerical curvature
           BC  C + B (exact ghost and exact wall curvature)
           X   A with a wall-compatible initial phi: signed distance to the
               BVP profile CONTINUED beyond r = R (extended_profile), instead
               of the profile truncated at the wall (whose level sets wrap
               around the endpoint (R, z_pin) with radius O(dx))
           XC  X + exact ghost;  XB  X + exact wall-column curvature
           K<d..> exact BVP curvature only in columns N-d (d = 1 is the wall
               column), e.g. K2, K12, K123 (K1 = B, K1234 = B4)
  early  THETA XI DX_MM      step 0/1/5/10/20 wall/axis dumps
  static [VARIANT] THETA DX_MM XI...   t = 0 geometry/stencil scan (no stepping)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

import air_vortex.contact_angle as ca  # noqa: E402
from air_vortex.contact_angle import initial_pin_height, reconstruct_pinned, _column_crossing  # noqa: E402
from air_vortex.curvature_single_phase import (crossing_positions, curvature_at_crossings,  # noqa: E402
                                               curvature_components_centers, interface_curvature)
from air_vortex.diagnostics import free_surface_height  # noqa: E402
from air_vortex.liquid_mask import classify  # noqa: E402
from air_vortex.meniscus import G, RHO, SIGMA  # noqa: E402
from air_vortex.pinned_phase import R_V, build_phase_solver, extended_profile, phase_of  # noqa: E402

METHOD, FIT, NCOLS = "reconstruct_ghost", "cubic", 4


def kappa_exact_r(ref, r):
    """BVP curvature at radius r (Young-Laplace at the BVP height eta(r))."""
    return ref.kappa_at_height(ref.eta(np.clip(r, 0.0, R_V)))


def exact_ghost(g, ref):
    rg = R_V + 0.5 * g.dr
    rs = np.linspace(-R_V - 3 * g.dr, R_V + 3 * g.dr, 40001)
    zs = extended_profile(ref, np.abs(rs))
    d2 = (rg - rs[None, :]) ** 2 + (g.z_c[:, None] - zs[None, :]) ** 2
    return np.sign(g.z_c - extended_profile(ref, rg)) * np.sqrt(d2.min(axis=1))


def extended_sdf(g, ref):
    rs = np.linspace(-g.r_v - 4 * g.dr, g.r_v + 4 * g.dr, 40001)
    zs = extended_profile(ref, np.abs(rs))
    out = np.empty(g.shape_center)
    for i in range(g.Nr):
        d2 = (g.r_c[i] - rs[None, :]) ** 2 + (g.z_c[:, None] - zs[None, :]) ** 2
        out[i] = np.sign(g.z_c - ref.eta(g.r_c[i])) * np.sqrt(d2.min(axis=1))
    return out


def make_case(theta, xi, dx_mm, variant="A"):
    dx = dx_mm * 1e-3
    s, ref = build_phase_solver(xi, dx, METHOD, FIT, theta_ref=theta)
    s.cfg.wall.pinned_fit_columns = NCOLS
    s.cfg.wall.pinned_skip_wall_column = False
    g = s.grid
    if variant.startswith("X"):
        s.fields.phi = extended_sdf(g, ref)
        variant = variant[1:] or "A"
    if variant in ("C", "BC"):
        ghost_ex = exact_ghost(g, ref)
        orig = ca.wall_ghost_column

        def patched(grid, phi, wall_cfg=None):
            if wall_cfg is not None and wall_cfg.contact_model == "pinned":
                return ghost_ex
            return orig(grid, phi, wall_cfg)
        ca.wall_ghost_column = patched
    if variant in ("B", "B4", "BC") or variant.startswith("K"):
        if variant.startswith("K"):          # K<offsets>: 1 = column N-1, 2 = N-2, ...
            cols = [grid_i for grid_i in (g.Nr - int(c) for c in variant[1:])]
        else:
            cols = list(range(g.Nr - (1 if variant in ("B", "BC") else NCOLS), g.Nr))

        def curv(grid, geom, phi):
            kr, kz = interface_curvature(grid, geom, phi, s.cfg.wall)
            rr, _, rz, _ = crossing_positions(grid, geom)
            for i in cols:
                m = np.isfinite(kz[i, :])
                kz[i, m] = kappa_exact_r(ref, rz[i, m])
                m = np.isfinite(kr[i, :])      # face (i-1)|i
                kr[i, m] = kappa_exact_r(ref, rr[i, m])
            return kr, kz
        s.curvature_fn = curv
    return s, ref


def umax_loc(s, geom):
    g, f = s.grid, s.fields
    ur = np.where(geom.ur_face_known(), np.abs(f.u_r), 0.0)
    uz = np.where(geom.uz_face_known(), np.abs(f.u_z), 0.0)
    if ur.max() >= uz.max():
        i, j = np.unravel_index(ur.argmax(), ur.shape)
        return float(ur.max()), "ur", int(i), int(j), float(g.r_f[i]), float(g.z_c[j])
    i, j = np.unravel_index(uz.argmax(), uz.shape)
    return float(uz.max()), "uz", int(i), int(j), float(g.r_c[i]), float(g.z_f[j])


def run_ts(variant, theta, xi, dx_mm, t_end, log_dt=0.01):
    s, ref = make_case(theta, xi, dx_mm, variant)
    g = s.grid
    eta0 = ref.eta(g.r_c)
    head = {"mode": "ts", "variant": variant, "theta": theta, "xi": xi, "dx_mm": dx_mm,
            "xi_wall": phase_of(s.cfg.wall.pinned_contact_height_m, g.dz),
            "xi_axis": phase_of(float(ref.eta(0.0)), g.dz), "theta_eq": ref.theta_deg,
            "Nr": g.Nr, "Nz": g.Nz}
    print(json.dumps(head), flush=True)
    next_t, win = log_dt, 0.0
    while s.fields.t < t_end - 1e-12:
        d = s.step()
        u = max(d.max_abs_ur_liquid, d.max_abs_uz_liquid)
        win = max(win, u)
        if not np.isfinite(u) or u > 5.0:
            print(json.dumps({"t": d.t, "blowup": True}), flush=True)
            return
        if d.t >= next_t - 1e-12:
            e = (free_surface_height(s.fields.phi, g) - eta0) / g.dr
            um, comp, i, j, r, z = umax_loc(s, s.geometry)
            print(json.dumps({"t": d.t, "step": s.fields.step, "U": u, "Uwin": win, "loc": [comp, i, j, r, z],
                              "e_axis": float(e[0]), "e_mid": float(e[g.Nr // 2]), "e_wall": float(e[-1]),
                              "e_rms": float(np.sqrt(np.nanmean(e**2))),
                              "ce_dx": (initial_pin_height(g, s.fields.phi, NCOLS)
                                        - s.cfg.wall.pinned_contact_height_m) / g.dz}), flush=True)
            next_t += log_dt
            win = 0.0


def wall_snapshot(s, ref, u_prev=None, dt=None):
    g, f, w = s.grid, s.fields, s.cfg.wall
    phi = f.phi
    geom = classify(phi)
    z_pin = w.pinned_contact_height_m
    idx = list(range(g.Nr - NCOLS, g.Nr))
    rec = reconstruct_pinned(g, phi, z_pin, FIT, NCOLS)
    x = (rec.r_samples - g.r_v) / g.dr
    A = np.vstack([x ** (k + 1) for k in range(3)]).T
    jcross = []
    for i in idx:
        col = phi[i, :]
        jj = [j for j in range(g.Nz - 1, 0, -1) if col[j] >= 0 and col[j - 1] < 0]
        jcross.append(jj[0] if jj else -1)
    ghost = ca.wall_ghost_column(g, phi, w)
    j0 = int(np.searchsorted(g.z_c, z_pin))
    rows = list(range(max(j0 - 4, 0), min(j0 + 4, g.Nz)))
    km_c, kt_c = curvature_components_centers(g, phi, w)
    pr, pz, kr, kz = s.interface_pressure_bc(geom, phi)
    rr, zr, rz, zz = crossing_positions(g, geom)
    wall_x = []
    for kind, K, P, R, Z, i in (("z", kz, pz, rz, zz, g.Nr - 1), ("r", kr, pr, rr, zr, g.Nr - 1)):
        m = np.nonzero(np.isfinite(K[i, :]))[0]
        for j in m:
            wall_x.append({"face": kind, "i": int(i), "j": int(j), "r": float(R[i, j]), "z": float(Z[i, j]),
                           "kappa": float(K[i, j]), "kappa_ex": float(kappa_exact_r(ref, R[i, j])),
                           "p_gamma": float(P[i, j]),
                           "p_hyd": float(ref.P0 - RHO * G * Z[i, j]),
                           "p_err": float(P[i, j] - (ref.P0 - RHO * G * Z[i, j]))})
    # radial crossings on the face N-2|N-1 are stored at kr[Nr-1, :]
    # (u_r face index Nr-1). all crossings: p error profile vs r
    pg = np.r_[pr[np.isfinite(pr)], pz[np.isfinite(pz)]]
    rx = np.r_[rr[np.isfinite(pr)], rz[np.isfinite(pz)]]
    zx = np.r_[zr[np.isfinite(pr)], zz[np.isfinite(pz)]]
    perr = pg - (ref.P0 - RHO * G * zx)
    k = int(np.argmax(np.abs(perr)))
    out = {
        "step": f.step, "t": f.t,
        "sample_cols": idx, "sample_jcross": jcross,
        "z_samples_dx": ((rec.z_samples - z_pin) / g.dz).tolist(),
        "coeffs": rec.a.tolist(), "cond": float(np.linalg.cond(A)),
        "wall_slope": rec.wall_slope, "wall_slope_ex": float(np.tan(ref.psi[-1])),
        "rows": rows, "ghost": ghost[rows].tolist(),
        "ghost_ex": exact_ghost(g, ref)[rows].tolist(),
        "kappa_m_col": km_c[-1, rows].tolist(), "kappa_t_col": kt_c[-1, rows].tolist(),
        "cell_class_wall5": geom.cell_class[-5:, rows].T.tolist(),
        "wall_crossings": wall_x,
        "p_liq_adj": [float(f.p[-1, j]) if geom.liquid[-1, j] else None for j in rows],
        "perr_all_max": float(perr[k]), "perr_all_max_rz": [float(rx[k]), float(zx[k])],
        "perr_rms_over_P0": float(np.sqrt(np.mean(perr**2)) / ref.P0),
        "perr_by_r": _bin_by_r(g, rx, perr),
        "ce_dx": (initial_pin_height(g, phi, NCOLS) - z_pin) / g.dz,
    }
    liq_ur, liq_uz = geom.ur_face_known(), geom.uz_face_known()
    um = umax_loc(s, geom)
    out["umax"] = um
    if u_prev is not None:
        ar = np.where(liq_ur, (f.u_r - u_prev[0]) / dt, 0.0)
        az = np.where(liq_uz, (f.u_z - u_prev[1]) / dt, 0.0)
        if np.abs(ar).max() >= np.abs(az).max():
            i, j = np.unravel_index(np.abs(ar).argmax(), ar.shape)
            out["accel_max"] = [float(ar[i, j]), "ur", int(i), int(j), float(g.r_f[i]), float(g.z_c[j])]
        else:
            i, j = np.unravel_index(np.abs(az).argmax(), az.shape)
            out["accel_max"] = [float(az[i, j]), "uz", int(i), int(j), float(g.r_c[i]), float(g.z_f[j])]
        # radial profile of |accel| (max over z per column) to locate the source
        a_col = np.maximum(np.abs(ar[:-1, :]).max(axis=1), np.abs(az).max(axis=1))
        out["accel_by_col"] = a_col.tolist()
    return out


def _bin_by_r(g, rx, perr):
    col = np.clip(((rx - 0.0) / g.dr).astype(int), 0, g.Nr - 1)
    return [float(np.abs(perr[col == i]).max()) if np.any(col == i) else None for i in range(g.Nr)]


def run_early(theta, xi, dx_mm, steps=(0, 1, 2, 5, 10, 20, 50, 100, 200)):
    s, ref = make_case(theta, xi, dx_mm, "A")
    g = s.grid
    print(json.dumps({"mode": "early", "theta": theta, "xi": xi, "dx_mm": dx_mm,
                      "xi_axis": phase_of(float(ref.eta(0.0)), g.dz), "Nr": g.Nr, "Nz": g.Nz,
                      "P0": ref.P0}), flush=True)
    print(json.dumps(wall_snapshot(s, ref)), flush=True)
    while s.fields.step < max(steps):
        prev = (s.fields.u_r.copy(), s.fields.u_z.copy())
        d = s.step()
        if s.fields.step in steps:
            print(json.dumps(wall_snapshot(s, ref, prev, d.dt)), flush=True)


def run_static(theta, dx_mm, xis, variant="A"):
    for xi in xis:
        s, ref = make_case(theta, xi, dx_mm, variant)
        snap = wall_snapshot(s, ref)
        g = s.grid
        keep = {k: snap[k] for k in ("sample_jcross", "z_samples_dx", "coeffs", "cond", "wall_slope",
                                     "wall_slope_ex", "perr_rms_over_P0", "perr_all_max", "perr_all_max_rz",
                                     "ce_dx")}
        wx = snap["wall_crossings"]
        keep.update({"theta": theta, "dx_mm": dx_mm, "xi": xi,
                     "xi_axis": phase_of(float(ref.eta(0.0)), g.dz),
                     "n_wall_cross": len(wx),
                     "wall_cross": [(c["face"], c["j"], round((c["z"] - s.cfg.wall.pinned_contact_height_m) / g.dz, 3),
                                     round(c["p_err"], 4), round(c["kappa"] - c["kappa_ex"], 3)) for c in wx],
                     "wall_perr_sum": float(sum(c["p_err"] for c in wx)),
                     "cell_class_wall5": snap["cell_class_wall5"]})
        print(json.dumps(keep), flush=True)


def main():
    mode = sys.argv[1]
    a = sys.argv[2:]
    if mode == "ts":
        run_ts(a[0], float(a[1]), float(a[2]), float(a[3]), float(a[4]), float(a[5]) if len(a) > 5 else 0.01)
    elif mode == "early":
        run_early(float(a[0]), float(a[1]), float(a[2]))
    elif mode == "static":
        variant = a[0] if a[0].isalpha() else "A"
        a = a[1:] if a[0].isalpha() else a
        run_static(float(a[0]), float(a[1]), [float(v) for v in a[2:]], variant)
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
