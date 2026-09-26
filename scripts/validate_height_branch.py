"""Validation gates H1-H5 of the opt-in single_phase_height research branch.

  geom       H1  geometry: node curvature components, well-balanced radial-crossing
                 curvature, face reconstruction and wall slope vs exact references
                 (paraboloid, zero-gravity spherical cap, pinned Young-Laplace menisci
                 theta 50..90), 10 phases, dx 0.5 / 0.25 / 0.125 mm            -> TABLE A
  transport  H2  conservative column transport with a prescribed discretely
                 divergence-free MAC field (no pressure, no sigma): instantaneous rate
                 K_H vs exact eta_t, and a FINITE-TIME run vs an exact Lagrangian
                 (RK4 particle) reference; discrete volume change            -> TABLE B
  run THETA XI DX_MM T_END  H5 static pinned meniscus, JSON lines every 0.01 s
  rigid DX_MM T_END OMEGA   H4 rigid body sigma = 0, matched rotating wall
  hydro DX_MM T_END         H3 off-grid flat surface, sigma = 0 and sigma > 0
  rot XI DX_MM T_END OMEGA  integrated gravity + capillarity + rotation equilibrium
  force RPM DX_MM T_END     QUALITATIVE bottom-stirrer forcing sanity (uncalibrated)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from air_vortex.height_benchmarks import build_height_flat, build_height_meniscus  # noqa: E402
from air_vortex.height_interface import (GraphInterface, column_areas, curvature_at_crossings,  # noqa: E402
                                         eta_rate, geometry_from_eta)
from air_vortex.meniscus import G, RHO, SIGMA  # noqa: E402
from air_vortex.pinned_phase import R_V, build_phase_solver, depth_for_phase, phase_of  # noqa: E402

XI10 = (0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95)


# ------------------------------------------------------------------ H1
def exact_geometry(kind, theta, xi, dx):
    """(grid, eta_fn, d1_fn, d2_fn, kappa_fn(r) -> (km, kt), z_pin)."""
    s, ref = build_phase_solver(xi, dx, "reconstruct_ghost", "cubic", theta_ref=max(theta, 50.0), ic="extended")
    g = s.grid
    zw = s.cfg.wall.pinned_contact_height_m
    if kind == "meniscus":
        from air_vortex.meniscus import solve_meniscus_pinned
        H, zp = depth_for_phase(xi, dx, theta)
        ref = solve_meniscus_pinned(zp, R_V, H)
        eta = ref.eta
        d1 = lambda r: np.tan(np.interp(r, ref.r, ref.psi))  # noqa: E731

        def kap(r):
            psi = np.interp(r, ref.r, ref.psi)
            kt = -np.sin(psi) / r
            return ref.kappa_at_height(ref.eta(r)) - kt, kt
        return g, eta, d1, kap, zp
    if kind == "paraboloid":
        A = (1.0 / np.tan(np.deg2rad(theta))) / (2 * R_V)
        z0 = zw - A * R_V**2
        eta = lambda r: z0 + A * np.asarray(r) ** 2  # noqa: E731
        d1 = lambda r: 2 * A * np.asarray(r)  # noqa: E731
        return g, eta, d1, lambda r: GraphInterface.kappa_parts(r, d1(r), 2 * A + 0 * r), float(eta(R_V))
    if kind == "cap":
        Rs = R_V / np.cos(np.deg2rad(theta))
        zb = zw - (Rs - np.sqrt(Rs**2 - R_V**2))
        eta = lambda r: zb + Rs - np.sqrt(Rs**2 - np.asarray(r) ** 2)  # noqa: E731
        d1 = lambda r: np.asarray(r) / np.sqrt(Rs**2 - np.asarray(r) ** 2)  # noqa: E731
        return g, eta, d1, lambda r: (-1 / Rs + 0 * r, -1 / Rs + 0 * r), float(eta(R_V))
    raise ValueError(kind)


def geom_row(kind, theta, xi, dx_mm):
    dx = dx_mm * 1e-3
    g, eta, d1, kap, zp = exact_geometry(kind, theta, xi, dx)
    gi = GraphInterface(g, eta(g.r_c), zp)
    d1n, d2n = gi.node_derivs()
    km, kt = GraphInterface.kappa_parts(g.r_c, d1n, d2n)
    kme, kte = kap(g.r_c)
    geom, rc = geometry_from_eta(g, gi)
    kr, _ = curvature_at_crossings(g, gi, geom, rc)
    m = np.isfinite(rc)
    rad_err = float(np.nanmax(np.abs(kr[m] - np.add(*kap(rc[m]))))) if m.any() else 0.0
    face_err = float(np.abs(gi.value(g.r_f[1:g.Nr]) - eta(g.r_f[1:g.Nr])).max() / dx)
    slope_err = float(abs(gi.derivs(np.array([R_V]))[1][0] - d1(R_V)))
    return {"geometry": f"{kind}{theta:g}", "dx_mm": dx_mm, "xi": xi,
            "eta_face_err_dx": face_err, "km_err": float(np.abs(km - kme).max()),
            "kt_err": float(np.abs(kt - kte).max()), "k_err": float(np.abs(km + kt - kme - kte).max()),
            "k_err_radial_crossings": rad_err, "wall_slope_err": slope_err,
            "k_scale": float(np.abs(kme + kte).max())}


def run_geom():
    cases = [("paraboloid", 60.0), ("cap", 60.0)] + [("meniscus", t) for t in (50.0, 60.0, 70.0, 80.0, 90.0)]
    for dx_mm in (0.5, 0.25, 0.125):
        for kind, th in cases:
            for xi in XI10:
                print(json.dumps(geom_row(kind, th, xi, dx_mm)), flush=True)


# ------------------------------------------------------------------ H2
def field_setup(theta, xi, dx):
    g, eta, d1, _, zp = exact_geometry("paraboloid", theta, xi, dx)
    Z = g.z_f[-1]
    fz = lambda z: z * (1 + z / Z)  # noqa: E731
    dfz = lambda z: 1 + 2 * z / Z  # noqa: E731
    psi = (g.r_f**2 * (R_V**2 - g.r_f**2) ** 2)[:, None] * fz(g.z_f)[None, :]
    ur = -(psi[:, 1:] - psi[:, :-1]) / (np.where(g.r_f == 0, 1, g.r_f)[:, None] * g.dz); ur[0] = 0
    uz = (psi[1:, :] - psi[:-1, :]) / (g.r_c[:, None] * g.dr)
    sc = 1e-3 / max(np.abs(ur).max(), np.abs(uz).max())
    ur_a = lambda r, z: -sc * r * (R_V**2 - r**2) ** 2 * dfz(z)  # noqa: E731
    uz_a = lambda r, z: sc * (2 * (R_V**2 - r**2) ** 2 - 4 * r**2 * (R_V**2 - r**2)) * fz(z)  # noqa: E731
    return g, eta, d1, zp, ur * sc, uz * sc, ur_a, uz_a


def lagrangian_reference(eta, ur_a, uz_a, T, r_eval, n=4001):
    """Exact graph at time T: RK4-trace interface particles of the analytic field."""
    r = np.linspace(0.0, R_V, n); z = eta(r)
    steps = 400; h = T / steps
    for _ in range(steps):
        def f(rr, zz):
            return ur_a(rr, zz), uz_a(rr, zz)
        k1 = f(r, z); k2 = f(r + h / 2 * k1[0], z + h / 2 * k1[1])
        k3 = f(r + h / 2 * k2[0], z + h / 2 * k2[1]); k4 = f(r + h * k3[0], z + h * k3[1])
        r = r + h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        z = z + h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
    return np.interp(r_eval, r, z)


def run_transport(T=0.5):
    for dx_mm in (0.5, 0.25, 0.125):
        for theta in (90.0, 75.0, 60.0, 50.0):
            for xi in (0.05, 0.25, 0.45, 0.65, 0.85):
                g, eta, d1, zp, ur, uz, ur_a, uz_a = field_setup(theta, xi, dx_mm * 1e-3)
                e0 = eta(g.r_c)
                A = column_areas(g)
                rate = eta_rate(g, e0, zp, ur)
                exact = uz_a(g.r_c, e0) - ur_a(g.r_c, e0) * d1(g.r_c)
                # finite-time SSPRK2 with the frozen discrete field, fixed dt
                dt = 0.25 * dx_mm * 1e-3 / 1e-3 * 0.5
                nst = int(np.ceil(T / dt)); dt = T / nst
                e = e0.copy()
                for _ in range(nst):
                    e1 = e + dt * eta_rate(g, e, zp, ur)
                    e = 0.5 * e + 0.5 * (e1 + dt * eta_rate(g, e1, zp, ur))
                eref = lagrangian_reference(eta, ur_a, uz_a, T, g.r_c)
                disp = np.abs(eref - e0).max()
                print(json.dumps({"dx_mm": dx_mm, "theta": theta, "xi": xi,
                                  "rate_rel_err": float(np.linalg.norm(rate - exact) / np.linalg.norm(exact)),
                                  "rate_rel_err_last8": float(np.linalg.norm((rate - exact)[-8:]) / np.linalg.norm(exact[-8:])),
                                  "volume_rate_rel": float(abs(np.sum(A * rate)) / np.sum(A * np.abs(rate))),
                                  "T": T, "interface_err_dx": float(np.abs(e - eref).max() / g.dr),
                                  "interface_err_rel_disp": float(np.abs(e - eref).max() / disp),
                                  "volume_change_rel": float(np.sum(A * (e - e0)) / np.sum(A * e0))}), flush=True)


# ------------------------------------------------------------------ H3-H5 time runs
def log_run(s, ref_eta, T, extra=None, every=0.01):
    V0 = s.volume
    nxt = 0.0
    while s.fields.t < T:
        d = s.step()
        U = max(d.max_abs_ur_liquid, d.max_abs_uz_liquid)
        if not np.isfinite(U) or U > 5.0:
            print(json.dumps({"t": d.t, "blowup": True}), flush=True)
            return
        if d.t >= nxt:
            row = {"t": d.t, "U": U, "dV": s.volume / V0 - 1, "div": d.max_divergence_liquid,
                   "max_slope": d.max_slope,
                   "rmse_dx": float(np.sqrt(np.mean((s.eta - ref_eta) ** 2)) / s.grid.dr),
                   "eta_axis_err_dx": float((s.eta[0] - ref_eta[0]) / s.grid.dr)}
            if extra:
                row.update(extra(s))
            print(json.dumps(row), flush=True)
            nxt += every


def run_meniscus(theta, xi, dx_mm, T):
    s, ref = build_height_meniscus(xi, dx_mm * 1e-3, theta)
    print(json.dumps({"mode": "meniscus", "theta": theta, "xi": xi, "dx_mm": dx_mm,
                      "xi_actual": phase_of(s.z_pin, s.grid.dz), "Nr": s.grid.Nr}), flush=True)
    log_run(s, ref.eta(s.grid.r_c), T)


def run_hydro(dx_mm, T):
    for sigma in (False, True):
        s = build_height_flat(0.00617, dx_mm * 1e-3, sigma=sigma)
        print(json.dumps({"mode": "hydro", "sigma": sigma, "dx_mm": dx_mm}), flush=True)
        log_run(s, np.full(s.grid.Nr, 0.00617), T, every=0.05)


def run_rigid(dx_mm, T, omega):
    from air_vortex.config import WallConfig
    from air_vortex.diagnostics import volume_consistent_parabola_constant
    from air_vortex.fields import allocate_fields
    from air_vortex.grid import build_grid
    from air_vortex.height_solver import SinglePhaseHeightSolver
    from air_vortex.meniscus import meniscus_config
    from air_vortex.single_phase_solver import WallBC
    dx = dx_mm * 1e-3
    H = 0.006
    cfg = meniscus_config(dx, None, H=H, contact_model="pinned")
    cfg.physics.free_surface_model = "single_phase_height"
    cfg.fluid.surface_tension = 0.0
    g = build_grid(cfg)
    V = np.pi * R_V**2 * H
    C = volume_consistent_parabola_constant(R_V, omega, G, V)
    eta_fn = lambda r: C + omega**2 * np.asarray(r) ** 2 / (2 * G)  # noqa: E731
    cfg.wall = WallConfig("pinned", None, float(eta_fn(R_V)))
    f = allocate_fields(g)
    r2, z2 = g.r_c[:, None] * np.ones(g.shape_center), g.z_c[None, :] * np.ones(g.shape_center)
    eta = eta_fn(g.r_c)
    liq = z2 < eta[:, None]
    f.u_theta = np.where(liq, omega * r2, 0.0)
    f.p = np.where(liq, RHO * (0.5 * omega**2 * r2**2 - G * (z2 - C)), 0.0)
    s = SinglePhaseHeightSolver(grid=g, cfg=cfg, fields=f, eta=eta, z_pin=float(eta_fn(R_V)),
                                wall_bc=WallBC.rotating(omega))
    depth = float(eta_fn(R_V) - eta_fn(0.0))
    print(json.dumps({"mode": "rigid", "dx_mm": dx_mm, "omega": omega, "depth_mm": depth * 1e3}), flush=True)
    log_run(s, eta, T, extra=lambda ss: {"eta_nrmse": float(np.sqrt(np.mean((ss.eta - eta) ** 2)) / depth),
                                         "center_err_dx": float((ss.eta[0] - eta[0]) / g.dr)}, every=0.05)


def run_rot(xi, dx_mm, T, omega):
    """Integrated gravity + capillarity + rotation equilibrium (pinned wall, matched
    rotating wall/bottom, reference: height_benchmarks.solve_meniscus_pinned_rotating)."""
    from air_vortex.height_benchmarks import build_height_rotating
    from air_vortex.height_interface import graph_validity
    s, ref = build_height_rotating(xi, dx_mm * 1e-3, omega)
    g = s.grid
    eta_ref = ref.eta(g.r_c)
    depth_ref = float(ref.eta(R_V) - ref.eta(0.0))
    print(json.dumps({"mode": "rot", "xi": xi, "dx_mm": dx_mm, "omega": omega, "depth_ref_mm": depth_ref * 1e3,
                      "Nr": g.Nr, **graph_validity(g, s.interface())}), flush=True)

    def extra(ss):
        v = graph_validity(g, ss.interface())
        liq = ss.geometry.liquid
        ut = np.where(liq, ss.fields.u_theta - omega * g.r_c[:, None], 0.0)
        return {"center_depression_err_dx": float(((ss.eta[-1] - ss.eta[0]) - (eta_ref[-1] - eta_ref[0])) / g.dr),
                "eta_nrmse_depth": float(np.sqrt(np.mean((ss.eta - eta_ref) ** 2)) / depth_ref),
                "utheta_dev_max": float(np.abs(ut).max()), "min_N_m": v["min_N_m"], "min_N_theta": v["min_N_theta"],
                "wall_value_err": float(ss.interface().value(np.array([R_V]))[0] - ss.z_pin)}
    log_run(s, eta_ref, T, extra=extra, every=0.02)


def run_force(rpm, dx_mm, T):
    """QUALITATIVE forcing sanity (uncalibrated tau_s, placeholder geometry): trends only."""
    from air_vortex.height_benchmarks import build_height_production
    from air_vortex.height_interface import graph_validity
    s = build_height_production(rpm, dx=dx_mm * 1e-3)
    g, cfg = s.grid, s.cfg
    H = cfg.geometry.water_height_m
    bar_top = cfg.geometry.stirbar_center_z_m + 0.5 * cfg.geometry.stirbar_diameter_m
    V0 = s.volume
    print(json.dumps({"mode": "force", "rpm": rpm, "dx_mm": dx_mm, "tau_s": cfg.stirrer.forcing_tau_s,
                      "calibrated": False, "R": g.r_v, "H": H, "bar_top": bar_top}), flush=True)
    nxt = 0.0
    while s.fields.t < T:
        d = s.step()
        v = graph_validity(g, s.interface())
        flag = None
        if v["max_slope"] > 5.0:
            flag = "graph slope > 5: outside the declared single-valued-graph scope"
        if s.eta[0] - bar_top < 2 * g.dz:
            flag = "interface within 2 cells of the bar top: first-contact regime reached"
        if d.t >= nxt or flag:
            f = s.fields
            liq = s.geometry.liquid
            uz_axis = np.where(liq[0], f.u_z[0, :-1], 0.0)
            print(json.dumps({"t": d.t, "u_theta_max": float(np.abs(np.where(liq, f.u_theta, 0)).max()),
                              "U_mer": max(d.max_abs_ur_liquid, d.max_abs_uz_liquid),
                              "p_center_bottom_minus_hydro": float(f.p[0, 0] - RHO * G * (H - g.z_c[0])),
                              "depression_mm": float((H - s.eta[0]) * 1e3),
                              "eta_wall_minus_axis_mm": float((s.eta[-1] - s.eta[0]) * 1e3),
                              "axis_uz_min": float(uz_axis.min()), "dV": s.volume / V0 - 1, **v,
                              "flag": flag}), flush=True)
            nxt += 0.05
            if flag:
                return


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "geom":
        run_geom()
    elif a[0] == "transport":
        run_transport()
    elif a[0] == "run":
        run_meniscus(float(a[1]), float(a[2]), float(a[3]), float(a[4]))
    elif a[0] == "hydro":
        run_hydro(float(a[1]), float(a[2]))
    elif a[0] == "force":
        run_force(float(a[1]), float(a[2]), float(a[3]))
    elif a[0] == "rot":
        run_rot(float(a[1]), float(a[2]), float(a[3]), float(a[4]))
    elif a[0] == "rigid":
        run_rigid(float(a[1]), float(a[2]), float(a[3]))
