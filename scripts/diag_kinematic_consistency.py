"""DIAGNOSTIC ONLY (V4b-P): does the instability disappear when the interface
moves with the mixed-face FLUX kinematics K_Q (the geometry through which the
sharp GFM pressure, gravity and capillarity do work) instead of the level-set
kinematics K_LS?

REDUCED HEIGHT MAP (also the Section-13 nonlinear prototype). State (h, u):
  h = column interface heights (authoritative; the pinned end (R, z_pin) is fixed),
  u = MAC velocities on the known faces of the base geometry.
One step:  phi = signed distance to a smooth graph through h (cubic spline,
  mirrored at the axis, continued past the wall; no reinitialization),
  run the production solver step (pressure, GFM capillarity, gravity,
  projection -- unchanged) -> u+, then
    LS : h+ = h + [heights(phi+) - heights(phi)]   (level-set transport, increment form)
    Q  : h+ = h + dt K_Q(u+),  K_Q u = (outward flux through column i's mixed
         faces, radial faces owned by the LIQUID-side column) / (2 pi r_i dr)
    Qv : as Q, radial mixed faces owned by the VOID-side column
Energy blocks (height representation, h_wall fixed):
  E_sigma = sigma * polyline surface-of-revolution area (as diag_adjoint_capillary)
  E_g = rho g sum_i a_i h_i^2 / 2, a_i = pi (r_f[i+1]^2 - r_f[i]^2) = 2 pi r_i dr
  H_p = H_g + H_sigma;  C_Q,total = dt P (-M_theta^-1 K_Q^T H_p)  replaces the WHOLE
  du+/dh block (gravity + capillarity) in the Q-paired "total EC" Jacobian.

Usage:
  python scripts/diag_kinematic_consistency.py jac THETA XI DX_MM PRE_T
  python scripts/diag_kinematic_consistency.py run THETA XI DX_MM PRE_T T_END KIN   (KIN = LS|Q|Qv)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
from scipy.interpolate import CubicSpline  # noqa: E402

from air_vortex.free_surface_bc import interface_pressure  # noqa: E402
from air_vortex.liquid_mask import FACE_LIQ_MINUS, FACE_LIQ_PLUS, classify, liquid_volume_subcell  # noqa: E402
from air_vortex.meniscus import G, RHO, SIGMA  # noqa: E402
from air_vortex.pinned_phase import R_V  # noqa: E402
from air_vortex.pressure_single_phase import project_liquid_velocity, solve_liquid_pressure  # noqa: E402
from air_vortex.wall_curvature import column_heights  # noqa: E402
from diag_adjoint_capillary import area_grad_hess  # noqa: E402
from diag_pinned_energy import make  # noqa: E402

OUT = ROOT / "results" / "validation_kinematics"


def graph_phi(g, h, z_pin, spp=64):
    """Signed distance to the cubic-spline graph through (+-r_i, h_i), (R, z_pin):
    exact point-to-SEGMENT distance to a fine polyline of the spline (point
    sampling is O(ds^2/d) wrong for nodes close to the curve)."""
    r = np.r_[-R_V, -g.r_c[::-1], g.r_c, R_V]
    z = np.r_[z_pin, h[::-1], h, z_pin]
    sp = CubicSpline(r, z, bc_type="not-a-knot", extrapolate=True)
    rs = np.linspace(-R_V - 4 * g.dr, R_V + 4 * g.dr, spp * (2 * g.Nr + 8) + 1)
    zs = sp(rs)
    ax, az = rs[:-1], zs[:-1]
    bx, bz = rs[1:] - ax, zs[1:] - az
    L2 = bx**2 + bz**2
    out = np.empty(g.shape_center)
    for i in range(g.Nr):
        # only segments within 8 cells of this column
        m = np.abs(ax - g.r_c[i]) < 8 * g.dr
        px = g.r_c[i] - ax[m][None, :]
        pz = g.z_c[:, None] - az[m][None, :]
        t = np.clip((px * bx[m] + pz * bz[m]) / L2[m], 0.0, 1.0)
        d2 = (px - t * bx[m]) ** 2 + (pz - t * bz[m]) ** 2
        out[i] = np.sign(g.z_c - sp(g.r_c[i])) * np.sqrt(d2.min(axis=1))
    return out


def col_flux(g, geom, ur, uz, owner="liquid"):
    Q = np.zeros(g.Nr)
    kz = geom.face_kind_z
    for kind, sgn in ((FACE_LIQ_MINUS, 1.0), (FACE_LIQ_PLUS, -1.0)):
        i, j = np.nonzero(kz == kind)
        np.add.at(Q, i, sgn * uz[i, j] * 2 * np.pi * g.r_c[i] * g.dr)
    kr = geom.face_kind_r
    for kind, sgn, off_l, off_v in ((FACE_LIQ_MINUS, 1.0, -1, 0), (FACE_LIQ_PLUS, -1.0, 0, -1)):
        i, j = np.nonzero(kr == kind)
        off = off_l if owner == "liquid" else off_v
        np.add.at(Q, np.clip(i + off, 0, g.Nr - 1), sgn * ur[i, j] * 2 * np.pi * g.r_f[i] * g.dz)
    return Q


class Reduced:
    def __init__(self, theta, xi, dx_mm, pre_t):
        s, ref, _ = make("F3", theta, xi, dx_mm)
        while s.fields.t < pre_t:                       # relax with the production path
            s.step()
        self.s, self.ref, self.g = s, ref, s.grid
        self.z_pin = s.cfg.wall.pinned_contact_height_m
        self.h0 = column_heights(self.g, s.fields.phi, np.arange(self.g.Nr))
        self.base = {k: getattr(s.fields, k).copy() for k in ("u_r", "u_z", "u_theta", "p")}
        self.t0, self.st0 = s.fields.t, s.fields.step
        self.dt = s.stable_timestep(s.fields.u_r, s.fields.u_z)
        phi0 = graph_phi(self.g, self.h0, self.z_pin)
        self.geom0 = classify(phi0)
        self.m_ur = self.geom0.ur_face_known().copy(); self.m_ur[0, :] = self.m_ur[-1, :] = False
        self.m_uz = self.geom0.uz_face_known().copy(); self.m_uz[:, 0] = False
        self.nur, self.nuz = int(self.m_ur.sum()), int(self.m_uz.sum())
        self.u0 = np.r_[self.base["u_r"][self.m_ur], self.base["u_z"][self.m_uz]]

    def unpack(self, uv):
        ur, uz = self.base["u_r"].copy(), self.base["u_z"].copy()
        ur[self.m_ur] = uv[:self.nur]; uz[self.m_uz] = uv[self.nur:]
        return ur, uz

    def step(self, h, uv, kin="LS", dt=None):
        """One reduced step; returns (h+ for LS/Q/Qv as a dict, u+ vector, phi, geom)."""
        s, f, g = self.s, self.s.fields, self.g
        dt = self.dt if dt is None else dt
        phi = graph_phi(g, h, self.z_pin)
        ur, uz = self.unpack(uv)
        f.phi, f.u_r, f.u_z = phi, ur, uz
        f.u_theta, f.p = self.base["u_theta"].copy(), self.base["p"].copy()
        f.t, f.step = self.t0, self.st0
        geom = classify(phi)
        s.step(dt)
        up = np.r_[f.u_r[self.m_ur], f.u_z[self.m_uz]]
        wcol = 2 * np.pi * g.r_c * g.dr
        # LS in INCREMENT form: the round trip h -> phi(h) -> heights is not exactly the
        # identity (a per-step grid-scale gain ~1.01 would be a spurious ~+75 /s mode)
        h_rt = column_heights(g, phi, np.arange(g.Nr))
        hp = {"LS": h + column_heights(g, f.phi, np.arange(g.Nr)) - h_rt,
              "Q": h + dt * col_flux(g, geom, f.u_r, f.u_z, "liquid") / wcol,
              "Qv": h + dt * col_flux(g, geom, f.u_r, f.u_z, "void") / wcol}
        return hp, up, phi, geom


def eig_summary(Jm, dt):
    mu, V = np.linalg.eig(Jm)
    lam = np.log(np.abs(mu)) / dt; fr = np.abs(np.angle(mu)) / (2 * np.pi * dt)
    k = int(np.argmax(lam)); o = fr > 1.0
    ko = int(np.nonzero(o)[0][np.argmax(lam[o])]) if o.any() else k
    return {"lambda_max": float(lam[k]), "f_max": float(fr[k]), "lambda_osc": float(lam[ko]),
            "f_osc": float(fr[ko])}, V[:, k], mu[k]


def jac(theta, xi, dx_mm, pre_t):
    tic = time.time()
    R = Reduced(theta, xi, dx_mm, pre_t)
    g, dt, h0, u0 = R.g, R.dt, R.h0, R.u0
    nh, nu = g.Nr, len(u0)
    eh, eu = 1e-7 * g.dr, 1e-9
    hp0, up0, _, geom0 = R.step(h0, u0)
    B = {k: np.empty((nh + nu, nh + nu)) for k in ("LS", "Q", "Qv")}
    for j in range(nh + nu):
        h, uv = h0.copy(), u0.copy()
        if j < nh:
            h[j] += eh; eps = eh
        else:
            uv[j - nh] += eu; eps = eu
        hp, up, _, _ = R.step(h, uv)
        for k in B:
            B[k][:nh, j] = (hp[k] - hp0[k]) / eps
            B[k][nh:, j] = (up - up0) / eps
    res = {"theta": theta, "xi": xi, "dx_mm": dx_mm, "pre_t": pre_t, "dt": dt, "nh": nh, "nu": nu}
    eig = {}
    vecs = {}
    for k, Jm in B.items():
        eig[k], vecs[k], _ = eig_summary(Jm, dt)
    # ---- operators at the base: K_LS (kinematic part only), K_Q, masses, projection, energy
    ur0, uz0 = R.unpack(u0)
    wcol = 2 * np.pi * g.r_c * g.dr
    KQ = np.empty((nh, nu)); KQv = np.empty((nh, nu)); KLS = np.empty((nh, nu))
    for j in range(nu):
        e = np.zeros(nu); e[j] = 1.0
        a, b = R.unpack(e); a[~R.m_ur] = 0; b[~R.m_uz] = 0
        KQ[:, j] = col_flux(g, geom0, a, b, "liquid") / wcol
        KQv[:, j] = col_flux(g, geom0, a, b, "void") / wcol
    # K_LS measured directly: extension + level-set advection of phi(h0), heights, / dt
    from air_vortex.levelset import advect_level_set_advective
    from air_vortex.velocity_extension import extend_velocity
    phi0 = graph_phi(g, h0, R.z_pin)
    ls, nl = R.s.cfg.levelset, R.s.cfg.physics.extension_layers_capillary

    def heights_after(a, b):
        ure, uze, _, _ = extend_velocity(g, geom0, phi0, a, b, R.base["u_theta"], nl)
        ph = advect_level_set_advective(phi0, ure, uze, g, dt, scheme=ls.advection_scheme,
                                        time_integrator=ls.time_integrator, limiter=ls.limiter)
        return column_heights(g, ph, np.arange(g.Nr))
    hb = heights_after(ur0, uz0)
    for j in range(nu):
        e = np.zeros(nu); e[j] = eu
        a, b = R.unpack(u0 + e)
        KLS[:, j] = (heights_after(a, b) - hb) / eu / dt
    th_r = np.where(np.isfinite(geom0.theta_r), geom0.theta_r, 1.0)
    th_z = np.where(np.isfinite(geom0.theta_z), geom0.theta_z, 1.0)
    wr = RHO * 2 * np.pi * g.r_f[:, None] * g.dr * g.dz * th_r
    wz = RHO * 2 * np.pi * g.r_c[:, None] * g.dr * g.dz * th_z
    Mth = np.r_[wr[R.m_ur], wz[R.m_uz]]
    pr0, pz0 = interface_pressure(geom0)
    P = np.empty((nu, nu))
    for j in range(nu):
        e = np.zeros(nu); e[j] = 1.0
        a, b = R.unpack(e); a[~R.m_ur] = 0; b[~R.m_uz] = 0
        p = solve_liquid_pressure(g, geom0, a, b, RHO, dt, pr0, pz0)
        a2, b2 = project_liquid_velocity(g, geom0, a, b, p, RHO, dt, pr0, pz0)
        P[:, j] = np.r_[a2[R.m_ur], b2[R.m_uz]]
    _, gS, HS = area_grad_hess(g.r_c, h0, R.z_pin)
    aring = np.pi * (g.r_f[1:] ** 2 - g.r_f[:-1] ** 2)
    gG, HG = RHO * G * aring * h0, RHO * G * np.diag(aring)
    # FD check of E_g
    v = np.random.default_rng(0).standard_normal(nh)
    Eg = lambda hh: RHO * G * np.sum(aring * hh**2 / 2)
    fdg = abs((Eg(h0 + 1e-7 * v) - Eg(h0 - 1e-7 * v)) / 2e-7 - gG @ v) / abs(gG @ v)
    gp, Hp = SIGMA * gS + gG, SIGMA * HS + HG
    # ---- total-EC paired Jacobians: replace the whole du+/dh block
    for kname, Kk in (("Q", KQ), ("Qv", KQv), ("LS", KLS)):
        Cuh = dt * P @ (-(Kk.T @ Hp) / Mth[:, None])
        Jm = B[kname].copy()
        Buu_k = Jm[nh:, nh:]
        Jm[nh:, :nh] = Cuh
        Jm[:nh, :nh] = np.eye(nh) + dt * Kk @ Cuh
        Jm[:nh, nh:] = dt * Kk @ Buu_k
        eig[f"{kname}_totalEC"], vecs[f"{kname}_totalEC"], _ = eig_summary(Jm, dt)
    # ---- structural residuals
    rng = np.random.default_rng(1)
    RK = []
    for _ in range(8):
        a4 = rng.standard_normal(4)
        psi_r = g.r_f**2 * (R_V**2 - g.r_f**2) ** 2
        zz = g.z_f / g.z_f[-1]
        psi = psi_r[:, None] * (1 + a4[3] * (g.r_f[:, None] / R_V) ** 2) * \
            (zz * (1 + a4[0] * zz + a4[1] * zz**2) * (1 + a4[2] * np.cos(np.pi * zz)))[None, :]
        ur = -(psi[:, 1:] - psi[:, :-1]) / (np.where(g.r_f == 0, 1, g.r_f)[:, None] * g.dz); ur[0] = 0
        uz = (psi[1:, :] - psi[:-1, :]) / (g.r_c[:, None] * g.dr)
        uv = np.r_[ur[R.m_ur], uz[R.m_uz]]; uv /= np.abs(uv).max()
        W = abs(gp @ (KQ @ uv))
        RK.append((gp @ ((KLS - KQ) @ uv)) / W)
    # modal: second-order work mismatch for the least-stable mode of the LS (current) map
    vm = vecs["LS"]; dh, du = vm[:nh], vm[nh:]
    modal = float(abs(np.conj(Hp @ dh) @ ((KLS - KQ) @ du)) /
                  max(abs(np.conj(Hp @ dh) @ (KQ @ du)), 1e-300))
    res.update({
        "seconds": round(time.time() - tic, 1), "eig": eig,
        "K_LS_vs_K_Q_rel": float(np.linalg.norm(KLS - KQ) / np.linalg.norm(KLS)),
        "K_LS_vs_K_Q_rel_last8": float(np.linalg.norm((KLS - KQ)[-8:]) / np.linalg.norm(KLS[-8:])),
        "K_LS_vs_K_Qv_rel": float(np.linalg.norm(KLS - KQv) / np.linalg.norm(KLS)),
        "Eg_fd_rel": float(fdg),
        "R_K_random_rms": float(np.sqrt(np.mean(np.square(RK)))), "R_K_random_max": float(np.max(np.abs(RK))),
        "R_K_modal_rel": modal,
    })
    print(json.dumps(res), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"kin_th{theta:g}_xi{xi:g}_dx{dx_mm:g}.npz", **{f"J_{k}": v for k, v in B.items()},
                        KLS=KLS, KQ=KQ, KQv=KQv, P=P, Mth=Mth, Hp=Hp, gp=gp, dt=dt, h0=h0)


def run(theta, xi, dx_mm, pre_t, t_end, kin):
    """Nonlinear reduced prototype: h authoritative, evolved with LS or Q kinematics."""
    R = Reduced(theta, xi, dx_mm, pre_t)
    g = R.g
    h, uv = R.h0.copy(), R.u0.copy()
    href = R.ref.eta(g.r_c)
    t, n = 0.0, 0
    aring = np.pi * (g.r_f[1:] ** 2 - g.r_f[:-1] ** 2)
    V0 = float(np.sum(aring * h))
    print(json.dumps({"mode": "run", "theta": theta, "xi": xi, "kin": kin, "pre_t": pre_t, "dt": R.dt}), flush=True)
    next_log = 0.0
    while t < t_end:
        hp, uv, phi, geom = R.step(h, uv, kin)
        h = hp[kin]
        t += R.dt; n += 1
        U = float(np.abs(uv).max())
        if not np.isfinite(U) or U > 5.0:
            print(json.dumps({"t": t, "blowup": True}), flush=True); break
        if t >= next_log:
            ur, uz = R.unpack(uv)
            K = 0.5 * RHO * float(np.sum(2 * np.pi * g.r_f[:, None] * g.dr * g.dz * ur**2 * R.m_ur)
                                  + np.sum(2 * np.pi * g.r_c[:, None] * g.dr * g.dz * uz**2 * R.m_uz))
            A, _, _ = area_grad_hess(g.r_c, h, R.z_pin)
            Eg = RHO * G * float(np.sum(aring * h**2 / 2))
            V = float(np.sum(aring * h))
            E = K + SIGMA * A + Eg - R.ref.P0 * V
            print(json.dumps({"t": t, "U": U, "K": K, "E_total_minus_P0V": E,
                              "rmse_dx": float(np.sqrt(np.mean((h - href) ** 2)) / g.dr),
                              "vol_drift": V / V0 - 1, "h_axis_dx": float((h[0] - href[0]) / g.dr)}), flush=True)
            next_log += 0.01


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "jac":
        jac(float(a[1]), float(a[2]), float(a[3]), float(a[4]))
    elif a[0] == "run":
        run(float(a[1]), float(a[2]), float(a[3]), float(a[4]), float(a[5]), a[6])
