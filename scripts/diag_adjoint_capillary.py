"""DIAGNOSTIC ONLY (V4b-P): is the remaining instability caused by a capillary
force that is not the discrete energetic adjoint of the ACTUAL interface
kinematics?  Everything is linearized about a RELAXED state of the production
path (pinned, reconstruct_ghost, level-set curvature; wall-compatible IC,
reinit OFF, PRE_T seconds of time integration).

Pieces (all measured from the solver's own operators, fixed dt0):
  x = (phi on |phi| < 6 dx, u_r on known faces, u_z on known faces)
  J3 = d step_F3 / dx  (production)       J2 = d step_F2 / dx  (curvature frozen as
                                            kappa_ref(r_crossing): no shape feedback)
  C_cur = (J3 - J2)[u rows, phi cols]     the capillary shape -> momentum block (per step)
  A_u  = d phi^+ / d u_new                extension + level-set advection of one step
  S    = d h / d phi                      column heights (wall_curvature.column_heights)
  K    = S A_u / dt                       measured kinematics u -> h_dot
  K_c  continuum graph kinematics eta_t = u_z - u_r eta_r (bilinear from MAC faces)
  K_Q  flux kinematics: h_dot_i = (outward volume flux through column i's mixed faces)
       / (2 pi r_i dr)   (the kinematics a column-owned sharp pressure is adjoint to)
  E(h) = sigma A(h), A = polyline surface of revolution through (0,h_0), (r_i,h_i),
       (R, z_pin) with h_wall = z_pin FIXED;  g = dE/dh, H = d2E/dh2 (analytic, FD-checked)
  M    = rho x axisymmetric face control volume (u_r: 2 pi r_f dr dz, u_z: 2 pi r_c dr dz);
       M_theta = M with mixed faces x their GFM fraction theta (P is M_theta-symmetric)
  P    = free-surface GFM projection (homogeneous interface Dirichlet), from the solver's
       solve_liquid_pressure + project_liquid_velocity
  C_EC = dt P (-M_theta^-1 K^T H S)       energy-consistent capillary block (per step)
Block replacement:  J_X = J2 + [[A_u C_X], [C_X]]  (phi rows get the advection of the
changed u_new); J_cur rebuilt this way must reproduce J3 (checked).

Usage: python scripts/diag_adjoint_capillary.py THETA XI DX_MM [PRE_T] [PRE_CASE F3|F2]
Writes one JSON summary line and results/validation_adjoint/adj_th*_xi*_dx*.npz
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

from air_vortex.free_surface_bc import interface_pressure  # noqa: E402
from air_vortex.levelset import advect_level_set_advective  # noqa: E402
from air_vortex.liquid_mask import FACE_LIQ_MINUS, FACE_LIQ_PLUS, classify  # noqa: E402
from air_vortex.meniscus import RHO, SIGMA  # noqa: E402
from air_vortex.pinned_phase import R_V  # noqa: E402
from air_vortex.pressure_single_phase import project_liquid_velocity, solve_liquid_pressure  # noqa: E402
from air_vortex.velocity_extension import extend_velocity  # noqa: E402
from air_vortex.wall_curvature import column_heights  # noqa: E402
from diag_pinned_energy import make  # noqa: E402

BAND = 6.0


# ------------------------------------------------------------ surface energy
def area_grad_hess(r_c, h, z_pin):
    """A, dA/dh, d2A/dh2 for the polyline (0,h0),(r_i,h_i),(R,z_pin); h_wall fixed."""
    n = len(h)
    r = np.r_[0.0, r_c, R_V]
    z = np.r_[h[0], h, z_pin]
    idx = np.r_[0, np.arange(n), -1]             # DOF of each polyline point (-1 = fixed)
    A = 0.0
    g = np.zeros(n)
    H = np.zeros((n, n))
    for k in range(len(r) - 1):
        c = np.pi * (r[k] + r[k + 1])
        dr, dz = r[k + 1] - r[k], z[k + 1] - z[k]
        L = np.hypot(dr, dz)
        A += c * L
        d1 = c * dz / L                          # dA_k / d(dz)
        d2 = c * dr**2 / L**3                    # d2A_k / d(dz)^2
        a, b = idx[k], idx[k + 1]
        for p, sp in ((a, -1.0), (b, 1.0)):
            if p >= 0:
                g[p] += sp * d1
        for p, sp in ((a, -1.0), (b, 1.0)):
            for q, sq in ((a, -1.0), (b, 1.0)):
                if p >= 0 and q >= 0:
                    H[p, q] += sp * sq * d2
    return A, g, H


def fd_check(r_c, h, z_pin):
    A0, g, H = area_grad_hess(r_c, h, z_pin)
    rng = np.random.default_rng(0)
    out = {}
    for eps in (1e-5, 1e-6, 1e-7, 1e-8):
        eg, eh = [], []
        for _ in range(5):
            v = rng.standard_normal(len(h))
            Ap = area_grad_hess(r_c, h + eps * v, z_pin)
            Am = area_grad_hess(r_c, h - eps * v, z_pin)
            eg.append(abs((Ap[0] - Am[0]) / (2 * eps) - g @ v) / abs(g @ v))
            eh.append(np.linalg.norm((Ap[1] - Am[1]) / (2 * eps) - H @ v) / np.linalg.norm(H @ v))
        out[f"{eps:g}"] = (float(max(eg)), float(max(eh)))
    return out


# ------------------------------------------------------------ main
def main(theta, xi, dx_mm, pre_t=1.0, pre_case="F3"):
    tic = time.time()
    s3, ref, _ = make("F3", theta, xi, dx_mm)
    g_, f = s3.grid, s3.fields
    s2, _, _ = make("F2", theta, xi, dx_mm)
    sp = s3 if pre_case == "F3" else s2          # relax with production, or with the stable F2
    while sp.fields.t < pre_t:
        sp.step()
    if sp is s2:
        for k in ("phi", "u_r", "u_z", "u_theta", "p"):
            setattr(f, k, getattr(s2.fields, k).copy())
        f.t, f.step = s2.fields.t, s2.fields.step
    base = {k: getattr(f, k).copy() for k in ("phi", "u_r", "u_z", "u_theta", "p")}
    t0, st0 = f.t, f.step
    dt = s3.stable_timestep(f.u_r, f.u_z)
    z_pin = s3.cfg.wall.pinned_contact_height_m
    nl = s3.cfg.physics.extension_layers_capillary
    ls = s3.cfg.levelset
    geom = classify(base["phi"])
    m_phi = np.abs(base["phi"]) < BAND * g_.dr
    m_ur = geom.ur_face_known().copy(); m_ur[0, :] = m_ur[-1, :] = False
    m_uz = geom.uz_face_known().copy(); m_uz[:, 0] = False
    nphi, nur, nuz = int(m_phi.sum()), int(m_ur.sum()), int(m_uz.sum())
    nu_ = nur + nuz
    N = nphi + nu_
    Nr = g_.Nr
    eps_phi, eps_u = 1e-7 * g_.dr, 1e-9

    def load(s, x=None):
        ff = s.fields
        for k, v in base.items():
            setattr(ff, k, v.copy())
        ff.t, ff.step = t0, st0
        if x is not None:
            ff.phi[m_phi] += x[:nphi]
            ff.u_r[m_ur] += x[nphi:nphi + nur]
            ff.u_z[m_uz] += x[nphi + nur:]

    def outv(s):
        ff = s.fields
        return np.r_[ff.phi[m_phi], ff.u_r[m_ur], ff.u_z[m_uz]]

    def jac(s):
        load(s); s.step(dt); y0 = outv(s)
        Jm = np.empty((N, N))
        for j in range(N):
            e = np.zeros(N); eps = eps_phi if j < nphi else eps_u; e[j] = eps
            load(s, e); s.step(dt)
            Jm[:, j] = (outv(s) - y0) / eps
        return Jm

    J3 = jac(s3)
    J2 = jac(s2)

    # ---- kinematics: A_u = d phi^+ / d u_new (extension + advection), about the base u
    def advect(ur, uz):
        ure, uze, _, _ = extend_velocity(g_, geom, base["phi"], ur, uz, base["u_theta"], nl)
        return advect_level_set_advective(base["phi"], ure, uze, g_, dt, scheme=ls.advection_scheme,
                                          time_integrator=ls.time_integrator, limiter=ls.limiter)
    phi_b = advect(base["u_r"], base["u_z"])
    A_u = np.empty((nphi, nu_))
    for j in range(nu_):
        ur, uz = base["u_r"].copy(), base["u_z"].copy()
        if j < nur:
            ur[m_ur] += eps_u * (np.arange(nur) == j)
        else:
            uz[m_uz] += eps_u * (np.arange(nuz) == j - nur)
        A_u[:, j] = (advect(ur, uz)[m_phi] - phi_b[m_phi]) / eps_u

    # ---- S = dh/dphi (all columns)
    cols = np.arange(Nr)
    h0 = column_heights(g_, base["phi"], cols)
    S = np.empty((Nr, nphi))
    for j in range(nphi):
        ph = base["phi"].copy(); ph[m_phi] += eps_phi * (np.arange(nphi) == j)
        S[:, j] = (column_heights(g_, ph, cols) - h0) / eps_phi
    K = S @ A_u / dt

    # ---- continuum graph kinematics and flux kinematics
    hr = np.gradient(np.r_[h0[0], h0, z_pin], np.r_[0.0, g_.r_c, R_V])[1:-1]
    def kin_cont(ur, uz):
        out = np.empty(Nr)
        for i in range(Nr):
            zq = h0[i]
            uzc = np.interp(zq, g_.z_f, uz[i, :])
            urc = 0.5 * (np.interp(zq, g_.z_c, ur[i, :]) + np.interp(zq, g_.z_c, ur[i + 1, :]))
            out[i] = uzc - urc * hr[i]
        return out
    def col_flux(ur, uz):
        Q = np.zeros(Nr)
        kz = geom.face_kind_z
        for kind, sgn in ((FACE_LIQ_MINUS, 1.0), (FACE_LIQ_PLUS, -1.0)):
            i, j = np.nonzero(kz == kind)
            np.add.at(Q, i, sgn * uz[i, j] * 2 * np.pi * g_.r_c[i] * g_.dr)
        kr = geom.face_kind_r
        for kind, sgn, off in ((FACE_LIQ_MINUS, 1.0, -1), (FACE_LIQ_PLUS, -1.0, 0)):
            i, j = np.nonzero(kr == kind)
            np.add.at(Q, i + off, sgn * ur[i, j] * 2 * np.pi * g_.r_f[i] * g_.dz)
        return Q
    Wcol = 2 * np.pi * g_.r_c * g_.dr
    Kc = np.empty((Nr, nu_)); KQ = np.empty((Nr, nu_))
    for j in range(nu_):
        ur, uz = np.zeros_like(base["u_r"]), np.zeros_like(base["u_z"])
        if j < nur:
            ur[m_ur] = (np.arange(nur) == j) * 1.0
        else:
            uz[m_uz] = (np.arange(nuz) == j - nur) * 1.0
        Kc[:, j] = kin_cont(ur, uz)
        KQ[:, j] = col_flux(ur, uz) / Wcol

    # ---- energy
    Aval, gA, HA = area_grad_hess(g_.r_c, h0, z_pin)
    gE, HE = SIGMA * gA, SIGMA * HA
    fd = fd_check(g_.r_c, h0, z_pin)
    kappa_var = gA / Wcol                         # pointwise variational curvature (sign test)
    kap_ref = ref.kappa_at_height(ref.eta(g_.r_c))

    # ---- mass matrix and projection
    wr = RHO * 2 * np.pi * g_.r_f[:, None] * g_.dr * g_.dz * np.ones((1, g_.Nz))
    wz = RHO * 2 * np.pi * g_.r_c[:, None] * g_.dr * g_.dz * np.ones((1, g_.Nz + 1))
    Mv = np.r_[wr[m_ur], wz[m_uz]]
    th_r = np.where(np.isfinite(geom.theta_r), geom.theta_r, 1.0)
    th_z = np.where(np.isfinite(geom.theta_z), geom.theta_z, 1.0)
    Mth = np.r_[(wr * th_r)[m_ur], (wz * th_z)[m_uz]]
    pr0, pz0 = interface_pressure(geom)
    P = np.empty((nu_, nu_))
    for j in range(nu_):
        ur, uz = np.zeros_like(base["u_r"]), np.zeros_like(base["u_z"])
        if j < nur:
            ur[m_ur] = (np.arange(nur) == j) * 1.0
        else:
            uz[m_uz] = (np.arange(nuz) == j - nur) * 1.0
        p = solve_liquid_pressure(g_, geom, ur, uz, RHO, dt, pr0, pz0)
        a, b = project_liquid_velocity(g_, geom, ur, uz, p, RHO, dt, pr0, pz0)
        P[:, j] = np.r_[a[m_ur], b[m_uz]]
    MP = Mv[:, None] * P
    MthP = Mth[:, None] * P
    proj = {"idempotency": float(np.linalg.norm(P @ P - P) / np.linalg.norm(P)),
            "M_symmetry": float(np.linalg.norm(MP - MP.T) / np.linalg.norm(MP)),
            "Mtheta_symmetry": float(np.linalg.norm(MthP - MthP.T) / np.linalg.norm(MthP))}

    # ---- capillary blocks (per step), rows = u known faces, cols = phi band
    C_cur = (J3 - J2)[nphi:, :nphi]
    Kx = {"measured": K, "continuum": Kc, "flux": KQ}
    # the projection is exactly symmetric in M_theta (mixed faces weighted by their GFM
    # fraction theta), not in the plain face-volume M: M_theta is the solver's energy metric
    C_EC = {f"{k}_Mtheta": dt * P @ ((-(Kk.T @ (HE @ S))) / Mth[:, None]) for k, Kk in Kx.items()}
    C_EC["measured_Mplain"] = dt * P @ ((-(K.T @ (HE @ S))) / Mv[:, None])
    # consistency of the block decomposition: phi rows of J3-J2 = A_u C_cur
    dphi_rows = (J3 - J2)[:nphi, :nphi]
    decomp_err = float(np.linalg.norm(dphi_rows - A_u @ C_cur) / max(np.linalg.norm(dphi_rows), 1e-300))

    def eig_summary(Jm):
        mu = np.linalg.eigvals(Jm)
        lam = np.log(np.abs(mu)) / dt; fr = np.abs(np.angle(mu)) / (2 * np.pi * dt)
        k = int(np.argmax(lam)); o = fr > 1.0
        ko = int(np.nonzero(o)[0][np.argmax(lam[o])]) if o.any() else k
        return {"lambda_max": float(lam[k]), "f_max": float(fr[k]),
                "lambda_osc": float(lam[ko]), "f_osc": float(fr[ko])}

    def assemble(C):
        Jx = J2.copy()
        Jx[nphi:, :nphi] += C
        Jx[:nphi, :nphi] += A_u @ C
        return Jx
    eig = {"J3_current": eig_summary(J3), "J2_no_shape_feedback": eig_summary(J2),
           "rebuilt_current": eig_summary(assemble(C_cur))}
    for k, C in C_EC.items():
        eig[f"EC_{k}"] = eig_summary(assemble(C))
    # EC scaled check: is the EC coupling of comparable strength (not just "weaker")?
    strength = {k: float(np.linalg.norm(C) / np.linalg.norm(C_cur)) for k, C in C_EC.items()}

    # ---- adjoint defect (projected, per unit time, in M-work units)
    def rel(a, b):
        return float(np.linalg.norm(a - b) / max(np.linalg.norm(a), np.linalg.norm(b)))
    D = {k: rel(Mth[:, None] * C_cur / dt, Mth[:, None] * C / dt) for k, C in C_EC.items()}
    cos = {k: float(np.sum(C_cur * C) / (np.linalg.norm(C_cur) * np.linalg.norm(C))) for k, C in C_EC.items()}

    # ---- work identity with random smooth divergence-free fields
    rng = np.random.default_rng(1)
    pr_b, pz_b, _, _ = s3.interface_pressure_bc(geom, base["phi"])
    work = []
    for trial in range(5):
        a = rng.standard_normal(4)
        psi_r = g_.r_f ** 2 * (R_V ** 2 - g_.r_f ** 2) ** 2
        zz = g_.z_f / g_.z_f[-1]
        psi_z = zz * (1 + a[0] * zz + a[1] * zz**2) * (1 + a[2] * np.cos(np.pi * zz))
        psi = (psi_r[:, None] * (1 + a[3] * (g_.r_f[:, None] / R_V) ** 2)) * psi_z[None, :]
        ur = -(psi[:, 1:] - psi[:, :-1]) / (np.where(g_.r_f == 0, 1, g_.r_f)[:, None] * g_.dz); ur[0] = 0
        uz = (psi[1:, :] - psi[:-1, :]) / (g_.r_c[:, None] * g_.dr)
        uv = np.r_[ur[m_ur], uz[m_uz]]
        uv *= 1e-3 / np.abs(uv).max()
        ur2, uz2 = np.zeros_like(ur), np.zeros_like(uz); ur2[m_ur] = uv[:nur]; uz2[m_uz] = uv[nur:]
        W_surf = float(gE @ (K @ uv))
        Qf = col_flux(ur2, uz2)
        # current sharp capillary pressure work (sigma kappa at the mixed faces)
        Wcur = 0.0
        kz = geom.face_kind_z
        for kind, sgn in ((FACE_LIQ_MINUS, 1.0), (FACE_LIQ_PLUS, -1.0)):
            i, j = np.nonzero(kz == kind)
            Wcur += float(np.sum(-pz_b[i, j] * sgn * uz2[i, j] * 2 * np.pi * g_.r_c[i] * g_.dr))
        kr = geom.face_kind_r
        for kind, sgn in ((FACE_LIQ_MINUS, 1.0), (FACE_LIQ_PLUS, -1.0)):
            i, j = np.nonzero(kr == kind)
            Wcur += float(np.sum(-pr_b[i, j] * sgn * ur2[i, j] * 2 * np.pi * g_.r_f[i] * g_.dz))
        W_EC = float(-(K.T @ gE) @ uv)                       # u^T M f, f = -M^-1 K^T g
        W_colp = float(-np.sum(gE / Wcol * Qf))               # column-owned pressure g_i/W_i
        work.append({"W_surface": W_surf, "W_fluid_current": Wcur, "W_fluid_EC": W_EC, "W_fluid_colpressure": W_colp,
                     "rel_current": abs(Wcur + W_surf) / max(abs(Wcur), abs(W_surf)),
                     "rel_EC": abs(W_EC + W_surf) / max(abs(W_EC), abs(W_surf)),
                     "rel_colpressure": abs(W_colp + W_surf) / max(abs(W_colp), abs(W_surf))})

    res = {"theta": theta, "xi": xi, "dx_mm": dx_mm, "pre_t": pre_t, "pre_case": pre_case, "dt": dt, "N": N,
           "n_phi_u": [nphi, nu_], "seconds": round(time.time() - tic, 1),
           "area_fd": fd,
           "kappa_var_vs_ref": {"max_abs_err_last8": float(np.abs(kappa_var - kap_ref)[-8:].max()),
                                "max_abs_err_interior": float(np.abs(kappa_var - kap_ref)[2:-4].max()),
                                "sign_agree_frac": float(np.mean(np.sign(kappa_var) == np.sign(kap_ref)))},
           "K_vs_continuum_rel": float(np.linalg.norm(K - Kc) / np.linalg.norm(K)),
           "K_vs_flux_rel": float(np.linalg.norm(K - KQ) / np.linalg.norm(K)),
           "K_vs_continuum_rel_last8": float(np.linalg.norm((K - Kc)[-8:]) / np.linalg.norm(K[-8:])),
           "K_vs_flux_rel_last8": float(np.linalg.norm((K - KQ)[-8:]) / np.linalg.norm(K[-8:])),
           "projection": proj, "block_decomposition_err": decomp_err,
           "adjoint_defect_rel": D, "cosine_C_cur_vs_EC": cos, "EC_strength_vs_current": strength,
           "eig": eig, "work": work}
    print(json.dumps(res), flush=True)
    od = ROOT / "results" / "validation_adjoint"; od.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(od / f"adj_th{theta:g}_xi{xi:g}_dx{dx_mm:g}_{pre_case}{pre_t:g}.npz", J3=J3, J2=J2, A_u=A_u, S=S, K=K, Kc=Kc,
                        KQ=KQ, HE=HE, gE=gE, P=P, Mv=Mv, dt=dt, nphi=nphi, h0=h0)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(float(a[0]), float(a[1]), float(a[2]), float(a[3]) if len(a) > 3 else 1.0,
         a[4] if len(a) > 4 else "F3")
