"""DIAGNOSTIC ONLY (V4b-P): does the production instability live in phi's
OFF-CONTOUR degrees of freedom (level-set perturbations that do not move the
zero contour but change the curvature stencil)?

About a relaxed production state (pinned, reconstruct_ghost, level-set
curvature, reinit OFF): J3 = FD Jacobian of one step on x = (phi band, u known).
  S = d h / d phi_band            (column heights, cubic root)
  R = d phi_band / d h            (exact signed distance of the spline graph through h)
  Pi = R S                        projection of a phi perturbation onto contour-consistent
                                  (signed-distance) perturbations
Reported:
  - eigenvalues of J3 and of D J3 D, D = diag(Pi, I): the production step with the
    off-contour phi component removed every step
  - for the least-stable modes: off-contour fraction ||dphi - Pi dphi|| / ||dphi||
Usage: python scripts/diag_offcontour_mode.py THETA XI DX_MM PRE_T
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

from air_vortex.liquid_mask import classify  # noqa: E402
from air_vortex.wall_curvature import column_heights  # noqa: E402
from diag_kinematic_consistency import graph_phi  # noqa: E402
from diag_pinned_energy import make  # noqa: E402


def main(theta, xi, dx_mm, pre_t):
    tic = time.time()
    s, ref, _ = make("F3", theta, xi, dx_mm)
    f, g = s.fields, s.grid
    while f.t < pre_t:
        s.step()
    dt = s.stable_timestep(f.u_r, f.u_z)
    base = {k: getattr(f, k).copy() for k in ("phi", "u_r", "u_z", "u_theta", "p")}
    t0, st0 = f.t, f.step
    geom = classify(base["phi"])
    m_phi = np.abs(base["phi"]) < 6 * g.dr
    m_ur = geom.ur_face_known().copy(); m_ur[0, :] = m_ur[-1, :] = False
    m_uz = geom.uz_face_known().copy(); m_uz[:, 0] = False
    nphi, nur, nuz = int(m_phi.sum()), int(m_ur.sum()), int(m_uz.sum())
    N = nphi + nur + nuz
    ep, eu = 1e-7 * g.dr, 1e-9

    def load(x=None):
        for k, v in base.items():
            setattr(f, k, v.copy())
        f.t, f.step = t0, st0
        if x is not None:
            f.phi[m_phi] += x[:nphi]; f.u_r[m_ur] += x[nphi:nphi + nur]; f.u_z[m_uz] += x[nphi + nur:]
    out = lambda: np.r_[f.phi[m_phi], f.u_r[m_ur], f.u_z[m_uz]]  # noqa: E731
    load(); s.step(dt); y0 = out()
    J = np.empty((N, N))
    for j in range(N):
        e = np.zeros(N); eps = ep if j < nphi else eu; e[j] = eps
        load(e); s.step(dt); J[:, j] = (out() - y0) / eps
    cols = np.arange(g.Nr)
    h0 = column_heights(g, base["phi"], cols)
    z_pin = s.cfg.wall.pinned_contact_height_m
    S = np.empty((g.Nr, nphi))
    for j in range(nphi):
        ph = base["phi"].copy(); ph[m_phi] += ep * (np.arange(nphi) == j)
        S[:, j] = (column_heights(g, ph, cols) - h0) / ep
    pg0 = graph_phi(g, h0, z_pin)
    Rm = np.empty((nphi, g.Nr))
    for j in range(g.Nr):
        hh = h0.copy(); hh[j] += ep
        Rm[:, j] = (graph_phi(g, hh, z_pin)[m_phi] - pg0[m_phi]) / ep
    Pi_approx = Rm @ S
    # exact (oblique) projection: S R is not exactly the identity (round-trip gain ~1.004), and a
    # non-idempotent Pi applied every step is itself a spurious per-step gain
    Pi = Rm @ np.linalg.solve(S @ Rm, S)
    D = np.eye(N); D[:nphi, :nphi] = Pi
    Da = np.eye(N); Da[:nphi, :nphi] = Pi_approx

    def summ(Jm, label):
        mu, V = np.linalg.eig(Jm)
        lam = np.log(np.abs(mu)) / dt; fr = np.abs(np.angle(mu)) / (2 * np.pi * dt)
        order = np.argsort(lam)[::-1]
        rows, seen = [], set()
        for k in order:
            key = (round(lam[k], 3), round(fr[k], 1))
            if key in seen:
                continue
            seen.add(key)
            dphi = V[:nphi, k]
            nf = np.linalg.norm(dphi)
            off = float(np.linalg.norm(dphi - Pi @ dphi) / nf) if nf > 0 else float("nan")
            share = float(nf / np.linalg.norm(V[:, k]))
            dh = np.abs(S @ dphi)                       # the mode in column-height space
            rows.append({"lambda": float(lam[k]), "f": float(fr[k]), "offcontour_frac": off, "phi_share": share,
                         "h_peak_col": int(np.argmax(dh)), "h_wall3_frac": float(np.sum(dh[-3:] ** 2) / max(np.sum(dh ** 2), 1e-300)),
                         "h_profile": [float(v) for v in (dh / max(dh.max(), 1e-300))[-8:]]})
            if len(rows) >= 5:
                break
        return {"label": label, "top": rows}
    # regional slaving: replace the phi perturbation by its contour-consistent version only in
    # cells of the given columns, keep the rest of the band unchanged
    colidx = np.nonzero(m_phi)[0]
    regional = {}
    for lab, cset in (("wall4", range(g.Nr - 4, g.Nr)), ("wall8", range(g.Nr - 8, g.Nr)),
                      ("wall12", range(g.Nr - 12, g.Nr)), ("interior_lt_N-8", range(0, g.Nr - 8)),
                      ("axis_quarter", range(0, g.Nr // 4))):
        m = np.isin(colidx, list(cset)).astype(float)
        Pr = np.diag(m) @ Pi + np.diag(1 - m)
        Dr = np.eye(N); Dr[:nphi, :nphi] = Pr
        regional[lab] = summ(Dr @ J @ Dr, f"slave {lab}")["top"][:2]
    res = {"theta": theta, "xi": xi, "dx_mm": dx_mm, "pre_t": pre_t, "dt": dt, "N": N,
           "Pi_idempotency": float(np.linalg.norm(Pi @ Pi - Pi) / np.linalg.norm(Pi)),
           "Pi_approx_idempotency": float(np.linalg.norm(Pi_approx @ Pi_approx - Pi_approx) / np.linalg.norm(Pi_approx)),
           "production": summ(J, "J3"), "contour_projected": summ(D @ J @ D, "D J3 D (exact projection)"),
           "contour_projected_approx": summ(Da @ J @ Da, "Da J3 Da (R S, approx)"),
           "regional": regional,
           "seconds": round(time.time() - tic, 1)}
    print(json.dumps(res), flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(float(a[0]), float(a[1]), float(a[2]), float(a[3]))
