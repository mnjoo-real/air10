"""DIAGNOSTIC ONLY (V4b-P mechanism pass): finite-difference Jacobian of ONE
fixed-dt solver step about the pinned BVP state, and its eigenvalues.

x = (phi on the band |phi| < BAND dx, u_r on liquid-known faces, u_z on
liquid-known faces), all taken from the base state (t = 0, u = 0, the
wall-compatible IC). Void-face velocities are not state: every step begins by
re-extending them from the liquid faces. Far-field phi is inert. Each column
J[:, j] = (S(x0 + eps e_j) - S(x0)) / eps with the solver's own step(dt0).

Per eigenvalue mu: growth rate lambda = ln|mu| / dt0, frequency arg(mu)/(2 pi dt0).
Printed: the eigenvalues with the largest lambda, their frequency, and where the
eigenvector lives (phi part by column, u part by column; fraction in the last
4 columns / axis quarter).

Usage: python scripts/diag_pinned_jacobian.py CASE THETA XI DX_MM [EPS_SCALE] [PRE_T]
       PRE_T > 0: run the solver PRE_T seconds first and linearize about that state.
       CASE as in diag_pinned_energy.py (F2, F3, I1..I4, W4).
Writes results/validation_pinned_phase/jacobian/J_<case>_th<theta>_xi<xi>_dx<dx>.npz
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
from diag_pinned_energy import make  # noqa: E402

BAND = 6.0
EPS_PHI = 1e-7        # x dx
EPS_U = 1e-9          # m/s


def main(case, theta, xi, dx_mm, eps_scale=1.0, pre_t=0.0):
    global EPS_PHI, EPS_U
    EPS_PHI, EPS_U = EPS_PHI * eps_scale, EPS_U * eps_scale
    s, ref, frozen = make(case, theta, xi, dx_mm)
    assert not frozen
    g, f = s.grid, s.fields
    while f.t < pre_t:                         # relax toward the discrete equilibrium first
        s.step()
    dt0 = s.stable_timestep(f.u_r, f.u_z)
    geom = classify(f.phi)
    m_phi = np.abs(f.phi) < BAND * g.dr
    m_ur = geom.ur_face_known().copy()
    m_uz = geom.uz_face_known().copy()
    m_ur[0, :] = m_ur[-1, :] = False          # axis / wall: Dirichlet 0
    m_uz[:, 0] = False                         # bottom
    n = (int(m_phi.sum()), int(m_ur.sum()), int(m_uz.sum()))
    N = sum(n)
    base = {k: getattr(f, k).copy() for k in ("phi", "u_r", "u_z", "u_theta", "p")}
    t0, st0 = f.t, f.step

    def restore(dphi=None, dur=None, duz=None):
        for k, v in base.items():
            setattr(f, k, v.copy())
        f.t, f.step = t0, st0
        if dphi is not None:
            f.phi[m_phi] += dphi
        if dur is not None:
            f.u_r[m_ur] += dur
        if duz is not None:
            f.u_z[m_uz] += duz

    def out():
        return np.r_[f.phi[m_phi], f.u_r[m_ur], f.u_z[m_uz]]

    restore()
    s.step(dt0)
    y0 = out()
    J = np.empty((N, N))
    tic = time.time()
    for j in range(N):
        e = np.zeros(N)
        if j < n[0]:
            eps = EPS_PHI * g.dr
            e[j] = eps
        else:
            eps = EPS_U
            e[j] = eps
        restore(e[:n[0]], e[n[0]:n[0] + n[1]], e[n[0] + n[1]:])
        s.step(dt0)
        J[:, j] = (out() - y0) / eps
    mu, vec = np.linalg.eig(J)
    lam = np.log(np.abs(mu)) / dt0
    freq = np.angle(mu) / (2 * np.pi * dt0)
    order = np.argsort(lam)[::-1]
    # column index of every state entry
    col_phi = np.nonzero(m_phi)[0]
    col_ur = np.clip(np.nonzero(m_ur)[0], 0, g.Nr - 1)
    col_uz = np.nonzero(m_uz)[0]
    cols = np.r_[col_phi, col_ur, col_uz]
    is_phi = np.r_[np.ones(n[0], bool), np.zeros(n[1] + n[2], bool)]
    rows = []
    seen = set()
    for k in order:
        key = (round(lam[k], 3), round(abs(freq[k]), 1))
        if key in seen:                        # complex-conjugate partner
            continue
        seen.add(key)
        v = vec[:, k]
        vp = np.abs(v[is_phi]) / (EPS_PHI * g.dr)   # scale phi and u parts comparably (per eps)
        vu = np.abs(v[~is_phi]) / EPS_U
        wp = np.bincount(cols[is_phi], vp**2, minlength=g.Nr)
        wu = np.bincount(cols[~is_phi], vu**2, minlength=g.Nr)
        rows.append({"lambda": float(lam[k]), "freq_Hz": float(abs(freq[k])),
                     "phi_wall4_frac": float(wp[-4:].sum() / wp.sum()),
                     "phi_axis_frac": float(wp[:g.Nr // 4].sum() / wp.sum()),
                     "phi_peak_col": int(np.argmax(wp)),
                     "u_wall4_frac": float(wu[-4:].sum() / wu.sum()),
                     "u_peak_col": int(np.argmax(wu))})
        if len(rows) >= 8:
            break
    osc = np.abs(freq) > 1.0
    ko = int(np.nonzero(osc)[0][np.argmax(lam[osc])]) if osc.any() else None
    res = {"eps_scale": eps_scale, "pre_t": pre_t, "case": case, "theta": theta,
           "least_damped_osc": None if ko is None else {"lambda": float(lam[ko]), "freq_Hz": float(abs(freq[ko]))}, "xi": xi, "dx_mm": dx_mm, "dt0": dt0, "N": N, "n_blocks": n,
           "fd_seconds": round(time.time() - tic, 1), "n_unstable": int(np.sum(lam > 1e-3)),
           "max_lambda": float(lam.max()), "top": rows}
    print(json.dumps(res), flush=True)
    od = ROOT / "results" / "validation_pinned_phase" / "jacobian"
    od.mkdir(parents=True, exist_ok=True)
    sfx = ("" if eps_scale == 1.0 else f"_eps{eps_scale:g}") + ("" if pre_t == 0 else f"_pre{pre_t:g}")
    np.savez_compressed(od / f"J_{case.replace(':', '_')}_th{theta:g}_xi{xi:g}_dx{dx_mm:g}{sfx}.npz", J=J, mu=mu, dt0=dt0,
                        m_phi=m_phi, m_ur=m_ur, m_uz=m_uz)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], float(a[1]), float(a[2]), float(a[3]), float(a[4]) if len(a) > 4 else 1.0,
         float(a[5]) if len(a) > 5 else 0.0)
