"""Jacobian gate of the single_phase_height branch: FD Jacobian of one fixed-dt step
on x = (eta (Nr column heights), u_r known faces, u_z known faces) -- NO level-set
values: off-contour modes cannot exist. For the least-stable modes: lambda, frequency,
and the relative volume content |sum_i A_i d eta_i| / sum_i A_i |d eta_i| (a growing
mode must not change the conserved discrete volume).

Usage: python scripts/diag_height_jacobian.py THETA XI DX_MM [PRE_T]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from air_vortex.height_benchmarks import build_height_meniscus  # noqa: E402
from air_vortex.height_interface import column_areas, geometry_from_eta  # noqa: E402


def main(theta, xi, dx_mm, pre_t=0.0):
    tic = time.time()
    s, ref = build_height_meniscus(xi, dx_mm * 1e-3, theta)
    g, f = s.grid, s.fields
    while f.t < pre_t:
        s.step()
    dt = s.stable_timestep(f.u_r, f.u_z)
    geom, _ = geometry_from_eta(g, s.interface())
    m_ur = geom.ur_face_known().copy(); m_ur[0, :] = m_ur[-1, :] = False
    m_uz = geom.uz_face_known().copy(); m_uz[:, 0] = False
    nh, nur, nuz = g.Nr, int(m_ur.sum()), int(m_uz.sum())
    N = nh + nur + nuz
    base = {k: getattr(f, k).copy() for k in ("u_r", "u_z", "u_theta", "p")}
    eta0, t0, st0 = s.eta.copy(), f.t, f.step

    def load(x=None):
        for k, v in base.items():
            setattr(f, k, v.copy())
        s.eta = eta0.copy(); f.t, f.step = t0, st0
        if x is not None:
            s.eta += x[:nh]; f.u_r[m_ur] += x[nh:nh + nur]; f.u_z[m_uz] += x[nh + nur:]
    out = lambda: np.r_[s.eta, f.u_r[m_ur], f.u_z[m_uz]]  # noqa: E731
    load(); s.step(dt); y0 = out()
    J = np.empty((N, N))
    eh, eu = 1e-7 * g.dr, 1e-9
    for j in range(N):
        e = np.zeros(N); eps = eh if j < nh else eu; e[j] = eps
        load(e); s.step(dt); J[:, j] = (out() - y0) / eps
    mu, V = np.linalg.eig(J)
    lam = np.log(np.abs(mu)) / dt; fr = np.abs(np.angle(mu)) / (2 * np.pi * dt)
    A = column_areas(g)
    rows, seen = [], set()
    for k in np.argsort(lam)[::-1]:
        key = (round(lam[k], 4), round(fr[k], 1))
        if key in seen:
            continue
        seen.add(key)
        dh = V[:nh, k]
        vol = float(abs(np.sum(A * dh)) / max(np.sum(A * np.abs(dh)), 1e-300))
        rows.append({"lambda": float(lam[k]), "freq_Hz": float(fr[k]), "rel_volume": vol,
                     "eta_share": float(np.linalg.norm(dh) / np.linalg.norm(V[:, k])),
                     "eta_peak_col": int(np.argmax(np.abs(dh)))})
        if len(rows) >= 6:
            break
    o = fr > 1.0
    ko = int(np.nonzero(o)[0][np.argmax(lam[o])]) if o.any() else None
    print(json.dumps({"theta": theta, "xi": xi, "dx_mm": dx_mm, "pre_t": pre_t, "dt": dt,
                      "state_dim": {"eta": nh, "u_r": nur, "u_z": nuz, "phi": 0, "total": N},
                      "max_lambda": float(lam.max()), "top": rows,
                      "least_damped_osc": None if ko is None else {"lambda": float(lam[ko]), "freq_Hz": float(fr[ko])},
                      "seconds": round(time.time() - tic, 1)}), flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(float(a[0]), float(a[1]), float(a[2]), float(a[3]) if len(a) > 3 else 0.0)
