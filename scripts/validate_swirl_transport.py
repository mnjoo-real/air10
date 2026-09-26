"""V7-T manufactured swirl (angular-momentum) transport tests -- no free surface, no forcing, nu = 0.

(1) 2-D axisymmetric: q = r u_theta advected by a prescribed, DISCRETELY divergence-free
    meridional flow from a corner streamfunction
        psi = A r^2 (1 - r/R)^2 sin^2(pi z / Z)   (z < Z; zero above),
        r_f u_r = -d psi/dz (face difference),  r_c u_z = d psi/dr (face difference),
    max |u| = 0.3 m/s, R = 45 mm, Z = 50 mm. Exact solution q(x, T) = q0(X(x, -T)) by RK4
    back-characteristics of the continuous flow. Profiles: "gauss" (w 4 mm at (22, 15) mm),
    "mode" (r/R)^2 (1 + 0.5 cos(pi r/R) cos(pi z/Z)), "solid" Omega r^2. T = 0.1 s (<= 30 mm travel).
    Metrics: L_z drift, r-weighted relative L2 error, peak loss, centroid (phase) error, order.
(2) 1-D periodic numerical viscosity: q = sin(k z), uniform u = 0.3 m/s, same face-value code and
    time integrator, Courant number 0.3 (as in the production runs); nu_num from the decay of the
    Fourier amplitude, a(t) = a0 exp(-nu_num k^2 t).
Prints JSON lines (and markdown tables at the end).
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from air_vortex.config import load_config  # noqa: E402
from air_vortex.grid import build_grid  # noqa: E402
from air_vortex.swirl_transport import advect_q, face_values  # noqa: E402

R, Z, U0, T = 0.045, 0.050, 0.3, 0.1
NU_WATER = 1.0e-6
METHODS = (("upwind1", 1, "vanleer"), ("muscl2_vanleer", 2, "vanleer"), ("muscl2_mc", 2, "mc"),
           ("muscl2_unlimited", 2, "none"))


def grid_for(dx):
    cfg = copy.deepcopy(load_config(ROOT / "configs" / "baseline.yaml"))
    cfg.grid.dr_m = cfg.grid.dz_m = dx
    return build_grid(cfg)


def psi_raw(r, z):
    zz = np.clip(z, 0.0, Z)
    return r**2 * (1 - r / R) ** 2 * np.sin(np.pi * zz / Z) ** 2


def vel_raw(r, z):
    """continuous (u_r, u_z) for A = 1."""
    inside = z < Z
    s2 = np.sin(np.pi * np.clip(z, 0, Z) / Z) ** 2
    ds2 = (np.pi / Z) * np.sin(2 * np.pi * np.clip(z, 0, Z) / Z)
    ur = -r * (1 - r / R) ** 2 * ds2
    uz = (2 * (1 - r / R) ** 2 - 2 * r * (1 - r / R) / R) * s2
    return np.where(inside, ur, 0.0), np.where(inside, uz, 0.0)


# amplitude so that max |u| = U0
_rr, _zz = np.meshgrid(np.linspace(0, R, 901), np.linspace(0, Z, 1001), indexing="ij")
_a, _b = vel_raw(_rr, _zz)
A = U0 / np.sqrt(_a**2 + _b**2).max()


def mac_velocity(g):
    rf, zf = g.r_f, g.z_f
    P = A * psi_raw(rf[:, None], zf[None, :])                      # corners (Nr+1, Nz+1)
    ur = np.zeros((g.Nr + 1, g.Nz))
    ur[1:] = -(P[1:, 1:] - P[1:, :-1]) / g.dz / rf[1:, None]
    uz = (P[1:, :] - P[:-1, :]) / (g.r_c[:, None] * g.dr)
    return ur, uz


def backtrace(r, z, t):
    n = max(200, int(t / 2e-4))
    h = -t / n
    for _ in range(n):
        def f(r_, z_):
            a, b = vel_raw(np.clip(r_, 0, R), z_)
            return A * a, A * b
        k1 = f(r, z); k2 = f(r + 0.5 * h * k1[0], z + 0.5 * h * k1[1])
        k3 = f(r + 0.5 * h * k2[0], z + 0.5 * h * k2[1]); k4 = f(r + h * k3[0], z + h * k3[1])
        r = r + h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        z = z + h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
    return np.clip(r, 0, R), z


PROFILES = {
    "gauss": lambda r, z: np.exp(-((r - 0.022) ** 2 + (z - 0.015) ** 2) / 0.004**2),
    "mode": lambda r, z: (r / R) ** 2 * (1 + 0.5 * np.cos(np.pi * r / R) * np.cos(np.pi * np.clip(z, 0, Z) / Z)),
    "solid": lambda r, z: 31.4159 * r**2,
}


def run2d(dx, method, order, limiter, prof):
    g = grid_for(dx)
    r2, z2 = np.meshgrid(g.r_c, g.z_c, indexing="ij")
    ur, uz = mac_velocity(g)
    rate = np.abs(ur).max() / g.dr + np.abs(uz).max() / g.dz
    nstep = int(np.ceil(T / (0.4 / rate)))
    dt = T / nstep
    q0 = PROFILES[prof](r2, z2)
    q = q0.copy()
    w = r2 * g.dr * g.dz
    L0 = np.sum(q * w)
    for _ in range(nstep):
        q = advect_q(g, q, ur, uz, dt, order, limiter)
    rb, zb = backtrace(r2, z2, T)
    qx = PROFILES[prof](rb, zb)
    out = {"test": "2d", "profile": prof, "method": method, "dx_mm": dx * 1e3, "steps": nstep,
           "courant_sum": rate * dt,
           "Lz_err": float(np.sum(q * w) / L0 - 1),
           "L2": float(np.sqrt(np.sum((q - qx) ** 2 * w) / np.sum(qx**2 * w))),
           "Linf": float(np.abs(q - qx).max() / np.abs(qx).max())}
    if prof == "gauss":
        out["peak_loss"] = float(1 - q.max() / qx.max())
        cn = [np.sum(q * w * c) / np.sum(q * w) for c in (r2, z2)]
        cx = [np.sum(qx * w * c) / np.sum(qx * w) for c in (r2, z2)]
        out["centroid_err_mm"] = float(np.hypot(cn[0] - cx[0], cn[1] - cx[1]) * 1e3)
        out["min_q"] = float(q.min())
    return out


def nu_num_1d(dx, order, limiter, lam, U=0.3, courant=0.3, t_end=None):
    """1-D periodic sine mode, same face_values + integrator; returns fitted nu_num."""
    n = int(round(lam / dx))
    z = (np.arange(n) + 0.5) * dx
    k = 2 * np.pi / lam
    q = np.sin(k * z)
    dt = courant * dx / U
    t_end = t_end or 0.5
    nstep = int(np.ceil(t_end / dt))
    vel = np.full(n + 1, U)

    def D(q):
        qp = np.concatenate([q[-2:], q, q[:2]])
        qf = face_values(qp, vel, order, limiter, 0)
        return U * (qf[1:] - qf[:-1]) / dx

    ts, amps = [], []
    for s in range(nstep + 1):
        if s % max(1, nstep // 50) == 0:
            c = np.fft.rfft(q)[1] * 2 / n
            ts.append(s * dt); amps.append(abs(c))
        if s == nstep:
            break
        if order == 1:
            q = q - dt * D(q)
        else:
            q1 = q - dt * D(q)
            q = 0.5 * q + 0.5 * (q1 - dt * D(q1))
    ts, amps = np.array(ts), np.array(amps)
    slope = np.polyfit(ts, np.log(amps), 1)[0]
    return -slope / k**2, float(amps[-1] / amps[0]), t_end


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    dxs = (1.0e-3, 0.5e-3, 0.25e-3, 0.125e-3)
    rows = []
    if what in ("all", "2d"):
        for prof in PROFILES:
            for m, o, lim in METHODS:
                for dx in dxs:
                    row = run2d(dx, m, o, lim, prof)
                    rows.append(row)
                    print(json.dumps(row), flush=True)
    if what in ("all", "nu"):
        for lam in (5e-3, 10e-3, 20e-3):
            for m, o, lim in METHODS:
                for dx in (1.0e-3, 0.5e-3, 0.25e-3):
                    nu, ratio, te = nu_num_1d(dx, o, lim, lam)
                    row = {"test": "nu_num", "lambda_mm": lam * 1e3, "method": m, "dx_mm": dx * 1e3,
                           "nu_num": nu, "nu_num_over_nu": nu / NU_WATER, "amp_ratio": ratio, "t_end": te,
                           "upwind_theory": 0.5 * 0.3 * dx * (1 - 0.3) if o == 1 else None}
                    print(json.dumps(row), flush=True)
