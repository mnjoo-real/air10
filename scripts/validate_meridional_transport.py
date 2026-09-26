"""V7-T section 11/13: how dissipative is the CURRENT meridional (u_r, u_z) momentum advection?

(a) Operator test (no projection): the MAC velocity of the smooth, discretely divergence-free
    streamfunction flow of validate_swirl_transport.py (max |u| 0.3 m/s, box R 45 x Z 50 mm) is
    fed to the predictor's advection expressions of SinglePhaseHeightSolver.step
    (first-order upwind, advective form). Compared with the analytic (u.grad)u at the faces:
    r-weighted relative L2 error in the interior (>= 3 cells from axis, wall, bottom and z = Z),
    near the axis (r < 3 dx) and near the side wall (R - r < 3 dx); observed order.
    Kinetic-energy production of the operator, P = -sum rho u . N(u) dV (zero for the exact
    operator with no-penetration walls), and nu_eff = -P / (rho int |omega|^2 dV), the
    viscosity that would dissipate the same power.
(b) Evolution test (with projection): the same flow as the initial condition of the height
    solver with a rigid free-slip lid at z = Z, rpm = 0, nu = 0, u_theta = 0, 0.2 s.
    Kinetic energy must stay constant; nu_eff(t) = -(dE/dt) / (rho int omega^2 dV).
Prints JSON lines.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

from air_vortex.operators import upwind_derivative, ur_at_uz_locations, uz_at_ur_locations  # noqa: E402
from validate_swirl_transport import A, R, Z, grid_for, mac_velocity, vel_raw  # noqa: E402

NU_WATER = 1e-6
RHO = 1000.0


def u_exact(r, z):
    a, b = vel_raw(r, z)
    return A * a, A * b


def conv_exact(r, z, h=1e-7):
    ur, uz = u_exact(r, z)
    dur_dr = (u_exact(r + h, z)[0] - u_exact(r - h, z)[0]) / (2 * h)
    dur_dz = (u_exact(r, z + h)[0] - u_exact(r, z - h)[0]) / (2 * h)
    duz_dr = (u_exact(r + h, z)[1] - u_exact(r - h, z)[1]) / (2 * h)
    duz_dz = (u_exact(r, z + h)[1] - u_exact(r, z - h)[1]) / (2 * h)
    return ur * dur_dr + uz * dur_dz, ur * duz_dr + uz * duz_dz, dur_dz - duz_dr


def operator_test(dx, scheme="advective"):
    g = grid_for(dx)
    u_r, u_z = mac_velocity(g)
    if scheme != "advective":
        from air_vortex.meridional_transport import advection_terms, mass_fluxes
        order = 1 if scheme == "conservative_upwind1" else 2
        adv_r, adv_z = advection_terms(g, u_r, u_z, mass_fluxes(g, u_r, u_z), order, "vanleer")
        return _operator_metrics(g, dx, u_r, u_z, adv_r, adv_z, scheme)
    w_at_ur = uz_at_ur_locations(u_z)
    adv_r = u_r * upwind_derivative(u_r, u_r, g.dr, axis=0) + w_at_ur * upwind_derivative(u_r, w_at_ur, g.dz, axis=1)
    u_at_uz = ur_at_uz_locations(u_r)
    adv_z = u_at_uz * upwind_derivative(u_z, u_at_uz, g.dr, axis=0) + u_z * upwind_derivative(u_z, u_z, g.dz, axis=1)
    return _operator_metrics(g, dx, u_r, u_z, adv_r, adv_z, scheme)


def _operator_metrics(g, dx, u_r, u_z, adv_r, adv_z, scheme):
    rr, zr = np.meshgrid(g.r_f, g.z_c, indexing="ij")          # u_r faces
    rz, zz = np.meshgrid(g.r_c, g.z_f, indexing="ij")          # u_z faces
    ex_r = conv_exact(rr, zr)[0]
    ex_z = conv_exact(rz, zz)[1]
    Vr = np.maximum(rr, 1e-30) * g.dr * g.dz
    Vz = rz * g.dr * g.dz
    out = {"test": "operator", "scheme": scheme, "dx_mm": dx * 1e3}
    regions = {
        "interior": (lambda r, z: (r >= 3 * dx) & (r <= R - 3 * dx) & (z >= 3 * dx) & (z <= Z - 3 * dx)),
        "near_axis": (lambda r, z: (r < 3 * dx) & (z >= 3 * dx) & (z <= Z - 3 * dx)),
        "near_wall": (lambda r, z: (r > R - 3 * dx) & (z >= 3 * dx) & (z <= Z - 3 * dx)),
    }
    for name, sel in regions.items():
        mr, mz = sel(rr, zr) & (rr > 0) & (rr < R), sel(rz, zz)
        num = np.sum(((adv_r - ex_r) ** 2 * Vr)[mr]) + np.sum(((adv_z - ex_z) ** 2 * Vz)[mz])
        den = np.sum((ex_r**2 * Vr)[mr]) + np.sum((ex_z**2 * Vz)[mz])
        out[f"L2_{name}"] = float(np.sqrt(num / den))
    # kinetic-energy production of the discrete operator (whole flow region)
    inside_r = (zr < Z) & (rr > 0) & (rr < R)
    inside_z = (zz > 0) & (zz < Z)
    P = -RHO * 2 * np.pi * (np.sum((u_r * adv_r * Vr)[inside_r]) + np.sum((u_z * adv_z * Vz)[inside_z]))
    P_ex = -RHO * 2 * np.pi * (np.sum((u_r * ex_r * Vr)[inside_r]) + np.sum((u_z * ex_z * Vz)[inside_z]))
    rc, zc = np.meshgrid(g.r_c, g.z_c, indexing="ij")
    om = conv_exact(rc, zc)[2]
    ens = RHO * 2 * np.pi * np.sum((om**2 * rc * g.dr * g.dz)[zc < Z])
    out.update({"KE_production_W": float(P), "KE_production_exact_op_W": float(P_ex),
                "nu_eff": float(-(P - P_ex) / ens), "nu_eff_over_nu": float(-(P - P_ex) / ens / NU_WATER)})
    return out


def swirl_operator_test(dx):
    """Same flow, same kind of measure for the swirl operators: q = r u_theta is materially
    conserved, so int q^2 dV is invariant under exact advection; nu_q = -(d/dt int q^2) /
    (2 int |grad q|^2) with the exact gradient. q = "mode" profile of validate_swirl_transport."""
    from validate_swirl_transport import PROFILES
    from air_vortex.swirl_transport import q_flux_divergence
    g = grid_for(dx)
    u_r, u_z = mac_velocity(g)
    rc, zc = np.meshgrid(g.r_c, g.z_c, indexing="ij")
    qf = PROFILES["mode"]
    q = qf(rc, zc)
    h = 1e-7
    gq2 = ((qf(rc + h, zc) - qf(rc - h, zc)) / (2 * h)) ** 2 + ((qf(rc, zc + h) - qf(rc, zc - h)) / (2 * h)) ** 2
    V = rc * g.dr * g.dz
    m = zc < Z
    out = {"test": "swirl_operator", "dx_mm": dx * 1e3}
    ut = q / rc
    u_r_c = 0.5 * (u_r[1:] + u_r[:-1]); u_z_c = 0.5 * (u_z[:, 1:] + u_z[:, :-1])
    legacy = rc * (u_r_c * upwind_derivative(ut, u_r_c, g.dr, 0) + u_z_c * upwind_derivative(ut, u_z_c, g.dz, 1)
                   + u_r_c * ut / rc)                      # dq/dt = -r (advective terms)
    for name, Dq in (("advective_legacy", legacy), ("upwind1", q_flux_divergence(g, q, u_r, u_z, 1)),
                     ("muscl2_vanleer", q_flux_divergence(g, q, u_r, u_z, 2, "vanleer"))):
        P = -2 * np.sum((q * Dq * V)[m])
        nu = -P / (2 * np.sum((gq2 * V)[m]))
        out[f"nu_q_{name}"] = float(nu)
        out[f"nu_q_{name}_over_nu"] = float(nu / NU_WATER)
    return out


def evolution_test(dx, T=0.2, visc_zero=True, scheme="advective"):
    from air_vortex.height_benchmarks import build_height_production
    from air_vortex.height_solver import SinglePhaseHeightSolver
    s0 = build_height_production(0.0, dx=dx)
    cfg = s0.cfg
    cfg.fluid.water_viscosity = 0.0 if visc_zero else cfg.fluid.water_viscosity
    s = SinglePhaseHeightSolver(grid=s0.grid, cfg=cfg, fields=s0.fields, eta=s0.eta, z_pin=s0.z_pin,
                                meridional_advection=scheme)
    s.rigid_lid = s.freeze_interface = True
    g = s.grid
    u_r, u_z = mac_velocity(g)                     # Z = H = 50 mm: psi vanishes on the lid
    s.fields.u_r, s.fields.u_z = u_r.copy(), u_z.copy()
    s.fields.u_theta[:] = 0.0
    rr, zr = np.meshgrid(g.r_f, g.z_c, indexing="ij")
    rz, zz = np.meshgrid(g.r_c, g.z_f, indexing="ij")
    rc, zc = np.meshgrid(g.r_c, g.z_c, indexing="ij")

    def energy(f):
        mr = (zr < Z)
        mz = (zz < Z) & (zz > 0)
        return 0.5 * RHO * 2 * np.pi * g.dr * g.dz * (np.sum((f.u_r**2 * rr)[mr]) + np.sum((f.u_z**2 * rz)[mz]))

    def enstrophy(f):
        # omega at cell corners from MAC velocities
        w = (f.u_r[:, 1:] - f.u_r[:, :-1]) / g.dz          # d u_r/dz at (r_f, z_f interior)
        v = (f.u_z[1:, :] - f.u_z[:-1, :]) / g.dr          # d u_z/dr at (r_f interior, z_f)
        om = w[1:-1, :] - v[:, 1:-1]
        rco = g.r_f[1:-1, None] * np.ones_like(om)
        zco = g.z_f[None, 1:-1] * np.ones_like(om)
        return RHO * 2 * np.pi * g.dr * g.dz * np.sum((om**2 * rco)[zco < Z])

    E0 = energy(s.fields)
    rows, nxt = [], 0.0
    while s.fields.t < T - 1e-12:
        dt = min(s.stable_timestep(s.fields.u_r, s.fields.u_z), T - s.fields.t)
        s.step(dt)
        if s.fields.t >= nxt - 1e-12:
            rows.append((s.fields.t, energy(s.fields), enstrophy(s.fields)))
            nxt += 0.01
    t, E, W = map(np.array, zip(*rows))
    dEdt = np.gradient(E, t)
    nu_eff = -dEdt / W
    return {"test": "evolution", "scheme": scheme, "dx_mm": dx * 1e3, "T": T, "E0": E0, "E_T_over_E0": float(E[-1] / E0),
            "nu_eff_mean": float(np.mean(nu_eff[1:])), "nu_eff_over_nu": float(np.mean(nu_eff[1:]) / NU_WATER),
            "nu_eff_t": [float(x) for x in nu_eff[::4]]}


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("all", "operator"):
        for sch in ("advective", "conservative_upwind1", "conservative_muscl2"):
            for dx in (1.0e-3, 0.5e-3, 0.25e-3, 0.125e-3):
                print(json.dumps(operator_test(dx, sch)), flush=True)
    if what in ("all", "swirl"):
        for dx in (1.0e-3, 0.5e-3, 0.25e-3, 0.125e-3):
            print(json.dumps(swirl_operator_test(dx)), flush=True)
    if what in ("all", "evolution"):
        schemes = sys.argv[2:] or ["advective"]
        for sch in schemes:
            for dx in (1.0e-3, 0.5e-3, 0.25e-3):
                print(json.dumps(evolution_test(dx, scheme=sch)), flush=True)
