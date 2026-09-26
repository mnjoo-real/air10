"""DIAGNOSTIC ONLY (V4b-P mechanism pass): frozen-interface separation and a
discrete mechanical-energy budget for the unforced pinned meniscus.

Nothing here changes production numerics. The solver is driven through its
own step(); this process only (a) wraps project_liquid_velocity to capture
u* and u^{n+1}, (b) uses the validation curvature_fn hook, (c) restores phi
after each step for the frozen-interface cases.

Usage:
  python scripts/diag_pinned_energy.py run  CASE THETA XI DX_MM T_END [WIN_DT] [HR_T0]
      CASE: F0  phi frozen, p_Gamma = sigma*kappa_ref(r) at the (fixed) crossings
            F1  phi frozen, numerical kappa computed once at t=0 and frozen
            F2  phi advected (pinned ghost active), kappa = kappa_ref(r_crossing),
                i.e. no curvature-shape feedback
            F3  fully coupled production path (reconstruct_ghost, cubic, 4 columns)
      W4  as F2 but NUMERICAL kappa in the 4 wall columns (reference elsewhere)
      I<n> numerical kappa everywhere EXCEPT the last n wall columns, which get
           the reference kappa(r) (I1 = B, I4 = B4)
      R<n> production + RS2 (contour-preserving) reinitialization every n steps
      X2  production, but phi TRANSPORT with a two-sided normal extension of the interface
          velocity (diagnostic of liquid-side level-set shear)
      SA  same geometry/IC with the static_angle wall instead of the pinned ghost
      G:<fit>:<order>:<band>:<blend>[:SA]  wall_curvature="graph" candidate
      All from the wall-compatible IC (pinned_phase ic="extended"), reinit OFF.
      WIN_DT: energy-window length (default 0.02 s). HR_T0: if given, per-step
      samples of eta(axis), eta(wall band) and U are written for t >= HR_T0.
  python scripts/diag_pinned_energy.py signcheck THETA XI DX_MM
      Prescribed, discretely divergence-free stream-function field on the pinned
      meniscus: compares sigma*dA/dt (graph kinematics) with -W_gamma
      (mixed-face pressure power) to fix the sign convention.

Energy bookkeeping (all per step, then summed over windows):
  K        = 1/2 rho sum w |u|^2, face weights w = face control volume x mean
             sub-cell liquid fraction of the two adjacent cells (same fraction
             as liquid_volume_subcell); u_theta at centres (0 here).
  Eg, Es, V from the interface graph eta(r) (free_surface_height, the same
             linear phi=0 crossing as all diagnostics), with the pinned point
             (R, z_pin) and the axis point (0, eta_0):
             V  = sum a_i eta_i, Eg = rho g sum a_i eta_i^2/2, a_i = ring area,
             Es = sigma * sum pi (r_a + r_b) |segment|.
  F        = Eg + Es - P0 V  (P0 = BVP pressure constant): stationary at the
             equilibrium for ANY volume change, so level-set volume drift does not
             masquerade as energy.
  P_visc   = rho nu sum w u.L(u) with the solver's laplacian_ur / laplacian_uz on
             the same extended velocity the predictor uses; D_mu = -P_visc.
  W_g      = rho g sum w_z u_z  (rate at which the flow raises Eg).
  W_gamma  = - sum_mixed faces p_Gamma * (outward volume flux of u^{n+1}),
             the pressure power of the free-surface Dirichlet data.
  Residuals (J per window):
    R_pred = dK_pred - dt P_visc + dt W_g        (advection / time error)
    R_proj = dK_proj - dt W_gamma                (projection vs boundary work)
    R_grav = dEg - dt W_g
    R_cap  = dEs + dt W_gamma                    (capillary consistency)
    R_vol  = -P0 dV
    dK_ext = K(next step's weights) - K(this step's weights)
    R_E    = dK + dEg + dEs - P0 dV + dt D_mu = sum of the above.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

import air_vortex.single_phase_solver as sps  # noqa: E402
from air_vortex.curvature_single_phase import crossing_positions, interface_curvature  # noqa: E402
from air_vortex.diagnostics import free_surface_height  # noqa: E402
from air_vortex.liquid_mask import FACE_LIQ_MINUS, FACE_LIQ_PLUS, classify  # noqa: E402
from air_vortex.meniscus import G, RHO, SIGMA  # noqa: E402
from air_vortex.operators import center_grad_r, center_grad_z, laplacian_ur, laplacian_uz  # noqa: E402
from air_vortex.pinned_phase import R_V, build_phase_solver  # noqa: E402
from air_vortex.velocity_extension import extend_velocity  # noqa: E402


# ---------------------------------------------------------------- geometry

def cell_fraction(g, phi):
    gr, gz = center_grad_r(g, phi), center_grad_z(g, phi)
    mag = np.sqrt(gr**2 + gz**2) + 1e-300
    L_n = (np.abs(gr) * g.dr + np.abs(gz) * g.dz) / mag
    return np.clip(0.5 - (phi / mag) / L_n, 0.0, 1.0)


def face_weights(g, phi, geom=None):
    """Face control volume x mean sub-cell liquid fraction, restricted to the
    faces the solver actually solves (liquid-liquid and mixed, i.e.
    ur_face_known / uz_face_known). Void-void faces are excluded: the
    projection returns them unchanged from u* (they still carry the -g dt
    increment) until velocity extension overwrites them."""
    geom = classify(phi) if geom is None else geom
    fr = cell_fraction(g, phi)
    dv = 2 * np.pi * g.dr * g.dz
    wr = np.zeros((g.Nr + 1, g.Nz))
    wr[1:g.Nr] = dv * g.r_f[1:g.Nr, None] * 0.5 * (fr[:-1] + fr[1:])
    wz = np.zeros((g.Nr, g.Nz + 1))
    wz[:, 1:g.Nz] = dv * g.r_c[:, None] * 0.5 * (fr[:, :-1] + fr[:, 1:])
    wz[:, g.Nz] = dv * g.r_c * 0.5 * fr[:, -1]
    wc = dv * g.r_c[:, None] * fr * geom.liquid
    return wr * geom.ur_face_known(), wz * geom.uz_face_known(), wc


def kinetic(w, u_r, u_z, u_t):
    wr, wz, wc = w
    return 0.5 * RHO * float(np.sum(wr * u_r**2) + np.sum(wz * u_z**2) + np.sum(wc * u_t**2))


def graph_energy(g, eta, z_pin):
    """(V, Eg, Es, per-column Es) from the graph eta(r_c) + (0, eta_0) + (R, z_pin)."""
    a = np.pi * (g.r_f[1:] ** 2 - g.r_f[:-1] ** 2)
    V = float(np.sum(a * eta))
    Eg = RHO * G * float(np.sum(a * eta**2 / 2))
    r = np.r_[0.0, g.r_c, R_V]
    z = np.r_[eta[0], eta, z_pin]
    seg = np.pi * (r[:-1] + r[1:]) * np.hypot(np.diff(r), np.diff(z))
    col = np.zeros(g.Nr)
    col[0] += seg[0]                     # axis -> column 0
    col[1:] += seg[1:g.Nr]               # column i-1 -> i assigned to i
    col[-1] += seg[g.Nr]                 # last column -> wall
    col *= SIGMA
    return V, Eg, float(col.sum()), col


def gamma_power(g, geom, u_r, u_z, pgr, pgz):
    """Per-column pressure power -p_Gamma * outward flux on mixed faces, and the
    per-column outward flux."""
    P = np.zeros(g.Nr)
    Q = np.zeros(g.Nr)
    Az = 2 * np.pi * g.r_c * g.dr
    kz = geom.face_kind_z
    for kind, sgn, col_of in ((FACE_LIQ_MINUS, 1.0, None), (FACE_LIQ_PLUS, -1.0, None)):
        i, j = np.nonzero(kz == kind)
        q = sgn * u_z[i, j] * Az[i]
        np.add.at(P, i, -pgz[i, j] * q)
        np.add.at(Q, i, q)
    kr = geom.face_kind_r
    for kind, sgn, off in ((FACE_LIQ_MINUS, 1.0, -1), (FACE_LIQ_PLUS, -1.0, 0)):
        i, j = np.nonzero(kr == kind)
        q = sgn * u_r[i, j] * 2 * np.pi * g.r_f[i] * g.dz
        np.add.at(P, i + off, -pgr[i, j] * q)
        np.add.at(Q, i + off, q)
    return P, Q


def regions(Nr):
    return {"wall1": list(range(Nr - 1, Nr)), "wall2": list(range(Nr - 2, Nr)),
            "wall4": list(range(Nr - 4, Nr)), "mid": list(range(Nr // 4, Nr - 4)),
            "axis": list(range(0, Nr // 4))}


def kappa_ref_r(ref, r):
    return ref.kappa_at_height(ref.eta(np.clip(r, 0.0, R_V)))


# ---------------------------------------------------------------- cases

def make(case, theta, xi, dx_mm):
    s, ref = build_phase_solver(xi, dx_mm * 1e-3, "reconstruct_ghost", "cubic", theta_ref=theta,
                                ic="extended")
    s.cfg.wall.pinned_fit_columns = 4
    g = s.grid
    frozen_phi = case in ("F0", "F1")
    if case == "X2":
        # DIAGNOSTIC: level-set TRANSPORT uses the interface velocity extended along the
        # normal on BOTH sides (seed = faces within 1 cell of phi=0), so the liquid-side
        # level sets are not sheared by the bulk flow (Adalsteinsson-Sethian idea).
        # Momentum, pressure, curvature, and the extension used by momentum unchanged.
        import air_vortex.single_phase_solver as _sps
        from air_vortex.operators import interp_center_to_ur, interp_center_to_uz
        from air_vortex.velocity_extension import center_normals, extend_field
        orig_adv = _sps.advect_level_set_advective
        nl = s.cfg.physics.extension_layers_capillary

        def two_sided(phi, u_r, u_z, grid, dt, **kw):
            nrc, nzc = center_normals(grid, phi)
            out = []
            for u, interp in ((u_r, interp_center_to_ur), (u_z, interp_center_to_uz)):
                nr, nz = interp(nrc), interp(nzc)
                mag = np.sqrt(nr**2 + nz**2); mag = np.where(mag > 0, mag, 1.0)
                nr, nz = nr / mag, nz / mag
                pf = interp(phi)
                seed = np.abs(pf) < 1.0 * grid.dr
                up, _ = extend_field(u, seed, nr, nz, grid.dr, grid.dz, nl)       # into the void
                dn, _ = extend_field(u, seed, -nr, -nz, grid.dr, grid.dz, nl)     # into the liquid
                q = np.where(seed, u, np.where(pf > 0, up, np.where(pf > -nl * grid.dr, dn, u)))
                out.append(q)
            out[0][0, :] = 0.0; out[0][-1, :] = 0.0; out[1][:, 0] = 0.0
            return orig_adv(phi, out[0], out[1], grid, dt, **kw)
        _sps.advect_level_set_advective = two_sided
        return s, ref, False
    if case[0] == "R" and case[1:].isdigit():
        # production path + the EXISTING contour-preserving RS2 reinitialization every n steps
        # (off-contour phi DOFs damped); everything else unchanged
        s.cfg.levelset.reinitialize_every = int(case[1:])
        s.cfg.levelset.reinitialization_method = "russo_smereka_subcell"
        s.cfg.levelset.reinitialization_trigger = "periodic"
        return s, ref, False
    if case.startswith("G:"):
        # production candidate: wall_curvature = "graph" (src/air_vortex/wall_curvature.py)
        # G:<fit>:<order>:<band>:<blend>[:SA][:W<window>]  (window default order + 1)
        from air_vortex.config import WallConfig
        p = case.split(":")
        if len(p) > 5 and p[5] == "SA":
            s.cfg.wall = WallConfig("static_angle", float(ref.theta_deg))
        w = s.cfg.wall
        w.wall_curvature, w.wall_curvature_fit = "graph", p[1]
        w.wall_curvature_order, w.wall_curvature_band, w.wall_curvature_blend = int(p[2]), int(p[3]), int(p[4])
        w.wall_curvature_window = next((int(q[1:]) for q in p[5:] if q.startswith("W")), int(p[2]) + 1)
        return s, ref, False
    if case in ("F0", "F2"):
        def curv(grid, geom, phi):
            kr, kz = interface_curvature(grid, geom, phi, s.cfg.wall)   # only for the mask
            rr, _, rz, _ = crossing_positions(grid, geom)
            m = np.isfinite(kr)
            kr[m] = kappa_ref_r(ref, rr[m])
            m = np.isfinite(kz)
            kz[m] = kappa_ref_r(ref, rz[m])
            return kr, kz
        s.curvature_fn = curv
    elif case == "F1":
        geom0 = classify(s.fields.phi)
        kr0, kz0 = interface_curvature(g, geom0, s.fields.phi, s.cfg.wall)
        s.curvature_fn = lambda grid, geom, phi: (kr0.copy(), kz0.copy())
    elif case in ("W4",) or (case[0] == "I" and case[1:].isdigit()):
        nw = 4 if case == "W4" else int(case[1:])
        wall = np.zeros(g.Nr + 1, bool)
        wall[g.Nr - nw:] = True         # columns N-nw..N-1; u_r faces N-nw..N-1 (N = wall face)

        def curv(grid, geom, phi):
            kr, kz = interface_curvature(grid, geom, phi, s.cfg.wall)
            rr, _, rz, _ = crossing_positions(grid, geom)
            sel_z = ~wall[:grid.Nr] if case[0] == "W" else wall[:grid.Nr]
            sel_r = ~wall if case[0] == "W" else wall
            m = np.isfinite(kz) & sel_z[:, None]
            kz[m] = kappa_ref_r(ref, rz[m])
            m = np.isfinite(kr) & sel_r[:, None]
            kr[m] = kappa_ref_r(ref, rr[m])
            return kr, kz
        s.curvature_fn = curv
    elif case == "SA":
        # same geometry and IC, but the V4b-S static_angle wall (theta = the BVP's
        # equilibrium angle) instead of the pinned ghost: is the defect pin-specific?
        from air_vortex.config import WallConfig
        s.cfg.wall = WallConfig("static_angle", float(ref.theta_deg))
    elif case != "F3":
        raise ValueError(case)
    return s, ref, frozen_phi


def run(case, theta, xi, dx_mm, t_end, win_dt=0.02, hr_t0=None):
    s, ref, frozen = make(case, theta, xi, dx_mm)
    g, f, cfg = s.grid, s.fields, s.cfg
    z_pin = cfg.wall.pinned_contact_height_m
    if z_pin is None:                      # static_angle: close the energy polyline at the BVP wall height
        z_pin = float(ref.z_wall)
    nu = cfg.fluid.water_viscosity / RHO
    nl = cfg.physics.extension_layers_capillary
    phi0 = f.phi.copy()
    eta_ref = ref.eta(g.r_c)
    reg = regions(g.Nr)

    cap = {}
    orig = sps.project_liquid_velocity

    def proj(grid, geom, urs, uzs, p, rho, dt, pgr, pgz):
        out = orig(grid, geom, urs, uzs, p, rho, dt, pgr, pgz)
        cap.update(urs=urs.copy(), uzs=uzs.copy(), urn=out[0].copy(), uzn=out[1].copy(),
                   pgr=pgr, pgz=pgz, geom=geom)
        return out
    sps.project_liquid_velocity = proj

    print(json.dumps({"mode": "run", "case": case, "theta": theta, "xi": xi, "dx_mm": dx_mm,
                      "frozen_phi": frozen, "Nr": g.Nr, "P0": ref.P0, "regions": {k: [v[0], v[-1]] for k, v in reg.items()}}),
          flush=True)
    keys = ["dK_pred", "dK_proj", "dK_ext", "dEg", "dEs", "P0dV", "dtD", "dtWg", "dtWgam", "dV"]
    acc = {k: 0.0 for k in keys}
    acc_reg = {k: {"dEs": 0.0, "dtWgam": 0.0} for k in reg}
    V0, Eg0, Es0, _ = graph_energy(g, free_surface_height(f.phi, g), z_pin)
    F_start = Eg0 + Es0 - ref.P0 * V0
    w = face_weights(g, f.phi)
    K_prev_new = None
    next_t = win_dt
    hr = []
    first = None
    while f.t < t_end - 1e-12:
        phi_n = f.phi.copy()
        geom_n = classify(phi_n)
        w = face_weights(g, phi_n, geom_n)
        K0 = kinetic(w, f.u_r, f.u_z, f.u_theta)
        if K_prev_new is not None:
            acc["dK_ext"] += K0 - K_prev_new
        ure, uze, ute, _ = extend_velocity(g, geom_n, phi_n, f.u_r, f.u_z, f.u_theta, nl)
        P_visc = RHO * nu * float(np.sum(w[0] * ure * laplacian_ur(g, ure)) + np.sum(w[1] * uze * laplacian_uz(g, uze)))
        W_g = RHO * G * float(np.sum(w[1] * uze))
        V1, Eg1, Es1, col1 = graph_energy(g, free_surface_height(phi_n, g), z_pin)
        d = s.step()
        if frozen:
            f.phi = phi0.copy()
        dt = d.dt
        Ks = kinetic(w, cap["urs"], cap["uzs"], f.u_theta)
        Kn = kinetic(w, cap["urn"], cap["uzn"], f.u_theta)
        Pcol, Qcol = gamma_power(g, cap["geom"], cap["urn"], cap["uzn"], cap["pgr"], cap["pgz"])
        V2, Eg2, Es2, col2 = graph_energy(g, free_surface_height(f.phi, g), z_pin)
        acc["dK_pred"] += Ks - K0
        acc["dK_proj"] += Kn - Ks
        acc["dEg"] += Eg2 - Eg1
        acc["dEs"] += Es2 - Es1
        acc["dV"] += V2 - V1
        acc["P0dV"] += ref.P0 * (V2 - V1)
        acc["dtD"] += -dt * P_visc
        acc["dtWg"] += dt * W_g
        acc["dtWgam"] += dt * float(Pcol.sum())
        for k, cols in reg.items():
            acc_reg[k]["dEs"] += float(np.sum(col2[cols] - col1[cols]))
            acc_reg[k]["dtWgam"] += dt * float(np.sum(Pcol[cols]))
        K_prev_new = Kn
        u = max(d.max_abs_ur_liquid, d.max_abs_uz_liquid)
        if first is None:
            ar, az = np.abs(cap["urn"]) / dt, np.abs(cap["uzn"]) / dt
            ar = np.where(cap["geom"].ur_face_known(), ar, 0)
            az = np.where(cap["geom"].uz_face_known(), az, 0)
            if ar.max() >= az.max():
                i, j = np.unravel_index(ar.argmax(), ar.shape); loc = ["ur", int(i), int(j), float(g.r_f[i]), float(g.z_c[j])]
            else:
                i, j = np.unravel_index(az.argmax(), az.shape); loc = ["uz", int(i), int(j), float(g.r_c[i]), float(g.z_f[j])]
            first = loc
            print(json.dumps({"first_residual": loc, "accel_max": float(max(ar.max(), az.max()))}), flush=True)
        if hr_t0 is not None and f.t >= hr_t0:
            eta = free_surface_height(f.phi, g)
            hr.append((f.t, float(eta[0] - eta_ref[0]), float(np.mean(eta[-4:] - eta_ref[-4:])), u))
        if not np.isfinite(u) or u > 5.0:
            print(json.dumps({"t": d.t, "blowup": True}), flush=True)
            break
        if f.t >= next_t - 1e-12:
            eta = free_surface_height(f.phi, g)
            Vn, Egn, Esn, _ = graph_energy(g, eta, z_pin)
            Fn = Egn + Esn - ref.P0 * Vn
            R = {
                "R_pred": acc["dK_pred"] + acc["dtD"] + acc["dtWg"],
                "R_proj": acc["dK_proj"] - acc["dtWgam"],
                "R_grav": acc["dEg"] - acc["dtWg"],
                "R_cap": acc["dEs"] + acc["dtWgam"],
                "R_vol": -acc["P0dV"],
                "dK_ext": acc["dK_ext"],
            }
            R["R_E"] = sum(R.values())
            rec = {"t": f.t, "U": u, "K": Kn, "F_minus_F0": Fn - F_start,
                   "e_rms_dx": float(np.sqrt(np.nanmean((eta - eta_ref) ** 2)) / g.dr),
                   "e_axis_dx": float((eta[0] - eta_ref[0]) / g.dr),
                   **{k: acc[k] for k in keys}, **R,
                   "Rcap_reg": {k: v["dEs"] + v["dtWgam"] for k, v in acc_reg.items()},
                   "Wgam_reg": {k: v["dtWgam"] for k, v in acc_reg.items()},
                   "dEs_reg": {k: v["dEs"] for k, v in acc_reg.items()}}
            print(json.dumps(rec), flush=True)
            acc = {k: 0.0 for k in keys}
            acc_reg = {k: {"dEs": 0.0, "dtWgam": 0.0} for k in reg}
            next_t += win_dt
    if hr:
        out = ROOT / "results" / "validation_pinned_phase" / "energy"
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / f"hr_{case}_th{theta:g}_xi{xi:g}_dx{dx_mm:g}.npy", np.array(hr))


# ---------------------------------------------------------------- sign check

def signcheck(theta, xi, dx_mm):
    s, ref, _ = make("F3", theta, xi, dx_mm)
    g = s.grid
    phi = s.fields.phi
    geom = classify(phi)
    z_pin = s.cfg.wall.pinned_contact_height_m
    pr_num, pz_num, _, _ = s.interface_pressure_bc(geom, phi)
    rr, zr, rz, zz = crossing_positions(g, geom)
    pr_ex = np.where(np.isfinite(pr_num), SIGMA * kappa_ref_r(ref, np.nan_to_num(rr)), np.nan)
    pz_ex = np.where(np.isfinite(pz_num), SIGMA * kappa_ref_r(ref, np.nan_to_num(rz)), np.nan)
    R = R_V
    out = []
    for name, gz_fn, dgz in (("psi=r^2(R^2-r^2)^2 z", lambda z: z, lambda z: 1.0 + 0 * z),
                             ("psi=r^2(R^2-r^2)^2 (z-z0)^2", lambda z: (z - 0.004) ** 2, lambda z: 2 * (z - 0.004))):
        A0 = 1e-3 / (R**5)
        fr = lambda r: r**2 * (R**2 - r**2) ** 2
        dfr = lambda r: 2 * r * (R**2 - r**2) ** 2 - 4 * r**3 * (R**2 - r**2)
        psi = A0 * fr(g.r_f)[:, None] * gz_fn(g.z_f)[None, :]          # corners
        u_r = -(psi[:, 1:] - psi[:, :-1]) / (np.where(g.r_f == 0, 1, g.r_f)[:, None] * g.dz)
        u_r[0] = 0.0
        u_z = (psi[1:, :] - psi[:-1, :]) / (g.r_c[:, None] * g.dr)
        # discrete divergence check on liquid cells
        div = ((g.r_f[1:, None] * u_r[1:] - g.r_f[:-1, None] * u_r[:-1]) / (g.r_c[:, None] * g.dr)
               + (u_z[:, 1:] - u_z[:, :-1]) / g.dz)
        res = {"field": name, "max_div": float(np.abs(div[geom.liquid]).max())}
        for lab, pgr, pgz in (("numerical_kappa", pr_num, pz_num), ("reference_kappa", pr_ex, pz_ex)):
            P, Q = gamma_power(g, geom, u_r, u_z, np.nan_to_num(pgr), np.nan_to_num(pgz))
            res[f"W_gamma[{lab}]"] = float(P.sum())
            res["sum_flux"] = float(Q.sum())
        # graph kinematics with the analytic field: eta_t = u_z - u_r eta_r
        eta = free_surface_height(phi, g)
        rpts = np.r_[0.0, g.r_c, R]
        zpts = np.r_[eta[0], eta, z_pin]
        ur_a = lambda r, z: -A0 * (fr(r) / np.where(r == 0, 1, r)) * dgz(z) * (r > 0)
        uz_a = lambda r, z: A0 * (dfr(r) / np.where(r == 0, 1, r)) * gz_fn(z) + (r == 0) * A0 * 2 * R**4 * gz_fn(z)
        slope = np.gradient(zpts, rpts)
        eta_t = uz_a(rpts, zpts) - ur_a(rpts, zpts) * slope
        eta_t[-1] = 0.0                                        # pinned point (u = 0 at the wall)

        def area(zp):
            return float(np.sum(np.pi * (rpts[:-1] + rpts[1:]) * np.hypot(np.diff(rpts), np.diff(zp))))
        h = 1e-6
        dA = (area(zpts + h * eta_t) - area(zpts - h * eta_t)) / (2 * h)
        a = np.pi * (g.r_f[1:] ** 2 - g.r_f[:-1] ** 2)
        res["sigma_dA_dt"] = SIGMA * dA
        res["dEg_dt_graph"] = RHO * G * float(np.sum(a * eta * eta_t[1:-1]))
        res["dV_dt_graph"] = float(np.sum(a * eta_t[1:-1]))
        w = face_weights(g, phi)
        res["W_g_volume"] = RHO * G * float(np.sum(w[1] * u_z))
        out.append(res)
        print(json.dumps(res), flush=True)


if __name__ == "__main__":
    mode, a = sys.argv[1], sys.argv[2:]
    if mode == "run":
        run(a[0], float(a[1]), float(a[2]), float(a[3]), float(a[4]),
            float(a[5]) if len(a) > 5 else 0.02, float(a[6]) if len(a) > 6 else None)
    elif mode == "signcheck":
        signcheck(float(a[0]), float(a[1]), float(a[2]))
    else:
        raise SystemExit(__doc__)
