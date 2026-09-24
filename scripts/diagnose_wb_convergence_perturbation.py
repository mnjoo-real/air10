"""DIAGNOSTIC (not a validated/gated script) -- Session 7, final validation of
the diagnostic well-balanced (WB) face-density construction:

  (1) does the WB equilibrium residual actually converge under grid
      refinement (vs. the current arithmetic-average face density)?
  (2) does WB preserve genuine non-equilibrium perturbation dynamics, or
      does it also suppress real physical secondary flow?

No production code is modified. WB remains a Psi-based, solid-body-specific
diagnostic construction throughout -- this script exists to understand the
numerical structure needed for a future production-compatible well-balanced
scheme, not to justify inserting this exact construction into pressure.py.

Usage:
    python scripts/diagnose_wb_convergence_perturbation.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.sparse as sp  # noqa: E402
import scipy.sparse.linalg as spla  # noqa: E402

from air_vortex.config import load_config  # noqa: E402
from air_vortex.grid import build_grid  # noqa: E402
from air_vortex.fields import initialize_rotating_equilibrium  # noqa: E402
from air_vortex.boundary import apply_velocity_bc  # noqa: E402
from air_vortex.diagnostics import volume_consistent_parabola  # noqa: E402
from air_vortex.operators import (  # noqa: E402
    interp_center_to_ur, interp_center_to_uz, interp_ur_to_center, interp_uz_to_center,
    laplacian_ur, laplacian_uz, u_theta_on_ur_faces, upwind_derivative,
    uz_at_ur_locations, ur_at_uz_locations, divergence, grad_p_to_ur_faces, grad_p_to_uz_faces,
)
from air_vortex.pressure import face_inv_rho_ur, face_inv_rho_uz, pressure_rhs  # noqa: E402
from air_vortex.properties import material_properties, interface_epsilon, smoothed_heaviside  # noqa: E402
from air_vortex.surface_tension import surface_tension_force  # noqa: E402

OUT = Path("results/diagnostics/wb_convergence_perturbation")
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def norms(a):
    a = a[np.isfinite(a)]
    return float(np.max(np.abs(a))), float(np.sqrt(np.mean(a ** 2)))


def discrete_curl(grid, S_r, S_z):
    Sr_c = interp_ur_to_center(S_r); Sz_c = interp_uz_to_center(S_z)
    curl = np.full(grid.shape_center, np.nan)
    curl[1:-1, 1:-1] = ((Sz_c[2:, 1:-1] - Sz_c[:-2, 1:-1]) / (2 * grid.dr)
                        - (Sr_c[1:-1, 2:] - Sr_c[1:-1, :-2]) / (2 * grid.dz))
    return curl


def build_matrix_facerho(grid, inv_rho_ur, inv_rho_uz, top_bc="open"):
    Nr, Nz = grid.Nr, grid.Nz
    dr, dz = grid.dr, grid.dz
    r_f = grid.r_f; r_c = grid.r_c[:, None]
    N = Nr * Nz
    c_west = np.zeros((Nr, Nz)); c_west[1:, :] = r_f[1:Nr, None] * inv_rho_ur[1:Nr, :] / (r_c[1:, :] * dr * dr)
    c_east = np.zeros((Nr, Nz)); c_east[:-1, :] = r_f[1:Nr, None] * inv_rho_ur[1:Nr, :] / (r_c[:-1, :] * dr * dr)
    c_south = np.zeros((Nr, Nz)); c_south[:, 1:] = inv_rho_uz[:, 1:Nz] / (dz * dz)
    c_north = np.zeros((Nr, Nz)); c_north[:, :-1] = inv_rho_uz[:, 1:Nz] / (dz * dz)
    c_top_ghost = np.zeros((Nr, Nz))
    if top_bc == "open":
        c_top_ghost[:, -1] = inv_rho_uz[:, Nz] / (0.5 * dz * dz)
    diag = -(c_west + c_east + c_south + c_north + c_top_ghost)
    A = sp.diags(diagonals=[c_west.ravel()[Nz:], c_south.ravel()[1:], diag.ravel(),
                             c_north.ravel()[:-1], c_east.ravel()[:-Nz]],
                 offsets=[-Nz, -1, 0, 1, Nz], shape=(N, N), format="csr")
    if top_bc == "closed":
        A = A.tolil(); A[0, :] = 0.0; A[0, 0] = 1.0; A = A.tocsr()
    return A


def P_facerho(grid, v_r, v_z, inv_rho_ur, inv_rho_uz, top_bc="open"):
    v_r = v_r.copy(); v_z = v_z.copy()
    v_r[0, :] = 0.0; v_r[-1, :] = 0.0
    v_z[:, 0] = 0.0; v_z[:, -1] = v_z[:, -2]
    A = build_matrix_facerho(grid, inv_rho_ur, inv_rho_uz, top_bc)
    b = pressure_rhs(grid, v_r, v_z, dt=1.0)
    if top_bc == "closed":
        b = b.copy(); b[0] = 0.0
    p = spla.spsolve(A.tocsc(), b, permc_spec="MMD_AT_PLUS_A").reshape(grid.Nr, grid.Nz)
    dpdr = grad_p_to_ur_faces(grid, p); dpdz = grad_p_to_uz_faces(grid, p)
    u_r = v_r - inv_rho_ur * dpdr; u_z = v_z - inv_rho_uz * dpdz
    u_r[0, :] = 0.0; u_r[-1, :] = 0.0; u_z[:, 0] = 0.0
    return u_r, u_z, p


def step_facerho(grid, cfg, u_r, u_z, u_theta, rho, mu, phi, dt, omega,
                  inv_rho_ur, inv_rho_uz, top_bc="open"):
    f_sigma_r, f_sigma_z = surface_tension_force(grid, phi, cfg)
    rho_ur = interp_center_to_ur(rho); rho_uz = interp_center_to_uz(rho)
    mu_ur = interp_center_to_ur(mu); mu_uz = interp_center_to_uz(mu)
    w_at_ur = uz_at_ur_locations(u_z)
    adv_r = u_r * upwind_derivative(u_r, u_r, grid.dr, axis=0) + w_at_ur * upwind_derivative(u_r, w_at_ur, grid.dz, axis=1)
    u_theta_ur = u_theta_on_ur_faces(grid, u_theta)
    r_f_safe = np.where(grid.r_f == 0.0, grid.dr, grid.r_f)
    centrifugal = u_theta_ur ** 2 / r_f_safe[:, None]
    visc_r = mu_ur * laplacian_ur(grid, u_r) / rho_ur
    u_r_star = u_r + dt * (-adv_r + centrifugal + visc_r + f_sigma_r / rho_ur)
    u_at_uz = ur_at_uz_locations(u_r)
    adv_z = u_at_uz * upwind_derivative(u_z, u_at_uz, grid.dr, axis=0) + u_z * upwind_derivative(u_z, u_z, grid.dz, axis=1)
    visc_z = mu_uz * laplacian_uz(grid, u_z) / rho_uz
    u_z_star = u_z + dt * (-adv_z - cfg.fluid.gravity + visc_z + f_sigma_z / rho_uz)
    u_theta_new = omega * grid.r_c[:, None] * np.ones_like(u_theta)
    u_r_star[0, :] = 0.0; u_r_star[-1, :] = 0.0
    u_z_star[:, 0] = 0.0; u_z_star[:, -1] = u_z_star[:, -2]
    A = build_matrix_facerho(grid, inv_rho_ur, inv_rho_uz, top_bc)
    b = pressure_rhs(grid, u_r_star, u_z_star, dt)
    p_new = spla.spsolve(A.tocsc(), b, permc_spec="MMD_AT_PLUS_A").reshape(grid.Nr, grid.Nz)
    dpdr = grad_p_to_ur_faces(grid, p_new); dpdz = grad_p_to_uz_faces(grid, p_new)
    u_r_new = u_r_star - dt * inv_rho_ur * dpdr
    u_z_new = u_z_star - dt * inv_rho_uz * dpdz
    u_r_new[0, :] = 0.0; u_r_new[-1, :] = 0.0; u_z_new[:, 0] = 0.0
    return u_r_new, u_z_new, u_theta_new, p_new


def run_multistep(grid, cfg, omega, dt, n_steps, phi0, rho0, mu0, inv_rho_ur, inv_rho_uz):
    u_r = np.zeros(grid.shape_ur); u_z = np.zeros(grid.shape_uz)
    u_theta = omega * grid.r_c[:, None] * np.ones(grid.shape_center)
    rows = []
    for step in range(1, n_steps + 1):
        u_r, u_z, u_theta, p = step_facerho(grid, cfg, u_r, u_z, u_theta, rho0, mu0, phi0, dt, omega,
                                             inv_rho_ur, inv_rho_uz)
        apply_velocity_bc(grid, type("F", (), {"u_r": u_r, "u_z": u_z, "u_theta": u_theta, "phi": phi0})())
        speed = np.sqrt(0.5 * (u_r[:-1, :] ** 2 + u_r[1:, :] ** 2) + 0.5 * (u_z[:, :-1] ** 2 + u_z[:, 1:] ** 2))
        ke = float(np.sum(0.5 * (speed ** 2) * (2 * np.pi * grid.r_c[:, None] * grid.dr * grid.dz)))
        rows.append({"step": step, "U_max": float(np.max(speed)), "K": ke,
                     "max_ur": float(np.max(np.abs(u_r))), "max_uz": float(np.max(np.abs(u_z))),
                     "max_div": float(np.max(np.abs(divergence(grid, u_r, u_z))))})
    return pd.DataFrame(rows)


def build_rho_face_WB(grid, omega, g, eta_r, eps0, rho_w, rho_a):
    Psi_surface = -g * float(eta_r[0])
    eps_Psi = g * eps0
    n_lut = 20001
    span = 0.5 * omega ** 2 * grid.r_v ** 2 + g * (grid.z_c[-1] - grid.z_c[0] + 10 * eps0) + abs(Psi_surface) + 1.0
    Psi_grid = np.linspace(-span, span, n_lut)
    rho_of_Psi = rho_w + (rho_a - rho_w) * smoothed_heaviside(Psi_grid - Psi_surface, eps_Psi)
    p_of_Psi = np.concatenate([[0.0], np.cumsum(0.5 * (rho_of_Psi[1:] + rho_of_Psi[:-1]) * np.diff(Psi_grid))])

    def p_lu(x):
        return np.interp(x, Psi_grid, p_of_Psi)

    def rho_lu(x):
        return np.interp(x, Psi_grid, rho_of_Psi)

    rho_face_WB_ur = np.zeros(grid.shape_ur)
    Psi_l = 0.5 * omega ** 2 * grid.r_c[:-1, None] ** 2 - g * grid.z_c[None, :]
    Psi_r = 0.5 * omega ** 2 * grid.r_c[1:, None] ** 2 - g * grid.z_c[None, :]
    dPsi = Psi_r - Psi_l
    small = np.abs(dPsi) < 1e-9
    rho_face_WB_ur[1:-1, :] = np.where(small, rho_lu(0.5 * (Psi_l + Psi_r)),
                                        (p_lu(Psi_r) - p_lu(Psi_l)) / np.where(small, 1.0, dPsi))

    rho_face_WB_uz = np.zeros(grid.shape_uz)
    Psi_b = 0.5 * omega ** 2 * grid.r_c[:, None] ** 2 - g * grid.z_c[None, :-1]
    Psi_t = 0.5 * omega ** 2 * grid.r_c[:, None] ** 2 - g * grid.z_c[None, 1:]
    dPsi_z = Psi_t - Psi_b
    small_z = np.abs(dPsi_z) < 1e-9
    rho_face_WB_uz[:, 1:-1] = np.where(small_z, rho_lu(0.5 * (Psi_b + Psi_t)),
                                        (p_lu(Psi_t) - p_lu(Psi_b)) / np.where(small_z, 1.0, dPsi_z))
    rho_face_WB_uz[:, -1] = rho_lu(0.5 * omega ** 2 * grid.r_c ** 2 - g * (grid.z_c[-1] + 0.5 * grid.dz))

    inv_ur = np.where(rho_face_WB_ur > 0, 1.0 / np.where(rho_face_WB_ur > 0, rho_face_WB_ur, 1.0), 0.0)
    inv_uz = np.where(rho_face_WB_uz > 0, 1.0 / np.where(rho_face_WB_uz > 0, rho_face_WB_uz, 1.0), 0.0)
    inv_ur[0, :] = 0.0; inv_ur[-1, :] = 0.0; inv_uz[:, 0] = 0.0
    return inv_ur, inv_uz


def setup_case(dx_mm, eps_mm):
    cfg = load_config("configs/validation_solid_body.yaml")
    cfg.fluid.surface_tension = 0.0
    cfg.grid.dr_m = dx_mm * 1e-3
    cfg.grid.dz_m = dx_mm * 1e-3
    cfg.levelset.interface_width_cells = eps_mm / dx_mm
    grid = build_grid(cfg)
    omega = 2 * np.pi * 100 / 60
    g = cfg.fluid.gravity
    rho_w, rho_a = cfg.fluid.water_density, cfg.fluid.air_density
    fields0 = initialize_rotating_equilibrium(grid, cfg, omega)
    phi0 = fields0.phi.copy()
    eta_r = volume_consistent_parabola(grid.r_c, grid.r_v, omega, g,
                                        np.pi * grid.r_v ** 2 * cfg.geometry.water_height_m)
    eps0 = interface_epsilon(grid, cfg)
    rho_current, mu_current = material_properties(phi0, grid, cfg)
    return cfg, grid, omega, g, rho_w, rho_a, phi0, eta_r, eps0, rho_current, mu_current


def main():
    dt = 2.0e-4

    # ============================================================
    # TABLE A: growth-window analysis, current vs WB
    # ============================================================
    print("=== Table A: growth-window analysis (current vs WB), 200 steps ===")
    cfg, grid, omega, g, rho_w, rho_a, phi0, eta_r, eps0, rho_current, mu_current = setup_case(1.5, 2.25)
    inv_ur_cur, inv_uz_cur = face_inv_rho_ur(rho_current), face_inv_rho_uz(rho_current)
    inv_ur_wb, inv_uz_wb = build_rho_face_WB(grid, omega, g, eta_r, eps0, rho_w, rho_a)

    df_cur = run_multistep(grid, cfg, omega, dt, 200, phi0, rho_current, mu_current, inv_ur_cur, inv_uz_cur)
    df_wb = run_multistep(grid, cfg, omega, dt, 200, phi0, rho_current, mu_current, inv_ur_wb, inv_uz_wb)
    df_cur.to_csv(OUT / "growth_current.csv", index=False)
    df_wb.to_csv(OUT / "growth_wb.csv", index=False)

    def window_slope(df, lo, hi):
        sub = df[(df["step"] >= lo) & (df["step"] <= hi)]
        if len(sub) < 2:
            return float("nan")
        return float(np.polyfit(sub["step"], sub["U_max"], 1)[0])

    rows = []
    for label, df in [("current", df_cur), ("WB", df_wb)]:
        rows.append({"method": label,
                     "step1": df["U_max"].iloc[0],
                     "slope_1_10": window_slope(df, 1, 10),
                     "slope_10_50": window_slope(df, 10, 50),
                     "slope_50_200": window_slope(df, 50, 200),
                     "U200": df["U_max"].iloc[-1]})
        print(f"  {label:8s}: step1={rows[-1]['step1']:.4e}  slope[1-10]={rows[-1]['slope_1_10']:.3e}  "
              f"slope[10-50]={rows[-1]['slope_10_50']:.3e}  slope[50-200]={rows[-1]['slope_50_200']:.3e}  "
              f"U200={rows[-1]['U200']:.4e}")
    pd.DataFrame(rows).to_csv(OUT / "growth_window_analysis.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.semilogy(df_cur["step"], df_cur["U_max"].clip(lower=1e-18), label="current")
    ax.semilogy(df_wb["step"], df_wb["U_max"].clip(lower=1e-18), label="WB")
    ax.set_xlabel("step"); ax.set_ylabel("U_max [m/s] (log)")
    ax.set_title("Figure A: U_max vs step, current vs WB")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG / "figureA_Umax_vs_step.png", dpi=200)

    # ============================================================
    # TABLE B: grid convergence, Family A (eps/dx fixed) and Family B (eps fixed)
    # ============================================================
    print("\n=== Table B: grid convergence Family A (eps/dx fixed = 1.5) ===")
    famA_rows = []
    for dx_mm in [3.0, 1.5, 0.75]:
        eps_mm = 1.5 * dx_mm
        cfg_, grid_, omega_, g_, rw_, ra_, phi0_, eta_r_, eps0_, rho_c_, mu_c_ = setup_case(dx_mm, eps_mm)
        inv_ur_c, inv_uz_c = face_inv_rho_ur(rho_c_), face_inv_rho_uz(rho_c_)
        inv_ur_w, inv_uz_w = build_rho_face_WB(grid_, omega_, g_, eta_r_, eps0_, rw_, ra_)
        S_cent = omega_ ** 2 * grid_.r_f[:, None] * np.ones(grid_.shape_ur)
        S_grav = -g_ * np.ones(grid_.shape_uz)
        curl_c = discrete_curl(grid_, interp_center_to_ur(rho_c_) * S_cent, interp_center_to_uz(rho_c_) * S_grav)
        curl_w = discrete_curl(grid_, (1.0 / np.where(inv_ur_w > 0, inv_ur_w, 1e30)) * S_cent,
                                (1.0 / np.where(inv_uz_w > 0, inv_uz_w, 1e30)) * S_grav)
        band = np.abs(phi0_) < 2 * eps0_
        df_c100 = run_multistep(grid_, cfg_, omega_, dt, 100, phi0_, rho_c_, mu_c_, inv_ur_c, inv_uz_c)
        df_w100 = run_multistep(grid_, cfg_, omega_, dt, 100, phi0_, rho_c_, mu_c_, inv_ur_w, inv_uz_w)
        famA_rows.append({"dx_mm": dx_mm, "eps_over_dx": 1.5,
                           "U100_current": df_c100["U_max"].iloc[-1], "U100_WB": df_w100["U_max"].iloc[-1],
                           "curl_current": np.nanmax(np.abs(curl_c[band])), "curl_WB": np.nanmax(np.abs(curl_w[band]))})
        print(f"  dx={dx_mm}mm: U100_current={famA_rows[-1]['U100_current']:.4e}  "
              f"U100_WB={famA_rows[-1]['U100_WB']:.4e}  curl_current={famA_rows[-1]['curl_current']:.3e}  "
              f"curl_WB={famA_rows[-1]['curl_WB']:.3e}")
    pd.DataFrame(famA_rows).to_csv(OUT / "convergence_family_A.csv", index=False)

    print("\n=== Table B: grid convergence Family B (physical eps=2.25mm fixed) ===")
    famB_rows = []
    for dx_mm in [3.0, 1.5, 0.75]:
        cfg_, grid_, omega_, g_, rw_, ra_, phi0_, eta_r_, eps0_, rho_c_, mu_c_ = setup_case(dx_mm, 2.25)
        inv_ur_c, inv_uz_c = face_inv_rho_ur(rho_c_), face_inv_rho_uz(rho_c_)
        inv_ur_w, inv_uz_w = build_rho_face_WB(grid_, omega_, g_, eta_r_, eps0_, rw_, ra_)
        S_cent = omega_ ** 2 * grid_.r_f[:, None] * np.ones(grid_.shape_ur)
        S_grav = -g_ * np.ones(grid_.shape_uz)
        curl_c = discrete_curl(grid_, interp_center_to_ur(rho_c_) * S_cent, interp_center_to_uz(rho_c_) * S_grav)
        curl_w = discrete_curl(grid_, (1.0 / np.where(inv_ur_w > 0, inv_ur_w, 1e30)) * S_cent,
                                (1.0 / np.where(inv_uz_w > 0, inv_uz_w, 1e30)) * S_grav)
        band = np.abs(phi0_) < 2 * eps0_
        df_c100 = run_multistep(grid_, cfg_, omega_, dt, 100, phi0_, rho_c_, mu_c_, inv_ur_c, inv_uz_c)
        df_w100 = run_multistep(grid_, cfg_, omega_, dt, 100, phi0_, rho_c_, mu_c_, inv_ur_w, inv_uz_w)
        famB_rows.append({"dx_mm": dx_mm, "eps_over_dx": 2.25 / dx_mm,
                           "U100_current": df_c100["U_max"].iloc[-1], "U100_WB": df_w100["U_max"].iloc[-1],
                           "curl_current": np.nanmax(np.abs(curl_c[band])), "curl_WB": np.nanmax(np.abs(curl_w[band]))})
        print(f"  dx={dx_mm}mm (eps/dx={famB_rows[-1]['eps_over_dx']:.2f}): "
              f"U100_current={famB_rows[-1]['U100_current']:.4e}  U100_WB={famB_rows[-1]['U100_WB']:.4e}  "
              f"curl_current={famB_rows[-1]['curl_current']:.3e}  curl_WB={famB_rows[-1]['curl_WB']:.3e}")
    pd.DataFrame(famB_rows).to_csv(OUT / "convergence_family_B.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    dfa = pd.DataFrame(famA_rows); dfb = pd.DataFrame(famB_rows)
    axes[0].loglog(dfa["dx_mm"], dfa["U100_current"], "o-", label="current")
    axes[0].loglog(dfa["dx_mm"], dfa["U100_WB"], "s-", label="WB")
    axes[0].set_xlabel("dx [mm]"); axes[0].set_ylabel("U100 [m/s]"); axes[0].set_title("Family A: eps/dx fixed")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].loglog(dfb["dx_mm"], dfb["U100_current"], "o-", label="current")
    axes[1].loglog(dfb["dx_mm"], dfb["U100_WB"], "s-", label="WB")
    axes[1].set_xlabel("dx [mm]"); axes[1].set_ylabel("U100 [m/s]"); axes[1].set_title("Family B: physical eps fixed")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG / "figureC_grid_convergence.png", dpi=200)

    # ============================================================
    # TABLE C: radial vs vertical WB decomposition
    # ============================================================
    print("\n=== Table C: radial vs vertical WB decomposition, 200 steps (baseline grid) ===")
    inv_ur_rz, inv_uz_rz = inv_ur_wb, inv_uz_wb
    inv_ur_ronly, inv_uz_ronly = inv_ur_wb, inv_uz_cur
    inv_ur_zonly, inv_uz_zonly = inv_ur_cur, inv_uz_wb

    df_ronly = run_multistep(grid, cfg, omega, dt, 200, phi0, rho_current, mu_current, inv_ur_ronly, inv_uz_ronly)
    df_zonly = run_multistep(grid, cfg, omega, dt, 200, phi0, rho_current, mu_current, inv_ur_zonly, inv_uz_zonly)

    rv_rows = [
        {"treatment": "current", "step1": df_cur["U_max"].iloc[0], "U200": df_cur["U_max"].iloc[-1]},
        {"treatment": "WB radial only", "step1": df_ronly["U_max"].iloc[0], "U200": df_ronly["U_max"].iloc[-1]},
        {"treatment": "WB vertical only", "step1": df_zonly["U_max"].iloc[0], "U200": df_zonly["U_max"].iloc[-1]},
        {"treatment": "WB radial+vertical", "step1": df_wb["U_max"].iloc[0], "U200": df_wb["U_max"].iloc[-1]},
    ]
    for r in rv_rows:
        print(f"  {r['treatment']:20s}: step1={r['step1']:.4e}  U200={r['U200']:.4e}")
    pd.DataFrame(rv_rows).to_csv(OUT / "wb_radial_vertical.csv", index=False)

    # ============================================================
    # TABLE D: perturbation preservation
    # ============================================================
    print("\n=== Table D: perturbation preservation ===")
    R_v, H = grid.r_v, cfg.geometry.water_height_m
    S_grav = -g * np.ones(grid.shape_uz)

    def accel_from_utheta(u_theta_field, inv_ur, inv_uz):
        u_theta_ur = u_theta_on_ur_faces(grid, u_theta_field)
        r_f_safe = np.where(grid.r_f == 0.0, grid.dr, grid.r_f)
        S_cent = u_theta_ur ** 2 / r_f_safe[:, None]
        a_r, a_z, _ = P_facerho(grid, S_cent, S_grav, inv_ur, inv_uz)
        return a_r, a_z

    u_theta_eq = omega * grid.r_c[:, None] * np.ones(grid.shape_center)
    A_eq_cur_r, A_eq_cur_z = accel_from_utheta(u_theta_eq, inv_ur_cur, inv_uz_cur)
    A_eq_wb_r, A_eq_wb_z = accel_from_utheta(u_theta_eq, inv_ur_wb, inv_uz_wb)

    pert_rows = []

    def report_perturbation(label, u_theta_pert):
        A_pert_cur_r, A_pert_cur_z = accel_from_utheta(u_theta_pert, inv_ur_cur, inv_uz_cur)
        A_pert_wb_r, A_pert_wb_z = accel_from_utheta(u_theta_pert, inv_ur_wb, inv_uz_wb)
        dA_cur_r = A_pert_cur_r - A_eq_cur_r
        dA_wb_r = A_pert_wb_r - A_eq_wb_r
        dA_cur_z = A_pert_cur_z - A_eq_cur_z
        dA_wb_z = A_pert_wb_z - A_eq_wb_z
        vec_cur = np.concatenate([dA_cur_r.ravel(), dA_cur_z.ravel()])
        vec_wb = np.concatenate([dA_wb_r.ravel(), dA_wb_z.ravel()])
        corr = np.corrcoef(vec_cur, vec_wb)[0, 1] if np.std(vec_cur) > 0 and np.std(vec_wb) > 0 else float("nan")
        mag_cur = np.sqrt(np.mean(vec_cur ** 2)); mag_wb = np.sqrt(np.mean(vec_wb ** 2))
        rel_l2 = np.sqrt(np.mean((vec_cur - vec_wb) ** 2)) / max(1e-30, mag_cur)
        ratio = mag_wb / max(1e-30, mag_cur)
        pert_rows.append({"perturbation": label, "correlation": corr, "magnitude_ratio": ratio, "relative_L2_diff": rel_l2})
        print(f"  {label:20s}: correlation={corr:.4f}  magnitude_ratio={ratio:.4f}  relative_L2_diff={rel_l2:.4f}")
        return dA_cur_r, dA_wb_r

    r2d = grid.r_c[:, None] * np.ones(grid.shape_center)
    z2d = grid.z_c[None, :] * np.ones(grid.shape_center)

    for amp in [0.01, 0.02, 0.05]:
        delta = amp * omega * r2d * np.sin(np.pi * r2d / R_v) * np.cos(np.pi * z2d / H)
        report_perturbation(f"sinusoidal_{amp}", u_theta_eq + delta)

    for label, r0, z0, sig_r, sig_z, A in [
        ("gaussian_bulk", R_v * 0.4, H * 0.3, R_v * 0.15, H * 0.15, 0.02 * omega * R_v),
        ("gaussian_interface", R_v * 0.85, H * 0.9, R_v * 0.1, H * 0.1, 0.02 * omega * R_v),
    ]:
        delta = A * np.exp(-((r2d - r0) ** 2 / sig_r ** 2 + (z2d - z0) ** 2 / sig_z ** 2))
        dA_cur_r, dA_wb_r = report_perturbation(label, u_theta_eq + delta)

    a_core = 0.5 * R_v
    Omega_core = omega
    Gamma = Omega_core * a_core ** 2
    u_theta_rankine = np.where(r2d <= a_core, Omega_core * r2d, Gamma / np.maximum(r2d, 1e-9))
    report_perturbation("rankine_like", u_theta_rankine)

    pert_df = pd.DataFrame(pert_rows)
    pert_df.to_csv(OUT / "perturbation_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(pert_df["perturbation"], pert_df["correlation"])
    ax.axhline(0.99, color="r", ls="--", label="0.99 guideline")
    ax.set_ylabel("correlation(Delta A_current, Delta A_WB)")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    ax.legend(); fig.tight_layout()
    fig.savefig(FIG / "figureF_perturbation_correlation.png", dpi=200)

    print(f"\nAll results saved under {OUT}")


if __name__ == "__main__":
    main()
