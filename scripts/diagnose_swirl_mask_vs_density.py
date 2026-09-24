"""DIAGNOSTIC (not a validated/gated script) -- Session 7, final split of
"density interface presence" from "prescribed swirl mask spatial variation".

T2 (stationary air, diffuse density, U200=4.97e-2) ~= T4 (stationary air,
UNIFORM density, U200=5.37e-2) from the previous round is a strong signal
that the density interface itself was NOT the dominant remaining mechanism
-- the water/air SWIRL MASK (u_theta jumping/varying spatially with phase),
even at uniform density, is enough to reproduce most of the growth. This
script separates the two mechanisms cleanly via:

  - uniform-density, swirl-mask-only controls (M0-M2)
  - curl(rho*S_eff) computed from the ACTUAL prescribed u_theta field (not
    just the pure centrifugal formula assumed spatially uniform)
  - the direct integrability equation d/dz(rho u_theta^2/r) = -g d(rho)/dr
  - a fully-integrable-by-construction manufactured equilibrium E0 (rho,
    u_theta, p built together from one potential Psi) vs. E1 (same rho/p,
    deliberately non-integrable swirl mask)
  - source-norm-normalized comparison to separate "smaller forcing" from
    "better-balanced forcing"

Also confirms: swirl_mode="prescribed" (global domain-wide Omega*r) is used
ONLY by validation/diagnostic scripts (validate_exact_rotating_equilibrium.py,
validate_solid_body_long.py, validate_spinup_transient.py); build_solver's
default and every calibration/sweep/benchmark script use swirl_mode="forced"
(local bottom stirrer forcing). This is a solid-body VALIDATION forcing
inconsistency, not a production bug.

No production code is modified.

Usage:
    python scripts/diagnose_swirl_mask_vs_density.py --steps 200
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.sparse.linalg as spla  # noqa: E402
from scipy.optimize import brentq  # noqa: E402

from air_vortex.config import load_config  # noqa: E402
from air_vortex.grid import build_grid  # noqa: E402
from air_vortex.fields import initialize_rotating_equilibrium  # noqa: E402
from air_vortex.boundary import apply_velocity_bc  # noqa: E402
from air_vortex.operators import (  # noqa: E402
    interp_center_to_ur, interp_center_to_uz, interp_ur_to_center, interp_uz_to_center,
    laplacian_ur, laplacian_uz, u_theta_on_ur_faces, upwind_derivative,
    uz_at_ur_locations, ur_at_uz_locations, divergence, grad_p_to_ur_faces, grad_p_to_uz_faces,
    center_grad_r, center_grad_z,
)
from air_vortex.pressure import (  # noqa: E402
    solve_pressure_poisson, pressure_projection, face_inv_rho_ur, face_inv_rho_uz, pressure_rhs,
)
from air_vortex.properties import material_properties, interface_epsilon, smoothed_heaviside  # noqa: E402
from air_vortex.surface_tension import surface_tension_force  # noqa: E402

OUT = Path("results/diagnostics/swirl_mask_vs_density")
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


def P_production(grid, v_r, v_z, rho):
    v_r = v_r.copy(); v_z = v_z.copy()
    v_r[0, :] = 0.0; v_r[-1, :] = 0.0
    v_z[:, 0] = 0.0; v_z[:, -1] = v_z[:, -2]
    p = solve_pressure_poisson(grid, v_r, v_z, rho, dt=1.0, method="direct", top_bc="open")
    return pressure_projection(grid, v_r, v_z, p, rho, dt=1.0)


def step_with_utheta(grid, cfg, u_r, u_z, u_theta_field_fn, p, rho, mu, phi, dt, u_theta_prev):
    """Diagnostic step that takes an arbitrary u_theta(r,z) callable (not
    necessarily Omega*r) applied fresh every step, exactly mirroring the
    production predictor+pressure+projection sequence otherwise."""
    f_sigma_r, f_sigma_z = surface_tension_force(grid, phi, cfg)
    rho_ur = interp_center_to_ur(rho); rho_uz = interp_center_to_uz(rho)
    mu_ur = interp_center_to_ur(mu); mu_uz = interp_center_to_uz(mu)
    w_at_ur = uz_at_ur_locations(u_z)
    adv_r = u_r * upwind_derivative(u_r, u_r, grid.dr, axis=0) + w_at_ur * upwind_derivative(u_r, w_at_ur, grid.dz, axis=1)
    u_theta_ur = u_theta_on_ur_faces(grid, u_theta_prev)
    r_f_safe = np.where(grid.r_f == 0.0, grid.dr, grid.r_f)
    centrifugal = u_theta_ur ** 2 / r_f_safe[:, None]
    visc_r = mu_ur * laplacian_ur(grid, u_r) / rho_ur
    u_r_star = u_r + dt * (-adv_r + centrifugal + visc_r + f_sigma_r / rho_ur)
    u_at_uz = ur_at_uz_locations(u_r)
    adv_z = u_at_uz * upwind_derivative(u_z, u_at_uz, grid.dr, axis=0) + u_z * upwind_derivative(u_z, u_z, grid.dz, axis=1)
    visc_z = mu_uz * laplacian_uz(grid, u_z) / rho_uz
    u_z_star = u_z + dt * (-adv_z - cfg.fluid.gravity + visc_z + f_sigma_z / rho_uz)
    u_theta_new = u_theta_field_fn()
    u_r_star[0, :] = 0.0; u_r_star[-1, :] = 0.0
    u_z_star[:, 0] = 0.0; u_z_star[:, -1] = u_z_star[:, -2]
    p_new = solve_pressure_poisson(grid, u_r_star, u_z_star, rho, dt, method="direct", p0=p)
    u_r_new, u_z_new = pressure_projection(grid, u_r_star, u_z_star, p_new, rho, dt)
    return u_r_new, u_z_new, u_theta_new, p_new


def run_multistep(grid, cfg, dt, n_steps, phi0, rho0, mu0, u_theta_field_fn):
    u_r = np.zeros(grid.shape_ur); u_z = np.zeros(grid.shape_uz)
    u_theta = u_theta_field_fn()
    p = np.zeros(grid.shape_center)
    rows = []
    for step in range(1, n_steps + 1):
        u_r, u_z, u_theta, p = step_with_utheta(grid, cfg, u_r, u_z, u_theta_field_fn, p, rho0, mu0, phi0, dt, u_theta)
        apply_velocity_bc(grid, type("F", (), {"u_r": u_r, "u_z": u_z, "u_theta": u_theta, "phi": phi0})())
        speed = np.sqrt(0.5 * (u_r[:-1, :] ** 2 + u_r[1:, :] ** 2) + 0.5 * (u_z[:, :-1] ** 2 + u_z[:, 1:] ** 2))
        rows.append({"step": step, "U_max": float(np.max(speed))})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=200)
    args = parser.parse_args()

    cfg = load_config("configs/validation_solid_body.yaml")
    cfg.fluid.surface_tension = 0.0
    grid = build_grid(cfg)
    omega = 2 * np.pi * 100 / 60
    g = cfg.fluid.gravity
    rho_w, rho_a = cfg.fluid.water_density, cfg.fluid.air_density
    dt = 2.0e-4
    eps0 = interface_epsilon(grid, cfg)

    fields0 = initialize_rotating_equilibrium(grid, cfg, omega)
    phi0 = fields0.phi.copy()
    rho_current, mu_current = material_properties(phi0, grid, cfg)
    water_mask = phi0 < 0.0
    r2d = grid.r_c[:, None] * np.ones(grid.shape_center)

    def u_theta_global():
        return omega * grid.r_c[:, None] * np.ones(grid.shape_center)

    def u_theta_water_only():
        return np.where(water_mask, omega * r2d, 0.0)

    def u_theta_phase_weighted():
        h = np.clip(0.5 - phi0 / (2 * eps0), 0.0, 1.0)
        return omega * r2d * h

    # ============================================================
    # SECTION 3/6: curl(rho*S_eff) and integrability eq. from ACTUAL prescribed fields
    # ============================================================
    print("=== curl(rho*S_eff) and integrability residual from ACTUAL prescribed u_theta fields ===")
    S_grav_z = -g * np.ones(grid.shape_uz)

    def analyze_mode(label, u_theta_field, rho_field):
        u_theta_ur = u_theta_on_ur_faces(grid, u_theta_field)
        r_f_safe = np.where(grid.r_f == 0.0, grid.dr, grid.r_f)
        S_cent = u_theta_ur ** 2 / r_f_safe[:, None]
        rho_ur = interp_center_to_ur(rho_field); rho_uz = interp_center_to_uz(rho_field)
        F_r, F_z = rho_ur * S_cent, rho_uz * S_grav_z
        curl_total = discrete_curl(grid, F_r, F_z)
        eps_band = np.abs(phi0) < 2 * eps0
        max_curl = float(np.nanmax(np.abs(curl_total)))
        rms_band = float(np.sqrt(np.nanmean(curl_total[eps_band] ** 2))) if np.any(eps_band) else float("nan")

        # integrability eq: d/dz(rho*u_theta^2/r) vs -g*d(rho)/dr, at cell centers
        r_c_safe = np.where(grid.r_c == 0.0, grid.dr, grid.r_c)
        S_cent_center = u_theta_field ** 2 / r_c_safe[:, None]
        lhs = center_grad_z(grid, rho_field * S_cent_center)
        rhs = -g * center_grad_r(grid, rho_field)
        mismatch = lhs - rhs
        lhs_rms = float(np.sqrt(np.mean(lhs ** 2))); rhs_rms = float(np.sqrt(np.mean(rhs ** 2)))
        mismatch_rms = float(np.sqrt(np.mean(mismatch ** 2)))

        return max_curl, rms_band, lhs_rms, rhs_rms, mismatch_rms

    curl_rows = []
    for label, uth_fn, rho_field, rho_label in [
        ("P0_global", u_theta_global, rho_current, "diffuse"),
        ("P1_water_only", u_theta_water_only, rho_current, "diffuse"),
        ("P2_phase_weighted", u_theta_phase_weighted, rho_current, "diffuse"),
        ("P1_water_only_uniform", u_theta_water_only, rho_w * np.ones(grid.shape_center), "uniform"),
        ("P2_phase_weighted_uniform", u_theta_phase_weighted, rho_w * np.ones(grid.shape_center), "uniform"),
    ]:
        max_curl, rms_band, lhs_rms, rhs_rms, mismatch_rms = analyze_mode(label, uth_fn(), rho_field)
        curl_rows.append({"mode": label, "density": rho_label, "max_curl_total": max_curl,
                           "interface_rms_curl": rms_band, "LHS_rms": lhs_rms, "RHS_rms": rhs_rms,
                           "mismatch_rms": mismatch_rms})
        print(f"  {label:28s} ({rho_label:8s}): max_curl={max_curl:.3e}  interface_RMS_curl={rms_band:.3e}  "
              f"LHS_rms={lhs_rms:.3e}  RHS_rms={rhs_rms:.3e}  mismatch_rms={mismatch_rms:.3e}")
    pd.DataFrame(curl_rows).to_csv(OUT / "table_B_integrability.csv", index=False)

    # ============================================================
    # SECTION 5: uniform-density, swirl-mask-only controls M0-M2
    # ============================================================
    print(f"\n=== Table A: uniform-density swirl-mask-only controls M0-M2, {args.steps} steps ===")
    rho_uniform_field = rho_w * np.ones(grid.shape_center)
    mu_zero = np.zeros(grid.shape_center)

    tableA_rows = []
    dfs = {}
    for label, uth_fn, rho_field, mu_field, rho_label in [
        ("M0_uniform_global", u_theta_global, rho_uniform_field, mu_zero, "uniform"),
        ("M1_uniform_water_only", u_theta_water_only, rho_uniform_field, mu_zero, "uniform"),
        ("M2_uniform_phase_weighted", u_theta_phase_weighted, rho_uniform_field, mu_zero, "uniform"),
        ("P0_diffuse_global", u_theta_global, rho_current, mu_current, "diffuse"),
        ("P1_diffuse_water_only", u_theta_water_only, rho_current, mu_current, "diffuse"),
        ("P2_diffuse_phase_weighted", u_theta_phase_weighted, rho_current, mu_current, "diffuse"),
    ]:
        df = run_multistep(grid, cfg, dt, args.steps, phi0, rho_field, mu_field, uth_fn)
        dfs[label] = df
        max_curl, rms_band, *_ = analyze_mode(label, uth_fn(), rho_field)
        tableA_rows.append({"case": label, "rho": rho_label, "integrability_residual": max_curl,
                             "step1": df["U_max"].iloc[0], "U200": df["U_max"].iloc[-1]})
        print(f"  {label:28s}: integrability_residual={max_curl:.3e}  step1={df['U_max'].iloc[0]:.4e}  "
              f"U{args.steps}={df['U_max'].iloc[-1]:.4e}")
        df.to_csv(OUT / f"{label}.csv", index=False)
    pd.DataFrame(tableA_rows).to_csv(OUT / "table_A_mask_vs_density.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, df in dfs.items():
        ax.semilogy(df["step"], df["U_max"].clip(lower=1e-18), label=label)
    ax.set_xlabel("step"); ax.set_ylabel("U_max [m/s] (log)")
    ax.set_title("Swirl-mask-only (uniform rho) vs. diffuse-density controls")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG / "table_A_mask_vs_density.png", dpi=200)

    # ============================================================
    # SECTION 8/9: E0 fully-integrable-by-construction vs E1 non-integrable mask
    # ============================================================
    print(f"\n=== E0 (fully integrable manufactured eq.) vs E1 (same rho, non-integrable mask), {args.steps} steps ===")
    Psi = 0.5 * omega ** 2 * grid.r_c[:, None] ** 2 - g * grid.z_c[None, :]
    eta_r = fields0.phi[:, 0] * 0  # placeholder, recompute eta from IC directly below
    # recover eta(r) the same way initialize_rotating_equilibrium did:
    from air_vortex.diagnostics import volume_consistent_parabola
    eta_r = volume_consistent_parabola(grid.r_c, grid.r_v, omega, g,
                                        np.pi * grid.r_v ** 2 * cfg.geometry.water_height_m)
    Psi_surface = 0.5 * omega ** 2 * grid.r_c ** 2 - g * eta_r  # Psi AT the interface, now r-dependent in general
    # For a pure solid-body Psi (this choice), Psi_surface should be constant (verify):
    print(f"  Psi_surface range: [{Psi_surface.min():.4f}, {Psi_surface.max():.4f}] "
          f"(should be ~constant for a solid-body-consistent interface)")
    Psi_surface_const = float(np.mean(Psi_surface))

    eps_Psi = g * eps0
    # sign convention: water has z<eta -> LARGER Psi (since Psi decreases with z);
    # so water <-> Psi > Psi_surface_const -> argument for smoothed_heaviside (which
    # gives 0 at negative arg = water, 1 at positive = air) must be NEGATIVE in water:
    chi_E0 = Psi_surface_const - Psi  # negative in water (Psi>Psi_surface), positive in air
    h_E0 = smoothed_heaviside(chi_E0, eps_Psi)
    rho_E0 = rho_w + (rho_a - rho_w) * h_E0

    # verify sign convention along a column
    j_check = grid.Nz // 2
    i_check = grid.Nr // 2
    col_phi = phi0[i_check, :]
    col_rho_E0 = rho_E0[i_check, :]
    water_col = col_phi < 0
    print(f"  sign check (column i={i_check}): rho_E0 in phi<0 (water) rows = "
          f"[{col_rho_E0[water_col].min():.1f},{col_rho_E0[water_col].max():.1f}] (expect ~{rho_w}); "
          f"rho_E0 in phi>0 (air) rows = [{col_rho_E0[~water_col].min():.1f},{col_rho_E0[~water_col].max():.1f}] (expect ~{rho_a})")

    # p(Psi): dp/dPsi = rho(Psi), integrate numerically
    n_lut = 20001
    span = 0.5 * omega ** 2 * grid.r_v ** 2 + g * (grid.z_c[-1] - grid.z_c[0] + 10 * eps0) + abs(Psi_surface_const) + 1.0
    Psi_grid = np.linspace(-span, span, n_lut)
    chi_grid = Psi_surface_const - Psi_grid
    rho_of_chi = rho_w + (rho_a - rho_w) * smoothed_heaviside(chi_grid, eps_Psi)
    # p(Psi) = integral of rho dPsi from a reference; increasing Psi_grid order:
    order = np.argsort(Psi_grid)
    Psi_sorted = Psi_grid[order]; rho_sorted = rho_of_chi[order]
    p_sorted = np.concatenate([[0.0], np.cumsum(0.5 * (rho_sorted[1:] + rho_sorted[:-1]) * np.diff(Psi_sorted))])

    def p_lookup(Psi_val):
        return np.interp(Psi_val, Psi_sorted, p_sorted)

    p_E0_raw = p_lookup(Psi)
    z_top_ghost = grid.z_c[-1] + 0.5 * grid.dz
    Psi_at_ghost_top = 0.5 * omega ** 2 * grid.r_c ** 2 - g * z_top_ghost
    p_at_ghost_top = p_lookup(Psi_at_ghost_top)
    p_E0 = p_E0_raw - p_at_ghost_top[:, None]  # shift so p~0 at the open-top ghost point, per-r column

    u_theta_E0 = omega * grid.r_c[:, None] * np.ones(grid.shape_center)  # sqrt(r*dPsi/dr) = sqrt(r*Omega^2 r) = Omega*r

    def u_theta_E0_fn():
        return u_theta_E0

    def u_theta_E1_fn():
        return u_theta_water_only()  # same non-integrable mask as P1, but paired with E0's rho/p

    df_E0 = run_multistep(grid, cfg, dt, args.steps, phi0, rho_E0, mu_zero, u_theta_E0_fn)
    df_E1 = run_multistep(grid, cfg, dt, args.steps, phi0, rho_E0, mu_zero, u_theta_E1_fn)
    max_curl_E0, rms_E0, *_ = analyze_mode("E0", u_theta_E0, rho_E0)
    max_curl_E1, rms_E1, *_ = analyze_mode("E1", u_theta_water_only(), rho_E0)
    print(f"  E0 (integrable):     integrability_residual={max_curl_E0:.3e}  step1={df_E0['U_max'].iloc[0]:.4e}  U{args.steps}={df_E0['U_max'].iloc[-1]:.4e}")
    print(f"  E1 (non-integrable): integrability_residual={max_curl_E1:.3e}  step1={df_E1['U_max'].iloc[0]:.4e}  U{args.steps}={df_E1['U_max'].iloc[-1]:.4e}")
    df_E0.to_csv(OUT / "E0_integrable.csv", index=False)
    df_E1.to_csv(OUT / "E1_nonintegrable.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.semilogy(df_E0["step"], df_E0["U_max"].clip(lower=1e-18), label="E0 (integrable by construction)")
    ax.semilogy(df_E1["step"], df_E1["U_max"].clip(lower=1e-18), label="E1 (same rho, non-integrable mask)")
    ax.set_xlabel("step"); ax.set_ylabel("U_max [m/s] (log)")
    ax.set_title("E0 vs E1: isolating swirl-mask integrability")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG / "E0_vs_E1.png", dpi=200)

    # ============================================================
    # SECTION 11: source-norm-normalized comparison
    # ============================================================
    print("\n=== Table C: source-norm-normalized comparison ===")
    tableC_rows = []
    for label, uth_fn, rho_field in [
        ("P0_global", u_theta_global, rho_current),
        ("P1_water_only", u_theta_water_only, rho_current),
        ("P2_phase_weighted", u_theta_phase_weighted, rho_current),
    ]:
        uth = uth_fn()
        u_theta_ur = u_theta_on_ur_faces(grid, uth)
        r_f_safe = np.where(grid.r_f == 0.0, grid.dr, grid.r_f)
        S_cent = u_theta_ur ** 2 / r_f_safe[:, None]
        Pr, Pz = P_production(grid, S_cent, S_grav_z, rho_field)
        S_norm = norms(S_cent)[0]
        P_norm = norms(Pr)[0]
        frac = P_norm / max(1e-30, S_norm)
        tableC_rows.append({"mode": label, "S_norm": S_norm, "P_S_norm": P_norm, "nonprojectable_fraction": frac})
        print(f"  {label:20s}: ||S||={S_norm:.4e}  ||P(S)||={P_norm:.4e}  nonprojectable_fraction={frac:.4e}")
    pd.DataFrame(tableC_rows).to_csv(OUT / "table_C_source_norms.csv", index=False)

    print(f"\nAll results saved under {OUT}")


if __name__ == "__main__":
    main()
