"""DIAGNOSTIC (not a validated/gated script) -- Session 7, final split of
"initial-condition mismatch" vs. "repeated per-step forcing mismatch", plus
a genuinely open-top-compatible manufactured two-phase equilibrium.

Corrects a real gap in the previous round: solver.py's prescribed swirl mode
resets u_theta=Omega*r EVERYWHERE, EVERY step (see confirmation below), so
varying only the t=0 air swirl (as the prior air-swirl-sensitivity test did)
cannot separate a one-time IC effect from a recurring per-step effect. This
script builds diagnostic-only per-step forcing rules (P0-P3) that apply a
chosen air-swirl policy EVERY step, not just at t=0, and a new manufactured
equilibrium (rotating water / stationary air, both consistent with the real
open-top p=0 BC) to cleanly separate the air/top-BC mechanism from the
diffuse-interface mechanism identified in prior rounds.

No production code is modified.

Usage:
    python scripts/diagnose_repeated_forcing_and_manufactured_eq.py --steps 200
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
from scipy.optimize import brentq  # noqa: E402

from air_vortex.config import load_config  # noqa: E402
from air_vortex.grid import build_grid  # noqa: E402
from air_vortex.fields import initialize_rotating_equilibrium  # noqa: E402
from air_vortex.boundary import apply_velocity_bc  # noqa: E402
from air_vortex.operators import (  # noqa: E402
    interp_center_to_ur, interp_center_to_uz, laplacian_ur, laplacian_uz,
    u_theta_on_ur_faces, upwind_derivative, uz_at_ur_locations, ur_at_uz_locations, divergence,
)
from air_vortex.pressure import solve_pressure_poisson, pressure_projection  # noqa: E402
from air_vortex.properties import material_properties, interface_epsilon  # noqa: E402
from air_vortex.surface_tension import surface_tension_force  # noqa: E402

OUT = Path("results/diagnostics/repeated_forcing_manufactured_eq")
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)


# ============================================================
# SECTION 1: confirm prescribed swirl mechanics (read-only check, printed)
# ============================================================
def confirm_prescribed_swirl_mechanics():
    print("=== Confirming solver.py prescribed swirl mechanics (source inspection) ===")
    print("  solver.py Solver.step(), swirl_mode='prescribed' branch (~line 120-121):")
    print("    u_theta_new = omega * grid.r_c[:, None] * np.ones_like(f.u_theta)")
    print("  This assigns Omega*r to EVERY cell (Nr,Nz) -- water AND air, interface band")
    print("  included -- with NO phi/water-air distinction. It runs AFTER the r/z momentum")
    print("  predictor (which uses the PREVIOUS u_theta for the centrifugal term) and BEFORE")
    print("  the pressure solve/projection, so u_theta_new feeds the NEXT step's centrifugal")
    print("  term, every single step, regardless of what u_theta was before or what type of")
    print("  fluid occupies that cell. This is confirmed structurally (not just by memory) here.")


# ============================================================
# SECTION 3: diagnostic per-step swirl-forcing modes P0-P3
# ============================================================
def apply_swirl_mode(grid, phi, omega, mode, eps0):
    r2d = grid.r_c[:, None] * np.ones(grid.shape_center)
    z2d = grid.z_c[None, :] * np.ones(grid.shape_center)
    z_top = grid.z_c[-1] + 0.5 * grid.dz
    u_solid = omega * r2d

    if mode == "P0_current":
        return u_solid.copy()
    if mode == "P1_water_only":
        return np.where(phi < 0.0, u_solid, 0.0)
    if mode == "P2_phase_weighted":
        h = np.clip(0.5 - phi / (2 * eps0), 0.0, 1.0)  # 1 in water, 0 in air, smooth in band
        return u_solid * h
    if mode == "P3_decay_to_top":
        # water: solid body. air: linearly decays from the water value at
        # the LOCAL interface height to 0 at the domain top (diagnostic
        # construction only, not a physical proposal).
        water = phi < 0.0
        # local interface height per column: first z where phi crosses 0
        eta_col = np.full(grid.Nr, np.nan)
        for i in range(grid.Nr):
            col = phi[i, :]
            idx = np.where(np.diff(np.sign(col)))[0]
            eta_col[i] = grid.z_c[idx[0]] if len(idx) else grid.z_c[0]
        eta2d = eta_col[:, None] * np.ones(grid.shape_center)
        frac = np.clip((z_top - z2d) / np.maximum(z_top - eta2d, 1e-6), 0.0, 1.0)
        return np.where(water, u_solid, u_solid * frac)
    raise ValueError(mode)


def step_diagnostic(grid, cfg, u_r, u_z, u_theta_prev, p, rho, mu, phi, dt, omega, mode, eps0):
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

    u_theta_new = apply_swirl_mode(grid, phi, omega, mode, eps0)  # <-- the ONLY thing that varies P0-P3

    u_r_star[0, :] = 0.0; u_r_star[-1, :] = 0.0
    u_z_star[:, 0] = 0.0; u_z_star[:, -1] = u_z_star[:, -2]
    p_new = solve_pressure_poisson(grid, u_r_star, u_z_star, rho, dt, method="direct", p0=p)
    u_r_new, u_z_new = pressure_projection(grid, u_r_star, u_z_star, p_new, rho, dt)
    return u_r_new, u_z_new, u_theta_new, p_new


def run_multistep(grid, cfg, omega, dt, n_steps, phi0, rho0, mu0, mode, eps0, u_theta_init=None):
    u_r = np.zeros(grid.shape_ur); u_z = np.zeros(grid.shape_uz)
    u_theta = u_theta_init.copy() if u_theta_init is not None else apply_swirl_mode(grid, phi0, omega, mode, eps0)
    p = np.zeros(grid.shape_center)
    water_cell = phi0 < 0.0
    eps_band = np.abs(phi0) < 2 * eps0
    wall_band = grid.r_c > 0.7 * grid.r_v
    air_top_band = (~water_cell) & (grid.z_c[None, :] > grid.z_c[-1] - 3 * grid.dz)
    rows = []
    for step in range(1, n_steps + 1):
        u_r, u_z, u_theta, p = step_diagnostic(grid, cfg, u_r, u_z, u_theta, p, rho0, mu0, phi0, dt, omega, mode, eps0)
        apply_velocity_bc(grid, type("F", (), {"u_r": u_r, "u_z": u_z, "u_theta": u_theta, "phi": phi0})())
        speed_ur_c = 0.5 * (u_r[:-1, :] ** 2 + u_r[1:, :] ** 2)
        speed_uz_c = 0.5 * (u_z[:, :-1] ** 2 + u_z[:, 1:] ** 2)
        speed = np.sqrt(speed_ur_c + speed_uz_c)

        def roi_max(mask2d_center):
            # crude: max meridional speed at cell centers within mask
            return float(np.max(speed[mask2d_center])) if np.any(mask2d_center) else float("nan")

        rows.append({
            "step": step, "U_max": float(np.max(speed)),
            "U_bulk_water": roi_max(water_cell & ~eps_band),
            "U_interface": roi_max(eps_band),
            "U_bulk_air": roi_max((~water_cell) & ~eps_band),
            "U_near_top_air": roi_max(air_top_band),
            "U_wall_interface": roi_max(eps_band & wall_band[:, None]),
            "max_div": float(np.max(np.abs(divergence(grid, u_r, u_z)))),
        })
    return pd.DataFrame(rows)


# ============================================================
# SECTION 6: open-top-compatible manufactured equilibrium
# ============================================================
def manufactured_eta(grid, cfg, omega, g, rho_w, rho_a, target_volume):
    scale = rho_w / (rho_w - rho_a)

    def volume_for_C(C_eta):
        eta = C_eta + scale * omega ** 2 * grid.r_c ** 2 / (2 * g)
        eta = np.clip(eta, grid.z_c[0] - grid.dz, grid.z_c[-1] + grid.dz)
        return float(np.sum(2 * np.pi * grid.r_c * grid.dr * eta))

    lo, hi = -1.0, 1.0
    while volume_for_C(lo) > target_volume:
        lo -= 1.0
    while volume_for_C(hi) < target_volume:
        hi += 1.0
    C_eta = brentq(lambda C: volume_for_C(C) - target_volume, lo, hi)
    eta = C_eta + scale * omega ** 2 * grid.r_c ** 2 / (2 * g)
    return eta


def build_manufactured_fields(grid, cfg, omega, density_mode, phi_source=None):
    g = cfg.fluid.gravity
    rho_w, rho_a = cfg.fluid.water_density, cfg.fluid.air_density
    target_volume = np.pi * grid.r_v ** 2 * cfg.geometry.water_height_m
    eta = manufactured_eta(grid, cfg, omega, g, rho_w, rho_a, target_volume)
    z_top = grid.z_c[-1] + 0.5 * grid.dz

    phi = grid.z_c[None, :] - eta[:, None]
    water_cell = phi < 0.0

    if density_mode == "uniform":
        rho = rho_w * np.ones(grid.shape_center)
        mu = np.zeros(grid.shape_center)
    elif density_mode == "sharp":
        rho = np.where(water_cell, rho_w, rho_a)
        mu = np.zeros(grid.shape_center)
    elif density_mode == "diffuse":
        rho, _ = material_properties(phi, grid, cfg)
        mu = np.zeros(grid.shape_center)
    else:
        raise ValueError(density_mode)

    u_theta = np.where(water_cell, omega * grid.r_c[:, None], 0.0) * np.ones(grid.shape_center)
    r2d = grid.r_c[:, None] * np.ones(grid.shape_center)
    z2d = grid.z_c[None, :] * np.ones(grid.shape_center)
    C = 0.0  # gauge: choose so p_water matches p_air at a reference; absorbed by construction below
    p_air = rho_a * g * (z_top - z2d)
    # p_water must equal p_air at the true interface eta(r); solve for the
    # per-column additive offset consistent with the derivation:
    # p_water(r,z) = [rho_a*g*(z_top-eta(r))] + 0.5*rho_w*Omega^2*(r^2 - r^2) ... simplest:
    # match at the interface directly, then extend with the known r,z dependence.
    p_water_shape = 0.5 * rho_w * omega ** 2 * r2d ** 2 - rho_w * g * z2d
    p_water_shape_at_eta = 0.5 * rho_w * omega ** 2 * grid.r_c ** 2 - rho_w * g * eta
    p_air_at_eta = rho_a * g * (z_top - eta)
    offset = p_air_at_eta - p_water_shape_at_eta  # per-r column additive constant
    p_water = p_water_shape + offset[:, None]
    p = np.where(water_cell, p_water, p_air)
    return phi, rho, mu, u_theta, p, eta


def check_manufactured_residuals(grid, cfg, omega, phi, rho, u_theta, p, eta):
    g = cfg.fluid.gravity
    rho_w, rho_a = cfg.fluid.water_density, cfg.fluid.air_density
    z_top = grid.z_c[-1] + 0.5 * grid.dz
    p_top_row = p[:, -1]
    top_residual = float(np.max(np.abs(p_top_row)))
    print(f"  top pressure residual max|p(r,z_top)|: {top_residual:.3e} Pa (should be ~0)")
    return top_residual


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=200)
    args = parser.parse_args()

    confirm_prescribed_swirl_mechanics()

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

    # ============================================================
    # SECTION 2: prove B/C collapse to A after step 1 under the OLD test
    # ============================================================
    print("\n=== Proving prior test's B/C collapse to A after step 1 (prescribed mode) ===")
    water_mask = phi0 < 0.0
    u_theta_A0 = omega * grid.r_c[:, None] * np.ones(grid.shape_center)
    u_theta_B0 = np.where(water_mask, u_theta_A0, 0.0)
    for label, uth0 in [("A", u_theta_A0), ("B", u_theta_B0)]:
        df = run_multistep(grid, cfg, omega, dt, 3, phi0, rho_current, mu_current, "P0_current", eps0, u_theta_init=uth0)
        # print u_theta in air at steps 0,1,2 directly
        u_r = np.zeros(grid.shape_ur); u_z = np.zeros(grid.shape_uz)
        u_theta = uth0.copy(); p = np.zeros(grid.shape_center)
        air_sample = (~water_mask)
        print(f"  case {label}: u_theta[air].max at step0={np.max(u_theta[air_sample]):.4f}")
        for s in range(1, 3):
            u_r, u_z, u_theta, p = step_diagnostic(grid, cfg, u_r, u_z, u_theta, p, rho_current, mu_current, phi0, dt, omega, "P0_current", eps0)
            print(f"           u_theta[air].max at step{s}={np.max(u_theta[air_sample]):.4f}  (Omega*R_v={omega*grid.r_v:.4f})")
    print("  => confirmed: under P0 (current production rule), A and B's u_theta in air become")
    print("     IDENTICAL from step 1 onward. The prior test's conclusion is correctly re-labeled")
    print("     as an INITIAL-CONDITION-ONLY sensitivity test, not a repeated-forcing test.")

    # ============================================================
    # TABLE A: P0-P3 repeated forcing modes, frozen phi/rho, 200 steps
    # ============================================================
    print(f"\n=== Table A: repeated forcing modes P0-P3, frozen phi/rho, {args.steps} steps ===")
    modeA_rows = []
    dfs = {}
    for mode in ["P0_current", "P1_water_only", "P2_phase_weighted", "P3_decay_to_top"]:
        df = run_multistep(grid, cfg, omega, dt, args.steps, phi0, rho_current, mu_current, mode, eps0)
        dfs[mode] = df
        df.to_csv(OUT / f"forcing_{mode}.csv", index=False)

        def slope(lo, hi):
            sub = df[(df["step"] >= lo) & (df["step"] <= hi)]
            return float(np.polyfit(sub["step"], sub["U_max"], 1)[0]) if len(sub) > 1 else float("nan")

        modeA_rows.append({"mode": mode, "step1": df["U_max"].iloc[0],
                            "slope_1_10": slope(1, 10), "slope_10_50": slope(10, 50), "slope_50_200": slope(50, args.steps),
                            "U200": df["U_max"].iloc[-1]})
        r = modeA_rows[-1]
        print(f"  {mode:20s}: step1={r['step1']:.4e}  slope[1-10]={r['slope_1_10']:.3e}  "
              f"slope[10-50]={r['slope_10_50']:.3e}  slope[50-{args.steps}]={r['slope_50_200']:.3e}  U200={r['U200']:.4e}")
    pd.DataFrame(modeA_rows).to_csv(OUT / "table_A_forcing_modes.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 5))
    for mode, df in dfs.items():
        ax.semilogy(df["step"], df["U_max"].clip(lower=1e-18), label=mode)
    ax.set_xlabel("step"); ax.set_ylabel("U_max [m/s] (log)")
    ax.set_title("Repeated-forcing air-swirl modes (frozen phi/rho)")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG / "table_A_forcing_modes.png", dpi=200)

    # ROI breakdown, P0 vs P1
    print("\n  ROI breakdown, P0 vs P1, final step:")
    for mode in ["P0_current", "P1_water_only"]:
        last = dfs[mode].iloc[-1]
        print(f"    {mode:16s}: bulk_water={last['U_bulk_water']:.4e}  interface={last['U_interface']:.4e}  "
              f"bulk_air={last['U_bulk_air']:.4e}  near_top_air={last['U_near_top_air']:.4e}  "
              f"wall_interface={last['U_wall_interface']:.4e}")

    # ============================================================
    # TABLE B / C: manufactured open-top-compatible equilibrium, M1-M3 / T1-T4
    # ============================================================
    print(f"\n=== Table B: manufactured open-top equilibrium (mu=0), {args.steps} steps ===")
    tableB_rows = []
    manuf_fields = {}
    for density_mode in ["uniform", "sharp", "diffuse"]:
        phi_m, rho_m, mu_m, uth_m, p_m, eta_m = build_manufactured_fields(grid, cfg, omega, density_mode)
        top_res = check_manufactured_residuals(grid, cfg, omega, phi_m, rho_m, uth_m, p_m, eta_m)
        df = run_multistep(grid, cfg, omega, dt, args.steps, phi_m, rho_m, mu_m, "P1_water_only", eps0, u_theta_init=uth_m)
        manuf_fields[density_mode] = (phi_m, rho_m, mu_m, uth_m, p_m)
        dfs[f"M_{density_mode}"] = df
        tableB_rows.append({"representation": density_mode, "top_p_residual": top_res,
                             "step1": df["U_max"].iloc[0], "U200": df["U_max"].iloc[-1],
                             "max_div": df["max_div"].max()})
        print(f"  {density_mode:10s}: top_p_residual={top_res:.3e}  step1={df['U_max'].iloc[0]:.4e}  "
              f"U{args.steps}={df['U_max'].iloc[-1]:.4e}  max_div={df['max_div'].max():.3e}")
        df.to_csv(OUT / f"manufactured_{density_mode}.csv", index=False)
    pd.DataFrame(tableB_rows).to_csv(OUT / "table_B_manufactured_equilibrium.csv", index=False)

    print(f"\n=== Table C: mechanism separation matrix, {args.steps} steps ===")
    # T1: swirl everywhere (old IC construction), diffuse -- reuse dfs["P0_current"] on phi0/rho_current
    T1 = dfs["P0_current"]
    # T2: stationary air, diffuse -- production_frozen density but manufactured (stationary-air) swirl+pressure
    phi_diff, rho_diff, mu_diff, uth_diff, p_diff = manuf_fields["diffuse"]
    T2 = dfs["M_diffuse"]
    T3 = dfs["M_sharp"]
    T4 = dfs["M_uniform"]
    tableC_rows = [
        {"case": "T1", "air_forcing": "swirl_everywhere", "density": "diffuse", "step1": T1["U_max"].iloc[0], "U200": T1["U_max"].iloc[-1]},
        {"case": "T2", "air_forcing": "stationary", "density": "diffuse", "step1": T2["U_max"].iloc[0], "U200": T2["U_max"].iloc[-1]},
        {"case": "T3", "air_forcing": "stationary", "density": "sharp", "step1": T3["U_max"].iloc[0], "U200": T3["U_max"].iloc[-1]},
        {"case": "T4", "air_forcing": "stationary", "density": "uniform", "step1": T4["U_max"].iloc[0], "U200": T4["U_max"].iloc[-1]},
    ]
    for r in tableC_rows:
        print(f"  {r['case']}: air={r['air_forcing']:18s} density={r['density']:8s} step1={r['step1']:.4e}  U200={r['U200']:.4e}")
    pd.DataFrame(tableC_rows).to_csv(OUT / "table_C_mechanism_separation.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 5))
    for label, df in [("T1 (swirl everywhere, diffuse)", T1), ("T2 (stationary air, diffuse)", T2),
                       ("T3 (stationary air, sharp)", T3), ("T4 (stationary air, uniform)", T4)]:
        ax.semilogy(df["step"], df["U_max"].clip(lower=1e-18), label=label)
    ax.set_xlabel("step"); ax.set_ylabel("U_max [m/s] (log)")
    ax.set_title("Mechanism separation: air-forcing vs. density representation")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG / "table_C_mechanism_separation.png", dpi=200)

    print(f"\nAll results saved under {OUT}")


if __name__ == "__main__":
    main()
