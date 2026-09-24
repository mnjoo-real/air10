"""DIAGNOSTIC (not a validated/gated script) -- Session 7, final root-cause
verification: full effective-force (centrifugal + gravity) integrability
across the diffuse density interface.

The previous diagnostic (scripts/diagnose_source_projection.py) found
curl(rho*S_cent) != 0 in the interface band and used that to implicate the
diffuse density representation. That test looked at the centrifugal
contribution ALONE. The correct equilibrium condition is

    grad(p) = rho * S_eff,   S_eff = Omega^2 r e_r - g e_z = grad(Psi),
    Psi(r,z) = 0.5 Omega^2 r^2 - g z

so it is curl(rho * S_eff) -- centrifugal AND gravity combined -- that must
vanish for a scalar pressure to exist. This script recomputes everything
with the full S_eff, adds a sharp-interface / smoothing-width sweep, a
potential-aligned density control, an unweighted Helmholtz-style gradient
decomposition of rho*S_eff, an interface-normal/S_eff alignment-angle map,
and wall-distance-binned curl -- all diagnostic-only, no production code
touched, sigma=0, viscosity reported separately.

Usage:
    python scripts/diagnose_effective_potential_integrability.py --steps 200
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

from air_vortex.config import load_config  # noqa: E402
from air_vortex.grid import build_grid  # noqa: E402
from air_vortex.fields import initialize_rotating_equilibrium  # noqa: E402
from air_vortex.boundary import apply_velocity_bc  # noqa: E402
from air_vortex.diagnostics import volume_consistent_parabola  # noqa: E402
from air_vortex.operators import (  # noqa: E402
    interp_center_to_ur, interp_center_to_uz, interp_ur_to_center, interp_uz_to_center,
    laplacian_ur, laplacian_uz, u_theta_on_ur_faces, upwind_derivative,
    uz_at_ur_locations, ur_at_uz_locations, divergence,
    grad_p_to_ur_faces, grad_p_to_uz_faces, center_grad_r, center_grad_z,
)
from air_vortex.pressure import (  # noqa: E402
    pressure_projection, solve_pressure_poisson, face_inv_rho_ur, face_inv_rho_uz,
)
from air_vortex.properties import material_properties, interface_epsilon, smoothed_heaviside  # noqa: E402
from air_vortex.surface_tension import surface_tension_force  # noqa: E402


# ============================================================
# Shared machinery (byte-identical to diagnose_source_projection.py)
# ============================================================
def P(grid, v_r, v_z, rho, top_bc="open"):
    v_r = v_r.copy(); v_z = v_z.copy()
    v_r[0, :] = 0.0; v_r[-1, :] = 0.0
    v_z[:, 0] = 0.0; v_z[:, -1] = v_z[:, -2]
    p = solve_pressure_poisson(grid, v_r, v_z, rho, dt=1.0, method="direct", top_bc=top_bc)
    return pressure_projection(grid, v_r, v_z, p, rho, dt=1.0)


def norms(a):
    a = a[np.isfinite(a)]
    return float(np.max(np.abs(a))), float(np.sqrt(np.mean(a ** 2)))


def discrete_curl(grid, S_r, S_z):
    Sr_c = interp_ur_to_center(S_r)
    Sz_c = interp_uz_to_center(S_z)
    curl = np.full(grid.shape_center, np.nan)
    curl[1:-1, 1:-1] = ((Sz_c[2:, 1:-1] - Sz_c[:-2, 1:-1]) / (2 * grid.dr)
                        - (Sr_c[1:-1, 2:] - Sr_c[1:-1, :-2]) / (2 * grid.dz))
    return curl


def momentum_pressure_step_custom(grid, cfg, u_r, u_z, u_theta, p, rho, mu, phi, dt, omega, top_bc="open"):
    f_sigma_r, f_sigma_z = surface_tension_force(grid, phi, cfg)
    rho_ur = interp_center_to_ur(rho); rho_uz = interp_center_to_uz(rho)
    mu_ur = interp_center_to_ur(mu); mu_uz = interp_center_to_uz(mu)

    w_at_ur = uz_at_ur_locations(u_z)
    adv_r = u_r * upwind_derivative(u_r, u_r, grid.dr, axis=0) \
        + w_at_ur * upwind_derivative(u_r, w_at_ur, grid.dz, axis=1)
    u_theta_ur = u_theta_on_ur_faces(grid, u_theta)
    r_f_safe = np.where(grid.r_f == 0.0, grid.dr, grid.r_f)
    centrifugal = u_theta_ur ** 2 / r_f_safe[:, None]
    visc_r_raw = laplacian_ur(grid, u_r)
    visc_r = mu_ur * visc_r_raw
    u_r_star = u_r + dt * (-adv_r + centrifugal + visc_r / rho_ur + f_sigma_r / rho_ur)

    u_at_uz = ur_at_uz_locations(u_r)
    adv_z = u_at_uz * upwind_derivative(u_z, u_at_uz, grid.dr, axis=0) \
        + u_z * upwind_derivative(u_z, u_z, grid.dz, axis=1)
    visc_z_raw = laplacian_uz(grid, u_z)
    visc_z = mu_uz * visc_z_raw
    u_z_star = u_z + dt * (-adv_z - cfg.fluid.gravity + visc_z / rho_uz + f_sigma_z / rho_uz)

    u_theta_new = omega * grid.r_c[:, None] * np.ones_like(u_theta)

    u_r_star[0, :] = 0.0; u_r_star[-1, :] = 0.0
    u_z_star[:, 0] = 0.0; u_z_star[:, -1] = u_z_star[:, -2]
    if top_bc == "closed":
        u_z_star[:, -1] = 0.0

    p_new = solve_pressure_poisson(grid, u_r_star, u_z_star, rho, dt, method="direct", p0=p, top_bc=top_bc)
    u_r_new, u_z_new = pressure_projection(grid, u_r_star, u_z_star, p_new, rho, dt)
    if top_bc == "closed":
        u_z_new[:, -1] = 0.0
    max_visc_contrib = float(max(np.max(np.abs(visc_r / rho_ur)), np.max(np.abs(visc_z / rho_uz))))
    return u_r_new, u_z_new, u_theta_new, p_new, max_visc_contrib


def run_frozen_multistep(grid, cfg, omega, dt, n_steps, phi0, rho0, mu0, top_bc="open"):
    u_r = np.zeros(grid.shape_ur); u_z = np.zeros(grid.shape_uz)
    u_theta = omega * grid.r_c[:, None] * np.ones(grid.shape_center)
    p = np.zeros(grid.shape_center)
    rows = []
    for step in range(1, n_steps + 1):
        u_r, u_z, u_theta, p, visc_c = momentum_pressure_step_custom(
            grid, cfg, u_r, u_z, u_theta, p, rho0, mu0, phi0, dt, omega, top_bc=top_bc)
        apply_velocity_bc(grid, type("F", (), {"u_r": u_r, "u_z": u_z, "u_theta": u_theta, "phi": phi0})())
        speed = np.sqrt(0.5 * (u_r[:-1, :] ** 2 + u_r[1:, :] ** 2) + 0.5 * (u_z[:, :-1] ** 2 + u_z[:, 1:] ** 2))
        rows.append({"step": step, "t": step * dt,
                     "max_meridional_speed": float(np.max(speed)),
                     "max_viscous_accel_contribution": visc_c})
    return pd.DataFrame(rows)


# ============================================================
# New: effective potential / S_eff / density representations
# ============================================================
def S_eff_faces(grid, omega, g):
    S_r = np.zeros(grid.shape_ur)
    S_r[:] = omega ** 2 * grid.r_f[:, None]
    S_z = -g * np.ones(grid.shape_uz)
    return S_r, S_z


def effective_potential(grid, omega, g):
    return 0.5 * omega ** 2 * grid.r_c[:, None] ** 2 - g * grid.z_c[None, :]


def build_density(h, rho_w, rho_a):
    return rho_w + (rho_a - rho_w) * h


def density_sharp(phi, rho_w, rho_a):
    return np.where(phi < 0.0, rho_w, rho_a)


def density_smoothed_phi(phi, eps, rho_w, rho_a):
    return build_density(smoothed_heaviside(phi, eps), rho_w, rho_a)


def density_potential_aligned(grid, omega, g, eta_r, eps, rho_w, rho_a):
    """Diagnostic-only: replace phi's smoothing coordinate with the
    effective-potential deviation chi = (Psi - Psi_surface)/|grad(Psi)|,
    normalized so a fixed eps corresponds to a fixed GEOMETRIC distance
    along the true equilibrium-consistent direction, unlike phi=z-eta(r)
    whose gradient magnitude sqrt(1+eta'(r)^2) grows with r (steeper
    interface slope near the wall), which anisotropically narrows the
    effective smoothing band there for the same nominal eps."""
    Psi = effective_potential(grid, omega, g)
    C = float(eta_r[0])
    Psi_surface = -g * C  # Psi(r, eta(r)) = -g*C for every r, by construction of eta(r)
    grad_Psi_r = omega ** 2 * grid.r_c[:, None] * np.ones(grid.shape_center)
    grad_Psi_z = -g * np.ones(grid.shape_center)
    grad_Psi_mag = np.sqrt(grad_Psi_r ** 2 + grad_Psi_z ** 2)
    chi = (Psi - Psi_surface) / grad_Psi_mag
    return build_density(smoothed_heaviside(chi, eps), rho_w, rho_a), chi


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

    out_dir = Path("results/diagnostics/effective_potential")
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    fields0 = initialize_rotating_equilibrium(grid, cfg, omega)
    phi0 = fields0.phi.copy()
    eta_r = volume_consistent_parabola(grid.r_c, grid.r_v, omega, g,
                                        np.pi * grid.r_v ** 2 * cfg.geometry.water_height_m)
    eps0 = interface_epsilon(grid, cfg)

    # ============================================================
    # Face-defined S_cent, S_grav, S_eff (shared by all curl computations)
    # ============================================================
    S_cent_r, S_eff_z = np.zeros(grid.shape_ur), np.zeros(grid.shape_uz)
    S_cent_r[:] = omega ** 2 * grid.r_f[:, None]
    S_grav_z = -g * np.ones(grid.shape_uz)
    S_grav_r = np.zeros(grid.shape_ur)
    S_eff_r = S_cent_r.copy()
    S_eff_z_full = S_grav_z.copy()

    def region_masks(phi_field):
        eps_band = np.abs(phi_field) < 2 * eps0
        wall_band = grid.r_c > 0.7 * grid.r_v
        axis_band = grid.r_c < 0.3 * grid.r_v
        bulk_water = (phi_field < -2 * eps0)
        bulk_air = (phi_field > 2 * eps0)
        return {
            "bulk_water": bulk_water & ~wall_band[:, None] & ~axis_band[:, None],
            "bulk_air": bulk_air & ~wall_band[:, None] & ~axis_band[:, None],
            "interface_band": eps_band,
            "wall_interface": eps_band & wall_band[:, None],
            "axis_interface": eps_band & axis_band[:, None],
        }

    def region_stat(field, mask):
        vals = field[mask]; vals = vals[np.isfinite(vals)]
        return float(np.max(np.abs(vals))) if vals.size else float("nan")

    def rms_stat(field, mask):
        vals = field[mask]; vals = vals[np.isfinite(vals)]
        return float(np.sqrt(np.mean(vals ** 2))) if vals.size else float("nan")

    # ============================================================
    # SECTION 3: full-force curl for current smoothed density
    # ============================================================
    rho_current, mu_current = material_properties(phi0, grid, cfg)
    rho_current_uz = interp_center_to_uz(rho_current)
    rho_current_ur = interp_center_to_ur(rho_current)

    F_cent_r = rho_current_ur * S_cent_r
    F_grav_z = rho_current_uz * S_grav_z
    F_eff_r, F_eff_z = F_cent_r, F_grav_z  # F_eff = rho*S_eff (only nonzero components)

    curl_cent = discrete_curl(grid, F_cent_r, np.zeros(grid.shape_uz))
    curl_grav = discrete_curl(grid, np.zeros(grid.shape_ur), F_grav_z)
    curl_eff = discrete_curl(grid, F_eff_r, F_eff_z)

    masks = region_masks(phi0)
    print("=== Table: full-force curl by region (current smoothed density) ===")
    rows = []
    for name, mask in masks.items():
        rows.append({"region": name,
                     "max_curl_cent": region_stat(curl_cent, mask),
                     "max_curl_gravity": region_stat(curl_grav, mask),
                     "max_curl_total_Seff": region_stat(curl_eff, mask)})
        print(f"  {name:16s}: curl_cent={rows[-1]['max_curl_cent']:.3e}  "
              f"curl_grav={rows[-1]['max_curl_gravity']:.3e}  "
              f"curl_Seff={rows[-1]['max_curl_total_Seff']:.3e}")
    pd.DataFrame(rows).to_csv(out_dir / "curl_by_region_current.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, field, title in [(axes[0], curl_cent, "curl(rho*S_cent)"),
                              (axes[1], curl_grav, "curl(rho*S_grav)"),
                              (axes[2], curl_eff, "curl(rho*S_eff) [combined]")]:
        im = ax.pcolormesh(grid.r_c * 1e3, grid.z_c * 1e3, field.T, shading="auto", cmap="RdBu_r")
        fig.colorbar(im, ax=ax)
        ax.contour(grid.r_c * 1e3, grid.z_c * 1e3, phi0.T, levels=[0.0], colors="k", linewidths=1.0)
        ax.set_title(title); ax.set_xlabel("r [mm]"); ax.set_ylabel("z [mm]")
    fig.suptitle("Full effective-force curl decomposition (does gravity cancel centrifugal curl?)")
    fig.tight_layout()
    fig.savefig(fig_dir / "curl_cent_vs_gravity_vs_total.png", dpi=200)
    print(f"Saved {fig_dir / 'curl_cent_vs_gravity_vs_total.png'}")

    # ============================================================
    # SECTION 4: density-representation sweep (R1-R5), curl(rho*S_eff) only
    # ============================================================
    print("\n=== Density-representation sweep: max|curl(rho*S_eff)| ===")
    sweep_rows = []

    def eval_density(name, rho_field, phi_for_mask=phi0):
        ur_ = interp_center_to_ur(rho_field); uz_ = interp_center_to_uz(rho_field)
        Fr, Fz = ur_ * S_cent_r, uz_ * S_grav_z
        curl_ = discrete_curl(grid, Fr, Fz)
        m = region_masks(phi_for_mask)
        max_all = region_stat(curl_, np.ones_like(curl_, dtype=bool))
        rms_band = rms_stat(curl_, m["interface_band"])
        sweep_rows.append({"case": name, "max_curl_Seff": max_all, "interface_band_rms_curl": rms_band})
        print(f"  {name:30s}: max|curl|={max_all:.3e}  interface-band RMS={rms_band:.3e}")
        return curl_

    rho_uniform = rho_w * np.ones(grid.shape_center)
    eval_density("R1_uniform_single_phase", rho_uniform)
    rho_sharp = density_sharp(phi0, rho_w, rho_a)
    eval_density("R2_sharp_no_smoothing", rho_sharp)
    eval_density("R3_current_smoothed(eps0)", rho_current)
    rho_half = density_smoothed_phi(phi0, eps0 / 2, rho_w, rho_a)
    eval_density("R4_smoothed(eps0/2)", rho_half)
    rho_double = density_smoothed_phi(phi0, eps0 * 2, rho_w, rho_a)
    eval_density("R5_smoothed(eps0*2)", rho_double)
    rho_potential, chi = density_potential_aligned(grid, omega, g, eta_r, eps0, rho_w, rho_a)
    eval_density("R6_potential_aligned(eps0)", rho_potential, phi_for_mask=phi0)

    sweep_df = pd.DataFrame(sweep_rows)
    sweep_df.to_csv(out_dir / "density_sweep_curl.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(sweep_df["case"], sweep_df["max_curl_Seff"])
    ax.set_ylabel("max |curl(rho*S_eff)|")
    ax.set_yscale("log")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(fig_dir / "density_sweep_curl.png", dpi=200)

    # ============================================================
    # SECTION 6: strongest A/B multi-step test (current phi vs potential-aligned)
    # ============================================================
    print(f"\n=== Multi-step A/B: current phi-smoothing vs potential-aligned smoothing ({args.steps} steps) ===")
    df_current = run_frozen_multistep(grid, cfg, omega, dt, args.steps, phi0, rho_current, mu_current)
    mu_potential = cfg.fluid.water_viscosity + (cfg.fluid.air_viscosity - cfg.fluid.water_viscosity) * \
        smoothed_heaviside(chi, eps0)
    df_potential = run_frozen_multistep(grid, cfg, omega, dt, args.steps, phi0, rho_potential, mu_potential)

    df_current.to_csv(out_dir / "multistep_current_phi_smoothing.csv", index=False)
    df_potential.to_csv(out_dir / "multistep_potential_aligned_smoothing.csv", index=False)

    slope_current = (df_current["max_meridional_speed"].iloc[9] - df_current["max_meridional_speed"].iloc[0]) / 9
    slope_potential = (df_potential["max_meridional_speed"].iloc[9] - df_potential["max_meridional_speed"].iloc[0]) / 9
    print(f"  current phi smoothing:     step1={df_current['max_meridional_speed'].iloc[0]:.4e}  "
          f"step{args.steps}={df_current['max_meridional_speed'].iloc[-1]:.4e}  slope={slope_current:.3e}")
    print(f"  potential-aligned smoothing: step1={df_potential['max_meridional_speed'].iloc[0]:.4e}  "
          f"step{args.steps}={df_potential['max_meridional_speed'].iloc[-1]:.4e}  slope={slope_potential:.3e}")
    print(f"  max viscous accel contribution (current): "
          f"{df_current['max_viscous_accel_contribution'].max():.3e}  "
          f"(reference scale vs centrifugal ~{omega**2*grid.r_v:.2f} m/s^2)")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.semilogy(df_current["step"], df_current["max_meridional_speed"].clip(lower=1e-18), label="current phi smoothing")
    ax.semilogy(df_potential["step"], df_potential["max_meridional_speed"].clip(lower=1e-18), label="potential-aligned smoothing")
    ax.set_xlabel("step"); ax.set_ylabel("max meridional speed [m/s] (log)")
    ax.set_title("Decisive A/B: does potential-aligned smoothing stop the growth?")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "ab_current_vs_potential_aligned.png", dpi=200)
    print(f"Saved {fig_dir / 'ab_current_vs_potential_aligned.png'}")

    # ============================================================
    # SECTION 7/8: Helmholtz-style decomposition + one-step residual correlation
    # ============================================================
    print("\n=== Helmholtz-style decomposition of rho*S_eff (unweighted, rho_ref=1) ===")
    rho_ref = np.ones(grid.shape_center)
    for name, Fr, Fz in [("current", F_cent_r, F_grav_z),
                          ("potential_aligned",
                           interp_center_to_ur(rho_potential) * S_cent_r,
                           interp_center_to_uz(rho_potential) * S_grav_z)]:
        F_nongrad_r, F_nongrad_z = P(grid, Fr, Fz, rho_ref)
        frac = norms(F_nongrad_r)[0] / max(1e-30, norms(Fr)[0])
        print(f"  {name:20s}: ||F_non_gradient||_inf / ||F||_inf = {frac:.4e}")

    # one-step actual residual (physically-weighted projection) vs P(F_eff) with real rho
    u_r0 = np.zeros(grid.shape_ur); u_z0 = np.zeros(grid.shape_uz)
    u_theta0 = omega * grid.r_c[:, None] * np.ones(grid.shape_center)
    u_r_new, u_z_new, _, _, _ = momentum_pressure_step_custom(
        grid, cfg, u_r0, u_z0, u_theta0, np.zeros(grid.shape_center), rho_current, mu_current, phi0, dt, omega)
    delta_u_actual_r = u_r_new / dt
    P_Seff_r, P_Seff_z = P(grid, S_cent_r, S_grav_z, rho_current)
    err_r = delta_u_actual_r - P_Seff_r
    linf_e, l2_e = norms(err_r)
    corr = np.corrcoef(delta_u_actual_r.ravel(), P_Seff_r.ravel())[0, 1]
    print(f"\n  ||Delta_u_actual - P(S_eff)||_r (with viscosity+advection at u=0, so should match exactly):"
          f" Linf={linf_e:.3e}  L2={l2_e:.3e}  correlation={corr:.6f}")

    # ============================================================
    # SECTION 9: interface-normal / S_eff alignment angle
    # ============================================================
    print("\n=== Interface-normal vs S_eff alignment angle ===")
    for name, rho_field in [("current_phi", rho_current), ("potential_aligned", rho_potential)]:
        grad_rho_r = center_grad_r(grid, rho_field)
        grad_rho_z = center_grad_z(grid, rho_field)
        S_eff_r_c = omega ** 2 * grid.r_c[:, None] * np.ones(grid.shape_center)
        S_eff_z_c = -g * np.ones(grid.shape_center)
        mag_rho = np.sqrt(grad_rho_r ** 2 + grad_rho_z ** 2)
        mag_S = np.sqrt(S_eff_r_c ** 2 + S_eff_z_c ** 2)
        band = np.abs(phi0) < 2 * eps0
        cos_theta = np.full(grid.shape_center, np.nan)
        valid = band & (mag_rho > 1e-6) & (mag_S > 1e-9)
        cos_theta[valid] = (grad_rho_r[valid] * S_eff_r_c[valid] + grad_rho_z[valid] * S_eff_z_c[valid]) / \
            (mag_rho[valid] * mag_S[valid])
        theta_deg = np.degrees(np.arccos(np.clip(np.abs(cos_theta), -1, 1)))
        mean_a = float(np.nanmean(theta_deg)); rms_a = float(np.sqrt(np.nanmean(theta_deg ** 2)))
        max_a = float(np.nanmax(theta_deg)) if np.any(np.isfinite(theta_deg)) else float("nan")
        print(f"  {name:20s}: mean_angle={mean_a:.3f} deg  RMS={rms_a:.3f} deg  max={max_a:.3f} deg")

    # ============================================================
    # SECTION 10: wall-distance-binned curl RMS
    # ============================================================
    print("\n=== Wall-distance-binned RMS |curl(rho*S_eff)| ===")
    bins = np.linspace(0, grid.r_v, 6)
    bin_rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (grid.r_c[:, None] >= lo) & (grid.r_c[:, None] < hi) & (np.abs(phi0) < 2 * eps0) * np.ones(grid.shape_center, dtype=bool)
        rms_ = rms_stat(curl_eff, mask)
        bin_rows.append({"r_lo_mm": lo * 1e3, "r_hi_mm": hi * 1e3, "rms_curl_Seff": rms_})
        print(f"  r in [{lo*1e3:.1f},{hi*1e3:.1f}]mm: RMS|curl(S_eff)|={rms_:.3e}")
    pd.DataFrame(bin_rows).to_csv(out_dir / "wall_distance_binned_curl.csv", index=False)

    # ============================================================
    # summary
    # ============================================================
    summary = {
        "step1_current": df_current["max_meridional_speed"].iloc[0],
        f"step{args.steps}_current": df_current["max_meridional_speed"].iloc[-1],
        "step1_potential_aligned": df_potential["max_meridional_speed"].iloc[0],
        f"step{args.steps}_potential_aligned": df_potential["max_meridional_speed"].iloc[-1],
        "max_curl_Seff_current": sweep_rows[2]["max_curl_Seff"],
        "max_curl_Seff_potential_aligned": sweep_rows[-1]["max_curl_Seff"],
        "identity_check_Linf": linf_e,
        "identity_check_correlation": corr,
    }
    pd.DataFrame([summary]).to_csv(out_dir / "summary.csv", index=False)
    print(f"\nSaved {out_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
