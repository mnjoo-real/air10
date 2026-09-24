"""DIAGNOSTIC (not a validated/gated script) -- Session 7, final split of the
two remaining unresolved threads: (1) is the ~1 m/s^2 viscous acceleration a
real bug or a downstream consequence of the already-grown spurious flow, and
(2) is the diffuse-interface curl a genuine well-balanced-discretization
defect that a well-balanced face-density reconstruction can fix?

No production code is modified. All alternate constructions (well-balanced
face density, no-viscosity mode) are diagnostic-only, built by duplicating
the relevant small piece of pressure.py's matrix assembly with a swapped-in
face-density array -- never by editing pressure.py/properties.py themselves.

Usage:
    python scripts/diagnose_wellbalanced_and_viscosity.py --steps 200
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
import scipy.sparse as sp  # noqa: E402
import scipy.sparse.linalg as spla  # noqa: E402

from air_vortex.config import load_config  # noqa: E402
from air_vortex.grid import build_grid  # noqa: E402
from air_vortex.fields import initialize_rotating_equilibrium  # noqa: E402
from air_vortex.boundary import apply_velocity_bc  # noqa: E402
from air_vortex.diagnostics import volume_consistent_parabola  # noqa: E402
from air_vortex.operators import (  # noqa: E402
    interp_center_to_ur, interp_center_to_uz, interp_ur_to_center, interp_uz_to_center,
    laplacian_ur, laplacian_uz, laplacian_utheta, u_theta_on_ur_faces, upwind_derivative,
    uz_at_ur_locations, ur_at_uz_locations, divergence, grad_p_to_ur_faces, grad_p_to_uz_faces,
)
from air_vortex.pressure import (  # noqa: E402
    pressure_projection, solve_pressure_poisson, face_inv_rho_ur, face_inv_rho_uz,
    pressure_rhs, _pressure_matrix_coefficients,
)
from air_vortex.properties import material_properties, interface_epsilon, smoothed_heaviside  # noqa: E402
from air_vortex.surface_tension import surface_tension_force  # noqa: E402


def norms(a):
    a = a[np.isfinite(a)]
    return float(np.max(np.abs(a))), float(np.sqrt(np.mean(a ** 2)))


def P_custom_facerho(grid, v_r, v_z, inv_rho_ur, inv_rho_uz, top_bc="open"):
    """Same diagnostic projection as before, but with the face 1/rho arrays
    supplied directly (so the well-balanced rho_face_WB can be substituted
    for the production face_inv_rho_ur/uz without touching pressure.py)."""
    v_r = v_r.copy(); v_z = v_z.copy()
    v_r[0, :] = 0.0; v_r[-1, :] = 0.0
    v_z[:, 0] = 0.0; v_z[:, -1] = v_z[:, -2]

    A = build_matrix_custom_facerho(grid, inv_rho_ur, inv_rho_uz, top_bc)
    b = pressure_rhs(grid, v_r, v_z, dt=1.0)
    if top_bc == "closed":
        b = b.copy(); b[0] = 0.0
    p_flat = spla.spsolve(A.tocsc(), b, permc_spec="MMD_AT_PLUS_A")
    p = p_flat.reshape(grid.Nr, grid.Nz)

    dpdr = grad_p_to_ur_faces(grid, p)
    dpdz = grad_p_to_uz_faces(grid, p)
    u_r = v_r - inv_rho_ur * dpdr
    u_z = v_z - inv_rho_uz * dpdz
    u_r[0, :] = 0.0; u_r[-1, :] = 0.0
    u_z[:, 0] = 0.0
    return u_r, u_z, p


def build_matrix_custom_facerho(grid, inv_rho_ur, inv_rho_uz, top_bc="open"):
    """Byte-identical structure to pressure._pressure_matrix_coefficients /
    build_pressure_matrix, EXCEPT the face 1/rho arrays are passed in
    directly instead of being computed from a cell-centered rho via simple
    arithmetic averaging -- this is the one and only diagnostic swap."""
    Nr, Nz = grid.Nr, grid.Nz
    dr, dz = grid.dr, grid.dz
    r_f = grid.r_f
    r_c = grid.r_c[:, None]
    N = Nr * Nz

    c_west = np.zeros((Nr, Nz))
    c_west[1:, :] = r_f[1:Nr, None] * inv_rho_ur[1:Nr, :] / (r_c[1:, :] * dr * dr)
    c_east = np.zeros((Nr, Nz))
    c_east[:-1, :] = r_f[1:Nr, None] * inv_rho_ur[1:Nr, :] / (r_c[:-1, :] * dr * dr)
    c_south = np.zeros((Nr, Nz))
    c_south[:, 1:] = inv_rho_uz[:, 1:Nz] / (dz * dz)
    c_north = np.zeros((Nr, Nz))
    c_north[:, :-1] = inv_rho_uz[:, 1:Nz] / (dz * dz)
    c_top_ghost = np.zeros((Nr, Nz))
    if top_bc == "open":
        c_top_ghost[:, -1] = inv_rho_uz[:, Nz] / (0.5 * dz * dz)
    diag = -(c_west + c_east + c_south + c_north + c_top_ghost)

    A = sp.diags(
        diagonals=[c_west.ravel()[Nz:], c_south.ravel()[1:], diag.ravel(),
                   c_north.ravel()[:-1], c_east.ravel()[:-Nz]],
        offsets=[-Nz, -1, 0, 1, Nz], shape=(N, N), format="csr",
    )
    if top_bc == "closed":
        A = A.tolil(); A[0, :] = 0.0; A[0, 0] = 1.0; A = A.tocsr()
    return A


def momentum_pressure_step_facerho(grid, cfg, u_r, u_z, u_theta, rho, mu, phi, dt, omega,
                                    inv_rho_ur, inv_rho_uz, top_bc="open", viscosity_on=True):
    f_sigma_r, f_sigma_z = surface_tension_force(grid, phi, cfg)
    rho_ur = interp_center_to_ur(rho); rho_uz = interp_center_to_uz(rho)
    mu_ur = interp_center_to_ur(mu); mu_uz = interp_center_to_uz(mu)

    w_at_ur = uz_at_ur_locations(u_z)
    adv_r = u_r * upwind_derivative(u_r, u_r, grid.dr, axis=0) \
        + w_at_ur * upwind_derivative(u_r, w_at_ur, grid.dz, axis=1)
    u_theta_ur = u_theta_on_ur_faces(grid, u_theta)
    r_f_safe = np.where(grid.r_f == 0.0, grid.dr, grid.r_f)
    centrifugal = u_theta_ur ** 2 / r_f_safe[:, None]
    visc_r = (mu_ur * laplacian_ur(grid, u_r) / rho_ur) if viscosity_on else 0.0
    u_r_star = u_r + dt * (-adv_r + centrifugal + visc_r + f_sigma_r / rho_ur)

    u_at_uz = ur_at_uz_locations(u_r)
    adv_z = u_at_uz * upwind_derivative(u_z, u_at_uz, grid.dr, axis=0) \
        + u_z * upwind_derivative(u_z, u_z, grid.dz, axis=1)
    visc_z = (mu_uz * laplacian_uz(grid, u_z) / rho_uz) if viscosity_on else 0.0
    u_z_star = u_z + dt * (-adv_z - cfg.fluid.gravity + visc_z + f_sigma_z / rho_uz)

    u_theta_new = omega * grid.r_c[:, None] * np.ones_like(u_theta)
    u_r_star[0, :] = 0.0; u_r_star[-1, :] = 0.0
    u_z_star[:, 0] = 0.0; u_z_star[:, -1] = u_z_star[:, -2]
    if top_bc == "closed":
        u_z_star[:, -1] = 0.0

    A = build_matrix_custom_facerho(grid, inv_rho_ur, inv_rho_uz, top_bc)
    b = pressure_rhs(grid, u_r_star, u_z_star, dt)
    if top_bc == "closed":
        b = b.copy(); b[0] = 0.0
    p_flat = spla.spsolve(A.tocsc(), b, permc_spec="MMD_AT_PLUS_A")
    p_new = p_flat.reshape(grid.Nr, grid.Nz)

    dpdr = grad_p_to_ur_faces(grid, p_new)
    dpdz = grad_p_to_uz_faces(grid, p_new)
    u_r_new = u_r_star - dt * inv_rho_ur * dpdr
    u_z_new = u_z_star - dt * inv_rho_uz * dpdz
    u_r_new[0, :] = 0.0; u_r_new[-1, :] = 0.0
    u_z_new[:, 0] = 0.0
    if top_bc == "closed":
        u_z_new[:, -1] = 0.0
    return u_r_new, u_z_new, u_theta_new, p_new


def run_multistep_facerho(grid, cfg, omega, dt, n_steps, phi0, rho0, mu0, inv_rho_ur, inv_rho_uz,
                           top_bc="open", viscosity_on=True):
    u_r = np.zeros(grid.shape_ur); u_z = np.zeros(grid.shape_uz)
    u_theta = omega * grid.r_c[:, None] * np.ones(grid.shape_center)
    rows = []
    for step in range(1, n_steps + 1):
        u_r, u_z, u_theta, p = momentum_pressure_step_facerho(
            grid, cfg, u_r, u_z, u_theta, rho0, mu0, phi0, dt, omega,
            inv_rho_ur, inv_rho_uz, top_bc=top_bc, viscosity_on=viscosity_on)
        apply_velocity_bc(grid, type("F", (), {"u_r": u_r, "u_z": u_z, "u_theta": u_theta, "phi": phi0})())
        speed = np.sqrt(0.5 * (u_r[:-1, :] ** 2 + u_r[1:, :] ** 2) + 0.5 * (u_z[:, :-1] ** 2 + u_z[:, 1:] ** 2))
        rows.append({"step": step, "max_meridional_speed": float(np.max(speed))})
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

    out_dir = Path("results/diagnostics/wellbalanced_viscosity")
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    fields0 = initialize_rotating_equilibrium(grid, cfg, omega)
    phi0 = fields0.phi.copy()
    eps0 = interface_epsilon(grid, cfg)
    eta_r = volume_consistent_parabola(grid.r_c, grid.r_v, omega, g,
                                        np.pi * grid.r_v ** 2 * cfg.geometry.water_height_m)
    rho_current, mu_current = material_properties(phi0, grid, cfg)

    # ============================================================
    # SECTION 1-3: viscosity at the EXACT IC (not contaminated by growth)
    # ============================================================
    print("=== Viscosity at the EXACT equilibrium IC (u_r=u_z=0 exactly) ===")
    u_r0 = np.zeros(grid.shape_ur); u_z0 = np.zeros(grid.shape_uz)
    Lap_ur0 = laplacian_ur(grid, u_r0)
    Lap_uz0 = laplacian_uz(grid, u_z0)
    print(f"  max|laplacian_ur(u_r=0)| = {np.max(np.abs(Lap_ur0)):.3e}  (u_r input IS zero -> trivially zero, any mu)")
    print(f"  max|laplacian_uz(u_z=0)| = {np.max(np.abs(Lap_uz0)):.3e}  (same)")
    print("  => visc_r, visc_z (the ONLY viscous terms used in prescribed swirl mode) are EXACTLY")
    print("     zero at the true equilibrium, for ANY mu(phi) -- V0/V1/V2/V3 are all identically 0/0.")

    u_theta0 = omega * grid.r_c[:, None] * np.ones(grid.shape_center)
    Lap_utheta = laplacian_utheta(grid, u_theta0)
    interior = Lap_utheta[1:-1, 1:-1]
    print(f"\n  max|laplacian_utheta(Omega r)| FULL domain = {np.max(np.abs(Lap_utheta)):.3e}")
    print(f"  max|laplacian_utheta(Omega r)| INTERIOR (excl. axis/wall/top/bottom rows) = {np.max(np.abs(interior)):.3e}")
    print("  => the large full-domain value is a WALL no-slip-ghost / exact-test-state mismatch")
    print("     artifact (same class as the earlier open-top BC mismatch): laplacian_utheta's wall")
    print("     BC assumes u_theta=0 there (correct for a REAL no-slip wall), but the idealized")
    print("     'solid-body all the way to the wall' test state violates that by construction.")
    print("     Also: 'prescribed' swirl mode NEVER calls laplacian_utheta at all -- irrelevant to")
    print("     the growth being investigated, which only uses prescribed mode.")

    # convergence check: does the INTERIOR identity hold at machine precision regardless of grid?
    conv_rows = []
    for Nr_, Nz_ in [(8, 10), (16, 20), (32, 40)]:
        cfg_ = load_config("configs/validation_solid_body.yaml")
        cfg_.grid.dr_m = grid.dr * grid.Nr / Nr_
        cfg_.grid.dz_m = cfg_.grid.dr_m
        try:
            grid_ = build_grid(cfg_)
            uth_ = omega * grid_.r_c[:, None] * np.ones(grid_.shape_center)
            lap_ = laplacian_utheta(grid_, uth_)
            interior_ = lap_[1:-1, 1:-1]
            conv_rows.append({"Nr": grid_.Nr, "Nz": grid_.Nz, "dx_mm": grid_.dr * 1e3,
                               "max_interior_error": float(np.max(np.abs(interior_)))})
        except Exception as e:
            conv_rows.append({"Nr": Nr_, "Nz": Nz_, "dx_mm": float("nan"), "max_interior_error": str(e)})
    conv_df = pd.DataFrame(conv_rows)
    print("\n  Interior azimuthal-Laplacian identity error vs grid resolution:")
    print(conv_df.to_string(index=False))
    conv_df.to_csv(out_dir / "azimuthal_identity_convergence.csv", index=False)

    # ============================================================
    # SECTION 5: growth with/without viscosity, full frozen two-phase case
    # ============================================================
    print(f"\n=== Growth with/without r/z viscosity contribution ({args.steps} steps, frozen two-phase) ===")
    inv_rho_ur_current = face_inv_rho_ur(rho_current)
    inv_rho_uz_current = face_inv_rho_uz(rho_current)
    df_visc_on = run_multistep_facerho(grid, cfg, omega, dt, args.steps, phi0, rho_current, mu_current,
                                        inv_rho_ur_current, inv_rho_uz_current, viscosity_on=True)
    df_visc_off = run_multistep_facerho(grid, cfg, omega, dt, args.steps, phi0, rho_current, mu_current,
                                         inv_rho_ur_current, inv_rho_uz_current, viscosity_on=False)
    print(f"  full predictor (viscosity ON):  step1={df_visc_on['max_meridional_speed'].iloc[0]:.4e}  "
          f"step{args.steps}={df_visc_on['max_meridional_speed'].iloc[-1]:.4e}")
    print(f"  viscosity disabled:              step1={df_visc_off['max_meridional_speed'].iloc[0]:.4e}  "
          f"step{args.steps}={df_visc_off['max_meridional_speed'].iloc[-1]:.4e}")
    df_visc_on.to_csv(out_dir / "growth_viscosity_on.csv", index=False)
    df_visc_off.to_csv(out_dir / "growth_viscosity_off.csv", index=False)

    # ============================================================
    # SECTION 9/10: well-balanced face-density construction
    # ============================================================
    print("\n=== Well-balanced face-density construction ===")
    Psi_surface = -g * float(eta_r[0])
    eps_Psi = g * eps0  # matches the chi=-g*phi scale identified earlier

    n_lut = 20001
    Psi_lo, Psi_hi = -g * (grid.z_c[-1] + eps0 * 5), -g * (grid.z_c[0] - eps0 * 5) + 0.5 * omega ** 2 * grid.r_v ** 2
    Psi_grid = np.linspace(Psi_lo - abs(Psi_surface) - 1.0, Psi_hi + abs(Psi_surface) + 1.0, n_lut)
    rho_of_Psi_grid = rho_w + (rho_a - rho_w) * smoothed_heaviside(Psi_grid - Psi_surface, eps_Psi)
    p_of_Psi_grid = np.concatenate([[0.0], np.cumsum(0.5 * (rho_of_Psi_grid[1:] + rho_of_Psi_grid[:-1])
                                                       * np.diff(Psi_grid))])

    def p_lookup(Psi_val):
        return np.interp(Psi_val, Psi_grid, p_of_Psi_grid)

    def rho_lookup(Psi_val):
        return np.interp(Psi_val, Psi_grid, rho_of_Psi_grid)

    Psi_center = 0.5 * omega ** 2 * grid.r_c[:, None] ** 2 - g * grid.z_c[None, :]

    # radial faces
    rho_face_WB_ur = np.zeros(grid.shape_ur)
    Psi_l = 0.5 * omega ** 2 * grid.r_c[:-1, None] ** 2 - g * grid.z_c[None, :]
    Psi_r_ = 0.5 * omega ** 2 * grid.r_c[1:, None] ** 2 - g * grid.z_c[None, :]
    dPsi = Psi_r_ - Psi_l
    small = np.abs(dPsi) < 1e-9
    rho_face_WB_ur[1:-1, :] = np.where(small, rho_lookup(0.5 * (Psi_l + Psi_r_)),
                                        (p_lookup(Psi_r_) - p_lookup(Psi_l)) / np.where(small, 1.0, dPsi))

    # axial faces
    rho_face_WB_uz = np.zeros(grid.shape_uz)
    Psi_b = 0.5 * omega ** 2 * grid.r_c[:, None] ** 2 - g * grid.z_c[None, :-1]
    Psi_t = 0.5 * omega ** 2 * grid.r_c[:, None] ** 2 - g * grid.z_c[None, 1:]
    dPsi_z = Psi_t - Psi_b
    small_z = np.abs(dPsi_z) < 1e-9
    rho_face_WB_uz[:, 1:-1] = np.where(small_z, rho_lookup(0.5 * (Psi_b + Psi_t)),
                                        (p_lookup(Psi_t) - p_lookup(Psi_b)) / np.where(small_z, 1.0, dPsi_z))
    rho_face_WB_uz[:, -1] = rho_lookup(0.5 * omega ** 2 * grid.r_c ** 2 - g * (grid.z_c[-1] + 0.5 * grid.dz))

    inv_rho_WB_ur = np.where(rho_face_WB_ur > 0, 1.0 / np.where(rho_face_WB_ur > 0, rho_face_WB_ur, 1.0), 0.0)
    inv_rho_WB_uz = np.where(rho_face_WB_uz > 0, 1.0 / np.where(rho_face_WB_uz > 0, rho_face_WB_uz, 1.0), 0.0)
    inv_rho_WB_ur[0, :] = 0.0; inv_rho_WB_ur[-1, :] = 0.0
    inv_rho_WB_uz[:, 0] = 0.0

    rho_face_current_ur = np.where(inv_rho_ur_current > 0, 1.0 / np.where(inv_rho_ur_current > 0, inv_rho_ur_current, 1.0), 0.0)
    diff_ur = np.where(inv_rho_ur_current[1:-1, :] > 0,
                        (rho_face_current_ur[1:-1, :] - rho_face_WB_ur[1:-1, :]) / rho_face_WB_ur[1:-1, :], 0.0)
    print(f"  max relative diff (current vs WB face density, radial faces): {np.max(np.abs(diff_ur)):.4%}")

    # ============================================================
    # SECTION 11: decisive WB A/B test
    # ============================================================
    print(f"\n=== Decisive WB A/B test: current vs well-balanced face density ({args.steps} steps) ===")
    S_cent_r = omega ** 2 * grid.r_f[:, None] * np.ones(grid.shape_ur)
    S_grav_z = -g * np.ones(grid.shape_uz)

    u_r_new_A, u_z_new_A, p_A = P_custom_facerho(grid, S_cent_r, S_grav_z, inv_rho_ur_current, inv_rho_uz_current)
    u_r_new_B, u_z_new_B, p_B = P_custom_facerho(grid, S_cent_r, S_grav_z, inv_rho_WB_ur, inv_rho_WB_uz)
    linf_A, l2_A = norms(u_r_new_A)
    linf_B, l2_B = norms(u_r_new_B)
    print(f"  one-step projected residual: current Linf={linf_A:.3e} L2={l2_A:.3e}")
    print(f"  one-step projected residual: WB      Linf={linf_B:.3e} L2={l2_B:.3e}")

    df_WB_A = run_multistep_facerho(grid, cfg, omega, dt, args.steps, phi0, rho_current, mu_current,
                                     inv_rho_ur_current, inv_rho_uz_current)
    df_WB_B = run_multistep_facerho(grid, cfg, omega, dt, args.steps, phi0, rho_current, mu_current,
                                     inv_rho_WB_ur, inv_rho_WB_uz)
    print(f"  200-step growth: current  step1={df_WB_A['max_meridional_speed'].iloc[0]:.4e}  "
          f"step{args.steps}={df_WB_A['max_meridional_speed'].iloc[-1]:.4e}")
    print(f"  200-step growth: WB       step1={df_WB_B['max_meridional_speed'].iloc[0]:.4e}  "
          f"step{args.steps}={df_WB_B['max_meridional_speed'].iloc[-1]:.4e}")
    df_WB_A.to_csv(out_dir / "wb_test_current.csv", index=False)
    df_WB_B.to_csv(out_dir / "wb_test_wellbalanced.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.semilogy(df_WB_A["step"], df_WB_A["max_meridional_speed"].clip(lower=1e-18), label="current face density")
    ax.semilogy(df_WB_B["step"], df_WB_B["max_meridional_speed"].clip(lower=1e-18), label="well-balanced face density")
    ax.set_xlabel("step"); ax.set_ylabel("max meridional speed [m/s] (log)")
    ax.set_title("Decisive test: well-balanced face density vs current")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "wb_ab_growth.png", dpi=200)
    print(f"Saved {fig_dir / 'wb_ab_growth.png'}")

    summary = {
        "azimuthal_interior_identity_error": float(np.max(np.abs(interior))),
        "viscosity_on_step1": df_visc_on["max_meridional_speed"].iloc[0],
        f"viscosity_on_step{args.steps}": df_visc_on["max_meridional_speed"].iloc[-1],
        "viscosity_off_step1": df_visc_off["max_meridional_speed"].iloc[0],
        f"viscosity_off_step{args.steps}": df_visc_off["max_meridional_speed"].iloc[-1],
        "one_step_residual_current": linf_A, "one_step_residual_WB": linf_B,
        "step1_WB_A": df_WB_A["max_meridional_speed"].iloc[0], f"step{args.steps}_WB_A": df_WB_A["max_meridional_speed"].iloc[-1],
        "step1_WB_B": df_WB_B["max_meridional_speed"].iloc[0], f"step{args.steps}_WB_B": df_WB_B["max_meridional_speed"].iloc[-1],
    }
    pd.DataFrame([summary]).to_csv(out_dir / "summary.csv", index=False)
    print(f"\nSaved {out_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
