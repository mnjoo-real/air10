"""DIAGNOSTIC (not a validated/gated script) -- Session 7, predictor-force /
pressure-projection discrete-compatibility investigation.

Session 7's temporal-feedback isolation (scripts/diagnose_temporal_feedback.py)
showed that Case A (phi/rho/mu completely frozen, momentum+pressure cycling
only) already reproduces the full spurious-growth accumulation -- rejecting
the Level-Set-feedback hypothesis. This script goes one level deeper: does
the momentum predictor's centrifugal forcing lie in the discrete
pressure-gradient subspace, or does it leave a small "solenoidal remainder"
that the projection cannot remove, and that gets re-injected identically
every step (since u_theta is re-prescribed fresh each step, not evolved)?

Defines a diagnostic discrete projection operator

    P(v) = v - B G L^{-1} D(v)

built ENTIRELY out of the actual production functions
(pressure.solve_pressure_poisson, pressure.pressure_projection,
pressure.face_inv_rho_ur/uz, operators.grad_p_to_ur/uz_faces,
operators.divergence) -- no new physics, no new operator -- so it is
provably the same L = D B G the real solver uses.

Nothing in src/air_vortex is modified. All source substitutions (Source A/B)
are diagnostic-only, applied inside this script's own copy of the predictor
loop, never in solver.py.

Usage:
    python scripts/diagnose_source_projection.py --steps 200
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
from air_vortex.operators import (  # noqa: E402
    interp_center_to_ur, interp_center_to_uz, interp_ur_to_center, interp_uz_to_center,
    laplacian_ur, laplacian_uz, u_theta_on_ur_faces, upwind_derivative,
    uz_at_ur_locations, ur_at_uz_locations, divergence,
    grad_p_to_ur_faces, grad_p_to_uz_faces,
)
from air_vortex.pressure import (  # noqa: E402
    pressure_projection, solve_pressure_poisson, face_inv_rho_ur, face_inv_rho_uz,
)
from air_vortex.properties import material_properties  # noqa: E402
from air_vortex.surface_tension import surface_tension_force  # noqa: E402


# ============================================================
# 3. Discrete projection operator P(v) = v - B G L^{-1} D(v)
# ============================================================
def P(grid, v_r, v_z, rho, top_bc="open"):
    """Reuses the EXACT production L = D B G (solve_pressure_poisson +
    pressure_projection), with dt=1.0 so this acts purely as the discrete
    Helmholtz/Hodge projector: P(v) is v with its B-gradient component
    removed. Boundary handling is byte-identical to production: the SAME
    pre-solve zeroing solver.py applies to u_star at axis/wall/bottom
    (lines 141-144) is applied here to v BEFORE the Poisson RHS is built --
    omitting that step is not "no BC", it silently changes the BC (an
    earlier version of this script had exactly this bug: the RHS saw the
    raw, un-zeroed v, which is a different, uncontrolled boundary condition,
    not "no BC at all")."""
    v_r = v_r.copy()
    v_z = v_z.copy()
    v_r[0, :] = 0.0
    v_r[-1, :] = 0.0
    v_z[:, 0] = 0.0
    v_z[:, -1] = v_z[:, -2]
    p = solve_pressure_poisson(grid, v_r, v_z, rho, dt=1.0, method="direct", top_bc=top_bc)
    return pressure_projection(grid, v_r, v_z, p, rho, dt=1.0)


def norms(a):
    a = a[np.isfinite(a)]
    return float(np.max(np.abs(a))), float(np.sqrt(np.mean(a ** 2)))


# ============================================================
# Predictor building blocks (byte-identical formulas to solver.py)
# ============================================================
def source_centrifugal_actual(grid, u_theta):
    u_theta_ur = u_theta_on_ur_faces(grid, u_theta)
    r_f_safe = np.where(grid.r_f == 0.0, grid.dr, grid.r_f)
    S_r = u_theta_ur ** 2 / r_f_safe[:, None]
    S_z = np.zeros(grid.shape_uz)
    return S_r, S_z


def source_potential_gradient(grid, omega):
    """Diagnostic-only Source B: the SAME discrete gradient operator G used
    by the pressure solve, applied to the exact continuum potential
    Psi=0.5*Omega^2*r^2 (a pure function of r). Its z-component is set to
    exactly zero directly (not via grad_p_to_uz_faces): that function's top
    face bakes in the production pressure field's p=0-at-open-top gauge
    (operators.py: "dpdz[:,-1] = (0.0 - p[:,-1])/(0.5*dz)"), which is a
    statement about the ACTUAL pressure field, not a generic dp/dz operator
    -- applying it to an arbitrary Psi that isn't gauge-matched to 0 at the
    top produces a spurious nonzero top-face value with nothing to do with
    Psi's true (exactly zero, since Psi has no z-dependence) z-gradient."""
    Psi = 0.5 * omega ** 2 * grid.r_c[:, None] ** 2 * np.ones(grid.shape_center)
    S_r = grad_p_to_ur_faces(grid, Psi)
    S_z = np.zeros(grid.shape_uz)
    return S_r, S_z


def source_gravity(grid, g):
    S_r = np.zeros(grid.shape_ur)
    S_z = -g * np.ones(grid.shape_uz)
    return S_r, S_z


def momentum_pressure_step_custom(grid, cfg, u_r, u_z, u_theta, p, rho, mu, phi, dt, omega,
                                   centrifugal_override=None, top_bc="open"):
    """Same as diagnose_temporal_feedback.momentum_pressure_step, with an
    optional override for the centrifugal source term only (diagnostic
    Source A/B substitution) -- everything else (advection, viscosity,
    gravity, wall BC, pressure solve, projection) is untouched production code."""
    f_sigma_r, f_sigma_z = surface_tension_force(grid, phi, cfg)

    rho_ur = interp_center_to_ur(rho)
    rho_uz = interp_center_to_uz(rho)
    mu_ur = interp_center_to_ur(mu)
    mu_uz = interp_center_to_uz(mu)

    w_at_ur = uz_at_ur_locations(u_z)
    adv_r = u_r * upwind_derivative(u_r, u_r, grid.dr, axis=0) \
        + w_at_ur * upwind_derivative(u_r, w_at_ur, grid.dz, axis=1)

    if centrifugal_override is None:
        u_theta_ur = u_theta_on_ur_faces(grid, u_theta)
        r_f_safe = np.where(grid.r_f == 0.0, grid.dr, grid.r_f)
        centrifugal = u_theta_ur ** 2 / r_f_safe[:, None]
    else:
        centrifugal = centrifugal_override

    visc_r = mu_ur * laplacian_ur(grid, u_r)
    u_r_star = u_r + dt * (-adv_r + centrifugal + visc_r / rho_ur + f_sigma_r / rho_ur)

    u_at_uz = ur_at_uz_locations(u_r)
    adv_z = u_at_uz * upwind_derivative(u_z, u_at_uz, grid.dr, axis=0) \
        + u_z * upwind_derivative(u_z, u_z, grid.dz, axis=1)
    visc_z = mu_uz * laplacian_uz(grid, u_z)
    u_z_star = u_z + dt * (-adv_z - cfg.fluid.gravity + visc_z / rho_uz + f_sigma_z / rho_uz)

    u_theta_new = omega * grid.r_c[:, None] * np.ones_like(u_theta)

    u_r_star[0, :] = 0.0
    u_r_star[-1, :] = 0.0
    u_z_star[:, 0] = 0.0
    u_z_star[:, -1] = u_z_star[:, -2]

    if top_bc == "closed":
        u_z_star[:, -1] = 0.0  # rigid lid: no penetration at the top face too
    p_new = solve_pressure_poisson(grid, u_r_star, u_z_star, rho, dt, method="direct", p0=p, top_bc=top_bc)
    u_r_new, u_z_new = pressure_projection(grid, u_r_star, u_z_star, p_new, rho, dt)
    if top_bc == "closed":
        u_z_new[:, -1] = 0.0
    return u_r_new, u_z_new, u_theta_new, p_new


def run_frozen_multistep(grid, cfg, omega, dt, n_steps, phi0, rho0, mu0, centrifugal_override_fn=None,
                          top_bc="open"):
    u_r = np.zeros(grid.shape_ur)
    u_z = np.zeros(grid.shape_uz)
    u_theta = omega * grid.r_c[:, None] * np.ones(grid.shape_center)
    p = np.zeros(grid.shape_center)
    rows = []
    for step in range(1, n_steps + 1):
        override = centrifugal_override_fn(grid, omega) if centrifugal_override_fn else None
        u_r, u_z, u_theta, p = momentum_pressure_step_custom(
            grid, cfg, u_r, u_z, u_theta, p, rho0, mu0, phi0, dt, omega,
            centrifugal_override=override[0] if override else None, top_bc=top_bc)
        apply_velocity_bc(grid, type("F", (), {"u_r": u_r, "u_z": u_z, "u_theta": u_theta, "phi": phi0})())
        speed = np.sqrt(0.5 * (u_r[:-1, :] ** 2 + u_r[1:, :] ** 2) + 0.5 * (u_z[:, :-1] ** 2 + u_z[:, 1:] ** 2))
        rows.append({"step": step, "t": step * dt,
                     "max_abs_ur": float(np.max(np.abs(u_r))),
                     "max_abs_uz": float(np.max(np.abs(u_z))),
                     "max_meridional_speed": float(np.max(speed))})
    return pd.DataFrame(rows), (u_r, u_z, p)


def discrete_curl(grid, S_r, S_z):
    """Simple cell-centered estimate of the r-z-plane curl of a face-defined
    vector field: d(S_z)/dr - d(S_r)/dz, via central differences of the
    cell-centered interpolations. Zero for any field that's an exact
    discrete gradient in the continuum sense; used only as a diagnostic
    indicator, not a new physical operator."""
    Sr_c = interp_ur_to_center(S_r)
    Sz_c = interp_uz_to_center(S_z)
    curl = np.full(grid.shape_center, np.nan)
    curl[1:-1, 1:-1] = ((Sz_c[2:, 1:-1] - Sz_c[:-2, 1:-1]) / (2 * grid.dr)
                        - (Sr_c[1:-1, 2:] - Sr_c[1:-1, :-2]) / (2 * grid.dz))
    return curl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=200)
    args = parser.parse_args()

    cfg = load_config("configs/validation_solid_body.yaml")
    cfg.fluid.surface_tension = 0.0
    grid = build_grid(cfg)
    omega = 2 * np.pi * 100 / 60
    dt = 2.0e-4
    g = cfg.fluid.gravity
    rho_water = cfg.fluid.water_density

    out_dir = Path("results/diagnostics/source_projection")
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    fields0 = initialize_rotating_equilibrium(grid, cfg, omega)
    phi0_2phase = fields0.phi.copy()
    rho0_2phase, mu0_2phase = material_properties(phi0_2phase, grid, cfg)
    phi0_1phase = -1.0 * np.ones(grid.shape_center)
    rho0_1phase = rho_water * np.ones(grid.shape_center)
    mu0_1phase = mu0_2phase[0, 0] * np.ones(grid.shape_center) * 0 + cfg.fluid.water_viscosity

    u_theta0 = omega * grid.r_c[:, None] * np.ones(grid.shape_center)

    # ============================================================
    # SECTION 4/5: identity check -- does P(S_total) == real one-step Delta u?
    # ============================================================
    print("=== Identity check: P(S_total) vs actual one-step Delta u (single-phase) ===")
    S_cent_r, S_cent_z = source_centrifugal_actual(grid, u_theta0)
    S_grav_r, S_grav_z = source_gravity(grid, g)
    S_tot_r, S_tot_z = S_cent_r + S_grav_r, S_cent_z + S_grav_z

    P_tot_r, P_tot_z = P(grid, S_tot_r, S_tot_z, rho0_1phase)

    u_r0 = np.zeros(grid.shape_ur)
    u_z0 = np.zeros(grid.shape_uz)
    u_r_new, u_z_new, _, p_new = momentum_pressure_step_custom(
        grid, cfg, u_r0, u_z0, u_theta0, np.zeros(grid.shape_center),
        rho0_1phase, mu0_1phase, phi0_1phase, dt, omega)
    delta_u_actual_r = u_r_new / dt
    delta_u_actual_z = u_z_new / dt

    err_r = delta_u_actual_r - P_tot_r
    err_z = delta_u_actual_z - P_tot_z
    linf_r, l2_r = norms(err_r)
    linf_z, l2_z = norms(err_z)
    print(f"  ||Delta_u_actual - P(S_total)||_r: Linf={linf_r:.3e}  L2={l2_r:.3e}")
    print(f"  ||Delta_u_actual - P(S_total)||_z: Linf={linf_z:.3e}  L2={l2_z:.3e}")
    print(f"  (reference scale: max|Delta_u_actual_r|={np.max(np.abs(delta_u_actual_r)):.3e})")

    # ============================================================
    # SECTION 6/7: S_cent_actual vs discrete-potential-gradient source
    # ============================================================
    S_pot_r, S_pot_z = source_potential_gradient(grid, omega)
    diff = S_cent_r - S_pot_r
    print("\n=== Source comparison: S_cent_actual vs G(0.5 Omega^2 r^2) ===")
    axis_mask = np.zeros(grid.Nr + 1, dtype=bool); axis_mask[1] = True
    wall_mask = np.zeros(grid.Nr + 1, dtype=bool); wall_mask[-2] = True
    bulk_mask = ~axis_mask & ~wall_mask
    bulk_mask[0] = bulk_mask[-1] = False
    for name, mask in [("axis-near face (i=1)", axis_mask), ("bulk faces", bulk_mask),
                        ("wall-near face (i=Nr-1)", wall_mask)]:
        vals = diff[mask, :]
        if vals.size:
            linf, l2 = norms(vals)
            rel = linf / max(1e-30, np.max(np.abs(S_pot_r[mask, :])))
            print(f"  {name}: Linf={linf:.4e}  L2={l2:.4e}  relative={rel:.4%}")

    face_err_df = pd.DataFrame({
        "face_index": np.arange(grid.Nr + 1),
        "r_face_mm": grid.r_f * 1e3,
        "S_cent_actual_mid": S_cent_r[:, grid.Nz // 2],
        "S_pot_grad_mid": S_pot_r[:, grid.Nz // 2],
        "u_theta_face_mid": u_theta_on_ur_faces(grid, u_theta0)[:, grid.Nz // 2],
        "u_theta_exact_mid": omega * grid.r_f,
    })
    face_err_df["u_theta_interp_error"] = face_err_df["u_theta_face_mid"] - face_err_df["u_theta_exact_mid"]
    face_err_df.to_csv(out_dir / "face_interpolation.csv", index=False)
    print(f"  saved {out_dir / 'face_interpolation.csv'}")

    P_cent_r, P_cent_z = P(grid, S_cent_r, S_cent_z, rho0_1phase)
    P_pot_r, P_pot_z = P(grid, S_pot_r, S_pot_z, rho0_1phase)
    linf_c, l2_c = norms(P_cent_r); linf_p, l2_p = norms(P_pot_r)
    print(f"\n  ||P(S_cent_actual)||: Linf={linf_c:.3e}  L2={l2_c:.3e}")
    print(f"  ||P(S_potential_grad)||: Linf={linf_p:.3e}  L2={l2_p:.3e}  (expect ~machine precision)")

    # gradient-annihilation identity with an ARBITRARY smooth q (physics-independent)
    q_arbitrary = (grid.r_c[:, None] ** 2 + grid.z_c[None, :] ** 2) * np.ones(grid.shape_center)
    Bq_r = face_inv_rho_ur(rho0_1phase) * grad_p_to_ur_faces(grid, q_arbitrary)
    Bq_z = face_inv_rho_uz(rho0_1phase) * grad_p_to_uz_faces(grid, q_arbitrary)
    P_Bq_r, P_Bq_z = P(grid, Bq_r, Bq_z, rho0_1phase)
    linf_bq, l2_bq = norms(P_Bq_r)
    print(f"  ||P(B G(q_arbitrary))||: Linf={linf_bq:.3e}  L2={l2_bq:.3e}  (pure L=DBG identity check)")

    # ============================================================
    # SECTION 6/7, CORRECTED: same P(S_cent) test but with a physically
    # matched BC -- top_bc="closed" (rigid lid), no gravity. The "open"
    # top_bc used above imposes p=0 at a FLAT top for every r, which is
    # only physically correct where the domain's top actually coincides
    # with a (possibly curved) free surface at p=0 -- for the two-phase
    # exact-equilibrium IC that's approximately true (the top few rows are
    # air, p~0 there anyway), but for a single-phase, all-water domain with
    # NO free surface, a flat p=0 top is not a valid boundary condition for
    # solid-body rotation (whose equilibrium p at any flat horizontal slice
    # varies with r, is not uniformly 0) -- it's an artifact of testing a
    # single-phase domain with the free-surface-specific "open" BC.
    # tests/test_rotation.py already validates the ONE-STEP closed-top case
    # to <1e-6 m/s; this repeats that with S_cent_actual/S_pot AND runs it
    # multi-step, which that existing test does not.
    # ============================================================
    cfg_nograv = load_config("configs/validation_solid_body.yaml")
    cfg_nograv.fluid.surface_tension = 0.0
    cfg_nograv.fluid.gravity = 0.0

    P_cent_r_closed, _ = P(grid, S_cent_r, S_cent_z, rho0_1phase, top_bc="closed")
    P_pot_r_closed, _ = P(grid, S_pot_r, S_pot_z, rho0_1phase, top_bc="closed")
    linf_c_cl, l2_c_cl = norms(P_cent_r_closed)
    linf_p_cl, l2_p_cl = norms(P_pot_r_closed)
    print("\n=== SAME test, corrected BC: top_bc='closed' (rigid lid), no gravity ===")
    print(f"  ||P(S_cent_actual)||_closed: Linf={linf_c_cl:.3e}  L2={l2_c_cl:.3e}")
    print(f"  ||P(S_potential_grad)||_closed: Linf={linf_p_cl:.3e}  L2={l2_p_cl:.3e}")

    # ============================================================
    # SECTION 1: single-phase (A0) vs frozen two-phase physical (A1), multi-step
    # ============================================================
    print(f"\n=== Multi-step control: A0 (single-phase) vs A1 (frozen two-phase), {args.steps} steps ===")
    df_A0, _ = run_frozen_multistep(grid, cfg, omega, dt, args.steps, phi0_1phase, rho0_1phase, mu0_1phase)
    df_A1, _ = run_frozen_multistep(grid, cfg, omega, dt, args.steps, phi0_2phase, rho0_2phase, mu0_2phase)
    df_A0.to_csv(out_dir / "multistep_A0_single_phase.csv", index=False)
    df_A1.to_csv(out_dir / "multistep_A1_frozen_twophase.csv", index=False)
    print(f"  A0 (single-phase): step1={df_A0['max_meridional_speed'].iloc[0]:.3e}  "
          f"step{args.steps}={df_A0['max_meridional_speed'].iloc[-1]:.3e}")
    print(f"  A1 (frozen 2-phase): step1={df_A1['max_meridional_speed'].iloc[0]:.3e}  "
          f"step{args.steps}={df_A1['max_meridional_speed'].iloc[-1]:.3e}")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(df_A0["step"], df_A0["max_meridional_speed"], label="A0: single-phase uniform")
    ax.plot(df_A1["step"], df_A1["max_meridional_speed"], label="A1: frozen two-phase physical")
    ax.set_xlabel("step"); ax.set_ylabel("max meridional speed [m/s]")
    ax.set_title("Single-phase vs frozen two-phase growth")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "single_vs_twophase_growth.png", dpi=200)

    print(f"\n=== Multi-step control, CORRECTED BC: A0_closed (single-phase, rigid lid, no gravity) ===")
    df_A0_closed, _ = run_frozen_multistep(grid, cfg_nograv, omega, dt, args.steps,
                                            phi0_1phase, rho0_1phase, mu0_1phase, top_bc="closed")
    df_A0_closed.to_csv(out_dir / "multistep_A0_closed_single_phase.csv", index=False)
    print(f"  A0_closed: step1={df_A0_closed['max_meridional_speed'].iloc[0]:.3e}  "
          f"step{args.steps}={df_A0_closed['max_meridional_speed'].iloc[-1]:.3e}")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.semilogy(df_A0["step"], df_A0["max_meridional_speed"].clip(lower=1e-18), label="A0: single-phase, OPEN top (mismatched BC)")
    ax.semilogy(df_A1["step"], df_A1["max_meridional_speed"].clip(lower=1e-18), label="A1: frozen two-phase, OPEN top (production-matched)")
    ax.semilogy(df_A0_closed["step"], df_A0_closed["max_meridional_speed"].clip(lower=1e-18), label="A0_closed: single-phase, CLOSED lid, no gravity")
    ax.set_xlabel("step"); ax.set_ylabel("max meridional speed [m/s] (log)")
    ax.set_title("Effect of top BC choice on single-phase rotation preservation")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "single_vs_twophase_growth_with_closed_control.png", dpi=200)

    # ============================================================
    # SECTION 2: density-ratio sweep, MULTI-STEP
    # ============================================================
    print(f"\n=== Density-ratio sweep, multi-step ({args.steps} steps each) ===")
    ratios = [1.0, 2.0, 10.0, 100.0, rho_water / 1.2]
    ratio_rows = []
    fig, ax = plt.subplots(figsize=(7, 5))
    for ratio in ratios:
        rho_air = rho_water / ratio
        rho_ratio = np.where(phi0_2phase < 0, rho_water, rho_air)
        # smooth via the same material_properties call, but force the ratio:
        cfg_r = load_config("configs/validation_solid_body.yaml")
        cfg_r.fluid.surface_tension = 0.0
        cfg_r.fluid.water_density = rho_water
        cfg_r.fluid.air_density = rho_air
        rho_r, mu_r = material_properties(phi0_2phase, grid, cfg_r)
        df_r, _ = run_frozen_multistep(grid, cfg_r, omega, dt, args.steps, phi0_2phase, rho_r, mu_r)
        df_r.to_csv(out_dir / f"multistep_density_ratio_{ratio:.0f}.csv", index=False)
        early_slope = (df_r["max_meridional_speed"].iloc[9] - df_r["max_meridional_speed"].iloc[0]) / 9
        ratio_rows.append({
            "density_ratio": ratio,
            "step1_residual": df_r["max_meridional_speed"].iloc[0],
            "step200_speed": df_r["max_meridional_speed"].iloc[-1],
            "early_growth_slope_per_step": early_slope,
        })
        ax.plot(df_r["step"], df_r["max_meridional_speed"], label=f"ratio={ratio:.0f}")
        print(f"  ratio={ratio:7.1f}: step1={df_r['max_meridional_speed'].iloc[0]:.3e}  "
              f"step{args.steps}={df_r['max_meridional_speed'].iloc[-1]:.3e}")
    ax.set_xlabel("step"); ax.set_ylabel("max meridional speed [m/s]")
    ax.set_title("Multi-step growth vs density ratio (phi/material frozen)")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "density_ratio_multistep.png", dpi=200)
    ratio_df = pd.DataFrame(ratio_rows)
    ratio_df.to_csv(out_dir / "multistep_density_ratio.csv", index=False)
    print(ratio_df.to_string(index=False))

    # ============================================================
    # SECTION 7: Source A/B multi-step growth comparison
    # ============================================================
    print(f"\n=== Source A/B multi-step test (single-phase, {args.steps} steps) ===")
    df_sourceA, _ = run_frozen_multistep(grid, cfg, omega, dt, args.steps, phi0_1phase, rho0_1phase, mu0_1phase,
                                          centrifugal_override_fn=lambda g_, o_: source_centrifugal_actual(g_, u_theta0))
    df_sourceB, _ = run_frozen_multistep(grid, cfg, omega, dt, args.steps, phi0_1phase, rho0_1phase, mu0_1phase,
                                          centrifugal_override_fn=lambda g_, o_: source_potential_gradient(g_, o_))
    df_sourceA.to_csv(out_dir / "source_ab_comparison_A.csv", index=False)
    df_sourceB.to_csv(out_dir / "source_ab_comparison_B.csv", index=False)
    print(f"  Source A (current u_theta^2/r):   step1={df_sourceA['max_meridional_speed'].iloc[0]:.3e}  "
          f"step{args.steps}={df_sourceA['max_meridional_speed'].iloc[-1]:.3e}")
    print(f"  Source B (G(0.5 Omega^2 r^2)):    step1={df_sourceB['max_meridional_speed'].iloc[0]:.3e}  "
          f"step{args.steps}={df_sourceB['max_meridional_speed'].iloc[-1]:.3e}")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.semilogy(df_sourceA["step"], df_sourceA["max_meridional_speed"].clip(lower=1e-18), label="Source A: current u_theta^2/r")
    ax.semilogy(df_sourceB["step"], df_sourceB["max_meridional_speed"].clip(lower=1e-18), label="Source B: discrete potential grad")
    ax.set_xlabel("step"); ax.set_ylabel("max meridional speed [m/s] (log)")
    ax.set_title("Source A/B growth comparison")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "source_ab_growth.png", dpi=200)

    print(f"\n=== Source A/B multi-step test, CORRECTED BC: closed lid, no gravity ({args.steps} steps) ===")
    df_sourceA_cl, _ = run_frozen_multistep(grid, cfg_nograv, omega, dt, args.steps, phi0_1phase, rho0_1phase, mu0_1phase,
                                             centrifugal_override_fn=lambda g_, o_: source_centrifugal_actual(g_, u_theta0),
                                             top_bc="closed")
    df_sourceB_cl, _ = run_frozen_multistep(grid, cfg_nograv, omega, dt, args.steps, phi0_1phase, rho0_1phase, mu0_1phase,
                                             centrifugal_override_fn=lambda g_, o_: source_potential_gradient(g_, o_),
                                             top_bc="closed")
    df_sourceA_cl.to_csv(out_dir / "source_ab_comparison_A_closed.csv", index=False)
    df_sourceB_cl.to_csv(out_dir / "source_ab_comparison_B_closed.csv", index=False)
    print(f"  Source A, closed lid:   step1={df_sourceA_cl['max_meridional_speed'].iloc[0]:.3e}  "
          f"step{args.steps}={df_sourceA_cl['max_meridional_speed'].iloc[-1]:.3e}")
    print(f"  Source B, closed lid:   step1={df_sourceB_cl['max_meridional_speed'].iloc[0]:.3e}  "
          f"step{args.steps}={df_sourceB_cl['max_meridional_speed'].iloc[-1]:.3e}")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.semilogy(df_sourceA["step"], df_sourceA["max_meridional_speed"].clip(lower=1e-18), label="Source A, OPEN top")
    ax.semilogy(df_sourceB["step"], df_sourceB["max_meridional_speed"].clip(lower=1e-18), label="Source B, OPEN top")
    ax.semilogy(df_sourceA_cl["step"], df_sourceA_cl["max_meridional_speed"].clip(lower=1e-18), "--", label="Source A, CLOSED lid")
    ax.semilogy(df_sourceB_cl["step"], df_sourceB_cl["max_meridional_speed"].clip(lower=1e-18), "--", label="Source B, CLOSED lid")
    ax.set_xlabel("step"); ax.set_ylabel("max meridional speed [m/s] (log)")
    ax.set_title("Source A/B growth, open vs closed top BC")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "source_ab_growth_with_closed.png", dpi=200)

    # ============================================================
    # SECTION 10/11: discrete circulation maps
    # ============================================================
    curl_cent = discrete_curl(grid, S_cent_r, S_cent_z)
    curl_pot = discrete_curl(grid, S_pot_r, S_pot_z)
    # curl of rho*S (force density) for the frozen two-phase case
    S_cent_r_2p, S_cent_z_2p = source_centrifugal_actual(grid, u_theta0)
    rho_face_ur = 1.0 / np.where(face_inv_rho_ur(rho0_2phase) > 0, face_inv_rho_ur(rho0_2phase), 1.0)
    rho_face_uz = 1.0 / np.where(face_inv_rho_uz(rho0_2phase) > 0, face_inv_rho_uz(rho0_2phase), 1.0)
    F_r = rho_face_ur * S_cent_r_2p
    F_z = rho_face_uz * S_cent_z_2p
    curl_F = discrete_curl(grid, F_r, F_z)

    eps_band = np.abs(phi0_2phase) < 3 * cfg.levelset.interface_width_cells * 0.5 * (grid.dr + grid.dz)
    wall_band = grid.r_c > 0.7 * grid.r_v
    bulk_band = ~eps_band & ~wall_band[:, None]

    def region_stat(field, mask):
        vals = field[mask]
        vals = vals[np.isfinite(vals)]
        return float(np.max(np.abs(vals))) if vals.size else float("nan")

    circ_rows = [
        {"region": "bulk", "current_source_curl": region_stat(curl_cent, bulk_band),
         "potential_source_curl": region_stat(curl_pot, bulk_band),
         "rho_S_curl_2phase": region_stat(curl_F, bulk_band)},
        {"region": "interface_band", "current_source_curl": region_stat(curl_cent, eps_band),
         "potential_source_curl": region_stat(curl_pot, eps_band),
         "rho_S_curl_2phase": region_stat(curl_F, eps_band)},
        {"region": "wall_adjacent", "current_source_curl": region_stat(curl_cent, wall_band[:, None] * np.ones_like(curl_cent, dtype=bool)),
         "potential_source_curl": region_stat(curl_pot, wall_band[:, None] * np.ones_like(curl_pot, dtype=bool)),
         "rho_S_curl_2phase": region_stat(curl_F, wall_band[:, None] * np.ones_like(curl_F, dtype=bool))},
    ]
    circ_df = pd.DataFrame(circ_rows)
    circ_df.to_csv(out_dir / "circulation.csv", index=False)
    print("\n=== Discrete circulation (max|curl|) by region ===")
    print(circ_df.to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, field, title in [(axes[0], curl_cent, "curl(S_cent_actual), single-phase"),
                              (axes[1], curl_F, "curl(rho * S_cent), frozen two-phase")]:
        im = ax.pcolormesh(grid.r_c * 1e3, grid.z_c * 1e3, field.T, shading="auto", cmap="RdBu_r")
        fig.colorbar(im, ax=ax)
        ax.contour(grid.r_c * 1e3, grid.z_c * 1e3, phi0_2phase.T, levels=[0.0], colors="k", linewidths=1.0)
        ax.set_title(title); ax.set_xlabel("r [mm]"); ax.set_ylabel("z [mm]")
    fig.tight_layout()
    fig.savefig(fig_dir / "discrete_curl_map.png", dpi=200)

    # ============================================================
    # SECTION 13: per-step residual prediction vs actual (early-time slope)
    # ============================================================
    eps_pred = float(np.max(np.abs(P_tot_r)))
    print(f"\n=== Linear-accumulation prediction check (single-phase, epsilon={eps_pred:.4e}) ===")
    pred_rows = []
    for N in [1, 5, 10, 20, 50]:
        U_pred = N * dt * eps_pred
        U_actual = df_A0[df_A0["step"] == N]["max_meridional_speed"].values[0]
        pred_rows.append({"N": N, "U_predicted_linear": U_pred, "U_actual": U_actual,
                           "ratio": U_actual / U_pred if U_pred else float("nan")})
    pred_df = pd.DataFrame(pred_rows)
    pred_df.to_csv(out_dir / "predicted_vs_actual_growth.csv", index=False)
    print(pred_df.to_string(index=False))

    fig, ax = plt.subplots(figsize=(7, 5))
    Ns = np.array(pred_df["N"])
    ax.plot(Ns, pred_df["U_predicted_linear"], "o--", label="predicted (N*dt*eps)")
    ax.plot(Ns, pred_df["U_actual"], "s-", label="actual (A0)")
    ax.set_xlabel("step N"); ax.set_ylabel("max meridional speed [m/s]")
    ax.set_title("Linear-accumulation prediction vs actual, early time")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "predicted_vs_actual_growth.png", dpi=200)

    # ============================================================
    # summary
    # ============================================================
    summary = {
        "identity_check_Linf_r": linf_r, "identity_check_L2_r": l2_r,
        "P_S_cent_actual_Linf": linf_c, "P_S_potential_grad_Linf": linf_p,
        "P_BGq_arbitrary_Linf": linf_bq,
        "A0_step1": df_A0["max_meridional_speed"].iloc[0], f"A0_step{args.steps}": df_A0["max_meridional_speed"].iloc[-1],
        "A1_step1": df_A1["max_meridional_speed"].iloc[0], f"A1_step{args.steps}": df_A1["max_meridional_speed"].iloc[-1],
        "sourceA_step1": df_sourceA["max_meridional_speed"].iloc[0], f"sourceA_step{args.steps}": df_sourceA["max_meridional_speed"].iloc[-1],
        "sourceB_step1": df_sourceB["max_meridional_speed"].iloc[0], f"sourceB_step{args.steps}": df_sourceB["max_meridional_speed"].iloc[-1],
    }
    pd.DataFrame([summary]).to_csv(out_dir / "summary.csv", index=False)
    print(f"\nSaved {out_dir / 'summary.csv'}")
    print("\nAll figures saved under", fig_dir)


if __name__ == "__main__":
    main()
