"""DIAGNOSTIC (not a validated/gated script) -- Session 7 investigation,
temporal-feedback isolation.

Wall pressure BC and the density-jump pressure discretization have both
been cleared as the primary one-step defect (see
scripts/diag_wall_interface_density.py and the Session 7 report). This
script isolates WHICH stage of the per-step loop

    momentum predictor -> pressure projection -> Level Set advection
    -> reinitialization -> rho/mu re-materialization

is responsible for turning a ~1e-4 m/s one-step residual into the
~1e-1 m/s spurious meridional flow observed after ~4000 steps of the
exact rotating equilibrium (sigma=0) run.

Five controlled cases, all from the SAME exact-equilibrium IC, SAME grid,
SAME fixed dt, SAME number of steps:

    A  : phi frozen,            rho/mu frozen,        momentum evolves
    B  : phi advects,           rho/mu frozen,        momentum evolves
    C  : phi advects,           rho/mu re-materializes, momentum evolves  (= production loop)
    D1 : phi advects,           velocity forced exact (u_r=u_z=0, u_theta=Omega r), no reinit
    D2 : phi advects + reinit,  velocity forced exact,

Plus a single-call idempotency check (Case E) and a discrete-balanced
pressure IC comparison (IC-1 continuum vs IC-2 discrete-balanced), both run
as Case A.

No production solver code is modified. This script only orchestrates the
existing operators/functions in a different sequence for isolation.

Usage:
    python scripts/diagnose_temporal_feedback.py --steps 200
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
from air_vortex.levelset import advect_level_set_configurable, reinitialize_level_set  # noqa: E402
from air_vortex.operators import (  # noqa: E402
    interp_center_to_ur, interp_center_to_uz, laplacian_ur, laplacian_uz,
    u_theta_on_ur_faces, upwind_derivative, uz_at_ur_locations,
    ur_at_uz_locations, divergence, grad_p_to_ur_faces,
)
from air_vortex.pressure import pressure_projection, solve_pressure_poisson, face_inv_rho_ur  # noqa: E402
from air_vortex.properties import material_properties, interface_epsilon  # noqa: E402
from air_vortex.surface_tension import surface_tension_force  # noqa: E402
from air_vortex.timestep import compute_stable_timestep  # noqa: E402
from air_vortex.diagnostics import free_surface_height  # noqa: E402
from air_vortex.solver import water_volume  # noqa: E402


def momentum_pressure_step(grid, cfg, u_r, u_z, u_theta, p, rho, mu, phi, dt, omega):
    """Exactly the predictor + wall-BC + pressure-Poisson + projection
    sequence from Solver.step() (solver.py lines ~92-152), parameterized on
    an externally-supplied rho/mu instead of recomputing them -- so Cases
    A/B can freeze material properties while still running the identical
    production momentum/pressure code path."""
    f_sigma_r, f_sigma_z = surface_tension_force(grid, phi, cfg)

    rho_ur = interp_center_to_ur(rho)
    rho_uz = interp_center_to_uz(rho)
    mu_ur = interp_center_to_ur(mu)
    mu_uz = interp_center_to_uz(mu)

    w_at_ur = uz_at_ur_locations(u_z)
    adv_r = u_r * upwind_derivative(u_r, u_r, grid.dr, axis=0) \
        + w_at_ur * upwind_derivative(u_r, w_at_ur, grid.dz, axis=1)
    u_theta_ur = u_theta_on_ur_faces(grid, u_theta)
    r_f_safe = np.where(grid.r_f == 0.0, grid.dr, grid.r_f)
    centrifugal = u_theta_ur**2 / r_f_safe[:, None]
    visc_r = mu_ur * laplacian_ur(grid, u_r)
    u_r_star = u_r + dt * (-adv_r + centrifugal + visc_r / rho_ur + f_sigma_r / rho_ur)

    u_at_uz = ur_at_uz_locations(u_r)
    adv_z = u_at_uz * upwind_derivative(u_z, u_at_uz, grid.dr, axis=0) \
        + u_z * upwind_derivative(u_z, u_z, grid.dz, axis=1)
    visc_z = mu_uz * laplacian_uz(grid, u_z)
    u_z_star = u_z + dt * (-adv_z - cfg.fluid.gravity + visc_z / rho_uz + f_sigma_z / rho_uz)

    u_theta_new = omega * grid.r_c[:, None] * np.ones_like(u_theta)  # prescribed swirl

    u_r_star[0, :] = 0.0
    u_r_star[-1, :] = 0.0
    u_z_star[:, 0] = 0.0
    u_z_star[:, -1] = u_z_star[:, -2]

    p_new = solve_pressure_poisson(grid, u_r_star, u_z_star, rho, dt, method="direct", p0=p)
    u_r_new, u_z_new = pressure_projection(grid, u_r_star, u_z_star, p_new, rho, dt)

    dpdr = grad_p_to_ur_faces(grid, p_new)
    return u_r_new, u_z_new, u_theta_new, p_new, dpdr, centrifugal, adv_r


def build_discrete_balanced_pressure(grid, cfg, phi0, rho0, omega, p_continuum):
    """IC-2: same z-structure as the continuum analytical pressure (hydrostatic
    part, already validated separately), but the RADIAL integration is done
    using the solver's own DISCRETE centrifugal term and face-averaged rho,
    at the SAME faces grad_p_to_ur_faces will later differentiate -- so that
    dp/dr (discrete) matches the discrete centrifugal term face-for-face, by
    construction, rather than merely matching the continuum formula."""
    u_theta0 = omega * grid.r_c[:, None] * np.ones(grid.shape_center)
    u_theta_ur = u_theta_on_ur_faces(grid, u_theta0)
    r_f_safe = np.where(grid.r_f == 0.0, grid.dr, grid.r_f)
    centrifugal = u_theta_ur**2 / r_f_safe[:, None]  # (Nr+1, Nz)

    inv_rho_ur = face_inv_rho_ur(rho0)
    rho_face = np.where(inv_rho_ur > 0, 1.0 / np.where(inv_rho_ur > 0, inv_rho_ur, 1.0), rho0[0, 0])

    p_discrete = np.empty_like(p_continuum)
    p_discrete[0, :] = p_continuum[0, :]  # anchor r=r_c[0] to the continuum value (preserves z-structure)
    for i in range(1, grid.Nr):
        p_discrete[i, :] = p_discrete[i - 1, :] + grid.dr * rho_face[i, :] * centrifugal[i, :]
    return p_discrete


def run_case(label, grid, cfg, omega, dt, n_steps, phi0, rho0, mu0, p_init,
             evolve_phi, evolve_material, force_exact_velocity, do_reinit):
    u_r = np.zeros(grid.shape_ur)
    u_z = np.zeros(grid.shape_uz)
    u_theta = omega * grid.r_c[:, None] * np.ones(grid.shape_center)
    p = p_init.copy()
    phi = phi0.copy()
    rho, mu = rho0.copy(), mu0.copy()

    eps = interface_epsilon(grid, cfg)
    interface_band0 = np.abs(phi0) < 2 * eps
    wall_roi = grid.r_c > 0.7 * grid.r_v
    eta0 = free_surface_height(phi0, grid)
    V0 = water_volume(grid, phi0)

    rows = []
    prev_ur, prev_uz = u_r.copy(), u_z.copy()

    for step in range(1, n_steps + 1):
        if force_exact_velocity:
            u_r[:] = 0.0
            u_z[:] = 0.0
            u_theta = omega * grid.r_c[:, None] * np.ones(grid.shape_center)
            dpdr = np.zeros(grid.shape_ur)
            centrifugal = np.zeros(grid.shape_ur)
            adv_r = np.zeros(grid.shape_ur)
        else:
            if evolve_material:
                rho, mu = material_properties(phi, grid, cfg)
            u_r, u_z, u_theta, p, dpdr, centrifugal, adv_r = momentum_pressure_step(
                grid, cfg, u_r, u_z, u_theta, p, rho, mu, phi, dt, omega)
            apply_velocity_bc(grid, type("F", (), {"u_r": u_r, "u_z": u_z, "u_theta": u_theta,
                                                     "phi": phi})())

        if evolve_phi:
            phi = advect_level_set_configurable(
                phi, u_r, u_z, grid, dt,
                scheme=cfg.levelset.advection_scheme,
                time_integrator=cfg.levelset.time_integrator,
                limiter=cfg.levelset.limiter,
            )
            if do_reinit and step % cfg.levelset.reinitialize_every == 0:
                phi = reinitialize_level_set(phi, grid, cfg.levelset.reinitialize_iterations)

        speed = np.sqrt(
            0.5 * (u_r[:-1, :] ** 2 + u_r[1:, :] ** 2)
            + 0.5 * (u_z[:, :-1] ** 2 + u_z[:, 1:] ** 2)
        )
        div = divergence(grid, u_r, u_z)

        water_cell = phi < 0.0
        wall_cell_mask = wall_roi
        interface_cell_mask = np.abs(phi) < 2 * eps
        wall_interface_mask = wall_cell_mask[:, None] & interface_cell_mask if interface_cell_mask.ndim == 2 else wall_cell_mask[:, None] & interface_cell_mask

        ke = float(np.sum(0.5 * water_cell * (speed ** 2) * (2 * np.pi * grid.r_c[:, None] * grid.dr * grid.dz)))
        V = water_volume(grid, phi)
        eta = free_surface_height(phi, grid)
        valid = np.isfinite(eta) & np.isfinite(eta0)
        interface_rms = float(np.sqrt(np.mean((eta[valid] - eta0[valid]) ** 2))) if np.any(valid) else float("nan")

        d_ur = u_r - prev_ur
        d_uz = u_z - prev_uz
        wall_ur_mask = np.zeros(grid.shape_ur, dtype=bool)
        wall_ur_mask[int(0.7 * grid.Nr):, :] = True

        rows.append({
            "step": step, "t": step * dt,
            "max_abs_ur": float(np.max(np.abs(u_r))),
            "max_abs_uz": float(np.max(np.abs(u_z))),
            "max_meridional_speed": float(np.max(speed)),
            "meridional_KE": ke,
            "max_divergence": float(np.max(np.abs(div))),
            "water_volume": V,
            "relative_volume_drift": (V - V0) / V0,
            "phi_L2_change": float(np.sqrt(np.mean((phi - phi0) ** 2))),
            "phi_Linf_change": float(np.max(np.abs(phi - phi0))),
            "rho_L2_change": float(np.sqrt(np.mean((rho - rho0) ** 2))),
            "interface_rms_displacement": interface_rms,
            "mean_delta_ur_wall": float(np.mean(d_ur[wall_ur_mask])),
            "mean_delta_uz_wall": float(np.mean(d_uz[wall_roi, :])),
            "max_abs_dpdr_minus_centrifugal_rho": float(np.max(np.abs(dpdr - centrifugal * np.where(
                face_inv_rho_ur(rho) > 0, 1.0 / np.where(face_inv_rho_ur(rho) > 0, face_inv_rho_ur(rho), 1.0), 1.0)))),
        })
        prev_ur, prev_uz = u_r.copy(), u_z.copy()

    df = pd.DataFrame(rows)
    return df, dict(u_r=u_r, u_z=u_z, phi=phi, rho=rho, p=p)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=200)
    args = parser.parse_args()

    cfg = load_config("configs/validation_solid_body.yaml")
    cfg.fluid.surface_tension = 0.0
    cfg.levelset.advection_scheme = "muscl2"
    cfg.levelset.time_integrator = "ssprk2"
    grid = build_grid(cfg)
    omega = 2 * np.pi * 100 / 60

    fields0 = initialize_rotating_equilibrium(grid, cfg, omega)
    phi0 = fields0.phi.copy()
    p0_continuum = fields0.p.copy()
    rho0, mu0 = material_properties(phi0, grid, cfg)

    dt = compute_stable_timestep(grid, fields0.u_r, fields0.u_z, rho0.min(), mu0.max(), cfg)
    print(f"Fixed dt used for ALL cases: {dt:.6e} s ({args.steps} steps -> t_end={dt*args.steps:.4f}s)")

    # idempotency check (Case E, single call, cheap)
    rho_check, mu_check = material_properties(phi0, grid, cfg)
    print(f"material_properties idempotency check: max|rho_check-rho0| = "
          f"{np.max(np.abs(rho_check-rho0)):.3e} (expect 0.0, exact machine precision)")

    out_dir = Path("results/diagnostics/temporal_feedback")
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    cases_spec = [
        ("A", dict(evolve_phi=False, evolve_material=False, force_exact_velocity=False, do_reinit=False)),
        ("B", dict(evolve_phi=True, evolve_material=False, force_exact_velocity=False, do_reinit=True)),
        ("C", dict(evolve_phi=True, evolve_material=True, force_exact_velocity=False, do_reinit=True)),
        ("D1", dict(evolve_phi=True, evolve_material=False, force_exact_velocity=True, do_reinit=False)),
        ("D2", dict(evolve_phi=True, evolve_material=False, force_exact_velocity=True, do_reinit=True)),
    ]

    results = {}
    for label, kwargs in cases_spec:
        print(f"Running Case {label} ({args.steps} steps)...")
        df, final = run_case(label, grid, cfg, omega, dt, args.steps, phi0, rho0, mu0,
                              p0_continuum, **kwargs)
        df.to_csv(out_dir / f"timeseries_case_{label}.csv", index=False)
        results[label] = (df, final)
        print(f"  final max|u_r|={df['max_abs_ur'].iloc[-1]:.4e}  "
              f"max|u_z|={df['max_abs_uz'].iloc[-1]:.4e}  "
              f"phi_L2_change={df['phi_L2_change'].iloc[-1]:.4e}  "
              f"rho_L2_change={df['rho_L2_change'].iloc[-1]:.4e}")

    # discrete-balanced IC comparison, run as Case A with IC-2
    p0_discrete = build_discrete_balanced_pressure(grid, cfg, phi0, rho0, omega, p0_continuum)
    print("Running Case A with discrete-balanced pressure IC (IC-2)...")
    df_A_discrete, _ = run_case("A_discrete", grid, cfg, omega, dt, args.steps, phi0, rho0, mu0,
                                 p0_discrete, evolve_phi=False, evolve_material=False,
                                 force_exact_velocity=False, do_reinit=False)
    df_A_discrete.to_csv(out_dir / "timeseries_case_A_discrete.csv", index=False)
    print(f"  Case A (discrete IC) final max|u_r|={df_A_discrete['max_abs_ur'].iloc[-1]:.4e}  "
          f"max|u_z|={df_A_discrete['max_abs_uz'].iloc[-1]:.4e}")

    # ---------------- Figure 1: max meridional velocity vs step ----------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for label, (df, _) in results.items():
        axes[0].plot(df["step"], df["max_meridional_speed"], label=f"Case {label}")
    axes[0].plot(df_A_discrete["step"], df_A_discrete["max_meridional_speed"],
                 "--", label="Case A (discrete-balanced IC)")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("max meridional speed [m/s]")
    axes[0].set_title("Linear scale")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    for label, (df, _) in results.items():
        axes[1].semilogy(df["step"], df["max_meridional_speed"].clip(lower=1e-16), label=f"Case {label}")
    axes[1].semilogy(df_A_discrete["step"], df_A_discrete["max_meridional_speed"].clip(lower=1e-16),
                      "--", label="Case A (discrete-balanced IC)")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("max meridional speed [m/s] (log)")
    axes[1].set_title("Log scale")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)
    fig.suptitle("Figure 1: max meridional velocity growth by mechanism")
    fig.tight_layout()
    fig.savefig(fig_dir / "diagnostic_growth_cases.png", dpi=200)
    fig.savefig(fig_dir / "diagnostic_growth_cases.pdf")
    print(f"Saved {fig_dir / 'diagnostic_growth_cases.png'}")

    # ---------------- Figure 2: phi displacement vs step ----------------
    fig, ax = plt.subplots(figsize=(7, 5))
    for label, (df, _) in results.items():
        ax.semilogy(df["step"], df["phi_L2_change"].clip(lower=1e-18), label=f"Case {label}")
    ax.set_xlabel("step")
    ax.set_ylabel("||phi^n - phi^0||_L2 (log)")
    ax.set_title("Figure 2: interface displacement by mechanism")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "phi_displacement.png", dpi=200)
    print(f"Saved {fig_dir / 'phi_displacement.png'}")

    # ---------------- Figure 3: density displacement vs step ----------------
    fig, ax = plt.subplots(figsize=(7, 5))
    for label, (df, _) in results.items():
        ax.semilogy(df["step"], df["rho_L2_change"].clip(lower=1e-18), label=f"Case {label}")
    ax.set_xlabel("step")
    ax.set_ylabel("||rho^n - rho^0||_L2 (log)")
    ax.set_title("Figure 3: material-property (re-materialization) displacement")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "rho_displacement.png", dpi=200)
    print(f"Saved {fig_dir / 'rho_displacement.png'}")

    # ---------------- Figure 4: per-step signed velocity increment, wall ROI ----------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for label, (df, _) in results.items():
        axes[0].plot(df["step"], df["mean_delta_ur_wall"], label=f"Case {label}")
        axes[1].plot(df["step"], df["mean_delta_ur_wall"].cumsum(), label=f"Case {label}")
    axes[0].axhline(0, color="k", lw=0.5)
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("mean(delta u_r), wall ROI (r>0.7 R_v)")
    axes[0].set_title("per-step signed increment")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("cumulative sum of mean(delta u_r)")
    axes[1].set_title("cumulative signed increment (coherence check)")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)
    fig.suptitle("Figure 4: wall-ROI signed velocity increment (coherent bias check)")
    fig.tight_layout()
    fig.savefig(fig_dir / "signed_increment_wall_roi.png", dpi=200)
    print(f"Saved {fig_dir / 'signed_increment_wall_roi.png'}")

    # ---------------- Figure 5: spatial snapshots ----------------
    snapshot_steps = [1, 10, 50, min(200, args.steps)]
    fig, axes = plt.subplots(3, len(snapshot_steps), figsize=(4 * len(snapshot_steps), 10), squeeze=False)
    for row, label in enumerate(["A", "B", "C"]):
        df, final = results[label]
        # re-run to capture snapshots (cheap re-run, steps are fast at this grid size)
        pass
    # Simpler: capture final-state field only (already have it) plus rerun to get
    # intermediate snapshots for A/B/C.
    for row, label in enumerate(["A", "B", "C"]):
        kwargs = dict(cases_spec)[label]
        u_r = np.zeros(grid.shape_ur); u_z = np.zeros(grid.shape_uz)
        u_theta = omega * grid.r_c[:, None] * np.ones(grid.shape_center)
        p = p0_continuum.copy(); phi = phi0.copy(); rho, mu = rho0.copy(), mu0.copy()
        col = 0
        for step in range(1, max(snapshot_steps) + 1):
            if kwargs["evolve_material"]:
                rho, mu = material_properties(phi, grid, cfg)
            u_r, u_z, u_theta, p, dpdr, centrifugal, adv_r = momentum_pressure_step(
                grid, cfg, u_r, u_z, u_theta, p, rho, mu, phi, dt, omega)
            apply_velocity_bc(grid, type("F", (), {"u_r": u_r, "u_z": u_z, "u_theta": u_theta, "phi": phi})())
            if kwargs["evolve_phi"]:
                phi = advect_level_set_configurable(phi, u_r, u_z, grid, dt,
                                                     scheme=cfg.levelset.advection_scheme,
                                                     time_integrator=cfg.levelset.time_integrator,
                                                     limiter=cfg.levelset.limiter)
                if kwargs["do_reinit"] and step % cfg.levelset.reinitialize_every == 0:
                    phi = reinitialize_level_set(phi, grid, cfg.levelset.reinitialize_iterations)
            if step in snapshot_steps:
                speed = np.sqrt(0.5 * (u_r[:-1, :] ** 2 + u_r[1:, :] ** 2) + 0.5 * (u_z[:, :-1] ** 2 + u_z[:, 1:] ** 2))
                ax = axes[row][col]
                im = ax.pcolormesh(grid.r_c * 1e3, grid.z_c * 1e3, speed.T, shading="auto", cmap="inferno")
                ax.contour(grid.r_c * 1e3, grid.z_c * 1e3, phi.T, levels=[0.0], colors="cyan", linewidths=1.0)
                ax.axvline(grid.r_v * 1e3, color="w", lw=0.8, ls=":")
                fig.colorbar(im, ax=ax, label="speed [m/s]")
                ax.set_title(f"Case {label}, step {step}")
                ax.set_xlabel("r [mm]"); ax.set_ylabel("z [mm]")
                col += 1
    fig.suptitle("Figure 5: spatial speed field snapshots, interface (cyan) + wall (dotted)")
    fig.tight_layout()
    fig.savefig(fig_dir / "spatial_snapshots.png", dpi=180)
    print(f"Saved {fig_dir / 'spatial_snapshots.png'}")

    # ---------------- summary table ----------------
    summary_rows = []
    for label, (df, final) in results.items():
        summary_rows.append({
            "case": label,
            "final_max_meridional_speed": df["max_meridional_speed"].iloc[-1],
            "final_phi_L2_change": df["phi_L2_change"].iloc[-1],
            "final_interface_rms_displacement": df["interface_rms_displacement"].iloc[-1],
            "final_rho_L2_change": df["rho_L2_change"].iloc[-1],
            "final_volume_drift": df["relative_volume_drift"].iloc[-1],
        })
    summary_rows.append({
        "case": "A_discrete_IC",
        "final_max_meridional_speed": df_A_discrete["max_meridional_speed"].iloc[-1],
        "final_phi_L2_change": df_A_discrete["phi_L2_change"].iloc[-1],
        "final_interface_rms_displacement": df_A_discrete["interface_rms_displacement"].iloc[-1],
        "final_rho_L2_change": df_A_discrete["rho_L2_change"].iloc[-1],
        "final_volume_drift": df_A_discrete["relative_volume_drift"].iloc[-1],
    })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "summary.csv", index=False)
    print("\n" + summary_df.to_string(index=False))
    print(f"\nSaved {out_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
