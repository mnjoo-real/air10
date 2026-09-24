"""DIAGNOSTIC (not a validated/gated script) -- Session 7 investigation into
the exact-rotating-equilibrium spurious-velocity defect.

The wall pressure Neumann BC was cleared by hand-derivation + single-phase
instrumentation (proven algebraically equivalent to the projection-consistent
Neumann substitution for a stationary wall). This script isolates the
REMAINING candidate: the variable-density face-coefficient / pressure-
gradient discretization across the Level-Set density jump, specifically
near the wall-interface intersection.

For a sequence of controlled cases (density ratio swept from 1 to the
physical ~832, with/without the Level Set interface present, with/without a
corrected u_theta wall-face interpolation), runs ONE real solver timestep
from the exact rotating equilibrium IC and records, at every wall-adjacent
and interface-adjacent face:

    centrifugal term, actual dp/dr, required dp/dr, face 1/rho, predictor
    velocity, corrected velocity, radial momentum residual

Then plots max spurious meridional speed vs. density ratio and a spatial map
of the radial momentum residual for the physical-ratio case.

Usage:
    python scripts/diag_wall_interface_density.py
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

from air_vortex.config import load_config  # noqa: E402
from air_vortex.grid import build_grid  # noqa: E402
from air_vortex.fields import initialize_rotating_equilibrium  # noqa: E402
from air_vortex.operators import (  # noqa: E402
    laplacian_ur, upwind_derivative, uz_at_ur_locations, divergence,
    grad_p_to_ur_faces, interp_center_to_ur,
)
from air_vortex.properties import material_properties  # noqa: E402
from air_vortex.pressure import solve_pressure_poisson, pressure_projection, face_inv_rho_ur  # noqa: E402
from air_vortex.boundary import apply_velocity_bc  # noqa: E402


def u_theta_wall_corrected(grid, u_theta):
    """A/B variant of u_theta_on_ur_faces: linear extrapolation to the two
    boundary faces instead of nearest-cell copy (interp_center_to_ur's
    current behaviour). Only used in the dedicated A/B case below -- NOT a
    change to the production solver."""
    out = interp_center_to_ur(u_theta)
    # axis: symmetric extrapolation (u_theta(0)=0 by construction of a
    # solid-body/any-physical swirl field with no singular core), leave as-is
    # (nearest-neighbor at the axis is fine -- u_theta[0,:] is already ~0-ish
    # and the centrifugal term is safety-guarded there).
    # wall: linear extrapolation from the last two cell centers to r=R_v.
    out[-1, :] = 2.0 * u_theta[-1, :] - u_theta[-2, :]
    return out


def run_one_step(grid, cfg, omega, dt, rho_water, rho_air, levelset_on,
                  correct_wall_utheta=False):
    fields = initialize_rotating_equilibrium(grid, cfg, omega)
    if not levelset_on:
        fields.phi = -1.0 * np.ones(grid.shape_center)  # uniform water everywhere

    cfg2 = load_config("configs/validation_solid_body.yaml")
    cfg2.fluid.surface_tension = 0.0
    cfg2.fluid.water_density = rho_water
    cfg2.fluid.air_density = rho_air
    rho, mu = material_properties(fields.phi, grid, cfg2)

    w_at_ur = uz_at_ur_locations(fields.u_z)
    adv_r = (fields.u_r * upwind_derivative(fields.u_r, fields.u_r, grid.dr, axis=0)
             + w_at_ur * upwind_derivative(fields.u_r, w_at_ur, grid.dz, axis=1))

    if correct_wall_utheta:
        u_theta_ur = u_theta_wall_corrected(grid, fields.u_theta)
    else:
        u_theta_ur = interp_center_to_ur(fields.u_theta)

    r_f_safe = np.where(grid.r_f == 0.0, grid.dr, grid.r_f)
    centrifugal = u_theta_ur**2 / r_f_safe[:, None]

    inv_rho_ur = face_inv_rho_ur(rho)
    rho_ur_face = np.where(inv_rho_ur > 0, 1.0 / np.where(inv_rho_ur > 0, inv_rho_ur, 1.0), rho_water)
    visc_r = np.where(rho_ur_face > 0, mu[0, 0] * laplacian_ur(grid, fields.u_r) / rho_ur_face, 0.0)

    u_r_star = fields.u_r + dt * (-adv_r + centrifugal + visc_r)
    u_r_star_raw = u_r_star.copy()

    u_r_star[0, :] = 0.0
    u_r_star[-1, :] = 0.0
    u_z_star = fields.u_z.copy()
    u_z_star[:, 0] = 0.0
    u_z_star[:, -1] = u_z_star[:, -2]

    p_new = solve_pressure_poisson(grid, u_r_star, u_z_star, rho, dt, method="direct", p0=fields.p)
    dpdr = grad_p_to_ur_faces(grid, p_new)

    u_r_new, u_z_new = pressure_projection(grid, u_r_star, u_z_star, p_new, rho, dt)
    fields.u_r, fields.u_z, fields.p = u_r_new, u_z_new, p_new
    apply_velocity_bc(grid, fields)

    # radial momentum residual at u_r faces (interior faces only -- boundary
    # faces are Dirichlet, residual there is not meaningful the same way):
    # R_r = (u_r_new - u_r_old)/dt - centrifugal + (1/rho) dp/dr - viscous
    # u_r_old == 0 everywhere (exact equilibrium IC), so:
    residual = (u_r_new - 0.0) / dt - centrifugal + inv_rho_ur * dpdr - visc_r
    residual[0, :] = np.nan
    residual[-1, :] = np.nan

    phi = fields.phi
    water_cell = phi < 0.0
    water_ur = np.zeros(grid.shape_ur, dtype=bool)
    water_ur[1:-1, :] = water_cell[1:, :] & water_cell[:-1, :]
    water_ur[0, :] = water_cell[0, :]
    water_ur[-1, :] = water_cell[-1, :]
    water_uz = np.zeros(grid.shape_uz, dtype=bool)
    water_uz[:, 1:-1] = water_cell[:, 1:] & water_cell[:, :-1]
    water_uz[:, 0] = water_cell[:, 0]
    water_uz[:, -1] = water_cell[:, -1]

    speed = np.sqrt(
        np.pad(fields.u_r, ((0, 0), (0, 0)))[:-1, :] ** 2  # placeholder, replaced below
    ) if False else None

    result = {
        "max_ur_all": float(np.max(np.abs(fields.u_r))),
        "max_ur_water": float(np.max(np.abs(fields.u_r[water_ur]))) if water_ur.any() else float("nan"),
        "max_ur_air": float(np.max(np.abs(fields.u_r[~water_ur]))) if (~water_ur).any() else float("nan"),
        "max_uz_all": float(np.max(np.abs(fields.u_z))),
        "max_uz_water": float(np.max(np.abs(fields.u_z[water_uz]))) if water_uz.any() else float("nan"),
        "max_uz_air": float(np.max(np.abs(fields.u_z[~water_uz]))) if (~water_uz).any() else float("nan"),
        "max_divergence": float(np.max(np.abs(divergence(grid, u_r_new, u_z_new)))),
        "max_abs_residual": float(np.nanmax(np.abs(residual))),
    }
    return result, residual, fields, rho, dpdr, centrifugal, u_r_star_raw, u_r_new


def main():
    cfg = load_config("configs/validation_solid_body.yaml")
    cfg.fluid.surface_tension = 0.0
    grid = build_grid(cfg)
    omega = 2 * np.pi * 100 / 60
    dt = 1e-4
    rho_water = 998.0

    cases = []

    # 1. Level Set OFF, uniform density (control)
    r, *_ = run_one_step(grid, cfg, omega, dt, rho_water, rho_water, levelset_on=False)
    cases.append({"label": "LS_OFF_uniform", "density_ratio": 1.0, "levelset": False,
                  "wall_utheta_fix": False, **r})

    # 2. Level Set ON, equal densities both sides (interface present, no density jump)
    r, *_ = run_one_step(grid, cfg, omega, dt, rho_water, rho_water, levelset_on=True)
    cases.append({"label": "LS_ON_ratio1", "density_ratio": 1.0, "levelset": True,
                  "wall_utheta_fix": False, **r})

    # 3. Level Set ON, density ratio sweep
    ratios = [2.0, 10.0, 100.0, rho_water / 1.2]
    last_physical_residual = None
    last_physical_fields = None
    for ratio in ratios:
        rho_air = rho_water / ratio
        r, residual, fields, rho_field, dpdr, centrifugal, ur_star_raw, ur_new = run_one_step(
            grid, cfg, omega, dt, rho_water, rho_air, levelset_on=True)
        label = f"LS_ON_ratio{ratio:.0f}" if ratio < 500 else "LS_ON_ratio_physical"
        cases.append({"label": label, "density_ratio": ratio, "levelset": True,
                      "wall_utheta_fix": False, **r})
        if ratio > 500:
            last_physical_residual = residual
            last_physical_fields = fields

    # 4. physical ratio + corrected u_theta wall-face interpolation (A/B)
    r, residual_fix, fields_fix, *_ = run_one_step(
        grid, cfg, omega, dt, rho_water, 1.2, levelset_on=True, correct_wall_utheta=True)
    cases.append({"label": "LS_ON_ratio_physical_utheta_fixed", "density_ratio": rho_water / 1.2,
                  "levelset": True, "wall_utheta_fix": True, **r})

    df = pd.DataFrame(cases)
    out_dir = Path("results/diagnostics")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "wall_interface_density_sweep.csv"
    df.to_csv(csv_path, index=False)
    print(df.to_string(index=False))
    print(f"\nSaved {csv_path}")

    # --- Plot 1: max spurious meridional speed vs density ratio ---
    sweep = df[df["levelset"] & ~df["wall_utheta_fix"]].sort_values("density_ratio")
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(sweep["density_ratio"], sweep["max_ur_all"] * 1e3, "o-", label="max|u_r| (mm/s)")
    ax.plot(sweep["density_ratio"], sweep["max_uz_all"] * 1e3, "s-", label="max|u_z| (mm/s)")
    ax.set_xscale("log")
    ax.set_xlabel("density ratio (rho_water / rho_air)")
    ax.set_ylabel("max spurious velocity after 1 step [mm/s]")
    ax.set_title("Spurious meridional velocity vs. density ratio\n(one step from exact equilibrium)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "spurious_velocity_vs_density_ratio.png", dpi=200)
    print(f"Saved {out_dir / 'spurious_velocity_vs_density_ratio.png'}")

    # --- Plot 2: spatial map of radial momentum residual, physical ratio ---
    if last_physical_residual is not None:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        for ax, res, title in [(axes[0], last_physical_residual, "u_theta: nearest-neighbor (current)"),
                                (axes[1], residual_fix, "u_theta: linear-extrapolated (A/B fix)")]:
            im = ax.pcolormesh(grid.r_f * 1e3, grid.z_c * 1e3, np.abs(res).T,
                                shading="auto", cmap="inferno")
            fig.colorbar(im, ax=ax, label="|R_r| [m/s^2]")
            ax.set_xlabel("r [mm]")
            ax.set_ylabel("z [mm]")
            ax.set_title(title)
            # overlay interface
            phi = last_physical_fields.phi if title.startswith("u_theta: nearest") else fields_fix.phi
            ax.contour(grid.r_c * 1e3, grid.z_c * 1e3, phi.T, levels=[0.0], colors="cyan", linewidths=1.2)
        fig.suptitle("Radial momentum residual R_r, physical density ratio (~832), one step")
        fig.tight_layout()
        fig.savefig(out_dir / "residual_map_physical_ratio.png", dpi=200)
        print(f"Saved {out_dir / 'residual_map_physical_ratio.png'}")


if __name__ == "__main__":
    main()
