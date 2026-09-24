"""Main report plots A-E (README section 25) plus the result-visualization
plots (README "Result Visualization"): validation figures, single-run
snapshots, and sweep-scaling figures. The latter group reuses
:mod:`air_vortex.diagnostics` for every physical quantity (vortex tip,
depth, free-surface height, air-core masks) so a figure's numbers never
diverge from the solver's own definitions."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .diagnostics import find_tip_z, free_surface_height, vortex_depth
from .grid import Grid
from .viz_common import (
    air_core_masks,
    cell_center_velocity,
    field,
    has_field,
    meridional_speed,
    meshgrid_mm,
    robust_clim,
    scalar,
    stirrer_region_contour,
    to_mm,
)


def plot_a_depth_vs_n_squared(N: np.ndarray, d_infinity: np.ndarray, ax=None):
    """Plot A: d_infinity vs N^2 (README 25)."""
    ax = ax or plt.gca()
    ax.plot(N**2, d_infinity, "o-")
    ax.set_xlabel(r"$N^2$ [rpm$^2$]")
    ax.set_ylabel(r"$d_\infty$ [m]")
    ax.set_title("Plot A: vortex depth vs. $N^2$")
    return ax


def plot_b_critical_n_squared_vs_h_eff(H_eff: np.ndarray, N_c: np.ndarray, ax=None):
    """Plot B: N_c^2 vs H_eff (README 25)."""
    ax = ax or plt.gca()
    ax.plot(H_eff, N_c**2, "o-")
    ax.set_xlabel(r"$H_{\mathrm{eff}}$ [m]")
    ax.set_ylabel(r"$N_c^2$ [rpm$^2$]")
    ax.set_title("Plot B: critical $N_c^2$ vs. $H_{eff}$")
    return ax


def plot_c_froude_vs_depth_ratio(H_over_Rm: np.ndarray, Fr_c: np.ndarray, ax=None):
    """Plot C: Fr_Omega,c vs H/R_m (README 25)."""
    ax = ax or plt.gca()
    ax.plot(H_over_Rm, Fr_c, "o-")
    ax.set_xlabel(r"$H/R_m$")
    ax.set_ylabel(r"$Fr_{\Omega,c}$")
    ax.set_title("Plot C: dimensionless critical-condition diagram")
    return ax


def plot_d_air_core_radius(grid: Grid, r_air_by_rpm: dict[float, np.ndarray], ax=None):
    """Plot D: r_air(z) for several RPM above N_c (README 25)."""
    ax = ax or plt.gca()
    for rpm, r_air in r_air_by_rpm.items():
        ax.plot(r_air, grid.z_c, label=f"{rpm:.0f} rpm")
    ax.set_xlabel(r"$r_{\mathrm{air}}(z)$ [m]")
    ax.set_ylabel("z [m]")
    ax.set_title("Plot D: air-core radius profile")
    ax.legend()
    return ax


def plot_e_velocity_field(grid: Grid, u_theta: np.ndarray, u_r: np.ndarray,
                           u_z: np.ndarray, fig=None):
    """Plot E: u_theta(r,z), (r,z)-plane streamlines, centerline u_z(z)."""
    from .operators import interp_ur_to_center, interp_uz_to_center

    fig = fig or plt.figure(figsize=(12, 4))
    axes = fig.subplots(1, 3)

    R, Z = np.meshgrid(grid.r_c, grid.z_c, indexing="ij")

    im0 = axes[0].pcolormesh(R, Z, u_theta, shading="auto")
    axes[0].set_title(r"$u_\theta(r,z)$")
    axes[0].set_xlabel("r [m]")
    axes[0].set_ylabel("z [m]")
    fig.colorbar(im0, ax=axes[0])

    u_r_c = interp_ur_to_center(u_r)
    u_z_c = interp_uz_to_center(u_z)
    axes[1].streamplot(grid.r_c, grid.z_c, u_r_c.T, u_z_c.T)
    axes[1].set_title("streamlines (r-z plane)")
    axes[1].set_xlabel("r [m]")
    axes[1].set_ylabel("z [m]")

    axes[2].plot(u_z_c[0, :], grid.z_c)
    axes[2].set_title(r"centerline $u_z(z)$")
    axes[2].set_xlabel(r"$u_z$ [m/s]")
    axes[2].set_ylabel("z [m]")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Result-visualization plots (README "Result Visualization" / Figures 1-10).
# ---------------------------------------------------------------------------


def plot_hydrostatic_validation(grid: Grid, cfg, snap, fig=None, r_index=None, V0=None):
    """Figure 1: still-water pressure vs. the analytic hydrostatic profile
    p(z) = rho_w * g * (H - z) (README section 43 Milestone 1 / Test 1)."""
    fig = fig or plt.figure(figsize=(9, 4.5))
    axes = fig.subplots(1, 2)

    p = field(snap, "p")
    Nr = grid.Nr
    i = r_index if r_index is not None else Nr // 2

    H = cfg.geometry.water_height_m
    rho_w = cfg.fluid.water_density
    g = cfg.fluid.gravity

    z_mm = to_mm(grid.z_c)
    p_num = p[i, :]
    water = grid.z_c < H
    p_exp = np.where(water, rho_w * g * (H - grid.z_c), 0.0)

    ax = axes[0]
    ax.plot(p_exp[water], z_mm[water], "-", color="tab:gray", lw=2, label="analytic")
    ax.plot(p_num, z_mm, "o", ms=3, color="tab:blue", label="numerical")
    ax.axhline(H * 1e3, color="k", lw=0.8, ls=":", label="H (interface)")
    ax.set_xlabel("pressure [Pa]")
    ax.set_ylabel("z [mm]")
    ax.set_title(f"pressure profile at r={grid.r_c[i]*1e3:.1f} mm")
    ax.legend(fontsize=8)

    ax2 = axes[1]
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_err = np.where(p_exp > 0, (p_num - p_exp) / p_exp * 100.0, np.nan)
    ax2.plot(rel_err[water], z_mm[water], "o-", ms=3, color="tab:red")
    ax2.axvline(0.0, color="k", lw=0.8)
    ax2.set_xlabel("relative error [%]")
    ax2.set_ylabel("z [mm]")
    ax2.set_title("numerical vs. analytic error")

    max_speed = max(np.max(np.abs(field(snap, "u_r"))), np.max(np.abs(field(snap, "u_z"))))
    caption = f"t = {scalar(snap, 't'):.4g} s,  max|u| = {max_speed:.2e} m/s"
    if V0 is not None:
        from .solver import water_volume
        V1 = water_volume(grid, field(snap, "phi"))
        caption += f",  volume drift = {abs(V1 - V0) / V0 * 100:.3f}%"
    fig.suptitle("Figure 1: hydrostatic equilibrium validation\n" + caption, fontsize=10)
    fig.tight_layout()
    return fig


def plot_solid_body_validation(grid: Grid, cfg, snap, omega: float, fig=None, z_index=None):
    """Figure 2: prescribed solid-body swirl vs. the analytic centrifugal
    pressure balance (README section 8) and parabolic free surface
    (README section 31 Test 2). Note: unlike dp/dr (which the pressure
    solve satisfies quickly), the free-surface panel only matches the
    analytic parabola once the interface has had time to relax via
    advection -- a short run will show dp/dr converged well before eta(r)
    has caught up; that gap is real physics (slow gravity-wave / advective
    relaxation), not a solver bug."""
    from .operators import center_grad_r

    fig = fig or plt.figure(figsize=(10, 4.5))
    axes = fig.subplots(1, 2)

    p = field(snap, "p")
    j = z_index if z_index is not None else grid.Nz // 2
    rho = field(snap, "rho") if has_field(snap, "rho") else np.full(grid.shape_center, cfg.fluid.water_density)

    dpdr = center_grad_r(grid, p)[:, j]
    exp_dpdr = rho[:, j] * omega**2 * grid.r_c

    ax = axes[0]
    ax.plot(to_mm(grid.r_c), exp_dpdr, "-", color="tab:gray", lw=2, label="analytic")
    ax.plot(to_mm(grid.r_c), dpdr, "o", ms=3, color="tab:blue", label="numerical")
    ax.set_xlabel("r [mm]")
    ax.set_ylabel(r"$dp/dr$ [Pa/m]")
    ax.set_title(rf"radial balance at z={grid.z_c[j]*1e3:.1f} mm, $\Omega$={omega:.2f} rad/s")
    ax.legend(fontsize=8)

    eta = free_surface_height(field(snap, "phi"), grid)
    valid = np.isfinite(eta)
    g = cfg.fluid.gravity
    eta0 = eta[valid][0] if np.any(valid) else cfg.geometry.water_height_m
    eta_exp = eta0 + omega**2 * grid.r_c**2 / (2.0 * g)

    ax2 = axes[1]
    ax2.plot(to_mm(grid.r_c), to_mm(eta_exp), "-", color="tab:gray", lw=2, label="analytic parabola")
    ax2.plot(to_mm(grid.r_c[valid]), to_mm(eta[valid]), "o", ms=3, color="tab:blue", label="numerical")
    ax2.set_xlabel("r [mm]")
    ax2.set_ylabel(r"$\eta(r)$ [mm]")
    ax2.set_title("free-surface shape")
    ax2.legend(fontsize=8)

    if np.any(valid):
        err = np.abs(eta[valid] - eta_exp[valid])
        caption = f"t = {scalar(snap, 't'):.4g} s,  mean|eta err| = {np.mean(err)*1e3:.3f} mm"
    else:
        caption = f"t = {scalar(snap, 't'):.4g} s,  no interface crossing found in domain"
    fig.suptitle("Figure 2: solid-body-rotation validation\n" + caption, fontsize=10)
    fig.tight_layout()
    return fig


def plot_freesurface_timeaverage(grid: Grid, eta_mean: np.ndarray, eta_std: np.ndarray,
                                  omega: float, g: float, eta0_ref: float, fig=None):
    """Long-run solid-body-rotation validation: time-averaged numerical
    eta(r) (over a window where the interface has had time to relax past
    its initial inertial-oscillation transient, README section 31 Test 2)
    vs. the analytic parabola eta(r) = eta(0) + Omega^2 r^2 / (2g), plus
    the pointwise error. ``eta0_ref`` anchors the analytic curve to the
    *numerical* eta(0) (volume conservation, not the flat-water H, sets the
    true absolute level -- see README "Performance"/validation notes on
    level-set volume drift) so the comparison isolates the *shape*, not an
    unrelated volume-drift offset."""
    fig = fig or plt.figure(figsize=(10, 4.5))
    axes = fig.subplots(1, 2)

    eta_exp = eta0_ref + omega**2 * grid.r_c**2 / (2.0 * g)
    valid = np.isfinite(eta_mean)

    ax = axes[0]
    ax.plot(to_mm(grid.r_c), to_mm(eta_exp), "-", color="tab:gray", lw=2, label="analytic parabola")
    ax.errorbar(to_mm(grid.r_c[valid]), to_mm(eta_mean[valid]), yerr=to_mm(eta_std[valid]),
                fmt="o", ms=3, capsize=2, color="tab:blue", label="numerical (time-avg ± std)")
    ax.set_xlabel("r [mm]")
    ax.set_ylabel(r"time-averaged $\eta(r)$ [mm]")
    ax.set_title(rf"free-surface shape, $\Omega$={omega:.2f} rad/s")
    ax.legend(fontsize=8)

    err = eta_mean - eta_exp
    ax2 = axes[1]
    ax2.plot(to_mm(grid.r_c[valid]), to_mm(err[valid]), "o-", ms=3, color="tab:red")
    ax2.axhline(0.0, color="k", lw=0.8)
    ax2.set_xlabel("r [mm]")
    ax2.set_ylabel(r"$\eta_{num} - \eta_{analytic}$ [mm]")
    ax2.set_title("pointwise error")

    fig.tight_layout()
    return fig


def plot_center_depression_convergence(t: np.ndarray, eta0: np.ndarray,
                                        analytic_eta0: float | None = None,
                                        window: tuple[float, float] | None = None, fig=None):
    """Center free-surface height eta(0) vs. time, for a long solid-body-
    rotation validation run: shows the initial transient, whatever
    inertial/gravity-wave oscillation persists around the mean, and
    (optionally) the analytic target and the window used to compute the
    time-average shown in :func:`plot_freesurface_timeaverage`."""
    fig = fig or plt.figure(figsize=(6.5, 4.5))
    ax = fig.add_subplot(111)

    ax.plot(t, to_mm(eta0), "-", color="tab:blue", lw=1)
    if analytic_eta0 is not None:
        ax.axhline(to_mm(analytic_eta0), color="tab:gray", lw=2, ls="--", label="analytic eta(0)")
    if window is not None:
        ax.axvspan(window[0], window[1], color="tab:green", alpha=0.15, label="averaging window")

    ax.set_xlabel("t [s]")
    ax.set_ylabel(r"$\eta(0)$ [mm]")
    ax.set_title("center free-surface height vs. time")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_interface_snapshots(grid: Grid, cfg, snapshots, rpm: float | None = None, fig=None, ncols=3):
    """Figure 3: phi=0 interface at several times, one panel per snapshot.
    ``snapshots`` is a sequence of (t, snap) pairs."""
    n = len(snapshots)
    ncols = min(ncols, n) or 1
    nrows = int(np.ceil(n / ncols))
    fig = fig or plt.figure(figsize=(4 * ncols, 3.2 * nrows))
    axes = fig.subplots(nrows, ncols, squeeze=False).ravel()

    chi = stirrer_region_contour(grid, cfg)
    R, Z = meshgrid_mm(grid)

    for idx, (t, snap) in enumerate(snapshots):
        ax = axes[idx]
        phi = field(snap, "phi")
        water = phi < 0
        ax.pcolormesh(R, Z, water.astype(float), cmap="Blues", vmin=-0.5, vmax=1.5, shading="auto")
        ax.contour(R, Z, phi, levels=[0.0], colors="k", linewidths=1.2)
        ax.contour(R, Z, chi, levels=[0.5], colors="tab:orange", linewidths=1.0, linestyles="--")

        z_tip = find_tip_z(phi, grid, cfg)
        d = vortex_depth(phi, grid, cfg)
        ax.axhline(z_tip * 1e3, color="tab:red", lw=0.8, ls=":")

        label = f"t={t:.3g}s"
        if rpm is not None:
            label += f", {rpm:.0f} rpm"
        label += f"\nd={d*1e3:.2f} mm"
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("r [mm]")
        ax.set_ylabel("z [mm]")

    for idx in range(n, len(axes)):
        axes[idx].axis("off")

    fig.suptitle("Figure 3: free-surface evolution", fontsize=11)
    fig.tight_layout()
    return fig


def plot_pressure_interface(grid: Grid, cfg, snap, fig=None):
    """Figure 4: gauge pressure field with the interface, stirrer region,
    and vortex tip overlaid; robust (percentile-based) color scaling."""
    fig = fig or plt.figure(figsize=(6.5, 5))
    ax = fig.add_subplot(111)

    p = field(snap, "p")
    phi = field(snap, "phi")
    R, Z = meshgrid_mm(grid)

    vmin, vmax = robust_clim([p], low=1, high=99)
    im = ax.pcolormesh(R, Z, p, cmap="RdBu_r", vmin=vmin, vmax=vmax, shading="auto")
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("gauge pressure [Pa]")

    ax.contour(R, Z, phi, levels=[0.0], colors="k", linewidths=1.4)
    chi = stirrer_region_contour(grid, cfg)
    ax.contour(R, Z, chi, levels=[0.5], colors="lime", linewidths=1.2, linestyles="--")

    z_tip = find_tip_z(phi, grid, cfg)
    ax.plot([0.0], [z_tip * 1e3], marker="v", color="yellow", ms=10,
            markeredgecolor="k", label="vortex tip")

    ax.set_xlabel("r [mm]")
    ax.set_ylabel("z [mm]")
    ax.set_title(f"Figure 4: pressure field, t={scalar(snap, 't'):.3g} s")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


def plot_meridional_flow(grid: Grid, cfg, snap, fig=None, use_streamlines=True):
    """Figure 5: meridional (r-z plane) flow -- background |u_meridional|
    or u_z, overlaid with streamlines/quiver, to show the central downward
    jet, outer return flow, and toroidal recirculation."""
    fig = fig or plt.figure(figsize=(6.5, 5))
    ax = fig.add_subplot(111)

    u_r_c, u_z_c = cell_center_velocity(grid, field(snap, "u_r"), field(snap, "u_z"))
    speed = meridional_speed(u_r_c, u_z_c)
    R, Z = meshgrid_mm(grid)

    vmin, vmax = robust_clim([speed], low=0, high=99)
    im = ax.pcolormesh(R, Z, speed, cmap="viridis", vmin=vmin, vmax=vmax, shading="auto")
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("meridional speed [m/s]")

    if use_streamlines:
        try:
            ax.streamplot(to_mm(grid.r_c), to_mm(grid.z_c), u_r_c.T, u_z_c.T,
                           color="white", linewidth=0.6, density=1.0, arrowsize=0.8)
        except ValueError:
            pass  # degenerate (e.g. all-zero) velocity field: skip streamlines
    else:
        step = max(1, grid.Nr // 16)
        ax.quiver(R[::step, ::step], Z[::step, ::step],
                   u_r_c[::step, ::step], u_z_c[::step, ::step], color="white")

    ax.contour(R, Z, field(snap, "phi"), levels=[0.0], colors="k", linewidths=1.2)

    ax.set_xlabel("r [mm]")
    ax.set_ylabel("z [mm]")
    ax.set_title(f"Figure 5: meridional flow, t={scalar(snap, 't'):.3g} s")
    fig.tight_layout()
    return fig


def plot_azimuthal_velocity(grid: Grid, cfg, snap, z_slices=None, fig=None):
    """Figure 6: u_theta(r,z) contour plus u_theta(r) at a few z levels.
    r^1 (solid-body core) / r^-1 (free-vortex outer region) reference
    curves are drawn only as visual guides -- fit against the data, never
    asserted to hold (README "physical caution" requirement)."""
    fig = fig or plt.figure(figsize=(11, 4.5))
    axes = fig.subplots(1, 2)

    u_theta = field(snap, "u_theta")
    R, Z = meshgrid_mm(grid)

    im = axes[0].pcolormesh(R, Z, u_theta, cmap="plasma", shading="auto")
    fig.colorbar(im, ax=axes[0], label=r"$u_\theta$ [m/s]")
    axes[0].contour(R, Z, field(snap, "phi"), levels=[0.0], colors="w", linewidths=1.0)
    axes[0].set_xlabel("r [mm]")
    axes[0].set_ylabel("z [mm]")
    axes[0].set_title(r"$u_\theta(r,z)$")

    if z_slices is None:
        z_slices = [grid.z_c[grid.Nz // 4], grid.z_c[grid.Nz // 2]]

    ax2 = axes[1]
    for z_target in z_slices:
        j = int(np.argmin(np.abs(grid.z_c - z_target)))
        ax2.plot(to_mm(grid.r_c), u_theta[:, j], "o-", ms=3, label=f"z={grid.z_c[j]*1e3:.1f} mm")

    r_peak = grid.r_c[np.argmax(np.abs(u_theta[:, grid.Nz // 4]))]
    u_peak = u_theta[np.argmax(np.abs(u_theta[:, grid.Nz // 4])), grid.Nz // 4]
    if abs(u_peak) > 0:
        r_ref = grid.r_c
        core_ref = u_peak * r_ref / r_peak
        outer_ref = u_peak * r_peak / np.maximum(r_ref, r_peak * 1e-3)
        ax2.plot(to_mm(r_ref), core_ref, "--", color="gray", lw=1, label=r"reference $u_\theta \propto r$")
        ax2.plot(to_mm(r_ref), outer_ref, ":", color="gray", lw=1, label=r"reference $u_\theta \propto 1/r$")

    ax2.set_xlabel("r [mm]")
    ax2.set_ylabel(r"$u_\theta$ [m/s]")
    ax2.set_title(r"$u_\theta(r)$ profiles (reference curves only, not a fit)")
    ax2.legend(fontsize=7)

    fig.suptitle(f"Figure 6: azimuthal velocity, t={scalar(snap, 't'):.3g} s", fontsize=11)
    fig.tight_layout()
    return fig


def plot_vortex_depth_time(runs: dict, fig=None):
    """Figure 7: d(t) for several RPM runs on one axis. ``runs`` maps a
    label (e.g. "1200 rpm") to a diagnostics DataFrame with columns "t"
    and "d" (the *same* column the solver itself writes, README section
    24) -- this function does not recompute vortex depth."""
    fig = fig or plt.figure(figsize=(6.5, 4.5))
    ax = fig.add_subplot(111)

    for label, df in runs.items():
        ax.plot(df["t"], df["d"] * 1e3, label=label)

    ax.set_xlabel("t [s]")
    ax.set_ylabel("vortex depth d [mm]")
    ax.set_title("Figure 7: vortex depth vs. time")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def _linregress_summary(x: np.ndarray, y: np.ndarray):
    from scipy import stats
    res = stats.linregress(x, y)
    return res.slope, res.intercept, res.rvalue**2, res.stderr


def plot_depth_scaling(N_rpm: np.ndarray, d_infinity: np.ndarray, fig=None,
                        dimensionless: bool = False, R_m: float | None = None,
                        g: float = 9.81, yerr=None):
    """Figure 8: d_infinity vs N^2 (dimensional) or d/R_m vs Fr_Omega
    (dimensionless, requires R_m). Linear regression is fit and reported,
    never asserted to be exact (README "physical caution": d ~ N^2 is a
    hypothesis to test, not an assumed law)."""
    fig = fig or plt.figure(figsize=(6, 4.5))
    ax = fig.add_subplot(111)

    N_rpm = np.asarray(N_rpm, dtype=float)
    d_infinity = np.asarray(d_infinity, dtype=float)

    if dimensionless:
        if R_m is None:
            raise ValueError("R_m is required for the dimensionless depth-scaling plot")
        omega = 2 * np.pi * N_rpm / 60.0
        x = omega**2 * R_m / g  # Fr_Omega
        y = d_infinity / R_m
        xlabel, ylabel = r"$Fr_\Omega = \Omega^2 R_m / g$", r"$d_\infty / R_m$"
    else:
        x = N_rpm**2
        y = d_infinity
        xlabel, ylabel = r"$N^2$ [rpm$^2$]", r"$d_\infty$ [m]"

    if yerr is not None:
        ax.errorbar(x, y, yerr=yerr, fmt="o", ms=4, capsize=3, label="data")
    else:
        ax.plot(x, y, "o", ms=5, label="data")

    if len(x) >= 2:
        slope, intercept, r2, stderr = _linregress_summary(x, y)
        x_fit = np.linspace(x.min(), x.max(), 50)
        ax.plot(x_fit, slope * x_fit + intercept, "--", color="tab:red",
                label=f"fit: slope={slope:.3g}±{stderr:.1g}, R²={r2:.3f}")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    title = "Figure 8: depth scaling" + (" (dimensionless)" if dimensionless else "")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_critical_scaling(H: np.ndarray, N_c: np.ndarray, fig=None,
                           dimensionless: bool = False, R_m: float | None = None,
                           g: float = 9.81, H_eff: np.ndarray | None = None,
                           Nc_err=None):
    """Figure 9: N_c^2 vs H_eff (dimensional) or Fr_Omega,c vs H/R_m
    (dimensionless). N_c^2 ~ H_eff is treated as a hypothesis, per the
    Level 0 reduced model (README section 4), not asserted as exact."""
    fig = fig or plt.figure(figsize=(6, 4.5))
    ax = fig.add_subplot(111)

    H = np.asarray(H, dtype=float)
    N_c = np.asarray(N_c, dtype=float)
    H_x = np.asarray(H_eff, dtype=float) if H_eff is not None else H

    if dimensionless:
        if R_m is None:
            raise ValueError("R_m is required for the dimensionless critical-scaling plot")
        omega_c = 2 * np.pi * N_c / 60.0
        x = H_x / R_m
        y = omega_c**2 * R_m / g
        xlabel, ylabel = r"$H/R_m$", r"$Fr_{\Omega,c}$"
    else:
        x = H_x
        y = N_c**2
        xlabel, ylabel = r"$H_{\mathrm{eff}}$ [m]", r"$N_c^2$ [rpm$^2$]"

    if Nc_err is not None and not dimensionless:
        yerr = 2 * N_c * np.asarray(Nc_err)
        ax.errorbar(x, y, yerr=yerr, fmt="o", ms=4, capsize=3, label="data")
    else:
        ax.plot(x, y, "o", ms=5, label="data")

    if len(x) >= 2:
        slope, intercept, r2, stderr = _linregress_summary(x, y)
        x_fit = np.linspace(x.min(), x.max(), 50)
        ax.plot(x_fit, slope * x_fit + intercept, "--", color="tab:red",
                label=f"fit: slope={slope:.3g}±{stderr:.1g}, R²={r2:.3f}")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    title = "Figure 9: critical-RPM scaling" + (" (dimensionless)" if dimensionless else "")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_regime_map(df, fig=None):
    """Figure 10: regime/phase diagram in (Fr_Omega, H/R_m). ``df`` needs
    columns "Fr_Omega", "H_over_Rm", "regime" with regime in
    {"no_core", "transient_contact", "stable_core"} (from the same
    connectivity + persistence criteria as the solver, README section 22).
    A boundary is drawn only as a simple guide line through the transition
    points actually present in the data -- no extrapolated fit."""
    fig = fig or plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111)

    style = {
        "no_core": dict(marker="o", color="tab:blue", label="no core"),
        "transient_contact": dict(marker="^", color="tab:orange", label="transient contact"),
        "stable_core": dict(marker="s", color="tab:red", label="stable air core"),
    }

    for regime, kw in style.items():
        sub = df[df["regime"] == regime]
        if len(sub):
            ax.scatter(sub["Fr_Omega"], sub["H_over_Rm"], **kw, s=40, edgecolor="k", linewidth=0.5)

    ax.set_xlabel(r"$Fr_\Omega = \Omega^2 R_m / g$")
    ax.set_ylabel(r"$H/R_m$")
    ax.set_title("Figure 10: regime map")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig
