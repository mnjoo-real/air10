"""Video renderers for saved simulation runs (README "Result Visualization").

Reads already-saved frames via :mod:`air_vortex.run_io`; never re-runs the
solver. Uses matplotlib.animation with the ffmpeg writer when available,
otherwise falls back to a PNG frame sequence with a clear warning (spec
section "Video generator").

Every video redraws its axes from scratch each frame (``ax.clear()`` then
replot) rather than mutating artists in place. This is somewhat slower than
blitting, but far more robust against the different artist types involved
(pcolormesh, contour, scatter, streamplot all redrawn together each frame),
and rendering speed is explicitly not the priority for this feature.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
from matplotlib import animation as mpl_animation
import numpy as np

from .connectivity import ConnectivityState, is_geometrically_connected, update_persistence
from .diagnostics import find_tip_z, vortex_depth
from .run_io import RunData
from .viz_common import (
    air_core_masks,
    cell_center_velocity,
    field,
    meridional_speed,
    meshgrid_mm,
    robust_clim,
    scalar,
    stirrer_region_contour,
    to_mm,
)


def ffmpeg_available() -> bool:
    try:
        return mpl_animation.writers.is_available("ffmpeg")
    except Exception:
        return False


def _subsample_indices(n_frames: int, max_samples: int = 20) -> list[int]:
    if n_frames <= max_samples:
        return list(range(n_frames))
    step = max(1, n_frames // max_samples)
    return list(range(0, n_frames, step))


def _render(fig, update_fn, n_frames: int, out_path: str | Path, fps: int) -> dict:
    """Shared drive loop: real MP4 via ffmpeg if available, else a PNG
    frame sequence (with a printed warning) so the pipeline never silently
    produces nothing."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if ffmpeg_available():
        anim = mpl_animation.FuncAnimation(fig, update_fn, frames=n_frames, blit=False)
        anim.save(str(out_path), writer=mpl_animation.FFMpegWriter(fps=fps, bitrate=1800))
        plt.close(fig)
        return {"video": out_path}

    frames_dir = out_path.parent / f"{out_path.stem}_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    print(f"[air_vortex.animation] WARNING: ffmpeg not found; cannot write "
          f"'{out_path.name}' as MP4. Saving a PNG frame sequence to "
          f"'{frames_dir}' instead. Once ffmpeg is installed you can "
          f"combine them with e.g.\n"
          f"  ffmpeg -framerate {fps} -i {frames_dir.name}/frame_%04d.png "
          f"-pix_fmt yuv420p {out_path.name}")
    for i in range(n_frames):
        update_fn(i)
        fig.savefig(frames_dir / f"frame_{i:04d}.png", dpi=150)
    plt.close(fig)
    return {"frames_dir": frames_dir}


def _annotate(ax, t, rpm, d, connected=None, stable=None):
    lines = [f"t = {t:.3g} s", f"RPM = {rpm:.0f}", f"vortex depth = {d*1e3:.2f} mm"]
    if connected is not None:
        state = "stable air core" if stable else ("contact" if connected else "no contact")
        lines.append(f"air core: {state}")
    ax.text(0.02, 0.98, "\n".join(lines), transform=ax.transAxes,
            va="top", ha="left", fontsize=8,
            bbox=dict(boxstyle="round", fc="white", alpha=0.8))


def render_interface_video(run: RunData, out_path: str | Path, fps: int = 10,
                            max_frames: Optional[int] = None) -> dict:
    """Video A: water/air regions, phi=0 interface, stirrer forcing region,
    vortex tip, and (once persistence is satisfied) a "stable air core" tag."""
    run.require("cfg", "grid", "frames")
    grid, cfg = run.grid, run.cfg
    rpm = cfg.stirrer.rpm

    frames = run.frames[:max_frames] if max_frames else run.frames
    R, Z = meshgrid_mm(grid)
    chi = stirrer_region_contour(grid, cfg)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    state = ConnectivityState(connected=False)

    def update(i):
        nonlocal state
        ax.clear()
        data = frames[i].load()
        phi = data["phi"]
        t = float(data["t"])

        water = phi < 0
        ax.pcolormesh(R, Z, water.astype(float), cmap="Blues", vmin=-0.5, vmax=1.5, shading="auto")
        ax.contour(R, Z, phi, levels=[0.0], colors="k", linewidths=1.2)
        ax.contour(R, Z, chi, levels=[0.5], colors="tab:orange", linewidths=1.0, linestyles="--")

        connected_now = is_geometrically_connected(phi, grid, cfg)
        state = update_persistence(state, connected_now, t, cfg, rpm)
        d = vortex_depth(phi, grid, cfg)
        z_tip = find_tip_z(phi, grid, cfg)
        ax.plot([0.0], [z_tip * 1e3], marker="v", color="yellow", ms=10, markeredgecolor="k")

        ax.set_xlabel("r [mm]")
        ax.set_ylabel("z [mm]")
        ax.set_xlim(0, to_mm(grid.r_v))
        ax.set_ylim(0, to_mm(grid.z_max))
        _annotate(ax, t, rpm, d, connected=connected_now, stable=state.connected)
        return []

    return _render(fig, update, len(frames), out_path, fps)


def render_pressure_velocity_video(run: RunData, out_path: str | Path, fps: int = 10,
                                    max_frames: Optional[int] = None,
                                    color_subsample: int = 20) -> dict:
    """Video B: pressure (left) + meridional flow (right), with fixed
    (whole-run, robust) color limits computed once up front so the color
    scale doesn't jump around frame to frame."""
    run.require("cfg", "grid", "frames")
    grid, cfg = run.grid, run.cfg
    rpm = cfg.stirrer.rpm

    frames = run.frames[:max_frames] if max_frames else run.frames
    R, Z = meshgrid_mm(grid)

    sample_idx = _subsample_indices(len(frames), color_subsample)
    p_samples, speed_samples = [], []
    for i in sample_idx:
        data = frames[i].load()
        u_r_c, u_z_c = cell_center_velocity(grid, data["u_r"], data["u_z"])
        p_samples.append(data["p"])
        speed_samples.append(meridional_speed(u_r_c, u_z_c))
    p_lim = robust_clim(p_samples, low=1, high=99)
    speed_lim = robust_clim(speed_samples, low=0, high=99)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    def update(i):
        for ax in axes:
            ax.clear()
        data = frames[i].load()
        t = float(data["t"])
        phi = data["phi"]

        im0 = axes[0].pcolormesh(R, Z, data["p"], cmap="RdBu_r",
                                  vmin=p_lim[0], vmax=p_lim[1], shading="auto")
        axes[0].contour(R, Z, phi, levels=[0.0], colors="k", linewidths=1.2)
        axes[0].set_title("pressure [Pa]")
        axes[0].set_xlabel("r [mm]")
        axes[0].set_ylabel("z [mm]")

        u_r_c, u_z_c = cell_center_velocity(grid, data["u_r"], data["u_z"])
        speed = meridional_speed(u_r_c, u_z_c)
        axes[1].pcolormesh(R, Z, speed, cmap="viridis",
                            vmin=speed_lim[0], vmax=speed_lim[1], shading="auto")
        try:
            axes[1].streamplot(to_mm(grid.r_c), to_mm(grid.z_c), u_r_c.T, u_z_c.T,
                                color="white", linewidth=0.5, density=0.8, arrowsize=0.7)
        except ValueError:
            pass
        axes[1].contour(R, Z, phi, levels=[0.0], colors="k", linewidths=1.2)
        axes[1].set_title("meridional flow [m/s]")
        axes[1].set_xlabel("r [mm]")

        d = vortex_depth(phi, grid, cfg)
        connected_now = is_geometrically_connected(phi, grid, cfg)
        fig.suptitle(f"t={t:.3g}s  RPM={rpm:.0f}  d={d*1e3:.2f}mm  "
                     f"air-core contact={connected_now}", fontsize=10)
        return []

    return _render(fig, update, len(frames), out_path, fps)


def render_air_core_onset_video(run: RunData, out_path: str | Path, fps: int = 10,
                                 max_frames: Optional[int] = None,
                                 r_max: float = 0.020, z_max: Optional[float] = None) -> dict:
    """Video C: zoomed-in close-up of the connectivity criterion itself --
    the top-connected air component, the stirrer target region B_delta,
    and whether they currently intersect (README section 22)."""
    run.require("cfg", "grid", "frames")
    grid, cfg = run.grid, run.cfg
    rpm = cfg.stirrer.rpm
    z_max = z_max if z_max is not None else min(grid.z_max, max(0.050, 2 * cfg.geometry.stirbar_top_z_m))

    frames = run.frames[:max_frames] if max_frames else run.frames
    R, Z = meshgrid_mm(grid)

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    state = ConnectivityState(connected=False)

    def update(i):
        nonlocal state
        ax.clear()
        data = frames[i].load()
        phi = data["phi"]
        t = float(data["t"])

        masks = air_core_masks(phi, grid, cfg)
        ax.pcolormesh(R, Z, masks["all_air"].astype(float), cmap="Greys", vmin=0, vmax=2, shading="auto", alpha=0.4)
        ax.contourf(R, Z, masks["top_connected_air"].astype(float), levels=[0.5, 1.5],
                    colors=["tab:cyan"], alpha=0.6)
        ax.contour(R, Z, masks["stirbar_target"].astype(float), levels=[0.5], colors="tab:orange", linewidths=1.5)
        ax.contour(R, Z, phi, levels=[0.0], colors="k", linewidths=1.2)

        z_tip = find_tip_z(phi, grid, cfg)
        ax.plot([0.0], [z_tip * 1e3], marker="v", color="yellow", ms=10, markeredgecolor="k")

        connected_now = is_geometrically_connected(phi, grid, cfg)
        state = update_persistence(state, connected_now, t, cfg, rpm)
        d = vortex_depth(phi, grid, cfg)

        ax.set_xlim(0, to_mm(r_max))
        ax.set_ylim(0, to_mm(z_max))
        ax.set_xlabel("r [mm]")
        ax.set_ylabel("z [mm]")
        _annotate(ax, t, rpm, d, connected=connected_now, stable=state.connected)
        ax.set_title("air-core connectivity close-up\n"
                      "cyan = top-connected air, orange = stirrer target region", fontsize=9)
        return []

    return _render(fig, update, len(frames), out_path, fps)
