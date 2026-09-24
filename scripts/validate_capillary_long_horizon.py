"""Long-horizon isolated static drop (CAP-B, reinit OFF, volume correction
OFF): does the phi=0 displacement saturate, oscillate, or grow secularly?
Also runs the extension-halo check (capillary extension width below / at
/ above the derived minimum).

Writes results/validation_contact_angle/long_horizon_capillary.csv and
extension_halo.csv, and spurious_velocity.png.

Usage: python scripts/validate_capillary_long_horizon.py [--t-end 4.0]
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from air_vortex.capillary_benchmarks import build_sphere_solver  # noqa: E402
from air_vortex.curvature_single_phase import crossing_positions, interface_curvature  # noqa: E402
from air_vortex.liquid_mask import classify, liquid_volume_subcell  # noqa: E402
from air_vortex.reinit_diagnostics import contour_displacement  # noqa: E402

RS = 0.006


def run(n, t_end, sample_dt=0.02, layers=None):
    s, z0 = build_sphere_solver(RS / n, RS)
    if layers is not None:
        s.cfg.physics.extension_layers_capillary = layers
        s.cfg.physics.allow_unsafe_extension_for_diagnostics = True
    g = s.grid
    phi0 = s.fields.phi.copy()
    V0 = liquid_volume_subcell(g, phi0)
    rows, next_t, umax_window = [], 0.0, 0.0
    while s.fields.t < t_end:
        d = s.step()
        umax_window = max(umax_window, d.max_abs_ur_liquid, d.max_abs_uz_liquid)
        if s.fields.t >= next_t:
            geom = classify(s.fields.phi)
            rr, zr, rz, zz = crossing_positions(g, geom)
            pts = np.array([(a, b) for a, b in zip(np.r_[rr[np.isfinite(rr)], rz[np.isfinite(rz)]],
                                                   np.r_[zr[np.isfinite(zr)], zz[np.isfinite(zz)]])])
            rad = np.hypot(pts[:, 0], pts[:, 1] - z0)
            kr, kz = interface_curvature(g, geom, s.fields.phi)
            k = np.r_[kr[np.isfinite(kr)], kz[np.isfinite(kz)]]
            cd = contour_displacement(g, phi0, s.fields.phi)
            rows.append({"R_over_dx": n, "layers": s.cfg.physics.extension_layers_capillary,
                         "t": s.fields.t, "shift_max_dx": cd.max_shift_dx, "shift_rms_dx": cd.rms_shift_dx,
                         "volume_drift": liquid_volume_subcell(g, s.fields.phi) / V0 - 1,
                         "U_window_max": umax_window, "mean_radius_err_dx": (rad.mean() - RS) / g.dr,
                         "kappa_rms_rel": float(np.sqrt(np.mean((k * RS / 2 - 1) ** 2)))})
            umax_window = 0.0
            next_t += sample_dt
    return rows


def late_slope(rows, key, frac=0.5):
    t = np.array([r["t"] for r in rows])
    y = np.array([r[key] for r in rows])
    m = t >= t[-1] * (1 - frac)
    return float(np.polyfit(t[m], y[m], 1)[0])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--t-end", type=float, default=4.0)
    ap.add_argument("--results-root", default=str(ROOT / "results"))
    args = ap.parse_args()
    out = Path(args.results_root) / "validation_contact_angle"
    out.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for n in (8, 12, 16):
        rows = run(n, args.t_end)
        all_rows += rows
        print(f"R/dx={n}: final shift {rows[-1]['shift_max_dx']:.3f} dx, late slope "
              f"{late_slope(rows, 'shift_max_dx'):+.2e} dx/s, late U {max(r['U_window_max'] for r in rows[len(rows)//2:]):.1e}, "
              f"dV {rows[-1]['volume_drift']:+.1e}", flush=True)
    with open(out / "long_horizon_capillary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0]))
        w.writeheader()
        w.writerows(all_rows)

    halo = []
    for layers in (4, 5, 6, 12):
        rows = run(8, 2.5, layers=layers)
        halo.append({"layers": layers, "t_end": rows[-1]["t"],
                     "late_U": max(r["U_window_max"] for r in rows[len(rows) // 2:]),
                     "final_shift_dx": rows[-1]["shift_max_dx"], "kappa_rms_end": rows[-1]["kappa_rms_rel"]})
        print(f"halo layers={layers}: {halo[-1]}", flush=True)
    with open(out / "extension_halo.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(halo[0]))
        w.writeheader()
        w.writerows(halo)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {8: "#2a78d6", 12: "#eb6834", 16: "#1baf7a"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    for ax, key, lab in ((axes[0], "shift_max_dx", "max phi=0 displacement (dx)"),
                         (axes[1], "U_window_max", "max |u| in liquid (m/s)")):
        ax.set_facecolor("#fcfcfb")
        for n in (8, 12, 16):
            rr = [r for r in all_rows if r["R_over_dx"] == n]
            ax.plot([r["t"] for r in rr], [r[key] for r in rr], color=colors[n], lw=2, label=f"R/dx = {n}")
        ax.set_xlabel("time (s)", fontsize=9, color="#52514e")
        ax.set_ylabel(lab, fontsize=9, color="#52514e")
        ax.grid(True, color="#e4e3df", lw=0.6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(labelsize=8, colors="#52514e")
    axes[1].set_yscale("log")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Isolated static drop, R = 6 mm, reinit OFF, volume correction OFF", fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / "spurious_velocity.png")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
