"""Render the single-run figures (3-6) from a saved run directory, without
re-running the solver.

Usage:
    python scripts/render_run.py --run results/run_H050_RPM1200
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")
import numpy as np  # noqa: E402

from air_vortex.plotting import (  # noqa: E402
    plot_azimuthal_velocity,
    plot_interface_snapshots,
    plot_meridional_flow,
    plot_pressure_interface,
)
from air_vortex.run_io import MissingFieldError, load_run  # noqa: E402
from air_vortex.viz_common import save_figure  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--n-snapshots", type=int, default=6,
                         help="Number of time panels in Figure 3 (evenly spaced).")
    args = parser.parse_args()

    run = load_run(args.run)
    try:
        run.require("cfg", "grid", "frames")
    except MissingFieldError as e:
        print(f"Cannot render: {e}")
        return

    rpm = run.cfg.stirrer.rpm

    n = min(args.n_snapshots, run.n_frames)
    idxs = sorted(set(int(round(x)) for x in np.linspace(0, run.n_frames - 1, n)))
    snapshots = [(run.frames[i].t, run.load_frame(i)) for i in idxs]
    fig3 = plot_interface_snapshots(run.grid, run.cfg, snapshots, rpm=rpm)
    saved = save_figure(fig3, run.paths, "fig03_interface_evolution")
    print(f"Saved {saved['png']}\nSaved {saved['pdf']}")

    last = run.load_frame(-1)

    fig4 = plot_pressure_interface(run.grid, run.cfg, last)
    saved = save_figure(fig4, run.paths, "fig04_pressure_interface")
    print(f"Saved {saved['png']}\nSaved {saved['pdf']}")

    fig5 = plot_meridional_flow(run.grid, run.cfg, last)
    saved = save_figure(fig5, run.paths, "fig05_meridional_flow")
    print(f"Saved {saved['png']}\nSaved {saved['pdf']}")

    fig6 = plot_azimuthal_velocity(run.grid, run.cfg, last)
    saved = save_figure(fig6, run.paths, "fig06_azimuthal_velocity")
    print(f"Saved {saved['png']}\nSaved {saved['pdf']}")


if __name__ == "__main__":
    main()
