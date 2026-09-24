"""Render the hydrostatic (Figure 1) or solid-body-rotation (Figure 2)
validation figure from a saved run, without re-running the solver.

Usage:
    python scripts/render_validation.py --run results/validation_hydrostatic
    python scripts/render_validation.py --run results/validation_solid_body
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")

from air_vortex.plotting import plot_hydrostatic_validation, plot_solid_body_validation  # noqa: E402
from air_vortex.run_io import MissingFieldError, load_run  # noqa: E402
from air_vortex.solver import water_volume  # noqa: E402
from air_vortex.viz_common import save_figure  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="Run directory, e.g. results/validation_hydrostatic")
    parser.add_argument("--frame", type=int, default=-1, help="Frame index to render (default: last)")
    args = parser.parse_args()

    run = load_run(args.run)
    try:
        run.require("cfg", "grid", "frames")
    except MissingFieldError as e:
        print(f"Cannot render: {e}")
        return

    kind = run.metadata.get("kind", "production")
    snap = run.load_frame(args.frame)

    if kind == "hydrostatic":
        V0 = None
        if run.n_frames > 0:
            V0 = water_volume(run.grid, run.load_frame(0)["phi"])
        fig = plot_hydrostatic_validation(run.grid, run.cfg, snap, V0=V0)
        saved = save_figure(fig, run.paths, "fig01_hydrostatic_validation")
        print(f"Saved {saved['png']}\nSaved {saved['pdf']}")

    elif kind == "solid_body_rotation":
        omega = run.cfg.stirrer.omega(float(snap["t"]))
        fig = plot_solid_body_validation(run.grid, run.cfg, snap, omega=omega)
        saved = save_figure(fig, run.paths, "fig02_solid_body_validation")
        print(f"Saved {saved['png']}\nSaved {saved['pdf']}")

    else:
        print(f"Run kind '{kind}' has no defined validation figure "
              f"(expected metadata.json 'kind' of 'hydrostatic' or "
              f"'solid_body_rotation'; produced by run_single.py --kind ...).")


if __name__ == "__main__":
    main()
