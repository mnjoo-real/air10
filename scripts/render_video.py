"""Render one of the three result-visualization videos from a saved run,
without re-running the solver.

Usage:
    python scripts/render_video.py --run results/run_H050_RPM1200 --type interface
    python scripts/render_video.py --run results/run_H050_RPM1200 --type pressure_velocity
    python scripts/render_video.py --run results/run_H050_RPM1200 --type air_core
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")

from air_vortex.animation import (  # noqa: E402
    render_air_core_onset_video,
    render_interface_video,
    render_pressure_velocity_video,
)
from air_vortex.run_io import MissingFieldError, load_run  # noqa: E402

VIDEO_NAMES = {
    "interface": "video_interface_evolution.mp4",
    "pressure_velocity": "video_pressure_velocity.mp4",
    "air_core": "video_air_core_onset.mp4",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--type", required=True, choices=list(VIDEO_NAMES))
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--max-frames", type=int, default=None,
                         help="Limit the number of frames rendered (useful for quick previews).")
    args = parser.parse_args()

    run = load_run(args.run)
    try:
        run.require("cfg", "grid", "frames")
    except MissingFieldError as e:
        print(f"Cannot render: {e}")
        return

    out_path = run.paths.videos_dir / VIDEO_NAMES[args.type]

    if args.type == "interface":
        result = render_interface_video(run, out_path, fps=args.fps, max_frames=args.max_frames)
    elif args.type == "pressure_velocity":
        result = render_pressure_velocity_video(run, out_path, fps=args.fps, max_frames=args.max_frames)
    else:
        result = render_air_core_onset_video(run, out_path, fps=args.fps, max_frames=args.max_frames)

    if "video" in result:
        print(f"Saved {result['video']}")
    else:
        print(f"ffmpeg unavailable; saved PNG frame sequence to {result['frames_dir']}")


if __name__ == "__main__":
    main()
