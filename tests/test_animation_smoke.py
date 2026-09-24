"""Frame-generation smoke tests for the video renderers (README section 13:
"animation은 full render 대신 frame generation smoke test 정도면 충분하다").
Does not require ffmpeg: when unavailable, _render() falls back to a PNG
frame sequence, which is exactly what these tests check for."""
from pathlib import Path

from air_vortex.animation import (
    render_air_core_onset_video,
    render_interface_video,
    render_pressure_velocity_video,
)
from air_vortex.run_io import load_run
from _run_helpers import make_tiny_run


def _check_output(result, out_path: Path):
    if "video" in result:
        assert result["video"].exists() and result["video"].stat().st_size > 0
    else:
        frames_dir = result["frames_dir"]
        pngs = list(frames_dir.glob("frame_*.png"))
        assert len(pngs) >= 1
        assert all(p.stat().st_size > 0 for p in pngs)


def test_render_interface_video_smoke(tmp_path):
    paths = make_tiny_run(tmp_path, run_id="anim_interface", n_frames=3)
    run = load_run(paths.root)

    out_path = paths.videos_dir / "video_interface_evolution.mp4"
    result = render_interface_video(run, out_path, fps=5, max_frames=2)
    _check_output(result, out_path)


def test_render_pressure_velocity_video_smoke(tmp_path):
    paths = make_tiny_run(tmp_path, run_id="anim_pv", n_frames=3)
    run = load_run(paths.root)

    out_path = paths.videos_dir / "video_pressure_velocity.mp4"
    result = render_pressure_velocity_video(run, out_path, fps=5, max_frames=2)
    _check_output(result, out_path)


def test_render_air_core_onset_video_smoke(tmp_path):
    paths = make_tiny_run(tmp_path, run_id="anim_air_core", n_frames=3)
    run = load_run(paths.root)

    out_path = paths.videos_dir / "video_air_core_onset.mp4"
    result = render_air_core_onset_video(run, out_path, fps=5, max_frames=2)
    _check_output(result, out_path)
