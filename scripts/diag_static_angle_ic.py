"""DIAGNOSTIC ONLY: re-run the V4b-S static_angle long-horizon meniscus
(theta = 60, 0.25 mm; recorded blow-up ~2.6 s) from the wall-compatible
initial phi (pinned_phase.wall_compatible_sdf) instead of the distance to
the profile cut at r = R. Prints one JSON line per 0.01 s.

Usage: python scripts/diag_static_angle_ic.py DX_MM T_END {truncated|extended}"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from air_vortex.meniscus import build_meniscus_solver  # noqa: E402
from air_vortex.pinned_phase import wall_compatible_sdf  # noqa: E402
from air_vortex.single_phase_solver import SinglePhaseSolver  # noqa: E402

dx, t_end, ic = float(sys.argv[1]) * 1e-3, float(sys.argv[2]), sys.argv[3]
s, ref = build_meniscus_solver(dx, 60.0, contact_model="static_angle")
if ic == "extended":
    s.fields.phi = wall_compatible_sdf(s.grid, ref)
    s = SinglePhaseSolver(grid=s.grid, cfg=s.cfg, fields=s.fields)
next_t, win = 0.01, 0.0
while s.fields.t < t_end:
    d = s.step()
    u = max(d.max_abs_ur_liquid, d.max_abs_uz_liquid)
    win = max(win, u)
    if not np.isfinite(u) or u > 5.0:
        print(json.dumps({"t": d.t, "blowup": True}), flush=True)
        break
    if d.t >= next_t:
        print(json.dumps({"t": d.t, "U": u, "Uwin": win, "e_rms": 0.0, "ce_dx": 0.0}), flush=True)
        next_t += 0.01
        win = 0.0
