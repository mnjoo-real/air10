"""Statistically-steady-state detection for the vortex depth d(t) = H -
z_tip(t) (README section 21), used to end a run early instead of always
running to a fixed t_end.

Two criteria are evaluated over a trailing window of length ``window_s``:

A. relative range: (max(d) - min(d)) / max(mean(d), eps) < relative_range_tolerance
B. |slope| of a linear-regression fit of d(t) over the window < slope_tolerance

Both must hold on ``required_windows`` consecutive evaluations (each
evaluation uses the full trailing window, so this requires the criteria to
hold continuously for roughly ``window_s + (required_windows-1) *
<time between update() calls>``) before ``is_steady`` becomes True.

This deliberately targets the *physical observable* d(t) via time
statistics (mean, range, trend), not a numerical solver residual -- d(t)
is allowed a bounded oscillation (e.g. from interface dynamics near an
air-core) and can still be judged steady if that oscillation stays inside
tolerance; it does not need to be exactly constant.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SteadyStateConfig:
    enabled: bool = True
    window_s: float = 0.5
    min_time_s: float = 1.0
    relative_range_tolerance: float = 0.01
    slope_tolerance_mm_per_s: float = 0.1
    required_windows: int = 3


@dataclass
class SteadyStateDetector:
    cfg: SteadyStateConfig
    is_steady: bool = False
    steady_time_s: float | None = None
    _t_hist: list = field(default_factory=list)
    _d_hist: list = field(default_factory=list)
    _consecutive_pass: int = 0
    last_relative_range: float = float("nan")
    last_slope_mm_per_s: float = float("nan")

    def update(self, t: float, d: float) -> bool:
        """Feed one new (t, d) observation (d in meters, matching the
        solver's own d(t) convention -- README section 21). Returns True
        exactly once, on the update where ``is_steady`` transitions to True."""
        if not self.cfg.enabled or self.is_steady:
            return False

        self._t_hist.append(t)
        self._d_hist.append(d)

        window_start = t - self.cfg.window_s
        while len(self._t_hist) > 1 and self._t_hist[0] < window_start:
            self._t_hist.pop(0)
            self._d_hist.pop(0)

        # The trailing window can never quite reach a span of exactly
        # window_s at discrete sampling: once the oldest point would exceed
        # window_s in age it gets trimmed, so the retained span asymptotes
        # to window_s - <sampling interval>, not window_s itself. A 0.9
        # factor comfortably accepts that for any reasonable sampling rate
        # (would only reject genuinely-too-short histories, e.g. right
        # after startup) without silently requiring an unreachable span.
        if t < self.cfg.min_time_s or (self._t_hist[-1] - self._t_hist[0]) < self.cfg.window_s * 0.9:
            self._consecutive_pass = 0
            return False

        t_arr = np.asarray(self._t_hist, dtype=float)
        d_arr = np.asarray(self._d_hist, dtype=float)

        d_mean = float(np.mean(d_arr))
        rel_range = float((np.max(d_arr) - np.min(d_arr)) / max(abs(d_mean), 1e-12))

        if np.ptp(t_arr) > 0:
            slope = float(np.polyfit(t_arr, d_arr, 1)[0])  # m/s
        else:
            slope = 0.0
        slope_mm_per_s = abs(slope) * 1e3

        self.last_relative_range = rel_range
        self.last_slope_mm_per_s = slope_mm_per_s

        passed = (rel_range < self.cfg.relative_range_tolerance) and \
                 (slope_mm_per_s < self.cfg.slope_tolerance_mm_per_s)
        self._consecutive_pass = self._consecutive_pass + 1 if passed else 0

        if self._consecutive_pass >= self.cfg.required_windows:
            self.is_steady = True
            self.steady_time_s = t
            return True
        return False
