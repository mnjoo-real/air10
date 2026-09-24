import numpy as np

from air_vortex.steady_state import SteadyStateConfig, SteadyStateDetector

CFG = SteadyStateConfig(
    enabled=True, window_s=0.5, min_time_s=0.5,
    relative_range_tolerance=0.02, slope_tolerance_mm_per_s=1.0,
    required_windows=3,
)


def _feed(cfg, d_of_t, t_end, dt=0.01):
    det = SteadyStateDetector(cfg)
    t = 0.0
    first_steady_t = None
    while t <= t_end:
        d = d_of_t(t)
        just_steady = det.update(t, d)
        if just_steady and first_steady_t is None:
            first_steady_t = t
        t += dt
    return det, first_steady_t


def test_constant_signal_is_steady():
    det, t_steady = _feed(CFG, lambda t: 0.010, t_end=2.0)
    assert det.is_steady
    assert t_steady is not None
    # cannot be declared before min_time_s (not enough elapsed simulated time)
    assert t_steady >= CFG.min_time_s - 1e-9


def test_exponential_approach_becomes_steady_eventually():
    tau = 0.05  # fast decay relative to window_s=0.5
    d_target, d0 = 0.012, 0.002

    def d_of_t(t):
        return d_target + (d0 - d_target) * np.exp(-t / tau)

    det, t_steady = _feed(CFG, d_of_t, t_end=2.0)
    assert det.is_steady
    assert t_steady is not None
    # by the time it's declared steady, it should actually be close to the target
    assert abs(d_of_t(t_steady) - d_target) < 1e-4


def test_linearly_increasing_signal_is_never_steady():
    det, t_steady = _feed(CFG, lambda t: 0.001 + 0.02 * t, t_end=2.0)  # 20 mm/s, >> 1 mm/s tolerance
    assert not det.is_steady
    assert t_steady is None


def test_small_oscillation_within_tolerance_is_steady():
    mean, amp = 0.010, 0.00005  # 0.05 mm amplitude around 10 mm -> well under 2% range tolerance

    def d_of_t(t):
        return mean + amp * np.sin(2 * np.pi * t / 0.05)  # fast oscillation vs. window_s

    det, t_steady = _feed(CFG, d_of_t, t_end=2.0)
    assert det.is_steady
    assert t_steady is not None


def test_large_oscillation_is_not_steady():
    mean, amp = 0.010, 0.003  # 3mm amplitude on a 10mm mean -> 60% range, way over tolerance

    def d_of_t(t):
        return mean + amp * np.sin(2 * np.pi * t / 0.05)

    det, t_steady = _feed(CFG, d_of_t, t_end=2.0)
    assert not det.is_steady
    assert t_steady is None


def test_discontinuous_jump_is_not_immediately_declared_steady():
    """A step change (e.g. air-core onset) must not be mistaken for steady
    state right after it happens, even though the signal is flat again
    immediately post-jump -- the trailing window still straddles the jump.

    jump_t is placed before min_time_s so the pre-jump plateau itself can
    never be (correctly) declared steady first; this isolates the case the
    detector must get right: the window that straddles the jump."""
    jump_t = 0.2
    assert jump_t < CFG.min_time_s  # so only the post-jump window can ever pass

    def d_of_t(t):
        return 0.005 if t < jump_t else 0.015

    det = SteadyStateDetector(CFG)
    t = 0.0
    became_steady_while_window_straddles_jump = False
    while t <= jump_t + CFG.window_s:  # window still contains pre- and post-jump values
        just_steady = det.update(t, d_of_t(t))
        if just_steady:
            became_steady_while_window_straddles_jump = True
        t += 0.01

    assert not became_steady_while_window_straddles_jump
    assert not det.is_steady

    # but it DOES eventually settle once a full window is entirely post-jump
    while t <= jump_t + CFG.window_s + 1.0:
        det.update(t, d_of_t(t))
        t += 0.01
    assert det.is_steady
    assert det.steady_time_s > jump_t + CFG.window_s
