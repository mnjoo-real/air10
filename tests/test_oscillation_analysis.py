import numpy as np

from air_vortex.oscillation_analysis import damping_from_peaks, dominant_frequency


def test_dominant_frequency_recovers_known_sine():
    dt = 0.001
    t = np.arange(0, 2.0, dt)
    f_true = 7.5  # Hz
    signal = np.sin(2 * np.pi * f_true * t)

    result = dominant_frequency(t, signal)

    assert result.f_dominant is not None
    assert abs(result.f_dominant - f_true) < 0.5  # within one FFT bin or so
    assert abs(result.period - 1.0 / f_true) < 0.05


def test_dominant_frequency_none_for_too_short_series():
    result = dominant_frequency(np.array([0.0, 0.1, 0.2]), np.array([1.0, 2.0, 1.0]))
    assert result.f_dominant is None


def test_damping_from_peaks_recovers_known_decay_rate():
    dt = 0.001
    t = np.arange(0, 3.0, dt)
    f_true, lambda_true = 5.0, 1.0  # Hz, 1/s
    signal = np.exp(-lambda_true * t) * np.sin(2 * np.pi * f_true * t)

    result = damping_from_peaks(t, signal)

    assert result.resolvable
    assert abs(result.lambda_ - lambda_true) / lambda_true < 0.15
    assert abs(result.tau_d - 1.0 / lambda_true) / (1.0 / lambda_true) < 0.15
    assert result.fit_quality_r2 > 0.9


def test_damping_not_resolvable_for_growing_oscillation():
    dt = 0.001
    t = np.arange(0, 3.0, dt)
    signal = np.exp(+1.0 * t) * np.sin(2 * np.pi * 5.0 * t)  # growing, not decaying

    result = damping_from_peaks(t, signal)

    assert not result.resolvable


def test_damping_not_resolvable_with_too_few_peaks():
    t = np.linspace(0, 0.1, 20)
    signal = np.sin(2 * np.pi * 2.0 * t)  # barely more than one cycle

    result = damping_from_peaks(t, signal, min_peaks=5)

    assert not result.resolvable


def test_damping_not_resolvable_for_pure_undamped_sine():
    """A constant-amplitude oscillation has lambda ~ 0 -- must not be
    reported as a resolved (positive) decay rate."""
    dt = 0.001
    t = np.arange(0, 3.0, dt)
    signal = np.sin(2 * np.pi * 5.0 * t)  # no decay at all

    result = damping_from_peaks(t, signal)

    assert not result.resolvable
