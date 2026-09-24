"""Free-surface oscillation frequency and damping-rate analysis (README
"oscillation frequency 분석" / "damping rate 분석") for a scalar time series
such as center_depression(t).

Deliberately refuses to report a damping rate when the data doesn't
support one (too few peaks, a poor or non-decaying fit) -- returns
``resolvable=False`` rather than a fabricated number (README: "damping이
너무 작아 신뢰할 수 없으면 억지로 값을 내지 말고 'damping not resolvable' 로 보고한다").
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FrequencyResult:
    f_dominant: float | None  # Hz
    period: float | None      # s
    freqs: np.ndarray
    power: np.ndarray


def dominant_frequency(t: np.ndarray, signal: np.ndarray) -> FrequencyResult:
    """FFT-based dominant oscillation frequency of a (adaptively-sampled)
    time series: resamples onto a uniform grid at the median dt, removes
    the mean, and reports the frequency of the largest non-DC power bin."""
    t = np.asarray(t, dtype=float)
    signal = np.asarray(signal, dtype=float)
    if len(t) < 8:
        return FrequencyResult(None, None, np.array([]), np.array([]))

    dt = float(np.median(np.diff(t)))
    if dt <= 0:
        return FrequencyResult(None, None, np.array([]), np.array([]))

    t_uniform = np.arange(t[0], t[-1], dt)
    if len(t_uniform) < 8:
        return FrequencyResult(None, None, np.array([]), np.array([]))

    sig_uniform = np.interp(t_uniform, t, signal)
    sig_uniform = sig_uniform - np.mean(sig_uniform)

    freqs = np.fft.rfftfreq(len(t_uniform), d=dt)
    power = np.abs(np.fft.rfft(sig_uniform)) ** 2

    if len(freqs) < 2 or np.all(power[1:] == 0):
        return FrequencyResult(None, None, freqs, power)

    idx = 1 + int(np.argmax(power[1:]))
    f_dom = float(freqs[idx])
    period = 1.0 / f_dom if f_dom > 0 else None
    return FrequencyResult(f_dom, period, freqs, power)


@dataclass
class DampingResult:
    resolvable: bool
    lambda_: float | None = None
    tau_d: float | None = None
    fit_quality_r2: float | None = None
    peak_times: np.ndarray | None = None
    peak_amplitudes: np.ndarray | None = None


def find_local_maxima(signal: np.ndarray) -> np.ndarray:
    """Simple positive-local-maxima detector (no scipy.signal dependency):
    index i is a peak if signal[i] is a strict local max and positive."""
    signal = np.asarray(signal)
    peaks = []
    for i in range(1, len(signal) - 1):
        if signal[i] > signal[i - 1] and signal[i] >= signal[i + 1] and signal[i] > 0:
            peaks.append(i)
    return np.array(peaks, dtype=int)


def damping_from_peaks(t: np.ndarray, signal: np.ndarray, min_peaks: int = 3) -> DampingResult:
    """Fits an exponential envelope A(t) ~= A0*exp(-lambda*t) through the
    successive positive peaks of a mean-removed oscillating signal (linear
    least squares on ln(A) vs t). Reports resolvable=False (not a forced
    number) if there are too few peaks, any non-positive peak amplitude,
    a non-decaying (lambda<=0) trend, or a poor fit (R^2 < 0.3)."""
    t = np.asarray(t, dtype=float)
    signal = np.asarray(signal, dtype=float)
    sig0 = signal - np.mean(signal)

    peak_idx = find_local_maxima(sig0)
    if len(peak_idx) < min_peaks:
        return DampingResult(resolvable=False)

    t_peaks = t[peak_idx]
    a_peaks = sig0[peak_idx]
    if np.any(a_peaks <= 0):
        return DampingResult(resolvable=False, peak_times=t_peaks, peak_amplitudes=a_peaks)

    log_a = np.log(a_peaks)
    A = np.vstack([t_peaks, np.ones_like(t_peaks)]).T
    (slope, intercept), *_ = np.linalg.lstsq(A, log_a, rcond=None)
    pred = A @ np.array([slope, intercept])
    ss_res = float(np.sum((log_a - pred) ** 2))
    ss_tot = float(np.sum((log_a - np.mean(log_a)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    lam = -float(slope)
    if lam <= 0 or r2 < 0.3:
        return DampingResult(resolvable=False, lambda_=lam, fit_quality_r2=r2,
                              peak_times=t_peaks, peak_amplitudes=a_peaks)

    return DampingResult(resolvable=True, lambda_=lam, tau_d=1.0 / lam, fit_quality_r2=r2,
                          peak_times=t_peaks, peak_amplitudes=a_peaks)
