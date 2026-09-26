"""Spectrum of the per-step samples written by diag_pinned_energy.py (HR_T0):
columns t, eta_axis - eta_ref, mean(eta_wallband - eta_ref), U. Interpretation aid only."""
import sys
from pathlib import Path

import numpy as np

f = Path(sys.argv[1])
d = np.load(f)
t, ea, ew, U = d.T
T = t[-1] - t[0]
dt_s = np.median(np.diff(t))
n = int(T / dt_s)
tu = np.linspace(t[0], t[-1], n)
print(f"{f.name}: {len(t)} samples over {T:.3f} s, median step {dt_s*1e3:.3f} ms "
      f"(Nyquist {0.5/dt_s:.0f} Hz), resolution 1/T = {1/T:.2f} Hz")
for name, y in (("eta_axis", ea), ("eta_wall4", ew), ("U", U)):
    yu = np.interp(tu, t, y)
    yu = (yu - np.polyval(np.polyfit(tu, yu, 1), tu)) * np.hanning(n)
    P = np.abs(np.fft.rfft(yu)) ** 2
    fr = np.fft.rfftfreq(n, tu[1] - tu[0])
    m = fr > 2.0
    k = np.argmax(P[m]); fpk = fr[m][k]
    top = np.argsort(P[m])[::-1][:6]
    peaks = sorted({round(float(fr[m][i]), 1) for i in top})
    print(f"  {name:9s}: peak {fpk:7.2f} Hz +- {0.5/T:.2f}  (top bins {peaks}); "
          f"peak/median power {P[m][k]/np.median(P[m]):.0f}")
R, H, g, s, rho = 0.008, 0.006, 9.81, 0.072, 998.0
for lab, zeros in (("pinned edge J0(kR)=0", (2.405, 5.520, 8.654)), ("free edge J1(kR)=0", (3.832, 7.016, 10.173))):
    fs = [np.sqrt((g * z / R + s * (z / R) ** 3 / rho) * np.tanh(z / R * H)) / (2 * np.pi) for z in zeros]
    print(f"  flat-surface capillary-gravity reference, {lab}: " + ", ".join(f"kR={z}: {q:.1f} Hz" for z, q in zip(zeros, fs)))
