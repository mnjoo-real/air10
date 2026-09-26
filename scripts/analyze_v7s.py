"""V7-S TABLES A/B: stirrer-model torque decomposition and grid convergence from
results/validation_v7s/<dir>/lid_rpm300_dx<dx>_<model>.jsonl (validate_forcing_convergence.py logs).

TABLE A at t = 0.05 / 0.1 / 0.2 / 0.5 s: T_stir, T_bottom_under (r < R_m), T_bottom_outside, T_side,
dL/dt (from the log), time-integrated budget residual (cum_residual_rel).
TABLE B: successive changes Delta1 = |q(0.5) - q(1)|, Delta2 = |q(0.25) - q(0.5)| relative to |q(0.25)|,
ratio Delta1 / Delta2 (~2 first order, ~4 second order, <= 1 non-convergent), for T_bottom_under,
T_bottom_outside, T_side, T_stir, L_z. "converging" = Delta2 < Delta1 / 1.5.
Usage: python scripts/analyze_v7s.py [early|lid]
"""
from __future__ import annotations

import glob
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SUB = sys.argv[1] if len(sys.argv) > 1 else "early"
D = ROOT / "results" / "validation_v7s" / SUB
QS = ("T_bottom_under", "T_bottom_outside", "T_side", "T_stir", "L_z")
TIMES = (0.05, 0.1, 0.2, 0.5) if SUB == "early" else (0.5, 1.0, 2.0, 3.0)


def load(path):
    L = [json.loads(l) for l in open(path) if l.strip()]
    return L[0], [x for x in L[1:] if "L_z" in x]


def budget_rel_L(rows, x):
    """(L(t) - L(first log) - int T dt) / L(t) from the per-window logs (model-independent
    normalisation; M2 has T_stir = 0)."""
    dl = sum(y.get("window_dL", 0.0) for y in rows if y["t"] <= x["t"] + 1e-12)
    it = sum(y.get("window_intT", 0.0) for y in rows if y["t"] <= x["t"] + 1e-12)
    return (dl - it) / x["L_z"] if x["L_z"] else float("nan")


def at(rows, t, tol=0.03):
    x = min(rows, key=lambda y: abs(y["t"] - t))
    return x if abs(x["t"] - t) <= tol else None


runs = {}
for p in glob.glob(str(D / "lid_rpm300_dx*_*.jsonl")):
    m = re.search(r"dx([\d.]+?)_([\w.]+)\.jsonl$", p)
    runs[(m.group(2), float(m.group(1)))] = load(p)[1]
models = sorted({k[0] for k in runs})

print(f"## TABLE A -- torque decomposition ({SUB}; S1am, rigid lid, 300 rpm, tau_s 5 ms fixed)\n")
print("| model | dx | t | T_stir | T_bottom_under | T_bottom_outside | T_side | dL/dt | L_z | budget residual / L_z |")
print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
for mod in models:
    for dx in (1.0, 0.5, 0.25, 0.125):
        if (mod, dx) not in runs:
            continue
        for t in TIMES:
            x = at(runs[(mod, dx)], t)
            if x is None:
                continue
            print(f"| {mod} | {dx:g} | {t:g} | {x['T_stir']:.3e} | {x.get('T_bottom_under', float('nan')):.3e} | "
                  f"{x.get('T_bottom_outside', float('nan')):.3e} | {x['T_side']:.3e} | "
                  f"{(x['dLdt'] or 0):.3e} | {x['L_z']:.4e} | {budget_rel_L(runs[(mod, dx)], x):+.1e} |")

print(f"\n## TABLE B -- successive changes (relative to the 0.25 mm value)\n")
print("| model | t | quantity | 1->0.5 | 0.5->0.25 | ratio | converging? |")
print("|---|---:|---|---:|---:|---:|---|")
for mod in models:
    if not all((mod, dx) in runs for dx in (1.0, 0.5, 0.25)):
        continue
    for t in TIMES:
        xs = [at(runs[(mod, dx)], t) for dx in (1.0, 0.5, 0.25)]
        if any(x is None for x in xs):
            continue
        for q in QS:
            v = [x.get(q) for x in xs]
            if any(vv is None for vv in v) or max(abs(vv) for vv in v) < 1e-12:
                continue
            ref = abs(v[2]) if abs(v[2]) > 0 else 1.0
            d1, d2 = abs(v[1] - v[0]) / ref, abs(v[2] - v[1]) / ref
            ratio = d1 / d2 if d2 > 0 else float("inf")
            ok = "yes" if d2 < d1 / 1.5 else ("marginal" if d2 < d1 else "NO")
            print(f"| {mod} | {t:g} | {q} | {d1:.3f} | {d2:.3f} | {ratio:.2f} | {ok} |")
