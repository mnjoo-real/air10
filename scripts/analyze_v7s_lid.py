"""V7-S rigid-lid three-grid convergence (TABLE A: ell_z sensitivity, TABLE B: D10_05, D05_025, R).

Runs: results/validation_v7s/lid/lid_rpm300_dx{1.0,0.5,0.25}_{M1_1.75,M1_3.5}.jsonl/.npz (S1am, M1,
tau_s 5 ms fixed). Observables at matched physical times and as window means over [1.5, 2.0] s:
  GLOBAL: T_stir, T_bottom (total), T_side, T_total = sum, L_z, U_mer, swirl arrival at (5,25) and
          (5,40) mm (5/10/20 % of Omega R_m), upper-L fraction (z > 25 mm), bulk u_theta profile.
  LOCAL:  T_bottom_under (wall stress under the bar), T_bottom_outside.
D10_05 = |q(1) - q(0.5)|, D05_025 = |q(0.5) - q(0.25)| (relative to |q(0.25)|), R = D05_025 / D10_05.
Bulk profile metric: r-weighted relative L2 of u_theta over the liquid EXCLUDING the 2 mm next to the
bottom and the side wall (bulk), finer grid restricted (block mean) to the coarser grid; reported
for 1 vs 0.5 and 0.5 vs 0.25 at t = 1.0 / 1.5 / 2.0 s.
Usage: python scripts/analyze_v7s_lid.py [T_WINDOW_END]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "results" / "validation_v7s" / "lid"
DXS = ("1.0", "0.5", "0.25")
U_TIP = 300 * 2 * np.pi / 60 * 0.015
T_END = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0
WIN = (T_END - 0.5, T_END)


def rows(model, dx):
    L = [json.loads(l) for l in open(D / f"lid_rpm300_dx{dx}_{model}.jsonl") if l.strip()]
    return [x for x in L[1:] if "L_z" in x]


def arrival(R, i, f):
    return next((x["t"] for x in R if x["probes"][i] >= f * U_TIP), None)


def restrict(a, k):
    return a.reshape(a.shape[0] // k, k, a.shape[1] // k, k).mean(axis=(1, 3))


def upper_frac(snap, t):
    s = np.load(snap)
    k = f"t{t:.1f}"
    if k + "_ut" not in s:
        return None
    rc, zc = s["r_c"], s["z_c"]
    w = np.where(s[k + "_liq"], s[k + "_ut"] * rc[:, None] ** 2, 0.0)
    return float(w[:, zc > 0.025].sum() / w.sum())


def bulk_profile_diff(coarse, fine, t, margin=2e-3):
    c, f = np.load(coarse), np.load(fine)
    k = f"t{t:.1f}"
    if k + "_ut" not in c or k + "_ut" not in f:
        return None
    kk = int(round(len(f["r_c"]) / len(c["r_c"])))
    rc, zc = c["r_c"], c["z_c"]
    a, b = c[k + "_ut"], restrict(f[k + "_ut"], kk)
    liq = c[k + "_liq"] & (restrict(f[k + "_liq"].astype(float), kk) > 0.99)
    bulk = liq & (zc[None, :] > margin) & (rc[:, None] < 0.045 - margin)
    w = rc[:, None] * np.ones_like(a)
    return float(np.sqrt(np.sum(((a - b) ** 2 * w)[bulk]) / np.sum((b**2 * w)[bulk])))


def observables(model, dx):
    R = rows(model, dx)
    W = [x for x in R if WIN[0] <= x["t"] <= WIN[1] + 1e-9]
    m = lambda f: float(np.mean([f(x) for x in W]))  # noqa: E731
    o = {"T_stir": m(lambda x: x["T_stir"]), "T_bottom": m(lambda x: x["T_bottom"]), "T_side": m(lambda x: x["T_side"]),
         "T_bottom_under": m(lambda x: x["T_bottom_under"]), "T_bottom_outside": m(lambda x: x["T_bottom_outside"]),
         "U_mer": m(lambda x: x["U_mer"]), "u_theta(10,30)": m(lambda x: x["probes"][5]),
         "u_theta(20,10)": m(lambda x: x["probes"][1])}
    o["T_total"] = o["T_stir"] + o["T_bottom"] + o["T_side"]
    for t in (1.0, 1.5, 2.0):
        x = min(R, key=lambda y: abs(y["t"] - t))
        o[f"L_z({t})"] = x["L_z"]
    for name, i in (("(5,25)", 2), ("(5,40)", 4)):
        for f in (0.05, 0.10, 0.20):
            o[f"arr{name}@{int(f*100)}"] = arrival(R, i, f)
    for t in (1.0, 1.5, 2.0):
        o[f"upperL({t})"] = upper_frac(D / f"lid_rpm300_dx{dx}_{model}.npz", t)
    o["t_reached"] = R[-1]["t"]
    return o


if __name__ == "__main__":
    models = ("M1_1.75", "M1_3.5")
    O = {m: {dx: observables(m, dx) for dx in DXS} for m in models}
    keys = [k for k in O[models[0]]["1.0"] if k != "t_reached"]
    print(f"## TABLE B -- rigid lid 300 rpm, M1 / S1am, window means over [{WIN[0]}, {WIN[1]}] s unless a time is given\n")
    print("| model | observable | 1 mm | 0.5 mm | 0.25 mm | D10_05 | D05_025 | R |")
    print("|---|---|---:|---:|---:|---:|---:|---:|")
    for m in models:
        for k in keys:
            v = [O[m][dx][k] for dx in DXS]
            if any(x is None for x in v):
                print(f"| {m} | {k} | " + " | ".join("-" if x is None else f"{x:.4g}" for x in v) + " | - | - | - |")
                continue
            isarr = k.startswith("arr")
            ref = 1.0 if isarr else (abs(v[2]) if v[2] else 1.0)
            d1, d2 = abs(v[1] - v[0]) / ref, abs(v[2] - v[1]) / ref
            R = d2 / d1 if d1 > 0 else float("inf")
            unit = " s" if isarr else ""
            print(f"| {m} | {k} | {v[0]:.4g} | {v[1]:.4g} | {v[2]:.4g} | {d1:.3g}{unit} | {d2:.3g}{unit} | {R:.2f} |")
    print("\n## Bulk u_theta profile difference (excluding 2 mm wall/bottom bands)\n")
    for m in models:
        for t in (1.0, 1.5, 2.0):
            a = bulk_profile_diff(D / f"lid_rpm300_dx1.0_{m}.npz", D / f"lid_rpm300_dx0.5_{m}.npz", t)
            b = bulk_profile_diff(D / f"lid_rpm300_dx0.5_{m}.npz", D / f"lid_rpm300_dx0.25_{m}.npz", t)
            print(f"  {m} t = {t}: 1 vs 0.5 {a if a is None else round(a, 3)}   0.5 vs 0.25 {b if b is None else round(b, 3)}")
    print("\n## TABLE A -- ell_z sensitivity at equal dx (relative difference 1.75 vs 3.5 mm)\n")
    print("| dx | observable | ell 1.75 | ell 3.5 | rel diff |")
    print("|---:|---|---:|---:|---:|")
    for dx in DXS:
        for k in ("T_stir", "T_bottom", "T_side", "T_total", "L_z(1.0)", "L_z(2.0)", "U_mer", "u_theta(10,30)",
                  "arr(5,25)@10", "arr(5,40)@10", "upperL(1.0)", "upperL(2.0)"):
            a, b = O["M1_1.75"][dx][k], O["M1_3.5"][dx][k]
            if a is None or b is None:
                continue
            rd = (b - a) if k.startswith("arr") else (b - a) / abs(a)
            print(f"| {dx} | {k} | {a:.4g} | {b:.4g} | {rd:+.3f}{' s' if k.startswith('arr') else ''} |")
