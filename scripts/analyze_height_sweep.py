"""TABLE D for the height branch: phase statistics per (theta, dx) from run_*.jsonl."""
import glob
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

D = Path(__file__).resolve().parents[1] / "results" / "validation_height"
rows = defaultdict(list)
for f in glob.glob(str(D / "*" / "run_th*_xi*_dx*.jsonl")):
    L = [json.loads(l) for l in open(f) if l.strip()]
    R = [x for x in L if "U" in x]
    if not R:
        continue
    th, xi, dx = map(float, re.findall(r"th([\d.]+)_xi([\d.]+)_dx([\d.]+?)\.jsonl", f)[0])
    T = R[-1]["t"]
    U3 = max(x["U"] for x in R if x["t"] > 0.9 * T)
    U1 = max((x["U"] for x in R if 0.9 < x["t"] <= 1.0), default=np.nan)
    rows[(th, dx)].append({"xi": xi, "T": T, "U3": U3, "U1": U1, "rmse": R[-1]["rmse_dx"],
                           "dV": max(abs(x["dV"]) for x in R), "blow": any("blowup" in x for x in L)})
print("| theta | dx | n | T_end | median U_end | worst U_end (xi) | phase std U | median RMSE/dx | worst RMSE/dx (xi) | max |dV| | unstable |")
print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
for (th, dx), L in sorted(rows.items()):
    U = np.array([x["U3"] for x in L]); E = np.array([x["rmse"] for x in L])
    w, we = L[int(U.argmax())], L[int(E.argmax())]
    unst = sum(x["blow"] or x["U3"] > 1e-3 or (np.isfinite(x["U1"]) and x["U3"] > x["U1"] and x["U3"] > 1e-4) for x in L)
    print(f"| {th:g} | {dx:g} | {len(L)} | {min(x['T'] for x in L):.2f} | {np.median(U):.1e} | {U.max():.1e} ({w['xi']:g}) | {U.std():.1e} | "
          f"{np.median(E):.4f} | {E.max():.4f} ({we['xi']:g}) | {max(x['dV'] for x in L):.0e} | {unst} |")
