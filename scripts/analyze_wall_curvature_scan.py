"""TABLE B / C summaries of Jacobian scans (eig_*.json from diag_pinned_jacobian.py)."""
import glob

import numpy as np
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

TOL = 0.02          # 1/s: FD/eigensolver uncertainty + near-neutral real modes (see docs)
D = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "results" / "validation_wall_curvature" / "jac"
rows = []
for f in glob.glob(str(D / "eig_*.json")):
    t = open(f).read().strip()
    if not t:
        continue
    r = json.loads(t)
    osc = [x for x in r["top"] if x["freq_Hz"] > 1.0]
    worst_osc = max(osc, key=lambda x: x["lambda"]) if osc else None
    if r.get("least_damped_osc"):
        worst_osc = r["least_damped_osc"]
    else:                                   # older runs: recompute from the saved spectrum
        npz = (Path(__file__).resolve().parents[1] / "results" / "validation_pinned_phase" / "jacobian" /
               f"J_{r['case'].replace(':', '_')}_th{r['theta']:g}_xi{r['xi']:g}_dx{r['dx_mm']:g}.npz")
        if npz.exists():
            A = np.load(npz); mu, dt = A["mu"], float(A["dt0"])
            lam = np.log(np.abs(mu)) / dt; fr = np.abs(np.angle(mu)) / (2 * np.pi * dt)
            o = fr > 1.0
            if o.any():
                k = np.nonzero(o)[0][np.argmax(lam[o])]
                worst_osc = {"lambda": float(lam[k]), "freq_Hz": float(fr[k])}
    rows.append({"case": r["case"], "theta": r["theta"], "xi": r["xi"], "dx": r["dx_mm"],
                 "lam": r["max_lambda"], "f": r["top"][0]["freq_Hz"],
                 "lam_osc": worst_osc["lambda"] if worst_osc else None,
                 "f_osc": worst_osc["freq_Hz"] if worst_osc else None})
by = defaultdict(list)
for r in rows:
    by[r["case"]].append(r)
print(f"| method | cases | worst lambda (1/s) | at (theta, xi, dx) | frequency (Hz) | unstable (lambda > {TOL}) | least-damped OSCILLATORY lambda (freq) |")
print("|---|---:|---:|---|---:|---:|---|")
for c, L in sorted(by.items(), key=lambda kv: max(x["lam"] for x in kv[1])):
    w = max(L, key=lambda x: x["lam"])
    n_un = sum(x["lam"] > TOL for x in L)
    lo = [x for x in L if x["lam_osc"] is not None]
    wo = max(lo, key=lambda x: x["lam_osc"]) if lo else None
    ostr = f"{wo['lam_osc']:+.3f} ({wo['f_osc']:.0f} Hz; th{wo['theta']:g} xi{wo['xi']:g})" if wo else "-"
    print(f"| {c} | {len(L)} | {w['lam']:+.3f} | ({w['theta']:g}, {w['xi']:g}, {w['dx']:g}) | {w['f']:.1f} | {n_un} | {ostr} |")
if "--detail" in sys.argv:
    for c, L in sorted(by.items()):
        print("==", c)
        for x in sorted(L, key=lambda x: (x["dx"], x["theta"], x["xi"])):
            print(f"   dx={x['dx']:g} th={x['theta']:g} xi={x['xi']:<5g} lam={x['lam']:+.3f} f={x['f']:.1f}"
                  + (f"  least-damped oscillatory: {x['lam_osc']:+.3f} @ {x['f_osc']:.1f} Hz" if x['lam_osc'] is not None else ""))
