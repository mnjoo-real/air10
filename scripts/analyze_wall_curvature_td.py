"""Time-domain summary of diag_pinned_energy.py runs in results/validation_wall_curvature/td."""
import glob
import json
import sys
from pathlib import Path

import numpy as np

D = Path(__file__).resolve().parents[1] / "results" / "validation_wall_curvature" / "td"
pat = sys.argv[1] if len(sys.argv) > 1 else "G99_th*.jsonl"
print("| run | t_end | U(0.3) | U(1.0) | U(end) | lambda_K/2 (t>1 s) | e_rms/dx end | contact err/dx | vol drift | status |")
print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
for f in sorted(glob.glob(str(D / pat))):
    L = [json.loads(l) for l in open(f) if l.strip()]
    R = [x for x in L if "U" in x]
    if not R:
        continue
    t = np.array([x["t"] for x in R]); U = np.array([x["U"] for x in R]); K = np.array([x["K"] for x in R])
    at = lambda tc: float(U[np.argmin(np.abs(t - tc))])
    m = t > 1.0
    lam = 0.5 * np.polyfit(t[m], np.log(K[m]), 1)[0] if m.sum() > 5 else float("nan")
    ce = R[-1].get("ce_dx", float("nan"))
    blow = any("blowup" in x for x in L)
    bad = blow or lam > 0.05 or R[-1]["e_rms_dx"] > 0.01 or U[-1] > 1e-3
    print(f"| {Path(f).stem} | {t[-1]:.2f} | {at(0.3):.1e} | {at(1.0):.1e} | {U[-1]:.1e} | {lam:+.2f} | "
          f"{R[-1]['e_rms_dx']:.4f} | {ce if isinstance(ce, str) else ce:.4} | {sum(x['dV'] for x in R):+.1e} | "
          f"{'BLOWUP' if blow else ('FAIL' if bad else 'ok')} |")
