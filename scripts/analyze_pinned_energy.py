"""Summarize results/validation_pinned_phase/energy/*.jsonl (diag_pinned_energy.py run)."""
import glob
import json
import sys
from pathlib import Path

import numpy as np

E = Path(__file__).resolve().parents[1] / "results" / "validation_pinned_phase" / "energy"
T0 = float(sys.argv[1]) if len(sys.argv) > 1 else 0.3        # start of the averaging window
comp = ["R_mom", "R_grav", "R_cap", "R_vol", "dK_ext", "R_E"]
for f in sorted(glob.glob(str(E / "[FWI][0-9]*.jsonl"))):
    L = [json.loads(l) for l in open(f)]
    h = L[0]
    first = next((x for x in L if "first_residual" in x), None)
    R = [x for x in L if "R_E" in x]
    for x in R:
        x["R_mom"] = x["R_pred"] + x["R_proj"]    # the O(dt) gravity-splitting term cancels here
    if len(R) < 5:
        continue
    t = np.array([x["t"] for x in R]); U = np.array([x["U"] for x in R]); K = np.array([x["K"] for x in R])
    m = t >= T0
    lam = np.polyfit(t[m], np.log(U[m]), 1)[0] if m.sum() > 4 else np.nan
    lamK = 0.5 * np.polyfit(t[m], np.log(K[m]), 1)[0] if m.sum() > 4 else np.nan
    Tw = t[m][-1] - t[m][0] + (t[1] - t[0])
    s = {k: sum(x[k] for x, mm in zip(R, m) if mm) / Tw for k in comp}          # mean rate, W
    dtD = sum(x["dtD"] for x, mm in zip(R, m) if mm) / Tw
    reg = {k: sum(x["Rcap_reg"][k] for x, mm in zip(R, m) if mm) / Tw for k in R[0]["Rcap_reg"]}
    wg = {k: sum(abs(x["Wgam_reg"][k]) for x, mm in zip(R, m) if mm) / Tw for k in R[0]["Wgam_reg"]}
    print(f"== {Path(f).name}  t_end={t[-1]:.2f}  U_end={U[-1]:.2e}  K_end={K[-1]:.2e} J  lambda_U={lam:+.2f}/s  lambda_K/2={lamK:+.2f}/s  first={first['first_residual'][:3] if first else None}")
    print("   mean rates over t>=%.2f [W]: " % T0 + "  ".join(f"{k}={v:+.2e}" for k, v in s.items()) + f"  D_mu={dtD:+.2e}")
    print("   R_cap by region [W]: " + "  ".join(f"{k}={v:+.2e}" for k, v in reg.items())
          + "   | mean|W_gamma| by region: " + "  ".join(f"{k}={v:.1e}" for k, v in wg.items()))
