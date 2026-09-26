"""Time-windowed angular-momentum budget for validate_forcing_convergence.py logs.

Per window [a, b): Delta L = sum of window_dL, int T dt = sum of window_intT (torques integrated
every step, trapezoid), residual = (Delta L - int T dt) / |int T dt|. Also the window means of
L_z, T_stir and the wall torque T_side + T_bottom.
Usage: python scripts/analyze_forcing_budget.py LOG.jsonl [LOG.jsonl ...]
"""
import json
import sys

import numpy as np

WINDOWS = ((0.0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 3.0), (3.0, 4.0), (4.0, 6.0))


def windows(path):
    R = [json.loads(l) for l in open(path) if '"window_dL"' in l]
    t = np.array([x["t"] for x in R])
    out = []
    for a, b in WINDOWS:
        m = (t >= a) & (t < b)
        if m.sum() < 3:
            continue
        S = [x for x, k in zip(R, m) if k]
        dL = sum(x["window_dL"] for x in S)
        iT = sum(x["window_intT"] for x in S)
        out.append({"a": a, "b": b, "L": np.mean([x["L_z"] for x in S]),
                    "T_stir": np.mean([x["T_stir"] for x in S]),
                    "T_wall": np.mean([x["T_side"] + x["T_bottom"] for x in S]),
                    "dL": dL, "intT": iT, "res": (dL - iT) / abs(iT)})
    return out


if __name__ == "__main__":
    for p in sys.argv[1:]:
        print("==", p.replace("\\", "/").split("results/validation_forcing/")[-1])
        for w in windows(p):
            print(f"   [{w['a']:.1f},{w['b']:.1f}) L {w['L']:.3e}  T_stir {w['T_stir']:.3e}  T_wall {w['T_wall']:.3e}"
                  f"  dL {w['dL']:.3e}  intT {w['intT']:.3e}  residual {w['res']:+.3f}")
