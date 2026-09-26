"""Summaries (TABLES B-E) of results/validation_forcing/{lid,free}_rpm*_dx*_dtf*.jsonl."""
import glob
import json
import re
from pathlib import Path

import numpy as np

D = Path(__file__).resolve().parents[1] / "results" / "validation_forcing"


def load(f):
    L = [json.loads(l) for l in open(f) if l.strip()]
    R = [x for x in L if "L_z" in x]
    stop = next((x for x in L if "stop" in x), None)
    return R, stop


def at(R, t, key):
    x = min(R, key=lambda y: abs(y["t"] - t))
    return x[key] if abs(x["t"] - t) < 0.03 else None


rows = []
for f in sorted(glob.glob(str(D / "*_rpm*_dx*_dtf*.jsonl"))):
    m = re.search(r"(\w+)_rpm([\d.]+)_dx([\d.]+)_dtf([\d.]+)\.jsonl", f)
    mode, rpm, dx, dtf = m.group(1), float(m.group(2)), float(m.group(3)), float(m.group(4))
    R, stop = load(f)
    if not R:
        continue
    late = [x for x in R if x["t"] > R[-1]["t"] - 0.5]
    res = [x["balance_residual_rel"] for x in late if "balance_residual_rel" in x]
    Ncurv = min(min(x["N_m"], x["N_theta"]) for x in R)
    rows.append({"mode": mode, "rpm": rpm, "dx": dx, "dtf": dtf, "t_end": R[-1]["t"], "stop": stop,
                 "L_z": R[-1]["L_z"], "T_stir": R[-1]["T_stir"], "T_visc": R[-1]["T_side"] + R[-1]["T_bottom"],
                 "uth_max": R[-1]["u_theta_max"], "U_mer": R[-1]["U_mer"], "probes": R[-1]["probes"],
                 "res": float(np.mean(res)) if res else float("nan"),
                 "depr": R[-1]["depression_mm"], "Ncurv": Ncurv, "R": R})
print("| mode | rpm | dx | dt factor | t_end | L_z | T_stir | T_visc | max u_theta | U_mer | probes (u_theta) | budget residual (late) | depression mm | N_curv min | stop |")
print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---|")
for r in sorted(rows, key=lambda r: (r["mode"], r["rpm"], r["dtf"], -r["dx"])):
    print(f"| {r['mode']} | {r['rpm']:g} | {r['dx']:g} | {r['dtf']:g} | {r['t_end']:.2f} | {r['L_z']:.3e} | {r['T_stir']:.3e} | {r['T_visc']:.3e} | "
          f"{r['uth_max']:.3f} | {r['U_mer']:.3f} | {' '.join('%.3f' % p for p in r['probes'])} | {r['res']:+.3f} | {r['depr']:.3f} | {r['Ncurv']:.0f} | "
          f"{'N<8 @ %.2f s' % r['stop']['t'] if r['stop'] else ''} |")
# common-time comparison
print("\nCommon-time comparison (t = 1.0 / 2.0 / 3.0 s): L_z, T_stir, depression")
for mode, rpm in sorted({(r["mode"], r["rpm"]) for r in rows}):
    for r in sorted([r for r in rows if r["mode"] == mode and r["rpm"] == rpm], key=lambda r: (r["dtf"], -r["dx"])):
        vals = []
        for t in (1.0, 2.0, 3.0):
            L, T, d = at(r["R"], t, "L_z"), at(r["R"], t, "T_stir"), at(r["R"], t, "depression_mm")
            vals.append("t%.0f: L=%s T=%s d=%s" % (t, "%.3e" % L if L else "-", "%.3e" % T if T else "-", "%.3f" % d if d is not None else "-"))
        print(f"  {mode} {rpm:g} rpm dx {r['dx']:g} dtf {r['dtf']:g}: " + " | ".join(vals))
