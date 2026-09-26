"""V7-T / continuation Phase F: low-RPM free-surface convergence tables from
validate_forcing_convergence.py free-mode logs (results/validation_v7t/free by default).

t_swirl(probe, f): first log time with u_theta(probe) >= f * Omega R_m (Omega = the run's rpm,
  R_m = 15 mm), probes (5, 25) and (5, 40) mm, f = 5 / 10 / 20 %.
t_deepen(d0): first log time with depression d >= d0 that stays >= d0 for the following 0.1 s
  (or until the run ends / stops); d0 = 0.3 / 0.5 / 1.0 mm (0.5 mm is the headline value; the
  pre-transition surface stays within ~0.05 mm).
Delta t_response = t_deepen(0.5 mm) - t_swirl((5, 40), 10 %).
N_curv = min(N_m, N_theta) over the run up to t (UNRESOLVED < 8 <= MARGINAL < 12 <= PREFERRED).
d_late: mean depression over the last 0.5 s if the run reached its end time without a stop, and
  its drift over that window.
Usage: python scripts/analyze_v7t_free.py [DIR]
"""
from __future__ import annotations

import glob
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
D = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "validation_v7t" / "free"
PROBES = {"(5,25)": 2, "(5,40)": 4}
FRACS = (0.05, 0.10, 0.20)
D0S = (0.3, 0.5, 1.0)


def t_deepen(rows, d0, hold=0.1):
    for k, x in enumerate(rows):
        if x["depression_mm"] >= d0:
            later = [y for y in rows[k:] if y["t"] <= x["t"] + hold]
            if all(y["depression_mm"] >= d0 for y in later):
                return x["t"]
    return None


def summarize(path):
    L = [json.loads(l) for l in open(path) if l.strip()]
    head = L[0]
    rows = [x for x in L[1:] if "L_z" in x]
    stop = next((x for x in L if "stop" in x), None)
    m = re.search(r"rpm([\d.]+?)_dx([\d.]+?)_([\w.]+)\.jsonl$", path)
    rpm, dx, tag = float(m.group(1)), float(m.group(2)), m.group(3)
    U_tip = rpm * 2 * np.pi / 60 * 0.015
    out = {"run": Path(path).stem, "rpm": rpm, "dx": dx, "tag": tag, "t_end": rows[-1]["t"],
           "T_target": None, "stop": stop["t"] if stop else None}
    for name, idx in PROBES.items():
        for f in FRACS:
            out[f"ts{name}@{int(f*100)}"] = next((x["t"] for x in rows if x["probes"][idx] >= f * U_tip), None)
    for d0 in D0S:
        out[f"td@{d0}"] = t_deepen(rows, d0)
    ts, td = out["ts(5,40)@10"], out["td@0.5"]
    out["dt_resp"] = None if ts is None or td is None else td - ts
    nc = [min(x["N_m"], x["N_theta"]) for x in rows]
    out["Ncurv_min"] = float(min(nc))
    out["t_N12"] = next((x["t"] for x in rows if min(x["N_m"], x["N_theta"]) < 12), None)
    out["t_N8"] = next((x["t"] for x in rows if min(x["N_m"], x["N_theta"]) < 8), None)
    out["d_max"] = max(x["depression_mm"] for x in rows)
    out["d_end"] = rows[-1]["depression_mm"]
    out["max_slope"] = max(x["max_slope"] for x in rows)
    late = [x for x in rows if x["t"] >= rows[-1]["t"] - 0.5]
    out["d_late"] = None if stop else float(np.mean([x["depression_mm"] for x in late]))
    out["d_late_drift"] = None if stop else float(late[-1]["depression_mm"] - late[0]["depression_mm"])
    out["T_stir_late"] = float(np.mean([x["T_stir"] for x in late]))
    out["L_late"] = float(np.mean([x["L_z"] for x in late]))
    out["U_mer_late"] = float(np.mean([x["U_mer"] for x in late]))
    out["dV_max"] = max(abs(x["dV"]) for x in rows)
    out["budget_cum"] = rows[-1].get("cum_residual_rel")
    # state just before the N_curv = 12 / 8 crossings (for near-onset records)
    for lab, thr in (("12", 12), ("8", 8)):
        k = next((i for i, x in enumerate(rows) if min(x["N_m"], x["N_theta"]) < thr), None)
        if k is not None and k > 0:
            x = rows[k - 1]
            out[f"pre{lab}"] = {"t": x["t"], "d": x["depression_mm"], "T_stir": x["T_stir"], "L": x["L_z"],
                                "N_m": x["N_m"], "N_theta": x["N_theta"], "slope": x["max_slope"], "U_mer": x["U_mer"]}
    return out


def fmt(v, f="%.2f"):
    return "-" if v is None else f % v


if __name__ == "__main__":
    S = [summarize(p) for p in sorted(glob.glob(str(D / "free_*.jsonl")))]
    S.sort(key=lambda x: (x["rpm"], x["tag"], -x["dx"]))
    print("## Low-RPM free surface (tau_s 5 ms FIXED, uncalibrated; placeholder geometry)\n")
    print("| RPM | method | dx | t_end | stop (N<8) | t_swirl (5,25) 5/10/20 % | t_swirl (5,40) 5/10/20 % | "
          "t_deepen 0.3/0.5/1.0 mm | dt_response | d_late (drift) | d_max | N_curv,min | t(N<12) | T_stir late | L_z late | U_mer late | budget cum | max dV |")
    print("|---:|---|---:|---:|---:|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for x in S:
        a = "/".join(fmt(x[f"ts(5,25)@{p}"]) for p in (5, 10, 20))
        b = "/".join(fmt(x[f"ts(5,40)@{p}"]) for p in (5, 10, 20))
        c = "/".join(fmt(x[f"td@{d}"]) for d in D0S)
        dl = "-" if x["d_late"] is None else f"{x['d_late']:.3f} ({x['d_late_drift']:+.3f})"
        print(f"| {x['rpm']:g} | {x['tag']} | {x['dx']:g} | {x['t_end']:.2f} | {fmt(x['stop'])} | {a} | {b} | {c} | "
              f"{fmt(x['dt_resp'])} | {dl} | {x['d_max']:.2f} | {x['Ncurv_min']:.1f} | {fmt(x['t_N12'])} | "
              f"{x['T_stir_late']:.3e} | {x['L_late']:.3e} | {x['U_mer_late']:.3f} | {fmt(x['budget_cum'], '%+.1e')} | {x['dV_max']:.0e} |")
    print("\nState just before N_curv crossings:")
    for x in S:
        for lab in ("12", "8"):
            if f"pre{lab}" in x:
                p = x[f"pre{lab}"]
                print(f"  {x['run']} before N<{lab}: t {p['t']:.2f} d {p['d']:.3f} mm N_m {p['N_m']:.1f} N_theta {p['N_theta']:.1f} "
                      f"slope {p['slope']:.2f} T_stir {p['T_stir']:.3e} L {p['L']:.3e} U_mer {p['U_mer']:.3f}")
