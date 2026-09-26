"""V7-T rigid-lid analysis (TABLES B and C) from validate_forcing_convergence.py logs + snapshots.

Arrival time: first logged time at which u_theta(probe) >= f * U_tip, U_tip = Omega R_m
(300 rpm, R_m 15 mm -> 0.471 m/s), f = 5 / 10 / 20 %; a grid-independent physical scale, so
the definition is objective and identical for every run. Probes (r, z) mm: (5, 25), (10, 30),
(5, 40), (3, 45) -- the upper-axis region.
Budget: instantaneous = RMS of the per-window residual (Delta L - int T dt) / |int T dt| over all
windows; integrated = (L(t) - L(0) - int_0^t sum T dt) / int_0^t T_stir dt at the end.
Profile difference: the finer snapshot restricted (block mean) to the coarser grid; relative L2
(r-weighted, liquid) of u_theta and of the meridional speed at common snapshot times.
Boundary layers: delta95 / delta99 = wall distance at which u reaches 95 / 99 % of the first
local maximum away from the wall (linear interpolation from the no-slip value 0 at the wall).
Usage: python scripts/analyze_v7t_lid.py [DIR]
"""
from __future__ import annotations

import glob
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
D = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "validation_v7t" / "lid"
U_TIP = 300 * 2 * np.pi / 60 * 0.015
FRACS = (0.05, 0.10, 0.20)
PROBE_IDX = {"(5,25)": 2, "(10,30)": 5, "(5,40)": 4, "(3,45)": 6}
R_V = 0.045


def load(path):
    L = [json.loads(l) for l in open(path) if l.strip()]
    return L[0], [x for x in L[1:] if "L_z" in x]


def arrival(rows, idx, f):
    return next((x["t"] for x in rows if x["probes"][idx] >= f * U_TIP), None)


def summarize(path):
    head, rows = load(path)
    m = re.search(r"dx([\d.]+)_dtf([\d.]+)_(\w+)\.jsonl", path)
    out = {"run": Path(path).stem, "method": m.group(3), "dx": float(m.group(1)), "dtf": float(m.group(2)),
           "t_end": rows[-1]["t"]}
    res = [x["balance_residual_rel"] for x in rows if "balance_residual_rel" in x and x["t"] > 0.1]
    out["budget_rms"] = float(np.sqrt(np.mean(np.square(res)))) if res else None
    out["budget_cum"] = rows[-1].get("cum_residual_rel")
    for name, idx in PROBE_IDX.items():
        for f in FRACS:
            out[f"arr{name}@{int(f*100)}"] = arrival(rows, idx, f)
    late = [x for x in rows if x["t"] >= rows[-1]["t"] - 0.5]
    out["T_stir_late"] = float(np.mean([x["T_stir"] for x in late]))
    out["T_wall_late"] = float(np.mean([x["T_side"] + x["T_bottom"] for x in late]))
    out["U_mer_late"] = float(np.mean([x["U_mer"] for x in late]))
    out["ut_max"] = max(x["u_theta_max"] for x in rows)
    for t in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0):
        x = next((x for x in rows if x["t"] >= t - 1e-9), None)
        out[f"L@{t}"] = x["L_z"] if x else None
    return out


def restrict(a, k):
    return a.reshape(a.shape[0] // k, k, a.shape[1] // k, k).mean(axis=(1, 3))


def profile_diff(coarse, fine, t):
    c, f = np.load(coarse), np.load(fine)
    key = f"t{t:.1f}"
    if key + "_ut" not in c or key + "_ut" not in f:
        return None
    k = int(round(len(f["r_c"]) / len(c["r_c"])))
    rc = c["r_c"][:, None] * np.ones_like(c[key + "_ut"])
    liq = c[key + "_liq"] & (restrict(f[key + "_liq"].astype(float), k) > 0.99)
    out = {}
    for name, fn in (("ut", lambda d: d[key + "_ut"]),
                     ("umer", lambda d: np.hypot(d[key + "_ur"], d[key + "_uz"]))):
        a, b = fn(c), restrict(fn(f), k)
        out[name] = float(np.sqrt(np.sum(((a - b) ** 2 * rc)[liq]) / np.sum((b**2 * rc)[liq])))
    return out


def delta(dist, u, frac):
    """dist: wall distances of cell centres (increasing), u: |velocity| there; wall value 0."""
    i_max = next((i for i in range(1, len(u) - 1) if u[i] >= u[i - 1] and u[i] >= u[i + 1] and dist[i] <= 0.010), None)
    if i_max is None:
        return None, None
    ue = u[i_max]
    d = np.concatenate([[0.0], dist[:i_max + 1]])
    v = np.concatenate([[0.0], u[:i_max + 1]])
    j = int(np.argmax(v >= frac * ue))
    return float(d[j - 1] + (frac * ue - v[j - 1]) / (v[j] - v[j - 1]) * (d[j] - d[j - 1])), float(ue)


def boundary_layers(snap, t):
    s = np.load(snap)
    key = f"t{t:.1f}"
    if key + "_ut" not in s:
        return []
    rc, zc = s["r_c"], s["z_c"]
    dx = rc[1] - rc[0]
    ut, ur, uz = s[key + "_ut"], s[key + "_ur"], s[key + "_uz"]
    rows = []
    for zmm in (10, 25, 40):
        j = int(np.argmin(np.abs(zc - zmm * 1e-3)))
        for comp, fld in (("u_theta", ut), ("u_z", uz)):
            u = np.abs(fld[::-1, j])
            d95, ue = delta(R_V - rc[::-1], u, 0.95)
            d99, _ = delta(R_V - rc[::-1], u, 0.99)
            rows.append({"loc": f"side wall z={zmm} mm", "comp": comp, "dx": dx, "U_e": ue, "d95": d95, "d99": d99})
    for rmm in (20, 30, 40):
        i = int(np.argmin(np.abs(rc - rmm * 1e-3)))
        for comp, fld in (("u_theta", ut), ("u_r", ur)):
            u = np.abs(fld[i, :])
            d95, ue = delta(zc, u, 0.95)
            d99, _ = delta(zc, u, 0.99)
            rows.append({"loc": f"bottom r={rmm} mm", "comp": comp, "dx": dx, "U_e": ue, "d95": d95, "d99": d99})
    return rows


if __name__ == "__main__":
    S = [summarize(p) for p in sorted(glob.glob(str(D / "lid_*.jsonl")))]
    print("## TABLE B -- rigid lid 300 rpm (tau_s 5 ms fixed, uncalibrated; placeholder geometry)\n")
    print("| method | dx | dtf | t_end | budget RMS | budget cum | L(1.5) | L(2.0) | L(3.0) | T_stir late | T_wall late "
          "| arr (5,25) 5/10/20 % | arr (5,40) 5/10/20 % | U_mer late | u_theta max |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|")
    fmt = lambda v, f="%.3e": "-" if v is None else f % v  # noqa: E731
    for x in sorted(S, key=lambda x: (x["method"], -x["dx"], -x["dtf"])):
        a1 = "/".join(fmt(x[f"arr(5,25)@{p}"], "%.2f") for p in (5, 10, 20))
        a2 = "/".join(fmt(x[f"arr(5,40)@{p}"], "%.2f") for p in (5, 10, 20))
        print(f"| {x['method']} | {x['dx']:g} | {x['dtf']:g} | {x['t_end']:.2f} | {fmt(x['budget_rms'], '%.1e')} | "
              f"{fmt(x['budget_cum'], '%+.1e')} | {fmt(x['L@1.5'])} | {fmt(x['L@2.0'])} | {fmt(x['L@3.0'])} | "
              f"{x['T_stir_late']:.3e} | {x['T_wall_late']:.3e} | {a1} | {a2} | {x['U_mer_late']:.3f} | {x['ut_max']:.3f} |")
    print("\nArrival sensitivity, all probes (s) at 5/10/20 %:")
    for x in sorted(S, key=lambda x: (x["method"], -x["dx"])):
        print(f"  {x['run']}: " + "; ".join(f"{n} " + "/".join(fmt(x[f'arr{n}@{p}'], '%.2f') for p in (5, 10, 20))
                                             for n in PROBE_IDX))
    print("\n## Profile differences (relative L2, finer restricted to coarser)\n")
    for meth in sorted({x["method"] for x in S}):
        for a, b in (("1.0", "0.5"), ("0.5", "0.25")):
            ca, fb = D / f"lid_rpm300_dx{a}_dtf1.0_{meth}.npz", D / f"lid_rpm300_dx{b}_dtf1.0_{meth}.npz"
            if not (ca.exists() and fb.exists()):
                continue
            line = []
            for t in (1.0, 1.5, 2.0, 2.5, 3.0):
                d = profile_diff(ca, fb, t)
                if d:
                    line.append(f"t{t}: ut {d['ut']:.3f} umer {d['umer']:.3f}")
            print(f"  {meth} {a} vs {b}: " + "; ".join(line))
    print("\n## TABLE C -- boundary layers (snapshot t = 2.0 s and latest)\n")
    print("| method | dx | t | location | component | U_e | delta95 (mm) | delta99 (mm) | cells/d95 | cells/d99 |")
    print("|---|---:|---:|---|---|---:|---:|---:|---:|---:|")
    for snap in sorted(glob.glob(str(D / "lid_*dtf1.0_*.npz"))):
        meth = re.search(r"dtf1.0_(\w+)\.npz", snap).group(1)
        keys = sorted({k.split("_")[0] for k in np.load(snap).files if k.startswith("t")})
        for tk in sorted({"t2.0", keys[-1]} & set(keys)):
            for r in boundary_layers(snap, float(tk[1:])):
                if r["d95"] is None:
                    continue
                print(f"| {meth} | {r['dx']*1e3:g} | {tk[1:]} | {r['loc']} | {r['comp']} | {r['U_e']:.3f} | "
                      f"{r['d95']*1e3:.3f} | {r['d99']*1e3:.3f} | {r['d95']/r['dx']:.1f} | {r['d99']/r['dx']:.1f} |")


def upper_fraction(snap):
    """Integral transport metric from snapshots: fraction of the liquid angular momentum in
    z > 25 mm, and in the upper-axis box r < 10 mm, z > 25 mm, at each snapshot time."""
    s = np.load(snap)
    rc, zc = s["r_c"], s["z_c"]
    out = {}
    for k in sorted({k.split("_")[0] for k in s.files if k.startswith("t")}, key=lambda k: float(k[1:])):
        ut, liq = s[k + "_ut"], s[k + "_liq"]
        w = np.where(liq, ut * rc[:, None] ** 2, 0.0)
        tot = w.sum()
        up = w[:, zc > 0.025].sum()
        box = w[rc < 0.010][:, zc > 0.025].sum()
        out[float(k[1:])] = (up / tot, box / tot)
    return out


if __name__ == "__main__":
    print("\n## Integral swirl transport: fraction of L_z above z = 25 mm (and in r < 10 mm, z > 25 mm)\n")
    for snap in sorted(glob.glob(str(D / "lid_*dtf1.0_*.npz"))):
        uf = upper_fraction(snap)
        print(f"  {Path(snap).stem}: " + "  ".join(f"t{t:g} {a:.3f}/{b:.4f}" for t, (a, b) in uf.items()))
