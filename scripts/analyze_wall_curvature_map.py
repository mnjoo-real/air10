"""Per-(theta, xi, dx) map of lambda_max / least-damped oscillatory mode for one candidate (TABLE C)."""
import glob
import json
import sys
from pathlib import Path

D = Path(__file__).resolve().parents[1] / "results" / "validation_wall_curvature" / (sys.argv[2] if len(sys.argv) > 2 else "jac")
case = sys.argv[1]
n = case.replace(":", "_")
rows = []
for f in glob.glob(str(D / f"eig_{n}_th*_dx*.json")):
    t = open(f).read().strip()
    if not t:
        continue
    r = json.loads(t)
    if r["case"] != case:
        continue
    o = r.get("least_damped_osc") or {"lambda": float("nan"), "freq_Hz": float("nan")}
    rows.append((r["dx_mm"], r["theta"], r["xi"], r["max_lambda"], r["top"][0]["freq_Hz"], r["top"][0]["phi_peak_col"],
                 o["lambda"], o["freq_Hz"]))
rows.sort()
TOL = 0.02
print(f"{case}: {len(rows)} cases, unstable (lambda > {TOL}): {sum(r[3] > TOL for r in rows)}")
for dx, th, xi, lam, f, col, lo, fo in rows:
    flag = "UNSTABLE" if lam > TOL else ""
    print(f"  dx={dx:<6g} th={th:<3g} xi={xi:<5g} lambda_max={lam:+9.3f} (f={f:6.1f} Hz, phi peak col {col:2d})  least-damped osc={lo:+.3f} @ {fo:6.1f} Hz {flag}")
