"""Tables A-D from results/validation_adjoint/adj_*.json."""
import glob
import json
from pathlib import Path

D = Path(__file__).resolve().parents[1] / "results" / "validation_adjoint"
R = []
import sys
pat = sys.argv[1] if len(sys.argv) > 1 else "adj_th*.json"
for f in sorted(glob.glob(str(D / pat))):
    t = open(f).read().strip()
    if t:
        R.append(json.loads(t))
R.sort(key=lambda r: (r["theta"], r["xi"], r.get("pre_case", "F3"), r["pre_t"]))
for r in R:
    r["xi_lab"] = f"{r['xi']:g} [{r.get('pre_case', 'F3')} {r['pre_t']:g}s]"
print("TABLE A  | case | grad-E FD err (eps 1e-7) | Hessian FD err (eps 1e-8) | ||K-K_cont||/||K|| (last 8) | ||K-K_flux||/||K|| (last 8) | P idempotency | P sym in M / M_theta | block decomposition err |")
for r in R:
    fd = r["area_fd"]; p = r["projection"]
    print(f"| th{r['theta']:g} xi{r['xi_lab']} | {fd['1e-07'][0]:.1e} | {fd['1e-08'][1]:.1e} | {r['K_vs_continuum_rel']:.2f} ({r['K_vs_continuum_rel_last8']:.2f}) | "
          f"{r['K_vs_flux_rel']:.2f} ({r['K_vs_flux_rel_last8']:.2f}) | {p['idempotency']:.0e} | {p['M_symmetry']:.2f} / {p['Mtheta_symmetry']:.0e} | {r['block_decomposition_err']:.3f} |")
print("\nTABLE B  | case | W_fluid current | W_fluid EC | W_fluid column-pressure | -W_surface | rel current | rel EC | rel col-p |  (max over 5 random div-free fields)")
for r in R:
    w = max(r["work"], key=lambda x: x["rel_current"])
    print(f"| th{r['theta']:g} xi{r['xi_lab']} | {w['W_fluid_current']:+.3e} | {w['W_fluid_EC']:+.3e} | {w['W_fluid_colpressure']:+.3e} | {-w['W_surface']:+.3e} | "
          f"{max(x['rel_current'] for x in r['work']):.2e} | {max(x['rel_EC'] for x in r['work']):.1e} | {max(x['rel_colpressure'] for x in r['work']):.2e} |")
print("\nTABLE C  | theta | xi | current lambda_max | rel D_adj (measured K, M_theta) | cosine(C_cur, C_EC) | ||C_EC||/||C_cur|| |")
for r in R:
    e = r["eig"]["J3_current"]
    print(f"| {r['theta']:g} | {r['xi_lab']} | {e['lambda_max']:+.3f} ({e['f_max']:.0f} Hz) | {r['adjoint_defect_rel']['measured_Mtheta']:.3f} | "
          f"{r['cosine_C_cur_vs_EC']['measured_Mtheta']:.3f} | {r['EC_strength_vs_current']['measured_Mtheta']:.2f} |")
print("\nTABLE D  | case | current lambda_max (f) | rebuilt current | J2 (no shape feedback) | EC measured K, M_theta | EC measured K, M plain | EC continuum K | EC flux K |  [lambda_max (f Hz) ; least-damped oscillatory]")
for r in R:
    e = r["eig"]
    fmt = lambda x: f"{x['lambda_max']:+.3f} ({x['f_max']:.0f}); osc {x['lambda_osc']:+.2f}@{x['f_osc']:.0f}"
    print(f"| th{r['theta']:g} xi{r['xi_lab']} | {fmt(e['J3_current'])} | {e['rebuilt_current']['lambda_max']:+.3f} | {fmt(e['J2_no_shape_feedback'])} | "
          f"{fmt(e['EC_measured_Mtheta'])} | {fmt(e['EC_measured_Mplain'])} | {fmt(e['EC_continuum_Mtheta'])} | {fmt(e['EC_flux_Mtheta'])} |")
print("\nkappa_var sign/accuracy:", [(r['theta'], r['xi'], r['kappa_var_vs_ref']) for r in R][:2])
