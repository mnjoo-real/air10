"""First-order sensitivity of the unstable eigenvalue of J_F3 to the curvature-
shape feedback dJ = J_F3 - J_F2 (same base state, same dt), split by the radial
column of the INPUT (which phi perturbation changes kappa) and of the OUTPUT
(where the resulting change acts). d(lambda) for removing block B of dJ:
  dmu = -(y^H dJ_B x)/(y^H x),  dlambda = Re(conj(mu) dmu)/|mu|^2 / dt.
Also: exact eigenvalues of J_F2 + dJ restricted to the wall band (4 columns) and
to the rest, both by input and by output (a linear decomposition, no mixed
curvature definitions)."""
import sys
from pathlib import Path

import numpy as np

D = Path(__file__).resolve().parents[1] / "results" / "validation_pinned_phase" / "jacobian"
tag = sys.argv[1] if len(sys.argv) > 1 else "th60_xi0.25_dx0.25"
A = np.load(D / f"J_F3_{tag}.npz"); B = np.load(D / f"J_F2_{tag}.npz")
J3, J2, dt = A["J"], B["J"], float(A["dt0"])
m_phi, m_ur, m_uz = A["m_phi"], A["m_ur"], A["m_uz"]
Nr = m_phi.shape[0]
col = np.r_[np.nonzero(m_phi)[0], np.clip(np.nonzero(m_ur)[0], 0, Nr - 1), np.nonzero(m_uz)[0]]
nphi = int(m_phi.sum())
kind = np.r_[np.full(nphi, "phi"), np.full(len(col) - nphi, "u")]
dJ = J3 - J2
mu, X = np.linalg.eig(J3)
lam = np.log(np.abs(mu)) / dt
band = (np.abs(np.angle(mu)) / (2 * np.pi * dt) > 100) & (np.abs(np.angle(mu)) / (2 * np.pi * dt) < 250)
k = int(np.argmax(lam)) if lam.max() > 0.05 else int(np.nonzero(band)[0][np.argmax(lam[band])])
m = mu[k]; x = X[:, k]
mut, Y = np.linalg.eig(J3.T)
y = np.conj(Y[:, np.argmin(np.abs(mut - m))])    # left eigenvector: y^H J = mu y^H  ->  J^T conj(y) = mu conj(y)
yH = np.conj(y)
norm = yH @ x
print(f"unstable mode: lambda={lam[k]:+.3f}/s  f={abs(np.angle(m))/(2*np.pi*dt):.1f} Hz")
print(f"relative size of curvature feedback: |dJ|/|J3| = {np.linalg.norm(dJ)/np.linalg.norm(J3):.2e}")

def dlam(mask_rows, mask_cols):
    blk = np.zeros_like(dJ); blk[np.ix_(mask_rows, mask_cols)] = dJ[np.ix_(mask_rows, mask_cols)]
    dmu = (yH @ blk @ x) / norm              # contribution of this block to mu
    return float(np.real(np.conj(m) * dmu) / abs(m) ** 2 / dt)

regions = {"wall1": [Nr - 1], "wall2": [Nr - 2], "wall3": [Nr - 3], "wall4": [Nr - 4],
           "wall5-8": list(range(Nr - 8, Nr - 4)), "mid": list(range(Nr // 4, Nr - 8)), "axis": list(range(0, Nr // 4))}
allr = np.ones(len(col), bool)
print("\ncontribution of the curvature feedback to lambda (1/s), first order; total = sum:")
print(f"  all regions: {dlam(allr, allr):+.3f}")
print("  by INPUT column (phi perturbed there):  " + "  ".join(f"{r}={dlam(allr, np.isin(col, c)):+.3f}" for r, c in regions.items()))
print("  by OUTPUT column (force/response there): " + "  ".join(f"{r}={dlam(np.isin(col, c), allr):+.3f}" for r, c in regions.items()))

def exact(mask_rows, mask_cols, label):
    Jt = J2.copy(); Jt[np.ix_(mask_rows, mask_cols)] += dJ[np.ix_(mask_rows, mask_cols)]
    mt = np.linalg.eigvals(Jt); lt = np.log(np.abs(mt)) / dt; i = int(np.argmax(lt))
    print(f"  {label:48s} max lambda={lt[i]:+.3f}/s at {abs(np.angle(mt[i]))/(2*np.pi*dt):6.1f} Hz")
w4 = np.isin(col, range(Nr - 4, Nr))
print("\nexact eigenvalues of J_F2 + (part of dJ):")
exact(allr, allr, "full feedback (= J_F3)")
exact(allr, w4, "feedback from phi in the 4 wall columns only")
exact(allr, ~w4, "feedback from phi outside the 4 wall columns")
exact(w4, allr, "feedback acting in the 4 wall columns only")
exact(~w4, allr, "feedback acting outside the 4 wall columns")
