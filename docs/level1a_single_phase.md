# Level 1A single-phase free-surface solver: Milestone 1 notes

Companion to `README.md` (sections 5–7, 9, 14, 19–20). This file records
**how** Milestone 1 was implemented, the mathematical choices that the README leaves
open, and the plan for the remaining gates. Measured numbers come from
`scripts/validate_single_phase_milestone1.py` (`results/validation_single_phase_m1/`).

## 1. Runtime paths

| `physics.free_surface_model` | class | status |
|---|---|---|
| `two_phase_diffuse_ls` (default, transition) | `solver.Solver` | legacy Level 1B, bit-for-bit unchanged |
| `single_phase_ls` | `single_phase_solver.SinglePhaseSolver` | Level 1A, Milestone 1 (σ = 0, no stirring) |

The only branch point is `solver.build_solver()`. A config without a `physics`
section is legacy, so every existing config, script, test and the golden fixture
run the same code path as before.

The single-phase path does not import `properties.py` (ρ(φ), μ(φ)),
`surface_tension.py` (CSF) or `pressure.py` (variable-density Poisson). A test
enforces this (`tests/test_single_phase_config.py`).

### Guards

- `single_phase_ls` with σ ≠ 0 → `NotImplementedError`. The capillary jump is Milestone D, and its sign comes from Gate V4, not from notation.
- `single_phase_ls` with stirrer RPM ≠ 0 → `NotImplementedError`. Forcing is Milestone E.
- `single_phase_ls` with `swirl_mode="prescribed"` → `ValueError`.
- Legacy with `swirl_mode="prescribed"` → `ValidationOnlyWarning`.
- `run_single.py --swirl-mode prescribed --kind production` → refused.
- The validation-only co-rotating wall (`WallBC.rotating`) is a constructor argument, not a YAML key, so a production config cannot select it.

## 2. Modules

| module | responsibility |
|---|---|
| `liquid_mask.py` | cell classes (void / interface / liquid), face kinds, linear sub-cell crossing θ, θ floor, sub-cell volume |
| `free_surface_bc.py` | interface pressure p_Γ (σ = 0: p_atm = 0 gauge); the single ghost-fluid face-gradient formula |
| `pressure_single_phase.py` | liquid-only r-weighted symmetric matrix, solve, projection, liquid divergence; 1D reference solver |
| `velocity_extension.py` | constant-along-normal narrow-band extension of (u_r, u_z, u_θ/r) |
| `single_phase_solver.py` | time step, `WallBC`, rigid-body manufactured initialiser |
| `curvature_single_phase.py` | axisymmetric kappa = div(grad phi/|grad phi|) at centers, interpolated to the pressure-BC crossings |
| `capillary_benchmarks.py` | CAP-A / CAP-B static-sphere setups (analytic vs numerical curvature) |
| `levelset.py` (appended) | advective-form MUSCL2 (`−div(uφ) + φ div u`), for a non-solenoidal extended velocity |
| `volume_correction.py` | optional `volume_fn` (default unchanged); single-phase passes the sub-cell volume |

## 3. Discretisation choices

### 3.1 Interface pressure (ghost-fluid Dirichlet)

On the grid line from liquid center P (φ_P < 0) to void center Q (φ_Q ≥ 0):

    θ = φ_P / (φ_P − φ_Q) ∈ (0, 1]        (linear zero crossing; never snapped)
    (∂p/∂n)_face = (p_Γ − p_P) / (θ h)

This is the Gibou–Fedkiw–Cheng–Kang (2002) formulation. The same
`ghost_face_gradient` is used in the matrix and in the projection, so every liquid
cell is divergence-free to solver precision after projection.

Property worth knowing: the face gradient is **exact for any p that is affine in φ**.
With φ = z − η(r), rotating hydrostatics gives p = −ρgφ exactly, so the rigid-body
test with that φ only measures round-off. Gate V3 therefore uses the **signed-distance**
φ (`initialize_rigid_body_single_phase(..., phi_form="signed_distance")`). That makes
p non-affine in φ and exercises the O(h²κ) crossing error for real.

### 3.2 Small-θ robustness

θ_eff = max(θ, 10⁻⁶). This is equivalent to moving the Dirichlet point by at most 10⁻⁶ h.
It changes p_P by at most 10⁻⁶ h |∇p|, which is below the O(h²) truncation error on any grid
used here. It is active only when the interface is within 10⁻⁶ cells of a node. Tests measure
the error for θ down to 10⁻¹² (1D P1) and at α = 10⁻⁴ and 0.999 (2D hydrostatic).

A center exactly on φ = 0 is classified as void, so θ = 1 from its liquid neighbour and never 0.

### 3.3 Velocity extension

The extension is not air flow. It solves n·∇q = 0 by first-order upwind
(fast-marching-style) weights, layer by layer, for 3 layers (`physics.extension_layers`),
and sets zero beyond the band. The swirl is extended as **ω = u_θ/r**: the zero tangential
stress for an azimuthal flow is τ_θn = μ r n·∇(u_θ/r) = 0, so constant-ω extension is the
discrete stress-free condition and preserves Ωr exactly. Constant-u_θ extension would
put a spurious shear on a rigidly rotating surface.

### 3.4 Level Set transport

MUSCL2 + MC + SSP-RK2 (unchanged reconstruction), in advective form
−div(uφ) + φ div u. The extended velocity is solenoidal in the liquid but not in the
band, and the flux form alone would add −φ div u there.

### 3.5 Wall BC

Production is `WallBC.no_slip()`: ghost-cell Dirichlet u_θ = 0 at wall and bottom
(same operator as legacy `laplacian_utheta`, verified bit-for-bit). Unlike legacy, it
does not additionally zero the first interior u_θ cell. Gate V3 uses
`WallBC.rotating(Ω)` (container co-rotating: u_θ = Ωr on wall and bottom). A negative-control
test shows Ωr is *not* preserved with the stationary wall.

### 3.6 Time step

Legacy `compute_stable_timestep` with ρ_w, μ_w, plus a gravity-wave limit
dt ≤ 0.5 √(h/g) for the explicit free-surface kinematics.

## 4. Known open issues after Milestone 1

1. **Reinitialisation** is now Gate V5b/V5c; see section 7. The legacy Godunov method is
   kept, but in single_phase_ls it is superseded by `russo_smereka_subcell`.
2. The legacy staircase volume correction chases quantisation noise (it degraded rigid-body
   NRMSE about 8×). The single-phase path uses the sub-cell volume target instead;
   the legacy default is unchanged.
3. Trapped (non-top-connected) void bubbles would also receive p_atm. That is outside
   Level-1A scope (README §23), but the run should flag it once post-contact is simulated.
4. The pressure matrix is rebuilt each step. It has not been profiled (README §14.3: correctness first).
5. Liquid reaching the top row raises `LiquidReachedTopError`. There is no top BC by design.

## 5. Validation status

| gate | status |
|---|---|
| V1 operators / interface pressure | PASS |
| V2 hydrostatic | PASS (round-off) |
| V3 rigid-body core, reinit OFF | PASS |
| V4 surface tension, isolated interface | PASS R/dx >= 8 (reinit OFF / inactive quality trigger); WARN R/dx = 6; FAIL R/dx = 4. Long horizon: shape error bounded (<= 0.2 dx), but a free drop DRIFTS rigidly along the axis (net spurious axial force; neutral mode at g = 0) -- open |
| V4b-S static wall contact geometry | geometry PASS (angle convention, CA0-CA2, wall curvature after ghost fix); Young-Laplace meniscus short-time PASS; LONG-HORIZON wall equilibrium FAIL (static_angle: exponential growth, blow-up ~2.6 s at 0.25 mm, dt-independent; pinned: sub-cell-position-dependent sustained sloshing up to 8e-2 m/s). These long runs started from the distance to the profile cut at r = R (O(1/dx) wall-curvature error, sec. 9.10); rerun of static_angle 0.25 mm from the wall-compatible IC (`scripts/diag_static_angle_ic.py`): starts 10x lower (1e-4 m/s) but still grows exponentially, 1e-3 at 1.4 s, 0.6 m/s at 3 s -- verdict unchanged |
| V4b-P pinned contact line, grid-phase invariance | PARTIAL (sec. 9.10): compatible-IC sweep good on most phases of 0.5 / 0.25 / 0.125 mm, but a reproducible instability island (dx ~ 0.25 mm, xi ~ 0.22-0.26, theta = 60, lambda ~ +1/s, U3s 6e-3) whose root cause is not identified |
| V4b-M moving contact line | NOT IMPLEMENTED (planning only, sec. 9.7) |
| V5a Level Set advection / zero-flow invariance | PASS |
| V5b reinitialization interface preservation | PASS for planes and the V3 paraboloid (RS order 2); legacy Godunov FAIL; under-resolved curvature (circle radius <= 8 cells) still drifts under repeated calls |
| V5c full equilibrium with operational reinit | PARTIAL: shape/volume targets PASS, but a reinit-induced meridional-velocity floor of 1.7-2.8e-3 m/s (2.5-7.7x reinit OFF) converges only weakly |
| V6 timestep | pending (new production path) |
| V7 grid | pending (new production path) |
| V8 stirrer forcing | pending |
| V9 calibration | pending |
| V10 experiment validation | pending |

The pressure/free-surface core (V1-V3) and reinitialization (V5b/V5c) are separate gates.
A reinit failure is not a V3 failure.

## 6. Planned `check_validation_gate.py` restructure (not yet applied)

```
NUMERICAL / FREE-SURFACE VERIFICATION
  V1 operator tests            pytest: operators, liquid_mask, single_phase_pressure (P1/P2, 2D manufactured)
  V2 hydrostatic single-phase  validate_single_phase_milestone1.py  section C
  V3 rigid body, matched BC    validate_single_phase_milestone1.py  section D  (NRMSE<0.05, center<5%, no O(1e-2) flow)
  V4 static capillary jump     (Milestone D, gravity=0 sphere, determines s_kappa)
  V5 Level Set transport/volume  correction OFF and ON, reinit drift  (reinit fix required)
  V6 timestep convergence      single_phase_ls version of run_timestep_convergence.py
  V7 grid convergence          three grids, subcritical + near-contact
PRODUCTION MODEL
  V8 local stirrer forcing sanity   (Milestone E; non-equilibrium, no integrability test)
  V9 experimental forcing calibration (one measured subcritical depth, tau_s frozen)
  V10 validation at unused RPMs
HISTORICAL (reported, never gating)
  legacy two-phase prescribed-swirl exact-equilibrium runs (validation_exact_equilibrium*)
```

The current script is left untouched in this pass, so its legacy report stays reproducible.

## 6b. README migration proposal

- Done (2026-09-24): `README.md` is now the Level-1A architecture document (the former README_rewritten file), with a status/equations summary at the top.
- The full Session 1–7 README is already archived verbatim in
  `docs/legacy_two_phase_diagnostics.md`, so no history is lost by the swap.
- After V1–V7 pass, flip the dataclass default to `single_phase_ls` and add an explicit
  `physics: {free_surface_model: two_phase_diffuse_ls}` to `benchmark.yaml` and the legacy
  validation configs. The golden test then keeps pinning the legacy path explicitly.

## 7. Reinitialization (Gate V5b / V5c)

`levelset.reinitialization_method`: `legacy_godunov` (default; the only method the
two_phase_diffuse_ls solver accepts, unchanged bit-for-bit) or `russo_smereka_subcell`
(`reinit_subcell.py`, single_phase_ls only).

### Method

Sussman's `phi_tau + sgn(phi0)(|grad phi| - 1) = 0` with the Russo–Smereka (2000) subcell fix.
It is implemented in the dimension-by-dimension Min–Gibou form, stated equation by equation in
C. Min, JCP 229 (2010) 2764 §2.2–3:

- ENO2 one-sided differences with minmod second-difference correction;
- at a node whose frozen phi0 changes sign with a neighbour, the difference toward that
  neighbour uses phi = 0 at the interface point (quadratic-ENO root of phi0, falling back
  to the Milestone-1 linear `crossing_fraction`);
- sharp sgn(phi0), Godunov Hamiltonian, local dtau = 0.45 min(dx+-, dz+-), TVD-RK2.

phi0, and therefore the sign, crossings and dtau, is frozen for the whole call. The legacy
method also freezes its sign function per call.

The original RS 2D normalisation Δφ0 is not reproduced here. Only the 1D formula was available
from secondary sources (du Chéné, Min & Gibou 2008 §4); the JCP 2000 paper itself was not
accessible.

`reinitialization_order = 1` (linear crossing, no ENO correction) is implemented for
comparison. It is **rejected operationally**: at 1000 calls it gave NRMSE 9 % and +6 % volume.

### Legacy mechanism (measured, `levelset.reinitialize_level_set`)

The dominant cause is the **across-interface upwind stencil**. At an interface-adjacent node the
one-sided difference reads the value on the other side of phi=0. That value is itself being
updated, so the interface is never a boundary condition and phi=0 slides. The smoothed sign
phi0/sqrt(phi0^2+dx^2) only slows it; a sharp sign alone moves it 4–6x more. Replacing that one
stencil by the subcell Dirichlet point cuts the R4 per-call shift from 4.5e-2 dx to 2.3e-4 dx.

### Metric caveats

- Shifts are **normal** displacements. Along-line crossing shifts diverge where a grid line is
  nearly tangent to the interface.
- phi0's own linear crossing is biased by O(h^2 phi''/phi') when phi0 is nonlinear. A scheme that
  places the interface more accurately (quadratic crossing) therefore shows a nonzero "shift" while
  its error against the exact interface falls. Both are reported (`reinit_benchmarks.py`).

### Key results (`results/validation_reinit/summary.md`)

- **Planes:** exact to round-off when a call is converged. Truncated 5-iteration calls lock in
  <= 1e-4 dx; that error does not grow. Tilted plane after 100 calls: 1.3e-3..9.4e-3 dx, versus
  0.26–0.31 dx for legacy.
- **V3 paraboloid:** 1 call 1.8–4.5e-4 dx and 100 calls 0.8–1.0 % dx on all four grids, versus
  0.55–0.64 dx for legacy.
- **Circle, radius 4–12 cells:** per call ~1–3 % dx; after 100 calls 0.34 / 0.14 / 0.10 / 0.049 dx
  on 1.5 / 1.0 / 0.75 / 0.5 mm (order ~2.7 in mm). Legacy does not converge (0.7 dx).
- **Repeated-call drift** is a property of the converged fixed point, not of truncation. It is
  identical for 5, 20 or 60 iterations per call. Each converged output's crossing differs from
  the pinned crossing by O(h^3 kappa^2), so reinit **frequency** matters, not only the method.
- **Translation (D2):** RS2 adds <= 0.002 dx on top of advection-only error, i.e. no bias.
  Legacy adds 0.02–0.05 dx and −1.2 to −1.8 % volume.
- **V3 operational, every 5 steps (1000 calls/s):** NRMSE 1.3e-3 / 1.1e-3 / 9.5e-4, center
  0.36 / 0.19 / 0.19 %, volume <= 0.06 %. Late meridional velocity is 2.8 / 2.2 / 1.65 e-3 m/s,
  against 1.1 / 0.47 / 0.22 e-3 with reinit OFF and 3–4e-2 for legacy.
- **Quality trigger** (E_sd > 0.01): never fired in the V3 equilibrium and reproduced reinit OFF
  exactly. The threshold is not yet a production default.

### Recommendation

- Use `russo_smereka_subcell`, order 2, whole domain. The band option is kept but not
  recommended while phi0 is strongly distorted, because it is keyed on phi0 values, not distance.
- Reinitialize **only when needed** (quality-triggered or infrequent), not every 5 steps.
- If the high-curvature vortex-tip region needs frequent reinit, evaluate a constrained scheme
  next, in this order: Sussman–Fatemi 1999, then Hartmann–Meinke–Schröder 2008 (JCP 227).

## 8. Surface tension (Gate V4)

Sharp jump only (no CSF anywhere in single_phase_ls):
`p_Gamma = p_atm + S_KAPPA sigma kappa`, with `S_KAPPA = +1` (`free_surface_bc.py`).
`kappa = div(grad phi/|grad phi|)` is positive for a liquid sphere. The sign was fixed by CAP-A:
+1 gives p_liquid = +2 sigma/R; -1 gives -2 sigma/R. A unit test encodes this.

### Curvature operator (`curvature_single_phase.py`)

- Closed-form div(n) from central differences, including the hoop term phi_r/(r|grad phi|).
- There is no node on r = 0. The first column (r = dr/2) uses the even mirror ghost, which
  is exact for phi = f(z) + b r^2 (tested to 2e-10), so the axis limit 2 dn_r/dr + dn_z/dz is
  reproduced without a separate formula.
- kappa is interpolated linearly to each crossing at the SAME theta the pressure BC uses.
- No clipping or smoothing is applied.
- On signed-distance spheres the error is 2nd order: RMS 0.94 / 0.43 / 0.26 / 0.11 / 0.06 / 0.03 %
  at R/dx = 4 / 6 / 8 / 12 / 16 / 24. Cylinder (hoop term only): 2nd order. Plane: exactly 0.

### Pressure-jump vs curvature error (kept separate)

- **CAP-A** (exact kappa): Delta p exact to 1e-15 relative, U ~ 1e-16 m/s, for 2694 steps.
  This isolates and verifies the Dirichlet-at-crossing BC and the projection. It is easy by
  construction, because p_Gamma is constant.
- **CAP-B** (numerical kappa, 1 s, reinit OFF, volume correction OFF): see
  `results/validation_surface_tension/summary.md`. Late U_spurious ≈ 1.6e-3 / 7.7e-4 / 4.3e-4 /
  4.5e-4 / 1.3e-4 m/s at R/dx = 6 / 8 / 12 / 16 / 24. Delta p error <= 0.4 %.
  U_late ≈ (0.6–1.2 m/s) x std(p_Gamma)/Delta p across all resolved cases: the chain
  curvature error -> p_Gamma nonuniformity -> spurious current holds quantitatively.
- R/dx = 4 **fails**: kappa error grows to 83 %, U -> 0.27 m/s, volume −21 % within 1 s.
- Time-step independent (dt factor 1 / 0.5 / 0.25: U_late 3.6e-4 for all).
  dt_sigma = factor * sqrt(rho dx^3 / (4 pi sigma)) (Brackbill-Kothe-Zemach 1992, gas density 0).

### Extension-band finding

With the sigma = 0 width (3 layers), the static drop was stable for about 0.75 s and then grew
exponentially (E_sd -> 0.8, U -> 0.5 m/s by 1.25 s), independent of dt. Cause: the extended
velocity drops to zero at the band edge, and the resulting |grad phi| kink (measured at
phi ≈ +2.4 dx) sits within reach of the curvature and MUSCL stencils.

`physics.extension_layers_capillary` (default 6) is now used when sigma > 0; 6 and 12 layers
behave identically. The sigma = 0 path still uses `extension_layers` = 3, so V2/V3 are
bitwise unchanged. Unifying at 6 should be revisited before production (stirring).

### Reinitialization with surface tension

- The quality trigger (E_sd > 0.01) never fired on the static drop.
- Periodic RS2 (every 50 steps) **worsened** curvature at every event (RMS 1.1e-3 -> 2.4e-3 at the
  first event), p_Gamma spread, and U_late (4.3e-4 -> 2.7e-3). It also did not improve E_sd on
  this near-distance field.
- Recommendation: with sigma > 0, reinitialize only when triggered by a quality metric, never
  periodically. A curvature-aware check (kappa noise before/after) should accompany any reinit.

### Resolution threshold (production relevance)

For curvature RMS < 0.5 %, Delta p error < 0.5 % and U_spurious < 1e-3 m/s over 1 s:
**R/dx >= 8**; R/dx >= 12 is comfortable. So the minimum trustworthy interface radius is
R_min ≈ 8 dx. That is 4 mm at dx = 0.5 mm, 2 mm at 0.25 mm, and 1 mm at 0.125 mm. An air-core
tip with a smaller radius of curvature is NOT resolved on that grid.

### V4b (planning only): wall contact

- The existing capillary-rotating reference (`capillary_equilibrium.py`) assumes eta'(R_v) = 0,
  i.e. a 90-degree contact angle.
- The single-phase curvature and RS2 stencils use linear-extrapolation ghosts at the wall, so the
  solver does NOT currently impose 90 degrees. The wall behaviour is whatever the local phi slope
  implies.
- Before the integrated capillary-gravity rotating test, V4b must choose and implement an explicit
  contact condition. That means a configurable static angle theta_c, enforced on phi's wall ghost,
  with sensitivity runs; the water–glass angle should be measured, not assumed.
- Only then is the comparison with the capillary-corrected reference clean.

## 9. Wall contact (Gate V4b)

Results: `results/validation_contact_angle/` (summary.md, CSVs, figures) and
`scripts/validate_capillary_long_horizon.py` (isolated drop, extension halo).

### 9.1 Long-horizon isolated drop (V4 refinement)

- 4 s runs at R/dx = 8 / 12 / 16. The total phi=0 displacement grows at a late slope of
  +0.5 / +0.5 / +1.6 dx/s.
- Measured against a sphere TRANSLATED with the liquid centroid, the shape error stays
  bounded at <= 0.18 dx (R/dx 8) and <= 0.12 dx (R/dx 12).
- So the secular part is a rigid axial drift of the free drop, a neutral mode at g = 0 driven by
  a small net spurious axial force. It cannot occur for liquid resting on the bottom, but it is a
  momentum-consistency defect and stays OPEN.

### 9.2 Extension halo (structural)

`velocity_extension.minimum_capillary_extension_layers()` = 1 + 2 + 2 = 5, from the actual stencils:

- the curvature evaluation node Q (layer 1);
- the 3x3 curvature stencil (Manhattan reach 2);
- MUSCL2 + SSPRK2 transport dependence (2).

The solver refuses sigma > 0 with fewer layers unless
`physics.allow_unsafe_extension_for_diagnostics` is set. The default 6 is the minimum plus one.

Measured over 2.5 s at R/dx = 8: 4 / 5 / 6 / 12 layers behave alike, and 3 layers diverged by
1.25 s. So the derived minimum is slightly conservative, which is the safe direction.

### 9.3 Contact-angle convention (verified)

- theta is measured through the liquid. eta'(R) = cot(theta) and n . e_r = -cos(theta), with n
  pointing liquid -> air and e_r the wall's outward normal.
- theta < 90 rises toward the wall. This is encoded in tests/test_contact_angle.py.
- In axisymmetry a straight meridional line is a CONE. Its exact curvature is the hoop term
  -cos(theta)/r, not 0, except at 90 deg.

### 9.4 Wall ghost (contact_angle.py) -- Outcome B found and fixed

The first construction failed to converge at the wall:

- A linear-extrapolation ghost forces phi_rr = 0 in the wall column: 30-45 % wall curvature error
  on a spherical cap, non-convergent.
- Imposing theta on every level set is inconsistent with a curved distance function: 6 % error,
  non-convergent.

The current construction:

- base = cubic extrapolation (4, -6, 4, -1);
- static_angle adds one wall-slope correction h*dg evaluated only at the contact point, with a
  local cubic in z.

Result: spherical-cap wall curvature error <= 1.2e-4 relative on all grids (sub-cell floor
~2e-5, no clean order); bulk curvature 2nd order. The isolated V4 drop is bitwise unchanged.

### 9.5 Wall models tested dynamically (no-slip wall, reinit OFF, theta = 60 deg meniscus)

| model | behaviour |
|---|---|
| `extrapolate` (angle free) | **unstable**: round-off on a flat 90-degree surface grows to 0.2 m/s within 0.3 s. Not usable. |
| `static_angle` | short-time shape good (RMSE ~0.016 dx). Long runs: 0.5 mm decays to a ~4e-3 m/s tail; 0.25 mm decays to 1.6e-3, then grows exponentially and blows up at ~2.6 s. The growth is independent of dt (factor 0.5 blows up at the same time) and is not cured by quality-triggered RS2 (which kept E_sd < 0.01). 0.125 mm reaches 7e-2 m/s within 0.75 s. |
| `pinned` (CL-P) | contact height held to <= 0.01 dx, but the dynamics depend on WHERE z_pin falls inside its cell. On one 0.25 mm grid, sub-cell fraction 0.02 decays cleanly (2.6e-5 m/s), 0.74 sustains 2.4e-2 m/s, and 0.46 sustains 2-8e-2 m/s. The clean 0.5 / 0.125 mm runs were favourable positions. This is a pinned-ghost defect to fix before this model can be trusted. |

A flat start with a prescribed 60-degree angle shows the contact line climbing 0.34 / 0.40 mm (of
the 0.81 mm equilibrium rise) within 0.2 s at dx = 0.5 / 0.25 mm. With a no-slip wall this motion
can only happen through the half-cell numerical slip. Its weak grid dependence is the logarithmic
slip-length dependence expected of contact-line physics. That is Outcome C: fixed-angle transient
motion with no-slip is set by numerics, not physics.

### 9.6 Contact-line policy (Table D)

| model | line mobile? | prescribed theta? | no-slip compatible as used? | status |
|---|---|---|---|---|
| pinned (CL-P) | no | no (angle free) | yes | preferred physical candidate, but its ghost is sub-cell-position dependent (sustained 1e-2 m/s sloshing for unfavourable z_pin) -> needs a fix |
| fixed-angle static equilibrium | equilibrium only | yes | only if nothing needs to move | geometry validated; long runs unstable -> not usable |
| mobile fixed-angle (CL-A) | yes | yes | NO (numerical slip only) | not physically posed; do not use in production |

### 9.7 V4b-M (planning only)

If the experiment shows a moving contact line, candidates in order:

1. Navier slip on the side wall near the contact line (slip length as a parameter, grid-converged);
2. static angle + slip;
3. advancing/receding hysteresis;
4. a dynamic contact-angle law.

None is implemented, and none should be without experimental evidence.

### 9.8 Experiment needed (add to the experiment plan)

- Side-view recording of the liquid-wall intersection during an RPM ramp, recording z_contact(t)
  and the apparent angle theta_app(t). At minimum: does z_contact stay pinned?
- Baseline static meniscus/contact angle of water before stirring.
- Ramp up AND ramp down, to detect hysteresis.
- Outcome D (pinned over the onset range) justifies the pinned model. Outcome E (substantial
  motion) makes V4b-M necessary.

### 9.9 Resolution criterion

- The isolated-sphere rule R/dx >= 8 is renamed the **isolation-benchmark curvature criterion**.
  It is not a universal air-core-tip guarantee.
- `curvature_single_phase.principal_resolution` reports N_m = R_m/dx and N_theta = R_theta/dx
  separately. The minimum over the tip region, not 1/|kappa_total|, must be used in production.

### 9.10 V4b-P: grid-phase sweep and the dx = 0.25 mm / xi = 0.25 anomaly (session 2026-09-25)

Artifacts: `results/validation_pinned_phase/` (`dyn_sweep_grids.jsonl`, `dyn_sweep_025.jsonl`:
historical truncated-IC sweeps; `sweep_ic_extended/`: compatible-IC sweep; `diag/`: mechanism
runs, `diag/invalid_nonint_Nr/`: quarantined invalid runs). Scripts:
`scripts/run_pinned_phase_case.py` (now takes `ic=extended`, records `max_div`),
`scripts/diag_pinned_phase_mechanism.py` (DIAGNOSTIC ONLY: ts / early / static modes, exact-BVP
injection variants), `scripts/diag_static_angle_ic.py`.

**Phase shift implementation (continuum problem unchanged).** `pinned_phase.depth_for_phase`
adds a uniform liquid layer delta = ((xi - xi0) mod 1) dz. In a flat-bottomed cylinder this is an
exact vertical translation of the gravity-capillary meniscus: z_pin, H and V = pi R^2 H shift by
delta and pi R^2 delta, the pinned BVP shape is identical (theta_eq = 60.000000 deg to 1e-12 in every
record), vessel radius, g, sigma and rho are fixed. The air layer (unused by the single-phase
solver) keeps its thickness, so the domain top moves with delta; it never meets the liquid.

**Result recovery.** The ten 0.125 mm truncated-IC runs launched by the expired session all
completed after it expired (last record 2026-09-24 17:43, T = 3.00 s, no `blowup_t`); none was
rerun. No process survived.

#### Finding 1 -- the validation initial condition was incompatible with the wall (fixed in the harness)

`single_phase_solver.signed_distance_to_profile` samples the profile only for |r| <= R. For a tilted
meniscus the level sets phi = +-O(dx) next to the wall are then distances to the ENDPOINT
(R, z_pin): circles of radius O(dx). The level-set curvature of the wall column is therefore
O(1/dx) wrong, and without reinitialization it is never removed. Measured at t = 0 (theta = 60,
all phases):

| dx (mm) | wall-crossing p_Gamma error | kappa error | kappa error x dx |
|---:|---:|---:|---:|
| 0.5 | +1.5..+1.9 Pa | +21..+25 1/m | 12 mm/m |
| 0.25 | +2.7..+3.2 Pa | +38..+45 1/m | 11 |
| 0.20 | +3.4..+4.0 Pa | +47..+55 1/m | 11 |
| 0.125 | +5.7..+6.4 Pa | +79..+89 1/m | 11 |

(P0 ~ 50 Pa.) It is the global maximum pressure error, and it explains why the historical
`p_resid_t0` INCREASED under refinement (7e-3 -> 1e-2 -> 1.5e-2). theta = 75 has a 30x smaller error
(flatter wall slope), so theta = 75 was NOT a like-for-like control.

Fix (validation harness only, no solver change): `pinned_phase.wall_compatible_sdf` = distance to
the BVP profile continued beyond R (`extended_profile`), selected by
`build_phase_solver(..., ic="extended")`. The t = 0 wall error drops to -0.4..-1.3 Pa (0.25 mm:
3.2 -> 0.66 Pa). The default stays `ic="truncated"` so the historical artifacts remain reproducible.
The same truncated IC is used by `meniscus.build_meniscus_solver`, so the V4b-S `static_angle`
long-horizon verdict (sec. 9.5) was also measured from an incompatible IC (see below).

#### Finding 2 -- non-integer R/dx grids are a different continuum problem (guard added)

`grid.build_grid` uses Nr = round(R/dr); for R/dx not an integer the wall face sits at Nr dr != R
while the reconstruction and BVP use R. The first intermediate-dx runs (0.35, 0.30, 0.275, 0.225,
0.175 mm) showed wall pressure errors of 20-200 Pa for this reason; they are quarantined in
`diag/invalid_nonint_Nr/`. `build_phase_solver` now raises for non-integer R/dx. Valid intermediate
grids are dx = 8 mm / N.

#### Finding 3 -- the anomaly is a narrow, intrinsic discrete instability island

theta = 60, production method (reconstruct_ghost, cubic, 4 columns), growth rate lambda = d ln U/dt
fitted over the last 0.7 s:

| dx (mm) | N = R/dx | xi | lambda (1/s) | note |
|---:|---:|---:|---:|---|
| 0.333 | 24 | 0.25 | -1.6 | decays |
| 0.286 | 28 | 0.25 | -1.2 | decays |
| **0.25** | 32 | 0.20 | -0.9 | decays |
| **0.25** | 32 | **0.225** | **+0.7** | grows |
| **0.25** | 32 | **0.25** | **+1.35** | grows (U3s 1.6e-2) |
| **0.25** | 32 | 0.275 / 0.30 | -1.0 / -0.7 | decays |
| 0.222 | 36 | 0.25 | -1.5 | decays |
| 0.20 | 40 | 0.25 | -1.5 | decays |
| 0.182 | 44 | 0.25 | -0.5 | decays (slowly) |
| 0.167 | 48 | 0.25 | -1.7 | decays |
| 0.125 | 64 | 0.25 | (U3s 8.4e-5) | decays |

The unstable set is roughly dx ~ 0.25 mm (neighbouring valid grids stable), xi ~ 0.22-0.26.
It is NOT an artifact of the incompatible IC: from the compatible IC (variant X) the same case
starts 5x lower but still grows (lambda ~ +1.0/s; U = 1.5e-3 m/s at 2 s > 1e-3 target).

**Source vs response.** Early-step dumps (steps 0/1/2/5/10/20/50/100/200) put the FIRST residual at
the wall: at step 1 the largest acceleration is on the u_z face of the wall column at the
interface row (wall/axis ratio ~800). The axis response appears only after ~100-200 steps.
The growing mode is axis-peaked (interface amplitude axis : mid : wall = 0.034 : 0.009 : 0.003 dx),
dominant ~42 Hz in the 100 Hz-sampled log (aliasing not excluded) -- consistent with a low-order
physical axisymmetric capillary-gravity sloshing mode (free-edge 2nd mode 38 Hz; a pinned edge
raises it), not a grid-scale mode. So the r = 0 maximum is the response of a global mode fed at
the wall.

**No discrete branch switching** was found: sample rows, reconstruction coefficients (cond = 147
at every phase/grid, set by the fixed sample abscissae), cell classification and stencil are
constant over steps 0-200 in the bad case and its neighbours. At t = 0 the bad case and its stable
neighbours (xi = 0.15, 0.35) have the same wall error (3.14 / 3.22 / 3.19 Pa), so the static error
does not select the phase; the phase selects the sign of a small net damping.

**A/B/C exact-injection (diagnostic only, never production):**

| variant | U(0.1) | U(1.0) | late lambda | reading |
|---|---:|---:|---:|---|
| A production | 2.1e-3 | 4.8e-3 | +1.35 | unstable |
| B exact BVP curvature at wall-column crossings | 1.4e-4 | 1.1e-4 | +1.25 (U3s 1.4e-3) | seed 15x lower, SAME growth rate |
| C exact static ghost, numerical curvature | 1.2e-3 | 1.7e-2 | saturates ~1.5e-2 | worse (static ghost inconsistent with truncated interior) |
| BC | = B | | | ghost only enters via the wall-column curvature, which B replaces |
| B4 exact curvature in the 4 fit columns | 9.7e-5 | 7.1e-6 | -1.9 | stable |
| X compatible IC | 6.1e-4 | 6.9e-4 | +1.0 | unstable (U2s 1.5e-3) |
| XB compatible IC + B | 1.4e-4 | 1.1e-4 | +1.3 | unstable |
| K2 / K12 / K123 exact curvature in column subsets | | | explodes / ~0 / +4.8 | mixed exact/numerical columns are themselves inconsistent: inconclusive |

Interpretation under the agreed rule: B unstable and C unstable, so neither the wall-curvature
value nor the ghost geometry alone is causal. The wall-column curvature error sets the SEED
amplitude; the growth is a weakly anti-damped (~ +1/s) global sloshing mode whose energy input is
the discrete near-wall capillary coupling (removing capillary feedback from the 4 wall columns, B4,
removes it; larger viscosity suppresses it; dt and extension width do not). The root cause of the
negative damping (pressure/velocity wall stencil vs level-set kinematics at the pinned column) is
NOT identified.

#### Phase-marginalized sweep, theta = 60 (3 s, 10 phases per grid)

"truncated" = historical IC, "extended" = wall-compatible IC (`sweep_ic_extended/`, per-phase
rows in `summary_table.md`). U3s = max liquid |u| over [2.7, 3.0] s.

| IC | dx | n | median U3s | worst U3s (xi) | best U3s | phase std | log10 std | median RMSE | worst RMSE (xi) | RMSE>0.01 | U3>1e-3 | U3>3e-4 | growing U3>U1 | max|ce| dx | max|vol drift| | max div |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| truncated | 0.5 | 10 | 1.87e-04 | 4.06e-04 (0.45) | 2.48e-05 | 1.29e-04 | 0.37 | 6.79e-03 | 9.20e-03 (0.15) | 0 | 0 | 2 | [] | 0.021 | 1.4e-04 | nan |
| truncated | 0.25 | 10 | 7.46e-05 | 1.57e-02 (0.25) | 2.37e-05 | 4.92e-03 | 0.81 | 6.62e-03 | 1.35e-02 (0.15) | 2 | 1 | 2 | [0.25] | 0.012 | 9.9e-05 | nan |
| truncated | 0.125 | 10 | 6.59e-05 | 2.21e-04 (0.65) | 1.47e-05 | 6.37e-05 | 0.36 | 7.52e-03 | 1.48e-02 (0.35) | 4 | 0 | 0 | [0.65] | 0.009 | 2.8e-05 | nan |
| extended | 0.5 | 10 | 1.61e-04 | 2.04e-04 (0.85) | 1.04e-04 | 3.33e-05 | 0.10 | 1.04e-03 | 2.44e-03 (0.15) | 0 | 0 | 0 | [] | 0.015 | 1.1e-04 | 6.1e-10 |
| extended | 0.25 | 10 | 4.27e-05 | 6.28e-03 (0.25) | 2.13e-05 | 1.97e-03 | 0.71 | 6.39e-04 | 1.43e-02 (0.25) | 1 | 1 | 1 | [0.25] | 0.005 | 6.1e-05 | 5.6e-10 |
| extended | 0.125 | 10 | 6.07e-06 | 4.75e-04 (0.35) | 2.42e-06 | 1.48e-04 | 0.63 | 4.24e-04 | 1.47e-02 (0.35) | 1 | 0 | 1 | [] | 0.005 | 4.4e-06 | 2.0e-08 |

Worst-case behaviour under refinement (compatible IC) is NOT monotonic: worst U3s 2.0e-4 -> 6.3e-3 ->
4.8e-4 m/s, worst RMSE 2.4e-3 -> 1.43e-2 -> 1.47e-2 dx (target 0.01 dx fails on both finer grids),
and the bad phase moves (0.25 at 0.25 mm; 0.35 at 0.125 mm, a burst to 2.6e-3 m/s at 1 s). The
median converges well (1.6e-4 -> 4.3e-5 -> 6.1e-6 m/s), the contact point stays within 0.005 dx and
volume drift is <= 1.1e-4. No run diverged.

#### Finding 3b -- the unstable window moves with the wall slope (not an xi = 0.25 special case)

Compatible IC (X), dx = 0.25 mm, 1.5 s, lambda over the last 0.7 s:

| theta | xi = 0.05 | 0.25 | 0.45 | 0.65 | 0.85 |
|---:|---:|---:|---:|---:|---:|
| 50 | +0.16 (U1 1.3e-3) | +0.33 (U1 6.8e-4) | **+2.0 (U1 2.3e-2, RMSE 0.067 dx)** | -1.5 | -1.5 |
| 60 | (sweep: stable) | **+1.0..+1.3** | stable | stable | stable |
| 70 | -0.8 | -0.5 | -0.4 | -3.2 | -1.2 |

A steeper wall meniscus makes the phase sensitivity stronger and moves the unstable window in xi.
theta = 75 (the earlier "control") and 70 are benign. This matters for production: rotation
steepens the interface at the wall.

Grid dependence at theta = 50 (compatible IC, lambda in 1/s; valid grids R/dx integer):

| dx (mm) | xi = 0.25 | xi = 0.45 | xi = 0.65 |
|---:|---:|---:|---:|
| 0.5 | -0.97 | -1.03 | -1.06 |
| 0.333 / 0.286 | | -0.7 / -1.1 | |
| 0.25 | +0.33 | **+2.0** | -1.5 |
| 0.20 | **+1.3** (U1 5.0e-3) | -0.4 | -0.3 |
| 0.167 | | **+3.3** (U1 9.7e-3) | |
| 0.125 | -1.7 | -1.8 | -1.3 (xi 0.05: -0.24, U1 8e-4; xi 0.85: -2.4) |

The unstable window appears on several grids, moves in xi with dx, and the finest grid tested so
far grows fastest. This is the "migrating / multiple phase windows" outcome: the pinned
reconstruct_ghost wall is NOT grid-phase robust. (For theta = 60 alone the sweep grids 0.5 / 0.125
looked stable and 0.25 had one island; that view was too narrow.)

#### Classification and recommendation

V4b-P = **FAIL** for the current pinned formulation as a general wall treatment. It is not
"PASS WITH KNOWN ANOMALY": the anomaly is not a single intermediate-resolution island, its root
cause is not identified, and no production guard can avoid it (the unstable (dx, xi) set moves with
the local wall slope, which rotation changes). The integrated gravity + capillarity + rotation test
and the stirrer forcing remain BLOCKED.

What the evidence already excludes: sample/stencil/classification switching, reconstruction
conditioning (cond = 147 fixed), dt, extension width, the IC artifact (seed only), the wall-column
curvature value alone (B), the ghost geometry alone (C).
What it points to: a weakly anti-damped (O(1/s)) coupling between the pin, which is enforced ONLY
through the explicit wall-column curvature (a stiff capillary spring whose discrete stiffness depends
on the cut-cell geometry), and the low axisymmetric sloshing modes. The level-set kinematics do not
know about the pin.

Next architectural investigation (not started; each is a design decision for the user):

1. Discrete energy budget of the pinned equilibrium (kinetic + gravity + surface energy per step) to
   locate the term that does net positive work. It is cheap and decides between 2 and 3.
2. Enforce the pin kinematically in the transport (zero contact-point velocity in the wall-column
   extension/advection) instead of only through curvature.
3. Wall curvature from height functions (the standard contact-line treatment in VOF/LS codes)
   and/or a semi-implicit surface-tension step, which damps capillary modes.
4. A linear eigenvalue analysis of the frozen discrete operator about the BVP equilibrium, to map
   the unstable (dx, xi, theta) set directly instead of by time marching.

Independently, the side-view contact-line experiment (sec. 9.8) decides whether pinned is the right
physics at all.

#### Mechanism pass (2026-09-25, second session): which discrete coupling grows?

Scripts (DIAGNOSTIC ONLY): `scripts/diag_pinned_energy.py` (frozen cases F0-F3, I<n>, W4, SA;
energy budget; per-step modal samples), `scripts/diag_pinned_jacobian.py` (finite-difference
Jacobian of one fixed-dt step about the BVP state, ~1 min per case at 0.25 mm), analysis in
`scripts/analyze_pinned_{energy,mode,jacobian}.py`. Results in `results/validation_pinned_phase/
{energy,jacobian}/`. All cases start from the wall-compatible IC, reinit OFF, theta = 60,
dx = 0.25 mm, xi = 0.25 unless noted.

**Corrections to the previous session.** (1) The growing mode is at **157 Hz**, not ~42 Hz: per-step
sampling (7.6 kHz, 0.4 s, 2.5 Hz resolution) gives 157.4 +- 1.3 Hz in eta(axis) and eta(wall band);
the 100 Hz log had aliased it (157 - 100 = 57 -> 43 Hz). It is a SHORT capillary wave (flat
capillary-gravity scale kR ~ 19, wavelength ~ 2.6 mm ~ 10 dx), not a low-order sloshing mode.
(2) "Negative damping from the pin's capillary spring" is REJECTED as stated: the wall column's own
curvature carries almost none of the growth (below).

**Frozen / unfrozen matrix (time domain):**

| case | phi evolves | kappa evolves | U_end | growth | verdict |
|---|---|---|---:|---:|---|
| F0 | no | no (reference kappa(r)) | 3.5e-3 | saturating (steady forced flow) | stable |
| F1 | no | no (numerical, t = 0) | 2.65e-2 | flat after 0.3 s | stable |
| F2 | yes | no (reference kappa(r) at every crossing) | 1.3e-5 | -2.3 /s | stable |
| F3 | yes | yes (production) | 1.6e-3 | +1.0..+1.3 /s | UNSTABLE |
| I4 | yes | yes except reference in the 4 wall columns | 7e-7 | -1.8 /s | stable |
| W4 | yes | only in the 4 wall columns | 4e-6 | -0.8 /s | stable (cannot host the mode) |

With phi frozen the flow through the fixed interface saturates at the level set by the static
p_Gamma error. There is no operator instability in momentum/projection/wall BC. Growth needs the
curvature-shape feedback.

**Energy budget.** K (face control volume x sub-cell liquid fraction on liquid-known faces),
Eg, Es, V from the interface graph, F = Eg + Es - P0 V, D_mu from the solver's own viscous
operator, W_gamma = -sum p_Gamma x outward mixed-face flux. Sign check (discretely divergence-free
stream-function field on the real meniscus): W_gamma = -2.92e-7 W vs sigma dA/dt = +2.88e-7 W
(1.6 %; 1.2 % with reference kappa), W_g vs dEg/dt within 2 %. So R_cap = dEs + dt W_gamma
has the right sign. RESULT: INCONCLUSIVE. The budget is a cancellation of ~1e-9 W first-order terms
(R_grav vs R_vol vs R_cap) down to residuals of ~1e-12 W, while the injection needed is ~1e-11 W.
The regional R_cap shows the same "wall +, mid -" pattern in stable and unstable cases (a
region-assignment artifact) and is not used as evidence. (A first attempt that weighted void-void
faces was wrong -- those faces still carry the -g dt predictor increment -- and is archived in
`energy/superseded_voidweights/`.)

**Linear stability of the discrete step (Jacobian, eigenvalues mu, lambda = ln|mu|/dt):**

| wall | theta | xi | max lambda (1/s) | frequency | time-domain lambda |
|---|---:|---:|---:|---:|---:|
| pinned | 60 | 0.25 | **+1.35** | **156.8 Hz** (axis 79 %) | +1.25..+1.35, 157 Hz |
| pinned | 60 | 0.35 | stable | | decays |
| pinned | 50 | 0.45 | **+3.24** | 183.5 Hz | +2.0 |
| pinned | 50 | 0.25 | +0.59 | 22 Hz | +0.3 |
| pinned | 50 | 0.65 | stable | | -1.5 |
| pinned | 70 / 80 / 90 | 0.25 | stable | | stable |
| pinned, F2 (no shape feedback) | 60 | 0.25 / 0.35 | stable | | stable |
| pinned, I4 | 60 | 0.25 | stable | | stable |
| pinned, I1 / I2 / I3 | 60 | 0.25 | +0.60 / +0.23 / +5.1 | 133 / 143 / 151 Hz | |
| **static_angle** (same geometry, IC) | 60 | 0.15 / 0.25 / 0.35 / 0.65 | **+2.8 / +2.7** / +0.26 / stable | 114 / 148 / 17 Hz | |
| **static_angle** | 50 | 0.45 | **+4.2** | 141 Hz | |

The Jacobian reproduces every time-domain sign and the frequency. It is a fast (minutes)
phase-marginalized stability test and should gate any future wall/curvature change.

First-order eigenvalue sensitivity of the unstable mode to the curvature feedback
dJ = J_F3 - J_F2: wall column (input) -0.47 /s, column N-2 -7.5, N-3 -3.6, N-4 +28, N-5..N-8 +26,
mid -24, axis -13 (sum +5.1). lambda is a small net of large opposing contributions from the SLOPED
near-wall band, which is consistent with a phase-sensitive sign. The feedback is not a small
perturbation (|dJ|/|J| ~ 1), so removing blocks of it (or mixing exact and numerical kappa by
column, I1-I3, K2/K123) breaks the capillary operator. Those results are not used for localization.

**Best-supported mechanism (confidence labels):**
- CONFIRMED: the anomaly is a linear instability of the discrete coupled step (unstable eigenpair
  matching the time-domain growth and the resolved 157 Hz frequency).
- CONFIRMED (this case): it needs the curvature-shape feedback. Momentum, projection and the wall BC
  alone (F0, F1) and the kinematics with shape-independent p_Gamma (F2) are stable.
- SUPPORTED: it is NOT specific to the pinned ghost/reconstruction. The static_angle wall on the same
  geometry has the same family of axis-peaked unstable capillary modes, phase-dependent, even
  stronger. Exact ghost (C) and exact wall-column curvature (B, sensitivity -0.47/s) do not
  remove it.
- SUPPORTED: it gets stronger with the wall slope (theta 50 > 60 > 70-90 stable) and its sign flips
  with the sub-cell phase.
- PLAUSIBLE: the phase-dependent sign comes from the cut-cell capillary coupling (level-set curvature
  at cell centres interpolated to the crossings, ghost-fluid pressure) on the SLOPED near-wall part of
  the meniscus (~8 columns), acting on a marginally resolved short capillary wave (~10 dx).
- REJECTED: time-level coupling (dt-independent) -> semi-implicit surface tension is not the first
  fix; energy injection by the projection (F0/F1 stable); the pinned ghost as root cause.
- NOT RESOLVED: the energy-budget location of the injection.

Decision-tree outcome: closest to **D** (energy residual unresolved, linear mode grows) plus the
F-matrix part of **A**. The linear analysis has been done.

Recommended next pass (not started): a geometric (height-function / local graph) curvature for the
near-wall SLOPED band, not only the wall column. That is B4's effect, which the Jacobian confirms is
stable, but it needs a consistent definition instead of injected BVP values. It should apply to both
wall models, and be judged by a Jacobian eigen scan over theta x xi x dx (no unstable eigenvalue)
before any long run. A height function only in the wall column is not expected to help
(sensitivity -0.47/s).

#### Wall-near geometric curvature redesign (third session): NOT a fix yet

Name of the defect from now on: **phase-dependent linear instability in wall-near curvature-shape
feedback**, not "pinned-contact-line instability" (the static_angle wall shows the same modes).

Code (opt-in, default unchanged): `src/air_vortex/wall_curvature.py`, selected by
`wall.wall_curvature = "graph"` (`wall_curvature_{band,order,fit,window,blend}`). With the default
`"level_set"` every path is bitwise the old one (legacy 21/21, sigma = 0 15/15, V4 12/12).
- Column heights z_i come from the phi = 0 root of a local cubic along each column (O(dx^4)).
- kappa_m and kappa_theta come from a constrained polynomial in x = (r - r*)/dx, evaluated at the
  SAME crossing coordinates the GFM pressure BC uses. There is no cell-centre -> crossing
  interpolation.
- The wall constraint is a single row in a shared algorithm: pinned eta(R) = z_pin, static_angle
  eta'(R) = cot(theta).
- band >= Nr covers the whole radius (heights mirrored at the axis).
Scripts: `diag_wall_curvature_geometry.py` (TABLE A), `diag_pinned_jacobian.py` (now with ε scaling,
pre-relaxation and the least-damped oscillatory mode), `analyze_wall_curvature_{scan,map,td}.py`.
Results: `results/validation_wall_curvature/`.

**Jacobian gate validated (ε x0.01 .. x100):** theta 60 / xi 0.25: mu = 0.99182 +- 0.12899j,
lambda = +1.346..+1.353 /s, 156.78-156.79 Hz, eigenvector correlation 1.00000; theta 50 / xi 0.45:
+3.19..+3.25 /s, 183.53 Hz, correlation 1.00000. Tolerance used: lambda > 0.02 /s counts as
unstable. Real near-neutral modes of stable baselines sit at +0.002..+0.008.
**Caveat found:** linearizing about the BVP state (t = 0) instead of the relaxed discrete equilibrium
produces spurious modes (graph candidate: +10.7 /s at t = 0 -> +0.016 after 0.3-1 s relaxation;
+0.022 -> +0.002). For weakly damped modes (|lambda| < ~0.4 /s) of the graph operator, the sign even
depends on WHEN in the residual sloshing the state is frozen: 21 Hz mode +0.28 / +0.38 / -0.07 after
0.3 / 0.5 / 1.0 s. That operator is piecewise smooth (root-window and nearest-column switches).
The Jacobian is a reliable gate for strong instabilities only. Weak modes need the time domain.

**TABLE A (geometry, dx = 0.25 mm, 10 phases; worst over crossings in the last 8 columns;
total kappa error median / worst in 1/m, phase std).**

| method | theta 50 | theta 60 | theta 70 | theta 80 | theta 90 | cap 60 (g = 0) |
|---|---|---|---|---|---|---|
| LS (current div n) | 28.5 / 44.5, std 8.5 | 11.7 / 19.2, std 3.4 | 3.4 / 5.9 | 0.16 / 0.56 | 0.01 / 0.05 | 5.6 / 8.6 |
| graph, cubic 4-pt interpolating | **2.7 / 3.1, std 0.40** | **0.80 / 1.03, std 0.17** | 0.26 / 2.7 | 0.09 / 0.57 | 0.01 / 0.05 | **0.03 / 0.25** |
| graph, quadratic 3-pt (height function) | 15.9 / 39.6, std 10 | 8.1 / 19.7, std 4.6 | 2.7 / 8.1 | 0.79 / 1.6 | 0.01 / 0.05 | 0.60 / 1.3 |
| graph, quadratic 5-pt least squares | 48.8, std 0.01 | 26.9, std 0 | 13.6 | 5.6 | 0.00 | 2.1 |
| graph cubic, static-angle constraint | 2.9 / 3.9 | 0.91 / 1.26 | 0.28 / 2.7 | 0.09 / 0.56 | 0.01 / 0.05 | 0.03 / 0.16 |

kappa_theta errors <= 0.11 1/m for every graph variant (no hidden cancellation). The cubic graph is
10-40x more accurate than LS on steep menisci and essentially phase-independent.

**Lesson 1 (restoring property).** Least-squares windows (Savitzky-Golay) are anti-restoring at the
grid scale (5 points: sawtooth response +4/7 instead of -4). Every LSQ candidate had lambda up to
+1400 /s. The stencils must interpolate (window = order + 1). This is encoded in
`tests/test_wall_curvature.py`.

**TABLE B (Jacobian at t = 0, dx = 0.25 mm; canonical 7 cases, or all 5 theta x 10 xi = 50).**

| method | cases | worst lambda | frequency | unstable | least-damped oscillatory |
|---|---:|---:|---:|---:|---|
| LS (production) | 7 canonical | +3.24 (th50 xi.45) | 184 Hz | 2 | +3.24 |
| graph cubic band 6 hard | 50 | +109 (th50 xi.85) | 0 (band-edge column) | 8 | +0.92 (21 Hz) |
| graph cubic band 6 blend 2 | 50 | +15.9 (th50 xi.85) | 0 (band edge) | 8 | +3.42 (90 Hz) |
| graph cubic full radius | 50 | +10.7 (th50 xi.45; +0.016 after relaxation) | 0 | 9 | +2.45 (20 Hz) |
| graph cubic band 6, static_angle | 50 | +109 | 0 (band edge) | 8 | +1.31 (142 Hz) |
| graph cubic band 4 / 8 hard | 7 | +76 / +80 | 0 (band edge) | 4 / 2 | |
| graph quadratic 3-pt, any band | 7 | +17..+23 | 0-19 Hz | 4 | |
| graph quadratic 5-pt LSQ | 7 | +583 | 0 | 7 | |

**Lesson 2 (seam).** Any band edge between graph and level-set curvature creates a localized real
mode AT the edge column. A 2-column linear blend does not remove it. (Outcome C.)

**Time domain (full-radius cubic graph, reinit OFF, 2 s, dx = 0.25 mm):**

| case | old LS lambda | graph: U(1 s) | U(end) | lambda_K/2 (t > 1 s) | RMSE/dx | result |
|---|---:|---:|---:|---:|---:|---|
| theta 60, xi 0.25 (historical bad) | +1.35 | 1.2e-5 | 1.2e-5 | -0.59 | 0.004 | **fixed** |
| theta 50, xi 0.45 (historical bad) | +2.0..+3.2 | 1.3e-4 | 1.1e-5 | -1.22 | 0.012 | fixed (RMSE marginal) |
| theta 60, all 10 xi | only 0.225-0.25 unstable | <= 1.1e-4 | <= 4e-5 | -0.44..-0.78 | 0.004-0.014 | all decay; 5/10 RMSE > 0.01 |
| theta 50, all 10 xi | 0.25 / 0.45 unstable | up to 6.3e-3 (bursts) | up to 1.8e-3 | xi 0.95: **+0.87** | 0.036-0.20 | **9/10 FAIL** |

static_angle wall with the same graph curvature (constraint eta'(R) = cot theta), time domain 2 s:
theta 60 xi 0.25 decays (lambda_K -1.0, RMSE 0.004), theta 50 xi 0.45 decays (RMSE 0.012),
theta 50 xi 0.05 grows (+0.56 /s, RMSE 0.16) -- the same pattern as pinned. The remaining failures
are not specific to the wall BC (not Outcome D).

The graph operator in height space is only mildly non-symmetric (5-6 %, at the one-sided wall
columns), positive definite, and identical for stable and unstable phases. So the remaining phase
dependence enters through the crossing <-> flow mapping (GFM pressure placement with its theta
fractions, pressure applied at vertical AND radial crossings, level-set kinematics), not through the
curvature operator.

**Outcome: B (primary) with C (banded variants).** Curvature accuracy and co-location alone do not
make the coupled free-surface operator stable. They cure the original high-frequency capillary mode
in the two historical failures but leave or create other phase windows on steeper menisci, and every
band edge adds a seam mode. V4b-P remains FAIL. `wall_curvature = "graph"` stays opt-in and
experimental.

**Next pass (recommended):** make the capillary coupling ENERGY-/ADJOINT-CONSISTENT. The discrete
work of p_Gamma through the mixed faces should equal the change of a discrete surface energy under
the same interface motion. Concretely: a variational curvature (kappa_i = dA/dh_i / (2 pi r_i dr),
the Hessian of the polyline area, symmetric by construction) applied with a pressure placement
whose flux weights are the transpose of the height update (vertical AND radial crossings of a column
consistent), checked first with the energy budget (R_cap) and the relaxed-state Jacobian.
Semi-implicit surface tension and projection rewrites remain unjustified (dt-independent;
F0/F1 stable).

#### Energy/adjoint-consistency diagnosis of the capillary coupling (fourth session): hypothesis NOT sufficient

Script: `scripts/diag_adjoint_capillary.py` (DIAGNOSTIC ONLY; tables via
`scripts/analyze_adjoint.py`, results `results/validation_adjoint/`). No production code changed.
Everything is linearized about a relaxed state (production path F3 for 1 s or 0.3 s, or the stable
F2 path for 1 s), dx = 0.25 mm, from the wall-compatible IC.
- Interface DOFs: column heights h (cubic-root crossings), h_wall = z_pin fixed.
- E = sigma A(h), A = exact area of the piecewise-linear surface of revolution through
  (0, h_0), (r_i, h_i), (R, z_pin); analytic g, H.
- Measured kinematics K = S A_u / dt: S = dh/dphi; A_u = d phi+/d u_new through the solver's own
  extension and level-set advection.
- M = rho x face control volumes; P = the solver's free-surface GFM projection.
- C_cur = (J_F3 - J_F2)[u, phi]; C_EC = dt P (-M^-1 K^T H S).
- Block replacement: J_F2 + [[A_u C], [C]]. With C_cur this reproduces J_F3 on clean bases
  (lambda 1.179 vs 1.180; decomposition error <= 0.6 %).

Established:
- A(h): gradient error vs FD 2-5e-6 (eps 1e-7, then round-off), Hessian 3e-9. The variational
  curvature dA/dh_i / (2 pi r_i dr) has the sign of the verified convention (100 % of columns;
  error 1.3-3.3 1/m in the interior, 60-90 1/m at the half-cell end segment next to the pin).
- K (actual level-set kinematics) differs from the continuum graph kinematics eta_t = u_z - u_r eta_r
  by 27-33 % (29-37 % in the last 8 columns). It differs from the FLUX kinematics
  K_Q = (column mixed-face outward flux) / (2 pi r_i dr), which a sharp GFM pressure does work
  through, by 64-70 % (67-90 % in the last 8 columns).
- Projection: P^2 = P to 6e-16. P is NOT symmetric in the plain face-volume metric (14-16 %) but
  exactly symmetric (4e-16) in M_theta = M with mixed faces x their GFM fraction theta. M_theta is
  the solver's energy metric; no projection issue (Outcome D not triggered).
- Work identity (5 random discretely divergence-free fields): current sharp p_Gamma work vs
  -g^T K u mismatch 2.7-6.0 %; column-owned pressure g_i/(2 pi r_i dr) 1.5-2.9 %; EC exact
  (1e-16, by construction).
- Adjoint defect ||M(C_cur - C_EC)|| / max = 0.42-0.52 in EVERY case (bad theta60 xi.25 0.42,
  good theta60 xi.35 0.42, good theta70 0.45). It does NOT correlate with the unstable phases.

Decision table (lambda_max 1/s (frequency); least-damped oscillatory):

| case (base) | current | EC, measured K, M_theta | EC, flux K_Q | EC, measured K, plain M |
|---|---|---|---|---|
| th60 xi.25 (F3 1 s) | +0.87 (157 Hz) | +3.03 (0 Hz); osc -3.72 | +0.064; osc -0.31 | +5.69 |
| th60 xi.35 (F3 1 s, good) | +0.008 | +2.48 (0 Hz); osc -3.64 | +0.039 | +4.63 |
| th50 xi.45 (F3 0.3 s) | +4.67 (184 Hz) | +2.10 (0 Hz); osc -1.20 | +0.34 (0) | +4.86 |
| th50 xi.25 (F3 0.3 s) | 0.000 | +2.67 (0 Hz); osc -3.62 | **+5.58 (365 Hz)** | +4.36 |
| th50 xi.65 (F3 1 s, good) | 0.000 | +2.23 (0 Hz) | +0.002 | +4.62 |
| th70 xi.25 (F3 1 s, good) | -0.001 | +2.75 (0 Hz); osc -3.05 | -0.001 | +5.54 |

Reading:
- The energy-consistent capillary block built on the ACTUAL kinematics damps every oscillatory
  capillary mode (-1.2..-3.7 /s). But it adds a real divergent mode (+1.9..+3.0 /s) in EVERY case,
  including phases that are stable today.
- On the flux kinematics it is near-neutral in most cases but unstable at theta 50 xi 0.25 (365 Hz).
- Base-state sensitivity: the current operator's lambda at theta 50 xi 0.25 is 0.000 / +1.77 /
  +21.9 about three different relaxed states, and on F2-relaxed bases the block decomposition no
  longer reproduces J_F3. Linearization of this piecewise-smooth scheme is trustworthy only for
  strong modes on clean bases.

**Outcome C: a capillary force that is non-adjoint to the kinematics is NOT a sufficient
explanation.** Swapping only the capillary block for its energetic adjoint exchanges one
instability for another. As the decision logic requires, no nonlinear "variational_adjoint"
prototype was built.

The most specific structural finding is the 64-90 % mismatch between the interface motion the
level set actually produces (K) and the face-flux kinematics through which every sharp pressure
boundary term does work (K_Q). That concerns the hydrostatic/gravity restoring as much as
capillarity. An energy-consistent capillary force paired with K then fights a gravity coupling paired
with K_Q (real divergent modes). One paired with K_Q inherits K_Q's mismatch with the actual motion.
Candidate next investigation, if the user chooses: make the free-surface kinematics and the
pressure-BC work use one operator (e.g. interface motion driven by the same mixed-face fluxes,
or a pressure BC derived from the level-set kinematics), and check it on the WHOLE free-surface
coupling (gravity + capillarity) with the energy budget first. This is an architectural change of
the GFM/level-set coupling, i.e. a user decision.

#### Kinematic-consistency test K_LS vs K_Q and the off-contour mechanism (fifth session, unattended)

Scripts (DIAGNOSTIC ONLY): `diag_kinematic_consistency.py` (reduced height map: h authoritative,
phi = exact signed distance of a spline graph through h each step, solver step unchanged, h+ by
level-set increment or by mixed-face flux K_Q; paired and total-energy Jacobians; nonlinear reduced
prototype), `diag_kinematics_manufactured.py`, `diag_offcontour_mode.py`. Results:
`results/validation_kinematics/`. Recovery log: `results/autonomous_2026-09-25/status.md`.
Invalid runs are kept in `superseded_pointsampled_sdf/`: a point-sampled SDF gave a round-trip gain
of 1.106 per step, which alone is a spurious +75 /s mode. The exact point-to-segment SDF round-trips
to 3.6e-6 dx.

1. **Paired Jacobians (reduced map, dx 0.25 mm, relaxed states):** the historical cases (theta60
   xi.25, theta50 xi.45) are STABLE with EITHER level-set or K_Q kinematics once phi is rebuilt each
   step as an exact signed distance of its contour. theta50 xi.05 is unstable with both (LS +6.2 real;
   K_Q +16 @ 473 Hz). Pairing gravity + capillarity to ONE K (total-energy EC, H_p = H_g + H_sigma,
   E_g FD-verified to 1e-11) is stable in all 6 cases for EITHER K. The void-side ownership variant
   of K_Q fails (+62 @ 613 Hz; Nyquist blow-ups when paired). **The K_LS vs K_Q choice is not what
   separates stable from unstable.**
2. **Manufactured kinematics (no pressure, no sigma; paraboloid, divergence-free field, 5 phases):**
   - K_LS error 6.0 / 2.9 / 1.3 % at theta50 (dx 0.5 / 0.25 / 0.125), first order. It is dominated
     by the VELOCITY EXTENSION (exact field everywhere: 0.1-0.9 %). MUSCL vs upwind, the capillary dt
     and the crossing extraction change it by < 0.5 %.
   - K_Q error 21 % at theta50 on EVERY grid (17 % at theta60, 11 % at theta75, converges only for a
     flat surface). Assigning radial mixed-face fluxes to columns is not a consistent discretization
     of eta_t, whichever side owns them. ||K_LS - K_Q|| does not decrease with refinement because of
     K_Q.
3. **Off-contour modal test (production Jacobian about the t = 0 exact-SDF state):** the unstable
   eigenvector is 65 % (theta60 xi.25) / 83 % (theta50 xi.45) OFF-CONTOUR: phi perturbations that
   change the curvature stencil without moving phi = 0. The same Jacobian with phi slaved each step to
   the exact SDF of its contour (D J D, D = diag(R S, I)) is STABLE: +1.347 -> -0.54 @ 22 Hz,
   +3.240 -> -0.60 @ 21 Hz, good phase still stable.
   Generality scan (25 cases: theta 50/60 x 10 xi, theta 70 x 5; EXACT oblique projection
   R (S R)^-1 S, idempotent to 3e-16; `offc_scan/summary_exact_projection.md`):
   - slaving removes EVERY oscillatory capillary instability (historical 157 / 184 / 185 / 66 / 93 Hz;
     the least-damped oscillatory mode becomes the viscous 21-22 Hz sloshing mode at -0.4..-0.6 /s);
   - but it EXPOSES static (0 Hz) divergent modes at 7/25 phases: +3.5..+10 /s at theta50
     xi .05/.15/.25/.95 and +0.3..+1.2 /s at theta60 xi .05/.85/.95, i.e. where the pin sits near a
     cell face. All 7 have the same height shape: single-signed, axis-peaked, ~0 at the pinned wall
     (`offc_loc/`). That is a volume-changing shape mode, which divergence-free flow cannot drive, so
     it points at the non-conservative part of the interface kinematics near the pin. Production
     shows the same family only weakly at those phases (+0.04..+0.17, real).
   **CONFIRMED (linear, SDF state): the OSCILLATORY capillary instability family lives in phi's
   off-contour degrees of freedom. A second, STATIC volume-shape divergence (pin near a cell face)
   is independent of it and remains when the off-contour DOFs are removed.**
   Caveats: slaving about a 1 s-evolved (non-SDF) base is inconsistent (unstable). Slaving only a
   region (wall 4 / 8 / 12 columns, interior) creates seam modes (+74..+117 /s), so the mechanism
   cannot be localized to a wall band.
4. **No existing tool realizes the slaving nonlinearly:**
   - RS2 reinitialization every 1 / 5 / 20 steps: U 1e-3..2e-2 m/s, RMSE 0.17-0.8 dx at 2 s, even at
     the good phase. The one-step R1 Jacobian is +145 / +1571 /s.
   - The nonlinear reduced prototype: LS increment 1.4e-3..3e-2 m/s, secular axis drift, energy
     rising. K_Q locks into a sustained ~8e-3 m/s through-interface current with flat total energy.
   - Diagnostic two-sided normal extension of the transport velocity (X2): cures theta50 xi.45 but
     worsens theta60 xi.25 (+3.48 @ 160 Hz).

**Verdict: the kinematic (K_LS vs K_Q) hypothesis is REJECTED as root cause, and the K_Q transport
route is closed** (stop rules 1-2 of the unattended plan: no flux-consistent transport implemented).
The best-supported picture is now TWO mechanisms:
(i) **oscillatory:** the off-contour structure of phi (the level sets next to the contour, which the
level-set curvature stencil reads) evolves freely without reinitialization and couples into the
capillary pressure. Keeping phi an exact signed distance of its contour removes it linearly at every
tested phase.
(ii) **static:** a volume-changing, axis-peaked shape mode diverges when the pin sits near a cell face
(xi ~ 0.85-0.25). It is exposed once (i) is removed, and consistent with non-conservative interface
kinematics near the pin.
No existing wall-compatible tool realizes (i) nonlinearly, and (ii) is uncharacterized beyond its shape.
Recommended next design question (user decision): a contour-preserving, wall-compatible signed-distance
maintenance (or a curvature that does not read off-contour phi at all, together with pressure/kinematic
consistency). Any candidate is to be gated by the global slaving Jacobian test and time-domain runs.

#### Pin-consistency contract (tests)

`test_pinned_static_meniscus_holds_contact_point` now states the contract: z_pin equals the initial
interface's own wall height (wall-trace crossing within 0.005 dx of z_pin BEFORE time integration,
from the wall-compatible IC), and the meniscus then holds (umax 1.4e-3 < 5e-3 m/s in 0.1 s).
`test_pinned_meniscus_is_sensitive_to_a_small_pin_mismatch` keeps the known sensitivity: the
automatic capture `initial_pin_height` is biased by -0.0133 dx on a curved meniscus, and a 0.013 dx
mismatch drives ~10x the flow (1.5e-2 m/s). Production starts flat, where the capture is exact.

#### Finding 4 -- pre-existing failing unit test (resolved by the contract above)

`tests/test_contact_angle.py::test_pinned_static_meniscus_holds_contact_point` fails in the working
tree left by the previous session (Umax 2.3e-2 > 5e-3 within 0.1 s). Cause: with
`pinned_method = reconstruct_ghost` (the default since 2026-09-24 17:09) the auto-captured pin
`initial_pin_height` is 0.013 dx below the BVP wall height (legacy capture: 0.002 dx), so the IC is
not an equilibrium of the pinned problem; the compatible IC does not cure it (1.4-1.5e-2). A
0.013 dx pin mismatch drives 2e-2 m/s: the pinned model is very sensitive to z_pin consistency.
The test was not edited.

### 9.11 Checkpoint

The current state is a natural checkpoint: single-phase architecture, RS2 reinit, sharp surface
tension, V4b geometry, legacy bitwise regression. Suggested tag: `milestone-v4-single-phase-capillary`.
Not committed (awaiting user permission).

## 10. Height-function free surface (opt-in research branch `single_phase_height`)

Code (additive; `single_phase_ls` and all defaults unchanged): `src/air_vortex/height_interface.py`
(graph reconstruction, geometry, curvature, conservative fluxes), `height_solver.py`
(`SinglePhaseHeightSolver`), `height_benchmarks.py` (meniscus, flat, rotating builders and the
rotating Young-Laplace reference). Validation: `scripts/validate_height_branch.py`,
`scripts/diag_height_jacobian.py`, `scripts/analyze_height_sweep.py`; results in
`results/validation_height/` (recovery log `status.md`). Tests: `tests/test_height_branch.py`.

**Why.** Sec. 9.10 found two freedoms behind the Level-Set wall instability:
(i) off-contour phi DOFs (oscillatory modes) and (ii) non-conservative zero-contour motion (static
volume-changing mode). The height branch removes both by construction.

**Formulation.**
- The interface is z = eta(r), one height eta_i per radial cell centre. It is authoritative. There is
  no evolved level set; `fields.phi = z - eta` is derived plotting/compat data and never fed back
  (tested).
- Reconstruction: a piecewise cubic whose interval stencils are FIXED 4 nodes. The nodes are the
  axis-mirrored columns (even: eta_r(0) = 0), eta_i, and the pinned wall point (R, z_pin), with z_pin
  continuous and never snapped. No least-squares window (Savitzky-Golay windows are anti-restoring,
  sec. 9.10).
- Motion: d(A_i eta_i)/dt = F_{i-1/2} - F_{i+1/2}, F = 2 pi r_f int_0^eta u_r dz with the staggered
  u_r and the top partial cell weighted by its liquid fraction. Axis and wall fluxes are 0, so
  sum_i A_i eta_i is conserved ALGEBRAICALLY (no volume correction). SSPRK2 with the projected
  velocity.
- Geometry for the unchanged ghost-fluid pressure/projection comes straight from eta:
  - liquid iff z_c < eta_i;
  - vertical crossings at eta_i;
  - radial crossings WELL-BALANCED: linear interpolant of the two adjacent heights, carrying the
    linear interpolation of the two node curvatures. A node equilibrium
    sigma kappa_i + rho g eta_i = P0 then balances every crossing exactly. With cubic radial crossings
    the pressure BC is overdetermined (more crossings than height DOFs): at theta60 xi.25 that gave a
    steady 4.8e-3 m/s current, against 3.9e-5 and decaying with the well-balanced form.
- Curvature at column nodes: mean of the two adjacent interval cubics,
  kappa_m = -eta''/(1+eta'^2)^1.5, kappa_theta = -eta'/(r sqrt(1+eta'^2)), p_Gamma = sigma kappa.
- Velocity extension (only for momentum stencils) uses the graph normal of the derived psi.

**Scope (declared).** Single-valued graphs only: flat surface, vortex depression, first air-core
contact. No overturning, pinch-off or multi-valued interfaces. `height_interface.graph_validity`
reports max |eta_r| and min N_m, N_theta.

**Gates.**
- H1 geometry: node kappa error theta60 1.27 / 0.48 1/m (0.5 / 0.25 mm; LS div n: 11.7 median /
  19.2 worst at 0.25 mm), theta50 3.2 / 1.4. kappa_theta <= 0.03. Paraboloid exact, cap 0.012 1/m.
  Wall-slope error 2e-4 (theta60, 0.25). Radial-crossing kappa phase std 0.07 1/m (theta60).
- H2 conservative transport (prescribed divergence-free MAC field): rate error theta50 0.79 / 0.21 %,
  theta60 0.65 / 0.17 / 0.05 % (dx 0.5 / 0.25 / 0.125) -> second order (Level Set: 6.0 / 2.9 / 1.3 %,
  first order). Finite-time interface error / displacement 1.5 -> 0.4 %. Volume change <= 1.5e-16.
- H3 hydrostatic (flat off-grid, sigma = 0 and > 0): round-off (U 1e-15..1e-17, dV <= 3e-16).
- H4 rigid body sigma = 0, Omega = 20, matched rotating wall, 2 s: U 1.5e-15, eta NRMSE 2e-15
  (LS V3: 6e-4).
- Jacobian gate, 50 cases (theta 50-90 x 10 xi, 0.25 mm): state = eta + u only (phi dimension 0).
  lambda_max within [-0.0021, +0.0018] (FD noise). The top mode is the NEUTRAL conserved-volume
  direction (rel. volume 1.0, lambda = 0: a family of equilibria parametrized by volume), not a growing
  mode. The least-damped oscillatory mode is the viscous 21-23 Hz sloshing mode (-0.24..-0.53 /s).
  The Level-Set 157 / 184 Hz modes and the growing volume mode do not exist.
- H5 static pinned meniscus sweep, dx 0.25 mm, 10 phases each (theta 50/60/70 to 3 s, theta 80/90 to
  1.5 s by compute budget), NO volume correction, NO reinitialization:
  theta50 worst U 4.2e-5, worst RMSE 0.0013 dx; theta60 1.9e-5 / 0.0005; theta70 8.1e-6 / 0.0003;
  theta80 4.7e-6 / 0.0001; theta90 round-off. 0/50 unstable. Max |dV| 4e-15.
  The Level-Set failure phases (theta60 xi .25; theta50 xi .05 / .15 / .25 / .45 / .95) are all stable.
- Integrated gravity + capillarity + rotation equilibrium (no forcing). Independent reference:
  `height_benchmarks.solve_meniscus_pinned_rotating`, the pinned Young-Laplace BVP with
  P0 + rho Omega^2 r^2/2 - rho g z = sigma kappa, fixed volume, axis symmetry, pinned wall height.
  It reproduces the static P0 exactly at Omega = 0; BVP residual ~3e-11.
  Omega = 25 rad/s: rise 1.67 mm vs 1.38 mm static, max slope 0.38, wall angle 69 deg, no air core,
  N_m = N_theta >= 36 cells even at 0.5 mm. Matched rotating wall and bottom, u_theta = Omega r.

  | dx (mm) | time | meridional U | late lambda_U | eta RMSE | depression error | u_theta - Omega r | volume / pin |
  |---:|---:|---:|---:|---:|---:|---:|---|
  | 0.5 | 3 s | 5-7e-6 | -0.28 / -0.34 | 4-5e-5 dx | <= 9e-5 dx | <= 1.9e-6 | exact |
  | 0.25 | 2.6 s | 0.9-1.3e-6 | -0.27 / -0.58 | 4-6e-5 dx | <= 1.1e-4 dx | <= 2.4e-7 | exact |
  | 0.125 | 1.0 s | 4.4e-7 | -0.16 | 5e-5 dx | 9e-5 dx | 1.5e-7 | exact |

  Stable, no secular growth, and the spurious flow converges with refinement -> integrated free-surface
  physics PASS for the height branch.

- Forcing sanity (QUALITATIVE; forcing.py term ported opt-in; tau_s = 5 ms UNCALIBRATED; baseline
  vessel/bar geometry is a placeholder, not measured; stationary no-slip wall; pinned flat start).
  At dx 1 mm, 2 s, 150 / 300 / 450 rpm:
  - u_theta max 0.22 / 0.41 / 0.63 m/s;
  - p(centre bottom) - hydrostatic -2.3 / -76 / -205 Pa;
  - central depression 0.04 / 4.7 / 16.8 mm;
  - axis u_z min -0.09 / -0.13 / -0.14 m/s (central downward jet);
  - strong meridional circulation; volume exact; graph single-valued (slope <= 2.6), far from the bar.
  So the trends are right (rpm up -> swirl up, centre pressure down, depression up).
  NOT grid-converged: at 300 rpm u_theta and U_mer agree between 0.5 and 1 mm, but the SURFACE response
  does not (0.95 s: 0.014 mm at 0.5 mm vs 0.19 mm at 1 mm). The 1 mm depression is a smooth, broad
  central motion (not an axis-column spike) that oscillates and then deepens. N_m falls to 3-6 cells
  (< 8). The flow is Re ~ 2e4; an axisymmetric laminar solve at 1 mm is under-resolved. This is a
  production-grid convergence (V6/V7) issue, not a free-surface-equilibrium defect. It blocks
  calibration.
- V4b phase acceptance (theta 60, 10 phases per grid, 3 s, no volume correction, no reinit):

  | dx (mm) | median U | worst U | phase std | median RMSE | worst RMSE | max dV | unstable |
  |---:|---:|---:|---:|---:|---:|---:|---:|
  | 0.5 | 7.5e-5 | 9.0e-5 | 1.1e-5 | 0.0010 dx | 0.0013 dx | 3e-15 | 0 |
  | 0.25 | 1.5e-5 | 1.9e-5 | 1.4e-6 | 0.0005 dx | 0.0005 dx | 4e-15 | 0 |
  | 0.125 | 3.2e-6 | 3.5e-6 | 1.6e-7 | 0.0003 dx | 0.0003 dx | 5e-15 | 0 |

  The worst case IMPROVES under refinement (Level Set, same test: 2.0e-4 / 6.3e-3 / 4.8e-4 m/s,
  0.0024 / 0.0143 / 0.0147 dx). Together with 0/50 unstable at 0.25 mm over theta 50-90 and the
  Jacobian gate: **V4b (pinned wall) PASS for the height-function branch.**
- Regressions after this branch: legacy 21/21, single-phase LS sigma = 0 15/15, isolated LS V4 12/12
  bitwise; pytest 235 passed (11 new in tests/test_height_branch.py).
- Open for the branch:
  (a) the forcing surface response is not grid-converged, and there is an abrupt-deepening episode at
      450 rpm on both grids (needs dt and 0.25 mm checks);
  (b) the graph scope ends at overturning or pinch-off;
  (c) V6/V7 production-grid convergence;
  (d) calibration needs measured data.

### 10.x V6/V7 production stirrer forcing: grid and time-step convergence (session 2026-09-26)

Setting: `single_phase_height`, baseline.yaml geometry (R 45 mm, H 50 mm; **numerical-validation
placeholder, not measured**), stationary no-slip walls, pinned flat start, **tau_s = 0.005 s FIXED and
UNCALIBRATED**. Scripts: `validate_forcing_operator.py`, `validate_forcing_convergence.py`
(modes swirl / lid / free), `analyze_forcing_convergence.py`, `analyze_forcing_budget.py`.
Results: `results/validation_forcing/` (see status.md there).

**Table A: forcing operator** (reference = 20x-finer midpoint quadrature)

| dx (mm) | V_chi rel err | T0 rel err | cells across tanh edge | forcing step error |
|---:|---:|---:|---:|---:|
| 1.0 | 3.6e-4 | -2.7e-5 | 2.2 | 1e-18 |
| 0.5 | 9.2e-5 | -3.9e-8 | 4.4 | 1e-18 |
| 0.25 | 2.3e-5 | < 1e-8 | 8.8 | 1e-18 |
| 0.125 | 5.7e-6 | < 1e-8 | 17.6 | 1e-18 |

T0(300 rpm, u = 0) = 3.567e-3 N m. The operator is grid-independent (the torque varies far below 10 %)
-> Outcome B (forcing geometry) excluded.

**Angular-momentum budget** (dL_z/dt = T_stir + T_side + T_bottom, torques integrated every step).
- Swirl-only (frozen meridional flow): closes to < 4e-4.
- Rigid free-slip lid (new diagnostic `rigid_lid`). A bug was found and fixed during this pass:
  the lid faces are FACE_INACTIVE, so `extend_velocity` overwrote the no-penetration u_z at the start
  of every step. The legacy lid runs carry it; for the advective form its effect is small (<= 2 % in
  L_z at 1 mm).
- New opt-in `swirl_advection = "conservative"`: flux form of q = r u_theta with first-order upwind
  face values. The default "advective" form is unchanged, and the bitwise regressions pass.

  | lid 300 rpm, window | 1 mm advective | 1 mm conservative | 0.5 mm advective | 0.5 mm conservative |
  |---|---:|---:|---:|---:|
  | [0, 0.5) s | -0.10 | -0.001 | -0.10 | -0.001 |
  | [0.5, 1) s | -0.12 | +0.001 | -0.31 | +0.000 |
  | [1, 1.5) s | -0.16 | +0.002 | -0.40 | +0.001 |
  | [1.5, 2) s | -0.03 | +0.003 | -0.37 | +0.001 |
  | [2, 3) s | +0.38 | +0.007 | -0.42 | +0.002 (to 2.4 s) |
  | [3, 4) s | +0.30 | +0.011 | (pre-fix run: -0.79) | not run |

  (Budget residual (Delta L - int T dt) / |int T dt|.) The legacy non-conservative upwind swirl
  advection u.grad(u_theta) + u_r u_theta / r creates or destroys 10-80 % of the angular momentum;
  the flux form closes it to <= 1 %.

  Unit test: inviscid, unforced lid with meridional flow. The flux form conserves L_z to 0.0 in
  40 steps; the advective form drifts 4e-5 (`test_rigid_lid_conservative_swirl_conserves_angular_momentum_inviscid`).

**Table B: rigid-lid spin-up, 300 rpm, common times**

| quantity | 1 mm adv | 0.5 mm adv | diff | 1 mm cons | 0.5 mm cons | diff |
|---|---:|---:|---:|---:|---:|---:|
| L_z(1.0 s) | 3.427e-4 | 2.881e-4 | 17 % | 3.649e-4 | 3.668e-4 | 0.5 % |
| L_z(1.5 s) | 5.74e-4 | 4.50e-4 | 24 % | 6.21e-4 | 6.15e-4 | 0.9 % |
| T_stir(1.0 s) | 6.67e-4 | 6.52e-4 | 2 % | 6.34e-4 | 6.36e-4 | 0.3 % |
| L_z(2.0 s) (after the 1 mm arrival) | 7.72e-4 | 6.23e-4 | 24 % | 7.92e-4 | 8.64e-4 | 9 % |
| swirl reaches (r, z) = (5, 25) mm | 1.75 s | 2.35 s | 0.6 s | 1.70 s | 2.10 s | 0.4 s |

With the conservative form the integral quantities converge (<= 1 %). The TIMING of the swirl
transport into the upper/axis region does not; that transport is carried by the meridional
circulation (first-order upwind momentum advection, unresolved Ekman layers).

**Table D: time step** (dx 0.5 mm, dt factors 1 / 0.5 / 0.25; compared with dt 0.25)

| case | t | L_z | T_stir, u_theta max, probe, depression |
|---|---:|---:|---|
| lid 300 | 0.4 s | +6.5e-4 / +2.2e-4 / 0 | agree to 4 digits |
| lid 300 | 0.8 s | -1.7e-4 / -2.1e-4 / 0 | agree to 4 digits |
| free 150 | 0.4 s | +4.7e-4 / +1.6e-4 / 0 | depression 0.0132 / 0.0132 / 0.0131 mm |
| free 150 | 0.75 s | -1.9e-4 / -6.4e-5 / 0 | depression 0.0082 (all) |

Time-step error is <= 7e-4 relative and negligible next to the grid error -> the time step is converged.

**Table C: free-surface response** (N_curv = min(N_m, N_theta); free runs stop at N_curv < 8)

| rpm | dx | swirl reaches (5, 25) mm (> 0.02 m/s) | depression > 0.3 mm | N_curv < 12 | N_curv < 8 (stop) | last state |
|---:|---:|---:|---:|---:|---:|---|
| 150 | 1.0 | 3.40 s | 3.40 s | 3.65 s (min 9.2, recovers to 23) | never | d 1.88 mm at 6 s, N 23, still creeping |
| 150 | 0.5 | 4.25 s | 5.15 s | never (so far) | never | d 0.48 mm at 5.2 s (event under way; run continuing) |
| 300 | 1.0 | not before stop | 1.55 s | 1.25 s | 1.45 s (at d ~ 0; stop 1.60 s) | d 1.57 mm, off-axis slope 0.34 |
| 300 | 0.5 | 2.35 s | 2.45 s | 2.65 s | 2.65 s | d 3.06 mm, N 4 |
| 450 | 1.0 | not before stop | 0.75 s | 0.80 s | 0.85 s (stop 1.15 s) | d 2.81 mm, slope 0.96 |
| 450 | 0.5 | 1.65 s | 1.60 s | 1.75 s | 1.75 s | d 5.01 mm, N 7.4 |
| any | 0.25 | not reached | - | - | - | stopped for budget at 0.25 s (about 0.1 s of physical time per 40 min under contention) |

(Advective, i.e. default, swirl transport in all free runs.)

**Old 300 rpm discrepancy explained.** "1 mm -> 4.7 mm vs 0.5 mm -> 0.09 mm" compared two grids at
the same time before and after a grid-dependent transport event. On both grids the depression stays
at <= 0.2 mm until the swirl reaches the upper axis region; then it deepens within 0.1-0.2 s. That
arrival is 0.8-0.9 s later at 0.5 mm (150 rpm: 3.40 -> 4.25 s). At 1 mm, 300 and 450 rpm lose
curvature resolution (N_curv < 8) BEFORE the event, through an off-axis slope, so those 1 mm events
are unresolved. The
0.5 mm run deepens in the same way at 2.45-2.65 s.

**Table E: 450 rpm sudden deepening.**
- 1 mm: N_curv < 8 from 0.85 s, BEFORE the event (d 0.5 mm, off-axis slope 0.3-0.96). **UNRESOLVED.**
- 0.5 mm: N_curv >= 29 until 1.70 s. The onset (1.60-1.70 s, d 0.3 -> 1.7 mm) is resolved, but it
  collapses to 7.4 within 0.05 s. **Onset RESOLVED; deep state UNRESOLVED.**
- 0.25 mm: **NOT REACHED** (stopped for budget).

The event is real in the sense that it occurs on both grids, after the same trigger (swirl
reaching the upper axis). Its time and depth are not converged.

**Viscous-scale resolution.**
- Ekman layer sqrt(nu/Omega): 0.18 mm at 300 rpm (0.15 mm at 450), i.e. 0.2 / 0.4 / 0.7 cells at
  1 / 0.5 / 0.25 mm.
- First-order upwind numerical viscosity |u| dx / 2: about 2e-4 / 1e-4 / 5e-5 m^2/s, i.e.
  200 / 100 / 50 x nu.

The meridional circulation, and so the swirl transport time, is set by numerics on every tested grid.

**Outcome: C.**
- The forcing converges, the time step converges and the angular-momentum input converges
  (conservative form).
- The swirl/meridional transport timescale does not converge, so the surface-event times and
  depressions are grid-dependent.
- Minimum credible grid for the surface response: none among 1 / 0.5 / 0.25 mm with the present
  momentum advection.

Requirements before V6/V7 can pass:
- conservative (flux-form) swirl advection as the production default (validated here);
- a higher-order, low-dissipation meridional momentum advection;
- wall/bottom boundary layers resolved to >= 3 cells (about 0.05 mm) by a stretched near-wall grid,
  or a wall model;
- then a repeat of Table C.

Calibration of tau_s stays blocked.

### 10.y V7-T: low-dissipation conservative momentum transport (session 2026-09-26)

Setting as §10.x: tau_s = 0.005 s FIXED (uncalibrated); placeholder geometry; rigid-lid and free-surface
runs from `scripts/validate_forcing_convergence.py`. Results: `results/validation_v7t/`. Recovery log:
`results/autonomous_v7_continuation/status.md`.

**New code (opt-in / height-only).**
- `swirl_transport.py`: q = r u_theta flux form on the MAC face mass fluxes. Options: upwind1, or MUSCL2
  with the van Leer limiter (TVD, chosen because the unlimited scheme undershoots, q_min = -4e-3 on the
  Gaussian). Time stepping SSPRK2; MUSCL CFL bound (|u_r|/dr + |u_z|/dz) dt <= 0.45.
  `angular_momentum_viscous` is the flux form of (1/r^2) d/dr(r^3 d(u/r)/dr); its interior stresses
  telescope and its wall flux equals the torque diagnostic.
- `meridional_transport.py`: flux-form MUSCL2 for u_r, u_z with r-weighted CV mass fluxes (discretely
  divergence-free on every staggered CV: a uniform field gives |N| ~ 1e-13).
- height_solver switches `swirl_advection`, `swirl_limiter`, `swirl_viscous`, `meridional_advection`;
  LU cache for frozen geometry (bitwise equal to spsolve, 2x faster).
- Height defaults: swirl conservative_muscl2 + angular_momentum viscous; meridional advective.
  The LS models ignore these fields. "advective" swirl warns unless height_legacy_swirl_for_validation.

**Isolated gates.**
- Manufactured swirl transport (TABLE A, `results/validation_v7t/tableA_manufactured.md`): MUSCL2 L2
  order 1.76 / 1.88 / 1.98 (Gaussian), 2.05 / 2.03 / 2.02 (smooth mode), 2.07 / 2.06 / 2.07 (solid-body
  profile); upwind1 0.8-1.0. Gaussian peak loss at 1 mm: 26 % (upwind1) vs 9 % (MUSCL2). L_z exact
  (1e-16).
- Measured numerical viscosity (1-D sine mode, U = 0.3 m/s, Courant 0.3):
  - upwind1 at lambda 10 mm: 106 / 53 / 26 x nu, matching U dx (1 - C)/2 within 1 %;
  - MUSCL2 at lambda 10 mm: 33 / 6.1 / 1.0 x nu (dx 1 / 0.5 / 0.25), i.e. ~dx^2.5;
  - MUSCL2 at lambda 20 mm: 12 / 1.8 / 0.2 x nu.
- Meridional operator (current advective upwind): first order; energy-based nu_eff 15 / 7.5 / 3.7 /
  1.9 x nu. Full-solver inviscid decay test: KE loss 14 % / 8 % / 4.4 % in 0.2 s. MUSCL2 meridional:
  nu_eff 0.08 / 0.01 x nu, KE loss 1.5 % / 0.7 %.
- Height gates with MUSCL2 swirl (+ angular-momentum viscous): H4 U ~1e-15; integrated rotation
  U 7.1e-6 m/s, RMSE 5.5e-5 dx at 3 s (unchanged). H3 and V4b-H have u_theta = 0 (unaffected).
- Regressions: legacy 21/21, sigma0 15/15, V4 12/12 bitwise; pytest 247 passed.

**Rigid lid 300 rpm (S0 = upwind1 swirl, S1 = MUSCL2 swirl, S1am = S1 + angular-momentum viscous,
S2 = S1 + MUSCL2 meridional).**
- Budget, time-integrated: S1am -1e-5 (1 mm) / -3e-5 (0.5 mm); S0/S1 +1e-3..+2e-3.
- Compact dt check (0.5 mm, dt vs dt/2): S1 <= 1.2e-3 PASS; S2 6 % T_stir and 17 % U_mer at 1 s
  (chaotic sensitivity), so S2 is rejected as the candidate.
- Late window [3, 4] s, 1 vs 0.5 mm (L / T_stir / T_wall / U_mer): S0 4.4 / 8 / 17 / 5 %;
  S1am 2.7 / 2.5 / 8 / 6 %; S2 8.4 / 13 / 31 / 17 %.
- Upper-axis arrival at probe (5, 25) mm, thresholds 5/10/20 % of Omega R_m, 1 -> 0.5 mm gap:
  S0 +0.40/+0.40/+0.35 s, S1 +0.25/+0.15/+0.10 s.
- 0.25 mm (to ~1.1 s): at t = 1 s L = 3.774 / 3.827 / 4.213e-4 (1 / 0.5 / 0.25) and wall torque
  -1.50 / -1.69 / -1.17e-4. The 0.5 -> 0.25 change is NOT smaller than 1 -> 0.5.
- Cause (measured):
  - bottom torque ~1/dx at early times (t = 0.1 s: 1.9 / 3.6 / 6.2e-6 N m);
  - under the bar the first-cell u_theta is fixed by the forcing (0.080 / 0.080 / 0.074 m/s), so the
    wall shear u1/(dz/2) diverges. This is a forcing-wall contact layer of thickness
    ~sqrt(nu tau_s) = 0.07 mm, a property of the stirrer MODEL (chi touches the no-slip bottom);
  - with a diagnostic moving bottom under the footprint, the 1 -> 0.5 mm bottom-torque change at 0.3 s
    falls from 42 % to 10 %. The remainder is the genuine bottom / wall layers (spin-up sqrt(nu t)
    ~0.3 mm; bottom radial jet delta95 0.46-0.49 mm = 1-2 cells at 0.25 mm; side-wall delta95 1.2-1.4
    cells at 0.5 mm, grid-bound).

**Free surface (S1am).**
- Deepening starts when the swirl reaches the upper axis: Delta t_response = t_deepen(0.5 mm) -
  t_swirl((5,40), 10 %) = 0.00 / -0.05 s (150 rpm) and -0.05 / -0.10 s (200 rpm) on 1 / 0.5 mm, i.e.
  grid-independent within the 0.05 s logging. t_swirl itself shifts +0.5 s from 1 -> 0.5 mm.
- 300 rpm, 1 mm: N_m < 8 at 1.30 s AT THE WALL (the surface rises above the pinned contact point),
  before the axis deepens: unresolved. 300 rpm, 0.5 mm: onset at 1.80 s resolved (N >= 17), then
  collapse d 2 -> 8 mm in 0.1 s with N < 8 at 2.10 s.

**Rigid lid at 0.25 mm (final, to t = 2.4-2.55 s).**

| method | dx | arrival (5,25) 5/10/20 % | arrival (5,40) | L(1.5) | L(2.0) | T_stir(2.0) | T_wall(2.0) | budget (cum) |
|---|---:|---|---|---:|---:|---:|---:|---:|
| S0 | 1 / 0.5 / 0.25 | 1.70/1.75/1.85 ; 2.10/2.15/2.20 ; 2.00/2.00/2.05 | 1.60 ; 2.05 ; 1.95 | 6.21 / 6.15 / 6.80e-4 | 7.92 / 8.64 / 9.09e-4 | 3.9 / 6.6 / 5.1e-4 | -1.66 / -2.08 / -1.66e-4 | +2e-3 / +1e-3 / +2e-4 |
| S1 | 1 / 0.5 / 0.25 | 1.95/2.10/2.15 ; 2.20/2.25/2.25 ; 2.00/2.05/2.05 | 2.10 ; 2.00 ; 1.95 | 6.32 / 6.36 / 7.04e-4 | 8.01 / 8.88 / 9.48e-4 | 4.4 / 5.8 / 3.8e-4 | -1.68 / -1.98 / -1.44e-4 | +2e-3 / +9e-4 / +2e-4 |
| S2 | 1 / 0.5 / 0.25 | 1.75/1.80/1.80 ; 1.75/1.75/1.95 ; 1.80/1.80/1.85 | 1.50 ; 1.90 ; 1.70 | 5.98 / 6.05 / 5.84e-4 | 7.37 / 7.97 / 7.56e-4 | 4.6 / 5.2 / 5.6e-4 | -1.56 / -1.88 / -1.76e-4 | +2e-3 / +1e-3 / +2e-4 |

(S1am = S1 to within 1e-3 in every quantity except the budget, which closes to 1e-5.)
- Profile L2 differences (u_theta / meridional speed): 1 vs 0.5 mm 0.42-0.70 / 0.44-0.70; 0.5 vs 0.25 mm
  0.42-0.56 / 0.50-0.64. Refinement from 0.5 to 0.25 mm is NOT smaller than from 1 to 0.5 mm in arrival
  times (S1: +0.25 then -0.20 s), fields, or torques; L(2 s) differs +10.9 % then +6.7 % (slow, non-asymptotic).
- S0 and S1 agree at 0.25 mm (arrival 2.00 vs 2.00-2.05 s), so the swirl-scheme choice no longer matters
  there; the remaining dependence is not in swirl transport.
- Transient: S0 0.25 mm u_theta max 1.49 m/s at t = 2.2 s for one log interval, as the swirl front
  converges on the upper axis (q conserved, r_c = 0.125 mm). Back to 0.44 m/s at 2.5 s; budget closed.
  Axis intensification, not a conservation error.

**Where the torque does not converge (S1 snapshots).**

| dx | T_bottom under bar, t = 2 s | T_bottom outside | T_side |
|---:|---:|---:|---:|
| 1.0 | 5.5e-6 | 6.3e-5 | 1.00e-4 |
| 0.5 | 1.0e-5 | 9.3e-5 | 0.96e-4 |
| 0.25 | 2.0e-5 | 6.1e-5 | 0.63e-4 |

- Under the bar: grows ~1/dx (forcing-wall contact layer ~sqrt(nu tau_s) = 0.07 mm). The 0.125 mm
  discriminator (reached t = 0.05 s only) continues the growth: 0.90 / 1.73 / 3.05 / 4.50e-6 N m.
- Side wall and outer bottom: non-monotone.

**TABLE C (measured layers, `results/validation_v7t/tableC_boundary_layers.md`), t = 2 s, S1.**
- bottom radial jet (u_r, r = 20 / 30 mm): delta95 = 0.54 / 0.54 mm and 0.62 / 0.55 mm on 0.5 / 0.25 mm
  (2.2 cells at 0.25);
- bottom swirl layer (r = 30 mm): 0.65 / 0.62 mm (2.5 cells);
- side-wall swirl layer (z = 10 mm): 0.72 / 0.82 mm (3.3 cells); side-wall up-flow u_z: 0.60 / 1.18 mm
  (NOT converged).
On 1 mm the maxima sit in the first cell (layer < 0.5 mm unmeasurable). The laminar Ekman estimate
0.18 mm is too thin; the measured layers are ~0.55-0.8 mm but only 2-3 cells at 0.25 mm.

**Free surface (S1am, final).**

| RPM | dx | t_swirl (5,40) 10 % | t_deepen (0.5 mm) | Delta t_response | d_late (last 1 s) | N_curv,min |
|---:|---:|---:|---:|---:|---:|---:|
| 150 | 1.0 | 3.10 | 3.10 | 0.00 | 2.17 +- 0.05 | 12.2 |
| 150 | 0.5 | 3.60 | 3.55 | -0.05 | 2.15 +- 0.27 | 11.7 |
| 200 | 1.0 | 2.30 | 2.25 | -0.05 | 3.74 +- 0.27 | 10.1 |
| 200 | 0.5 | 2.80 | 2.70 | -0.10 | 4.15 +- 0.87 | 8.8 |
| 300 | 1.0 | - | 1.50 (stop) | - | unresolved at the wall pin (N < 8 at 1.30 s) | 6.4 |
| 300 | 0.5 | 2.00 | 1.80 | -0.20 | collapse 2 -> 8 mm, N < 8 at 2.10 s | 4.7 |

The conditional response (deepening starts when swirl reaches the upper axis) is grid-independent
within the 0.05 s logging. The swirl arrival itself, and so the absolute deepening time, is not
converged. Late depth at 150 rpm agrees (1 %) but the 0.5 mm surface oscillates. 0.25 mm free-surface
runs were not feasible in-session (~12 h each); 450 rpm was not run (low-RPM gate not passed).

**Classification: Outcome D. Final status label: B (UNIFORM GRID RESOLUTION STILL BLOCKING).**
- Swirl transport is fixed:
  - 2nd order, conservative to round-off, budget to 1e-5;
  - measured nu_num ~1-6 x nu at 0.25-0.5 mm, from 26-53 x nu.
- The blocker is the wall/bottom viscous and shear layers (0.55-0.8 mm, 2-3 cells at 0.25 mm; torques
  and the wall up-flow not converged) plus, as a separate MODEL issue, the forcing-wall contact layer
  under the bar. With the stirrer's relaxation forcing touching the no-slip bottom, that layer thins to
  ~sqrt(nu tau_s) = 0.07 mm and its torque diverges ~1/dx on every uniform grid tested; a diagnostic
  moving bottom under the footprint makes the early bottom torque converge (10 % -> 2 %).
- Second-order meridional transport (S2) is implemented and validated in isolation, but it is not
  adopted: it is dt-sensitive (chaotic) and does not improve grid convergence at 1-0.25 mm.
- Next (user decisions):
  1. choose the stirrer-bottom model (keep the no-slip bottom under the bar and accept a 0.07 mm
     layer, or treat the bar footprint as a moving surface / lift the kernel);
  2. then a stretched grid in r at the wall and z at the bottom (docs/nonuniform_grid_plan.md);
     uniform refinement below 0.25 mm is ~20x too expensive per simulated second.
- tau_s calibration stays blocked.

### 10.z V7-S: stirrer-bottom model consistency (2026-09-26/27)

tau_s = 0.005 s FIXED, uncalibrated; placeholder geometry; transport S1am (MUSCL2 swirl + angular-
momentum viscous + legacy meridional). Results: `results/validation_v7s/` (status.md, early/, lid/, free/,
gate1_static.jsonl, tableE_boundary_layers.md). Scripts: validate_stirrer_models.py, analyze_v7s.py,
analyze_v7s_lid.py; runner switches `stirrer=` and `taper=`.

**Models (config `stirrer.model`, opt-in; default unchanged).**
- M0 `wall_touching_volume` (legacy): chi reaches the no-slip bottom.
- M1 `tapered_volume`: chi_z times the C2 smoothstep S(z / ell_z), zero at the bottom. ell_z is a
  physical length. Tested ell_z = eps_f (1.0 mm), D_m/4 (1.75 mm), D_m/2 (3.5 mm). Balance estimate:
  forcing-viscous layer (nu tau_s ell^3)^(1/5) = 0.35 / 0.49 / 0.73 mm, instead of
  sqrt(nu tau_s) = 0.07 mm.
- M2 `moving_footprint` (height branch only): no volume forcing; the bottom moves with
  Omega(t) r w(r), w = (1 - tanh((r - R_m)/eps))/2 (fixed 1 mm transition). An effective moving
  surface, not a rotating glass bottom and not the real bar.

**Gate 1 (static).** All support volumes / moments / initial torques converge (<= 7e-4 at 1 mm).
M1 T0 is -7 / -12.5 / -25 % vs M0 (ell 1 / 1.75 / 3.5); M2 moments converge to 1e-9. Not renormalised.

**Gate 2 (early rigid lid, 1 / 0.5 / 0.25 mm, to 0.5 s).**
- M0 under-bar bottom torque doubles per refinement at every time (singular).
- M1 bounded for every ell (t = 0.5 s: 3.0/4.2/4.0, 2.4/2.6/2.3, 1.4/1.3/1.1e-6 N m).
- M2 +60 %, +21 %, ~100x weaker drive; its whole wall shear grows with refinement (disk layer
  ~0.18 mm unresolved); the edge peak stays at a fixed r ~13.1-13.5 mm with a constant ratio to the
  interior, so there is no separate edge singularity.
- Budget residual / L_z ~ -2.4e-4 at 0.25 mm and 0.5 s for every model (scales with dx; spin-up
  integration of the diagnostic); 6e-5 at 150 rpm, 7 s.

**Rigid lid, 300 rpm, to 3 s (R = D05_025 / D10_05).**

| observable (ell 1.75 mm) | 1 mm | 0.5 mm | 0.25 mm | D10_05 | D05_025 | R |
|---|---:|---:|---:|---:|---:|---:|
| L_z(2 s) | 8.16e-4 | 9.17e-4 | 9.08e-4 | 0.110 | 0.010 | 0.09 |
| L_z(3 s) | 9.98e-4 | 1.081e-3 | 1.077e-3 | 0.077 | 0.004 | 0.05 |
| arrival (5,25) 10 % | 2.05 s | 2.25 s | 2.25 s | 0.20 s | 0.00 s | 0.00 |
| arrival (5,40) 10 % | 1.75 | 2.05 | 2.45 | 0.30 | 0.40 | 1.33 |
| T_stir [2.5, 3] | 3.29e-4 | 2.16e-4 | 2.72e-4 | 0.41 | 0.20 | 0.49 |
| T_side [2.5, 3] | -1.07e-4 | -6.57e-5 | -7.33e-5 | 0.56 | 0.10 | 0.18 |
| T_bottom [2.5, 3] | -6.33e-5 | -8.16e-5 | -6.83e-5 | 0.27 | 0.19 | 0.73 |
| U_mer [2.5, 3] | 0.308 | 0.303 | 0.350 | 0.015 | 0.135 | 8.8 |
| upper-L fraction (3 s) | 0.482 | 0.452 | 0.429 | 0.07 | 0.05 | 0.75 |
| bulk u_theta L2 (3 s, walls excluded) | | 0.280 (1 vs 0.5) | 0.467 (0.5 vs 0.25) | | | |

ell 3.5 mm: L_z R 1.05 (2.5 s) / 0.53 (3 s), T_stir R 1.1.
Outcome: angular-momentum content and near-axis arrival converge; the meridional circulation
(U_mer), late torques and instantaneous bulk fields do not (R2 for the meridional flow).

**ell_z sensitivity (M1 1.75 vs 3.5 mm).**
- Rigid lid 0.25 mm: L_z(3 s) 2.6 %, L_z(2 s) 9 %, upper-L 1.5 %, U_mer 9 %, late T_stir 19 %.
- 150 rpm free surface 0.5 mm: late depth 2.28 vs 2.29 mm (0.4 %), L_z 3.3 %, T_stir 10 %.
- The bulk and depth response is weakly sensitive; torques are moderately sensitive. ell_z is NOT a
  calibration parameter: it is fixed at D_m/4 = 1.75 mm (the smallest geometric taper that converges;
  1.0 mm is one cell at 1 mm and its outer-bottom torque does not converge).
- Any global drive-strength change is absorbed by tau_s, but torque-based observables would carry
  ~10-20 % taper uncertainty.

**TABLE E (post-M1 layers, t = 2 s, ell 1.75).**
- under-bar bottom layer 2.3-2.6 mm (taper-controlled, 9-10 cells at 0.25);
- outer bottom radial jet 0.6-0.9 mm (2.5-3.5 cells);
- outer bottom swirl 0.76 mm (3 cells);
- side-wall swirl 1.09 mm (4.3 cells);
- side-wall up-flow u_z 0.63 -> 1.53 mm (not converged);
- episodic impingement layer under the bar at 2.5-3 s ~0.3-0.5 mm (omega overshoots to 40-44 rad/s;
  1.5-2 cells at 0.25).
The old 0.07 mm forcing-wall layer is gone.

**150 rpm free surface (M1 1.75 mm).**

| dx | d_late [5, 7] s (std) | last 0.5 s | T_stir | L_z | t_swirl (5,40) 10 % | N_curv,min | U_mer |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1.0 | N < 8 at 3.45 s (outside the credible regime) | - | - | - | 3.25 | 7.1 | - |
| 0.5 | 2.28 (0.43) | 2.37 | 9.43e-5 | 6.271e-4 | 3.45 | 19.4 | 0.168 |
| 0.25 | 2.13 (0.31) | 2.39 | 9.50e-5 | 6.279e-4 | 3.25 | 41.8 | 0.189 |

The 0.5 and 0.25 mm grids agree:
- L_z 0.1 %;
- T_stir 0.8 %;
- depth 7 % (window mean), inside the surface oscillation (std 0.3-0.4 mm);
- U_mer 12 %; arrival 0.2 s.
Only two grids are credible (1 mm unresolved), so a converging three-grid sequence is NOT
demonstrated. 200 / 300 / 450 rpm were not run.

**Decision: B. M1 ACCEPTED -- STRETCHED GRID REQUIRED FOR GLOBAL CONVERGENCE.**
- The model singularity is removed and M1 is the numerical production stirrer-model candidate
  (ell_z = D_m/4; tau_s stays the single calibration parameter).
- The meridional circulation still changes 12-16 % from 0.5 to 0.25 mm (rigid lid and 150 rpm), with
  the near-wall layers at 1.5-4 cells.
- Next: stretched grid per docs/nonuniform_grid_plan.md (dz_min ~0.1 mm bottom, dr_min ~0.2 mm wall,
  ~1.6e4 cells) as the third, finer credible grid.
- Calibration stays blocked, although 150 rpm is the leading candidate calibration point
  (N_curv >= 19, depth agreeing within the oscillation).
- M2 is kept as a diagnostic reference only.
