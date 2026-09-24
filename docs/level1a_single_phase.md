# Level 1A single-phase free-surface solver: Milestone 1 notes

Companion to `README_rewritten.md` (sections 5–7, 9, 14, 19–20). This file records
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
| V4b-S static wall contact geometry | geometry PASS (angle convention, CA0-CA2, wall curvature after ghost fix); Young-Laplace meniscus short-time PASS; LONG-HORIZON wall equilibrium FAIL (static_angle: exponential growth, blow-up ~2.6 s at 0.25 mm, dt-independent; pinned: sub-cell-position-dependent sustained sloshing up to 8e-2 m/s) |
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

- Replace `README.md` with `README_rewritten.md` once the user approves.
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

### 9.10 Checkpoint

The current state is a natural checkpoint: single-phase architecture, RS2 reinit, sharp surface
tension, V4b geometry, legacy bitwise regression. Suggested tag: `milestone-v4-single-phase-capillary`.
Not committed (awaiting user permission).
