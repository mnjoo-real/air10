# KYPT / IYPT 2027 Problem 10 — Air Vortex
## Revised simulation architecture after numerical-diagnostics review

**Revision:** 2026-09-24  
**Primary recommendation:** use a **single-phase, water-only, axisymmetric free-surface solver** for the main IYPT parameter study. Keep the existing full two-phase diffuse Level-Set solver as a legacy research/diagnostic branch, not as the production path for predicting vortex depth or the air-core contact threshold.

---

# Status and direction at a glance

> **Where we are:** the Level-1A single-phase solver (water only, sharp free surface) is
> implemented and verified up to isolated-interface surface tension. Wall contact is in
> progress. Stirrer forcing, calibration and RPM sweeps have **not** started.
> **PRODUCTION READY: NO.**
>
> Detailed numbers, methods and open issues: [`docs/level1a_single_phase.md`](docs/level1a_single_phase.md).
> Session 1–7 history of the legacy two-phase solver: [`docs/legacy_two_phase_diagnostics.md`](docs/legacy_two_phase_diagnostics.md).

## A. Direction

```text
Experiment question:  d(N) = vortex depth,   N_c(H) = RPM at which the air core reaches the stir bar
Model:                Level 1A = axisymmetric water-only Navier-Stokes + swirl
                      + moving free surface (Level Set, geometry only)
                      + atmosphere / capillarity as a SHARP pressure condition on phi = 0
Strategy:             verify each numerical piece on a manufactured case with an exact answer
                      -> then connect the stirrer -> calibrate tau_s on ONE measured depth
                      -> predict the other RPMs / depths without refitting
```

Per time step, the production path is:

```text
phi^n -> liquid / void / interface geometry (sub-cell theta)
      -> extend u^n into a void band (>= 5 layers when sigma > 0)
      -> liquid predictor u*  (rho_w, mu_w constant; no CSF, no air)
      -> p_Gamma = p_atm + sigma * kappa  at every sub-cell crossing
      -> liquid-only pressure solve (ghost-fluid Dirichlet)  -> projection
      -> extend u^{n+1}  -> Level Set transport (MUSCL2 + SSP-RK2)
      -> reinitialization only if a quality metric requires it (RS2)
      -> diagnostics: z_tip, d(t), top-connected air vs stir-bar region
```

## B. Validation gate status

| gate | what it checks | status |
|---|---|---|
| V1 | operators, ghost-fluid pressure BC (1D/2D manufactured, 2nd order) | **PASS** |
| V2 | hydrostatic free surface, off-grid | **PASS** (round-off) |
| V3 | rigid-body rotating liquid, co-rotating (matched) wall, $\sigma = 0$ | **PASS** with reinit OFF (NRMSE $\le 6\times10^{-4}$) |
| V4 | isolated capillary interface (static drop) | **PASS** for $R/\Delta x \ge 8$; WARN at 6; FAIL at 4. Open: slow rigid drift of a free drop (net spurious axial force) |
| V4b-S | wall contact geometry (angle convention, wall curvature, Young–Laplace meniscus) | geometry **PASS**; long-time wall equilibrium **FAIL** |
| V4b-P | pinned contact line invariant to where the pin sits inside a cell | **IN PROGRESS**: new reconstruction cuts median spurious flow 7x, one phase still unstable |
| V5a | Level Set transport, zero-flow invariance | **PASS** |
| V5b | reinitialization keeps $\phi = 0$ fixed (RS2 subcell fix) | **PASS** for planes and the rigid-body paraboloid; drifts on radii $\lesssim 8$ cells |
| V5c | rigid body with operational reinit | **PARTIAL** (shape PASS, small velocity floor) |
| V6–V7 | time-step / grid convergence of the production path | pending |
| V8–V10 | stirrer forcing, calibration, experiment | pending (blocked on V4b) |

## C. Equations used, in one place

**Liquid region and interface** ( $\phi<0$ liquid, $\phi>0$ air/void ):

```math
\frac{1}{r}\frac{\partial (r u_r)}{\partial r}+\frac{\partial u_z}{\partial z}=0,
\qquad
\rho_w\Big(\frac{\partial \mathbf u}{\partial t}+\mathbf u\cdot\nabla\mathbf u\Big)
=-\nabla p+\mu_w\nabla^2\mathbf u+\rho_w\mathbf g+\rho_w f_\theta\,\mathbf e_\theta
```

```math
\frac{\partial\phi}{\partial t}+\mathbf u_{\rm ext}\cdot\nabla\phi=0,
\qquad
p_\Gamma=p_{\rm atm}+\sigma\kappa,
\qquad
\kappa=\nabla\cdot\frac{\nabla\phi}{|\nabla\phi|}
=\underbrace{\frac{\phi_{rr}\phi_z^2-2\phi_r\phi_z\phi_{rz}+\phi_{zz}\phi_r^2}{|\nabla\phi|^3}}_{\kappa_m\ \text{(meridional)}}
+\underbrace{\frac{\phi_r}{r\,|\nabla\phi|}}_{\kappa_\theta\ \text{(hoop)}}
```

Sign convention (measured, not assumed): a liquid sphere has $\kappa=+2/R$ and $p_{\rm liquid}=p_{\rm atm}+2\sigma/R$.

**Projection with the sharp free surface** ($\theta$ = sub-cell crossing fraction from liquid node $P$ to void node $Q$):

```math
\nabla^2 p=\frac{\rho_w}{\Delta t}\nabla\cdot\mathbf u^{*},
\qquad
\mathbf u^{n+1}=\mathbf u^{*}-\frac{\Delta t}{\rho_w}\nabla p,
\qquad
\theta=\frac{\phi_P}{\phi_P-\phi_Q},
\qquad
\left.\frac{\partial p}{\partial n}\right|_{\rm face}=\frac{p_\Gamma-p_P}{\theta\,h}
```

**Velocity extension** (not air flow; swirl extended as angular velocity so rigid rotation stays exact):

```math
\mathbf n\cdot\nabla u_r=\mathbf n\cdot\nabla u_z=0,
\qquad
\mathbf n\cdot\nabla\!\left(\frac{u_\theta}{r}\right)=0
\quad\text{in the void band}
```

**Reinitialization** (Russo–Smereka subcell fix, Min–Gibou form; $\phi^0$ frozen; the difference toward a neighbour across $\phi^0=0$ uses $\phi=0$ at the crossing distance $\Delta x^{\pm}$):

```math
\frac{\partial\phi}{\partial\tau}+\operatorname{sgn}(\phi^0)\big(|\nabla\phi|-1\big)=0,
\qquad
D^{+}_{x}\phi_i=\frac{0-\phi_i}{\Delta x^{+}}-\frac{\Delta x^{+}}{2}\,\mathrm{minmod}(D_{xx}\phi_i,D_{xx}\phi_{i+1})
```

**Wall contact** ($\theta_c$ measured through the liquid, $\mathbf e_r$ = wall normal):

```math
\text{static angle: }\ \mathbf n\cdot\mathbf e_r=-\cos\theta_c\ \ \big(\eta'(R)=\cot\theta_c\big),
\qquad
\text{pinned line: }\ \eta(R)=z_{\rm pin},\ \theta \text{ free}
```

**Reference equilibria used for validation**

```math
\text{rigid body: }\ \eta(r)=C+\frac{\Omega^2 r^2}{2g},
\qquad
\text{meniscus (Young–Laplace): }\ \sigma\kappa=P_0-\rho_w g z\ \text{ on } \Gamma,\ \ \text{fixed volume}
```

**Stirrer forcing** (production; connected only after the numerical gates pass):

```math
f_\theta=\chi(r,z)\,\frac{\Omega_m r-u_\theta}{\tau_s},
\qquad
\Omega_m=\frac{2\pi N_{\rm actual}}{60}
```

**Time step**

```math
\Delta t=\min\!\Big(\Delta t_{\rm CFL},\ \Delta t_\nu,\ \tfrac12\sqrt{\Delta x/g},\ \sqrt{\frac{\rho_w\Delta x^3}{4\pi\sigma}}\Big)
```

**Observables**

```math
d(t)=H-z_{\rm tip}(t),
\qquad
\text{contact: }A_{\rm top}\cap\mathcal B_\delta\neq\varnothing\ \text{ for }\ t\ge n_p\frac{60}{N}
```

## D. Resolution rules learned so far

- A curved interface needs **at least 8 cells per radius of curvature**; 12 is comfortable. This comes from the isolated-drop benchmark. Check the meridional and azimuthal radii separately ($N_m = R_m/\Delta x$, $N_\theta = R_\theta/\Delta x$). An air-core tip thinner than about $8\Delta x$ is **not** resolved (4 mm at $\Delta x = 0.5$ mm).
- With surface tension, the velocity-extension band must be at least **5 layers** (curvature stencil 1+2 plus transport 2); the default is 6.
- Reinitialize only when needed. Periodic reinitialization degrades curvature.
- Validate on manufactured cases whose PDE, initial condition and boundary conditions are mutually consistent. For example, never grade solid-body rotation against a stationary no-slip wall.

## E. Next steps

1. Finish V4b-P: make the pinned contact line independent of the pin's sub-cell position, checked by median **and** worst phase on three grids.
2. Integrated gravity + capillarity + rotation equilibrium.
3. V6–V7 convergence of the production path. Then connect the stirrer forcing, measure the contact-line behaviour and the actual RPM, and calibrate $\tau_s$ on one depth.

**Experiment needed now:** film the liquid–wall contact line from the side during an RPM ramp, up and down, to decide between the pinned and moving contact-line models.

---


# 0. Executive decision

## 0.1 Research goal

The numerical model is intended to predict how a rotating magnetic stir bar deforms the liquid free surface and to estimate the critical condition at which the **top-connected air region reaches the rotating stir bar**.

The central observables are:

```math
d_\infty(N),
\qquad
N_c(H),
\qquad
r_{\rm air}(z),
```

where:

- $N$: actual stir-bar RPM,
- $d_\infty$: steady or statistically steady vortex depth,
- $N_c$: critical RPM for persistent air-core contact,
- $H$: liquid depth.

The simulation is **not** required to model bubble pinch-off, dispersed gas, or post-contact gas entrainment as its first objective.

## 0.2 Architecture change

The original Level-1 design solved water and air simultaneously with a diffuse Level-Set material transition:

```math
\rho=\rho(\phi),\qquad \mu=\mu(\phi),
```

plus a variable-density pressure projection and CSF surface tension.

That formulation is retained as **Level 1B / legacy two-phase mode**, but it is no longer the recommended production path.

The new main model is:

```math
\boxed{
\text{Level 1A: axisymmetric water-only Navier--Stokes + swirl + moving free surface}
}
```

with:

- water equations solved only in the liquid region,
- no explicit air momentum equation,
- atmospheric pressure imposed directly at the liquid free surface,
- surface tension imposed as an interfacial pressure jump rather than as a diffuse volumetric CSF force,
- Level Set retained primarily for geometry, kinematics, curvature, and top-connected-air detection.

## 0.3 Why this redesign is justified

The redesign is based on both the project diagnostics and the literature.

### Project diagnostics

The extensive validation campaign established that:

- the production variable-density projection algebra is internally consistent for BC-compatible manufactured gradient fields,
- the pressure matrix and projection use matching face coefficients,
- divergence is reduced to solver tolerance / machine-level values,
- several large apparent errors came from **invalid validation constructions**, especially prescribing solid-body swirl in the air while also imposing a flat $p=0$ atmospheric top boundary,
- spatially masked prescribed swirl can itself generate non-integrable forcing and is therefore unsuitable as a strict equilibrium test,
- the diagnostic $\Psi$-based well-balanced face-density construction reduced long-time growth but failed grid-convergence and perturbation-preservation tests, so it must **not** be productionized,
- the global `swirl_mode="prescribed"` is validation-only; the actual production model uses the intended **local bottom stirrer forcing**.

Therefore the final Session-7 conclusion is not “the production solver has a single identified projection bug.” Rather, the validation path itself had become entangled with the numerical difficulties of a high-density-ratio diffuse two-phase formulation.

### Literature rationale

Relevant prior work supports separating free-surface deformation from full gas entrainment:

1. **Halász et al. (2007)** measured magnetic-stirrer vortex flow and found an approximately ideal-vortex-like tangential field outside a core together with a strong central downward jet.  
   DOI: `10.1119/1.2772287`

2. **Mahmud et al. (2009)** studied free-surface turbulent flow generated by a cylindrical magnetic stirrer using experiments plus CFD with a VOF interface model and turbulence closures. Even a mature CFD approach showed quantitative difficulty in vortex-depth / radial-velocity prediction.  
   DOI: `10.1016/j.ces.2009.06.059`

3. **Kumar et al. (2018)** investigated vortex formation and air entrainment above a rotating submerged disk. The vortex-tip/contact condition and the post-contact gas-entrainment regime are physically distinguishable, which supports stopping the present Level-1 model at persistent contact.  
   DOI: `10.1021/acs.iecr.8b00379`

4. **Schillaci et al. (2016)** presented a Level-Set-aided **single-phase free-surface** method that deactivates the light phase and imposes free-surface pressure directly. The paper reports improved stability and computational performance compared with full two-phase treatment for problems in which gas stresses are negligible.  
   DOI: `10.1016/j.compfluid.2016.09.014`

5. **Raessi & Pitsch (2012)** and **Nangia et al. (2019)** show why high-density-ratio Level-Set solvers are numerically demanding: consistent mass/momentum transport and well-balanced body-force treatment are important to avoid non-physical errors.  
   DOI: `10.1016/j.compfluid.2012.04.002`  
   DOI: `10.1016/j.jcp.2019.03.042`

6. **Olsson & Kreiss (2005)** provide a conservative Level-Set formulation that may be considered later if volume conservation remains limiting in the new single-phase solver.  
   DOI: `10.1016/j.jcp.2005.04.007`

7. COMSOL provides both a **single-phase free-surface mixer** workflow and a fully two-phase Level-Set mixer workflow. This supports using the simpler free-surface formulation for surface-deformation work and reserving full two-phase calculations for gas-entrainment questions.

---

# 1. Experimental system

## 1.1 Magnetic stirrer

Current stirrer:

- model: **DAIHAN MSH-20D**,
- nominal speed range: **80–1500 rpm**,
- setting resolution: **5 rpm**,
- current stir bar:
  - length $L_m=30\ \mathrm{mm}$,
  - diameter $D_m=7\ \mathrm{mm}$.

The actual stir-bar RPM must be measured experimentally because magnetic slip / step-out can make the real bar speed differ from the display setting.

## 1.2 Vessel

Current vessel:

- DURAN low-form beaker with spout,
- nominal volume: 600 mL,
- nominal diameter: $D_v=90\ \mathrm{mm}$,
- nominal radius: $R_v=45\ \mathrm{mm}$,
- height: 125 mm.

For final simulations, use the **measured internal diameter**, not the catalog nominal diameter.

## 1.3 Stir-bar target geometry

```math
R_m=\frac{L_m}{2}=15\ \mathrm{mm}.
```

For a circular 7 mm cross-section resting on the bottom,

```math
z_{\rm bar,top}\approx7\ \mathrm{mm}.
```

The air core need only reach the stir-bar target region, not the vessel bottom.

---

# 2. Main research questions

## Q1. Vortex depth versus RPM

```math
d_\infty=f(N).
```

A reduced model suggests approximately

```math
d_\infty\propto N^2
```

over a limited regime. This is a hypothesis to test, not an imposed law.

## Q2. Critical air-core contact

Define $N_c$ as the smallest actual stir-bar RPM for which the top-connected gas/void region reaches the stir-bar target region and remains connected for a prescribed persistence time.

```math
N_c=f(H,\mu,\rho,\sigma,L_m,D_m,R_v,\ldots).
```

## Q3. Water-depth dependence

Measure or simulate

```math
N_c(H).
```

A first scaling estimate gives

```math
N_c\propto\sqrt{H_{\rm eff}},
\qquad
H_{\rm eff}=H-z_{\rm bar,top}.
```

## Q4. Air-core geometry prior to breakup

For conditions above onset but before strong bubble entrainment, characterize

```math
r_{\rm air}(z),
\qquad
D_{\rm air}(z)=2r_{\rm air}(z).
```

---

# 3. Modeling hierarchy

Use the following hierarchy.

```text
Level 0: reduced analytical / Rankine-type scaling
    ↓
Level 1A: 2D axisymmetric single-phase water free-surface CFD  ← MAIN MODEL
    ↓
Experiment: d(N), N_c(H), flow structure
    ↓
Level 1B: legacy full two-phase Level-Set solver, selected diagnostics only
    ↓
Level 2: selected 3D SPH / VOF-style validation cases
```

## 3.1 Level 0

Purpose:

- estimate useful RPM ranges,
- test dimensional trends,
- provide initial guesses.

## 3.2 Level 1A — main model

Purpose:

- parameter sweeps,
- vortex depth,
- free-surface shape,
- central downward flow,
- critical air-core contact.

## 3.3 Level 1B — legacy two-phase model

Purpose:

- numerical research,
- selected comparison cases,
- future gas-entrainment work if required.

Do **not** make the IYPT study depend on resolving all high-density-ratio diffuse-interface issues in this branch.

## 3.4 Level 2

Use only representative cases:

- clearly below onset,
- near onset,
- clearly above onset.

The 3D model is for rod-end effects, non-axisymmetric wakes, precession, and post-contact behavior.

---

# 4. Level 0 reduced vortex model

Let

- $\Omega_m$: actual stir-bar angular speed,
- $\omega_f$: effective vortex-core angular speed,
- $a$: effective core radius,
- $\beta$: stirrer-to-core coupling parameter.

```math
\Omega_m=\frac{2\pi N}{60},
\qquad
\omega_f=\beta\Omega_m.
```

A Rankine-type estimate gives

```math
d\sim\frac{\omega_f^2a^2}{g}
=\frac{\beta^2\Omega_m^2a^2}{g}.
```

Air-core contact is roughly expected when

```math
d_c\approx H-z_{\rm bar,top},
```

so

```math
\Omega_c\sim
\frac{\sqrt{g(H-z_{\rm bar,top})}}{\beta a}.
```

Use this only for parameter selection and scaling checks.

---

# 5. Level 1A governing equations

## 5.1 Coordinates and unknowns

Axisymmetric cylindrical coordinates:

```math
(r,\theta,z),
\qquad
\frac{\partial}{\partial\theta}=0,
\qquad
u_\theta\neq0.
```

The liquid region is

```math
\Omega_l(t)=\{(r,z):\phi(r,z,t)<0\}.
```

The solved velocity components are

```math
u_r,\quad u_z,\quad u_\theta.
```

Air is not solved as a fluid phase in Level 1A.

## 5.2 Incompressibility

Inside the liquid:

```math
\boxed{
\frac{1}{r}\frac{\partial(ru_r)}{\partial r}
+\frac{\partial u_z}{\partial z}=0
}
```

## 5.3 Liquid momentum

Inside $\Omega_l(t)$:

```math
\rho_w\left(
\frac{\partial\mathbf u}{\partial t}
+\mathbf u\cdot\nabla\mathbf u
\right)
=
-\nabla p
+\nabla\cdot\left[\mu_w(\nabla\mathbf u+\nabla\mathbf u^T)\right]
+\rho_w\mathbf g
+\rho_w\mathbf f_{\rm stir}.
```

The important radial swirl term is

```math
\frac{\partial u_r}{\partial t}+\cdots
-\frac{u_\theta^2}{r}
=-\frac1{\rho_w}\frac{\partial p}{\partial r}+\cdots
```

and the azimuthal equation contains

```math
\frac{\partial u_\theta}{\partial t}
+u_r\frac{\partial u_\theta}{\partial r}
+u_z\frac{\partial u_\theta}{\partial z}
+\frac{u_ru_\theta}{r}
=\text{viscous}+f_\theta.
```

## 5.4 Free-surface kinematics

The Level Set is advected with an extended liquid velocity:

```math
\boxed{
\frac{\partial\phi}{\partial t}
+\mathbf u_{\rm ext}\cdot\nabla\phi=0
}
```

where $\mathbf u_{\rm ext}$ equals the physical liquid velocity in water and is extended only a few cells into the void/gas side for numerical interface transport.

The extended velocity is **not** interpreted as physical air flow.

## 5.5 Dynamic free-surface condition

Neglecting gas inertia and gas viscous stress, the normal-stress balance is imposed directly at the interface.

First implementation:

```math
\boxed{
p_\Gamma=p_{\rm atm}+s_\kappa\sigma\kappa
}
```

where $s_\kappa=\pm1$ depends on the code's curvature sign convention.

The sign must be validated with a static capillary benchmark; do not hard-code the sign based only on notation.

A later refinement may include liquid viscous normal stress explicitly:

```math
(-p\mathbf I+\boldsymbol\tau)\mathbf n
=
-p_{\rm atm}\mathbf n+\sigma\kappa\mathbf n.
```

Tangential gas stress is approximated as zero.

## 5.6 Why atmospheric pressure is acceptable

The Level-1A assumption is that the top-connected gas core remains close to atmospheric pressure up to the contact threshold. This is appropriate for predicting **surface depression and first persistent contact**, but not bubble pinch-off or trapped-gas dynamics.

---

# 6. Level 1A pressure projection

Because only water is solved,

```math
\rho=\rho_w=\text{constant}
```

inside the active fluid domain.

The predictor is

```math
\frac{\mathbf u^*-\mathbf u^n}{\Delta t}
=
-\mathbf u\cdot\nabla\mathbf u
+\text{viscous}
+\mathbf g
+\mathbf f_{\rm stir}.
```

The pressure Poisson equation becomes

```math
\boxed{
\nabla^2p^{n+1}
=
\frac{\rho_w}{\Delta t}\nabla\cdot\mathbf u^*
}
```

inside liquid cells.

Velocity correction:

```math
\boxed{
\mathbf u^{n+1}
=
\mathbf u^*-\frac{\Delta t}{\rho_w}\nabla p^{n+1}.
}
```

## 6.1 Interface pressure treatment

For a liquid cell center $P$ whose stencil crosses the interface at $I$, impose

```math
p_I=p_\Gamma.
```

If the normal/grid-line distance from $P$ to $I$ is $d_{PI}$, use an interface-aware one-sided / ghost relation such as

```math
\left.\frac{\partial p}{\partial n}\right|_P
\approx
\frac{p_I-p_P}{d_{PI}}.
```

The exact discrete implementation may use ghost-fluid, cut-cell, or equivalent embedded-boundary algebra, but it must satisfy:

- the interface Dirichlet pressure at the actual sub-cell crossing,
- consistent use in matrix assembly and velocity projection,
- second-order behavior away from the interface,
- no explicit air pressure unknown.

## 6.2 No diffuse material transition in the production path

Level 1A must not construct

```math
\rho(\phi),\qquad \mu(\phi)
```

for the pressure/momentum solve.

The Level Set is a geometric marker, not a material-interpolation field for the production Level-1A solver.

---

# 7. Boundary conditions

## Axis $r=0$

```math
u_r=0,
\qquad
u_\theta=0,
\qquad
\partial_r u_z=0,
\qquad
\partial_r p=0.
```

## Vessel wall $r=R_v$

No slip:

```math
u_r=u_z=u_\theta=0.
```

## Bottom $z=0$

No slip:

```math
u_r=u_z=u_\theta=0.
```

The magnetic stirrer is represented by a volumetric azimuthal source above the bottom.

## Free surface

Pressure / stress is imposed directly at $\phi=0$.

There is no separate rectangular-domain atmospheric pressure boundary acting on a simulated air phase in Level 1A.

If the liquid interface reaches the computational top, that condition must be handled explicitly; normal production runs should include enough void headspace that this does not occur.

---

# 8. Stirrer forcing — production model

The physical rod is not axisymmetric, so Level 1A retains the effective bottom forcing model.

## 8.1 Forcing region

Approximate swept region:

```math
0\le r\lesssim R_m=15\ \mathrm{mm},
\qquad
0\le z\lesssim D_m=7\ \mathrm{mm}.
```

Use a smooth mask $\chi(r,z)$.

## 8.2 Azimuthal relaxation forcing

```math
\boxed{
f_\theta
=\chi(r,z)
\frac{\Omega_mr-u_\theta}{\tau_s}
}
```

where

```math
\Omega_m=\frac{2\pi N_{\rm actual}}{60}.
```

Important:

- this local forcing is the **production** swirl model,
- `swirl_mode="prescribed"` must remain validation-only,
- never prescribe $u_\theta=\Omega r$ throughout the entire water/void domain in production.

## 8.3 Calibration

$\tau_s$ is an effective coupling parameter.

Use one experimental subcritical condition:

1. measure actual bar RPM,
2. measure vortex depth,
3. fit $\tau_s$,
4. freeze $\tau_s$,
5. predict all other RPM and water-depth cases without re-fitting.

---

# 9. Level Set in the revised solver

## 9.1 Sign convention

```math
\phi<0:\text{ liquid},
\qquad
\phi>0:\text{ void / top-connected gas region},
\qquad
\phi=0:\text{ free surface}.
```

## 9.2 Recommended advection

Keep the already implemented:

- MUSCL2 reconstruction,
- MC limiter as default,
- SSP-RK2 time integration.

The old upwind1+Euler path may remain for regression only.

## 9.3 Velocity extension

Because the interface can move into previously inactive cells, construct a narrow-band extension $\mathbf u_{\rm ext}$ into $\phi>0$.

Recommended initial implementation:

- copy/extrapolate liquid velocity along the local interface normal,
- use only a 2–4 cell narrow band,
- do not interpret the extension as air dynamics.

A more formal extension equation can be added later if necessary.

## 9.4 Reinitialization

Reinitialization is allowed only if interface drift is measured and controlled.

Known diagnostic result:

- reinitialization can move $\phi$ even at zero physical velocity.

Therefore:

- use the minimum reinitialization frequency required for signed-distance quality,
- record interface displacement caused by reinitialization,
- keep a zero-velocity reinitialization regression test.

## 9.5 Volume conservation

Retain the existing global volume-correction option as a temporary safeguard, but do not let it hide transport errors.

Production acceptance requires comparison of:

- correction OFF,
- correction ON.

If long-time volume conservation remains problematic after the architecture change, consider a conservative Level-Set formulation rather than repeatedly increasing correction strength.

---

# 10. Surface tension in the revised solver

The preferred Level-1A production treatment is a **sharp interfacial pressure condition**, not diffuse CSF.

Use

```math
p_\Gamma=p_{\rm atm}+s_\kappa\sigma\kappa.
```

Advantages:

- removes $\delta_\epsilon(\phi)$-distributed surface-tension body force,
- avoids high-density-ratio force balancing across a diffuse material band,
- naturally fits the water-only free-surface formulation.

Curvature remains computed from the Level Set:

```math
\mathbf n=\frac{\nabla\phi}{|\nabla\phi|+\varepsilon_n},
\qquad
\kappa=\nabla\cdot\mathbf n.
```

Curvature noise must be monitored carefully because the pressure boundary condition depends directly on $\kappa$.

---

# 11. Air-core detection without solving air

The void region is

```math
A=\{\phi>0\}.
```

Find the component connected to the open headspace / computational top:

```math
A_{\rm top}.
```

Define the expanded stir-bar target region $\mathcal B_\delta$.

Air-core contact occurs when

```math
\boxed{
A_{\rm top}\cap\mathcal B_\delta\neq\varnothing.
}
```

Require persistence for several stir-bar rotations:

```math
t_{\rm connected}\ge n_pT,
\qquad
T=\frac{60}{N},
```

with initial recommendation $n_p=5$.

The primary Level-1A simulation may terminate once persistent contact is confirmed.

Do not interpret Level 1A after extensive bubble pinch-off or detached-gas formation.

---

# 12. Vortex depth

Let the initial free surface be $z=H$.

Let $z_{\rm tip}$ be the lowest point of the top-connected void region in the central region.

```math
\boxed{
d(t)=H-z_{\rm tip}(t).
}
```

Use the existing sub-grid $\phi=0$ interpolation; never revert to grid-face quantization.

Use time-averaged or statistically steady depth where appropriate.

---

# 13. Dimensionless parameters

Use the stir-bar half-length

```math
R_m=15\ \mathrm{mm}
```

as the characteristic radius unless another definition is explicitly stated.

## Rotational Reynolds number

```math
\boxed{
Re_\Omega=\frac{\rho\Omega R_m^2}{\mu}
}
```

## Rotational Froude number

```math
\boxed{
Fr_\Omega=\frac{\Omega^2R_m}{g}
}
```

## Weber number

```math
\boxed{
We_\Omega=\frac{\rho\Omega^2R_m^3}{\sigma}
}
```

## Bond number

```math
\boxed{
Bo=\frac{\rho gR_m^2}{\sigma}
}
```

## Geometry

```math
H^*=\frac{H}{R_m},
\qquad
C=\frac{R_m}{R_v}.
```

Always state the exact definitions because rotational Froude/Reynolds conventions differ between papers.

---

# 14. Numerical implementation

## 14.1 Grid

Retain the axisymmetric MAC-type grid.

Initial production candidate:

```math
\Delta r=\Delta z=0.5\ \mathrm{mm}
```

only after the single-phase grid-convergence gate passes.

## 14.2 Active liquid cells

Maintain masks such as:

- liquid cell,
- interface-cut cell,
- inactive void cell.

Pressure is solved only for liquid / cut-cell unknowns required by the chosen embedded-boundary formulation.

## 14.3 Pressure matrix

The old optimized sparse assembly infrastructure can be reused, but the matrix topology now changes as the interface moves.

Potential implementation strategies:

1. rebuild the active sparse matrix when the interface topology changes,
2. retain a full rectangular matrix but deactivate void unknowns with robust row treatment,
3. later optimize with cached topology if profiling justifies it.

Correctness takes priority over preserving the old full-domain matrix optimization.

## 14.4 Time step

Use

```math
\Delta t
=\min(
\Delta t_{\rm adv},
\Delta t_\nu,
\Delta t_\sigma,
\Delta t_{\rm forcing}
).
```

Surface tension can still impose a capillary time-step restriction even though it is imposed through an interface pressure jump.

## 14.5 Pressure solver

Start with direct sparse solve for validation grids.

Only revisit AMG / iterative methods after the new formulation is correct and profiled.

---

# 15. Revised verification and validation gates

The previous global two-phase prescribed-solid-body test is **retired as a strict equilibrium gate**.

It mixed:

- simulated rotating air,
- atmospheric top pressure,
- spatial swirl masking,
- diffuse density,
- free-surface geometry,

and was therefore not a clean manufactured equilibrium.

The new gate sequence is below.

## Gate V1 — discrete operators

Required:

- cylindrical divergence tests,
- gradient tests,
- pressure matrix / projection consistency,
- axis symmetry,
- sub-grid interface interpolation,
- velocity extension zero-field invariance.

## Gate V2 — hydrostatic free surface

Water only, no stirring.

Expected:

```math
\mathbf u\approx0,
\qquad
p(z)=p_{\rm atm}+\rho_wg(\eta-z),
```

with a flat interface.

Acceptance targets:

- max divergence near solver tolerance,
- no secular meridional velocity growth,
- volume drift <1%, preferably <0.5%.

## Gate V3 — rigid-body rotating liquid, $\sigma=0$, with matched tangential BCs

This is a **manufactured verification case**, not the production beaker wall condition. A stationary no-slip wall is incompatible with exact solid-body rotation $u_\theta=\Omega r$. Therefore use one of the following matched diagnostic boundary sets:

- rotating cylindrical wall/bottom with tangential speed $u_\theta=\Omega r$, or
- inviscid / tangential-slip diagnostic boundaries that do not overwrite the analytical swirl.

Then initialize in the liquid:

```math
u_\theta=\Omega r,
\qquad
u_r=u_z=0.
```

Do **not** grade this equilibrium while simultaneously imposing the production stationary-wall no-slip condition.

Analytical surface:

```math
\eta(r)=C+\frac{\Omega^2r^2}{2g}
```

with $C$ set by liquid volume.

Analytical pressure:

```math
p=p_{\rm atm}
+\rho_w\left[
\frac12\Omega^2r^2-g(z-\eta_0)
\right]
```

up to an equivalent constant convention.

This is now a genuine one-phase free-surface equilibrium test.

Acceptance target:

- NRMSE <0.05,
- center-depression relative error <5%,
- no $O(10^{-2})$ or larger secular meridional growth.

## Gate V4 — static capillary benchmark

Choose a geometry with known curvature / pressure jump.

Verify:

```math
\Delta p=\sigma\kappa
```

with the implemented sign convention.

This gate determines the correct $s_\kappa$.

## Gate V5 — moving free-surface transport

Use a standard advection / sloshing-style test.

Measure:

- interface position,
- water volume,
- reinitialization drift,
- time-step convergence.

## Gate V6 — grid convergence

Compare at least three grids for representative subcritical and near-contact cases.

Track:

- mean vortex depth,
- RMS depth oscillation,
- interface profile,
- central downward velocity,
- contact threshold bracket.

## Gate V7 — local stirrer forcing sanity

Use the actual production forcing only.

Check that increasing actual RPM gives physically sensible trends:

- increasing swirl,
- decreasing central pressure,
- increasing vortex depth,
- emergence of a central downward jet / toroidal circulation.

Do not require the forcing field to be an equilibrium gradient; it is intentionally a non-equilibrium momentum source.

## Gate V8 — experimental calibration

Fit $\tau_s$ at one measured subcritical condition.

Then freeze it.

Production-ready status must remain **NO** until this real calibration exists.

---

# 16. Experimental plan

## 16.1 Measurements required before final CFD

Measure:

- vessel internal diameter,
- actual liquid height,
- liquid temperature,
- actual stir-bar RPM,
- stir-bar position / clearance,
- vortex depth versus RPM for at least one calibration condition.

## 16.2 Recommended first calibration condition

Use approximately:

- $H=40$–50 mm,
- moderate RPM,
- clearly subcritical air core,
- stable visible vortex.

Measure $d_{\rm exp}$, fit $\tau_s$, then freeze it.

## 16.3 First validation sweep

After calibration:

```math
N=500,\ 900,\ 1200,\ 1500\ \mathrm{rpm}
```

or a comparable set using **actual bar RPM**.

Then bracket the contact threshold with bisection.

---

# 17. Data products

Every production run should save:

## Interface

- $d(t)$,
- $z_{\rm tip}(t)$,
- $r_{\rm air}(z,t)$,
- contact / persistence state,
- water volume.

## Flow

- $u_r(r,z,t)$,
- $u_z(r,z,t)$,
- $u_\theta(r,z,t)$,
- liquid pressure $p(r,z,t)$.

## Scalars

- max $u_\theta$,
- max downward $u_z$,
- minimum central gauge pressure,
- max divergence,
- time step,
- solver runtime,
- termination reason,
- actual RPM source (`actual` or explicit fallback).

---

# 18. Main plots

The final report should prioritize:

1. $d_\infty$ vs $N^2$,
2. $N_c^2$ vs $H_{\rm eff}$,
3. $d/R_m$ vs $Fr_\Omega$,
4. critical $Fr_{\Omega,c}$ vs $H/R_m$,
5. $u_\theta(r,z)$,
6. meridional streamlines / central $u_z(z)$,
7. free-surface profiles for several RPMs,
8. air-core profile $r_{\rm air}(z)$ before post-contact breakup becomes important.

All scaling laws are to be tested against data, not imposed.

---

# 19. Repository migration plan

Do not destroy the current codebase.

## 19.1 Preserve legacy

Create a tag or branch before architecture changes, for example:

```text
legacy-two-phase-session7
```

Keep the full two-phase solver runnable for historical diagnostics.

## 19.2 New configuration switch

Recommended:

```yaml
physics:
  free_surface_model: single_phase_ls   # default production path
```

Optional legacy value:

```yaml
physics:
  free_surface_model: two_phase_diffuse_ls
```

## 19.3 Suggested new modules

```text
src/air_vortex/
    liquid_mask.py
    free_surface_bc.py
    velocity_extension.py
    pressure_single_phase.py
```

Reuse existing:

```text
grid.py
operators.py
forcing.py
levelset.py
diagnostics.py
connectivity.py
run_io.py
plotting.py
animation.py
```

## 19.4 Production path

The default physical-run path should become:

```text
phi
  ↓
liquid / interface geometry
  ↓
liquid-only predictor
  ↓
interface p = p_atm + sigma*kappa
  ↓
liquid-only pressure solve
  ↓
projection
  ↓
velocity extension into narrow void band
  ↓
Level Set advection
  ↓
reinitialization if required
  ↓
diagnostics / connectivity
```

---

# 20. Implementation milestones

## Milestone A — freeze and regression

- archive current Session-7 behavior,
- keep all existing tests,
- add explicit legacy-mode tests,
- do not update golden fixtures merely to hide differences.

## Milestone B — liquid-only pressure solver, fixed interface

Implement a flat stationary interface first.

Pass:

- hydrostatic test,
- manufactured interface Dirichlet pressure test.

## Milestone C — moving interface without surface tension

Add:

- active-liquid mask,
- sub-cell pressure boundary,
- narrow-band velocity extension,
- Level Set transport.

Pass rigid-body rotating-liquid equilibrium at $\sigma=0$.

## Milestone D — sharp surface tension

Add interfacial $\sigma\kappa$ pressure jump.

Pass static capillary benchmark.

## Milestone E — production stirrer forcing

Connect the existing local bottom $f_\theta$ model.

Run only short qualitative pilots until calibration data exist.

## Milestone F — calibration

Use one real measured vortex depth to fit $\tau_s$.

## Milestone G — production sweeps

Only after all numerical gates and calibration pass:

- RPM sweep,
- depth sweep,
- threshold bisection,
- optional hysteresis study.

---

# 21. Legacy Session-7 findings that must not be forgotten

The following conclusions are retained as regression knowledge.

## 21.1 Projection algebra

For exactly BC-compatible manufactured fields, the variable-density projection annihilated $BGq$ to numerical precision.

Therefore the production projection core should not be treated as “known broken.”

## 21.2 Invalid global rotating-air equilibrium

The old two-phase solid-body validation imposed rotating air together with a flat atmospheric top pressure. These conditions are not a stationary continuum equilibrium.

That test is retired as a strict equilibrium gate.

## 21.3 Prescribed swirl mask

Water-only / phase-weighted prescribed swirl can itself produce a spatially non-integrable forcing field.

Therefore it is not a clean strict-equilibrium construction.

## 21.4 Production forcing is different

The actual production solver uses local bottom relaxation forcing, not global prescribed swirl.

Do not transfer conclusions from the artificial prescribed-swirl equilibrium diagnostics directly to the production stirrer source.

## 21.5 Rejected WB construction

The $\Psi$-specific well-balanced face-density diagnostic:

- reduced long-time accumulation,
- did not converge to zero under grid refinement,
- significantly changed perturbation dynamics,
- gave very poor response correlation for Rankine-like vortex perturbations.

It must not be adopted as a production scheme.

## 21.6 Level Set transport

MUSCL2+SSPRK2 substantially reduced numerical diffusion compared with first-order upwind.

Sub-grid vortex-tip extraction is validated.

Reinitialization still causes measurable drift and must remain monitored.

---

# 22. Turbulence and 3D limitations

The real stirrer flow can become transitional or turbulent at high RPM.

The direct magnetic-stirrer CFD literature used turbulence closures and still found quantitative discrepancies.

Therefore:

1. first validate the revised free-surface numerics,
2. calibrate the local stirrer coupling,
3. compare with experiment,
4. only then decide whether a turbulence / effective-viscosity / 3D correction is required.

Do not add an arbitrary turbulence model before the simpler model has been experimentally tested.

The axisymmetric model cannot represent:

- rod-end wakes,
- eccentricity,
- vortex precession,
- beaker spout asymmetry,
- asymmetric attachment to the rod.

Use selected 3D cases only if these effects materially limit the comparison.

---

# 23. When Level 1A is no longer valid

Stop treating the single-phase model as quantitatively reliable when:

- the gas core pinches off into detached bubbles,
- gas entrainment becomes sustained,
- trapped gas pressure becomes important,
- aerodynamic shear from air is no longer negligible,
- post-contact bubble dynamics are the quantity of interest.

Those regimes require Level 1B or Level 2 treatment.

For the primary IYPT onset question, persistent first contact is the natural Level-1A stopping condition.

---

# 24. Success criteria

The revised numerical model is considered useful when it reproduces, after one-point forcing calibration:

1. increasing vortex depth with RPM,
2. vortex-like tangential velocity structure,
3. central downward flow / toroidal recirculation,
4. a funnel-shaped free surface,
5. a reproducible critical contact RPM,
6. increasing critical RPM with liquid depth,
7. stable predictions under grid/time refinement,
8. consistent predictions at RPMs not used for calibration.

The strongest final IYPT result remains:

```math
\boxed{
\text{theory}
\leftrightarrow
\text{revised Python CFD}
\leftrightarrow
\text{experiment}
}
```

for

```math
d(N)
\qquad\text{and}\qquad
N_c(H).
```

---

# 25. Recommended first revised baseline

After the new single-phase validation gates pass:

```text
vessel radius:          measured value, nominal ~45 mm
water height:           50 mm initial baseline
stir bar:               30 mm x 7 mm
fluid:                   water only in momentum solve
free surface:            Level Set geometry
atmosphere:              direct interface pressure BC
surface tension:         sharp pressure jump
stirring:                local bottom relaxation forcing
Level Set advection:     MUSCL2 + SSPRK2
volume correction:       compare OFF and ON
initial grid candidate:  0.5 mm
```

Do not begin the full 500/900/1200/1500-rpm production sweep until:

- V1–V7 pass,
- a real calibration point exists,
- actual stir-bar RPM is available or the setpoint fallback is explicitly acknowledged.

---

# 26. Key references

1. Halász, G., Gyüre, B., Jánosi, I. M., Szabó, K. G., Tél, T. (2007). *Vortex flow generated by a magnetic stirrer*. American Journal of Physics 75, 1092–1098. DOI: `10.1119/1.2772287`.

2. Mahmud et al. (2009). *Measurements and modelling of free-surface turbulent flows induced by a magnetic stirrer in an unbaffled stirred tank reactor*. Chemical Engineering Science 64, 4197–4209. DOI: `10.1016/j.ces.2009.06.059`.

3. Kumar, P., Prajapati, M., Das, A. K., Mitra, S. K. (2018). *Vortex Formation and Subsequent Air Entrainment inside a Liquid Pool*. Industrial & Engineering Chemistry Research 57, 6538–6552. DOI: `10.1021/acs.iecr.8b00379`.

4. Schillaci, E., Jofre, L., Balcázar, N., Lehmkuhl, O., Oliva, A. (2016). *A level-set aided single-phase model for the numerical simulation of free-surface flow on unstructured meshes*. Computers & Fluids 140, 97–110. DOI: `10.1016/j.compfluid.2016.09.014`.

5. Raessi, M., Pitsch, H. (2012). *Consistent mass and momentum transport for simulating incompressible interfacial flows with large density ratios using the level set method*. Computers & Fluids 63, 70–81. DOI: `10.1016/j.compfluid.2012.04.002`.

6. Nangia, N., Griffith, B. E., Patankar, N. A., Bhalla, A. P. S. (2019). *A robust incompressible Navier–Stokes solver for high density ratio multiphase flows*. Journal of Computational Physics 390, 548–594. DOI: `10.1016/j.jcp.2019.03.042`.

7. Olsson, E., Kreiss, G. (2005). *A conservative level set method for two phase flow*. Journal of Computational Physics 210, 225–246. DOI: `10.1016/j.jcp.2005.04.007`.

8. COMSOL Application Library: *Free Surface Mixer* — single-phase free-surface rotating-mixer example.

9. COMSOL Application Library: *Free Surface Mixer Level Set* — time-dependent two-phase Level-Set rotating-mixer example.

---

# 27. Immediate next task

> **Status (2026-09-24):** the Level-1A solver and its first verification gates are implemented; see *Status and direction at a glance* at the top of this document.

The next coding task is **not** another Session-7 patch to the legacy diffuse two-phase solver.

It is:

```math
\boxed{
\text{implement and verify Level 1A: single-phase water + sharp free-surface pressure BC}
}
```

in a way that preserves the legacy branch and reuses as much validated infrastructure as possible.

The first milestone is again deliberately simple:

```math
\boxed{
\text{stationary single-phase water remains stationary under a tracked free surface}
}
```

The second milestone is:

```math
\boxed{
\text{a rotating liquid preserves the analytical free-surface equilibrium without any simulated air phase}
}
```

Only then should the local magnetic-stirrer forcing be reconnected.
