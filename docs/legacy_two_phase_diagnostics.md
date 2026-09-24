# Legacy Level-1B two-phase diffuse Level-Set solver — Session 1–7 history

> **Archived verbatim** from the repository `README.md` as it stood before the
> Level-1A single-phase redesign (2026-09-24). Nothing below this note has been
> edited. The architecture source of truth is now `README_rewritten.md`; the
> legacy solver described here remains runnable as
> `physics.free_surface_model: two_phase_diffuse_ls` (the transition default).
> Session-7 conclusions that remain binding are summarised in
> `README_rewritten.md` section 21.

---

# KYPT / IYPT 2027 Problem 10 — Air Vortex
## Python-based simulation design for a magnetic-stirrer air core

> **Goal**  
> Build a Python simulation that predicts how a rotating magnetic stir bar deforms the free surface and determines the **critical condition for an air channel (air core) to connect the free surface to the stir bar**.

---

## 1. Experimental system

### 1.1 Magnetic stirrer

Current stirrer:

- Model: **DAIHAN MSH-20D**
- Speed range: **80–1500 rpm**
- Speed setting resolution: **5 rpm**
- Included stir bar: **3 cm**
- Usable stir bar length: up to **5 cm**
- Current stir bar:
  - length \(L_m = 30\ \mathrm{mm}\)
  - diameter \(D_m = 7\ \mathrm{mm}\)

The MSH-20D values above come from the uploaded DAIHAN catalog (`wisd-n-125.pdf`, p. 125).

### 1.2 Vessel

Current vessel:

- **DURAN® low-form beaker with spout, 600 mL**
- nominal diameter: \(D_v = 90\ \mathrm{mm}\)
- nominal radius: \(R_v = 45\ \mathrm{mm}\)
- height: \(125\ \mathrm{mm}\)

> **Important:** the catalog diameter may not be the exact internal diameter.  
> For final CFD input, measure the **internal diameter** directly.

### 1.3 Baseline geometry

For the first simulation:

\[
R_v = 45\ \mathrm{mm},\qquad
L_m = 30\ \mathrm{mm},\qquad
D_m = 7\ \mathrm{mm}
\]

The stir-bar half-length is

\[
R_m=\frac{L_m}{2}=15\ \mathrm{mm}.
\]

Assuming a circular 7 mm cross-section resting on the bottom,

\[
z_{\mathrm{bar,center}}\approx 3.5\ \mathrm{mm},
\qquad
z_{\mathrm{bar,top}}\approx 7\ \mathrm{mm}.
\]

This matters because the air core does **not** need to reach \(z=0\); it only needs to reach the rotating bar.

---

# 2. Research questions

The simulation should answer four primary questions.

### Q1. How does vortex depth depend on rotation rate?

\[
d_\infty = f(N)
\]

where

- \(N\): actual stir-bar rotation rate
- \(d_\infty\): steady or statistically steady vortex depth

A simple vortex model predicts approximately

\[
d_\infty\propto N^2
\]

over a limited regime.

---

### Q2. What is the critical rotation rate for air-core formation?

Define

\[
N_c
\]

as the smallest rotation rate for which the top-connected air region reaches the stir bar and remains connected for a prescribed persistence time.

The central target is

\[
N_c=f(H,\mu,\rho,\sigma,L_m,D_m,R_v,\ldots).
\]

---

### Q3. How does water depth change the critical condition?

Measure or simulate

\[
N_c(H).
\]

A first-order model suggests a square-root-type trend,

\[
N_c\propto \sqrt{H_{\mathrm{eff}}},
\]

where

\[
H_{\mathrm{eff}}
=
H-z_{\mathrm{bar,top}}.
\]

---

### Q4. What controls air-core geometry after formation?

Possible outputs:

\[
r_{\mathrm{air}}(z),\qquad
D_{\mathrm{air}}(z)=2r_{\mathrm{air}}(z),
\]

as functions of RPM, viscosity, water depth, and stir-bar geometry.

---

# 3. Overall simulation strategy

Do **not** start with a full 3D two-phase CFD simulation.

Use three levels.

```text
Level 0: reduced analytical model
        ↓
Level 1: 2D axisymmetric + swirl + free-surface solver
        ↓
Level 2: selected 3D SPH / particle simulations
        ↓
Experiment ↔ simulation validation
```

Recommended use:

| Level | Model | Main purpose | Cost |
|---|---|---|---|
| 0 | reduced Rankine-type model | estimate useful RPM range | very low |
| 1 | 2D axisymmetric CFD + swirl + Level Set | parameter sweep and \(N_c\) prediction | moderate |
| 2 | 3D SPH with rotating rod | validate non-axisymmetric rod effects | high |

The main research model should be **Level 1**.

---

# 4. Level 0 — reduced vortex model

This model is not a CFD solution. It is used to estimate parameter ranges before running the numerical solver.

Let

- \(\Omega_m\): actual stir-bar angular velocity
- \(\omega_f\): effective angular velocity of the vortex core
- \(a\): effective vortex-core radius
- \(\beta\): fluid–stirrer coupling coefficient

with

\[
\Omega_m=\frac{2\pi N}{60},
\]

and

\[
\omega_f=\beta\Omega_m.
\]

A Rankine-type estimate gives

\[
d\sim\frac{\omega_f^2a^2}{g}.
\]

Therefore

\[
d\sim
\frac{\beta^2\Omega_m^2a^2}{g}.
\]

An approximate air-core threshold is

\[
d_c\approx H-z_{\mathrm{bar,top}},
\]

which gives

\[
\boxed{
\Omega_c
\sim
\frac{\sqrt{g(H-z_{\mathrm{bar,top}})}}{\beta a}
}
\]

or

\[
\boxed{
N_c
\sim
\frac{60}{2\pi}
\frac{\sqrt{g(H-z_{\mathrm{bar,top}})}}{\beta a}
}.
\]

### Important limitation

\(a\) and \(\beta\) are **effective parameters**, not directly equal to stir-bar radius or a universal constant.

Use this model only for:

- selecting initial RPM values
- checking expected scaling
- generating initial guesses for the CFD runs

Do not use it as the final quantitative model.

---

# 5. Level 1 — 2D axisymmetric CFD with swirl

## 5.1 Why 2D axisymmetric?

The physical stir bar is not axisymmetric, but most of the air core is approximately centered on the vessel axis.

A 2D \((r,z)\) model with all three velocity components

\[
u_r,\quad u_z,\quad u_\theta
\]

captures:

- radial circulation
- axial circulation
- azimuthal swirl
- centrifugal pressure depression
- free-surface deformation
- formation of a central air core

at a fraction of the cost of a full 3D calculation.

The missing 3D stir-bar geometry is replaced by an **effective azimuthal momentum source** near the bottom.

---

# 6. Coordinate system

Use cylindrical coordinates:

\[
(r,\theta,z)
\]

with

- \(r=0\): vessel axis
- \(z=0\): beaker bottom
- \(z=H\): initial water surface

Axisymmetry means

\[
\frac{\partial}{\partial\theta}=0,
\]

but

\[
u_\theta\neq 0.
\]

Computational domain:

\[
0\le r\le R_v
\]

and

\[
0\le z\le Z_{\max}.
\]

Choose

\[
Z_{\max}=H+H_{\mathrm{air}},
\]

with typically

\[
H_{\mathrm{air}}=20\text{–}30\ \mathrm{mm}.
\]

Example for \(H=50\ \mathrm{mm}\):

\[
0\le r\le45\ \mathrm{mm},
\qquad
0\le z\le80\ \mathrm{mm}.
\]

---

# 7. Governing equations

## 7.1 Incompressibility

For axisymmetric flow,

\[
\boxed{
\frac{1}{r}
\frac{\partial (ru_r)}{\partial r}
+
\frac{\partial u_z}{\partial z}
=0
}
\]

---

## 7.2 Momentum equation

The compact one-fluid two-phase form is

\[
\rho
\left(
\frac{\partial\mathbf u}{\partial t}
+
\mathbf u\cdot\nabla\mathbf u
\right)
=
-\nabla p
+
\nabla\cdot
\left[
\mu
\left(
\nabla\mathbf u+\nabla\mathbf u^T
\right)
\right]
+
\rho\mathbf g
+
\mathbf F_\sigma
+
\mathbf F_{\mathrm{stir}}.
\]

Here

- \(\rho\): local density
- \(\mu\): local dynamic viscosity
- \(p\): pressure
- \(\mathbf g=(0,0,-g)\)
- \(\mathbf F_\sigma\): surface-tension force
- \(\mathbf F_{\mathrm{stir}}\): effective stir-bar forcing

---

## 7.3 Cylindrical momentum coupling

The important swirl terms are:

### Radial

\[
\frac{\partial u_r}{\partial t}
+
u_r\frac{\partial u_r}{\partial r}
+
u_z\frac{\partial u_r}{\partial z}
-
\frac{u_\theta^2}{r}
=
-\frac{1}{\rho}\frac{\partial p}{\partial r}
+\text{viscous terms}
+f_r.
\]

The term

\[
-\frac{u_\theta^2}{r}
\]

is responsible for the radial centrifugal-pressure balance.

### Azimuthal

\[
\frac{\partial u_\theta}{\partial t}
+
u_r\frac{\partial u_\theta}{\partial r}
+
u_z\frac{\partial u_\theta}{\partial z}
+
\frac{u_ru_\theta}{r}
=
\text{viscous terms}
+
f_\theta.
\]

### Axial

\[
\frac{\partial u_z}{\partial t}
+
u_r\frac{\partial u_z}{\partial r}
+
u_z\frac{\partial u_z}{\partial z}
=
-\frac{1}{\rho}\frac{\partial p}{\partial z}
+
\text{viscous terms}
-g
+f_z.
\]

---

# 8. Why the vortex depresses the surface

For a nearly steady rotating flow,

\[
\frac{\partial p}{\partial r}
\approx
\rho\frac{u_\theta^2}{r}.
\]

Thus pressure increases with radius and is lower near the axis.

At a free surface,

\[
p\approx p_{\mathrm{atm}}.
\]

The surface therefore moves downward near the low-pressure center.

A simplified free-surface slope relation is

\[
\boxed{
g\frac{d\eta}{dr}
\approx
\frac{u_\theta^2}{r}
}
\]

where

\[
z=\eta(r)
\]

is the free-surface profile.

This relation provides a useful diagnostic check for the numerical solution.

---

# 9. Two-phase representation with a Level Set

Define a Level Set function

\[
\phi(r,z,t).
\]

Convention:

\[
\phi<0:\text{ water},
\]

\[
\phi>0:\text{ air},
\]

\[
\phi=0:\text{ air–water interface}.
\]

The interface is advected by

\[
\boxed{
\frac{\partial\phi}{\partial t}
+
u_r\frac{\partial\phi}{\partial r}
+
u_z\frac{\partial\phi}{\partial z}
=0
}
\]

because the problem is axisymmetric.

---

## 9.1 Material properties

Use a smoothed Heaviside function \(H_\epsilon(\phi)\):

\[
\rho(\phi)
=
\rho_w
+
(\rho_a-\rho_w)H_\epsilon(\phi),
\]

\[
\mu(\phi)
=
\mu_w
+
(\mu_a-\mu_w)H_\epsilon(\phi).
\]

A common smooth transition width is

\[
\epsilon\approx1.5\Delta x.
\]

---

## 9.2 Surface tension

Unit normal:

\[
\mathbf n
=
\frac{\nabla\phi}
{|\nabla\phi|+\varepsilon_n}.
\]

Curvature:

\[
\kappa
=
\nabla\cdot\mathbf n.
\]

Continuum Surface Force form:

\[
\boxed{
\mathbf F_\sigma
=
\sigma
\kappa
\delta_\epsilon(\phi)
\mathbf n
}
\]

where

- \(\sigma\): water–air surface tension
- \(\delta_\epsilon\): smoothed delta function around the interface

---

## 9.3 Level Set reinitialization

Numerical advection distorts \(|\nabla\phi|=1\).

Periodically solve in pseudo-time \(\tau\):

\[
\frac{\partial\phi}{\partial\tau}
=
S(\phi_0)
\left(
1-|\nabla\phi|
\right)
\]

for several small pseudo-time iterations.

Do not reinitialize too aggressively because this can move the interface and cause volume loss.

---

# 10. Effective stir-bar forcing

A 2D axisymmetric solver cannot explicitly represent a 30 mm × 7 mm rod rotating in the horizontal plane.

Replace the rod with a smooth bottom forcing zone.

## 10.1 Forcing region

Approximate swept radial region:

\[
0\le r\lesssim R_m=15\ \mathrm{mm}.
\]

Approximate vertical region:

\[
0\le z\lesssim D_m=7\ \mathrm{mm}.
\]

Define a smooth mask

\[
0\le\chi(r,z)\le1.
\]

Example:

\[
\chi(r,z)
=
\frac12
\left[
1-\tanh
\left(
\frac{r-R_m}{\epsilon_f}
\right)
\right]
\,
\frac12
\left[
1-\tanh
\left(
\frac{z-D_m}{\epsilon_f}
\right)
\right].
\]

---

## 10.2 Azimuthal forcing law

Target solid-body speed:

\[
u_{\theta,\mathrm{target}}
=
\Omega_m r.
\]

Use relaxation forcing:

\[
\boxed{
f_\theta
=
\chi(r,z)
\frac{
u_{\theta,\mathrm{target}}-u_\theta
}{
\tau_s
}
}
\]

where

\[
\Omega_m=\frac{2\pi N}{60}.
\]

Parameter:

- \(\tau_s\): effective stirrer-to-fluid momentum-transfer timescale

Smaller \(\tau_s\):

- stronger coupling
- larger swirl
- deeper vortex

Larger \(\tau_s\):

- weaker coupling
- smaller swirl
- shallower vortex

### Calibration

\(\tau_s\) is **not known a priori**.

Recommended procedure:

1. choose one moderate RPM condition below air-core onset
2. measure experimental vortex depth
3. adjust \(\tau_s\) until simulated depth matches that one condition
4. freeze \(\tau_s\)
5. predict all remaining RPM and water-depth conditions

Do not recalibrate \(\tau_s\) for each RPM.

---

# 11. Fluid properties

Baseline at approximately room temperature:

| Property | Water | Air |
|---|---:|---:|
| Density \(\rho\) | \(998\ \mathrm{kg/m^3}\) | \(1.2\ \mathrm{kg/m^3}\) |
| Dynamic viscosity \(\mu\) | \(1.00\times10^{-3}\ \mathrm{Pa\,s}\) | \(1.8\times10^{-5}\ \mathrm{Pa\,s}\) |

Water–air surface tension:

\[
\sigma\approx0.072\ \mathrm{N/m}.
\]

Gravity:

\[
g=9.81\ \mathrm{m/s^2}.
\]

These should be updated for the actual measured liquid temperature.

If glycerol is added, update **all** of

\[
\rho,\qquad
\mu,\qquad
\sigma,
\]

not viscosity alone.

---

# 12. Boundary conditions

## Axis \(r=0\)

\[
u_r=0,
\]

\[
u_\theta=0,
\]

\[
\frac{\partial u_z}{\partial r}=0,
\]

\[
\frac{\partial p}{\partial r}=0.
\]

---

## Vessel wall \(r=R_v\)

No slip:

\[
u_r=u_\theta=u_z=0.
\]

---

## Bottom \(z=0\)

No slip:

\[
u_r=u_\theta=u_z=0.
\]

The stirrer effect is introduced through the volumetric forcing zone above the bottom.

---

## Top boundary

Use an atmospheric pressure opening:

\[
p=p_{\mathrm{atm}}.
\]

In gauge pressure,

\[
p=0.
\]

Use zero-gradient or open-boundary treatment for velocity so that air can adjust without artificial pressurization.

---

# 13. Initial condition

At

\[
t=0
\]

set

\[
\mathbf u=0.
\]

Water:

\[
z<H.
\]

Air:

\[
z>H.
\]

Level Set initialization:

\[
\phi(r,z,0)=z-H.
\]

The stirrer should preferably be ramped rather than switched instantaneously from 0 to full RPM.

Example smooth ramp:

\[
\Omega(t)
=
\Omega_{\mathrm{target}}
\left[
1-\exp\left(-\frac{t}{t_r}\right)
\right].
\]

Typical initial guess:

\[
t_r=0.2\text{–}0.5\ \mathrm{s}.
\]

If the experimental startup curve is measured, use that instead.

---

# 14. Numerical method

Recommended first implementation:

- staggered MAC-type grid
- finite differences / finite-volume-like fluxes
- fractional-step projection method
- sparse pressure Poisson solve
- Level Set interface tracking

---

## 14.1 Projection algorithm

For every time step:

### Step 1 — material properties

Compute

\[
\rho(\phi),\qquad\mu(\phi).
\]

### Step 2 — interface geometry

Compute

\[
\mathbf n,\qquad
\kappa,\qquad
\mathbf F_\sigma.
\]

### Step 3 — predictor velocity

Calculate provisional velocity

\[
\mathbf u^*
\]

without the new pressure.

Symbolically,

\[
\frac{\mathbf u^*-\mathbf u^n}{\Delta t}
=
-\mathbf u^n\cdot\nabla\mathbf u^n
+
\text{viscous}
+
\mathbf g
+
\frac{\mathbf F_\sigma}{\rho}
+
\mathbf F_{\mathrm{stir}}.
\]

### Step 4 — pressure Poisson equation

Enforce incompressibility through

\[
\boxed{
\nabla\cdot
\left(
\frac{1}{\rho}\nabla p^{n+1}
\right)
=
\frac{1}{\Delta t}
\nabla\cdot\mathbf u^*
}
\]

### Step 5 — velocity correction

\[
\boxed{
\mathbf u^{n+1}
=
\mathbf u^*
-
\Delta t
\frac{1}{\rho}
\nabla p^{n+1}
}
\]

### Step 6 — Level Set advection

\[
\phi^{n+1}
=
\phi^n
-
\Delta t\,
\mathbf u^{n+1}\cdot\nabla\phi.
\]

### Step 7 — reinitialization

Perform several Level Set reinitialization iterations if required.

### Step 8 — diagnostics

Measure:

- vortex depth
- air-core status
- mass/volume conservation
- maximum velocity
- CFL
- residuals

---

# 15. Grid

## First-resolution run

Recommended initial uniform grid:

\[
\Delta r=\Delta z=0.5\ \mathrm{mm}.
\]

For

\[
R_v=45\ \mathrm{mm},
\quad
H=50\ \mathrm{mm},
\quad
H_{\mathrm{air}}=30\ \mathrm{mm},
\]

this gives approximately

\[
N_r=90,
\qquad
N_z=160,
\]

or roughly \(1.4\times10^4\) cells.

This is small enough for Python.

---

## Convergence study

At minimum compare

\[
\Delta x=
1.0,\ 0.5,\ 0.25\ \mathrm{mm}
\]

for selected representative conditions.

Track convergence of:

\[
d_\infty
\]

and

\[
N_c.
\]

Suggested acceptance condition:

\[
\frac{|d_{\mathrm{fine}}-d_{\mathrm{medium}}|}
{d_{\mathrm{fine}}}
<5\%
\]

and similarly for \(N_c\).

---

# 16. Time step

Use an adaptive time step.

## Convective constraint

\[
\Delta t_{\mathrm{adv}}
\le
C_{\mathrm{CFL}}
\min
\left(
\frac{\Delta r}{|u_r|+\varepsilon},
\frac{\Delta z}{|u_z|+\varepsilon}
\right).
\]

Use initially

\[
C_{\mathrm{CFL}}\approx0.2\text{–}0.4.
\]

---

## Viscous constraint for explicit diffusion

\[
\Delta t_\nu
\lesssim
C_\nu
\frac{\rho\Delta x^2}{\mu}.
\]

---

## Capillary constraint

Surface tension can impose a stricter limit:

\[
\Delta t_\sigma
\lesssim
C_\sigma
\sqrt{
\frac{\rho\Delta x^3}{\sigma}
}.
\]

Use

\[
\boxed{
\Delta t
=
\min
(
\Delta t_{\mathrm{adv}},
\Delta t_\nu,
\Delta t_\sigma,
\Delta t_{\mathrm{forcing}}
)
}
\]

with

\[
\Delta t_{\mathrm{forcing}}
\ll\tau_s.
\]

For \(\Delta x=0.5\ \mathrm{mm}\), a practical first test range is approximately

\[
10^{-4}\text{–}5\times10^{-4}\ \mathrm{s},
\]

but the solver should determine the step adaptively.

---

# 17. Dimensionless parameters

Because definitions vary in the literature, always state the exact definition used.

Let the characteristic radius be the stir-bar half-length

\[
R_m=15\ \mathrm{mm}.
\]

---

## Rotational Reynolds number

\[
\boxed{
Re_\Omega
=
\frac{\rho\Omega R_m^2}{\mu}
}
\]

It compares rotational inertia with viscosity.

---

## Rotational Froude number

\[
\boxed{
Fr_\Omega
=
\frac{\Omega^2R_m}{g}
}
\]

It compares rotational/centrifugal acceleration with gravity.

---

## Weber number

\[
\boxed{
We_\Omega
=
\frac{\rho\Omega^2R_m^3}{\sigma}
}
\]

It compares rotational inertia with surface tension.

---

## Bond number

\[
\boxed{
Bo=
\frac{\rho gR_m^2}{\sigma}
}
\]

It compares gravity with surface tension.

---

## Geometrical ratios

Water-depth ratio:

\[
\boxed{
H^*=\frac{H}{R_m}
}
\]

Confinement ratio:

\[
\boxed{
C=\frac{R_m}{R_v}
}
\]

For the current geometry,

\[
C=\frac{15}{45}=0.333.
\]

Bar aspect ratio:

\[
\boxed{
AR=\frac{L_m}{D_m}
=
\frac{30}{7}
\approx4.29.
}
\]

---

# 18. RPM conversion

\[
\boxed{
\Omega=\frac{2\pi N}{60}
}
\]

where \(N\) is in rpm.

For the MSH-20D:

\[
80\le N\le1500\ \mathrm{rpm}.
\]

Thus

\[
8.38\lesssim\Omega\lesssim157.1\ \mathrm{rad/s}.
\]

At 1500 rpm, the nominal stir-bar tip speed is

\[
U_{\mathrm{tip}}
=
\Omega R_m
\approx
157.1\times0.015
\approx
2.36\ \mathrm{m/s}.
\]

> This uses actual stir-bar RPM.  
> The displayed stirrer setting should not automatically be assumed to equal actual bar RPM at high load.

---

# 19. Water-depth study

Recommended first sweep:

\[
H=
20,\ 30,\ 40,\ 50,\ 60\ \mathrm{mm}.
\]

If the 90 mm diameter is treated as an internal cylindrical diameter, approximate volumes are:

| \(H\) | approximate volume |
|---:|---:|
| 20 mm | 127 mL |
| 30 mm | 191 mL |
| 40 mm | 254 mL |
| 50 mm | 318 mL |
| 60 mm | 382 mL |

These are only geometric estimates.

Use **measured water height**, not beaker graduations, as the final simulation input.

---

# 20. RPM sweep

Do not initially run every 5 rpm.

Use a coarse sweep:

\[
N=
300,\ 500,\ 700,\ 900,\ 1100,\ 1300,\ 1500\ \mathrm{rpm}.
\]

For each \(H\), find the interval containing the transition.

Example:

```text
1100 rpm → no core
1300 rpm → core
```

Then test

```text
1200 rpm
```

and repeatedly bisect the interval.

Continue until the uncertainty is about the experimental setting resolution:

\[
\Delta N_c\sim5\text{–}10\ \mathrm{rpm}.
\]

---

# 21. Definition of vortex depth

Let the initial undisturbed water surface be

\[
z=H.
\]

Let the lowest point of the **top-connected central air region** be

\[
z_{\mathrm{tip}}.
\]

Then

\[
\boxed{
d(t)=H-z_{\mathrm{tip}}(t)
}
\]

and

\[
d_\infty
\]

is the long-time or time-averaged depth after transients have decayed.

To avoid selecting a wall meniscus, determine the tip within a central region such as

\[
r\le R_m.
\]

---

# 22. Objective definition of air-core formation

A visual decision such as “it looks connected” is not sufficiently reproducible.

Use a connectivity criterion.

### Binary air mask

\[
A(r,z)=
\begin{cases}
1,&\phi>0\\
0,&\phi\le0
\end{cases}
\]

Find the connected air component touching the top boundary.

Call it

\[
A_{\mathrm{top}}.
\]

Define an expanded stir-bar target region

\[
\mathcal B_\delta
\]

within approximately 1–2 grid cells of the effective stir-bar region.

The air core is geometrically connected when

\[
\boxed{
A_{\mathrm{top}}
\cap
\mathcal B_\delta
\neq\emptyset
}
\]

---

## Persistence criterion

To exclude momentary interface contact, require persistence for several stir-bar rotations.

For example:

\[
\boxed{
t_{\mathrm{connected}}
\ge
5T
}
\]

where

\[
T=\frac{60}{N}.
\]

Therefore

\[
t_{\mathrm{connected}}
\ge
\frac{300}{N}.
\]

This defines a reproducible critical RPM.

---

# 23. Air-core radius

At a given height \(z\), identify the central interval belonging to the top-connected air component.

If its radial extent is

\[
0\le r\le r_{\mathrm{air}}(z),
\]

then

\[
\boxed{
D_{\mathrm{air}}(z)
=
2r_{\mathrm{air}}(z).
}
\]

Store this profile after the core forms.

---

# 24. Primary simulation outputs

Every run should automatically save:

### Free-surface quantities

\[
d(t)
\]

\[
d_\infty
\]

\[
z_{\mathrm{tip}}(t)
\]

\[
r_{\mathrm{air}}(z,t)
\]

### Flow quantities

\[
u_r(r,z,t)
\]

\[
u_z(r,z,t)
\]

\[
u_\theta(r,z,t)
\]

\[
p(r,z,t)
\]

### Scalar diagnostics

- maximum \(u_\theta\)
- maximum downward \(u_z\)
- minimum central pressure
- water volume
- volume drift
- CFL number
- pressure-solver residual
- air-core state
- persistence time

---

# 25. Main plots

The first report should contain at least the following.

## Plot A

\[
\boxed{
d_\infty\ \text{vs.}\ N^2
}
\]

Tests the approximate quadratic scaling.

---

## Plot B

\[
\boxed{
N_c^2\ \text{vs.}\ H_{\mathrm{eff}}
}
\]

where

\[
H_{\mathrm{eff}}
=
H-z_{\mathrm{bar,top}}.
\]

---

## Plot C

\[
\boxed{
Fr_{\Omega,c}\ \text{vs.}\ H/R_m
}
\]

This forms a dimensionless critical-condition diagram.

---

## Plot D

\[
\boxed{
r_{\mathrm{air}}(z)
}
\]

for several values of RPM above \(N_c\).

---

## Plot E

Velocity field:

- \(u_\theta(r,z)\)
- streamlines in the \(r-z\) plane
- centerline \(u_z(z)\)

This is useful for identifying the central downward jet and toroidal recirculation.

---

# 26. Parameter studies

## Study A — RPM

Fixed:

- \(H\)
- viscosity
- stir-bar dimensions
- vessel dimensions

Vary:

\[
N.
\]

Outputs:

\[
d_\infty(N),\qquad N_c.
\]

---

## Study B — water depth

Vary:

\[
H=
20,\ 30,\ 40,\ 50,\ 60\ \mathrm{mm}.
\]

Output:

\[
N_c(H).
\]

---

## Study C — viscosity

Possible simulation values:

\[
\mu=
1,\ 2,\ 5,\ 10,\ 20\ \mathrm{mPa\,s}.
\]

When comparing to water–glycerol experiments, also change density and surface tension consistently.

---

## Study D — stir-bar size

Current:

\[
30\times7\ \mathrm{mm}.
\]

Possible length study:

\[
L_m=
20,\ 30,\ 40,\ 50\ \mathrm{mm}.
\]

The MSH-20D catalog states that bars up to 5 cm are usable.

---

## Study E — vessel diameter

Current nominal diameter:

\[
D_v=90\ \mathrm{mm}.
\]

If other vessels are available, test confinement through

\[
C=\frac{R_m}{R_v}.
\]

---

## Study F — hysteresis

Run both:

### ramp up

\[
N_1<N_2<\cdots
\]

and

### ramp down

\[
N_1>N_2>\cdots.
\]

Compare

\[
N_{\mathrm{form}}
\]

and

\[
N_{\mathrm{collapse}}.
\]

A difference indicates hysteresis.

---

# 27. Python implementation

Recommended packages:

```text
numpy
scipy
numba
matplotlib
pandas
scikit-image
pyyaml
tqdm
h5py
```

Optional:

```text
pyamg
taichi
```

---

## Package roles

### NumPy

- field arrays
- finite-difference operators
- vectorized diagnostics

### SciPy

- sparse matrices
- pressure Poisson solver
- iterative linear solvers

### Numba

- JIT acceleration of numerical kernels
- advection
- curvature
- forcing
- diagnostics

### scikit-image

- connected-component labeling
- air-core connectivity
- contour extraction

### Matplotlib

- free-surface plots
- velocity maps
- animation

### pandas

- parameter sweep summary
- CSV output

### h5py

- full field snapshots without enormous CSV files

### PyYAML

- experiment configuration files

### Taichi — optional

Use later if Python/Numba becomes too slow, especially for 3D particle simulations.

---

# 28. Suggested repository structure

```text
air-vortex/
│
├── README.md
├── pyproject.toml
├── requirements.txt
│
├── configs/
│   ├── baseline.yaml
│   ├── sweep_rpm.yaml
│   └── sweep_depth.yaml
│
├── src/
│   └── air_vortex/
│       ├── __init__.py
│       ├── config.py
│       ├── grid.py
│       ├── fields.py
│       ├── operators.py
│       ├── boundary.py
│       ├── properties.py
│       ├── forcing.py
│       ├── levelset.py
│       ├── surface_tension.py
│       ├── pressure.py
│       ├── timestep.py
│       ├── solver.py
│       ├── diagnostics.py
│       ├── connectivity.py
│       ├── sweep.py
│       └── plotting.py
│
├── scripts/
│   ├── run_single.py
│   ├── run_rpm_sweep.py
│   ├── run_depth_sweep.py
│   └── analyze_results.py
│
├── tests/
│   ├── test_operators.py
│   ├── test_pressure.py
│   ├── test_levelset.py
│   ├── test_hydrostatic.py
│   └── test_rotation.py
│
└── results/
    ├── raw/
    ├── figures/
    └── summary/
```

---

# 29. Example baseline configuration

```yaml
geometry:
  vessel_radius_m: 0.045
  vessel_height_m: 0.125

  water_height_m: 0.050
  air_height_m: 0.030

  stirbar_length_m: 0.030
  stirbar_diameter_m: 0.007
  stirbar_center_z_m: 0.0035

fluid:
  water_density: 998.0
  water_viscosity: 0.00100

  air_density: 1.20
  air_viscosity: 0.000018

  surface_tension: 0.072
  gravity: 9.81

stirrer:
  rpm: 900
  ramp_time_s: 0.3

  forcing_tau_s: 0.005
  forcing_smoothing_m: 0.001

grid:
  dr_m: 0.0005
  dz_m: 0.0005

time:
  t_end_s: 8.0
  cfl: 0.30
  dt_max_s: 0.0005

levelset:
  interface_width_cells: 1.5
  reinitialize_every: 5
  reinitialize_iterations: 3

air_core:
  contact_tolerance_cells: 2
  persistence_rotations: 5

output:
  save_every_s: 0.02
```

`forcing_tau_s` is only an initial numerical guess and must be calibrated.

---

# 30. Solver pseudocode

```python
initialize_grid()
initialize_velocity()
initialize_level_set()
initialize_pressure()

while t < t_end:

    omega = stirrer_ramp(t)

    rho, mu = material_properties(phi)

    normal = interface_normal(phi)
    curvature = interface_curvature(phi)
    f_sigma = surface_tension_force(phi, curvature)

    f_stir = stirrer_forcing(
        u_theta,
        omega,
        grid,
        geometry,
        tau_s,
    )

    dt = compute_stable_timestep(
        velocity,
        rho,
        mu,
        sigma,
        grid,
    )

    u_star = momentum_predictor(
        velocity,
        pressure,
        rho,
        mu,
        f_sigma,
        f_stir,
        gravity,
        dt,
    )

    pressure = solve_pressure_poisson(
        u_star,
        rho,
        dt,
    )

    velocity = pressure_projection(
        u_star,
        pressure,
        rho,
        dt,
    )

    phi = advect_level_set(
        phi,
        velocity,
        dt,
    )

    if step % reinit_interval == 0:
        phi = reinitialize_level_set(phi)

    diagnostics = analyze(
        phi,
        velocity,
        pressure,
    )

    save_if_needed()

    t += dt
```

---

# 31. Numerical verification tests

Before simulating the real vortex, pass simpler tests.

## Test 1 — hydrostatic water

No stirring:

\[
\Omega=0.
\]

Expected:

\[
\mathbf u\approx0,
\]

\[
p(z)\approx\rho g(H-z),
\]

and flat free surface.

---

## Test 2 — rigid-body swirl

Prescribe approximately

\[
u_\theta=\omega r.
\]

Expected free-surface shape:

\[
\eta(r)
=
\eta(0)
+
\frac{\omega^2r^2}{2g}.
\]

The solver should reproduce a paraboloid.

---

## Test 3 — static capillary interface

Use a known curved interface and verify the Laplace-pressure jump.

---

## Test 4 — water-volume conservation

Track

\[
V_w(t).
\]

Define relative drift:

\[
\epsilon_V
=
\frac{|V_w(t)-V_w(0)|}{V_w(0)}.
\]

Target initially:

\[
\epsilon_V<1\%.
\]

Preferably reduce below

\[
0.5\%.
\]

---

## Test 5 — grid convergence

Check representative values with multiple grid spacings.

---

## Test 6 — time-step convergence

Run the same case with

\[
\Delta t,\qquad
\frac{\Delta t}{2}
\]

and compare

\[
d_\infty
\]

and interface shape.

---

# 32. Calibration and validation strategy

The model should not be tuned separately for every experiment.

Recommended sequence:

### Calibration experiment

Choose one condition such as:

- water depth around 40–50 mm
- moderate RPM
- no air core
- stable visible vortex

Measure:

\[
d_{\mathrm{exp}}.
\]

Fit only the effective coupling parameter

\[
\tau_s.
\]

---

### Validation experiments

Without changing \(\tau_s\), compare:

\[
d(N)
\]

for other RPM values.

Then compare:

\[
N_c(H)
\]

for multiple water depths.

Finally compare:

\[
r_{\mathrm{air}}(z)
\]

above the critical RPM.

---

# 33. Actual stir-bar RPM measurement

The numerical model needs

\[
N_{\mathrm{bar}},
\]

not merely the stirrer display setting.

At high speed, magnetic slip or step-out may occur.

Recommended measurement:

1. mark one end of the stir bar
2. record high-speed video
3. extract angular position frame by frame
4. calculate

\[
N_{\mathrm{bar}}
=
\frac{60\,n_{\mathrm{rev}}}{\Delta t}.
\]

Store both:

```text
set_rpm
actual_bar_rpm
```

in the experimental dataset.

Use `actual_bar_rpm` for CFD input.

---

# 34. Level 2 — optional 3D SPH model

The axisymmetric model replaces the rod with an effective forcing region.

A 3D model is needed to investigate:

- the two ends of the rod
- non-axisymmetric wake
- vortex precession
- asymmetric air-core attachment
- rod-induced oscillation
- eccentric stirring

A practical Python route is a particle-based SPH solver implemented with NumPy/Numba or Taichi.

---

## 34.1 One-phase free-surface SPH

Initially simulate only water.

The empty region represents atmospheric air.

This can predict:

- surface depression
- air-core geometry as a void
- contact of the free surface with the rotating rod

It cannot accurately predict:

- air pressure dynamics
- bubble pinch-off
- two-phase entrainment

Those require a two-phase extension.

---

## 34.2 SPH density

A common formulation is

\[
\rho_i
=
\sum_j
m_j W_{ij}.
\]

---

## 34.3 Pressure

Weakly compressible SPH may use a Tait-type equation of state:

\[
p_i
=
\frac{c_0^2\rho_0}{\gamma}
\left[
\left(
\frac{\rho_i}{\rho_0}
\right)^\gamma
-1
\right].
\]

Choose \(c_0\) sufficiently large so that density fluctuations remain small.

---

## 34.4 Momentum

Generic SPH form:

\[
\frac{d\mathbf u_i}{dt}
=
-\sum_j
m_j
\left(
\frac{p_i}{\rho_i^2}
+
\frac{p_j}{\rho_j^2}
\right)
\nabla_i W_{ij}
+
\mathbf a_{\nu,i}
+
\mathbf g.
\]

---

## 34.5 Rotating rod

Represent the stir bar as rigid boundary particles.

For each boundary particle,

\[
\mathbf u_b
=
\boldsymbol\Omega\times
(\mathbf x_b-\mathbf x_c).
\]

Use the actual capsule-like geometry:

- overall length: 30 mm
- diameter: 7 mm

The rod rotates around the vertical vessel axis.

---

# 35. Why not start with 3D?

At a particle spacing of about

\[
1\ \mathrm{mm},
\]

a several-hundred-mL water volume already requires on the order of \(10^5\)–\(10^6\) particles depending on the simulated fill.

At

\[
0.5\ \mathrm{mm},
\]

particle count increases by approximately a factor of 8.

Therefore:

\[
\boxed{
\text{2D parameter sweep first}
\rightarrow
\text{3D validation second}
}
\]

is far more efficient.

---

# 36. Experimental parameter matrix

Recommended first campaign:

| Study | Parameter | Values |
|---|---|---|
| A | RPM | 300, 500, 700, 900, 1100, 1300, 1500 |
| B | \(H\) | 20, 30, 40, 50, 60 mm |
| C | \(\mu\) | 1, 2, 5, 10, 20 mPa·s |
| D | \(L_m\) | 20, 30, 40, 50 mm |
| E | RPM direction | ramp-up / ramp-down |

Do **A + B first**.

Do not run the full Cartesian product of all parameters initially.

---

# 37. Suggested run sequence

## Phase 1

Implement Level 0 reduced model.

Deliverables:

- estimated \(N_c\)
- RPM range recommendation
- dimensionless parameter table

---

## Phase 2

Implement hydrostatic 2D solver without swirl.

Deliverables:

- stable water–air interface
- correct hydrostatic pressure

---

## Phase 3

Add imposed solid-body swirl.

Deliverables:

- parabolic free surface
- validation against analytical solution

---

## Phase 4

Add bottom stirrer forcing.

Deliverables:

- \(u_\theta(r,z)\)
- central pressure depression
- vortex formation

---

## Phase 5

Add complete Level Set deformation and connectivity analysis.

Deliverables:

- \(d(t)\)
- \(d_\infty\)
- automatic air-core detection

---

## Phase 6

Run RPM sweep.

Deliverables:

\[
d_\infty(N)
\]

and

\[
N_c.
\]

---

## Phase 7

Run water-depth sweep.

Deliverable:

\[
N_c(H).
\]

---

## Phase 8

Validate experimentally.

Compare:

\[
d_{\mathrm{sim}}
\quad\text{vs.}\quad
d_{\mathrm{exp}}
\]

and

\[
N_{c,\mathrm{sim}}
\quad\text{vs.}\quad
N_{c,\mathrm{exp}}.
\]

---

## Phase 9

Run selected 3D simulations.

Use only representative conditions:

- clearly below threshold
- near threshold
- clearly above threshold

---

# 38. Key limitations

The Level 1 model deliberately simplifies the real system.

### Axisymmetry

The real 30 mm rod is not axisymmetric.

The model replaces it by an averaged torque source.

### Stir-bar coupling

The effective forcing parameter must be calibrated.

### Turbulence

At high RPM the real flow may be transitional or turbulent.

The initial solver may use molecular viscosity only.

Later options:

- effective eddy viscosity
- simple mixing-length closure
- selected 3D calculations

### Beaker spout

The real DURAN beaker has a spout and is therefore not perfectly axisymmetric.

Ignore it in Level 1.

### Exact beaker geometry

The nominal 90 mm diameter may differ from the actual internal diameter.

Measure the vessel.

### Heating

The MSH-20D is a hotplate stirrer, but the baseline model is isothermal.

If temperature is intentionally varied, use temperature-dependent

\[
\rho(T),\quad
\mu(T),\quad
\sigma(T).
\]

### Bubble breakup

The Level 1 goal is air-core onset and geometry.

Detailed bubble pinch-off and gas entrainment are outside the first model.

---

# 39. Main success criteria

The simulation is considered useful if it can reproduce all of the following qualitatively and at least approximately quantitatively:

1. vortex depth increases strongly with RPM
2. outer rotational flow exhibits vortex-like decay with radius
3. a central downward flow develops
4. free surface forms a funnel
5. a threshold RPM exists for contact with the stir bar
6. critical RPM increases with water depth
7. predictions remain consistent after one-point forcing calibration
8. mesh refinement does not substantially change \(N_c\)

The strongest final IYPT result would be a combined comparison of

\[
\boxed{
\text{theory}
\leftrightarrow
\text{Python CFD}
\leftrightarrow
\text{experiment}
}
\]

for

\[
d(N)
\]

and

\[
N_c(H).
\]

---

# 40. Recommended first baseline

Until the actual standard fill height is selected, use:

\[
H=50\ \mathrm{mm}
\]

with

\[
D_v=90\ \mathrm{mm},
\quad
L_m=30\ \mathrm{mm},
\quad
D_m=7\ \mathrm{mm}.
\]

Initial RPM cases:

\[
N=
500,\ 700,\ 900,\ 1100,\ 1300,\ 1500\ \mathrm{rpm}.
\]

Initial grid:

\[
\Delta r=\Delta z=0.5\ \mathrm{mm}.
\]

Initial fluid:

- water + air
- room-temperature properties
- surface tension enabled

Initial simulation duration:

\[
5\text{–}10\ \mathrm{s}
\]

or until the vortex-depth signal becomes statistically steady.

---

# 41. Experimental data schema

Recommended CSV format:

```text
run_id
water_height_mm
water_volume_ml
water_temperature_C
set_rpm
actual_rpm
stirbar_length_mm
stirbar_diameter_mm
vessel_internal_diameter_mm
vortex_depth_mm
air_core_formed
air_core_min_diameter_mm
formation_time_s
collapse_rpm
notes
```

This allows direct comparison between experiment and numerical runs.

---

# 42. References and examples

Useful starting literature and software examples:

1. **Halász et al. — Vortex flow generated by a magnetic stirrer**  
   American Journal of Physics (2007)  
   DOI: `10.1119/1.2772287`

2. **Mahmud et al. — CFD / VOF study of free-surface flow generated by a magnetic stirrer**  
   Chemical Engineering Science (2009)  
   Relevant for magnetic-stirrer CFD and free-surface modeling.

3. **Vortex formation and subsequent air entrainment by a rotating submerged disk**  
   Industrial & Engineering Chemistry Research (2018)  
   DOI: `10.1021/acs.iecr.8b00379`

4. **COMSOL Free Surface Mixer — Level Set example**  
   Useful as a reference for rotating-domain, free-surface, and time-dependent vortex modeling.

5. **ANSYS Fluent transient stirred-tank VOF example**  
   Useful as a reference for VOF + rotating/sliding-mesh formulation.

These references are conceptual guides only. The proposed implementation in this repository is intended to be a custom Python numerical solver rather than a GUI CFD workflow.

---

# 43. Immediate next implementation task

Implement the following in order:

```text
1. baseline YAML config
2. axisymmetric grid
3. differential operators
4. hydrostatic pressure test
5. projection solver
6. swirl equation
7. stirrer forcing
8. Level Set advection
9. surface tension
10. air-core connectivity detector
11. RPM sweep
12. water-depth sweep
```

The first milestone should **not** be an air vortex.

The first milestone should be:

\[
\boxed{
\text{stationary water remains stationary and conserves volume}
}
\]

The second milestone should be:

\[
\boxed{
\text{imposed solid-body rotation reproduces the analytical parabolic surface}
}
\]

Only after these validation tests pass should the magnetic-stirrer forcing be trusted.

---

# 44. Result Visualization

A post-processing layer that turns saved simulation results into figures and
videos, without ever re-running the solver. It is purely an I/O and
plotting layer on top of the existing `Fields` / `Config` / `Grid` objects
and the existing `diagnostics.py` / `connectivity.py` definitions — it does
not change solver physics.

## 44.1 Run directory layout

`scripts/run_single.py` now writes each run to `results/<run_id>/`:

```text
results/<run_id>/
├── config.yaml           # full Config (README section 29 schema)
├── metadata.json         # kind, rpm, water_height_mm, grid shape, ...
├── diagnostics.csv       # one row per SOLVER step (section 24)
├── fields/
│   ├── frame_000000.npz  # one row per OUTPUT SAMPLE (output.save_every_s),
│   ├── frame_000001.npz  # NOT one per solver step
│   ├── ...
│   └── index.csv         # index, t, step, path
├── figures/
│   ├── png/
│   └── pdf/
└── videos/
```

Solver dt (~1e-4 s) and output sampling interval are intentionally
decoupled: a frame is written only when `t >= next_save_t`, advancing
`next_save_t` by `output.save_every_s` — exactly the field already present
in the section 29 config schema.

## 44.2 Figures

| # | Script | What it shows |
|---|---|---|
| 1 | `render_validation.py` (kind=hydrostatic) | still-water pressure vs. analytic hydrostatic profile, error, max\|u\|, volume drift |
| 2 | `render_validation.py` (kind=solid_body_rotation) | dp/dr vs. analytic centrifugal balance; eta(r) vs. analytic parabola |
| 3 | `render_run.py` | phi=0 interface at several times, one run |
| 4 | `render_run.py` | gauge pressure field + interface + stirrer region + vortex tip |
| 5 | `render_run.py` | meridional (r-z) flow: speed background + streamlines |
| 6 | `render_run.py` | u_theta(r,z) contour + u_theta(r) profiles (reference r / 1/r curves only, never asserted) |
| 7 | `render_sweep.py` | d(t) for several RPM runs |
| 8 | `render_sweep.py` | d_infinity vs. N^2 (+ dimensionless d/R_m vs. Fr_Omega), with linear-regression fit and R² |
| 9 | `render_sweep.py` | N_c^2 vs. H_eff (+ dimensionless Fr_Omega,c vs. H/R_m), with fit |
| 10 | (plotting.py `plot_regime_map`) | (Fr_Omega, H/R_m) regime map: no core / transient contact / stable core |

Every figure is saved as both `png` (300 dpi) and `pdf` (vector) under the
run's `figures/` directory. Figures 7-10 need real sweep data (from
`scripts/run_rpm_sweep.py` / `run_depth_sweep.py`, or a directory of
run-directories); the plotting functions are unit-tested with synthetic
data but have not yet been exercised on a real multi-RPM sweep (see 44.5).

## 44.3 Videos

`scripts/render_video.py --run <dir> --type {interface,pressure_velocity,air_core}`
produces:

- **interface**: water/air regions, phi=0 interface, stirrer region, vortex
  tip, live "stable air core" tag (from the same connectivity + persistence
  logic as the solver).
- **pressure_velocity**: pressure (left) + meridional flow (right), with
  fixed, whole-run, percentile-based color limits computed once up front.
- **air_core**: zoomed-in close-up (default r ∈ [0, 20mm]) of the
  connectivity criterion itself: top-connected air component vs. the
  stirrer target region B_delta.

If `ffmpeg` is available, a real `.mp4` is written via
`matplotlib.animation.FFMpegWriter`. If not, a PNG frame sequence is
written instead with a printed warning (never a silent no-op) and a ready
to use `ffmpeg -framerate ... -i frame_%04d.png ...` command line.

## 44.4 CLI examples

```bash
python scripts/run_single.py --config configs/validation_hydrostatic.yaml \
    --kind hydrostatic --run-id validation_hydrostatic
python scripts/render_validation.py --run results/validation_hydrostatic

python scripts/run_single.py --config configs/validation_solid_body.yaml \
    --swirl-mode prescribed --kind solid_body_rotation --run-id validation_solid_body
python scripts/render_validation.py --run results/validation_solid_body

python scripts/run_single.py --config configs/baseline.yaml --run-id run_H050_RPM1200 --rpm 1200
python scripts/render_run.py --run results/run_H050_RPM1200
python scripts/render_video.py --run results/run_H050_RPM1200 --type interface
python scripts/render_video.py --run results/run_H050_RPM1200 --type pressure_velocity
python scripts/render_video.py --run results/run_H050_RPM1200 --type air_core

python scripts/render_sweep.py --results results/rpm_sweep
python scripts/render_sweep.py --results results/depth_sweep
```

## 44.5 Interpretation caveats

- **A short run is a pipeline/solver sanity check, not a physical result.**
  The 0.02 s baseline smoke run (and the small validation/demo runs used to
  exercise this pipeline) only prove the code runs and the numbers are
  self-consistent. Vortex evolution, air-core onset, N_c, and d_infinity
  must only be read off long runs that have reached (or averaged over) a
  statistically steady state.
- Figure 2's two panels can legitimately disagree at short times: dp/dr
  converges to the analytic centrifugal balance quickly (a single pressure
  solve), but eta(r) only converges once the free surface has had time to
  relax via advection (many gravity-wave periods) — a gap between the two
  panels is expected physics at short t, not a solver bug.
- Diagnostics (vortex tip, vortex depth, air-core connected, stable air
  core, persistence time, water volume) are always read from
  `diagnostics.py` / `connectivity.py`, never recomputed ad hoc in a
  plotting function, so a figure's numbers cannot drift from the solver's
  own definitions.
- Do not overclaim from a figure: u_theta ∝ 1/r, d ∝ N², and N_c² ∝ H are
  hypotheses/scalings to test against data (Figures 6, 8, 9 draw them only
  as reference/fit lines with a reported R², never as an assumed law), and
  the axisymmetric model does not claim to fully reproduce the real
  30 mm × 7 mm rod's non-axisymmetric geometry (README section 38).

---

# 45. Performance

The pressure Poisson solve (README section 14.1 step 4) is, by a wide
margin, the dominant per-step cost (confirmed by profiling, not assumed --
see below). This section documents what was optimized, how it was
verified not to change any result, and what was tried but did not help.

## 45.1 Methodology: profile first, then a regression benchmark

`scripts/benchmark_solver.py` runs a fixed, deterministic scenario
(`configs/benchmark.yaml`: same dr/dz as the physical baseline but a
smaller vessel, ~2880 cells, `swirl_mode="forced"`) for a fixed number of
steps, and can:

- `--profile`: run under `cProfile` to show where time actually goes.
- `--save-golden PATH`: save the final field state as a reference.
- `--compare-golden PATH`: rerun and diff every field against a saved
  reference (max abs/relative difference per field).

Before touching any code, a golden reference was captured with the
original (unoptimized) implementation
(`tests/fixtures/regression_golden.npz`). Every optimization below was
verified against it before being kept, and `tests/test_regression_benchmark.py`
reruns this comparison automatically as part of the test suite so a future
change can't silently drift the physics while "just" refactoring.

## 45.2 What was profiled and fixed

Profiling `configs/benchmark.yaml` showed the per-cell Python loop that
rebuilt the pressure matrix from scratch every step -- not the linear
solve itself -- was **~73% of total step time** (matrix assembly 0.306s
vs. `spsolve` 0.079s, out of 0.417s for 15 steps).

**Fix**: `pressure.build_pressure_matrix` now computes the same five
stencil coefficients (west/east/south/north/diagonal) as full `(Nr, Nz)`
numpy arrays in one vectorized pass (`_pressure_matrix_coefficients`) and
assembles the sparse matrix via `scipy.sparse.diags` at fixed offsets
(`-Nz, -1, 0, 1, Nz` -- the natural row-major 5-point-stencil structure),
instead of a `for i: for j:` loop with per-cell `list.append`. The
coefficients and their placement in the matrix are unchanged; this is
purely how they get assembled. Regression check: **bit-for-bit identical**
output (`max_abs_diff = 0.0` on every field).

**Fix**: `spsolve`'s column ordering was switched from SuperLU's default
`COLAMD` to `MMD_AT_PLUS_A`, which measured ~25-30% faster for this
matrix's structure with no change to the linear system solved (still an
exact direct solve; differences vs. the golden reference are pure
floating-point round-off from a different elimination order, ~1e-13 to
1e-17 in absolute terms).

## 45.3 Measured speedup

| grid | before | after | speedup |
|---|---:|---:|---:|
| benchmark.yaml, 2880 cells | 13.56 ms/step | 5.60-5.75 ms/step | ~2.4x |
| baseline.yaml, 14400 cells | ~238 ms/step (original smoke test) | 28.3 ms/step | ~8.4x |

## 45.4 What was tried and did NOT help (and why)

After the assembly fix, `spsolve` itself became the dominant cost again
(~78-83% of total). Direct sparse LU for a 2D 5-point-stencil matrix is
not asymptotically cheap, so an iterative solver looked promising. It was
implemented and benchmarked (`pressure.solve_pressure_poisson(method="cg")`,
`pressure._build_spd_system`) but, at the grid sizes tested so far,
**did not beat the optimized direct solve**:

- The natural system is *not* symmetric (the cylindrical `1/r` weighting
  makes row `i` and row `i-1`'s coupling coefficients differ). Multiplying
  each equation by its cell weight `r_c[i]` symmetrizes it exactly (a
  standard finite-volume row-scaling) so `cg` applies -- but that same
  `r_c[i]` weighting gives the matrix a wide dynamic range of row
  magnitudes (axis rows vs. wall rows), which badly hurts simple
  preconditioners:
  - **No preconditioner**: did not converge in 2000 iterations.
  - **Jacobi**: converged, but needed ~260-275 iterations at 2880 cells --
    slower than direct.
  - **ILU** (`scipy.sparse.linalg.spilu`, default drop tolerance): did
    *not* converge at all (the wide coefficient range defeats a
    magnitude-based drop tolerance).
  - **Algebraic multigrid** (`pyamg`, optional dependency): converged
    cleanly in ~20-22 iterations at both 2880 and 14400 cells, matching
    the direct solve to ~1e-12 -- but AMG hierarchy setup cost
    (10-60 ms) plus the CG solve itself was still slower in total than
    the (now-optimized) direct solve at these sizes.
- Conclusion: for the ~3000-15000-cell 2D grids used so far, direct sparse
  LU is genuinely competitive (2D fill-in is not catastrophic), and the
  iterative path's overhead isn't recovered. `method="cg"` is kept,
  AMG-preconditioned when `pyamg` is installed, as a documented option for
  much finer future grids (README section 15's convergence study, e.g.
  Δx=0.25mm) where that balance may flip -- it is **not** the current
  default (`Solver.pressure_method` stays `"direct"`), and it always falls
  back to the direct solve rather than returning an under-converged result
  if it fails to reach tolerance.
- Also tried and not pursued further: `permc_spec="NATURAL"` (naively
  exploiting the row-major banded structure) was ~10x *slower* than
  `COLAMD`/`MMD_AT_PLUS_A` -- fill-reducing reordering matters more than
  raw bandwidth here.

## 45.5 Remaining bottleneck and what real runs still need

Even after these fixes, `spsolve` is still ~78% of per-step time at
baseline scale, and solver dt remains capillary-constrained to roughly
2e-5 s (README section 16) -- an 8-second physical run at baseline
resolution is still on the order of hours of wall-clock time, not minutes.
Getting a full baseline run (README section 40) into a practical iteration
loop needs more than this pass covered: candidates include a coarser
exploratory grid first (the convergence-study workflow README section 15
already recommends), or a fundamentally different pressure-solve strategy
at larger grid sizes (AMG becomes attractive there per 45.4). Both are
follow-up work, not part of this optimization pass.

---

# 46. Pre-physical-simulation validation status (superseded by section 47)

This section is the honest current answer to "can this numerical model be
used to generate real Air Vortex data yet" (not "does the code run").
Run `python scripts/check_validation_gate.py --production-config
<pilot config>` for the up-to-date, automatically-checked version of this.

**This section's specific numbers (NRMSE 0.588, volume drift 6.5%,
grid-quantized vortex depth) are the PRE-improvement baseline.** Section 47
documents what changed (2nd-order Level Set advection, optional volume
correction, sub-grid tip interpolation) and the current numbers -- read
that section for the up-to-date status; this one is kept for the before/
after record.

## 46.1 What is validated

- **Hydrostatic equilibrium** (README section 43 Milestone 1 / Test 1):
  passes. Still water stays still, conserves volume, and matches the
  analytic hydrostatic pressure profile.
- **Solid-body-rotation radial pressure balance** (README section 43
  Milestone 2 / Test 2, dp/dr = rho * Omega^2 * r): passes.
- **Two real bugs were found and fixed** while producing the long
  solid-body validation run (46.2): a near-axis curvature miscalculation
  in the surface-tension force (`operators.center_grad_r` was using the
  wrong stencil at r=0, violating the r=0 mirror-symmetry that an
  axisymmetric field must satisfy), and a boundary-condition conflict at
  the open top face that discarded the pressure projection's correct,
  divergence-consistent velocity there. Both are described with full
  diagnosis in git history / inline comments (`operators.py`,
  `boundary.py`) and covered by regression tests.

## 46.2 What is NOT yet validated -- known, diagnosed, open limitation

**Solid-body free-surface shape does not yet converge to the analytic
parabola over a long run.** dp/dr converges quickly (Test 2 passes), but
`eta(r)` does not: `scripts/validate_solid_body_long.py` shows normalized
RMSE ~0.59 against `eta(r) = eta(0) + Omega^2 r^2 / (2g)` over a
0.15-0.4s window at Omega~10.5 rad/s.

Diagnosis (per README section 31's "diagnose before changing the solver"
instruction, isolated one factor at a time):

- NOT the pressure-interface coupling or boundary conditions (both fixed
  and verified separately, 46.1).
- NOT reinitialization frequency: disabling reinitialization entirely
  changes total volume drift by only ~2 percentage points out of the
  ~7-30% observed over 0.5-2s runs.
- **IS the level-set advection scheme itself**: `levelset.advect_level_set`
  uses first-order upwind differencing, which is numerically diffusive.
  Water volume drifts monotonically upward over a long run (the free
  surface visibly rises well above its correct level once drift exceeds a
  few percent -- see `solid_body_center_depression_convergence.png` in any
  `validation_solid_body_long*` run directory) instead of settling near the
  analytic level. This is a genuine, unfixed numerical-accuracy limitation,
  not a hidden or reinterpreted "pass."
- A secondary, related finding: `diagnostics.find_tip_z` (and therefore
  `vortex_depth`) reads the *discrete* `grid.z_f[j_min]` face rather than
  interpolating the phi=0 crossing (the way `diagnostics.free_surface_height`
  already does). `scripts/run_grid_convergence.py` shows this makes
  reported vortex depth jump by almost exactly one grid cell between
  resolutions (3.0mm -> 1.5mm -> -3.0mm at dx=3.0/1.5/0.75mm) instead of
  converging smoothly, and the same quantization made a calibration-search
  self-test (46.4) noisy/non-monotonic. Not fixed in this pass -- flagged
  as a concrete, well-localized follow-up (reuse the interpolation
  `free_surface_height` already implements).

**Practical implication**: production vortex-depth predictions (pilot
sweeps, calibration) inherit both of these limitations until they are
addressed. The validation gate (section 46.5) fails "solid-body free
surface" and "volume conservation" for exactly this reason -- on purpose.

## 46.3 Time-step and grid convergence

`scripts/run_timestep_convergence.py` and `scripts/run_grid_convergence.py`
implement and were run against the small validation geometry (not the full
50mm-water baseline, which is impractically slow per section 45.5).
Results in `results/convergence/`. Two honest caveats found while running
them, not smoothed over:

- The time-step sweep's `dt_max*0.75` and `dt_max` (baseline) settings
  produced *identical* results (0.0% difference) because neither was
  actually the binding constraint -- the adaptive time-stepper's own
  capillary/viscous limit was already smaller than both. Only `dt_max/2`
  actually changed the effective dt, and did so by a modest amount
  (interface RMSE 0.05mm, volume-drift change ~0.03 percentage points).
  This is useful signal (reasonable time-step independence in the tested
  range) but the sweep as configured doesn't test three genuinely distinct
  time steps -- a follow-up should pick dt_max values validated to actually
  bind.
- The grid sweep's `d_final` values are corrupted by the discretization
  quantization described in 46.2 and should not be read as converging;
  `volume_drift` (2.9% -> 0.9% -> 0.6% coarse to fine) is the more
  trustworthy convergence signal from that run, and is consistent with a
  first-order-accurate (in dx) numerical-diffusion source.

## 46.4 Calibration framework

`scripts/calibrate_forcing.py` implements a bracketed, log-space
root-search for `forcing_tau_s` against one real (measured RPM, measured
depth) point, per README section 10.2 ("do not recalibrate tau_s for each
RPM" -- this script only ever fits one point per invocation and expects
that value reused unchanged elsewhere). **No real measured depth exists
for this project yet**, so no `results/calibration/calibrated_forcing.yaml`
has been produced, and none should be fabricated.

The script and search algorithm were verified with a self-consistent
"twin experiment" (simulate at a known tau_s, feed that simulation's own
output back in as the "measured" target, check the search recovers it) --
explicitly a software test, not a calibration result, and its output
artifact was deleted afterward rather than left in `results/`. That test
also surfaced the find_tip_z quantization issue (46.2) making the search
noisier than it should be.

## 46.5 Production readiness

`stirrer.calibrated: false` is set in every `configs/pilot_H050_RPM*.yaml`
file; `scripts/check_validation_gate.py` currently reports
`PRODUCTION READY: NO`, blocked on: solid-body free surface, volume
conservation, and forcing calibration (46.2, 46.4). The four pilot configs
exist (RPM 500/900/1200/1500 at H=50mm, README section 40 geometry) but
have deliberately not been run as production sweeps -- only as the
software-level checks described above.

## 46.6 Turbulence modeling limitation

The current Level 1 axisymmetric solver uses molecular viscosity only
(README section 38) and does not include a turbulence closure. At the
high end of the RPM range (up to 1500 rpm, README section 18), the real
flow may be transitional or turbulent, which this solver does not
represent. Low/moderate-RPM trends may still be useful once the section
46.2 and 46.4 limitations are addressed, but a systematic discrepancy that
grows with RPM should be interpreted as a turbulence/effective-viscosity/
3D-effects question (README section 38), not necessarily a stirrer-forcing
calibration problem. No turbulence model has been added in this pass --
that remains explicitly out of scope until the more fundamental
free-surface and calibration gaps above are closed.

---

# 47. Level Set advection accuracy, volume conservation, and sub-grid vortex-tip extraction

Follow-up to section 46's diagnosis. Scope: (A) Level Set advection
accuracy, (B) volume conservation, (C) sub-grid vortex-tip extraction,
(D) re-passing the solid-body free-surface validation. Forcing calibration,
pilot RPM runs, and a turbulence model are explicitly NOT part of this pass.

## 47.1 What changed

- **Selectable Level Set advection** (`config.levelset.advection_scheme` /
  `time_integrator` / `limiter`): the original `upwind1`+`euler` scheme is
  kept byte-for-byte (default, and what the regression golden fixture still
  validates against) alongside a new `muscl2`+`ssprk2` scheme -- a
  conservative, flux-form MUSCL-TVD reconstruction (MC or minmod limiter)
  with SSP-RK2 time integration, velocity held fixed across the RK
  sub-stages (`levelset.py`).
- **Optional global volume correction** (`config.levelset.volume_correction`,
  default OFF): a uniform phi shift, found by 1D root-search
  (`scipy.optimize.brentq`), that restores the water volume to its t=0
  value every `every_n_steps` steps. Never reshapes the interface locally
  (`volume_correction.py`).
- **Sub-grid vortex-tip interpolation**: `diagnostics.find_tip_z` now
  linearly interpolates the phi=0 crossing (sharing the same helper as
  `free_surface_height`) instead of returning the raw grid face -- fixes
  the grid-quantized vortex depth found in section 46.2.
  `tests/test_diagnostics_and_connectivity.py` verifies this recovers a
  known off-grid interface position to <1e-9 m at three different grid
  spacings.
- **Volume-consistent analytic free surface**: the parabola
  `eta(r) = C + Omega^2 r^2/(2g)` compared against is now anchored by
  `diagnostics.volume_consistent_parabola`, which picks C so the analytic
  surface encloses exactly the *initial* water volume -- not fit to the
  simulation's own (possibly volume-drifted) center height, which would
  have let a drifted case grade its own homework.

## 47.2 Case A-D comparison (`scripts/validate_solid_body_long.py`)

0.4s run, `configs/validation_solid_body.yaml`, RPM=100 (Omega=10.47 rad/s),
averaging window [0.15, 0.4]s:

| Case | scheme+integrator | correction | volume drift (final) | NRMSE | center depression error | ms/step |
|---|---|---|---:|---:|---:|---:|
| A | upwind1+Euler | OFF | 6.48% | 0.302 | 2.50 mm | 5.27 |
| B | MUSCL2+SSPRK2 | OFF | 1.66% | 0.184 | 1.88 mm | 5.80 |
| C | upwind1+Euler | ON | 0.46% | 0.221 | 2.35 mm | ~5.3\* |
| D | MUSCL2+SSPRK2 | ON | **-0.20%** | 0.198 | 1.81 mm | ~5.8\* |

\*correction adds negligible per-step cost (one `brentq` call every 10
steps); not separately re-measured.

Full CSVs/figures: `results/validation_solid_body_long/`
(`case_comparison_summary.csv`, per-case time-averaged free-surface
figures, `case_comparison_timeseries.png` for center-depression(t),
volume_error(t), interface_RMSE(t)).

## 47.3 Extended-duration finding (not in the table above)

Case B run to t=1.2s: volume drift grew to **-8.17%** (MUSCL2 alone slows
but does not stop long-run drift) and NRMSE worsened to 0.438. Case D
(correction ON) run to t=1.2s: volume drift stayed at 0.10%, but **NRMSE
was 0.207 -- essentially unchanged from the 0.4s value (0.198)**.

The `center_depression(t)` time series (`case_comparison_timeseries.png`)
shows why: B/C/D all show a persistent, only mildly-damped oscillation in
eta(0) (period ~0.15-0.2s) that has not decayed by t=1.2s. Given water's
molecular viscosity, the pure-diffusion damping timescale for this
geometry is ~R_v^2/nu ~ 500s -- so this oscillation is not expected to
damp out on any timescale reachable by more simulated time at reasonable
cost. **The dominant remaining error source is this persistent oscillation,
not residual numerical diffusion** -- MUSCL2 + volume correction have
already brought volume drift and shape RMSE close to their practical floor
for an *instantaneous or short-window* comparison against a *static*
analytic equilibrium.

## 47.4 Grid and time-step convergence (re-run)

Both re-run with MUSCL2+SSPRK2, `configs/validation_solid_body.yaml`,
t=0.15s (still within the fast initial transient -- see caveat below).

**Grid** (`scripts/run_grid_convergence.py`):

| dx | d_final | interface RMSE vs. finest | volume drift |
|---:|---:|---:|---:|
| 3.00 mm | 1.656 mm | 3.559 mm | 2.86% |
| 1.50 mm | 2.278 mm | 3.136 mm | 1.20% |
| 0.75 mm | 4.439 mm | 0.000 mm (reference) | 2.41% |

**Timestep** (`scripts/run_timestep_convergence.py`; the natural/binding dt
was measured first, ~1.19e-4 s -- the config's own `dt_max_s` never
actually bound, same issue as section 46.3, now fixed at the source):

| mean dt | d_final | interface RMSE vs. finest | volume drift |
|---:|---:|---:|---:|
| 1.186e-4 s | 2.278 mm | 0.606 mm | 1.20% |
| 5.929e-5 s | 1.667 mm | 0.602 mm | 0.88% |
| 2.965e-5 s | 2.372 mm | 0.000 mm (reference) | 3.09% |

**Caveat, reported rather than hidden**: `d_final` and `volume_drift` do
NOT converge monotonically in either table. `interface_rmse_vs_finest`
does (well-behaved, decreasing toward the finest resolution). This is the
same phenomenon as 47.3: at t=0.15s the interface is mid-way through its
first oscillation swing, so a single-instant point comparison (d_final) is
sensitive to exactly which phase of that swing each resolution/timestep
happens to land on, independent of whether the underlying numerics have
converged. `find_tip_z`'s quantization (section 46.2) is confirmed fixed
(d_final varies smoothly across resolutions -- 1.656/2.278/4.439mm --
instead of jumping in exact multiples of dx as before); the remaining
non-monotonicity is the oscillation-phase issue, not requantization.

## 47.5 Answers to the required questions

1. **Numerical diffusion actually reduced?** Yes: volume drift at t=0.4s
   drops from 6.48% (Case A) to 1.66% (Case B, accuracy improvement alone,
   no correction) -- a ~4x reduction from advection accuracy alone.
2. **Volume drift < 1%?** Only with correction ON (Case C: 0.46%, Case D:
   -0.20%). MUSCL2 alone (Case B) does not reach it by t=0.4s and drifts to
   -8.17% by t=1.2s -- correction is currently necessary for long runs, not
   just nice-to-have.
3. **Reproduces the solid-body analytical surface?** Partially. NRMSE
   improved from 0.588 to 0.184-0.221 across cases (a real, ~2.7-3.2x
   improvement) but the recommended <0.10 target is NOT met by any case.
   Per section 47.3, the shortfall is attributed to a persistent,
   weakly-damped oscillation rather than residual advection error.
4. **find_tip_z quantization resolved?** Yes -- verified to <1e-9m at three
   grid spacings on a synthetic known-position test
   (`test_find_tip_z_subgrid_interpolation_is_grid_independent`), and
   `d_final` no longer jumps in exact dx multiples in the grid-convergence
   run.
5. **Good enough with correction OFF?** Not for volume drift over long
   runs (see Q2); shape accuracy (NRMSE) is close to as good as WITH
   correction (0.184 vs 0.198-0.221), so correction's main value here is
   volume conservation, not shape accuracy.
6. **Is correction needed?** Yes, for volume drift specifically, given
   current advection accuracy and the durations of interest. It is NOT a
   substitute for accurate advection (Case B's shape RMSE is actually
   slightly better than Case D's -- correction does not fix shape error,
   consistent with it being a volume-only, non-local-reshaping operation
   by design).
7. **Is WENO/CLSVOF needed next?** Not clearly, given the diagnosis in
   47.3: the dominant remaining gap is a physical/numerical oscillation
   persistence issue, not advection-scheme order. A higher-order scheme
   would not obviously fix an underdamped free-surface response. Candidates
   worth investigating instead, in rough priority order: (a) whether a
   physically-motivated damping mechanism (e.g. numerical or explicit
   Rayleigh damping, or resolving the real viscous/Ekman boundary layer
   better) is what's actually missing, before reaching for a fancier
   conservative-transport scheme; (b) only if advection error is still
   implicated after that, WENO5 or a conservative/CLSVOF-style level set.
8. **Ready for forcing calibration?** Closer, not fully: divergence,
   volume conservation (with correction), grid/timestep-convergence
   infrastructure, and sub-grid tip extraction all now pass. The
   free-surface NRMSE gate still fails its recommended threshold, and that
   metric feeds directly into what calibrate_forcing.py optimizes against
   (vortex depth) -- so calibration run against noisy/oscillating depth
   values would itself be noisy. Recommend investigating 47.3 before
   spending real experimental measurements on calibration.
9. **Validation gate status**: `all pytest tests`, `hydrostatic`,
   `solid-body pressure`, `volume conservation`, `divergence`,
   `sub-grid tip extraction`, `timestep convergence`, `grid convergence`
   all **PASS**. `solid-body free surface` **FAILS** (NRMSE 0.198 vs. 0.10
   threshold, best case). `forcing calibration` **FAILS** (no real
   measured data, as required). **PRODUCTION READY: NO** -- as expected
   and required (README section 18 safeguards), since forcing calibration
   alone already blocks it regardless of the numerical-transport gates.

---

# 48. Exact-equilibrium preservation vs. flat-start transient (these are different questions)

Follow-up to section 47, prompted by re-examining whether the ~0.15-0.20
NRMSE found there was really "transient oscillation hasn't settled yet."
**It is not, or not only that.** This section separates two genuinely
different numerical-verification questions that section 47 (and the
original solid-body validation) had implicitly conflated:

- "Does a flat surface spin up into the analytic parabola?" (a transient
  physics question, confounded by startup excitation and weak damping)
- "If the solver is handed the EXACT analytical equilibrium as its initial
  condition, does it stay there?" (a pure numerical-verification question)

## 48.1 Surface tension was active in the "parabola" comparison

`configs/validation_solid_body.yaml` has `surface_tension: 0.072` (the
physical water-air value) -- section 47's comparisons against the simple
sigma=0 parabola were therefore not a strict analytical validation. Two
cases are now kept strictly separate and never mixed:

- **CASE P**: sigma forced to 0 -- compared against the exact
  `eta = C + Omega^2 r^2/(2g)` parabola.
- **CASE C**: physical sigma -- compared against a capillary-corrected
  reference (48.4), not the simple parabola.

Capillary length: l_c = sqrt(sigma/(rho g)) = **2.71 mm**. For the
validation geometry: l_c/R_v = 0.113, l_c/R_m = 0.452, l_c/dx = 1.8 grid
cells -- surface tension is a double-digit-percent effect at this vessel
scale and only marginally resolved by the grid, confirming CASE P/C must
not be mixed.

## 48.2 Exact Equilibrium Preservation Test (CASE P)

New: `fields.initialize_rotating_equilibrium` constructs the EXACT
analytical state directly (u_theta=Omega r, u_r=u_z=0, phi matching the
volume-consistent parabola, p matching the rotating-hydrostatic pressure
field, verified to satisfy dp/dr=rho Omega^2 r and dp/dz=-rho g to
tight tolerance -- `tests/test_rotating_equilibrium_init.py`).
`scripts/validate_exact_rotating_equilibrium.py` starts from this state
(sigma=0) and checks whether the solver holds it, t=0.4s, MUSCL2+SSPRK2:

| correction | volume drift | eta NRMSE | center error | max\|u_r\| | max\|u_z\| |
|---|---:|---:|---:|---:|---:|
| OFF | -2.80% | 0.154 | 0.207 mm | 0.134 m/s | 0.128 m/s |
| ON | **-0.03%** | 0.145 | 0.803 mm | 0.144 m/s | 0.141 m/s |

**Finding: the solver does NOT preserve its own exact analytical
equilibrium.** Meridional velocities that should stay at ~0 instead grow to
~0.13-0.14 m/s over 0.4s (comparable to the flat-start transient's own
velocities), and eta NRMSE reaches a similar magnitude (~0.15) as the
flat-start case. This directly rules out "it just hasn't settled from the
transient yet" as the (sole) explanation.

Velocity localization (checked directly, not just at t_end): the maximum
|u_r|/|u_z| at every sampled step (steps 10 through 2000) is consistently
located near r=16.5-23mm (69-96% of R_v=24mm) and at the local interface
height there -- i.e. concentrated where the free surface meets the
outer-wall region, not spread through the bulk. This is consistent with
(not proof of) the wall pressure boundary condition: `pressure.py`'s
`build_pressure_matrix` uses homogeneous-Neumann dp/dr=0 at the wall
(standard for a no-penetration boundary, since u_r is Dirichlet-0 there
regardless of the pressure gradient the projection would otherwise apply),
but the true rotating-equilibrium pressure gradient at the wall is
dp/dr=rho*Omega^2*R_v -- NOT zero. This mismatch was already characterized
as a small (~7%), boundary-localized error in isolation during the
pressure-solver work; it appears to be the leading candidate for what's
driving this equilibrium-preservation failure, though not confirmed as
the sole cause in this pass.

Volume correction ON fixes volume drift (-0.03% vs -2.80%) but does NOT
fix eta NRMSE (0.145 vs 0.154, marginally different) and makes center
error WORSE (0.80mm vs 0.21mm) -- consistent with section 47's finding
that correction is a volume-only fix, not a shape fix.

## 48.3 Redefined convergence metrics (windowed, not instantaneous)

`run_grid_convergence.py` / `run_timestep_convergence.py` now report, over
a window [window_start, t_end]: mean depth, RMS oscillation, mean-
interface RMSE vs. the finest case, and volume drift; instantaneous
`d_final` is kept only as an auxiliary column.

**Grid** (t=0.6s, window=[0.2,0.6]s, MUSCL2+SSPRK2, correction OFF):

| dx | mean depth | RMS oscillation | mean-interface RMSE | volume drift |
|---:|---:|---:|---:|---:|
| 3.00mm | 1.186mm | 0.228mm | 0.868mm | 3.91% |
| 1.50mm | 2.637mm | 0.477mm | 1.011mm | 0.16% |
| 0.75mm | 5.038mm | 2.051mm | 0.000mm (ref) | 6.64% |

**Not monotonically converging even with windowing** -- and this is a
finding, not a bug: mean depth and RMS oscillation both GROW with grid
refinement. The most likely explanation, consistent with 48.2: refining
the grid reduces numerical (artificial) damping of a real, weakly-damped
oscillatory mode, so a finer grid resolves MORE of the true (larger)
oscillation rather than converging toward a quiet equilibrium.

**Timestep** (t=0.3s, window=[0.15,0.3]s, correction OFF):

| mean dt | mean depth | RMS oscillation | mean-interface RMSE | volume drift |
|---:|---:|---:|---:|---:|
| 1.186e-4s | 2.135mm | 0.746mm | 0.953mm | 0.03% |
| 5.929e-5s | 1.048mm | 0.192mm | 1.598mm | 5.31% |
| 2.965e-5s | 2.415mm | 0.098mm | 0.000mm (ref) | 3.61% |

RMS oscillation DOES converge monotonically with smaller dt (0.746 ->
0.192 -> 0.098mm, roughly halving each refinement) -- unlike the grid
case. This is a meaningfully different, better-behaved result: time
integration error is being reduced in the usual way, while spatial
refinement is instead un-masking a real under-resolved physical mode.
mean_depth/volume_drift remain non-monotonic here too (both uncorrected,
consistent with section 47's volume-drift characterization).

## 48.4 Capillary-corrected equilibrium reference (CASE C)

`capillary_equilibrium.py` solves the SMALL-SLOPE-LINEARIZED Young-Laplace
equilibrium in closed form (modified Bessel functions I_0/I_1, not a
nonlinear BVP shooting problem) -- justified quantitatively: the maximum
interface slope (from the sigma=0 parabola, an upper bound) is ~0.27,
giving sqrt(1+slope^2)-1 ~= 3.6%, a small correction to the linearized
curvature term (`tests/test_capillary_equilibrium.py`). Wall BC is
eta'(R_v)=0 (zero contact-angle model, matching the solver's own implicit
zero-gradient Level Set wall treatment, not an added physics assumption).
Verified: satisfies the volume constraint to 1.6e-5 relative error and the
wall slope condition exactly.

Deviation from the simple (sigma=0) parabola at the validation geometry:
max |eta_capillary - eta_simple| = **0.61mm** -- comparable in magnitude to
the eta RMSE values found throughout this and section 47's work, so this
is a real, non-negligible contribution to why the simple-parabola
comparison showed a large NRMSE, though 48.2 shows it is not the whole
story (CASE P, sigma=0, still fails equilibrium preservation).

A dedicated CASE C preservation run (paralleling 48.2 but with physical
sigma and this reference) was not completed in this pass --
`capillary_equilibrium.py` itself is implemented and unit-tested; the
validation gate reports this as WARN, not silently skipped.

## 48.5 Startup ramp sensitivity and oscillation frequency (transient characterization, not gated)

`scripts/validate_spinup_transient.py`, t=1.6s, volume correction ON
(otherwise volume drift contaminates the amplitude/mean metrics),
post-transient analysis window starting at max(3*ramp_time, 0.5*t_end):

| ramp time | max overshoot | dominant frequency | damping |
|---:|---:|---:|---|
| 0.05s | 5.48mm | 1.38 Hz | not resolvable |
| 0.2s | 5.42mm | 2.00 Hz | not resolvable |
| 0.5s | 4.38mm | 1.25 Hz | not resolvable |
| 1.0s | 0.65mm | 1.25 Hz | not resolvable |

**Max overshoot decreases with slower ramp** (a real, fairly clean trend:
5.48 -> 0.65mm from t_ramp=0.05s to 1.0s), supporting "the ramp itself
excites part of the response." **But this does not explain 48.2's
finding**: the exact-equilibrium test has NO ramp at all (Omega is exactly
constant from t=0) and still fails to hold its own equilibrium -- so
startup excitation is at most a partial, not the primary, explanation for
the persistent oscillation.

**Dominant frequency and damping rate are not reliably resolved** in this
pass: estimates vary between analysis windows/durations (an earlier,
shorter, uncorrected 0.05s-ramp run found ~3.75 Hz, closer to the
physically-expected inertial frequency 2*Omega/(2*pi) ~= 3.33 Hz for this
Omega; the results above, from a longer, volume-corrected, post-transient-
windowed analysis, found 1.25-2.0 Hz instead). This is reported as a
genuine unresolved uncertainty -- `oscillation_analysis.py`'s
`damping_from_peaks` correctly reports "not resolvable" in every case here
(README explicit requirement) rather than forcing a number, and the
frequency estimate itself should be treated as low-confidence pending a
longer/more carefully windowed run.

## 48.6 Answers to the required questions

1. **Can the solver hold its own exact analytical equilibrium?** No, not
   within the tested tolerance (eta NRMSE 0.145-0.154 vs. a 0.05
   threshold; max meridional velocity ~0.13-0.14 m/s vs. an ideal ~0).
2. **Is the ~0.2 NRMSE numerical failure, transient oscillation, or
   capillary correction?** Primarily an **numerical/equilibrium-
   preservation issue** (48.2: it persists with NO transient and NO
   ramp). Capillary correction is a real but secondary contributor
   (~0.6mm deviation, 48.4). Transient/startup excitation (48.5)
   measurably affects overshoot magnitude but is not the primary driver,
   since removing the transient entirely (48.2) does not fix it.
3. **Dominant oscillation frequency?** Not reliably resolved this pass;
   estimates ranged 1.25-3.75 Hz across different analysis windows. The
   physically-expected inertial frequency for this geometry/RPM is
   2*Omega/(2*pi) ~= 3.33 Hz.
4. **Does a slower ramp reduce oscillation?** Overshoot yes (5.48mm ->
   0.65mm, 0.05s -> 1.0s ramp); but 48.2 shows the underlying equilibrium-
   preservation problem exists even with zero ramp, so this only partially
   explains the flat-start transient's behavior.
5. **Does correction meaningfully change mean vortex depth?** In the
   exact-equilibrium test, eta NRMSE changes only marginally (0.154 ->
   0.145) but center error gets WORSE (0.21 -> 0.80mm) with correction ON
   -- correction is not neutral on shape, though its effect is secondary
   to the underlying preservation issue.
6. **Does grid/timestep convergence show up in time-averaged metrics?**
   Timestep: yes for RMS oscillation (monotonic, ~halving per refinement).
   Grid: no -- mean depth and RMS oscillation both grow with refinement,
   consistent with finer grids resolving more of a real under-damped mode
   rather than converging.
7. **Is a conservative Level Set / CLSVOF needed now?** Given 48.2's
   finding that even a MUSCL2+SSPRK2+correction run fails to hold the
   EXACT initial equilibrium, and that failure is spatially localized near
   the wall/interface intersection rather than being a general advection-
   diffusion symptom, a higher-order or conservative transport scheme is
   **not clearly the next fix** -- the wall pressure boundary condition
   (48.2) is a more specific, better-supported candidate to investigate
   first.
8. **Ready to move to forcing calibration?** No. The validation gate's
   NUMERICAL VERIFICATION tier now correctly fails on "exact rotating
   equilibrium, sigma=0" (the right thing to gate on), in addition to the
   still-missing forcing calibration data.

---

## Status

Current known experimental geometry:

```text
MSH-20D
RPM: 80–1500 rpm
resolution: 5 rpm

stir bar:
30 mm × 7 mm

beaker:
DURAN low-form 600 mL
nominal diameter: 90 mm
height: 125 mm
```

Still recommended to measure before final simulation:

```text
actual beaker internal diameter
actual water height for each run
actual stir-bar RPM
water temperature
bar-to-bottom position / effective clearance
```

These measurements determine the final numerical input values.
