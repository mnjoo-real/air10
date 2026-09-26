# Non-uniform (stretched) grid: design plan (NOT implemented)

Status (V7-S, 2026-09-26/27): design only; nothing in the solver changed.
- The V7-T requirement from the 0.07 mm forcing-wall contact layer is WITHDRAWN. That layer was an
  artifact of the wall-touching stirrer model (M0) and disappears with the tapered volume forcing M1
  (docs §10.z).
- The targets below use only the layers that remain after M1.

## 1. Post-M1 layers (measured; rigid lid 300 rpm, S1am, M1, tau_s 5 ms fixed)

delta95 = wall distance at which |u| reaches 95 % of the first local maximum within 10 mm of the wall
(`results/validation_v7s/tableE_boundary_layers.md`), t = 2 s, M1 ell_z = 1.75 mm:

| layer | dx 0.5 | dx 0.25 | cells at 0.25 | converged? |
|---|---:|---:|---:|---|
| bottom swirl under the bar (r = 5-10 mm) | 2.45-2.60 mm | 2.34-2.59 mm | 9-10 | yes (set by the taper, not a wall singularity) |
| bottom radial jet (u_r, r = 20 mm) | 0.90 | 0.87 | 3.5 | yes |
| bottom radial jet (u_r, r = 30 mm) | 0.70 | 0.61 | 2.5 | nearly |
| bottom swirl outside the bar (r = 30 mm) | 0.89 | 0.76 | 3.0 | nearly |
| side-wall swirl (z = 10 mm) | 0.93 | 1.09 | 4.3 | nearly |
| side-wall up-flow u_z (z = 10 mm) | 0.63 | 1.53 | 6.1 | NO |
| episodic impingement layer under the bar (late, t = 2.5-3 s) | < 0.5 (in the first cell) | ~0.3-0.5 (omega overshoots to 40-44 rad/s at z ~0.4-0.6 mm) | 1.5-2 | NO |

Thinnest physical layer: the episodic impingement / rotating-core layer under the bar,
~0.3-0.4 mm (about 2 sqrt(nu/Omega), sqrt(nu/Omega) = 0.18 mm at 300 rpm).

## 2. Is stretching a correctness requirement?

Rigid-lid three-grid result (M1 ell 1.75 mm; R = D05_025 / D10_05):
- L_z converges (R 0.09 / 0.36 / 0.05 at t = 2 / 2.5 / 3 s; 0.5 -> 0.25 change 1 % / 5 % / 0.4 %).
- Swirl arrival at (5, 25) converges (2.05 / 2.25 / 2.25 s).
- NOT converged:
  - U_mer (R 8.8; +16 % from 0.5 to 0.25);
  - late T_stir (R 0.49, still 20 %);
  - bottom torque (R 0.73);
  - instantaneous bulk profiles (0.3-0.5 relative L2);
  - arrival at (5, 40).
- Hence the meridional circulation, which the near-wall layers drive, is still grid dependent at 0.25 mm.
  Stretching is therefore a correctness requirement for the meridional flow and the late torques,
  while the angular-momentum content already converges on uniform 0.25 mm.

## 3. Targets and cost (measured layers, not the old estimate)

- z (bottom): dz_min ~= 0.1 mm (3-4 cells across the 0.3-0.4 mm episodic layer, ~9 across the 0.9 mm
  jet). Do not go to 0.06 mm unless 0.1 mm fails a convergence test.
- r (side wall): dr_min ~= 0.2 mm (5 cells across the ~1 mm side-wall swirl layer; the up-flow layer
  0.6-1.5 mm then has >= 3 cells).
- Interior 0.5 mm, growth ratio 1.1. The height interface lives on the radial eta nodes, so fine
  bottom dz does NOT refine the free surface; free-surface curvature needs r-resolution near the
  axis, which stays uniform.

| grid | Nr | Nz | cells | dt_adv (pessimistic / local) | dt_nu | dt_capillary (wall cell) |
|---|---:|---:|---:|---|---:|---:|
| dz_min 0.10, dr_min 0.20 | 94 | 169 | 1.6e4 | 6.7e-5 / 3.2e-4 s | 2.0e-3 s | 9.4e-5 s |
| dz_min 0.075, dr_min 0.20 | 94 | 171 | 1.6e4 | 5.5e-5 / 2.9e-4 | 1.1e-3 | 9.4e-5 |
| dz_min 0.06, dr_min 0.15 | 96 | 173 | 1.7e4 | 4.3e-5 / 2.6e-4 | 7.2e-4 | 6.1e-5 |
| uniform 0.25 (reference) | 180 | 320 | 5.8e4 | 1.25e-4 (MUSCL CFL) | 1.25e-2 | 1.9e-4 |
| uniform 0.125 (reference) | 360 | 640 | 2.3e5 | ~6e-5 | 3.1e-3 | 4.6e-5 |

- Explicit viscosity never limits.
- The capillary limit at the interface cell next to the wall (width dr_min) and the advective limit
  set dt ~ 7e-5 - 1e-4 s, i.e. like uniform 0.25 mm but with 3.6x fewer cells.
- Pressure LU on 1.6e4 unknowns is cheap.
- Expected cost per simulated second: ~2-4x cheaper than uniform 0.25 mm and ~10x cheaper than
  uniform 0.125 mm, with 2.5x finer bottom spacing than 0.25 mm.
- A per-direction / local CFL (instead of global max |u| over min spacing) would recover ~3x in dt.

## 4. Operators that assume uniform scalar dr / dz (audit)

About 250 uses of the scalar grid.dr / grid.dz in about 30 modules. For the single_phase_height
path the affected pieces are:

| component | file | change |
|---|---|---|
| grid construction | grid.py | arrays dr_c, dr_f, dz_c, dz_f; keep scalar fields for legacy grids |
| divergence / gradient | operators.py, pressure_single_phase.py | face-to-face distances per index |
| pressure matrix (liquid, GFM) | pressure_single_phase.py | variable coefficients; ghost-fluid theta uses local h |
| projection | pressure_single_phase.py | local h in the face gradients |
| viscous u_r, u_z, u_theta | operators.py, swirl_transport.angular_momentum_viscous | variable-spacing Laplacians (flux form already; needs h per face) |
| swirl / meridional transport | swirl_transport.py, meridional_transport.py | MUSCL slopes with non-uniform spacing; CV widths |
| forcing kernel | forcing.py | cell-centre sampling unchanged; volume weights local |
| height-function column fluxes | height_interface.py | column areas A_i = pi (r_f,i+1^2 - r_f,i^2) already per index; reconstruction stencils need non-uniform cubic |
| graph curvature | height_interface.py | derivatives from non-uniform nodes |
| crossings / liquid mask | height_interface.py, liquid_mask.py | local h |
| velocity extension | velocity_extension.py | upwind weights use h_r, h_z (per index) |
| stable time step | timestep.py, height_solver.stable_timestep | min over local u/h |
| diagnostics | torque_budget, validate_* scripts | local dV, one-sided wall gradients |

The LS paths (two_phase_diffuse_ls, single_phase_ls) should NOT be converted; they keep the
uniform Grid, so the legacy / LS bitwise regressions stay untouched.

## 5. Validation gates to rerun if implemented

V1 geometry and operators (manufactured, variable h: divergence, gradient, Laplacian order),
V2/H3 hydrostatic, V3/H4 rigid body, H1/H2 height geometry and transport, V4b-H pinned wall
(phases), integrated rotation equilibrium, V6/V7 Table A forcing moments, the V7-T manufactured
swirl/meridional tests, the rigid-lid spin-up, then the low-RPM free surface.

## 6. Risk

High. Every discrete operator of the verified height branch changes. The pinned-contact-line
and curvature results (V4b-H) depend on uniform near-wall r-spacing, exactly where r would be
refined. Estimated effort: a full pass for the operators and gates V1-H4, then another for
V4b-H / integrated / V7. Do not start it without the user's decision.
