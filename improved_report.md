# Improved Report — Round 2 (Synthetic400)

> **Note (added later):** kept as-is below for its original numbers/
> reasoning. Callisto has since been run as a pipeline/feasibility check
> (see the main README) -- the missing-drillhole blocker below is still
> real for it (no ground truth exists to validate against, regardless of
> resolution), so nothing in this report's per-body validation applies
> to it. The root-cause investigation below (sign-flip artifact,
> bound-constrained optimizer attempts) is Synthetic400-specific work
> and hasn't been re-run against Callisto.

Round 2, per the task's own scope narrowing: **Callisto is deferred, left
blank** (the RAM-at-native-resolution and missing-drillhole blockers from
round 1 are unchanged; nothing in this round's four improvements resolves
either). Synthetic400 only, below.

## Scope and constraints honoured

- No MPI, wavelet compression, or magnetisation vector inversion added.
- `regularization.WeightedLeastSquares`, `regularization.Sparse`, and
  `optimization.InexactGaussNewton` are used exactly as `grav_mag_
  inversion/src/invert.py`'s `run_smooth`/`run_sparse` use them -- none
  of the three classes was subclassed, monkeypatched, or edited (the
  shipped "improved" result switched from `WeightedLeastSquares` to
  `Sparse`/IRLS partway through the round -- see "Root cause
  investigation" -- but this is a choice of *which* SimPEG regularization
  class to call unmodified, same as `run_smooth` vs `run_sparse` already
  being two different functions in the untouched `invert.py`, not a
  change to either class's internals). The five new modules live
  entirely in this project (`src/topography.py`, `src/distance_
  weighting.py`, `src/kernel_cache.py`, `src/artefact_detector.py`,
  `src/invert_v2.py`) and do not modify `grav_mag_inversion/src/
  invert.py`, `forward.py`, or `mesh3d.py` -- verified directly (not just
  by intent): calling this round's converter with every new toggle left
  at its default reproduces round 1's baseline bit-for-bit (same
  chi=0.7127450929833306, same 6 iterations, same active-cell count
  62,976), confirming the existing Yilgarn (WA) and India workflows,
  which import those same unmodified files, are unaffected.

## Task 1 — Topography

`src/topography.py` builds a DEM from `MAG_n5f1.OBS`'s own receiver
elevation column (a clean native 400×400 grid at 25 m spacing -- see its
module docstring for why this was used over trying to recover topography
from `meshgrid_2depth.txt`'s ambiguous per-cell layering) and drapes
`mesh3d.build_mesh`'s (unchanged) active-cell mask below it, instead of
the flat z=0 plane round 1 used. Receivers are placed at
`dem(x,y) + height_above_surface` (0 m this round -- ground-based, not
airborne, per `TOMOPARAMS.TXT`).

| | Baseline (flat z=0) | Improved (draped) |
|---|---:|---:|
| Active cells | 62,976 | 65,144 |
| Receiver elevation range | 0 m (flat, by construction) | see DEM check below |
| Active cell z range | -10300 to 0 m | -10300 to 600 m |

Real topography spans about -680 to +701 m around the domain mean
(checked directly against the DEM) -- draping recovers roughly 1,700 m
of vertical relief round 1's flat approximation either wrongly excluded
(high ground) or wrongly included as "ground" (low ground, above true
local elevation but below the flat z=0 plane).

## Task 2 — Distance-based sensitivity weighting

`src/distance_weighting.py` implements Li & Oldenburg (1996b)'s
`w_j = 1/(r_j+r0)^(β/2)` (β=3 for magnetics), `r_j` = nearest-receiver
distance rather than depth-below-a-flat-surface -- injected via
`reg.set_weights(sensitivity=...)`, the same weight key `directives.
UpdateSensitivityWeights` itself writes to (confirmed by reading its
source), so this is a genuine substitution, not an additional term.
`r0` defaults to `core_cell_z_m` (50 m) -- a documented simplification,
not Li & Oldenburg's own fitted value; `src/distance_weighting.py` also
gained a `fit_r0()` function later (see "Root cause investigation") that
calibrates it against the actual sensitivity matrix instead of guessing.

**This is built and independently testable (`invert_v2.run_smooth_v2
(..., weighting="distance")`), but is NOT what the shipped "improved"
result uses.** It was the first thing tried once topography was on;
investigating why its recovered bodies looked wrong (see "Root cause
investigation") led to testing several alternatives, ending with a
switch to sparse/IRLS regularization (`invert_v2.run_sparse_v2`, using
the *default* real per-cell sensitivity weighting, not this distance
proxy) for the result actually shipped in Task 5 below. Kept as a
toggle and documented here since it was a real, working deliverable of
this round, just not the one that ended up in the final pipeline.

## Task 3 — Kernel caching

`src/kernel_cache.py` points `Simulation3DIntegral` at a hash-keyed disk
cache directory using SimPEG's own public `store_sensitivities="disk"` /
`sensitivity_path` mechanism (verified directly from source -- no private
attribute access, no subclassing). Demonstrated on a 5-value beta sweep,
each value run as a fresh mesh/survey/sim (simulating separate script
invocations), cache disabled vs enabled:

| Beta | Time, no cache (s) | Time, cached (s) | Cache result |
|---:|---:|---:|---|
| 10000 | 72.6 | 72.5 | miss (first run) |
| 3000 | 70.5 | 35.4 | hit |
| 1000 | 73.7 | 34.9 | hit |
| 300 | 72.5 | 32.7 | hit |
| 100 | 72.8 | 34.4 | hit |
| **total** | **362.0** | **209.8** | |

**Reduction on total sweep wall time: 42%** -- below the
task's 90% target, and the reason is disclosed rather than hidden:
caching eliminates the G-matrix computation (~instant on a hit, vs. tens
of seconds on a miss), but not the optimizer's own iteration cost, which
dominates total wall time at this problem size once G is cached. 90%
reduction would need the *solve* itself to also be cheap relative to G's
computation, which isn't the regime this benchmark's resolution sits in
-- see `scripts/sweep_beta.py`'s own printed caveat.

## Task 4 — Artefact detector

`src/artefact_detector.py` flags the four documented L2-smooth artefacts
against each relevant body (heuristic geometric checks with disclosed
thresholds -- see its module docstring; each row below carries its own
evidence, not just a bare flag):

| Artefact | Body | Baseline flagged | Improved flagged | Baseline evidence | Improved evidence |
|---|---|---|---|---|---|
| deep_tail_extension | 4 | no | no | significant recovery (30% of peak 0.0191 SI) spans -1000 to -25 m (975 m total) near Body 4; flag threshold 1200 m | significant recovery (30% of peak 0.0846 SI) spans -800 to 75 m (875 m total) near Body 4; flag threshold 1200 m |
| positive_halo_around_negative_body | 6 | no | no | core |peak| near Body 6 = 0.0071 SI; max positive value in the 1200-2500 m halo shell = 0.0022 SI (31% of core peak, flag threshold 50%) | core |peak| near Body 6 = 0.0146 SI; max positive value in the 1200-2500 m halo shell = 0.0058 SI (40% of core peak, flag threshold 50%) |
| thin_sheet_bottom_depth_overestimate | 2 | YES | YES | recovered significant-value extent near Body 2 (flat slab) = 950 m vs. true thickness ~50 m (19.0x, flag threshold 3x) | recovered significant-value extent near Body 2 (flat slab) = 1250 m vs. true thickness ~50 m (25.0x, flag threshold 3x) |
| thin_sheet_bottom_depth_overestimate | 3 | YES | no | recovered significant-value extent near Body 3 (dipping slab) = 450 m vs. true thickness ~10 m (45.0x, flag threshold 3x) | recovered significant-value extent near Body 3 (dipping slab) = 0 m vs. true thickness ~10 m (0.0x, flag threshold 3x) |
| thin_sheet_bottom_depth_overestimate | 5 | YES | YES | recovered significant-value extent near Body 5 (dipping slab) = 1350 m vs. true thickness ~20 m (67.5x, flag threshold 3x) | recovered significant-value extent near Body 5 (dipping slab) = 1475 m vs. true thickness ~20 m (73.8x, flag threshold 3x) |
| bowl_shaped_disc_overthickening | 1 | no | no | significant-cell count near Body 1 grows 1.0x from the shallowest depth bin (69 cells) to the widest bin (69 cells); flag threshold 1.8x | significant-cell count near Body 1 grows 1.5x from the shallowest depth bin (6 cells) to the widest bin (9 cells); flag threshold 1.8x |

**3 of 6 checks flagged on the baseline run,
2 of 6 on the improved run.**

## Task 5 — Comparison

| | Baseline | Improved |
|---|---:|---:|
| chi | 0.71 | 0.99 |
| Active cells | 62,976 | 65,144 |
| Wall time (load+inversion) | 79.7s | 166.0s |
| Peak RSS | 2.31 GB | 2.35 GB |
| Artefact flags | 3/6 | 2/6 |

### Per-body recovery, baseline vs improved

| Body | Shape | True χ (SI) | Same-sign peak χ base | Same-sign peak χ improved | Ratio base | Ratio improved | Centroid offset base (m) | Centroid offset improved (m) | Sign flip? (improved) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | flat disc | 0.0500 | 0.0081 | 0.0187 | 0.16 | 0.37 | 138 | 68 | YES |
| 2 | flat slab | 0.0700 | 0.0067 | 0.0106 | 0.10 | 0.15 | 399 | 191 | YES |
| 3 | dipping slab | 0.0300 | 0.0029 | 0.0089 | 0.10 | 0.30 | 379 | 548 | YES |
| 4 | vertical cylinder | 0.1000 | 0.0191 | 0.0629 | 0.19 | 0.63 | 741 | 832 | YES |
| 5 | dipping slab | 0.0200 | 0.0007 | 0.0012 | 0.03 | 0.06 | 141 | 529 | YES |
| 6 | vertical cylinder | 0.0015 | 0.0022 | 0.0058 | 1.46 | 3.88 | 37 | 243 | YES |

### Sign check — a real finding, not a table artefact

Checked directly (not assumed) after the raw per-body numbers first came
back with every "improved" body reading as strongly *negative*, including
the five bodies with a positive true contrast: within 1500 m of every
single body, the improved run has BOTH a correctly-signed peak AND a
larger-magnitude opposite-sign artifact nearby. The table above already
uses the same-sign peak (the metric that actually answers "how well was
this body recovered"); this table shows both peaks side by side so the
artifact isn't hidden by that choice:

| Body | Sign-agnostic peak base | Sign-agnostic peak improved | Same-sign peak improved | Opposite-sign artifact present? |
|---|---:|---:|---:|---|
| 1 | 0.0081 | -0.0192 | 0.0187 | YES -- larger than the body's own peak |
| 2 | 0.0067 | -0.0244 | 0.0106 | YES -- larger than the body's own peak |
| 3 | 0.0029 | -0.0133 | 0.0089 | YES -- larger than the body's own peak |
| 4 | 0.0191 | -0.0846 | 0.0629 | YES -- larger than the body's own peak |
| 5 | 0.0007 | -0.0017 | 0.0012 | YES -- larger than the body's own peak |
| 6 | -0.0071 | -0.0146 | 0.0058 | YES -- larger than the body's own peak |

**Every body in the improved run has an opposite-sign artifact larger
than its own correctly-signed peak within 1500 m -- and this is true for
EVERY configuration tried this round, including the sparse/IRLS result
actually shipped here.** This did not happen in the baseline run (all
six baseline peaks were correctly signed). See "Root cause investigation"
below for the full trail of what was tried (r0 calibration, real
sensitivity weighting, beta floors, bound constraints) and why none of
it eliminated this -- the short version: it is real topography itself,
not any particular weighting scheme or regularization family, that
requires a much less-regularized model to fit the data comparably well,
and that lower-regularization regime is where this artifact lives.

## Plain-language interpretation

- **Data fit improved substantially** (chi 0.71 -> 0.99,
  right at the target of 1.0) with topography + sparse/IRLS on, consistent
  with the model no longer having to explain real topographic relief
  using a flat-surface geometry it doesn't actually have, and with
  sparse/IRLS's minimum-support term giving genuinely better-recovered
  (not just better-fitting) correctly-signed peaks than smooth L2 managed
  at any weighting scheme tried (see "Root cause investigation").
- **Same-sign peak recovery ratio increased for all six bodies** (see the
  per-body table) -- consistent, not just an average effect. This is a
  change in behaviour, not on its own proof of "more correct" recovery --
  the ground-truth caveats from baseline_report.md (digitised,
  approximate body positions; no depth ground truth) still apply equally
  to both runs, and it needs to be read together with the sign-check
  finding immediately below: bigger correctly-signed peaks came bundled
  with bigger opposite-sign artifacts right next to them.
- **Every body now has a larger opposite-sign artifact nearby than its
  own peak** -- see "Sign check" above. This is the single most
  important finding of this round's comparison and should not be read
  past: the improved run's chi is closer to target, but its per-body
  recovered structure is arguably less trustworthy at face value than
  the baseline's, not more.
- **Artefact flag counts**: see the table above for the actual per-body
  verdicts -- read these next to their evidence strings, not as a bare
  pass/fail score, since the thresholds behind each check are disclosed
  heuristics (see src/artefact_detector.py), not a validated statistical
  test.
- **Runtime/memory**: the improved run's own single-pass cost is
  reported above; the caching benefit specifically only shows up across
  *multiple* runs sharing forward geometry (Task 3's sweep), not on a
  single run -- see that section.

## Root cause investigation — the sign-flip artifact

Once topography was on, every body's recovery showed a larger opposite-
sign artifact nearby (see "Sign check" above). This was investigated
across several configurations before settling on sparse/IRLS as the
shipped method -- all interactive, one-off runs (using the toggles `src/
invert_v2.py` and `src/distance_weighting.py` gained for this), not
separate permanent pipeline scripts.

| Configuration | chi | beta (final) | phi_m (final) | Bodies still sign-flipped |
|---|---:|---:|---:|---:|
| Baseline (flat topography, depth weighting, smooth L2) | 0.71 | 27.4 | 136 | 0 / 6 |
| Topo + distance weighting, r0=50 (guessed) | 0.95 | 0.051 | 10,116 | 6 / 6 |
| Topo + distance weighting, r0=18.6 (fitted to actual G) | 0.88 | 0.118 | 3,842 | 6 / 6 |
| Topo + real per-cell sensitivity weighting, smooth L2 | 0.95 | 0.231 | 892 | 6 / 6 |
| Topo + real sensitivity + beta floor 27.4, smooth L2 | 1.90 | 27.4 | 218 | 4 / 6 |
| Topo + real sensitivity + beta floor 10, smooth L2 | 1.26 | 10.0 | 401 | 4 / 6 |
| Topo + real sensitivity + beta floor 5, smooth L2 | 1.00 | 5.0 | 604 | 5 / 6 |
| Topo + real sensitivity + bounds [0, 1] SI, smooth L2 | -- | -- | -- | run failed, see below |
| **Shipped**: topo + real sensitivity, sparse/IRLS | 0.99 | -- (IRLS, not BetaSchedule) | -- | 6 / 6 |

**r0 calibration** (fit against the real sensitivity decay of the actual
G matrix, Li & Oldenburg's own calibration method, not the
`core_cell_z_m` guess): fitted value 18.6 m, smaller than the 50 m guess.
Using it moved beta/phi_m to a less extreme point, but removed zero sign
flips -- r0 was not the primary driver.

**Real per-cell sensitivity weighting** (the most accurate weighting
possible, computed directly from G): still 6/6 sign-flipped under smooth
L2. This isolated the cause away from distance-weighting's *approximation
quality* specifically. The common factor across every topography run
under smooth L2 was a beta cooling 100-500x further than the flat-
topography baseline needed for a comparable fit.

**Beta floor** (clamping beta from cooling below a set value, smooth L2):
a real, continuous trade-off, not a fix -- lower floor gave better chi
but more/larger sign flips, higher floor suppressed more flips at the
cost of chi. Floor=5 was the best balance found (chi=1.00, 5/6 still
flipped).

**Bound constraints** (checked against the paper itself): Ogarko et al.
state directly, "The positivity constraint was added via the alternating
direction method of multipliers (ADMM)-bound constraints... In all
susceptibility inversions a positivity constraint was applied to reduce
the model non-uniqueness (Li and Oldenburg, 1996a)" -- confirmed against
both benchmarks' own `TOMOPARAMS.TXT` (`inversion.admm.magn.bounds = 0.
1.`). Implemented via `optimization.ProjectedGNCG` (inherits directly
from `InexactGaussNewton` with a `Bounded` mixin -- confirmed from
source; no new optimizer algorithm, just per-step projection). Two
attempts, both failed numerically: starting exactly at the lower bound
(`m0=0`) left the optimizer stuck at the starting model for all 30
iterations; starting just inside the bound (`m0=0.001`) got the model
moving but the line search broke down at iteration 7. Debugging was
stopped at the user's instruction rather than pursued further -- this
remains, on the merits, the most principled fix tried and was not ruled
out by evidence, only by not-yet-solved SimPEG numerics.

**Sparse/IRLS** (`invert_v2.run_sparse_v2`, same `norms=(0,2,2,2)`,
`irls_cooling_factor=1.1`, `max_irls_iterations=12` as india_grav_mt_
inversion's own run_magnetics.py, real per-cell sensitivity weighting,
requested explicitly after the smooth-L2 investigation above): chi
reached 0.99, and same-sign peak recovery ratios improved
substantially over every smooth-L2 configuration tried (see the per-body
table above -- e.g. Body 4's ratio roughly doubled versus the best
smooth-L2 result). **But the opposite-sign artifact is still present in
all 6 bodies.** This was not the expected outcome -- grav_mag_inversion's
WA/India projects adopted sparse/IRLS specifically because its minimum-
support term suppresses exactly this kind of smooth-regularization
ringing, and it did measurably improve the correctly-signed recovery
here -- but it did not remove the companion artifact the way the
WA/India precedent might suggest it should. The most likely reading,
consistent with every configuration tested: the negative structure is
not a diffuse smoothing artifact that a sparsity prior naturally
suppresses -- it is compact enough (and useful enough for fitting the
data under real topography) that IRLS's own norms=(0,2,2,2) doesn't
penalise it away, which points back to the positivity-bound approach
(the paper's own actual answer, given the true bodies are all
non-negative) as the more direct fix, still unresolved due to the
ProjectedGNCG numerics issue above, not tried again this round per the
user's instruction to stop debugging.

**Verdict**: within this round's runs, the sign-flip artifact was
reduced (real sensitivity weighting, then sparse/IRLS both improved the
correctly-signed peak's own quality) but not eliminated by any
regularization or weighting choice tested, including switching
regularization families entirely. The shipped result (sparse/IRLS) is
the best available this round on data fit and correctly-signed recovery
quality, with the sign-flip caveat disclosed rather than hidden. Getting
`ProjectedGNCG` to converge with a positivity bound remains the most
likely fully-effective fix, given it would rule the artifact out by
construction rather than discourage it statistically -- flagged as the
lead candidate for a future round, needing numerical tuning time that
was not spent this round.

## Limitations carried over from round 1 (still apply)

No true-depth ground truth for any Synthetic400 body; body centroids are
still digitised from Fig. 4 (±300-500 m); Callisto deferred; noise model
inferred from a filename convention. See `baseline_report.md` for the
full list -- none of round 2's four improvements address these.
