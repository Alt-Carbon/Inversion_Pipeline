# Baseline Report — grav_mag_inversion pipeline vs. Tomofast-x 2.0 Synthetic400

> **Note (added later):** this is the round-1 investigation record, kept
> as-is below for its original numbers/reasoning. Callisto (called
> "deferred" throughout this report) has since been run -- see the main
> README's "Included example datasets" for current numbers on both. This
> report's per-body validation still applies to Synthetic400 only:
> Callisto has no published ground truth to validate against regardless
> of the RAM blocker discussed below.

Round 1, diagnostic only. `src/invert.py` was not modified. No topography,
distance weighting, kernel caching, or ADMM was added. This round covers
**Synthetic400 only** — Callisto is deferred (see "Scope" below).

## Scope

- **Synthetic400**: full round (converter, driver, per-body validation,
  runtime/memory logging) — this report.
- **Callisto**: deferred at the requester's instruction, after two
  blockers surfaced: (1) at native resolution the dense sensitivity
  matrix would be ~387 GB (this machine has 15 GB RAM) -- since resolved
  by decimating further, see the README, not by acquiring more RAM, and
  (2) no drill-hole location/assay data exists in the local file set or
  the Zenodo package to build the requested drill-hole susceptibility
  table -- this one is NOT resolved, Callisto's run has no ground-truth
  check.
  `tomofast_to_simpeg.py`'s readers work on Callisto's files too (same
  format), so `scripts/run_callisto.py` is a small follow-on once the
  resolution/drill-hole questions are resolved — not written this round.

## Data and Ground Truth — what's exact, what's approximate, what's missing

Reverse-engineered directly from the downloaded files (no format spec
ships with the Zenodo package) and cross-checked against the paper itself
(Ogarko et al., 2024, *Geosci. Model Dev.* 17, 2325–2345,
[doi:10.5194/gmd-17-2325-2024](https://doi.org/10.5194/gmd-17-2325-2024)):

| Item | Status | Source |
|---|---|---|
| Receiver locations, TMI values | Exact | `MAG_n5f1.OBS`, parsed directly |
| Inducing field (F=41770 nT, I=−21.4°, D=0.1°) | Exact | `TOMOPARAMS.TXT`, parsed directly |
| Body shape, susceptibility contrast (Table 3) | Exact | Table 3 **downloaded as XLSX** and parsed (not read off the rendered table image — that was checked first and found to be incomplete/error-prone) |
| Body thickness/radius (Body 1: 100 m disc, Body 2: 50 m sheet, Body 3: 10 m slab, Body 4: 250 m-radius pipe, Body 5: 20 m sheet, Body 6: 500 m-radius pipe) | Lower confidence | Article prose (Sect. 3.1), extracted via an LLM-summarised page fetch, not independently re-verified against a primary-source table the way Table 2/3 were |
| Per-station noise/uncertainty | **Not in the data** | `.OBS` files have exactly 4 columns (X, Y, Z, value) — no 5th column. See `standard_deviation()` in `tomofast_to_simpeg.py`: reconstructed from the paper's noise-generation description (1–10% of signal, 1 or 5 nT floor) plus the file name `MAG_n5f1.OBS` ("n5f1" read as 5%/1 nT — an inference, not a confirmed spec) |
| Body horizontal centroid (Synthetic400) | **Approximate, digitised** | Not in Table 2, not in Table 3 (both checked directly via XLSX, not just the rendered image), not stated numerically anywhere in the article text found. Read by eye off **Figure 4** (map-view total response with body outlines on a 2,500 m graticule), anchored to real UTM coordinates via the receiver footprint's own extent. Uncertainty band ±300–500 m per body — see `src/validation.py`'s `BODIES` table for the per-body reasoning |
| Body depth-to-top | **Not available anywhere** | No cross-section of the *true* model is published — Figure 6 shows the *inverted* result, not ground truth, and no numeric depth appears in the text or either table. Depth recovery below is reported qualitatively (which depth band the recovered signal falls in), not as a numeric error |
| Body true volume | **Not available with confidence** | Thickness/radius exist for some bodies (lower-confidence source above) but not full 3-D extent for any of them (slab/disc plan-view size is never given) — a true-vs-recovered volume *ratio* is not computed; only the recovered contour volume is reported |

**Table 2 and Table 3, checked exactly, for the record**, since an earlier
exchange in this session assumed they held the body coordinates —
verified they don't, by downloading each as XLSX and parsing it
(`/tmp/.../table2.xlsx`, `table3.xlsx`), not by trusting a rendered-image
summary:

- Table 3 columns: `Body | Shape | Susceptibility (SI) | Analogue` — 6
  rows, no coordinates, no dimensions.
- Table 2 columns: `Name | ΔEast | ΔNorth | Depth | Relief | Nx | Ny | Nz
  | Ndata` — domain/mesh sizing only (Synthetic400: 10,000×10,000 m core,
  +5,904 m padding each side, depth 3,290 m, Nx=Ny=400(+40), Nz=28,
  Ndata=158,006 — this Ndata differs slightly from the shipped
  `MAG_n5f1.OBS`'s 160,000; not investigated further, immaterial to this
  round since we decimate regardless).

## Compute Feasibility — why this isn't run at native resolution

Native Synthetic400: 440×440×28 = 5,420,800 mesh cells, 160,000
receivers. `Simulation3DIntegral` (unmodified, per constraints) builds a
**dense** sensitivity matrix of size `n_data × n_active_cells`:

```
160,000 × 5,420,800 × 8 bytes ≈ 6.3 TB
```

This machine has **15 GB RAM total** (checked directly, `free -h`). Even
Tomofast-x's own authors need wavelet compression (their
`forward.matrixCompression.type=1`) to run this at all — `invert.py` has
no such path and is not allowed one this round.

**Resolution taken**: receivers decimated from the native 400×400 raster
by stride 6 (real receiver locations and values, not synthesised — see
`decimate_receiver_grid`), giving a 67×67 = 4,489-receiver grid at ~150 m
spacing. Core cell size set to 100 m horizontally / 50 m vertically
(`mesh3d.build_mesh`, unmodified) — calibrated empirically (built the
mesh and measured active-cell count directly, not estimated on paper) to
land the dense matrix around 2 GB, the same order of magnitude that ran
reliably on comparable past runs on this class of machine (worked at
~4.5 GB, failed above ~9 GB):

| Quantity | Native (paper) | This round | Ratio |
|---|---|---|---|
| Receivers | 160,000 | 4,489 | 1/36 |
| Core cell (horiz.) | 25 m | 100 m | 4× coarser |
| Mesh cells (reduced-domain) | 5,420,800 | 119,792 (62,976 active) | — |
| Dense matrix | 6.3 TB | 2.26 GB | — |

**This is a materially lower-resolution run than the paper's** — the
per-body recovery numbers below should be read as "what this pipeline
does at a resolution ~4× coarser than the reference, on ~1/36 of the
data," not as a resolution-matched baseline. That gap is itself a
finding for round 2 (see "Next round" below), not something to paper
over.

One coordinate-system fix was needed to make the reused mesh code behave
correctly: Tomofast-x's Z is not elevation-above-sea-level in the usual
sense (Synthetic400 receivers sit at Z = −2,498 to −1,111 m in its own
local datum). `mesh3d.build_mesh`/`active_cells_flat_surface` assume a
flat surface at *z=0*; without recentring, only ~4,000 mostly-coarse
cells came out active (confirmed empirically, not assumed) because the
mesh's own internal refinement is centred on z=0 regardless of what
surface elevation is passed in. Fixed by shifting all Z by the task's own
instruction (flat topography at mean receiver elevation) so that mean
elevation lands at z=0 — a translation only, same axes and units,
documented in `flat_surface_z()`.

## Task 1/2 — Converter and Driver

- `src/tomofast_to_simpeg.py`: parses `.OBS`, `TOMOPARAMS.TXT`, and
  streams `meshgrid_*.txt` for its bounding box (not loaded cell-by-cell
  — see Compute Feasibility); builds the SimPEG survey via
  `forward.build_magnetic_simulation` (unmodified) and the mesh via
  `mesh3d.build_mesh` (unmodified); returns `dobs`/`std` arrays in the
  same calling convention `run_gravity.py`/`run_magnetics.py` already use
  for `invert.py`, plus a `simpeg.data.Data` object for convenience.
- `scripts/run_synthetic400.py`: calls `invert.run_smooth` — confirmed
  this is the method the task's Context names ("WeightedLeastSquares
  regularization and InexactGaussNewton"), i.e.
  `regularization.WeightedLeastSquares` + `optimization.
  InexactGaussNewton` + `directives.BetaSchedule(coolingFactor=2.0,
  coolingRate=1)` — **with no keyword arguments beyond `chi_target`**, so
  every other value (`max_iter=30, alpha_s=0.05, length_scale_x=1.0`) is
  `invert.py`'s own literal function-signature default, not a tuned
  value carried over from the WA/India projects (which use
  `run_sparse`/IRLS as their default, a different function).

## Results

### Convergence

```
n_data = 4,489   n_active_cells = 62,976
phi_d:  47,229 -> 27,584 -> 17,102 -> 10,067 -> 5,725 -> 3,200   (6 iterations)
target phi_d = 4,489 (chi_target=1.0)
final chi = phi_d / n_data = 0.71
```

Stopped on step-size tolerance (`|xc-x_last|` below tolX), not on
reaching the target misfit — chi=0.71 means the model **overfits**
slightly relative to the target (this is `run_smooth`'s own default
behaviour, unmodified; not a tuning choice made this round).

### Runtime and memory (Task 4)

| Quantity | Value |
|---|---|
| Load + convert time | 6.3 s |
| Inversion time | 61–111 s (two runs, same seed/result; wall-clock varied with machine load) |
| Peak RSS | 2.32 GB (started at 0.30 GB) |
| Dense sensitivity matrix | 4,489 × 62,976 × 8 B = 2.26 GB |

This is round 1's baseline number to compare round 2 against — e.g. if a
compression or tiling scheme is added later, its value is in letting
`n_data`/`n_active_cells` grow toward the native 160,000/5,420,800
without RAM blowing up the same way.

### Per-body recovery (Synthetic400, smooth L2, this round's resolution)

| Body | Shape | True χ (SI) | Peak recovered χ (SI) | Peak ratio | Centroid offset (m, ±300–500 m true-position uncertainty) | Volume ≥0.01 SI (m³, cells) | Depth band of significant recovery (m) |
|---|---|---:|---:|---:|---:|---:|---|
| 1 | flat disc | 0.0500 | 0.0081 | 0.16 | 138 | 0 (0) | −125 to −800 |
| 2 | flat slab | 0.0700 | 0.0067 | 0.10 | 399 | 0 (0) | −350 to −1300 |
| 3 | dipping slab | 0.0300 | 0.0029 | 0.10 | 379 | 0 (0) | −25 to −1000 |
| 4 | vertical cylinder | 0.1000 | 0.0191 | 0.19 | 741 | 4.08×10⁸ (94) | −25 to −800 |
| 5 | dipping slab | 0.0200 | 0.0007 | 0.03 | 141 | 0 (0) | −350 to −1700 |
| 6 | vertical cylinder | 0.0015 | −0.0071 | n/a | 37 | 0 (0) | −350 to −1000 |

(Raw numbers: `src/validation.py::per_body_report`, run against
`results/synthetic400_result.npz`.)

## Interpretation — plain language, per metric

**Convergence (chi=0.71).** The inversion ran to completion and stopped
on its own step-size criterion rather than hitting a wall — this is a
normal, healthy `run_smooth` run on this data, not a failure. Chi
slightly below 1 just means the recovered model fits the (assumed) noise
level a little more tightly than strictly required.

**Peak susceptibility recovery (10–19% of true value for 5 of 6
bodies).** This is expected, not a bug: smooth (L2) regularization is
known to spread a compact true anomaly's mass over a wider volume of
cells, diluting its peak value — and this round's mesh is 4× coarser
than the paper's, which dilutes further by volume-averaging within each
(larger) cell. The one body that gets closest in relative terms (Body 4,
19%) is also the strongest true source (0.10 SI) and the one with real
recovered volume above the 0.01 SI contour — consistent with "the
strongest source is the one method+resolution combination here can
partially resolve; the weaker ones get smoothed below the threshold
entirely."

**Body 6 recovering negative.** Body 6's true contrast (0.0015 SI) is
tiny — smaller than this run's own noise floor (1 nT) is likely to
resolve at all. The recovered value near it (−0.0071 SI) is most likely
the smooth model's own negative side-lobe from a nearby positive
anomaly, not a resolved "reversed" body. The `peak_recovery_ratio` for
this body is reported as n/a rather than the literal −4.7 the arithmetic
gives, since dividing by a true value this small produces a number that
looks alarming but isn't a meaningful ratio.

**Centroid offset (37–741 m).** Four of six bodies land within or close
to their own digitised-position uncertainty (±300–500 m) — i.e. not
distinguishable from "the recovered centroid matches, within how
precisely we could even read the true position off a figure." Body 4's
741 m is the one clear outlier, exceeding its own ±300 m uncertainty
band — plausibly genuine lateral smearing from smooth regularization
(consistent with it being the body with the most spatial extent above
the contour threshold) rather than a digitising-error artifact, but this
can't be fully disentangled without a real coordinate ground truth.

**Volume at the 0.01 SI contour (only Body 4 nonzero).** Not a
volume-recovery *ratio* (no reliable true volume exists to divide by —
see Data and Ground Truth) — just the raw recovered volume. Five of six
bodies show **zero** volume above this contour, including Body 6, for
which that's mechanically guaranteed regardless of recovery quality
(its own true value, 0.0015 SI, is below the 0.01 SI threshold). For the
other four (true values 0.02–0.07 SI), zero recovered volume at 0.01 SI
is consistent with the peak-recovery dilution above — their peaks (0.0007
to 0.0081 SI) never reach the contour threshold at this resolution/method.

## Limitations of this round

1. **Resolution gap vs. the paper is large** (4× coarser mesh, 1/36 the
  receivers) — required by this machine's 15 GB RAM, not a pipeline
  defect. See "Compute Feasibility."
2. **Body centroids are digitised by eye from a figure**, not exact —
  every centroid-offset number carries a real ±300–500 m uncertainty
  that's comparable to some of the offsets themselves.
3. **No depth ground truth exists anywhere found** for Synthetic400 — depth
  recovery is reported as a qualitative band only, per body, not a
  numeric error metric (the task's originally-requested "top-depth error
  in metres" could not be computed against any available source).
4. **No true-volume ground truth** — only recovered contour volume is
  reported, not a recovery ratio.
5. **Noise/std is inferred from a filename convention** (`n5f1`), not
  confirmed against a spec — see Data and Ground Truth.
6. **Callisto not run this round** — deferred (missing drill-hole data,
  and the same RAM ceiling applies at an even larger native scale, 387 GB).
7. **Body thickness/radius figures** (used only for context in this
  report, not in any computed metric) came from an LLM-summarised page
  fetch of the article text, not a primary-source table download the way
  Table 2/3 were — lower confidence than everything else in this report.

## Candidate directions for round 2 (not started, listed for context only)

- Closing the resolution gap (tiling, tighter active-cell restriction
  around the receiver footprint, or a compression scheme) to see whether
  the low peak-recovery ratios are mostly a resolution artifact or persist
  closer to native scale.
- Comparing `run_smooth` against `run_sparse`/IRLS on this same reduced
  data, since sparse regularization is known (from the WA/India projects)
  to produce more compact, higher-amplitude recovered bodies than smooth
  L2 — directly relevant to the low peak-ratio numbers above.
- If exact per-body ground truth (coordinates, depth, dimensions) can be
  obtained from the paper's authors or a supplementary source not found
  this round, redoing the per-body table with real numbers instead of
  digitised/qualitative ones.
