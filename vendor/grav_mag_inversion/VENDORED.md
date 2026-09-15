# Vendored from `grav_mag_inversion`

This directory is a **frozen snapshot**, taken 2026-09-15, of the parts of
the sibling `grav_mag_inversion` project (the WA/Yilgarn gravity+magnetics
inversion pipeline, not otherwise part of this repo) that this pipeline's
own code depends on:

| File | Source | What it's used for here |
|---|---|---|
| `invert.py` | `grav_mag_inversion/src/invert.py`, byte-for-byte | `run_smooth` (scripts/run_baseline.py), `InversionResult`/`_TIGHT_TOL` (src/invert_v2.py, scripts/sweep_beta.py) |
| `mesh3d.py` | `grav_mag_inversion/src/mesh3d.py`, byte-for-byte | `build_mesh`/`active_cells_flat_surface` (src/tomofast_to_simpeg.py, src/auto_calibrate.py) |
| `forward.py` | `grav_mag_inversion/src/forward.py`, byte-for-byte | `build_magnetic_simulation` (src/tomofast_to_simpeg.py) |
| `regional.py` | `grav_mag_inversion/src/regional.py`, byte-for-byte | `fit_polynomial_trend` (src/tomofast_to_simpeg.py) |
| `plotting.py` | extracted from `grav_mag_inversion/scripts/run_gravity.py` (`style`, `plot_convergence`, `plot_data_fit`, `plot_model_sections`, `_slice_locs`, colour constants only) | scripts/plot_figures.py |
| `igrf.py` | extracted from `grav_mag_inversion/scripts/run_magnetics.py` (`igrf_at_aoi` only) | src/readers.py -- inducing field for raster/geojson sources that carry no field data of their own |
| `io_raster.py` | `grav_mag_inversion/src/io_data.py`, byte-for-byte | src/readers.py's `read_raster` -- GeoTIFF/ERS grid loading (WA-format data) |
| `io_geojson.py` | `india_grav_mt_inversion/src/io_data.py`, byte-for-byte | src/readers.py's `read_geojson` -- point-station loading (India-format data) |

`plotting.py` is not a byte-for-byte copy of `run_gravity.py` -- that
file is a driver script coupled to `grav_mag_inversion`'s own
`config.py` and `io_data.py` (imported unconditionally at module load,
regardless of which function is actually called), neither of which this
repo has any other reason to pull in. Only the plotting functions
themselves were extracted; their bodies are unchanged.

## Why vendored rather than a path/submodule dependency

This project originally imported `grav_mag_inversion` directly via a
`sys.path` insert reaching the sibling directory it lives in on the
machine this was developed on. That worked locally but meant `git clone`
of this repo alone was not runnable -- the whole point of this pipeline
being "point it at input data and it runs" (see README.md) fails if a
core dependency isn't actually in the repo. Vendoring a snapshot was
chosen over a git submodule because `grav_mag_inversion` is not (as of
this writing) itself a published/public repository this project could
depend on that way.

## Consequence: this snapshot does NOT auto-update

If `grav_mag_inversion` changes upstream (bug fixes, new features), this
copy does not pick that up automatically -- re-copy the relevant file(s)
by hand and note the new snapshot date above if that's ever needed.
Conversely, per this project's own working constraints, none of these
files should be *edited* here in a way that would make them diverge
from a legitimate upstream fix -- if a bug is found in one of them,
fix it upstream in `grav_mag_inversion` first, then re-sync the copy.
