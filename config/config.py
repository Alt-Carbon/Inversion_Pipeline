"""Paths and parameters for the inversion pipeline.

Two ways to point it at data:

  1. A known, hand-calibrated benchmark from the BENCHMARKS registry below
     (this repo ships two: "synthetic400", "callisto"):

         TOMOFASTX_BENCHMARK=synthetic400 python -m scripts.run_baseline

  2. Any OTHER data folder -- the format is auto-detected (src/discover.
     py) from what's actually in the folder, one of:
       - a *.OBS file (+ TOMOPARAMS*.TXT + meshgrid*.txt) -- Tomofast-x's
         own format
       - a *.tif/*.tiff/*.ers file -- a geographic raster grid (the
         format grav_mag_inversion's own WA pipeline uses)
       - a *.geojson file -- point stations (the format india_grav_mt_
         inversion's own pipeline uses)
     Mesh/decimation parameters are calibrated automatically the first
     time (src/auto_calibrate.py, cached afterwards):

         TOMOFASTX_DATA_DIR=/path/to/your/survey python -m scripts.run_baseline

     Format-specific settings (all via env var, all optional except
     where noted):
       raster:  TOMOFASTX_RASTER_FILE (which file, if the folder has more
                  than one *.tif/*.ers -- required in that case only)
                TOMOFASTX_AOI ("west,east,south,north" degrees -- default:
                  the raster's own full extent)
                TOMOFASTX_VALUE_SCALE (default 1.0 -- e.g. WA's own
                  gravity grids need 10.0, micro-m/s^2 -> mGal)
                TOMOFASTX_FIELD ("magnetic" default, or "gravity" -- note
                  the inversion stage itself is magnetic-only so far, see
                  README "Known limitations")
                TOMOFASTX_IGRF_DATE (default src/load_survey.py's
                  DEFAULT_IGRF_DATE -- a generic placeholder; override to
                  your survey's own acquisition date)
       geojson: TOMOFASTX_GEOJSON_FILE (which file, if more than one --
                  required in that case only)
                TOMOFASTX_VALUE_KEY (REQUIRED -- which feature property
                  holds the observed value, e.g. "magnetic_a")
                TOMOFASTX_ELEVATION_KEY (optional -- a station-elevation
                  property; not currently used for receiver z, kept for
                  a future topography pass, same as india_grav_mt_
                  inversion's own io_data.py)
                TOMOFASTX_IGRF_AMPLITUDE_KEY (optional -- a real per-
                  station field-intensity property, e.g. "igrf_nt", used
                  for F instead of the generic IGRF-14 lookup)
                TOMOFASTX_AOI, TOMOFASTX_IGRF_DATE (as above)
     General overrides (any format, any dataset):
       TOMOFASTX_NAME (default: the data folder's own basename) -- output
         filename prefix (results/<name>_result.npz, etc).
       TOMOFASTX_RAM_BUDGET_GB (default 4.0) -- dense sensitivity matrix
         budget auto-calibration targets.
       TOMOFASTX_DECIMATE_STRIDE/TOMOFASTX_CORE_CELL_M/
         TOMOFASTX_CORE_CELL_Z_M/TOMOFASTX_DEPTH_CORE_M/
         TOMOFASTX_PAD_DISTANCE_M -- override any single auto-calibrated
         mesh value directly.
       TOMOFASTX_NOISE_PCT/TOMOFASTX_NOISE_FLOOR_NT (defaults 0.05/1.0).
       TOMOFASTX_SECTION_DEPTH_LIM/TOMOFASTX_VOLUME_DEPTH_LIM ("lo,hi"
         metres) -- display depth window for 2D sections / 3D viewer.

TOMOFASTX_DATA_DIR takes precedence over TOMOFASTX_BENCHMARK when both are
set -- explicit-path mode is assumed to be the deliberate choice.
"""
import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]        # .../Tomofastx2.0_models

TOMOPARAMS_FILE_HINT = "TOMOPARAMS.TXT"     # only used for the two registry entries below;
MESHGRID_FILE_HINT = "meshgrid_2depth.txt"  # arbitrary-folder mode auto-detects instead (src/discover.py)

BENCHMARKS = {
    "synthetic400": dict(
        folder="synthetic400", obs_file="MAG_n5f1.OBS",
        # Calibrated empirically against this machine's 15 GB RAM (see
        # baseline_report.md's "Compute Feasibility") -- not a resolution
        # choice made for its own sake. decimate_stride=6 on the native
        # 400x400 receiver raster -> 67x67 (4,489 receivers); core_cell_m
        # =100 m is close to half that decimated spacing (~150 m), the
        # usual rule of thumb, with some RAM headroom left over.
        decimate_stride=6, core_cell_m=100.0, core_cell_z_m=50.0,
        depth_core_m=3000.0,          # paper's own Table 2 core depth is 3290 m
        pad_distance_m=3000.0,
        section_depth_lim=(-2500.0, 500.0),
        volume_depth_lim=(-2500.0, 2000.0),
    ),
    "callisto": dict(
        folder="callisto", obs_file="MAG_2depth.OBS",
        # Calibrated the same way as synthetic400 (build the mesh, measure
        # active cells directly, target a dense matrix safely under the
        # ~4.5 GB level that's run reliably on this machine before). No
        # drill-hole data exists in the released package to validate
        # against -- this run is a pipeline/feasibility check, not a
        # ground-truth-validated result (see README "Included example
        # datasets").
        decimate_stride=3, core_cell_m=20.0, core_cell_z_m=10.0,
        depth_core_m=3000.0, pad_distance_m=2000.0,
        section_depth_lim=(-2500.0, 500.0), volume_depth_lim=(-2500.0, 500.0),
    ),
}

DATA_DIR_OVERRIDE = os.environ.get("TOMOFASTX_DATA_DIR")


def _parse_aoi(env_val):
    if not env_val:
        return None
    w, e, s, n = (float(v) for v in env_val.split(","))
    return dict(west=w, east=e, south=s, north=n)


if DATA_DIR_OVERRIDE:
    # ---- Arbitrary-folder mode -- any of the 3 supported formats ----
    DATA_DIR = Path(DATA_DIR_OVERRIDE).expanduser().resolve()
    if not DATA_DIR.is_dir():
        raise FileNotFoundError(f"TOMOFASTX_DATA_DIR={DATA_DIR_OVERRIDE!r} is not a directory")
    BENCHMARK = os.environ.get("TOMOFASTX_NAME", DATA_DIR.name)

    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    from src import discover                                # noqa: E402

    FORMAT, _info = discover.detect_format(DATA_DIR)

    RESULTS_DIR = PROJECT_ROOT / "results"
    RESULTS_DIR.mkdir(exist_ok=True)
    CALIBRATION_CACHE_PATH = RESULTS_DIR / f"{BENCHMARK}_calibration.json"

    _aoi_env = _parse_aoi(os.environ.get("TOMOFASTX_AOI"))
    _igrf_date_env = os.environ.get("TOMOFASTX_IGRF_DATE")

    # Format-specific kwargs for src.load_survey.load_survey(fmt, format_kwargs, ...).
    # OBS_FILE/TOMOPARAMS_FILE/MESHGRID_FILE below are kept as module-level
    # names for backward compatibility with code written against "obs"-
    # only datasets (they're None for the other two formats).
    OBS_FILE = TOMOPARAMS_FILE = MESHGRID_FILE = None
    if FORMAT == "obs":
        OBS_FILE, TOMOPARAMS_FILE, MESHGRID_FILE = (
            _info["obs_name"], _info["tomoparams_name"], _info["meshgrid_name"])
        FORMAT_KWARGS = dict(folder=str(DATA_DIR), obs_name=OBS_FILE,
                             tomoparams_name=TOMOPARAMS_FILE, meshgrid_name=MESHGRID_FILE)
    elif FORMAT == "raster":
        _raster_names = _info["raster_names"]
        _raster_file = os.environ.get("TOMOFASTX_RASTER_FILE")
        if _raster_file is None:
            if len(_raster_names) > 1:
                raise ValueError(
                    f"{DATA_DIR} has {len(_raster_names)} raster candidates {_raster_names} -- "
                    "set TOMOFASTX_RASTER_FILE to say which one.")
            _raster_file = _raster_names[0]
        FORMAT_KWARGS = dict(
            path=str(DATA_DIR / _raster_file), aoi=_aoi_env,
            value_scale=float(os.environ.get("TOMOFASTX_VALUE_SCALE", "1.0")),
            field=os.environ.get("TOMOFASTX_FIELD", "magnetic"),
            igrf_date=_igrf_date_env)
    elif FORMAT == "geojson":
        _geojson_names = _info["geojson_names"]
        _geojson_file = os.environ.get("TOMOFASTX_GEOJSON_FILE")
        if _geojson_file is None:
            if len(_geojson_names) > 1:
                raise ValueError(
                    f"{DATA_DIR} has {len(_geojson_names)} geojson candidates {_geojson_names} -- "
                    "set TOMOFASTX_GEOJSON_FILE to say which one.")
            _geojson_file = _geojson_names[0]
        _value_key = os.environ.get("TOMOFASTX_VALUE_KEY")
        if not _value_key:
            raise ValueError(
                f"{DATA_DIR / _geojson_file}: TOMOFASTX_VALUE_KEY is required for GeoJSON input "
                "-- which feature property holds the observed value? (see src/load_survey.py's "
                "read_geojson docstring)")
        FORMAT_KWARGS = dict(
            path=str(DATA_DIR / _geojson_file), aoi=_aoi_env, value_key=_value_key,
            elevation_key=os.environ.get("TOMOFASTX_ELEVATION_KEY"),
            igrf_amplitude_key=os.environ.get("TOMOFASTX_IGRF_AMPLITUDE_KEY"),
            igrf_date=_igrf_date_env)

    _stride_env = os.environ.get("TOMOFASTX_DECIMATE_STRIDE")
    _cell_env = os.environ.get("TOMOFASTX_CORE_CELL_M")
    _cellz_env = os.environ.get("TOMOFASTX_CORE_CELL_Z_M")
    _depth_env = os.environ.get("TOMOFASTX_DEPTH_CORE_M")
    _pad_env = os.environ.get("TOMOFASTX_PAD_DISTANCE_M")

    _all_mesh_given = _cell_env and _cellz_env and (FORMAT == "geojson" or _stride_env)
    if _all_mesh_given:
        # Every mesh parameter given explicitly -- skip auto-calibration
        # (and the data read it needs) entirely.
        DECIMATE_STRIDE = int(_stride_env) if _stride_env else None
        CORE_CELL_M = float(_cell_env)
        CORE_CELL_Z_M = float(_cellz_env)
        DEPTH_CORE_M = float(_depth_env) if _depth_env else 3000.0
        PAD_DISTANCE_M = float(_pad_env) if _pad_env else 2000.0
    elif CALIBRATION_CACHE_PATH.exists():
        _cached = json.loads(CALIBRATION_CACHE_PATH.read_text())
        DECIMATE_STRIDE = int(_stride_env) if _stride_env else _cached["decimate_stride"]
        CORE_CELL_M = float(_cell_env) if _cell_env else _cached["core_cell_m"]
        CORE_CELL_Z_M = float(_cellz_env) if _cellz_env else _cached["core_cell_z_m"]
        DEPTH_CORE_M = float(_depth_env) if _depth_env else _cached["depth_core_m"]
        PAD_DISTANCE_M = float(_pad_env) if _pad_env else _cached["pad_distance_m"]
    else:
        from src import load_survey as t2s                   # noqa: E402
        from src import auto_calibrate                        # noqa: E402
        _ram_budget = float(os.environ.get("TOMOFASTX_RAM_BUDGET_GB", "4.0"))
        _depth_arg = float(_depth_env) if _depth_env else 3000.0
        _pad_arg = float(_pad_env) if _pad_env else None
        print(f"[config] no cached calibration for {BENCHMARK!r} ({FORMAT} format) -- "
             f"auto-calibrating mesh/decimation against a {_ram_budget:.1f} GB budget "
             f"(one-time; cached to {CALIBRATION_CACHE_PATH.name} afterwards)...")

        if FORMAT == "obs":
            _x, _y, _z, _v, _n = t2s.read_obs(DATA_DIR / OBS_FILE)
            _calib = auto_calibrate.calibrate_resolution(
                _x, _y, ram_budget_gb=_ram_budget, depth_core_m=_depth_arg, pad_distance_m=_pad_arg)
        elif FORMAT == "raster":
            import numpy as np
            _window, _ = t2s.read_raster(FORMAT_KWARGS["path"], aoi=FORMAT_KWARGS["aoi"],
                                         value_scale=FORMAT_KWARGS["value_scale"],
                                         field=FORMAT_KWARGS["field"])
            _xg, _yg = np.meshgrid(_window.x_m, _window.y_m)
            _calib = auto_calibrate.calibrate_resolution(
                _xg.ravel(), _yg.ravel(), ram_budget_gb=_ram_budget,
                depth_core_m=_depth_arg, pad_distance_m=_pad_arg)
        else:   # geojson
            _station, _ = t2s.read_geojson(FORMAT_KWARGS["path"], aoi=FORMAT_KWARGS["aoi"],
                                           value_key=FORMAT_KWARGS["value_key"],
                                           elevation_key=FORMAT_KWARGS["elevation_key"])
            _calib = auto_calibrate.calibrate_resolution_sparse(
                _station.x, _station.y, ram_budget_gb=_ram_budget,
                depth_core_m=_depth_arg, pad_distance_m=_pad_arg)

        CALIBRATION_CACHE_PATH.write_text(json.dumps(
            {k: _calib[k] for k in
             ("decimate_stride", "core_cell_m", "core_cell_z_m", "depth_core_m", "pad_distance_m")},
            indent=2))
        DECIMATE_STRIDE = int(_stride_env) if _stride_env else _calib["decimate_stride"]
        CORE_CELL_M = float(_cell_env) if _cell_env else _calib["core_cell_m"]
        CORE_CELL_Z_M = float(_cellz_env) if _cellz_env else _calib["core_cell_z_m"]
        DEPTH_CORE_M = float(_depth_env) if _depth_env else _calib["depth_core_m"]
        PAD_DISTANCE_M = float(_pad_env) if _pad_env else _calib["pad_distance_m"]

    # Generic display-only depth window (2D sections / 3D viewer) -- both
    # hand-picked registry windows sit at about -0.83 * depth_core_m at
    # the bottom, +500 m at the top; no better generic default is
    # available without real topography info. Override with
    # TOMOFASTX_SECTION_DEPTH_LIM/TOMOFASTX_VOLUME_DEPTH_LIM ("lo,hi"
    # metres) if it clips your data.
    _section_env = os.environ.get("TOMOFASTX_SECTION_DEPTH_LIM")
    _volume_env = os.environ.get("TOMOFASTX_VOLUME_DEPTH_LIM")
    _default_depth_lim = (-0.83 * DEPTH_CORE_M, 500.0)
    SECTION_DEPTH_LIM = (tuple(float(v) for v in _section_env.split(","))
                         if _section_env else _default_depth_lim)
    VOLUME_DEPTH_LIM = (tuple(float(v) for v in _volume_env.split(","))
                        if _volume_env else _default_depth_lim)

else:
    # ---- Known-benchmark mode (always "obs" format) ----
    BENCHMARK = os.environ.get("TOMOFASTX_BENCHMARK", "synthetic400")
    if BENCHMARK not in BENCHMARKS:
        raise ValueError(
            f"TOMOFASTX_BENCHMARK={BENCHMARK!r} not in {list(BENCHMARKS)} -- for any OTHER "
            "dataset, point TOMOFASTX_DATA_DIR at its folder instead (see this module's "
            "docstring); it does not need a BENCHMARKS entry.")
    _b = BENCHMARKS[BENCHMARK]
    FORMAT = "obs"
    DATA_DIR = PROJECT_ROOT / _b["folder"]
    OBS_FILE = _b["obs_file"]
    TOMOPARAMS_FILE = TOMOPARAMS_FILE_HINT
    MESHGRID_FILE = MESHGRID_FILE_HINT
    FORMAT_KWARGS = dict(folder=str(DATA_DIR), obs_name=OBS_FILE,
                         tomoparams_name=TOMOPARAMS_FILE, meshgrid_name=MESHGRID_FILE)
    DECIMATE_STRIDE = _b["decimate_stride"]
    CORE_CELL_M = _b["core_cell_m"]
    CORE_CELL_Z_M = _b["core_cell_z_m"]
    DEPTH_CORE_M = _b["depth_core_m"]
    PAD_DISTANCE_M = _b["pad_distance_m"]
    SECTION_DEPTH_LIM = _b["section_depth_lim"]
    VOLUME_DEPTH_LIM = _b["volume_depth_lim"]

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"
VIEWER_DIR = PROJECT_ROOT / "viewer"
EXPERIMENTS_DIR = RESULTS_DIR / "experiments"   # one-off diagnostic runs, see improved_report.md

SEED = 42
CHI_TARGET = 1.0

# Distance-to-nearest-receiver threshold for the survey-footprint mask
# (src/coverage.py) -- scaled off the core cell size (2.5x, comfortably
# larger than the decimated receiver spacing) rather than a fixed metre
# value, so it stays sensible across very different survey scales.
COVERAGE_MAX_DIST_M = 2.5 * CORE_CELL_M

# Sparse/IRLS defaults -- same as india_grav_mt_inversion's own
# run_magnetics.py (see src/invert_v2.py's run_sparse_v2 docstring for
# why these were kept instead of SimPEG's own 1.2/8).
SPARSE_NORMS = (0.0, 2.0, 2.0, 2.0)
IRLS_COOLING_FACTOR = 1.1
MAX_IRLS_ITERATIONS = 12

# Noise model -- see load_survey.standard_deviation's docstring. Exact
# for Synthetic400 only (its .OBS filename literally encodes it); for any
# other dataset/format this is just a starting assumption, not derived
# from anything -- override via TOMOFASTX_NOISE_PCT/TOMOFASTX_NOISE_
# FLOOR_NT if you have real per-station uncertainty or a better estimate.
NOISE_PCT = float(os.environ.get("TOMOFASTX_NOISE_PCT", "0.05"))
NOISE_FLOOR_NT = float(os.environ.get("TOMOFASTX_NOISE_FLOOR_NT", "1.0"))
