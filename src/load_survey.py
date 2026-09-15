"""Converts survey data in any of three supported formats into the SimPEG
objects the existing grav_mag_inversion pipeline expects -- reuses
vendor.grav_mag_inversion's {forward,mesh3d,regional} and invert.py's
calling convention UNCHANGED (see vendor/grav_mag_inversion/VENDORED.md
for why this is a vendored snapshot).

Three input formats, auto-detected by src/discover.py from what's in a
data folder:

  "obs"     Tomofast-x 2.0's own plain-text format (Ogarko et al. 2024,
            GMD 17, 2325-2345; data at Zenodo 10.5281/zenodo.8397794) --
            a `*.OBS` file (line 1: n_data; then n_data rows of
            `X Y Z VALUE`, a regular receiver raster), a `*TOMOPARAMS*.
            TXT` file (`key = value` lines -- exact inducing field F/I/D,
            no external lookup needed), a `meshgrid*.txt` file (native
            mesh extent only, not reused cell-for-cell -- see
            build_reduced_mesh). Real per-receiver elevation -- the only
            format here that supports real topography draping.

  "raster"  A GeoTIFF/ERS grid in a geographic (lon/lat) CRS (the format
            grav_mag_inversion's own WA pipeline uses) -- a dense
            continuous grid, decimated the same way an .OBS raster is.
            No inducing-field data of its own (unlike "obs") -- computed
            via IGRF-14 at the data's own centroid (vendor.grav_mag_
            inversion.igrf) for a date you supply. No receiver elevation
            -- topography draping isn't available for this format.

  "geojson" A GeoJSON FeatureCollection of point stations (the format
            india_grav_mt_inversion's own pipeline uses) -- already-
            sparse discrete points, used as-is (no decimation). You must
            say which feature property holds the observed value
            (`value_key`) -- unlike file-FORMAT detection (unambiguous
            from the file extension), which property is "the data" isn't
            guessable from the file alone. If the file carries a real
            per-station field-intensity property (e.g. igrf_nt), pass it
            as `igrf_amplitude_key` for a better F than the generic IGRF-
            14 lookup; inclination/declination still come from IGRF-14
            (no format here carries real per-station direction data).

All three converge on the same shared pipeline after loading: optional
decimation, 2nd-degree regional trend removal, a mesh built the same way
regardless of source, and the same SimPEG magnetic simulation assembly.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # .../Tomofastx2.0_models
from vendor.grav_mag_inversion import forward, mesh3d, regional   # noqa: E402
from vendor.grav_mag_inversion import igrf as igrf_mod            # noqa: E402
from src import topography as topo                                 # noqa: E402


# ---------------------------------------------------------------- "obs" ----

def read_tomoparams(path):
    """`key = value` lines -> dict, values coerced to float where they
    parse cleanly (Fortran `0.1d0`-style exponents normalised to `e`),
    left as strings otherwise (e.g. file paths, `./output/`)."""
    out = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or set(line) <= {"="}:
            continue
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip()
        val_norm = val.replace("d0", "e0").replace("D0", "e0").rstrip(".")
        try:
            out[key] = float(val_norm)
        except ValueError:
            out[key] = val
    return out


def read_obs(path):
    """Returns (x, y, z, values, n_declared). n_declared is the header
    count, checked against the actual row count (raises if they disagree
    -- a truncated/corrupt download should fail loudly, not silently)."""
    with open(path) as f:
        n_declared = int(f.readline().split()[0])
    arr = np.loadtxt(path, skiprows=1)
    if arr.shape[0] != n_declared:
        raise ValueError(f"{path}: header declares {n_declared} rows, found {arr.shape[0]}")
    x, y, z, v = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
    return x, y, z, v, n_declared


def meshgrid_extent(path, sample_every=20000):
    """Streams the (large -- hundreds of MB) mesh grid file for just its
    bounding box and native cell-size range, without loading all n_cells
    rows into memory (we don't reuse their per-cell mesh -- see module
    docstring and build_reduced_mesh -- this is for the report's "their
    native resolution vs. ours" comparison only)."""
    x0min = y0min = z0min = np.inf
    x1max = y1max = z1max = -np.inf
    dx_samples, dz_samples = [], []
    with open(path) as f:
        n_declared = int(f.readline().split()[0])
        for i, line in enumerate(f):
            parts = line.split()
            if len(parts) < 6:
                continue
            x0, x1, y0, y1, z0, z1 = (float(p) for p in parts[:6])
            x0min, x1max = min(x0min, x0), max(x1max, x1)
            y0min, y1max = min(y0min, y0), max(y1max, y1)
            z0min, z1max = min(z0min, z0), max(z1max, z1)
            if i % sample_every == 0:
                dx_samples.append(x1 - x0)
                dz_samples.append(z1 - z0)
    return dict(
        n_cells=n_declared,
        x_extent=(x0min, x1max), y_extent=(y0min, y1max), z_extent=(z0min, z1max),
        dx_range=(min(dx_samples), max(dx_samples)) if dx_samples else None,
        dz_range=(min(dz_samples), max(dz_samples)) if dz_samples else None,
    )


def decimate_receiver_grid(x, y, z, v, stride):
    """Subsamples a regular receiver raster in file order (row-major, X
    fastest). Reshapes to (ny, nx) from the unique coordinate counts and
    strides both axes."""
    xu = np.unique(x)
    yu = np.unique(y)
    if xu.size * yu.size != x.size:
        raise ValueError(
            f"receivers are not a clean {yu.size}x{xu.size} raster "
            f"({xu.size * yu.size} != {x.size}) -- decimate_receiver_grid "
            "assumes a regular grid; not valid for scattered/flight-line data."
        )
    shp = (yu.size, xu.size)
    xg, yg, zg, vg = (a.reshape(shp) for a in (x, y, z, v))
    xg, yg, zg, vg = (a[::stride, ::stride] for a in (xg, yg, zg, vg))
    return xg.ravel(), yg.ravel(), zg.ravel(), vg.ravel()


def flat_surface_z(z_receivers):
    """Recentres a local vertical datum so its own mean sits at z=0 --
    required because Tomofast-x's own Z is NOT elevation-above-sea-level
    in the usual sense (Synthetic400's receivers sit at Z = -2498 to
    -1111 m) and grav_mag_inversion's mesh3d/forward code hardcodes a
    z=0 flat-surface convention throughout. Confirmed empirically: taking
    active_cells_flat_surface at the *unshifted* mean receiver Z returned
    only ~4,000 (mostly coarse padding) active cells vs. ~63,000 once
    recentred. A coordinate translation only (same axes/units)."""
    return float(np.mean(z_receivers))


def standard_deviation(dobs, pct=0.05, floor_nt=1.0):
    """Per-station uncertainty as a fraction of |signal| with a floor --
    the same convention regardless of source format. Exact values (pct/
    floor_nt) matter: Synthetic400's are inferred from its own MAG_n5f1.
    OBS filename convention (see baseline_report.md); any other dataset's
    are a starting assumption unless you have real per-station
    uncertainty of your own to use instead."""
    return np.maximum(pct * np.abs(dobs), floor_nt)


def build_reduced_mesh(x, y, core_cell_m, core_cell_z_m, depth_core_m, pad_distance_m,
                       dem=None):
    """Thin wrapper over the UNCHANGED vendor.grav_mag_inversion.mesh3d.
    build_mesh -- 'TreeMesh matching a real survey's model dimensions' is
    interpreted as matching the overall core *domain* (horizontal extent
    from the receiver footprint, a fixed core depth) rather than any
    source format's own native per-cell resolution, which this pipeline's
    dense-matrix method cannot afford at native density regardless of
    machine size (see README "How resolution is chosen").

    `dem`: optional callable dem(x, y) -> elevation (see src/topography.
    py) -- when given, active cells are draped below it instead of the
    flat z=0 plane. Only the "obs" format currently has a real elevation
    source to build a DEM from (see module docstring)."""
    mesh = mesh3d.build_mesh((x, y), core_cell_m, depth_core_m, pad_distance_m,
                             core_cell_z_m=core_cell_z_m)
    if dem is None:
        active = mesh3d.active_cells_flat_surface(mesh, z_surface=0.0)
    else:
        active = topo.drape_active_cells(mesh, dem)
    return mesh, active


# ------------------------------------------------------------- "raster" ----

def read_raster(path, aoi=None, value_scale=1.0, field="magnetic"):
    """WA-style GeoTIFF/ERS grid reader (vendor.grav_mag_inversion.
    io_raster, unchanged). Geographic (lon/lat) rasters only for now --
    checked via the file's own CRS; a projected-CRS raster raises rather
    than silently misapplying the lon/lat-degree local projection
    io_raster.py uses.

    `aoi` (dict: west/east/south/north, degrees): if not given, uses the
    raster's own full extent (its rasterio bounds) -- an explicit aoi
    only matters for windowing into part of a LARGER raster (e.g. WA's
    own statewide tiles); an already-cropped single-survey file doesn't
    need one.

    `value_scale`: WA's own gravity grid needs dividing by 10.0 (micro-
    m/s^2 -> mGal, see grav_mag_inversion/config/config.py's
    MICROMS2_PER_MGAL) before use; magnetic grids there are already nT
    (scale=1.0, the default) -- override for a different unit convention
    in your own data."""
    import rasterio
    from vendor.grav_mag_inversion import io_raster

    with rasterio.open(path) as s:
        crs = s.crs
        bounds = s.bounds
    if crs is None or not crs.is_geographic:
        raise ValueError(
            f"{path}: CRS is {crs} (not geographic lon/lat) -- read_raster only supports "
            "geographic rasters right now (matching grav_mag_inversion's own WA data); "
            "reproject to a lon/lat CRS (e.g. EPSG:4326) first.")
    if aoi is None:
        aoi = dict(west=bounds.left, east=bounds.right, south=bounds.bottom, north=bounds.top)

    if field == "gravity":
        window = io_raster.load_gravity_window(path, aoi, value_scale)
    else:
        window = io_raster.load_magnetic_window(path, aoi)
        if value_scale != 1.0:
            window.values = window.values / value_scale
    return window, aoi   # GridWindow: values(2D), mask, x_m(1D), y_m(1D), lon0, lat0


# ------------------------------------------------------------ "geojson" ----

def read_geojson(path, aoi=None, value_key=None, elevation_key=None, extra_keys=None):
    """India-style GeoJSON point-station reader (vendor.grav_mag_
    inversion.io_geojson, unchanged).

    `aoi`: sets the local-metric projection origin (its centroid) -- if
    not given, derived from the data's own lon/lat bounding box (these
    files already represent one confined survey, unlike a statewide
    raster, so no external aoi is normally needed).

    `value_key`: REQUIRED -- the GeoJSON feature property holding the
    observed value. Not auto-detected: guessing which of possibly
    several numeric properties is "the data" would be a silent, easily-
    wrong choice, unlike file-format detection itself (unambiguous from
    the extension) -- see src/discover.py."""
    from vendor.grav_mag_inversion import io_geojson

    if value_key is None:
        raise ValueError(
            f"{path}: value_key is required for GeoJSON input -- which feature property holds "
            "the observed value? (e.g. 'magnetic_a'/'bouguer_an' in the India dataset this was "
            "built against -- inspect the file's own properties) -- pass via TOMOFASTX_VALUE_KEY.")
    if aoi is None:
        with open(path) as f:
            data = json.load(f)
        lons = [feat["geometry"]["coordinates"][0] for feat in data["features"]]
        lats = [feat["geometry"]["coordinates"][1] for feat in data["features"]]
        aoi = dict(west=min(lons), east=max(lons), south=min(lats), north=max(lats))
    station = io_geojson.load_station_geojson(path, aoi, value_key, elevation_key, extra_keys)
    return station, aoi   # StationData: x, y, v, elevation, lon0, lat0, extra


# --------------------------------------------------------- unified entry ----

DEFAULT_IGRF_DATE = "2020-01-01"   # generic placeholder date for IGRF-14 lookups (raster/
                                   # geojson formats only -- "obs" gets an exact F/I/D from
                                   # TOMOPARAMS, real survey data should override this via
                                   # TOMOFASTX_IGRF_DATE to the survey's own acquisition date


def load_survey(fmt, format_kwargs, decimate_stride, core_cell_m, core_cell_z_m,
                depth_core_m, pad_distance_m, regional_trend_degree=2,
                noise_pct=0.05, noise_floor_nt=1.0, use_topography=False,
                receiver_height_above_surface=0.0):
    """Single entry point for all three formats -- loads raw (x, y, z, v)
    plus an inducing field, then runs the SAME shared pipeline regardless
    of source (decimation if the format supports it, regional trend
    removal, mesh building, SimPEG assembly). Returns a dict with
    everything scripts/run_*.py need (matches the old load_benchmark's
    return shape, with two format-generic additions: `format` and
    `topography_available`).

    `fmt`: "obs" | "raster" | "geojson" (see src/discover.py).
    `format_kwargs`: format-specific arguments --
      obs:     folder, obs_name, tomoparams_name, meshgrid_name
      raster:  path, aoi (optional), value_scale (optional), igrf_date (optional)
      geojson: path, aoi (optional), value_key, elevation_key (optional),
               igrf_amplitude_key (optional), igrf_date (optional)

    `use_topography=True` is only honoured for fmt="obs" (the only format
    with a real per-receiver elevation source) -- for "raster"/"geojson"
    it's silently treated as False (both source pipelines this was built
    against -- WA, India -- already assume flat z=0 receivers themselves;
    see io_raster.py/io_geojson.py's own docstrings), but the returned
    dict's `topography_available` flag says so explicitly rather than
    leaving a caller to infer it from silence."""
    if fmt == "obs":
        return _load_survey_obs(format_kwargs, decimate_stride, core_cell_m, core_cell_z_m,
                                depth_core_m, pad_distance_m, regional_trend_degree,
                                noise_pct, noise_floor_nt, use_topography,
                                receiver_height_above_surface)
    elif fmt == "raster":
        return _load_survey_grid(format_kwargs, decimate_stride, core_cell_m, core_cell_z_m,
                                 depth_core_m, pad_distance_m, regional_trend_degree,
                                 noise_pct, noise_floor_nt, use_topography, is_raster=True)
    elif fmt == "geojson":
        return _load_survey_grid(format_kwargs, decimate_stride, core_cell_m, core_cell_z_m,
                                 depth_core_m, pad_distance_m, regional_trend_degree,
                                 noise_pct, noise_floor_nt, use_topography, is_raster=False)
    else:
        raise ValueError(f"fmt must be 'obs', 'raster', or 'geojson', got {fmt!r}")


def _assemble(x, y, z_receiver, v, F, I, D, core_cell_m, core_cell_z_m, depth_core_m,
             pad_distance_m, regional_trend_degree, noise_pct, noise_floor_nt, dem):
    """Shared tail end for every format: mesh, SimPEG survey/sim, regional
    trend, noise model, Data object -- identical regardless of how
    (x, y, z_receiver, v, F, I, D) were obtained."""
    mesh, active = build_reduced_mesh(x, y, core_cell_m, core_cell_z_m,
                                      depth_core_m, pad_distance_m, dem=dem)
    survey, sim = forward.build_magnetic_simulation(mesh, active, x, y, F, I, D, z_m=z_receiver)
    trend, trend_info = regional.fit_polynomial_trend(x, y, v, regional_trend_degree)
    dobs = v - trend
    std = standard_deviation(dobs, pct=noise_pct, floor_nt=noise_floor_nt)

    from simpeg import data as simpeg_data
    data_obj = simpeg_data.Data(survey, dobs=dobs, standard_deviation=std)
    return mesh, active, survey, sim, trend, trend_info, dobs, std, data_obj


def _load_survey_obs(kw, decimate_stride, core_cell_m, core_cell_z_m, depth_core_m,
                     pad_distance_m, regional_trend_degree, noise_pct, noise_floor_nt,
                     use_topography, receiver_height_above_surface):
    folder = Path(kw["folder"])
    params = read_tomoparams(folder / kw["tomoparams_name"])
    mesh_meta = meshgrid_extent(folder / kw["meshgrid_name"])
    x_raw, y_raw, z_raw, v_raw, n_declared = read_obs(folder / kw["obs_name"])

    z0 = flat_surface_z(z_raw)
    x, y, z, v = decimate_receiver_grid(x_raw, y_raw, z_raw - z0, v_raw, decimate_stride)

    dem = None
    receiver_z = None
    if use_topography:
        dem = topo.load_dem(x_raw, y_raw, z_raw - z0)
        z = topo.drape_receiver_z(x, y, dem, receiver_height_above_surface)
        receiver_z = z

    F = params["forward.magneticField.intensity_nT"]
    I = params["forward.magneticField.inclination"]
    D = params["forward.magneticField.declination"]
    mesh, active, survey, sim, trend, trend_info, dobs, std, data_obj = _assemble(
        x, y, receiver_z, v, F, I, D, core_cell_m, core_cell_z_m, depth_core_m,
        pad_distance_m, regional_trend_degree, noise_pct, noise_floor_nt, dem)

    return dict(
        format="obs", topography_available=True,
        survey=survey, sim=sim, mesh=mesh, active_cells=active,
        dobs=dobs, std=std, data=data_obj,
        x=x, y=y, z=z,
        raw=dict(x=x_raw, y=y_raw, z=z_raw, v=v_raw, n=n_declared, z_shift_applied=z0),
        trend=trend, trend_info=trend_info,
        igrf=dict(F=F, I=I, D=D),
        native_mesh=mesh_meta, params=params,
        n_data_native=n_declared, n_data_used=x.size,
        n_active_cells=int(active.sum()), n_mesh_cells=int(mesh.n_cells),
        decimate_stride=decimate_stride, use_topography=use_topography, dem=dem,
    )


def _load_survey_grid(kw, decimate_stride, core_cell_m, core_cell_z_m, depth_core_m,
                      pad_distance_m, regional_trend_degree, noise_pct, noise_floor_nt,
                      use_topography, is_raster):
    igrf_date = kw.get("igrf_date") or DEFAULT_IGRF_DATE

    if is_raster:
        window, aoi = read_raster(kw["path"], aoi=kw.get("aoi"),
                                  value_scale=kw.get("value_scale", 1.0),
                                  field=kw.get("field", "magnetic"))
        x2d, y2d, v_flat = mesh3d.decimate_grid(window, decimate_stride or 1)
        x_raw, y_raw, v_raw = x2d, y2d, v_flat
        n_native = int(window.values.size)
    else:
        station, aoi = read_geojson(kw["path"], aoi=kw.get("aoi"), value_key=kw["value_key"],
                                    elevation_key=kw.get("elevation_key"),
                                    extra_keys=[kw["igrf_amplitude_key"]] if kw.get("igrf_amplitude_key") else None)
        x_raw, y_raw, v_raw = station.x, station.y, station.v
        n_native = int(x_raw.size)   # already sparse -- "native" == "used", no decimation axis

    amp_key = kw.get("igrf_amplitude_key")
    if not is_raster and amp_key and amp_key in station.extra:
        F = float(np.median(station.extra[amp_key]))
        _, I, D = igrf_mod.igrf_at_aoi(aoi, igrf_date)
    else:
        F, I, D = igrf_mod.igrf_at_aoi(aoi, igrf_date)

    if use_topography:
        print("[load_survey] use_topography=True requested but this format has no real "
             "elevation source -- proceeding with flat z=0 (see load_survey's module docstring).")

    mesh, active, survey, sim, trend, trend_info, dobs, std, data_obj = _assemble(
        x_raw, y_raw, None, v_raw, F, I, D, core_cell_m, core_cell_z_m, depth_core_m,
        pad_distance_m, regional_trend_degree, noise_pct, noise_floor_nt, dem=None)

    return dict(
        format="raster" if is_raster else "geojson", topography_available=False,
        survey=survey, sim=sim, mesh=mesh, active_cells=active,
        dobs=dobs, std=std, data=data_obj,
        x=x_raw, y=y_raw, z=np.zeros_like(x_raw),
        raw=dict(x=x_raw, y=y_raw, v=v_raw, n=n_native, z_shift_applied=0.0),
        trend=trend, trend_info=trend_info,
        igrf=dict(F=F, I=I, D=D, date=igrf_date),
        native_mesh=None, params=dict(aoi=aoi),
        n_data_native=n_native, n_data_used=x_raw.size,
        n_active_cells=int(active.sum()), n_mesh_cells=int(mesh.n_cells),
        decimate_stride=decimate_stride if is_raster else None,
        use_topography=False, dem=None,
    )
