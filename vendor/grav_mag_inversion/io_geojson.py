"""GeoJSON point-station loader for the India gravity/magnetic surveys --
different data shape from the WA project's rasters (WA: continuous grid,
decimated for tractability; India: already-sparse discrete station points,
used as-is, no decimation). Reuses the same local equirectangular
projection convention as grav_mag_inversion/src/io_data.py so downstream
code (mesh3d, forward, invert) is identical between the two projects.
"""
from dataclasses import dataclass, field
import json

import numpy as np


@dataclass
class StationData:
    x: np.ndarray
    y: np.ndarray
    v: np.ndarray          # bouguer_an (mGal) or magnetic_a (nT)
    elevation: np.ndarray  # metres, station surface elevation -- not yet used
                           # for receiver z (mesh assumes flat z=0, same as
                           # WA project); kept for a future topography pass.
    lon0: float
    lat0: float
    extra: dict = field(default_factory=dict)   # other requested columns, e.g. igrf_nt


def _local_xy(lon, lat, lon0, lat0):
    ew = 111320.0 * np.cos(np.deg2rad(lat0))
    return (lon - lon0) * ew, (lat - lat0) * 111320.0


def _dedupe(lon, lat, cols):
    """Average duplicate-coordinate stations (Puruliya has ~16% repeat
    coordinates -- re-surveys/toposheet-boundary overlaps, not decided by
    us) rather than silently keeping/dropping an arbitrary one of each
    group. `cols` is a dict of same-length arrays to average alongside
    lon/lat (elevation, and any extra_keys)."""
    coords = np.stack([lon, lat], axis=1)
    uniq, inverse = np.unique(coords, axis=0, return_inverse=True)
    inverse = inverse.ravel()
    n = len(uniq)
    out = {}
    for name, arr in cols.items():
        s = np.zeros(n)
        np.add.at(s, inverse, arr)
        out[name] = s
    cnt = np.zeros(n)
    np.add.at(cnt, inverse, 1)
    for name in out:
        out[name] = out[name] / cnt
    return uniq[:, 0], uniq[:, 1], out


def load_station_geojson(path, aoi, value_key, elevation_key=None, extra_keys=None):
    """Load one geojson FeatureCollection of Point stations. `aoi` sets the
    projection origin (lon0/lat0 = AOI centroid, matching config.AOI), not
    a crop -- these files are already one district each, no windowing
    needed. `value_key`/`elevation_key` are the properties keys to pull
    (gravity: 'bouguer_an'/'elevation_'; magnetics: 'magnetic_a', no
    elevation field in that file -- pass None). `extra_keys`: additional
    property names to load verbatim (deduped/averaged the same way as
    value/elevation), returned in `StationData.extra` -- e.g. `igrf_nt`,
    which is real per-station IGRF total-intensity data already present in
    the magnetic file (see run_magnetics.py for why this matters: it's a
    better source for the forward model's inducing-field amplitude than
    guessing a survey date)."""
    with open(path) as f:
        data = json.load(f)
    feats = data["features"]
    lon = np.array([f["geometry"]["coordinates"][0] for f in feats])
    lat = np.array([f["geometry"]["coordinates"][1] for f in feats])
    val = np.array([f["properties"][value_key] for f in feats], dtype=float)
    elev = (np.array([f["properties"][elevation_key] for f in feats], dtype=float)
           if elevation_key else np.zeros_like(val))
    extras = {k: np.array([f["properties"][k] for f in feats], dtype=float)
             for k in (extra_keys or [])}

    ok = np.isfinite(val)
    lon, lat, val, elev = lon[ok], lat[ok], val[ok], elev[ok]
    extras = {k: v[ok] for k, v in extras.items()}
    lon, lat, deduped = _dedupe(lon, lat, dict(value=val, elevation=elev, **extras))

    lon0 = (aoi["west"] + aoi["east"]) / 2
    lat0 = (aoi["south"] + aoi["north"]) / 2
    x, y = _local_xy(lon, lat, lon0, lat0)
    extra_out = {k: deduped[k] for k in extras}
    return StationData(x, y, deduped["value"], deduped["elevation"], lon0, lat0, extra_out)
