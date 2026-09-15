"""Load a WA gravity + magnetic AOI window into local metric coordinates.

Both source grids are GDA94 geographic (lat/lon). For a small AOI (tens of
km) a local equirectangular projection -- the same one used throughout this
project for distance calculations -- is accurate enough; a proper UTM/MGA
reprojection would matter for anything larger.
"""
from dataclasses import dataclass

import numpy as np
import rasterio
from rasterio.windows import from_bounds


@dataclass
class GridWindow:
    values: np.ndarray       # 2D, north-up (row 0 = north edge)
    mask: np.ndarray         # True = nodata
    x_m: np.ndarray          # 1D easting, metres, local to AOI centre
    y_m: np.ndarray          # 1D northing, metres, local to AOI centre
    lon0: float
    lat0: float


def _local_xy(lon, lat, lon0, lat0):
    ew = 111320.0 * np.cos(np.deg2rad(lat0))
    return (lon - lon0) * ew, (lat - lat0) * 111320.0


def load_gravity_window(path, aoi, micro_ms2_per_mgal):
    lon0 = (aoi["west"] + aoi["east"]) / 2
    lat0 = (aoi["south"] + aoi["north"]) / 2
    with rasterio.open(path) as s:
        w = from_bounds(aoi["west"], aoi["south"], aoi["east"], aoi["north"], s.transform)
        a = s.read(1, window=w, masked=True)
        tr = s.window_transform(w)
    vals = np.ma.filled(a, np.nan).astype(float) / micro_ms2_per_mgal   # -> mGal
    ny, nx = vals.shape
    lon = tr.c + (np.arange(nx) + 0.5) * tr.a
    lat = tr.f + (np.arange(ny) + 0.5) * tr.e   # tr.e is negative (north-up)
    x, _ = _local_xy(lon, lat0, lon0, lat0)
    _, y = _local_xy(lon0, lat, lon0, lat0)
    return GridWindow(vals, ~np.isfinite(vals), x, y, lon0, lat0)


def load_magnetic_window(path, aoi):
    lon0 = (aoi["west"] + aoi["east"]) / 2
    lat0 = (aoi["south"] + aoi["north"]) / 2
    with rasterio.open(path) as s:
        w = from_bounds(aoi["west"], aoi["south"], aoi["east"], aoi["north"], s.transform)
        a = s.read(1, window=w, masked=True)
        tr = s.window_transform(w)
    vals = np.ma.filled(a, np.nan).astype(float)   # already nT (see config.py note)
    ny, nx = vals.shape
    lon = tr.c + (np.arange(nx) + 0.5) * tr.a
    lat = tr.f + (np.arange(ny) + 0.5) * tr.e
    x, _ = _local_xy(lon, lat0, lon0, lat0)
    _, y = _local_xy(lon0, lat, lon0, lat0)
    return GridWindow(vals, ~np.isfinite(vals), x, y, lon0, lat0)
