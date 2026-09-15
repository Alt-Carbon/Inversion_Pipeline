"""Vendored from grav_mag_inversion/scripts/run_magnetics.py's `igrf_at_aoi`
(see VENDORED.md) -- computes the inducing field (F/I/D) at an AOI's
centroid via the IGRF-14 model (`ppigrf`), for data sources that don't
carry a real per-station or per-file inducing-field value of their own
(unlike Tomofast's TOMOPARAMS.TXT, or India's real per-station igrf_nt --
see src/readers.py's module docstring for when this is actually used)."""
import datetime

import numpy as np
import ppigrf


def igrf_at_aoi(aoi, date_str):
    """`aoi`: dict with west/east/south/north (degrees). `date_str`: ISO
    date, e.g. "2019-06-01" -- IGRF is time-varying, so this is a real,
    disclosed choice, not a formality (see callers for what date they use
    and why)."""
    lon0 = (aoi["west"] + aoi["east"]) / 2
    lat0 = (aoi["south"] + aoi["north"]) / 2
    d = datetime.datetime.fromisoformat(date_str)
    Be, Bn, Bu = ppigrf.igrf(lon0, lat0, 0.0, d)
    Be, Bn, Bu = float(Be[0]), float(Bn[0]), float(Bu[0])
    F = (Be**2 + Bn**2 + Bu**2) ** 0.5
    D = np.degrees(np.arctan2(Be, Bn))
    I = np.degrees(np.arctan2(-Bu, (Be**2 + Bn**2) ** 0.5))
    return F, I, D
