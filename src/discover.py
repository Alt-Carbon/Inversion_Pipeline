"""Auto-detects which of the three supported input formats a data folder
holds, and the specific file(s) needed -- so a new dataset can be pointed
at directly (TOMOFASTX_DATA_DIR) instead of requiring hand-written config.
Three formats, told apart by what's actually in the folder (never by a
user-supplied hint that could be wrong):

  "obs"     a *.OBS file (+ TOMOPARAMS*.TXT + meshgrid*.txt) -- Tomofast-x's
            own plain-text format.
  "raster"  a *.tif/*.tiff/*.ers file -- a geographic grid (the format
            grav_mag_inversion's own WA pipeline uses).
  "geojson" a *.geojson file -- point stations (the format india_grav_mt_
            inversion's own pipeline uses).

See src/load_survey.py's module docstring for what each format actually
contains and how it's read."""
import re
from pathlib import Path


def _find(folder, pattern):
    rx = re.compile(pattern, re.IGNORECASE)
    return sorted(p.name for p in folder.iterdir() if p.is_file() and rx.match(p.name))


def _find_one(folder, pattern, kind):
    matches = _find(folder, pattern)
    if len(matches) == 0:
        present = sorted(p.name for p in folder.iterdir() if p.is_file())
        raise FileNotFoundError(
            f"No {kind} file found in {folder} (looked for a name matching {pattern!r}). "
            f"Files present: {present}")
    if len(matches) > 1:
        raise ValueError(
            f"Multiple {kind} candidates found in {folder}: {matches} -- auto-detection needs "
            "exactly one match; pass the filename explicitly instead of relying on auto-"
            "detection, or move/rename the extra file.")
    return matches[0]


def discover_files(folder):
    """OBS format only (kept for backward compatibility with callers that
    already know they have OBS data) -- returns (obs_name, tomoparams_
    name, meshgrid_name). Prefer `detect_format` for a folder of unknown
    format."""
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"data folder not found: {folder}")
    obs_name = _find_one(folder, r".*\.obs$", "observation (*.OBS)")
    tomoparams_name = _find_one(folder, r".*tomoparams.*\.txt$", "TOMOPARAMS (*TOMOPARAMS*.TXT)")
    meshgrid_name = _find_one(folder, r"meshgrid.*\.txt$", "meshgrid (meshgrid*.txt)")
    return obs_name, tomoparams_name, meshgrid_name


def detect_format(folder):
    """Inspects `folder` and returns (fmt, info) where fmt is "obs",
    "raster", or "geojson", and info is a dict of the bare filename(s)
    found (interpretation -- which is OBS vs TOMOPARAMS vs meshgrid, or
    which raster/geojson file if several match -- is `discover_files`'s
    job for "obs", or the caller's choice for "raster"/"geojson" when
    more than one candidate exists). Priority when a folder somehow has
    more than one format's marker file: obs > raster > geojson (obs is
    the most specific/unambiguous marker -- a *.OBS file basically never
    means anything else); raise instead of guessing is NOT done here for
    the multi-raster/multi-geojson case specifically, since a real folder
    might legitimately hold e.g. both a gravity and a magnetic raster --
    see raster_names/geojson_names below, and pick the one you want via
    the relevant TOMOFASTX_* override (config.py)."""
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"data folder not found: {folder}")

    obs_names = _find(folder, r".*\.obs$")
    if obs_names:
        obs_name, tomoparams_name, meshgrid_name = discover_files(folder)
        return "obs", dict(obs_name=obs_name, tomoparams_name=tomoparams_name,
                           meshgrid_name=meshgrid_name)

    raster_names = _find(folder, r".*\.(tif|tiff|ers)$")
    if raster_names:
        return "raster", dict(raster_names=raster_names)

    geojson_names = _find(folder, r".*\.geojson$")
    if geojson_names:
        return "geojson", dict(geojson_names=geojson_names)

    present = sorted(p.name for p in folder.iterdir() if p.is_file())
    raise FileNotFoundError(
        f"No recognised input file found in {folder} -- looked for *.OBS (Tomofast-x format), "
        f"*.tif/*.tiff/*.ers (raster grid), or *.geojson (point stations). Files present: {present}")
