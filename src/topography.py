"""Topography draping for the Tomofast-x benchmarks -- Ogarko et al.
(2024) Sect. 2.1 describes draping the model grid below a DEM surface
(their own mesh's per-column z-shift, visible in Fig. 2/3) rather than
inverting on a flat z=0 plane the way round 1 did. This module reproduces
that behaviour as a post-hoc `active_cells` mask over an UNCHANGED
`mesh3d.build_mesh` mesh -- it does not change how the mesh itself is
refined, only which of its cells count as "ground" vs "air".

DEM source, and why it's the receiver elevations, not the mesh file:
Synthetic400's `meshgrid_2depth.txt` cells DO shift slightly per (ix, iy)
column (checked directly -- e.g. column ix=1,iy=1 spans z -1475 to
+1748 m; column ix=220,iy=220 spans -1932 to +1290 m), but there is no
air/ground flag in the file (its one per-cell scalar column is a
uniform starting-model value, not a lithology/air marker -- see
load_survey.py's module docstring), so the exact air/ground
boundary per column isn't cleanly recoverable from it. `MAG_n5f1.OBS`'s
own Z column, by contrast, is unambiguous: a clean 400x400 grid at 25 m
spacing (confirmed directly -- max elevation change between adjacent
25 m samples is 45 m, consistent with real terrain, not noise), matching
Fig. 3a's "Synthetic400 model topography" description, and it's the same
file already used for the viewer's topography surface. That's the DEM
used here.
"""
import numpy as np
from scipy.interpolate import RegularGridInterpolator


def load_dem(obs_x, obs_y, obs_z):
    """Builds a DEM callable dem(x, y) -> elevation from the receiver
    grid's own (x, y, z) -- assumes a clean regular raster (checked by
    the caller/load_survey.read_obs, not re-validated here).
    Queries outside the native receiver footprint are clamped to the
    nearest edge value (`bounds_error=False, fill_value=None` would
    extrapolate linearly instead, which can run away unrealistically over
    the ~3 km of mesh padding beyond the receivers -- clamping just holds
    the boundary elevation flat into the padding, a deliberately
    conservative choice, not a claim that real terrain is flat out there)."""
    xu = np.unique(obs_x)
    yu = np.unique(obs_y)
    if xu.size * yu.size != obs_x.size:
        raise ValueError(f"receivers are not a clean {yu.size}x{xu.size} raster "
                         f"({xu.size * yu.size} != {obs_x.size}) -- load_dem assumes one")
    zg = obs_z.reshape(yu.size, xu.size)
    interp = RegularGridInterpolator((yu, xu), zg, method="linear",
                                     bounds_error=False, fill_value=None)
    x0, x1 = xu.min(), xu.max()
    y0, y1 = yu.min(), yu.max()

    def dem(x, y):
        xc = np.clip(x, x0, x1)
        yc = np.clip(y, y0, y1)
        return interp(np.c_[yc, xc])

    return dem


def drape_active_cells(mesh, dem):
    """Cells at or below the local DEM elevation directly above them --
    replaces mesh3d.active_cells_flat_surface's constant z_surface=0.0
    with a spatially-varying one. No air cells: a cell whose centre sits
    above the real local ground surface is excluded, even where the flat
    z=0 approximation (round 1) would have wrongly included it (anywhere
    real elevation is below the domain mean) or wrongly excluded a real
    near-surface cell (anywhere real elevation is above the mean -- up to
    +701 m for Synthetic400, checked directly against the DEM)."""
    cc = mesh.cell_centers
    surface_z = dem(cc[:, 0], cc[:, 1])
    return cc[:, 2] <= surface_z


def drape_receiver_z(x, y, dem, height_above_surface=0.0):
    """Receiver elevation = local DEM value + a constant drape height
    (default 0 -- ground-based magnetometer sitting directly on the
    surface; pass a positive value for a fixed-height airborne drape).
    NOT an absolute z -- the whole point of draping is that this follows
    the DEM's spatial variation, unlike round 1's single shared z."""
    return dem(x, y) + height_above_surface
