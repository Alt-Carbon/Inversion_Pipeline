"""3D OcTree mesh for the AOI, built with discretize's mesh_builder_xyz --
using the library's own padded-mesh construction rather than hand-rolling
padding/expansion logic (the kind of thing that's easy to get subtly wrong,
as with the 1D MT mesh's half-space padding cell earlier in this project).
"""
import numpy as np
import discretize


def decimate_grid(window, stride):
    """Every `stride`-th pixel of a GridWindow, as (x, y, values) 1D arrays,
    nodata dropped. Keeps the receiver/data count tractable for the dense
    sensitivity matrix that Simulation3DIntegral builds."""
    v = window.values[::stride, ::stride]
    m = window.mask[::stride, ::stride]
    yy, xx = np.meshgrid(window.y_m[::stride], window.x_m[::stride], indexing="ij")
    ok = ~m
    return xx[ok], yy[ok], v[ok]


def _pow2_tensor(extent_m, cell_m):
    """Uniform cell-width array at the finest level, count rounded up to
    the next power of 2 (required for an octree)."""
    n = int(np.ceil(extent_m / cell_m))
    n = 2 ** int(np.ceil(np.log2(max(n, 1))))
    return np.full(n, cell_m)


def build_mesh(xy_points, core_cell_m, depth_core_m, pad_distance_m, air_pad_m=None,
              refine_level=-1, core_cell_z_m=None):
    """OcTree mesh built directly (not via mesh_builder_xyz, which produced
    a degenerate 1-2 cell mesh for this point cloud -- see git history /
    README for what was tried). Finest cells (core_cell_m) span the full
    padded domain at construction; refine_points then forces the finest
    level near the receivers, and the octree's own one-level-difference
    balance rule coarsens everything else automatically -- no manual
    expansion-factor schedule to get wrong.

    `core_cell_z_m`: vertical cell size at the finest level, independent of
    the horizontal `core_cell_m` -- defaults to `core_cell_m` (cubic cells,
    the original behaviour). discretize's TreeMesh only requires a common
    octree *index* across the three axes, not cubic cells -- each axis
    keeps its own base tensor, so a finer hz just means more vertical
    levels of refinement inside the same horizontal footprint. Set this
    independently of horizontal resolution because the two are governed by
    different things: horizontal cell size is tied to receiver spacing (see
    config.py CORE_CELL_M comment), while vertical resolution is what
    limits how sharply a sparse/IRLS inversion (see README) can localise a
    body in depth -- receiver spacing doesn't constrain that the same way.
    """
    air_pad_m = air_pad_m if air_pad_m is not None else pad_distance_m * 0.3
    core_cell_z_m = core_cell_z_m if core_cell_z_m is not None else core_cell_m
    x_extent = (xy_points[0].max() - xy_points[0].min()) + 2 * pad_distance_m
    y_extent = (xy_points[1].max() - xy_points[1].min()) + 2 * pad_distance_m
    z_extent = depth_core_m + pad_distance_m + air_pad_m

    hx = _pow2_tensor(x_extent, core_cell_m)
    hy = _pow2_tensor(y_extent, core_cell_m)
    hz = _pow2_tensor(z_extent, core_cell_z_m)

    x0 = (xy_points[0].min() + xy_points[0].max()) / 2 - hx.sum() / 2
    y0 = (xy_points[1].min() + xy_points[1].max()) / 2 - hy.sum() / 2
    z0 = air_pad_m - hz.sum()   # top of mesh sits air_pad_m above z=0

    mesh = discretize.TreeMesh([hx, hy, hz], origin=[x0, y0, z0])

    # refine_points only refines the specific leaf cell each point lands in
    # -- with ~20k receiver points that produced a nearly-empty mesh (123
    # cells total, only 8 at the finest level), not the densely-refined
    # core intended. refine_bounding_box refines the whole box spanned by
    # the points, which is what a survey-wide core region needs; padding
    # steps the refinement down gradually toward the domain edges.
    surface_xyz = np.c_[xy_points[0], xy_points[1], np.zeros_like(xy_points[0])]
    mesh.refine_bounding_box(surface_xyz, level=refine_level,
                             padding_cells_by_level=[2, 2, 2, 4, 4], finalize=False)
    mesh.finalize()
    return mesh


def active_cells_flat_surface(mesh, z_surface=0.0):
    """Cells at or below a flat surface at z=z_surface -- a simplification
    (no real topography/DEM used yet); documented as a limitation."""
    return mesh.cell_centers[:, 2] <= z_surface
