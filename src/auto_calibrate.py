"""Automatic mesh/decimation calibration for an arbitrary dataset --
generalises the manual process this project used for both shipped
benchmarks (build a candidate mesh, measure real active-cell count
directly, compute the dense sensitivity matrix size, target staying
safely under a RAM budget -- see config.py's per-benchmark comments for
the numbers this was checked against: Synthetic400 stride=6/100 m ->
2,489 receivers/~1.9 GB, Callisto stride=3/20 m -> 3,953 receivers/
2.65 GB, both measured, not guessed).

This module does the same search automatically for a NEW dataset (one
with no hand-calibrated entry in config.py's BENCHMARKS registry): try
decimation strides from fine to coarse, and at each one build the ACTUAL
reduced mesh (not an estimate) to measure active cells and the resulting
dense-matrix size, stopping at the finest stride that fits the budget.
Building a handful of candidate meshes costs a few seconds each --
trivial next to the inversion itself, which is why this checks the real
number instead of extrapolating from a formula.

`core_cell_m` at each candidate stride is set to a fixed fraction of the
decimated receiver spacing (`core_cell_ratio`, default 0.6 -- splits the
difference between Synthetic400's 100/150=0.67 and Callisto's 20/37.5=
0.53, both chosen independently for those two datasets), rounded to the
nearest 5 m. This is a reasonable starting point for a dataset with no
prior calibration, not a substitute for the hand-checked BENCHMARKS
entries this repo ships for its own two datasets -- config.py prefers
those whenever the benchmark name matches."""
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_HERE))
from vendor.grav_mag_inversion import mesh3d             # noqa: E402
from src import load_survey as t2s                       # noqa: E402

DEFAULT_STRIDES = (1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50)


def _round_to(value, step, minimum):
    return max(round(value / step) * step, minimum)


class _TimedOut(Exception):
    pass


def _build_and_count(xy_points, core_cell_m, depth_core_m, pad_distance_m, core_cell_z_m, q):
    """Runs in a CHILD PROCESS (see _build_mesh_with_timeout) -- returns
    only plain ints through the queue, not the mesh object itself (a
    discretize.TreeMesh isn't a cheap/reliable thing to pickle back
    across the process boundary, and nothing upstream of this needs the
    mesh object, only n_cells/n_active)."""
    try:
        mesh = mesh3d.build_mesh(xy_points, core_cell_m, depth_core_m, pad_distance_m,
                                 core_cell_z_m=core_cell_z_m)
        active = mesh3d.active_cells_flat_surface(mesh, z_surface=0.0)
        q.put(("ok", int(mesh.n_cells), int(active.sum())))
    except Exception as e:   # noqa: BLE001 -- reported back, not swallowed
        q.put(("error", str(e)))


def _build_mesh_with_timeout(xy_points, core_cell_m, depth_core_m, pad_distance_m,
                             core_cell_z_m, timeout_s):
    """Hard wall-clock guard around one candidate mesh build, via a
    SEPARATE PROCESS killed on timeout -- NOT signal/SIGALRM-based (that
    was tried first and does not work here: confirmed directly against a
    real hang -- discretize's TreeMesh.refine_bounding_box/finalize are
    compiled extension code that doesn't check for pending Python signals
    during its own internal loop, so a queued SIGALRM sits undelivered
    until that call returns on its own, which defeats the whole point of
    a timeout for exactly the pathological case it exists to catch).
    Only a real OS-level process kill is reliable against code that could
    be stuck in a C extension, which is why this crosses a process
    boundary at all despite the overhead (`fork` start method -- cheap,
    copy-on-write, no re-import cost) rather than staying in-process.

    Returns (n_cells, n_active); raises _TimedOut or RuntimeError (the
    latter wrapping any exception the child raised) rather than a mesh
    object -- see _build_and_count's docstring for why."""
    ctx = mp.get_context("fork")
    q = ctx.Queue()
    p = ctx.Process(target=_build_and_count,
                    args=(xy_points, core_cell_m, depth_core_m, pad_distance_m, core_cell_z_m, q))
    p.start()
    p.join(timeout_s)
    if p.is_alive():
        p.terminate()
        p.join(2)
        if p.is_alive():
            p.kill()
            p.join()
        raise _TimedOut()
    if q.empty():
        raise RuntimeError(f"mesh-build subprocess exited (code {p.exitcode}) without a result "
                           "-- likely killed by the OS (out of memory?) rather than crashing "
                           "with a catchable Python exception.")
    status, *rest = q.get()
    if status == "error":
        raise RuntimeError(f"mesh build failed: {rest[0]}")
    return rest   # [n_cells, n_active]


def calibrate_resolution(x_raw, y_raw, ram_budget_gb=4.0, core_cell_ratio=0.6,
                         core_cell_z_ratio=0.5, depth_core_m=3000.0, pad_distance_m=None,
                         strides=DEFAULT_STRIDES, verbose=True, mesh_build_timeout_s=30):
    """Returns a dict with decimate_stride/core_cell_m/core_cell_z_m/
    depth_core_m/pad_distance_m ready to pass straight to
    load_survey.load_survey, plus `diagnostics` (every
    candidate tried, so the choice is auditable, not a black box) and
    `chosen` (the same dict duplicated for convenience).

    `x_raw`/`y_raw`: the FULL native receiver coordinates (not yet
    decimated) -- must form a clean rectangular raster (same requirement
    as decimate_receiver_grid; checked here too, so this fails with the
    same clear error rather than a confusing one deeper in the search).

    `pad_distance_m` (default None): if not given, derived from the
    survey's own footprint (20% of the larger horizontal extent, clamped
    to [1500, 3000] m) rather than hard-coded -- Synthetic400 (10 km
    footprint) and Callisto (2.5 km footprint) both used values in that
    range but for different reasons (paper's stated core depth vs. a
    smaller survey needing less padding relative to its own size), so
    this scales with the actual data instead of assuming one or the
    other."""
    xu, yu = np.unique(x_raw), np.unique(y_raw)
    if xu.size * yu.size != x_raw.size:
        raise ValueError(
            f"receivers are not a clean {yu.size}x{xu.size} raster "
            f"({xu.size * yu.size} != {x_raw.size}) -- auto_calibrate needs the same regular-"
            "grid layout decimate_receiver_grid does; not valid for flight-line/scattered data.")
    dx = float(np.median(np.diff(xu))) if xu.size > 1 else 0.0
    dy = float(np.median(np.diff(yu))) if yu.size > 1 else 0.0
    native_spacing = (dx + dy) / 2 if (dx and dy) else max(dx, dy)
    if native_spacing <= 0:
        raise ValueError(f"could not determine a native receiver spacing (dx={dx}, dy={dy})")

    if pad_distance_m is None:
        extent = max(x_raw.max() - x_raw.min(), y_raw.max() - y_raw.min())
        pad_distance_m = float(np.clip(extent * 0.2, 1500.0, 3000.0))

    z_dummy = np.zeros_like(x_raw)
    v_dummy = np.zeros_like(x_raw)

    x_extent = float(x_raw.max() - x_raw.min()) + 2 * pad_distance_m
    y_extent = float(y_raw.max() - y_raw.min()) + 2 * pad_distance_m

    diagnostics = []
    chosen = None
    for stride in strides:
        if xu.size // stride < 2 or yu.size // stride < 2:
            break   # too few receivers left along one axis to form a mesh at all
        decimated_spacing = native_spacing * stride
        core_cell_m = _round_to(decimated_spacing * core_cell_ratio, 5.0, 5.0)
        core_cell_z_m = _round_to(core_cell_m * core_cell_z_ratio, 5.0, 5.0)

        # Cheap pre-filter BEFORE paying to build the actual mesh: build_mesh
        # rounds each horizontal axis up to the next power of 2 cells at
        # core_cell_m (mesh3d._pow2_tensor) and finely refines the whole
        # receiver footprint (refine_bounding_box) -- a footprint-cell
        # estimate this coarse catches a pathologically fine early stride
        # (seen in practice: stride=1 on Synthetic400's native 25 m spacing
        # tried to build a 15 m-core mesh over a 10 km domain and used 5.7
        # GB of RAM just constructing ONE candidate mesh) without spending
        # the time/RAM actually building it.
        est_nx = 2 ** int(np.ceil(np.log2(max(int(np.ceil(x_extent / core_cell_m)), 1))))
        est_ny = 2 ** int(np.ceil(np.log2(max(int(np.ceil(y_extent / core_cell_m)), 1))))
        if est_nx * est_ny > 300_000:
            diagnostics.append(dict(stride=stride, core_cell_m=core_cell_m, core_cell_z_m=core_cell_z_m,
                                    skipped="footprint-cell estimate too large to build "
                                            f"({est_nx}x{est_ny}={est_nx*est_ny:,})"))
            if verbose:
                print(f"  [calibrate] stride={stride:>3}  core_cell_m={core_cell_m:>6.0f}  "
                     f"skipped (estimated {est_nx}x{est_ny} footprint cells, too fine to even build)")
            continue

        x_dec, y_dec, _z, _v = t2s.decimate_receiver_grid(x_raw, y_raw, z_dummy, v_dummy, stride)
        try:
            _n_cells, n_active = _build_mesh_with_timeout(
                (x_dec, y_dec), core_cell_m, depth_core_m, pad_distance_m,
                core_cell_z_m, mesh_build_timeout_s)
        except _TimedOut:
            diagnostics.append(dict(stride=stride, core_cell_m=core_cell_m, core_cell_z_m=core_cell_z_m,
                                    skipped=f"mesh build exceeded {mesh_build_timeout_s}s timeout"))
            if verbose:
                print(f"  [calibrate] stride={stride:>3}  core_cell_m={core_cell_m:>6.0f}  "
                     f"skipped (mesh build exceeded {mesh_build_timeout_s}s -- estimate above was "
                     "wrong for this footprint shape, caught by the timeout instead)")
            continue
        n_receivers = int(x_dec.size)
        matrix_gb = n_receivers * n_active * 8 / 1e9

        row = dict(stride=stride, core_cell_m=core_cell_m, core_cell_z_m=core_cell_z_m,
                  n_receivers=n_receivers, n_active=n_active, matrix_gb=matrix_gb)
        diagnostics.append(row)
        if verbose:
            print(f"  [calibrate] stride={stride:>3}  core_cell_m={core_cell_m:>6.0f}  "
                 f"receivers={n_receivers:>7,}  active_cells={n_active:>9,}  "
                 f"matrix={matrix_gb:>7.2f} GB" + ("  <- chosen" if matrix_gb <= ram_budget_gb else ""))
        if matrix_gb <= ram_budget_gb:
            chosen = row
            break

    if chosen is None:
        coarsest = diagnostics[-1] if diagnostics else "none -- grid too small for any stride in `strides`"
        raise RuntimeError(
            f"No decimation stride up to {strides[-1]} brought the dense matrix under "
            f"{ram_budget_gb} GB (coarsest tried: {coarsest}). Pass a larger `ram_budget_gb`, a "
            "coarser `strides` list, or a smaller `core_cell_ratio`.")

    return dict(decimate_stride=chosen["stride"], core_cell_m=chosen["core_cell_m"],
               core_cell_z_m=chosen["core_cell_z_m"], depth_core_m=depth_core_m,
               pad_distance_m=pad_distance_m, chosen=chosen, diagnostics=diagnostics,
               native_spacing_m=native_spacing)


def calibrate_resolution_sparse(x, y, ram_budget_gb=4.0, core_cell_z_ratio=0.5,
                                depth_core_m=3000.0, pad_distance_m=None,
                                cell_multipliers=(1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64),
                                verbose=True, mesh_build_timeout_s=30):
    """Same idea as calibrate_resolution, for data with no decimation axis
    at all (GeoJSON point stations -- already-sparse, fixed receiver
    count, unlike a regular raster/receiver grid that can be strided).
    The receiver COUNT is fixed, so the only lever left is `core_cell_m`
    itself: start near the median nearest-neighbour station spacing
    (india_grav_mt_inversion's own rule of thumb -- see its config.py:
    "CORE_CELL_M picked as ~half each district's own median station
    nearest-neighbour spacing") and coarsen (`cell_multipliers`) until
    the dense matrix fits the budget."""
    from scipy.spatial import cKDTree

    n = x.size
    if n < 2:
        raise ValueError(f"only {n} station(s) -- need at least 2 to estimate a spacing")
    tree = cKDTree(np.c_[x, y])
    nn_dist, _ = tree.query(np.c_[x, y], k=2)   # k=1 is the point itself (dist 0)
    median_spacing = float(np.median(nn_dist[:, 1]))
    base_core_cell_m = _round_to(median_spacing * 0.5, 5.0, 5.0)

    if pad_distance_m is None:
        extent = max(x.max() - x.min(), y.max() - y.min())
        pad_distance_m = float(np.clip(extent * 0.2, 1500.0, 20000.0))

    x_extent = float(x.max() - x.min()) + 2 * pad_distance_m
    y_extent = float(y.max() - y.min()) + 2 * pad_distance_m

    diagnostics = []
    chosen = None
    for mult in cell_multipliers:
        core_cell_m = base_core_cell_m * mult
        core_cell_z_m = _round_to(core_cell_m * core_cell_z_ratio, 5.0, 5.0)

        est_nx = 2 ** int(np.ceil(np.log2(max(int(np.ceil(x_extent / core_cell_m)), 1))))
        est_ny = 2 ** int(np.ceil(np.log2(max(int(np.ceil(y_extent / core_cell_m)), 1))))
        if est_nx * est_ny > 300_000:
            diagnostics.append(dict(core_cell_m=core_cell_m, core_cell_z_m=core_cell_z_m,
                                    skipped=f"footprint-cell estimate too large ({est_nx}x{est_ny})"))
            if verbose:
                print(f"  [calibrate] core_cell_m={core_cell_m:>7.1f}  skipped (estimated "
                     f"{est_nx}x{est_ny} footprint cells, too fine to even build)")
            continue

        try:
            _n_cells, n_active = _build_mesh_with_timeout(
                (x, y), core_cell_m, depth_core_m, pad_distance_m,
                core_cell_z_m, mesh_build_timeout_s)
        except _TimedOut:
            diagnostics.append(dict(core_cell_m=core_cell_m, core_cell_z_m=core_cell_z_m,
                                    skipped=f"mesh build exceeded {mesh_build_timeout_s}s timeout"))
            if verbose:
                print(f"  [calibrate] core_cell_m={core_cell_m:>7.1f}  skipped (mesh build "
                     f"exceeded {mesh_build_timeout_s}s)")
            continue
        matrix_gb = n * n_active * 8 / 1e9

        row = dict(core_cell_m=core_cell_m, core_cell_z_m=core_cell_z_m,
                  n_receivers=n, n_active=n_active, matrix_gb=matrix_gb)
        diagnostics.append(row)
        if verbose:
            print(f"  [calibrate] core_cell_m={core_cell_m:>7.1f}  receivers={n:>7,}  "
                 f"active_cells={n_active:>9,}  matrix={matrix_gb:>7.2f} GB" +
                 ("  <- chosen" if matrix_gb <= ram_budget_gb else ""))
        if matrix_gb <= ram_budget_gb:
            chosen = row
            break

    if chosen is None:
        coarsest = diagnostics[-1] if diagnostics else "none tried"
        raise RuntimeError(
            f"No core_cell_m up to {base_core_cell_m * cell_multipliers[-1]:.0f} m brought the "
            f"dense matrix under {ram_budget_gb} GB (coarsest tried: {coarsest}). This dataset's "
            f"receiver count itself ({n:,}) may be the limiting factor (no decimation is possible "
            "for point-station data) -- pass a larger `ram_budget_gb` or a coarser "
            "`cell_multipliers` list.")

    return dict(decimate_stride=None, core_cell_m=chosen["core_cell_m"],
               core_cell_z_m=chosen["core_cell_z_m"], depth_core_m=depth_core_m,
               pad_distance_m=pad_distance_m, chosen=chosen, diagnostics=diagnostics,
               median_station_spacing_m=median_spacing)
