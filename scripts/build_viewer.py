"""Interactive 3D viewer, benchmark-agnostic (TOMOFASTX_BENCHMARK, see
config.py) -- BOTH the round-1 baseline (flat z=0 topography, depth
weighting) and round-2 improved (real topography, sparse/IRLS) recovered
susceptibility isosurfaces, independently toggleable, plus a real
topography surface. Body-position labels (see recovered_labels below)
are Synthetic400-only -- no published ground truth exists for any other
benchmark to label against.

Earlier versions marked the 6 true bodies' digitised (from Fig. 4, by
eye) approximate horizontal positions with a dashed line through the
whole model -- removed at the user's request (the reading was never
precise enough, +-300-500 m, no depth at all, to serve as a reliable
on-screen reference implying a known position). Replaced with a plain
number label placed at each body's actual RECOVERED peak location
(src/validation.py's recovered_blob_near, same-sign search near the
body's approximate true position -- so the label still answers "which
body is this," but its (x, y, z) comes from what the inversion itself
produced, not a guess at ground truth).

One label set only, computed from the IMPROVED result and shared by both
layers -- an earlier version computed separate labels per layer (each
model's own recovered peak), but that meant the same body's number sat
at a different spot depending on which layer was on, which read as
inconsistent/confusing when switching layers rather than as the real
finding it was (baseline and improved genuinely do recover each body at
different positions -- see src/validation.py's per-body centroid_offset_m
for that comparison numerically instead). A single shared reference
point per body is easier to read at a glance; the labels stay fixed
while you switch which isosurface is shown underneath them.

    cd "WEST AUS/Tomofastx2.0_models" && python -m scripts.build_viewer
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

_HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_HERE))
from config import config as cfg                      # noqa: E402
from src import coverage, load_survey as t2s            # noqa: E402
from src import topography as topo_mod                  # noqa: E402
from src import validation as val                       # noqa: E402

VIEWER_DIR = cfg.VIEWER_DIR

NX, NY, NZ = 34, 34, 28
TARGET_TOPO_POINTS_PER_AXIS = 60   # display resolution for the topo surface, not physical accuracy

# MARGIN/FAR_THRESHOLD used to be fixed metre values tuned for
# Synthetic400's ~10 km survey / 100 m core cells -- fine there, but on
# Callisto's much smaller ~2.5 km survey (20 m core cells) the same 1500 m
# MARGIN pushed the viewer's own bounding box to more than half again the
# actual survey width, and the same 400 m FAR_THRESHOLD sampled far
# coarser than the underlying 20 m mesh actually resolves. Both now scale
# with cfg.CORE_CELL_M -- the natural length scale of whichever dataset
# is currently selected -- instead of assuming Synthetic400's own scale.
MARGIN = 10.0 * cfg.CORE_CELL_M
FAR_THRESHOLD = 4.0 * cfg.CORE_CELL_M   # ~4x core_cell_m -- see WA/India build_viewer.py precedent


def topo_surface(z_shift):
    """Real receiver elevation, decimated for a smooth display surface --
    "obs"-format datasets only (the only format with a real per-receiver
    elevation source, see src/load_survey.py); callers must check
    cfg.FORMAT == "obs" before calling this. The decimation stride is
    picked here (not a fixed constant) to land near
    TARGET_TOPO_POINTS_PER_AXIS regardless of the native receiver grid's
    own size -- a fixed stride tuned for Synthetic400's 400x400 native
    grid (e.g. stride=7 -> ~57x57 points) would produce a wildly
    different point count on a dataset with a different native density
    (e.g. Callisto's native 200x176 grid at the same stride -> ~29x25,
    fine; but a much denser or sparser dataset could end up with either
    a near-empty surface or one far too heavy for the viewer). z_shift
    matches both models' own recentring (see load_survey.flat_surface_z)
    so the surface lines up with them -- both runs share the same z0
    (computed from the same raw receiver file), so one topo surface
    serves both."""
    x, y, z, v_raw, _n = t2s.read_obs(cfg.DATA_DIR / cfg.OBS_FILE)
    native_nx, native_ny = np.unique(x).size, np.unique(y).size
    topo_stride = max(1, min(native_nx, native_ny) // TARGET_TOPO_POINTS_PER_AXIS)
    xg, yg, zg, _ = t2s.decimate_receiver_grid(x, y, z - z_shift, v_raw, topo_stride)
    xu = np.unique(xg)
    yu = np.unique(yg)
    zg2d = zg.reshape(yu.size, xu.size)
    return dict(x=[round(float(v), 1) for v in xu], y=[round(float(v), 1) for v in yu],
               z=[[round(float(v), 1) for v in row] for row in zg2d]), zg2d, xu, yu


def build_volume(npz_path, x0, x1, y0, y1, z0, z1, ground_fn):
    """One model's isosurface volume. `ground_fn(x, y) -> elevation`
    defines "above ground" for masking -- a constant-zero function for
    the flat-topography baseline, the real DEM for the topography-draped
    improved run (using a flat z=0 cutoff for the draped run would wrongly
    hide real near-surface recovery wherever true terrain sits above 0 m,
    up to +701 m here -- see src/topography.py)."""
    d = np.load(npz_path)
    cc, model = d["cc"], d["model"]

    xa = np.linspace(x0, x1, NX)
    ya = np.linspace(y0, y1, NY)
    za = np.linspace(z0, z1, NZ)
    X, Y, Z = np.meshgrid(xa, ya, za, indexing="ij")
    pts = np.c_[X.ravel(), Y.ravel(), Z.ravel()]

    dist, idx = cKDTree(cc).query(pts, k=1)
    vals = model[idx]
    vals[dist > FAR_THRESHOLD] = np.nan

    ground_z = ground_fn(X.ravel(), Y.ravel())
    vals[Z.ravel() > ground_z] = np.nan

    return dict(
        x=[round(float(v), 1) for v in xa], y=[round(float(v), 1) for v in ya],
        z=[round(float(v), 1) for v in za],
        x_flat=[round(float(v), 1) for v in X.ravel()],
        y_flat=[round(float(v), 1) for v in Y.ravel()],
        z_flat=[round(float(v), 1) for v in Z.ravel()],
        values=[None if np.isnan(v) else round(float(v), 6) for v in vals],
        lim=float(np.nanpercentile(np.abs(vals), 99.5)),
        chi=float(d["chi"]),
        n_data=int(d["n_data_used"]), n_active_cells=int(d["n_active_cells"]),
        core_cell_m=float(d["core_cell_m"]), core_cell_z_m=float(d["core_cell_z_m"]),
    ), d


def recovered_labels(npz_path, search_radius_m=1500.0):
    """One label per body: its RECOVERED peak location in this specific
    model (same-sign search near the body's approximate true position,
    src/validation.py's recovered_blob_near) -- the position shown is
    what the inversion produced, not the digitised guess used only to
    know which recovered feature to look near. Skips a body if no
    same-sign cell was found nearby (recovered_blob_near returns None).

    val.BODIES is Synthetic400-specific (coordinates digitised from that
    benchmark's own Fig. 4, nowhere near Callisto's coordinate range) --
    return no labels for any other benchmark rather than silently relying
    on recovered_blob_near's search radius to come up empty."""
    if cfg.BENCHMARK != "synthetic400":
        return []
    d = np.load(npz_path)
    cc, model = d["cc"], d["model"]
    labels = []
    for b in val.BODIES:
        sign = 1 if b["susceptibility_si"] >= 0 else -1
        blob = val.recovered_blob_near(cc, model, b["true_xy"], search_radius_m, expected_sign=sign)
        if blob is None:
            continue
        x, y, z = blob["peak_xyz"]
        labels.append(dict(id=b["id"], shape=b["shape"], true_susceptibility_si=b["susceptibility_si"],
                           recovered_value=blob["peak_value"], x=round(float(x), 1),
                           y=round(float(y), 1), z=round(float(z), 1)))
    return labels


def _brand_text(display_name, n_labels):
    """Header name/subtitle -- format-aware (was hardcoded to Synthetic400's
    own text before, which meant every other dataset's page showed a
    false "SYNTHETIC400 -- ... Ogarko et al. 2024 benchmark" header;
    caught by actually reading a published WA/India page, not assumed)."""
    name = f"{display_name.upper()} -- RECOVERED SUSCEPTIBILITY"
    if cfg.BENCHMARK == "synthetic400":
        sub = "baseline vs improved &middot; Ogarko et al. 2024 benchmark, reduced resolution"
    elif cfg.FORMAT == "obs":
        sub = "baseline vs improved &middot; Tomofast-x format, pipeline/feasibility check"
    else:
        fmt_label = {"raster": "GeoTIFF/ERS raster", "geojson": "GeoJSON point"}[cfg.FORMAT]
        sub = (f"baseline vs improved &middot; real {fmt_label} survey, "
              "pipeline/feasibility check -- no ground truth")
    return name, sub


def _note_html(n_labels, has_topo):
    """Sidebar explanatory note -- format-aware for the same reason as
    _brand_text (this used to be one block of Synthetic400-specific text,
    hardcoded, shown unchanged on every dataset's page: real per-receiver-
    count ratios, report filenames, and a sign-flip finding that was only
    ever investigated ON Synthetic400 -- asserting it for any other
    dataset would be a claim this project has no evidence for)."""
    parts = []
    if cfg.BENCHMARK == "synthetic400":
        parts.append(
            "<b>Read with care:</b> this run uses ~1/36 of the native receivers and "
            "a coarser mesh than the paper (15&nbsp;GB RAM ceiling -- see "
            "baseline_report.md).")
    else:
        parts.append(
            "<b>Read with care:</b> resolution is decimated/coarsened to fit a RAM "
            "budget on one machine (no MPI/wavelet compression) -- see README.md "
            "\"How resolution is chosen\".")
    if has_topo:
        parts.append("The tan surface is the real per-station topography.")
    else:
        parts.append(
            "This data format has no real per-receiver elevation source (see README "
            "\"Supported input formats\") -- both layers assume a flat surface, so "
            "\"improved\" differs from baseline only by regularization method, not topography.")
    if n_labels:
        parts.append(
            "Numbered markers (#1-#6) sit at each body's actual RECOVERED peak in the "
            "IMPROVED model specifically (same position shown under both layers, so it "
            "doesn't jump when you switch -- baseline's own recovered peaks sit somewhere "
            "slightly different; see src/validation.py's centroid_offset_m for that "
            "difference numerically), not a digitised guess at ground truth. See "
            "baseline_report.md's \"Data and Ground Truth\" section for the approximate "
            "true positions used only to know which recovered feature is which body.")
        parts.append(
            "<b>Improved run caveat (see improved_report.md's \"Sign check\"):</b> every "
            "body's correctly-signed peak in the improved model sits next to a "
            "larger-magnitude opposite-sign artifact -- a real finding, not a display bug, "
            "if the improved layer looks messier than the baseline.")
    else:
        parts.append(
            "No published ground truth exists for this dataset to check recovered "
            "positions against -- this is a pipeline/feasibility result only, not a "
            "validated one (see README \"Known limitations\").")
    return " ".join(parts)


def main():
    base_npz = cfg.RESULTS_DIR / f"{cfg.BENCHMARK}_result.npz"
    impr_npz = cfg.RESULTS_DIR / f"{cfg.BENCHMARK}_improved_result.npz"

    d0 = np.load(base_npz)
    x_all, y_all = d0["x"], d0["y"]
    x0, x1 = x_all.min() - MARGIN, x_all.max() + MARGIN
    y0, y1 = y_all.min() - MARGIN, y_all.max() + MARGIN
    z0, z1 = cfg.VOLUME_DEPTH_LIM
    z_shift = float(d0["z_shift_applied"])

    boundary_loops = coverage.coverage_contour(x_all, y_all, cfg.COVERAGE_MAX_DIST_M)
    boundary = [dict(x=[round(float(v), 1) for v in loop[:, 0]],
                     y=[round(float(v), 1) for v in loop[:, 1]])
               for loop in boundary_loops]
    print(f"study area outline: {len(boundary)} loop(s)")

    # Real topography surface + draped ground function -- "obs" format
    # only (the only one with a real per-receiver elevation source, see
    # src/load_survey.py). Other formats fall back to a flat z=0 ground
    # for both layers, matching what run_improved.py actually built for
    # them (bool(d["use_topography"]) is False in their improved npz too
    # -- see plot_figures.py's make_figures for the same check).
    if cfg.FORMAT == "obs":
        topo, topo_zg, topo_xu, topo_yu = topo_surface(z_shift)
        dem = topo_mod.load_dem(np.repeat(topo_xu, topo_yu.size),
                                np.tile(topo_yu, topo_xu.size),
                                topo_zg.T.ravel())   # rebuild dem() from the same decimated grid used for display
        improved_ground_fn = lambda x, y: dem(x, y)
    else:
        topo = None
        improved_ground_fn = lambda x, y: np.zeros_like(x)

    baseline, _ = build_volume(base_npz, x0, x1, y0, y1, z0, z1,
                               ground_fn=lambda x, y: np.zeros_like(x))
    improved, _ = build_volume(impr_npz, x0, x1, y0, y1, z0, z1,
                               ground_fn=improved_ground_fn)

    labels = recovered_labels(impr_npz)   # shared by both layers -- see module docstring
    if cfg.BENCHMARK == "synthetic400":
        print(f"labels: {len(labels)}/6 bodies had a same-sign recovered peak nearby (from the improved result)")
    else:
        print("labels: none (no published ground-truth body list for this benchmark)")

    data = dict(
        baseline=baseline, improved=improved, topo=topo,
        labels=labels, boundary=boundary,
        meta=dict(),
    )

    print(f"baseline: grid {NX}x{NY}x{NZ}, 99.5th pctile |value| = {baseline['lim']:.4g}, "
         f"chi = {baseline['chi']:.2f}, active cells = {baseline['n_active_cells']:,}")
    print(f"improved: grid {NX}x{NY}x{NZ}, 99.5th pctile |value| = {improved['lim']:.4g}, "
         f"chi = {improved['chi']:.2f}, active cells = {improved['n_active_cells']:,}")

    template = (VIEWER_DIR / "template.html").read_text()
    # Short (<=2 char) words are treated as an acronym (e.g. "wa" -> "WA")
    # rather than title-cased into "Wa" -- a reasonable heuristic for
    # place-name abbreviations, not foolproof for every possible dataset name.
    display_name = " ".join(w.upper() if len(w) <= 2 else w.capitalize()
                            for w in cfg.BENCHMARK.replace("-", "_").split("_"))
    brand_name, brand_sub = _brand_text(display_name, len(labels))
    out = template.replace("__TITLE__", f"{display_name} Recovered Model")
    out = out.replace("__BRAND_NAME__", brand_name).replace("__BRAND_SUB__", brand_sub)
    out = out.replace("__NOTE_HTML__", _note_html(len(labels), cfg.FORMAT == "obs"))
    out = out.replace("__VOLUME_DATA_JSON__", json.dumps(data, separators=(",", ":")))

    # "model3d.html" is the name already shipped/linked for synthetic400
    # (README, published viewer) -- kept as-is so that link doesn't move;
    # any other benchmark gets its own file so it can't silently overwrite it.
    out_name = "model3d.html" if cfg.BENCHMARK == "synthetic400" else f"model3d_{cfg.BENCHMARK}.html"
    out_path = VIEWER_DIR / out_name
    out_path.write_text(out)
    print(f"-> {out_path}  ({len(out)/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
