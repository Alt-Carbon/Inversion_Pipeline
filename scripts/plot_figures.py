"""2D figures for any dataset's baseline + improved runs (TOMOFASTX_
BENCHMARK/TOMOFASTX_DATA_DIR, see config.py) -- reuses vendor.grav_mag_
inversion.plotting's helpers UNCHANGED (format-agnostic: they already
take raw x/y/model arrays, not anything WA/India-specific), including
their existing `boundary_loops` support for drawing a real (non-
rectangular) survey outline on the section locator panels -- wired up
here via src/coverage.py, ported from india_grav_mt_inversion for
exactly this.

    cd "WEST AUS/Tomofastx2.0_models" && python -m scripts.plot_figures
"""
import sys
from types import SimpleNamespace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_HERE))
from config import config as cfg                                    # noqa: E402
from vendor.grav_mag_inversion.plotting import (                     # noqa: E402
    plot_data_fit, plot_convergence, plot_model_sections, _slice_locs, style,
    C_S1, C_S2, C_INK, C_MUTED, C_GRID, C_SURF)
from vendor.grav_mag_inversion import mesh3d                         # noqa: E402
from src import coverage, validation as val                          # noqa: E402
from src import topography as topo                                   # noqa: E402
from src import load_survey as t2s                                    # noqa: E402


def plot_study_area(x, y, boundary_loops, improved_npz_path, out_path, mark_bodies):
    """Plan view: receivers used (decimated grid), the real survey outline
    (src/coverage.py -- Synthetic400's own footprint is a dense
    rectangle, so this mostly traces its edge, but the same code handles
    genuinely irregular coverage unchanged if pointed at it later), and --
    ONLY when `mark_bodies` (Synthetic400 has a known body list to check
    recovery against; Callisto has none, see baseline_report.md's "Scope"
    -- val.BODIES' coordinates are Synthetic400-specific and meaningless
    for any other benchmark's coordinate range, so this must stay opt-in,
    not benchmark-agnostic) -- each body's actual RECOVERED peak (from the
    improved result -- see build_viewer.py's module docstring for why a
    single shared reference per body, not a digitised guess at ground
    truth, was settled on after the 3D viewer's own body markers went
    through the same revision)."""
    fig, ax = plt.subplots(figsize=(6.5, 6.2))
    ax.scatter(x, y, s=3, c=C_GRID, linewidths=0, label="receivers used", zorder=1)
    for loop in boundary_loops:
        ax.plot(loop[:, 0], loop[:, 1], color=C_S2, lw=1.3, alpha=0.9, zorder=2,
               label="survey outline" if loop is boundary_loops[0] else None)

    if mark_bodies:
        d = np.load(improved_npz_path)
        labels = val.per_body_report(d["cc"], d["model"], d["cell_volumes"])
        for row in labels:
            if not row.get("recovered"):
                continue
            # per_body_report doesn't return the peak's own (x, y) directly --
            # recompute via recovered_blob_near the same way build_viewer.py
            # does, same-sign search near the body's approximate true position.
            b = next(bb for bb in val.BODIES if bb["id"] == row["id"])
            sign = 1 if b["susceptibility_si"] >= 0 else -1
            blob = val.recovered_blob_near(d["cc"], d["model"], b["true_xy"], expected_sign=sign)
            if blob is None:
                continue
            px, py, _pz = blob["peak_xyz"]
            ax.scatter([px], [py], s=60, c=[b["susceptibility_si"]], cmap="Reds",
                      vmin=0, vmax=max(bb["susceptibility_si"] for bb in val.BODIES),
                      edgecolors=C_INK, linewidths=1.1, zorder=5)
            ax.annotate(str(b["id"]), (px, py), ha="center", va="center", fontsize=8,
                       fontweight="600", color=C_INK, zorder=6)
        subtitle = "numbered points = RECOVERED peak (improved), not ground truth"
    else:
        subtitle = "no published ground truth to check recovery against -- see README Scope"
    ax.set_aspect("equal")
    # UTM northing (~9.8e6 for Synthetic400, ~6.8e6 for Callisto) makes
    # matplotlib draw a "1e6" offset label above the y-axis by default --
    # it collides with style()'s subtitle line every time (this was NOT
    # actually fixed by shortening the subtitle text alone, despite
    # earlier notes to that effect -- confirmed by re-inspecting the
    # rendered PNG). Plain (non-offset) tick labels avoid the collision
    # without touching style() itself.
    ax.ticklabel_format(style="plain", useOffset=False)
    style(ax, f"{cfg.BENCHMARK}: study area", subtitle,
         xlabel="easting (m)", ylabel="northing (m)")
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def make_figures(npz_name, out_prefix, label, boundary_loops):
    """Generates the same figure set for one saved result. Whether the
    mesh/active-cell mask reconstructed here (not stored in the npz, only
    the model values are -- rebuilt purely for plot_model_sections'
    geometry needs) should be topography-draped is read from the npz's
    OWN `use_topography` field (set by run_baseline.py/run_improved.py
    to what they actually built), not assumed from which of the two this
    call is for -- an "improved" run on a raster/geojson dataset has no
    topography source and was built flat regardless (see src/load_survey.
    py), so trusting a per-call guess here would silently reconstruct the
    wrong mesh and misalign active_idx against the saved model array."""
    d = np.load(cfg.RESULTS_DIR / npz_name)
    x, y, dobs, predicted = d["x"], d["y"], d["dobs"], d["predicted"]
    cc, model = d["cc"], d["model"]
    n_data = dobs.size
    use_topography = bool(d["use_topography"]) if "use_topography" in d else False

    res = SimpleNamespace(method=str(d["method"]), phi_d_history=list(d["phi_d_history"]),
                          phi_m_history=list(d["phi_m_history"]))

    rms = plot_data_fit(x, y, dobs, dobs, predicted, "nT",
                        cfg.FIGURES_DIR / f"{out_prefix}_data_fit.png", f"{cfg.BENCHMARK} {label}")
    print(f"[{label}] data fit: RMS = {rms:.3f} nT ({100*rms/np.abs(dobs).max():.1f}% of dobs peak)")

    plot_convergence(res, cfg.CHI_TARGET, n_data, cfg.FIGURES_DIR / f"{out_prefix}_convergence.png",
                     f"{cfg.BENCHMARK} {label}")

    # Reconstruct a throwaway mesh identical to the one the inversion used
    # (same seed/params, deterministic) purely for plot_model_sections'
    # geometry -- must drape the SAME way the actual run did, or active_idx
    # won't line up with the saved model array.
    mesh = mesh3d.build_mesh((x, y), float(d["core_cell_m"]), float(d["depth_core_m"]),
                             float(d["pad_distance_m"]), core_cell_z_m=float(d["core_cell_z_m"]))
    if use_topography:
        x_raw, y_raw, z_raw, v_raw, _n = t2s.read_obs(cfg.DATA_DIR / cfg.OBS_FILE)
        z0 = t2s.flat_surface_z(z_raw)
        dem = topo.load_dem(x_raw, y_raw, z_raw - z0)
        active = topo.drape_active_cells(mesh, dem)
    else:
        active = mesh3d.active_cells_flat_surface(mesh, z_surface=0.0)
    active_idx = np.where(active)[0]
    assert active_idx.size == model.size, (active_idx.size, model.size)

    depth_lim = cfg.VOLUME_DEPTH_LIM if use_topography else cfg.SECTION_DEPTH_LIM
    plot_model_sections(mesh, active_idx, model, x, y, _slice_locs(y.min(), y.max()), "SI",
                        cfg.FIGURES_DIR / f"{out_prefix}_model_section_ew.png",
                        f"{res.method} susceptibility ({label}) -- east-west sections",
                        normal="Y", label_start="A", depth_lim=depth_lim, boundary_loops=boundary_loops)
    plot_model_sections(mesh, active_idx, model, x, y, _slice_locs(x.min(), x.max()), "SI",
                        cfg.FIGURES_DIR / f"{out_prefix}_model_section_ns.png",
                        f"{res.method} susceptibility ({label}) -- north-south sections",
                        normal="X", label_start="D", depth_lim=depth_lim, boundary_loops=boundary_loops)


def main():
    cfg.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    base_npz = cfg.RESULTS_DIR / f"{cfg.BENCHMARK}_result.npz"
    impr_npz = cfg.RESULTS_DIR / f"{cfg.BENCHMARK}_improved_result.npz"
    d0 = np.load(base_npz)
    x_all, y_all = d0["x"], d0["y"]
    boundary_loops = coverage.coverage_contour(x_all, y_all, cfg.COVERAGE_MAX_DIST_M)
    print(f"study area outline: {len(boundary_loops)} loop(s)")

    make_figures(base_npz.name, cfg.BENCHMARK, "baseline (flat topography, smooth L2)",
                boundary_loops=boundary_loops)
    make_figures(impr_npz.name, f"{cfg.BENCHMARK}_improved", "improved (sparse/IRLS"
                + (", real topography)" if cfg.FORMAT == "obs" else ", flat topography -- "
                   "no topography source for this format)"), boundary_loops=boundary_loops)

    plot_study_area(x_all, y_all, boundary_loops, impr_npz, cfg.FIGURES_DIR / f"{cfg.BENCHMARK}_study_area.png",
                    mark_bodies=(cfg.BENCHMARK == "synthetic400"))

    print(f"-> {cfg.FIGURES_DIR}")


if __name__ == "__main__":
    main()
