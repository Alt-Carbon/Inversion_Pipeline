"""Baseline driver -- the existing (UNCHANGED) grav_mag_inversion smooth
L2 pipeline (invert.run_smooth: WeightedLeastSquares + InexactGaussNewton,
default depth weighting via UpdateSensitivityWeights, BetaSchedule cooling,
flat z=0 topography, induced magnetisation only) on ANY dataset selected
via config.py (TOMOFASTX_BENCHMARK for the two known, hand-calibrated
ones shipped with this repo, or TOMOFASTX_DATA_DIR for any other folder,
in any of the three supported formats -- see config.py's module
docstring). No changes to src/invert.py, no topography/distance-
weighting/kernel-caching/ADMM additions -- that's what scripts/
run_improved.py adds.

Runs at reduced resolution relative to whatever the source data's own
native receiver density is -- see src/load_survey.py's module docstring
and baseline_report.md for why (dense-matrix method, no MPI/wavelet
compression) and src/auto_calibrate.py for how the reduction is chosen
automatically for a new dataset.

    cd "WEST AUS/Tomofastx2.0_models" && python -m scripts.run_baseline
    TOMOFASTX_BENCHMARK=callisto python -m scripts.run_baseline
    TOMOFASTX_DATA_DIR=/path/to/your/survey python -m scripts.run_baseline
"""
import resource
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_HERE))
from config import config as cfg                  # noqa: E402
from vendor.grav_mag_inversion import invert       # noqa: E402
from src import load_survey as t2s                 # noqa: E402


def main():
    np.random.seed(cfg.SEED)
    cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rss_start = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    t_load0 = time.time()
    d = t2s.load_survey(
        cfg.FORMAT, cfg.FORMAT_KWARGS,
        decimate_stride=cfg.DECIMATE_STRIDE, core_cell_m=cfg.CORE_CELL_M,
        core_cell_z_m=cfg.CORE_CELL_Z_M, depth_core_m=cfg.DEPTH_CORE_M,
        pad_distance_m=cfg.PAD_DISTANCE_M, noise_pct=cfg.NOISE_PCT, noise_floor_nt=cfg.NOISE_FLOOR_NT,
    )
    t_load = time.time() - t_load0

    if d["decimate_stride"]:
        print(f"{cfg.BENCHMARK} ({d['format']}): native {d['n_data_native']:,} receivers -> "
             f"{d['n_data_used']:,} used (stride {d['decimate_stride']})")
    else:
        print(f"{cfg.BENCHMARK} ({d['format']}): {d['n_data_used']:,} stations (no decimation "
             "-- already-sparse point data)")
    if d["native_mesh"]:
        print(f"native mesh (source file's own, not reused): {d['native_mesh']['n_cells']:,} cells "
             f"(x {d['native_mesh']['x_extent']}, z {d['native_mesh']['z_extent']})")
    print(f"reduced mesh: {d['n_mesh_cells']:,} cells, {d['n_active_cells']:,} active")
    if d["format"] == "obs":
        igrf_source = "from TOMOPARAMS, exact"
    else:
        igrf_source = f"IGRF-14 model at survey centroid, {d['igrf'].get('date', '?')}"
    print(f"IGRF ({igrf_source}): F={d['igrf']['F']:.1f} nT, "
         f"I={d['igrf']['I']:.1f} deg, D={d['igrf']['D']:.1f} deg")
    print(f"dobs: {d['dobs'].min():.1f} to {d['dobs'].max():.1f} nT "
         f"(regional trend explained {100*d['trend_info']['var_explained']:.1f}% of raw variance)")
    print(f"assumed std: {d['std'].min():.2f} to {d['std'].max():.2f} nT "
         f"({100*cfg.NOISE_PCT:g}% of |dobs|, {cfg.NOISE_FLOOR_NT} nT floor -- "
         f"a starting assumption unless overridden, see config.py)")

    n_active = d["n_active_cells"]
    n_data = d["n_data_used"]
    G_bytes = n_data * n_active * 8
    print(f"dense sensitivity matrix: {n_data:,} x {n_active:,} = {G_bytes/1e9:.2f} GB (float64)")

    m0 = np.zeros(n_active)
    mref = m0.copy()

    print("\n--- smooth L2 (WeightedLeastSquares + InexactGaussNewton), invert.py UNCHANGED, "
         "function defaults only (max_iter=30, alpha_s=0.05, length_scale_x=1.0) ---")
    t_inv0 = time.time()
    res = invert.run_smooth(d["survey"], d["sim"], d["dobs"], d["std"], d["mesh"],
                            d["active_cells"], m0, mref, chi_target=cfg.CHI_TARGET)
    t_inv = time.time() - t_inv0

    n_data_ = d["dobs"].size
    chi = res.phi_d / n_data_
    rss_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    print(f"phi_d = {res.phi_d:.2f}  (target {res.phi_d_target:.2f}, chi = {chi:.2f}, "
         f"{len(res.phi_d_history)} iterations)")
    print(f"\nload time: {t_load:.1f}s   inversion time: {t_inv:.1f}s   total: {t_load+t_inv:.1f}s")
    print(f"peak RSS: {rss_peak/1e6:.2f} GB (started at {rss_start/1e6:.2f} GB)")

    active_idx = np.where(d["active_cells"])[0]
    cc = d["mesh"].cell_centers[active_idx]
    cell_volumes = d["mesh"].cell_volumes[active_idx]
    np.savez(cfg.RESULTS_DIR / f"{cfg.BENCHMARK}_result.npz",
            x=d["x"], y=d["y"], dobs=d["dobs"], cc=cc, cell_volumes=cell_volumes,
            model=res.model, predicted=res.predicted, method=res.method,
            phi_d_history=res.phi_d_history, phi_m_history=res.phi_m_history,
            beta_history=res.beta_history, format=d["format"],
            n_data_native=d["n_data_native"], n_data_used=d["n_data_used"],
            n_active_cells=n_active, n_mesh_cells=d["n_mesh_cells"],
            decimate_stride=d["decimate_stride"] or 0, core_cell_m=cfg.CORE_CELL_M,
            core_cell_z_m=cfg.CORE_CELL_Z_M, depth_core_m=cfg.DEPTH_CORE_M,
            pad_distance_m=cfg.PAD_DISTANCE_M, chi=chi,
            g_bytes=G_bytes, load_time_s=t_load, inversion_time_s=t_inv,
            peak_rss_gb=rss_peak / 1e6, use_topography=False,
            z_shift_applied=d["raw"]["z_shift_applied"])
    print(f"\n-> {cfg.RESULTS_DIR / (cfg.BENCHMARK + '_result.npz')}")
    return res, d


if __name__ == "__main__":
    main()
