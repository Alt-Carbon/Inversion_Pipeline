"""Improved driver -- real topography draping + sparse/IRLS regularization
on top of the same load path scripts/run_baseline.py uses, for whichever
dataset config.py currently points at (TOMOFASTX_BENCHMARK or
TOMOFASTX_DATA_DIR -- see config.py's module docstring). This is what's
shipped in results/<name>_improved_result.npz, figures/, and the 3D
viewer for both benchmarks this repo carries.

For Synthetic400 specifically, this configuration has a documented,
unresolved caveat -- see improved_report.md's "Root cause investigation":
every recovered body sits next to a larger opposite-sign structure
(traced through several fixes -- r0 calibration, real sensitivity
weighting, beta floors, bound-constrained/reparameterised optimizers --
none of which fully eliminated it this round). Sparse/IRLS measurably
improves same-sign recovery quality over smooth L2 regardless, which is
why it's the default here rather than a straight swap back.

Kernel caching (src/kernel_cache.py) is exercised here (topography
changes the active-cell set from the baseline run, so this is always a
fresh cache entry on a new dataset) -- see scripts/sweep_beta.py for the
caching speedup demonstration on repeated runs.

    cd "WEST AUS/Tomofastx2.0_models" && python -m scripts.run_improved
    TOMOFASTX_BENCHMARK=callisto python -m scripts.run_improved
    TOMOFASTX_DATA_DIR=/path/to/your/survey python -m scripts.run_improved
"""
import resource
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_HERE))
from config import config as cfg                      # noqa: E402
from src import load_survey as t2s                      # noqa: E402
from src import invert_v2, kernel_cache as kc            # noqa: E402


def main():
    np.random.seed(cfg.SEED)
    cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rss_start = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    t_load0 = time.time()
    d = t2s.load_survey(
        cfg.FORMAT, cfg.FORMAT_KWARGS,
        decimate_stride=cfg.DECIMATE_STRIDE, core_cell_m=cfg.CORE_CELL_M,
        core_cell_z_m=cfg.CORE_CELL_Z_M, depth_core_m=cfg.DEPTH_CORE_M,
        pad_distance_m=cfg.PAD_DISTANCE_M, use_topography=True,
        noise_pct=cfg.NOISE_PCT, noise_floor_nt=cfg.NOISE_FLOOR_NT,
    )
    t_load = time.time() - t_load0

    topo_note = "topography-draped" if d["topography_available"] else \
        "flat z=0 -- no topography source for this format, see load_survey.py"
    print(f"{cfg.BENCHMARK} (improved, {d['format']}): {d['n_data_used']:,} receivers, "
         f"{d['n_active_cells']:,} active cells ({topo_note})")

    locs = d["sim"].survey.source_field.receiver_list[0].locations
    if d["topography_available"]:
        print(f"receiver elevation range: {locs[:,2].min():.0f} to {locs[:,2].max():.0f} m")

    n_active = d["n_active_cells"]
    key = kc.cache_key(d["mesh"], d["active_cells"], locs,
                       d["igrf"]["F"], d["igrf"]["I"], d["igrf"]["D"])
    print(f"kernel cache key: {key}")

    m0 = np.zeros(n_active)
    mref = m0.copy()

    print(f"\n--- sparse IRLS (norms={cfg.SPARSE_NORMS}, irls_cooling_factor="
         f"{cfg.IRLS_COOLING_FACTOR}, max_irls_iterations={cfg.MAX_IRLS_ITERATIONS} -- "
         "same as india_grav_mt_inversion's run_magnetics.py), topography-draped mesh, "
         "real per-cell sensitivity weighting ---")
    t_inv0 = time.time()
    res, cache_was_new = invert_v2.run_sparse_v2(
        d["survey"], d["sim"], d["dobs"], d["std"], d["mesh"], d["active_cells"], m0, mref,
        chi_target=cfg.CHI_TARGET, norms=cfg.SPARSE_NORMS,
        irls_cooling_factor=cfg.IRLS_COOLING_FACTOR, max_irls_iterations=cfg.MAX_IRLS_ITERATIONS,
        cache_key=key)
    t_inv = time.time() - t_inv0

    chi = res.phi_d / d["dobs"].size
    rss_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    print(f"phi_d = {res.phi_d:.2f}  (target {res.phi_d_target:.2f}, chi = {chi:.2f}, "
         f"{len(res.phi_d_history)} iterations)")
    print(f"\nload time: {t_load:.1f}s   inversion time: {t_inv:.1f}s   total: {t_load+t_inv:.1f}s")
    print(f"peak RSS: {rss_peak/1e6:.2f} GB (started at {rss_start/1e6:.2f} GB)")

    active_idx = np.where(d["active_cells"])[0]
    cc = d["mesh"].cell_centers[active_idx]
    cell_volumes = d["mesh"].cell_volumes[active_idx]
    np.savez(cfg.RESULTS_DIR / f"{cfg.BENCHMARK}_improved_result.npz",
            x=d["x"], y=d["y"], dobs=d["dobs"], cc=cc, cell_volumes=cell_volumes,
            model=res.model, predicted=res.predicted, method=res.method, format=d["format"],
            phi_d_history=res.phi_d_history, phi_m_history=res.phi_m_history,
            beta_history=res.beta_history,
            n_data_native=d["n_data_native"], n_data_used=d["n_data_used"],
            n_active_cells=n_active, n_mesh_cells=d["n_mesh_cells"],
            decimate_stride=d["decimate_stride"] or 0, core_cell_m=cfg.CORE_CELL_M,
            core_cell_z_m=cfg.CORE_CELL_Z_M, depth_core_m=cfg.DEPTH_CORE_M,
            pad_distance_m=cfg.PAD_DISTANCE_M, chi=chi,
            load_time_s=t_load, inversion_time_s=t_inv, peak_rss_gb=rss_peak / 1e6,
            z_shift_applied=d["raw"]["z_shift_applied"],
            weighting="depth_real_sensitivity", regularization="sparse_IRLS",
            use_topography=d["topography_available"], cache_key=key, cache_was_new=cache_was_new)
    print(f"\n-> {cfg.RESULTS_DIR / (cfg.BENCHMARK + '_improved_result.npz')}")
    return res, d


if __name__ == "__main__":
    main()
