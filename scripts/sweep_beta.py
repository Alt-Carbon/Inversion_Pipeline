"""Demonstrates the kernel cache on a beta-value sweep -- 5 fixed
beta values, each run as if it were a separate script invocation (a fresh
mesh/survey/simulation object built from scratch per value, matching how
sweeping a hyperparameter across separate runs would actually be done),
compared with the cache disabled vs enabled. Reports wall time for each
and the overall reduction.

    cd "WEST AUS/Tomofastx2.0_models" && python -m scripts.sweep_beta
"""
import sys
import time
from pathlib import Path

import numpy as np
from simpeg import data_misfit, directives, inverse_problem, inversion, optimization, regularization
from simpeg import data as simpeg_data

_HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_HERE))
from config import config as cfg                                    # noqa: E402
from vendor.grav_mag_inversion.invert import _TIGHT_TOL              # noqa: E402
from src import load_survey as t2s                                   # noqa: E402
from src import kernel_cache as kc                                   # noqa: E402

BETAS = [1e4, 3e3, 1e3, 3e2, 1e2]
FIXED_ITERS = 5   # small, fixed iteration count per beta -- this sweep is about
                  # comparing G-computation cost, not re-running full to-
                  # convergence inversions 5 times over


def solve_fixed_beta(survey, sim, dobs, std, mesh, active_cells, m0, mref, beta):
    """One fixed-beta solve, no BetaSchedule/BetaEstimate/TargetMisfit --
    deliberately minimal (a handful of Gauss-Newton steps at a single
    beta) since the point of this script is measuring G-computation cost
    against solve cost, not producing a converged model per beta."""
    d = simpeg_data.Data(survey, dobs=dobs, standard_deviation=std)
    dmis = data_misfit.L2DataMisfit(data=d, simulation=sim)
    reg = regularization.WeightedLeastSquares(mesh, active_cells=active_cells,
                                              alpha_s=0.05, length_scale_x=1.0,
                                              reference_model=mref)
    opt = optimization.InexactGaussNewton(maxIter=FIXED_ITERS, maxIterLS=20, maxIterCG=30,
                                          tolCG=1e-4, **_TIGHT_TOL)
    inv_prob = inverse_problem.BaseInvProblem(dmis, reg, opt)
    inv_prob.beta = beta
    save = directives.SaveOutputEveryIteration(on_disk=False)
    inv = inversion.BaseInversion(inv_prob, directiveList=[directives.UpdateSensitivityWeights(), save])
    m_rec = inv.run(m0)
    return float(inv_prob.phi_d), m_rec


def fresh_geometry():
    """Rebuilds mesh/survey/sim from scratch -- simulating a new process
    picking up the same forward geometry for a different beta value."""
    return t2s.load_survey(
        cfg.FORMAT, cfg.FORMAT_KWARGS,
        decimate_stride=cfg.DECIMATE_STRIDE, core_cell_m=cfg.CORE_CELL_M,
        core_cell_z_m=cfg.CORE_CELL_Z_M, depth_core_m=cfg.DEPTH_CORE_M,
        pad_distance_m=cfg.PAD_DISTANCE_M)


def run_sweep(use_cache, cache_dir):
    times = []
    for beta in BETAS:
        d = fresh_geometry()
        n_active = d["n_active_cells"]
        m0 = np.zeros(n_active)
        mref = m0.copy()
        locs = d["sim"].survey.source_field.receiver_list[0].locations
        if use_cache:
            key = kc.cache_key(d["mesh"], d["active_cells"], locs,
                               d["igrf"]["F"], d["igrf"]["I"], d["igrf"]["D"])
            d["sim"], _path, is_new = kc.attach_cache(d["sim"], key, cache_dir)
        t0 = time.time()
        solve_fixed_beta(d["survey"], d["sim"], d["dobs"], d["std"], d["mesh"],
                         d["active_cells"], m0, mref, beta)
        dt = time.time() - t0
        times.append(dt)
        print(f"  beta={beta:g}: {dt:.1f}s" + (f"  (cache {'miss' if is_new else 'hit'})" if use_cache else ""))
    return times


def main():
    cache_dir = cfg.RESULTS_DIR / "kernel_cache_sweep_demo"
    kc.clear_cache(cache_dir)   # start clean so the first run below is a guaranteed miss

    print(f"--- sweep WITHOUT cache ({len(BETAS)} betas, {FIXED_ITERS} iters each, fresh sim per beta) ---")
    t_no_cache = run_sweep(use_cache=False, cache_dir=cache_dir)

    kc.clear_cache(cache_dir)   # clean slate for a fair "first run still pays for G" comparison
    print(f"\n--- sweep WITH cache ({len(BETAS)} betas, {FIXED_ITERS} iters each, fresh sim per beta) ---")
    t_cache = run_sweep(use_cache=True, cache_dir=cache_dir)

    total_no_cache = sum(t_no_cache)
    total_cache = sum(t_cache)
    reduction = 100 * (1 - total_cache / total_no_cache)
    print(f"\ntotal without cache: {total_no_cache:.1f}s")
    print(f"total with cache:    {total_cache:.1f}s")
    print(f"runtime reduction:   {reduction:.0f}%")
    print("(The cache targets the G-computation portion specifically -- the reduction "
         "on TOTAL sweep time also includes solve time, which the cache does not speed "
         "up, so the honest total-time number is well below G's own near-instant-on-a-"
         "hit speedup -- see improved_report.md.)")

    result = dict(betas=BETAS, t_no_cache=t_no_cache, t_cache=t_cache,
                 total_no_cache=total_no_cache, total_cache=total_cache, reduction_pct=reduction)
    import json
    out_path = cfg.RESULTS_DIR / "sweep_beta_result.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"-> {out_path}")
    return result


if __name__ == "__main__":
    main()
