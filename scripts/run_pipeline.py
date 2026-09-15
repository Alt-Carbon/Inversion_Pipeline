"""Runs the full pipeline end to end for one dataset with a single
command -- input data in, baseline + improved inversions, 2D figures, 3D
viewer, and (optionally) an S3 upload of the results. Each stage is also
runnable on its own (see README.md's "How to run it") -- this is a thin
convenience wrapper, not a replacement for understanding what each stage
does; it just sets the same TOMOFASTX_* environment variables config.py
already reads (see config.py's module docstring) and calls each stage's
existing main() in sequence.

    # a known, hand-calibrated benchmark shipped with this repo:
    python -m scripts.run_pipeline --benchmark callisto

    # any other dataset -- format (Tomofast-x .OBS, GeoTIFF/ERS raster,
    # or GeoJSON points) is auto-detected from what's in the folder:
    python -m scripts.run_pipeline --data-dir /path/to/your/survey --ram-budget-gb 8

    # ... and upload the results afterwards:
    python -m scripts.run_pipeline --benchmark callisto --upload-s3-bucket my-bucket
"""
import argparse
import os
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_HERE))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--benchmark", choices=["synthetic400", "callisto"], default=None,
                     help="a known, hand-calibrated benchmark from config.py's BENCHMARKS registry")
    src.add_argument("--data-dir", default=None,
                     help="path to any other data folder -- format (Tomofast-x .OBS, GeoTIFF/ERS "
                          "raster, or GeoJSON points) is auto-detected, see src/discover.py and "
                          "src/load_survey.py's module docstrings")
    ap.add_argument("--name", default=None,
                    help="output filename prefix for --data-dir mode (default: the folder's own "
                         "basename) -- ignored with --benchmark")
    ap.add_argument("--ram-budget-gb", type=float, default=None,
                    help="dense sensitivity matrix budget for auto-calibrating a --data-dir dataset "
                         "(default: 4.0, src/auto_calibrate.py's own default) -- ignored with "
                         "--benchmark, which uses its own hand-checked numbers instead")
    ap.add_argument("--skip-improved", action="store_true",
                    help="run only the baseline (flat topography, smooth L2) stage -- skips the "
                         "topography+sparse/IRLS stage and everything downstream of it that needs "
                         "both results (figures, viewer)")
    ap.add_argument("--upload-s3-bucket", default=None,
                    help="if given, upload results/figures/viewer to this S3 bucket afterwards "
                         "(see scripts/upload_to_s3.py -- needs the `aws` CLI already configured)")
    ap.add_argument("--upload-s3-prefix", default="tomofastx",
                    help="key prefix inside --upload-s3-bucket (default: tomofastx)")
    args = ap.parse_args()

    if args.benchmark is None and args.data_dir is None:
        ap.error("pass --benchmark <name> or --data-dir <path>")
    if args.benchmark:
        os.environ["TOMOFASTX_BENCHMARK"] = args.benchmark
        os.environ.pop("TOMOFASTX_DATA_DIR", None)
    else:
        os.environ["TOMOFASTX_DATA_DIR"] = args.data_dir
        if args.name:
            os.environ["TOMOFASTX_NAME"] = args.name
        if args.ram_budget_gb:
            os.environ["TOMOFASTX_RAM_BUDGET_GB"] = str(args.ram_budget_gb)

    # Imported AFTER the env vars above are set -- config.py resolves
    # DATA_DIR/DECIMATE_STRIDE/etc. (and, for a brand new --data-dir
    # dataset, runs auto-calibration) at import time.
    from config import config as cfg
    from scripts import run_baseline, run_improved, plot_figures, build_viewer

    print(f"\n{'='*70}\nPIPELINE: {cfg.BENCHMARK}\n{'='*70}")

    t0 = time.time()
    print(f"\n--- [1/4] baseline (flat topography, smooth L2) ---")
    run_baseline.main()

    if not args.skip_improved:
        print(f"\n--- [2/4] improved (real topography, sparse/IRLS) ---")
        run_improved.main()

        print(f"\n--- [3/4] figures ---")
        plot_figures.main()

        print(f"\n--- [4/4] 3D viewer ---")
        build_viewer.main()
    else:
        print("\n(--skip-improved: figures/viewer skipped -- both need the improved result too)")

    print(f"\n{'='*70}\nDONE ({cfg.BENCHMARK}) -- {time.time()-t0:.0f}s total\n{'='*70}")
    print(f"results: {cfg.RESULTS_DIR}/{cfg.BENCHMARK}_*.npz")
    if not args.skip_improved:
        print(f"figures: {cfg.FIGURES_DIR}/{cfg.BENCHMARK}_*.png")
        viewer_name = "model3d.html" if cfg.BENCHMARK == "synthetic400" else f"model3d_{cfg.BENCHMARK}.html"
        print(f"3D viewer: {cfg.VIEWER_DIR}/{viewer_name}")

    if args.upload_s3_bucket:
        print(f"\n--- uploading to s3://{args.upload_s3_bucket}/{args.upload_s3_prefix}/ ---")
        from scripts import upload_to_s3
        sys.argv = ["upload_to_s3", "--bucket", args.upload_s3_bucket,
                   "--prefix", args.upload_s3_prefix, "--name", cfg.BENCHMARK]
        upload_to_s3.main()


if __name__ == "__main__":
    main()
