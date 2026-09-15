"""Optional, opt-in sync of one dataset's outputs (results/<name>_*.npz,
figures/<name>_*.png, the 3D viewer html) to an S3 bucket -- e.g. after
running the pipeline on a bigger AWS instance (this project's dense-
matrix method needs more RAM than this repo's own dev machine has for
anything finer than the two shipped benchmarks) and wanting the outputs
somewhere reachable without copying files by hand.

Nothing here manages AWS credentials -- it shells out to the `aws` CLI,
which must already be configured (`aws configure`, or the standard
AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY/AWS_PROFILE environment
variables) exactly as any other `aws` command on this machine would be.
This script does not read, store, or transmit credentials itself.

Nothing in this pipeline calls this automatically -- run it by hand (or
from scripts/run_pipeline.py's own --upload-s3-bucket flag) once you
have results you actually want uploaded.

    aws configure   # one-time, if not already done
    python -m scripts.upload_to_s3 --bucket my-bucket --prefix tomofastx
    python -m scripts.upload_to_s3 --bucket my-bucket --prefix tomofastx --dry-run
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_HERE))
from config import config as cfg   # noqa: E402


def _sync_one(local_dir, name_prefix, bucket, prefix, dry_run, extra_args):
    """One `aws s3 sync` call per local directory, --exclude '*' --include
    '<name_prefix>*' so only this dataset's own files go up (results/ and
    figures/ hold every benchmark's outputs side by side, not just this
    one's) -- not a full-bucket mirror."""
    if not local_dir.is_dir():
        print(f"  (skip) {local_dir} does not exist")
        return
    dest = f"s3://{bucket}/{prefix}/{local_dir.name}/"
    cmd = ["aws", "s3", "sync", str(local_dir) + "/", dest,
          "--exclude", "*", "--include", f"{name_prefix}*"]
    if dry_run:
        cmd.append("--dryrun")
    cmd += extra_args
    print(f"  $ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bucket", required=True, help="S3 bucket name (no s3:// prefix)")
    ap.add_argument("--prefix", default="tomofastx", help="key prefix inside the bucket (default: tomofastx)")
    ap.add_argument("--name", default=None,
                    help="dataset name to upload (default: whichever config.py currently resolves "
                         "to, i.e. TOMOFASTX_BENCHMARK/TOMOFASTX_DATA_DIR's own name)")
    ap.add_argument("--include-viewer", action="store_true", default=True,
                    help="also sync viewer/ (the 3D html -- can be a few MB; default on)")
    ap.add_argument("--no-include-viewer", dest="include_viewer", action="store_false")
    ap.add_argument("--dry-run", action="store_true", help="show what would be uploaded without uploading")
    ap.add_argument("--aws-arg", action="append", default=[], dest="extra_args",
                    help="extra argument to pass through to `aws s3 sync` verbatim, repeatable "
                         "(e.g. --aws-arg=--profile --aws-arg=myprofile)")
    args = ap.parse_args()

    if shutil.which("aws") is None:
        print("error: the `aws` CLI is not on PATH -- install it and run `aws configure` first "
             "(https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html)", file=sys.stderr)
        sys.exit(1)

    name = args.name or cfg.BENCHMARK
    print(f"uploading {name!r}'s outputs to s3://{args.bucket}/{args.prefix}/" +
         (" (dry run)" if args.dry_run else ""))

    _sync_one(cfg.RESULTS_DIR, name, args.bucket, args.prefix, args.dry_run, args.extra_args)
    _sync_one(cfg.FIGURES_DIR, name, args.bucket, args.prefix, args.dry_run, args.extra_args)
    if args.include_viewer:
        # viewer/ files are named model3d.html (synthetic400 only) /
        # model3d_<name>.html -- "model3d" itself is the common prefix,
        # narrowed to this dataset's own file(s) by name below.
        viewer_prefix = "model3d.html" if name == "synthetic400" else f"model3d_{name}"
        _sync_one(cfg.VIEWER_DIR, viewer_prefix, args.bucket, args.prefix, args.dry_run, args.extra_args)

    print("done." if not args.dry_run else "dry run complete -- rerun without --dry-run to actually upload.")


if __name__ == "__main__":
    main()
