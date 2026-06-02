#!/usr/bin/env python
"""Pre-build the Henderson quanvolutional lookup tables (LUTs) locally.

QMedViTHenderson2019 replaces every stem ConvBNReLU with a frozen random
quanvolutional layer (Quanv2d). Because each circuit is non-trainable and fed a
binary threshold encoding, its output is a deterministic function of the
2**n_qubits possible patches, so Quanv2d precomputes a lookup table once at
construction. For the full stem that is ~8384 nine-qubit circuits and takes
~5-6 minutes (parallelized) on first build.

This script builds those tables locally and writes them to `.quanv_cache/` at
the repo root. Commit the resulting ~17 MB of `quanv_lut_*.pt` files (the repo's
.gitignore has an exception for that folder) and Kaggle will simply *load* them
via the matching, configuration-derived cache key instead of rebuilding — the
model then initializes in seconds.

IMPORTANT: the cache key is a hash of the circuit configuration, including the
seed. `--seed` MUST equal the `quanv_seed` you use at runtime (the notebook's
default is 0); otherwise the hashes differ and Kaggle rebuilds from scratch.

The model import pulls MedViT/QMedViT (timm, torch, pennylane), so run this in an
environment where those are installed. (medmnist is NOT required — it is only
used by the notebook's data pipeline, not by model construction.)

Usage:
    python scripts/prebuild_quanv_cache.py --seed 0 --n-jobs -1
"""
import argparse
import glob
import os
import sys

# Repo root = parent of the directory containing this script (scripts/..).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Ensure the repo root is importable regardless of CWD (running
# `python scripts/prebuild_quanv_cache.py` puts scripts/ on sys.path, not root).
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _human_size(n_bytes: int) -> str:
    size = float(n_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024.0 or unit == "GB":
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} GB"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pre-build Henderson quanvolutional LUTs into .quanv_cache/.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="quanv_seed; MUST match the runtime quanv_seed used on Kaggle.",
    )
    parser.add_argument(
        "--n-jobs", type=int, default=-1,
        help="Parallel workers for the build (-1 = all CPU cores, 1 = sequential).",
    )
    parser.add_argument(
        "--num-classes", type=int, default=2,
        help="Classifier head size; irrelevant to the LUTs, exposed for completeness.",
    )
    parser.add_argument(
        "--cache-dir", type=str, default=os.path.join(_REPO_ROOT, ".quanv_cache"),
        help="Where to write quanv_lut_*.pt (default matches the runtime default).",
    )
    parser.add_argument(
        "--rebuild", action="store_true",
        help="Force a rebuild, ignoring any existing cache files.",
    )
    args = parser.parse_args()

    cache_dir = os.path.abspath(args.cache_dir)
    os.makedirs(cache_dir, exist_ok=True)

    # Import here so --help works even without the heavy deps installed.
    try:
        from quantum_variants.quanv_2d_henderson_2019 import QMedViTHenderson2019
    except Exception as exc:  # noqa: BLE001 - surface a clear, actionable message
        print(
            f"ERROR: could not import QMedViTHenderson2019 ({exc!r}).\n"
            "This script needs MedViT's deps (timm, torch, pennylane) installed,\n"
            "and must be run from / importable at the repo root.",
            file=sys.stderr,
        )
        return 1

    print(
        f"Building quanv LUTs: seed={args.seed}, n_jobs={args.n_jobs}, "
        f"cache_dir={cache_dir}, rebuild={args.rebuild}"
    )

    # Constructing the model triggers Quanv2d._load_or_build_lut for each of the
    # four stem ConvBNReLU blocks (seeds = seed + 1000*idx), writing one
    # quanv_lut_<hash>.pt per block. Built via the real code path so the hashes
    # are guaranteed identical to what the model loads at runtime.
    QMedViTHenderson2019(
        stem_chs=[64, 32, 64],
        depths=[3, 4, 10, 3],
        path_dropout=0.1,
        num_classes=args.num_classes,
        quanv_seed=args.seed,
        quanv_n_jobs=args.n_jobs,
        quanv_lut_cache_dir=cache_dir,
        quanv_rebuild_lut=args.rebuild,
    )

    files = sorted(glob.glob(os.path.join(cache_dir, "quanv_lut_*.pt")))
    total = 0
    print(f"\nCache files in {cache_dir}:")
    for path in files:
        size = os.path.getsize(path)
        total += size
        print(f"  {os.path.basename(path):40s}  {_human_size(size):>10s}")
    print(f"  {'TOTAL':40s}  {_human_size(total):>10s}  ({len(files)} files)")

    if len(files) != 4:
        print(
            f"\nWARNING: expected 4 LUT files for the default stem, found {len(files)}. "
            "Other configs/seeds may have left extra files in this directory.",
            file=sys.stderr,
        )

    print(
        "\nNext: commit the tables so Kaggle loads them instead of rebuilding:\n"
        f"    git add -f .quanv_cache/ && git commit -m \"prebuilt quanv LUTs (seed={args.seed})\""
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
