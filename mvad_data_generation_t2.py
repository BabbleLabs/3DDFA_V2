#!/usr/bin/env python3
"""
mvad_data_generation_t2.py
==========================
Read .mat files from epoch-server subfolders, classify them by speaker
overlap, and write train / val / test splits as .wav + ground-truth .npy.

Source data
───────────
    /work/scratch/raflagan/epoch_server/
        classic_se_48k_v8d6d5_spkcount_v1/SE/epochs/
            epoch_YYYYMMDD_HHMMSS_ID/
                000/  001/  002/  003/
                    NNN.mat   (each contains `audio` struct)

Each .mat file stores a structured array ``audio`` with fields:
    • mixtures    – int16 mono audio (480 000 samples = 10 s @ 48 kHz)
    • extraData   – uint8 per-sample speaker count  (0 = silence, 1 = single, 2 = overlap)
    • mixturesPeak – float scalar (original peak before int16 quantisation)

Split composition
─────────────────
In every split the proportion of files is:
    15 % → "single-or-silence"  (max(extraData) ≤ 1, no overlap at all)
    85 % → "overlap"            (some samples have extraData == 2)

Output
──────
    multivoice_VAD_data_generation/
        config.json
        train/   train_0000.wav   train_0000_gt.npy  …
        val/     val_0000.wav     val_0000_gt.npy    …
        test/    test_0000.wav    test_0000_gt.npy   …

Usage
─────
    python3 mvad_data_generation_t2.py                       # defaults (single epoch)
    python3 mvad_data_generation_t2.py --epoch epoch_20260324_104215_W8P3NW
    python3 mvad_data_generation_t2.py --dry-run              # scan only, do not write
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from multiprocessing import Pool, cpu_count

import numpy as np
import soundfile as sf
from scipy.io import loadmat


# ============================================================================
#  Constants
# ============================================================================

EPOCHS_ROOT = (
    "/work/scratch/raflagan/epoch_server/"
    "classic_se_48k_v8d6d5_spkcount_v1/SE/epochs"
)
SR = 48_000
SINGLE_RATIO = 0.15   # fraction of files with max-1-speaker
OVERLAP_RATIO = 0.85   # fraction of files with overlap (some 2s)

# Default split sizes  (train / val / test)
DEFAULT_N_TRAIN = 300
DEFAULT_N_VAL   = 50
DEFAULT_N_TEST  = 100


# ============================================================================
#  Scanning – classify every .mat file (parallel-friendly)
# ============================================================================

def _classify_one(mat_path: str) -> tuple:
    """
    Return (mat_path, has_overlap: bool).
    has_overlap is True when extraData contains at least one 2.
    """
    try:
        mat = loadmat(mat_path, variable_names=["audio"])
        extra = mat["audio"][0, 0]["extraData"].flatten()
        has_overlap = bool(np.any(extra == 2))
        return (mat_path, has_overlap)
    except Exception as exc:
        print(f"  ⚠ skipping {mat_path}: {exc}", file=sys.stderr)
        return (mat_path, None)


def scan_mat_files(epochs_root: str, epoch_name: str = None,
                   n_workers: int = 8) -> tuple:
    """
    Find all .mat sample files and classify them.

    Parameters
    ----------
    epochs_root : str   – root dir containing epoch_* folders
    epoch_name  : str   – if given, scan only this single epoch subfolder
    n_workers   : int   – parallel workers for classification

    Returns
    -------
    overlap_files : list[str]   – paths where extraData has 2s
    single_files  : list[str]   – paths where max(extraData) ≤ 1
    """
    root = Path(epochs_root)
    if epoch_name:
        # Single epoch mode
        pattern = f"{epoch_name}/[0-9][0-9][0-9]/*.mat"
        all_mats = sorted(str(p) for p in root.glob(pattern))
        print(f"  Found {len(all_mats):,} .mat files in epoch {epoch_name}")
    else:
        # All epochs
        all_mats = sorted(str(p) for p in root.glob("epoch_*/[0-9][0-9][0-9]/*.mat"))
        print(f"  Found {len(all_mats):,} .mat files under {epochs_root}")

    if not all_mats:
        return [], []

    # Parallel classification
    workers = min(n_workers, cpu_count(), len(all_mats))
    print(f"  Classifying with {workers} workers …")
    t0 = time.time()

    with Pool(workers) as pool:
        results = pool.map(_classify_one, all_mats, chunksize=256)

    overlap_files, single_files = [], []
    skipped = 0
    for path, has_ovl in results:
        if has_ovl is None:
            skipped += 1
        elif has_ovl:
            overlap_files.append(path)
        else:
            single_files.append(path)

    elapsed = time.time() - t0
    print(f"  Classification done in {elapsed:.1f} s")
    print(f"    overlap (has 2s) : {len(overlap_files):>7,}")
    print(f"    single (max ≤ 1) : {len(single_files):>7,}")
    if skipped:
        print(f"    skipped (errors) : {skipped:>7,}")

    return overlap_files, single_files


# ============================================================================
#  Split allocation
# ============================================================================

def allocate_splits(
    overlap_files: list,
    single_files: list,
    n_train: int,
    n_val: int,
    n_test: int,
    rng: np.random.Generator,
) -> dict:
    """
    Allocate files to train / val / test with the desired
    15 % single / 85 % overlap ratio per split.

    Returns dict  split_name → list[str]  (file paths).
    """
    splits = {}
    # Shuffle both pools
    overlap_arr = np.array(overlap_files)
    single_arr  = np.array(single_files)
    rng.shuffle(overlap_arr)
    rng.shuffle(single_arr)

    ovl_offset = 0
    sng_offset = 0

    for name, n_total in [("train", n_train), ("val", n_val), ("test", n_test)]:
        n_single  = round(n_total * SINGLE_RATIO)
        n_overlap = n_total - n_single

        # Check availability
        avail_ovl = len(overlap_arr) - ovl_offset
        avail_sng = len(single_arr) - sng_offset

        if n_overlap > avail_ovl:
            print(f"  ⚠ {name}: requested {n_overlap} overlap files but only "
                  f"{avail_ovl} remain – reducing.")
            n_overlap = avail_ovl
            n_total = n_single + n_overlap

        if n_single > avail_sng:
            print(f"  ⚠ {name}: requested {n_single} single files but only "
                  f"{avail_sng} remain – reducing.")
            n_single = avail_sng
            n_total = n_single + n_overlap

        sel_ovl = list(overlap_arr[ovl_offset:ovl_offset + n_overlap])
        sel_sng = list(single_arr[sng_offset:sng_offset + n_single])
        ovl_offset += n_overlap
        sng_offset += n_single

        combined = sel_ovl + sel_sng
        rng.shuffle(combined)      # interleave
        splits[name] = combined

        print(f"  {name:>5s}: {len(combined):>6,} files  "
              f"(overlap={len(sel_ovl)}, single={len(sel_sng)}, "
              f"ratio {len(sel_sng)/max(1,len(combined))*100:.1f}%/"
              f"{len(sel_ovl)/max(1,len(combined))*100:.1f}%)")

    return splits


# ============================================================================
#  Writing split files
# ============================================================================

def write_split(split_name: str, file_list: list, output_root: Path):
    """
    For each .mat path in *file_list*:
      • load ``targets`` → save as WAV (48 kHz, int16)
      • load ``extraData`` → save as .npy (uint8)
    """
    out_dir = output_root / split_name
    out_dir.mkdir(parents=True, exist_ok=True)

    n = len(file_list)
    # Determine zero-padding width from total count
    width = max(4, len(str(n - 1)))

    t0 = time.time()
    for idx, mat_path in enumerate(file_list):
        mat = loadmat(mat_path, variable_names=["audio"])
        audio_struct = mat["audio"][0, 0]

        targets = audio_struct["targets"].flatten().astype(np.int16)
        extra   = audio_struct["extraData"].flatten().astype(np.uint8)

        tag = f"{split_name}_{idx:0{width}d}"
        wav_path = str(out_dir / f"{tag}.wav")
        gt_path  = str(out_dir / f"{tag}_gt.npy")

        sf.write(wav_path, targets, SR, subtype="PCM_16")
        np.save(gt_path, extra)

        if (idx + 1) % max(1, n // 10) == 0 or idx + 1 == n:
            elapsed = time.time() - t0
            rate = (idx + 1) / elapsed if elapsed > 0 else 0
            eta = (n - idx - 1) / rate if rate > 0 else 0
            print(f"    [{idx+1:>{len(str(n))}}/{n}]  "
                  f"{rate:.0f} files/s  ETA {eta:.0f} s")

    print(f"  ✓ {split_name}: {n} files → {out_dir}")


# ============================================================================
#  CLI
# ============================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Generate multi-voice VAD datasets from epoch .mat files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Each split contains 15%% files with max one speaker (no overlap)
and 85%% files with at least some overlapping speech.

Examples:
  python3 mvad_data_generation_t2.py
  python3 mvad_data_generation_t2.py --epoch epoch_20260324_104215_W8P3NW
  python3 mvad_data_generation_t2.py --dry-run
""")

    ap.add_argument(
        "--epochs-root", default=EPOCHS_ROOT,
        help="Root directory containing epoch_* subfolders "
             f"(default: {EPOCHS_ROOT})")
    ap.add_argument(
        "--epoch", default=None,
        help="Name of a single epoch subfolder to use, e.g. "
             "'epoch_20260324_104215_W8P3NW'. "
             "If omitted, all epoch_* subfolders are scanned.")
    ap.add_argument(
        "--output-dir", default="multivoice_VAD_data_generation",
        help="Output root directory (default: multivoice_VAD_data_generation)")

    g = ap.add_argument_group("Dataset sizes")
    g.add_argument("--n-train", type=int, default=DEFAULT_N_TRAIN,
                   help=f"Number of training files (default: {DEFAULT_N_TRAIN})")
    g.add_argument("--n-val",   type=int, default=DEFAULT_N_VAL,
                   help=f"Number of validation files (default: {DEFAULT_N_VAL})")
    g.add_argument("--n-test",  type=int, default=DEFAULT_N_TEST,
                   help=f"Number of test files (default: {DEFAULT_N_TEST})")

    g = ap.add_argument_group("Misc")
    g.add_argument("--seed", type=int, default=42,
                   help="Random seed (default: 42)")
    g.add_argument("--workers", type=int, default=8,
                   help="Parallel workers for scanning (default: 8)")
    g.add_argument("--dry-run", action="store_true",
                   help="Scan and allocate only – do not write output files")

    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    # Resolve output dir relative to this script's location
    script_dir = Path(__file__).resolve().parent
    output_root = script_dir / args.output_dir

    epoch_label = args.epoch if args.epoch else "(all epochs)"

    print(f"\n{'=' * 64}")
    print(f"  Multi-Voice VAD — Data Generation (T2: from epoch .mat files)")
    print(f"{'=' * 64}")
    print(f"  Epochs root : {args.epochs_root}")
    print(f"  Epoch       : {epoch_label}")
    print(f"  Output      : {output_root}")
    print(f"  Split sizes : train={args.n_train}  val={args.n_val}  test={args.n_test}")
    print(f"  Composition : {SINGLE_RATIO*100:.0f}% single/silence  "
          f"{OVERLAP_RATIO*100:.0f}% overlap")
    print(f"  Seed        : {args.seed}")
    print(f"{'=' * 64}\n")

    # ── Step 1: Scan & classify ──────────────────────────────────────────
    print("Step 1 — Scanning .mat files …")
    overlap_files, single_files = scan_mat_files(
        args.epochs_root, epoch_name=args.epoch, n_workers=args.workers)

    total_avail = len(overlap_files) + len(single_files)
    if total_avail == 0:
        print("  ✗ No valid .mat files found. Exiting.")
        return 1

    # ── Step 2: Allocate splits ──────────────────────────────────────────
    print("\nStep 2 — Allocating splits …")
    splits = allocate_splits(
        overlap_files, single_files,
        args.n_train, args.n_val, args.n_test, rng)

    total_selected = sum(len(v) for v in splits.values())
    print(f"\n  Total selected: {total_selected:,} / {total_avail:,} available")

    if args.dry_run:
        print("\n  --dry-run: skipping file writing.")
        return 0

    # ── Step 3: Write output files ───────────────────────────────────────
    output_root.mkdir(parents=True, exist_ok=True)

    for split_name in ("train", "val", "test"):
        flist = splits.get(split_name, [])
        if not flist:
            continue
        print(f"\nStep 3 — Writing {split_name} ({len(flist):,} files) …")
        write_split(split_name, flist, output_root)

    # ── Step 4: Save config ──────────────────────────────────────────────
    cfg = {
        "epochs_root": str(args.epochs_root),
        "epoch": args.epoch,
        "output_dir": str(output_root),
        "sr": SR,
        "seed": args.seed,
        "single_ratio": SINGLE_RATIO,
        "overlap_ratio": OVERLAP_RATIO,
        "total_mat_files_scanned": total_avail,
        "total_overlap_available": len(overlap_files),
        "total_single_available": len(single_files),
        "splits": {
            name: {
                "n_files": len(flist),
                "files": [os.path.basename(f) for f in flist],
            }
            for name, flist in splits.items()
        },
    }
    cfg_path = output_root / "config.json"
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"\n  Config → {cfg_path}")

    print(f"\n  ✓ Done!  Dataset in: {output_root}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
