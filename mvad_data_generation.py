#!/usr/bin/env python3
"""
multivoice_VAD_data_generation.py
=================================
Generate train / val / test datasets for multi-voice activity detection.

Each output file is a fixed-length audio clip (default **15 s**, 48 kHz mono)
paired with a **sample-level** ground-truth NumPy array:

    0 = silence  (no active speaker)
    1 = single speaker
    2 = overlapping speech  (≥ 2 speakers)

Key design decisions
────────────────────
• Target proportions are **global** (across the whole split), not per file.
  Some files are silence-only, some are single-speaker-only, and the rest
  are mixed — the proportions are balanced by dynamic budget tracking.
• Long continuous fragments (≥ 5 s by default) are drawn from source files,
  preserving natural speech rhythm with pauses.  Speech is never cut into
  tiny, choppy segments.
• Source single-speaker WAVs + .mat VAD are loaded once, then fragments are
  drawn at random and layered to create overlap.

Default proportions (configurable):
    silence 20 %  ·  single 50 %  ·  two-speaker 20 %  ·  three-speaker 10 %

Usage
─────
    python3 multivoice_VAD_data_generation.py                          # defaults
    python3 multivoice_VAD_data_generation.py --n-train 500 --seed 123
    python3 multivoice_VAD_data_generation.py --duration 20 --min-fragment-s 6

Output
──────
    multivoice_VAD_data_generation/
        config.json
        train/   train_0000.wav   train_0000_gt.npy  …
        val/     val_0000.wav     val_0000_gt.npy    …
        test/    test_0000.wav    test_0000_gt.npy   …
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.io import loadmat


# ===========================================================================
#  Source-pool loading
# ===========================================================================

def _find_speech_segments(vad, min_samples):
    """Return [(start, end), …] of contiguous VAD=1 regions ≥ *min_samples*."""
    segs, in_sp, start = [], False, 0
    for i in range(len(vad)):
        if vad[i] == 1 and not in_sp:
            start, in_sp = i, True
        elif vad[i] == 0 and in_sp:
            if i - start >= min_samples:
                segs.append((start, i))
            in_sp = False
    if in_sp and len(vad) - start >= min_samples:
        segs.append((start, len(vad)))
    return segs


def load_source_pool(input_dir, sr_target=48000, min_fragment_s=5.0):
    """
    Load every WAV + MAT pair into memory.

    Only source files ≥ *min_fragment_s* are kept (shorter ones cannot
    provide a usable fragment).  Returns a list of source dicts.
    """
    input_dir = Path(input_dir)
    min_n = int(min_fragment_s * sr_target)
    pool, skipped = [], 0
    for wp in sorted(input_dir.glob('*.wav')):
        mp = wp.with_suffix('.mat')
        if not mp.exists():
            continue
        audio, sr = sf.read(str(wp), dtype='float32')
        if sr != sr_target:
            continue
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        vad = loadmat(str(mp))['var'].flatten().astype(np.uint8)
        n = min(len(audio), len(vad))
        audio, vad = audio[:n], vad[:n]
        if n < min_n:
            skipped += 1
            continue
        segs = _find_speech_segments(vad, min_samples=int(0.05 * sr_target))
        if not segs:
            skipped += 1
            continue
        sp = np.concatenate([audio[s:e] for s, e in segs])
        rms = max(float(np.sqrt(np.mean(sp ** 2))), 1e-7)
        pool.append(dict(name=wp.stem, audio=audio, vad=vad,
                         speech_segments=segs, rms=rms, length=n))
    total_s = sum(p['length'] / sr_target for p in pool)
    print(f"  Loaded {len(pool)} source files ≥ {min_fragment_s}s  "
          f"({total_s:.1f} s total, {skipped} skipped)")
    return pool


# ===========================================================================
#  Fragment picking
# ===========================================================================

def _pick_fragment(pool, max_len, rng, min_frag_n, exclude_src=None):
    """
    Select a random source and extract a continuous sub-portion of
    [min_frag_n, max_len] samples.  The fragment contains both the original
    audio AND its sample-level VAD (speech + natural silence).
    """
    ok = [s for s in pool
          if s['length'] >= min_frag_n
          and (not exclude_src or s['source_idx'] not in exclude_src)]
    if not ok:
        ok = [s for s in pool if s['length'] >= min_frag_n]
    if not ok:
        return None

    src = ok[int(rng.integers(len(ok)))]

    # Determine fragment length: [min_frag_n .. min(source_length, max_len)]
    actual_max = min(src['length'], max_len)
    if actual_max < min_frag_n:
        return None
    frag_len = int(rng.integers(min_frag_n, actual_max + 1))

    # Random start position within the source
    space = src['length'] - frag_len
    start = int(rng.integers(0, space + 1)) if space > 0 else 0

    return dict(
        audio=src['audio'][start:start + frag_len].copy(),
        vad=src['vad'][start:start + frag_len].copy(),
        source_idx=src['source_idx'],
        rms=src['rms'],
        length=frag_len,
    )


def _pick_fragment_near_len(pool, target_len, max_len, rng, min_frag_n,
                            exclude_src=None):
    """
    Pick a fragment whose length is close to *target_len* (±40 %).
    Used in overlap segments so speakers have similar durations.
    """
    lo = max(min_frag_n, int(target_len * 0.6))
    hi = min(max_len, int(target_len * 1.4))

    # Find sources that can provide a fragment in [lo, hi]
    ok = [s for s in pool
          if s['length'] >= lo
          and (not exclude_src or s['source_idx'] not in exclude_src)]
    if not ok:
        ok = [s for s in pool if s['length'] >= lo]
    if not ok:
        return _pick_fragment(pool, max_len, rng, min_frag_n, exclude_src)

    src = ok[int(rng.integers(len(ok)))]

    actual_max = min(src['length'], hi)
    actual_min = lo
    if actual_max < actual_min:
        actual_max = min(src['length'], max_len)
    if actual_max < actual_min:
        return None
    frag_len = int(rng.integers(actual_min, actual_max + 1))

    space = src['length'] - frag_len
    start = int(rng.integers(0, space + 1)) if space > 0 else 0

    return dict(
        audio=src['audio'][start:start + frag_len].copy(),
        vad=src['vad'][start:start + frag_len].copy(),
        source_idx=src['source_idx'],
        rms=src['rms'],
        length=frag_len,
    )


# ===========================================================================
#  Helpers
# ===========================================================================

def _gain(audio, src_rms, tgt_rms, rng, db_range=4.0):
    """RMS-normalise + random ±db_range gain jitter."""
    s = (tgt_rms / src_rms) * 10.0 ** (rng.uniform(-db_range, db_range) / 20.0)
    return (audio * s).astype(np.float32)


def _normalise(mix):
    """Peak-normalise a float64 buffer to ±0.9."""
    pk = np.max(np.abs(mix))
    if pk > 1e-4:
        mix = mix / pk * 0.9
    return mix.astype(np.float32)


def _place(mix, cnt, pos, frag, trms, rng):
    """Place a fragment at *pos* in the output buffers.  Returns length."""
    blen = frag['length']
    a = _gain(frag['audio'], frag['rms'], trms, rng)
    mix[pos:pos + blen] += a.astype(np.float64)
    cnt[pos:pos + blen] += frag['vad'][:blen].astype(np.int32)
    return blen


# ===========================================================================
#  Per-file generators
# ===========================================================================

def gen_silence(n_samples):
    """File of pure silence — GT = 0 everywhere."""
    return np.zeros(n_samples, np.float32), np.zeros(n_samples, np.int32)


def gen_single(pool, n_samples, sr, rng, trms, min_frag_n,
               gap_range_s=(0.05, 0.30)):
    """
    Single-speaker file: long continuous fragments placed sequentially
    with short natural gaps.  Only one speaker at a time → GT ∈ {0, 1}.
    """
    mix = np.zeros(n_samples, np.float64)
    cnt = np.zeros(n_samples, np.int32)

    pos = int(rng.uniform(0.0, 0.2) * sr)

    while pos < n_samples:
        rem = n_samples - pos
        # Allow shorter tail fragment to fill the file
        actual_min = min(min_frag_n, rem)
        if actual_min < int(1.0 * sr):
            break
        frag = _pick_fragment(pool, rem, rng, min_frag_n=actual_min)
        if frag is None:
            break
        blen = _place(mix, cnt, pos, frag, trms, rng)
        pos += blen + int(rng.uniform(*gap_range_s) * sr)

    gt = np.minimum(cnt, 2).astype(np.int32)
    return _normalise(mix), gt


def gen_mixed(pool, n_samples, sr, rng, trms, min_frag_n, seg_weights,
              gap_range_s=(0.03, 0.20)):
    """
    Mixed file: silence + single + overlap segments built from long
    continuous source fragments.  *seg_weights* is adjusted externally
    by the budget tracker.
    """
    mix = np.zeros(n_samples, np.float64)
    cnt = np.zeros(n_samples, np.int32)

    types = list(seg_weights.keys())
    w = np.array([seg_weights[t] for t in types], np.float64)
    w = w / w.sum() if w.sum() > 0 else np.ones(len(w)) / len(w)

    pos = 0

    while pos < n_samples:
        rem = n_samples - pos
        actual_min = min(min_frag_n, rem)
        if actual_min < int(1.0 * sr):
            break

        seg_type = rng.choice(types, p=w)

        # ── silence ──────────────────────────────────────────────────────
        if seg_type == 'silence':
            gap = min(int(rng.uniform(0.3, 1.5) * sr), rem)
            pos += gap
            continue

        # ── speech (1, 2, or 3 speakers) ─────────────────────────────────
        n_spk = {'single': 1, 'two': 2, 'three': 3}[seg_type]
        used_src = set()
        placed = []

        for j in range(n_spk):
            if j == 0:
                frag = _pick_fragment(pool, rem, rng,
                                      min_frag_n=actual_min,
                                      exclude_src=used_src)
            else:
                frag = _pick_fragment_near_len(
                    pool, placed[0]['length'], rem, rng,
                    min_frag_n=actual_min, exclude_src=used_src)
            if frag is None:
                continue
            used_src.add(frag['source_idx'])
            placed.append(frag)

        if not placed:
            pos += int(0.5 * sr)
            continue

        max_adv = 0
        for frag in placed:
            blen = _place(mix, cnt, pos, frag, trms, rng)
            max_adv = max(max_adv, blen)

        pos += max_adv
        pos += int(rng.uniform(*gap_range_s) * sr)

    gt = np.minimum(cnt, 2).astype(np.int32)
    return _normalise(mix), gt


# ===========================================================================
#  Split generation with global budget tracking
# ===========================================================================

def generate_split(pool, out_dir, n_files, sr, duration, min_frag_n,
                   target_props, rng, prefix='sample', trms=0.10):
    """
    Generate *n_files* samples whose **aggregate** GT proportions approach
    *target_props*.  Uses dynamic budget tracking.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_samples = int(sr * duration)
    total_dur = n_files * duration

    # ── global budget (seconds of GT labels) ──────────────────────────────
    ovl_pct = target_props['two'] + target_props['three']
    budget = {
        0: target_props['silence'] * total_dur,
        1: target_props['single']  * total_dur,
        2: ovl_pct                 * total_dur,
    }
    two_ratio = (target_props['two'] / ovl_pct) if ovl_pct > 0 else 0.75

    # ── file-type allocation ──────────────────────────────────────────────
    n_sil  = max(1, round(0.02 * n_files))
    n_sing = max(2, round(0.15 * n_files))
    n_mix  = n_files - n_sil - n_sing

    # Pre-estimate GT from special files (for initial mixed-file weights)
    est_sil_from_silence   = n_sil  * duration
    est_sil_from_single    = n_sing * duration * 0.25
    est_single_from_single = n_sing * duration * 0.75

    mixed_budget = {
        0: max(0, budget[0] - est_sil_from_silence - est_sil_from_single),
        1: max(0, budget[1] - est_single_from_single),
        2: budget[2],
    }

    ftypes = (['silence'] * n_sil
              + ['single'] * n_sing
              + ['mixed']  * n_mix)
    order = rng.permutation(len(ftypes))
    ftypes = [ftypes[j] for j in order]

    # ── accumulators ──────────────────────────────────────────────────────
    actual = {0: 0.0, 1: 0.0, 2: 0.0}
    per_file = {'silence': [], 'single': [], 'overlap': []}

    for i, ft in enumerate(ftypes):
        if ft == 'silence':
            audio, gt = gen_silence(n_samples)

        elif ft == 'single':
            audio, gt = gen_single(pool, n_samples, sr, rng, trms,
                                   min_frag_n)

        else:
            mixed_actual = {
                k: actual[k] - (est_sil_from_silence + est_sil_from_single
                                if k == 0 else
                                est_single_from_single if k == 1 else 0)
                for k in budget
            }
            rem = {k: max(0.01, mixed_budget[k] - max(0, mixed_actual[k]))
                   for k in mixed_budget}
            # Overlap segments produce ~50–55 % GT=2 (rest leaks to GT=1/0
            # from natural pauses).  Mild boost to compensate.
            OVL_BOOST = 1.5
            seg_w = {
                'silence': rem[0],
                'single':  rem[1],
                'two':     rem[2] * two_ratio     * OVL_BOOST,
                'three':   rem[2] * (1 - two_ratio) * OVL_BOOST,
            }
            audio, gt = gen_mixed(pool, n_samples, sr, rng, trms,
                                  min_frag_n, seg_w)

        # ── save ──────────────────────────────────────────────────────────
        sf.write(str(out_dir / f'{prefix}_{i:04d}.wav'),
                 audio, sr, subtype='FLOAT')
        np.save(str(out_dir / f'{prefix}_{i:04d}_gt.npy'), gt)

        # ── update ────────────────────────────────────────────────────────
        n_gt = len(gt)
        for lab in (0, 1, 2):
            actual[lab] += np.sum(gt == lab) / sr

        per_file['silence'].append(np.sum(gt == 0) / n_gt * 100)
        per_file['single'].append(np.sum(gt == 1) / n_gt * 100)
        per_file['overlap'].append(np.sum(gt == 2) / n_gt * 100)

        if (i + 1) % 50 == 0 or i + 1 == n_files:
            print(f"    [{i+1:>4d}/{n_files}]")

    # ── report ────────────────────────────────────────────────────────────
    g_tot = sum(actual.values())
    print(f"    ──── Global proportions ────")
    for lab, name in [(0, 'Silence'), (1, 'Single'), (2, 'Overlap')]:
        print(f"    {name:10s}: {actual[lab]:7.1f} s  "
              f"= {actual[lab]/g_tot*100:5.1f} %  "
              f"(target {budget[lab]/total_dur*100:.0f} %)")
    print(f"    ──── Per-file averages ────")
    for k in ('silence', 'single', 'overlap'):
        v = per_file[k]
        print(f"    {k:10s}: {np.mean(v):5.1f} % ± {np.std(v):.1f}")

    from collections import Counter
    fc = Counter(ftypes)
    print(f"    ──── File types ────")
    for t in ('silence', 'single', 'mixed'):
        print(f"    {t:10s}: {fc.get(t, 0)}")

    return {
        'global': {name: actual[lab] / g_tot * 100
                   for lab, name in [(0, 'silence'), (1, 'single'), (2, 'overlap')]},
    }


# ===========================================================================
#  CLI
# ===========================================================================

def main():
    ap = argparse.ArgumentParser(
        description='Generate multi-voice VAD datasets (train / val / test)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Proportions are GLOBAL across each split, not per-file.
Some files will be silence-only or single-speaker-only;
mixed files compensate via dynamic budget tracking.

Examples:
  python3 multivoice_VAD_data_generation.py
  python3 multivoice_VAD_data_generation.py --n-train 500 --duration 20
  python3 multivoice_VAD_data_generation.py --min-fragment-s 6 --pct-single 50
""")

    ap.add_argument('--input-dir', default=(
        '/work/user_data/msulewsk/data/repos/leapFox/trunk_main188/'
        'test_suite/subjective_tests/TR/input'),
        help='Directory with source .wav + .mat files')
    ap.add_argument('--output-dir', default='multivoice_VAD_data_generation',
                    help='Output root directory')
    ap.add_argument('--sr', type=int, default=48000)
    ap.add_argument('--duration', type=float, default=15.0,
                    help='Duration of each output file in seconds (default: 15)')
    ap.add_argument('--min-fragment-s', type=float, default=5.0,
                    help='Minimum fragment duration in seconds (default: 5)')

    g = ap.add_argument_group('Dataset sizes')
    g.add_argument('--n-train', type=int, default=300)
    g.add_argument('--n-val',   type=int, default=50)
    g.add_argument('--n-test',  type=int, default=100)

    g = ap.add_argument_group('Global target proportions (%%)')
    g.add_argument('--pct-silence', type=float, default=20)
    g.add_argument('--pct-single',  type=float, default=50)
    g.add_argument('--pct-two',     type=float, default=20)
    g.add_argument('--pct-three',   type=float, default=10)

    g = ap.add_argument_group('Mixing')
    g.add_argument('--target-rms', type=float, default=0.10,
                   help='Nominal speech RMS after normalisation')
    g.add_argument('--seed', type=int, default=42,
                   help='Random seed for reproducibility')

    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    min_frag_n = int(args.min_fragment_s * args.sr)

    total_pct = (args.pct_silence + args.pct_single
                 + args.pct_two + args.pct_three)
    props = {
        'silence': args.pct_silence / total_pct,
        'single':  args.pct_single  / total_pct,
        'two':     args.pct_two     / total_pct,
        'three':   args.pct_three   / total_pct,
    }

    print(f"\n{'=' * 62}")
    print(f"  Multi-Voice VAD — Data Generation")
    print(f"{'=' * 62}")
    print(f"  Source     : {args.input_dir}")
    print(f"  Output     : {args.output_dir}")
    print(f"  Files      : train={args.n_train}  val={args.n_val}  test={args.n_test}")
    print(f"  Duration   : {args.duration} s per file  @ {args.sr} Hz")
    print(f"  Min frag.  : {args.min_fragment_s} s")
    print(f"  Proportions (global target):")
    for k, v in props.items():
        print(f"      {k:10s}: {v*100:5.1f} %")
    print(f"  Seed       : {args.seed}")
    print(f"{'=' * 62}\n")

    # ── load sources ──────────────────────────────────────────────────────
    print("  Loading source pool …")
    pool = load_source_pool(args.input_dir, sr_target=args.sr,
                            min_fragment_s=args.min_fragment_s)
    if len(pool) < 3:
        print(f"  ✗ Need ≥ 3 source files, found {len(pool)}")
        return 1

    # Annotate pool entries with their index for exclusion logic
    for idx, p in enumerate(pool):
        p['source_idx'] = idx

    base = Path(args.output_dir)
    base.mkdir(parents=True, exist_ok=True)

    # ── generate splits ───────────────────────────────────────────────────
    all_stats = {}
    for split, n in [('train', args.n_train),
                     ('val',   args.n_val),
                     ('test',  args.n_test)]:
        print(f"\n  ── {split.upper()} ({n} files) "
              f"{'─' * (45 - len(split) - len(str(n)))}")
        stats = generate_split(
            pool, base / split, n, args.sr, args.duration, min_frag_n,
            props, rng, prefix=split, trms=args.target_rms)
        all_stats[split] = stats['global']

    # ── save config ───────────────────────────────────────────────────────
    cfg = dict(
        input_dir=str(args.input_dir),
        sr=args.sr, duration=args.duration,
        min_fragment_s=args.min_fragment_s,
        n_train=args.n_train, n_val=args.n_val, n_test=args.n_test,
        target_proportions=props,
        seed=args.seed, target_rms=args.target_rms,
        n_source_files=len(pool),
        actual_global_proportions=all_stats,
    )
    cfg_path = base / 'config.json'
    with open(cfg_path, 'w') as f:
        json.dump(cfg, f, indent=2)
    print(f"\n  Config → {cfg_path}")
    print(f"\n  ✓ Done!  Dataset in: {base}/")
    return 0


if __name__ == '__main__':
    sys.exit(main())
