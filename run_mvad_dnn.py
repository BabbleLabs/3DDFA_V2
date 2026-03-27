#!/usr/bin/env python3
"""
run_mvad_dnn.py — Run MVAD V2 DNN inference on a WAV file with post-filtering.

Pipeline:
    1. Load audio from WAV file
    2. Run MVAD_V2 DNN model inference (per-frame predictions at 100 Hz / 10 ms hop)
    3. Apply sliding-window filter (removes isolated overlap/single-speaker frames)
    4. Apply hold/sustain filter (extends active labels to smooth gaps)
    5. Save results to .npz file

Output .npz contains:
    labels          — final post-filtered labels (int32, 0=silence, 1=single, 2=overlap)
    labels_raw      — raw DNN predictions before any filtering (int32)
    hop_sec         — time between frames in seconds (float64)
    sample_rate     — audio sample rate in Hz (int)

Usage:
    python3 run_mvad_dnn.py -i audio.wav -m mvad_dnn_v2_model_ep46.pt -o dumps/mvad_dnn.npz
    python3 run_mvad_dnn.py -i audio.wav  # uses defaults for model and output
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

# Import model architecture and feature extraction from the training script
from train_mvad_dnn_v2 import (
    MVAD_V2,
    compute_log_mel,
    FRAME_MS,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  Model Loading & Inference
# ═══════════════════════════════════════════════════════════════════════════════

def load_mvad_v2_model(model_path, device=None):
    """Load a trained MVAD_V2 model from checkpoint."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(str(model_path), map_location=device, weights_only=False)
    cfg = ckpt['config']
    model = MVAD_V2(
        n_mels=cfg.get('n_mels', 40),
        n_classes=cfg.get('num_classes', 3),
        hidden_ch=cfg.get('hidden_ch', 64),
        kernel_size=cfg.get('kernel_size', 15),
        n_blocks=cfg.get('n_blocks', 5),
        dropout=cfg.get('dropout', 0.1),
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    # Extract standardisation (mean/std fitted on training set)
    std_info = ckpt.get('standardisation', {})
    feat_mean = std_info.get('mean', None)
    feat_std = std_info.get('std', None)
    print(f'  DNN-V2 model loaded: arch={cfg["arch"]}, '
          f'hidden={cfg["hidden_ch"]}, k={cfg["kernel_size"]}, '
          f'blocks={cfg["n_blocks"]}, params={n_params:,}, device={device}')
    if 'epoch' in ckpt:
        print(f'  Best epoch: {ckpt["epoch"]}')
    return model, cfg, feat_mean, feat_std, device


def predict_file(audio, orig_sr, model, cfg, feat_mean, feat_std, device):
    """
    Run MVAD_V2 inference on a full audio file.

    Computes mel features at the file's native sample rate,
    applies z-score normalisation, and runs the model.
    Returns per-frame predictions at 100 Hz (10 ms hop).
    """
    # Compute log mel features
    mel = compute_log_mel(audio, orig_sr)  # (n_frames, n_mels)

    # Z-score normalisation
    if feat_mean is not None and feat_std is not None:
        mel = (mel - feat_mean) / feat_std

    # (n_frames, n_mels) → (1, n_mels, n_frames) for Conv1d
    mel_t = torch.from_numpy(mel.T.astype(np.float32)).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(mel_t)  # (1, n_classes, n_frames)
        predictions = logits.argmax(dim=1).squeeze(0).cpu().numpy()

    return predictions


# ═══════════════════════════════════════════════════════════════════════════════
#  Post-Filtering
# ═══════════════════════════════════════════════════════════════════════════════

def sliding_window_filter(labels, hop_sec, window_sec=1.0,
                          overlap_threshold=0.70, single_threshold=0.30):
    """
    Sliding-window post-filter (1-second centred window).

    Two checks (both read from the *original* unmodified labels):
      1. Overlap → Single: if <overlap_threshold of frames in the window
         are overlap, re-label as single speaker.
      2. Single → Silence: if <single_threshold of frames in the window
         are single speaker, re-label as silence.
    """
    filtered = labels.copy()
    half_win = int((window_sec / 2) / hop_sec)
    n = len(labels)
    for i in range(n):
        if labels[i] == 2:
            lo = max(0, i - half_win)
            hi = min(n, i + half_win + 1)
            window = labels[lo:hi]
            if np.sum(window == 2) / len(window) < overlap_threshold:
                filtered[i] = 1
        elif labels[i] == 1:
            lo = max(0, i - half_win)
            hi = min(n, i + half_win + 1)
            window = labels[lo:hi]
            if np.sum(window == 1) / len(window) < single_threshold:
                filtered[i] = 0
    return filtered


def hold_filter(labels, hop_sec, hold_ms=750):
    """
    Hold / sustain filter — extends overlap and single-speaker labels
    to the right by *hold_ms* milliseconds.

    Two passes (left → right):
      1. Overlap hold: every frame labelled 2 is held for hold_ms
         to the right, overwriting silence and single-speaker.
      2. Single-speaker hold: every frame labelled 1 (after pass 1)
         is held for hold_ms to the right, but only into silence (0).
         The hold stops when an overlap (2) frame is encountered.

    Overlap has priority — single-speaker never overwrites overlap.
    """
    hold_frames = max(1, int(hold_ms / 1000 / hop_sec))
    filtered = labels.copy()
    n = len(filtered)

    # Pass 1: extend overlap to the right
    counter = 0
    for i in range(n):
        if filtered[i] == 2:
            counter = hold_frames
        elif counter > 0:
            filtered[i] = 2
            counter -= 1

    # Pass 2: extend single-speaker to the right (stop at overlap)
    counter = 0
    for i in range(n):
        if filtered[i] == 2:
            counter = 0          # overlap encountered → stop single hold
        elif filtered[i] == 1:
            counter = hold_frames
        elif counter > 0:        # filtered[i] == 0 (silence)
            filtered[i] = 1
            counter -= 1

    return filtered


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Run MVAD V2 DNN inference with post-filtering')
    parser.add_argument('-i', '--input', required=True,
                        help='Path to input WAV file')
    parser.add_argument('-m', '--model', default='mvad_dnn_v2_model_ep46.pt',
                        help='Path to MVAD V2 model checkpoint '
                             '(default: mvad_dnn_v2_model_ep46.pt)')
    parser.add_argument('-o', '--output', default=None,
                        help='Output .npz path (default: dumps/mvad_dnn.npz)')
    parser.add_argument('--window-sec', type=float, default=1.0,
                        help='Sliding-window filter duration in seconds '
                             '(default: 1.0)')
    parser.add_argument('--overlap-threshold', type=float, default=0.70,
                        help='Min fraction of overlap frames to keep overlap '
                             'label (default: 0.70)')
    parser.add_argument('--single-threshold', type=float, default=0.30,
                        help='Min fraction of single-speaker frames to keep '
                             'single label (default: 0.30)')
    parser.add_argument('--hold-ms', type=float, default=750,
                        help='Hold/sustain duration in ms (default: 750)')
    args = parser.parse_args()

    input_path = Path(args.input)
    model_path = Path(args.model)
    output_path = Path(args.output) if args.output else Path('dumps/mvad_dnn.npz')

    if not input_path.exists():
        print(f'Error: Input WAV not found: {input_path}')
        sys.exit(1)
    if not model_path.exists():
        print(f'Error: Model checkpoint not found: {model_path}')
        sys.exit(1)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── 1. Load audio ─────────────────────────────────────────────────────
    print(f'\n{"="*60}')
    print(f'MVAD V2 DNN Inference Pipeline')
    print(f'{"="*60}')
    signal, sr = sf.read(str(input_path), dtype='float64')
    if signal.ndim > 1:
        signal = np.mean(signal, axis=1)
    duration = len(signal) / sr
    print(f'  Input    : {input_path}')
    print(f'  SR       : {sr} Hz')
    print(f'  Duration : {duration:.2f} s')
    print(f'  Samples  : {len(signal):,}')

    # ── 2. Load model ─────────────────────────────────────────────────────
    print(f'\nLoading MVAD V2 model...')
    model, cfg, feat_mean, feat_std, device = load_mvad_v2_model(model_path)

    # ── 3. Run inference ──────────────────────────────────────────────────
    print(f'\nRunning DNN inference...')
    dnn_labels = predict_file(signal, sr, model, cfg, feat_mean, feat_std, device)
    hop_sec = FRAME_MS / 1000.0  # 0.01 s (10 ms hop)

    n_frames = len(dnn_labels)
    print(f'  Raw DNN: {n_frames} frames ({n_frames * hop_sec:.2f} s)')
    for lab, name in [(0, 'Silence'), (1, 'Single'), (2, 'Overlap')]:
        cnt = np.sum(dnn_labels == lab)
        print(f'    {name:20s}: {cnt:6d} frames  ({cnt/n_frames*100:5.1f}%)'
              f'  {cnt*hop_sec:.2f}s')

    # ── 4. Sliding-window filter ──────────────────────────────────────────
    print(f'\nApplying sliding-window filter '
          f'(window={args.window_sec}s, O≥{args.overlap_threshold:.0%}, '
          f'S≥{args.single_threshold:.0%})...')
    labels_filtered = sliding_window_filter(
        dnn_labels, hop_sec,
        window_sec=args.window_sec,
        overlap_threshold=args.overlap_threshold,
        single_threshold=args.single_threshold,
    )

    n_ovl_before = np.sum(dnn_labels == 2)
    n_ovl_after = np.sum(labels_filtered == 2)
    removed = n_ovl_before - n_ovl_after
    print(f'  Overlap: {n_ovl_before} → {n_ovl_after} '
          f'(removed {removed} frames = {removed * hop_sec:.2f} s)')

    # ── 5. Hold / sustain filter ──────────────────────────────────────────
    print(f'\nApplying hold/sustain filter (hold={args.hold_ms:.0f} ms)...')
    labels_final = hold_filter(labels_filtered, hop_sec, hold_ms=args.hold_ms)

    print(f'  Final labels:')
    for lab, name in [(0, 'Silence'), (1, 'Single'), (2, 'Overlap')]:
        c_before = np.sum(labels_filtered == lab)
        c_after = np.sum(labels_final == lab)
        diff = c_after - c_before
        print(f'    {name:20s}: {c_before:6d} → {c_after:6d}  '
              f'({diff:+6d} frames = {diff * hop_sec:+.2f} s)')

    # ── 6. Save results ──────────────────────────────────────────────────
    np.savez(
        str(output_path),
        labels=labels_final.astype(np.int32),
        labels_raw=dnn_labels.astype(np.int32),
        hop_sec=np.float64(hop_sec),
        sample_rate=np.int64(sr),
    )
    print(f'\n✓ Results saved to: {output_path}')
    print(f'  Keys: labels, labels_raw, hop_sec, sample_rate')
    print(f'  labels shape: {labels_final.shape}, dtype: int32')
    print(f'  (0=silence, 1=single speaker, 2=overlap)')


if __name__ == '__main__':
    main()
