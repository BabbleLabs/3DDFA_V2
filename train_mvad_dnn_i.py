#!/usr/bin/env python3
"""
train_mvad_dnn_i.py — Train VAD_I_C2_L8 architecture for Multi-Voice Activity Detection

Uses the 1D convolutional encoder-decoder architecture described in
vad_i_c2_l8_architecture.html, adapted for 3-class multivoice VAD:
    0 = silence / no speech
    1 = single speaker
    2 = overlapping speech (≥ 2 speakers)

The network operates directly on 16 kHz raw audio (resampled from 48 kHz
source) and produces frame-level decisions at 125 Hz (128× decimation,
one decision every 8 ms).

Architecture (VAD_I_C2_L8):
    Forward Transform (Encoder, 3 layers, no residual):
        L1:  Conv1d(1→2,  k=7, s=2)  + BN + ReLU
        L2:  Conv1d(2→4,  k=7, s=2)  + BN + ReLU
        L3:  Conv1d(4→8,  k=7, s=2)  + BN + ReLU
    Core (7 layers, with residual skip connections):
        L6:  Conv1d(8→8,  k=7, s=2)  + BN + ReLU + skip
        L7:  Conv1d(8→8,  k=9, s=2)  + BN + ReLU + skip
        L8:  Conv1d(8→8,  k=9, s=2)  + BN + ReLU + skip
        L9:  Conv1d(8→8,  k=9, s=2)  + BN + ReLU + skip
        L10: Conv1d(8→8,  k=9, s=1)  + BN + ReLU + skip
        L11: Conv1d(8→8,  k=9, s=1)  + BN + ReLU + skip
        L12: Conv1d(8→8,  k=9, s=1)  + BN + ReLU + skip
    Inverse Transform (Decoder, 2 layers, no residual):
        L15: Conv1d(8→4,  k=9,   s=1) + BN + ReLU
        L16: Conv1d(4→3,  k=128, s=1) + BN + ReLU
    Output: 3 channels @ 125 Hz (128× decimation from 16 kHz)

Data layout (multivoice_VAD_data_generation/):
    train/   train_0000.wav  train_0000_gt.npy  …
    val/     val_0000.wav    val_0000_gt.npy    …
    test/    test_0000.wav   test_0000_gt.npy   …

GT files contain sample-level labels (0/1/2 at 48 kHz).
Audio is resampled to 16 kHz; labels are downsampled to 125 Hz by majority vote.

Usage:
    python3 train_mvad_dnn_i.py
    python3 train_mvad_dnn_i.py --epochs 200 --batch-size 64
    python3 train_mvad_dnn_i.py --augment --overlap-weight-boost 3.0
"""

import argparse
import json
import sys
import time
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from scipy.signal import resample_poly
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
from torch.utils.data import Dataset, DataLoader


# ═══════════════════════════════════════════════════════════════════════════════
#  Constants & Defaults
# ═══════════════════════════════════════════════════════════════════════════════

ORIG_SAMPLE_RATE = 48_000          # source audio sample rate
TARGET_SAMPLE_RATE = 16_000        # network input sample rate
DECIMATION_FACTOR = 128            # total network decimation (2^7)
OUTPUT_RATE_HZ = TARGET_SAMPLE_RATE / DECIMATION_FACTOR  # 125 Hz
FRAME_MS = 1000.0 / OUTPUT_RATE_HZ                      # 8 ms

NUM_CLASSES = 3
CLASS_NAMES = ['Silence', 'Single', 'Overlap']

# Layer definitions: (kernel_size, stride) for the full network
LAYER_DEFS = [
    (7, 2), (7, 2), (7, 2),           # fwd  L1–L3
    (7, 2), (9, 2), (9, 2), (9, 2),   # core L6–L9  (stride-2)
    (9, 1), (9, 1), (9, 1),           # core L10–L12 (stride-1)
    (9, 1),                            # inv  L15
    (128, 1),                          # inv  L16
]

DEFAULT_DATA_DIR = 'multivoice_VAD_data_generation'
DEFAULT_CHUNK_SAMPLES = 48_000     # 3 s at 16 kHz
DEFAULT_EPOCHS = 200
DEFAULT_BATCH_SIZE = 32
DEFAULT_LR = 1e-3
DEFAULT_OVERLAP_WEIGHT_BOOST = 2.0
DEFAULT_EARLY_STOP_PATIENCE = 25
DEFAULT_MODEL_PATH = 'mvad_dnn_i_model.pt'
DEFAULT_HISTORY_PATH = 'mvad_dnn_i_training_history.json'


# ═══════════════════════════════════════════════════════════════════════════════
#  Analytical output-length computation
# ═══════════════════════════════════════════════════════════════════════════════

def compute_output_length(input_length):
    """Compute output sequence length analytically for VAD_I_C2_L8.

    All convolutions use valid (no-padding) mode, so:
        output_len = (input_len - kernel_size) // stride + 1
    applied sequentially for each layer.
    """
    length = input_length
    for k, s in LAYER_DEFS:
        length = (length - k) // s + 1
        if length <= 0:
            raise ValueError(
                f"Input length {input_length} too short: became {length} "
                f"after layer (k={k}, s={s}).  "
                f"Minimum input: 21,339 samples (1.33 s @ 16 kHz).")
    return length


# ═══════════════════════════════════════════════════════════════════════════════
#  ResCbr1dGen Building Block
# ═══════════════════════════════════════════════════════════════════════════════

class ResCbr1dGen(nn.Module):
    """
    Residual Conv-BatchNorm-ReLU 1D block.

    - Valid (no-padding) 1D convolution
    - BatchNorm + ReLU activation
    - Optional residual skip connection (only when in_ch == out_ch)
    - For stride > 1 the skip path sub-samples the input to match output length
    - Right-aligned (causal): the skip is cropped from the right
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 residual=False):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=0, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.use_residual = residual and (in_channels == out_channels)
        self.stride = stride

    def forward(self, x):
        out = self.relu(self.bn(self.conv(x)))

        if self.use_residual:
            skip = x
            if self.stride > 1:
                skip = skip[:, :, ::self.stride]
            # Right-align: keep the last out_len samples of the skip
            out_len = out.size(2)
            skip = skip[:, :, -out_len:]
            out = out + skip

        return out


# ═══════════════════════════════════════════════════════════════════════════════
#  VAD_I_C2_L8 Model
# ═══════════════════════════════════════════════════════════════════════════════

class VAD_I_C2_L8(nn.Module):
    """
    1D Convolutional Encoder-Decoder for Multi-Voice Activity Detection.

    Implements the architecture from vad_i_c2_l8_architecture.html:

    Forward Transform (Encoder, 3 layers — no residual):
        L1:  Conv1d(1→2,  k=7,  s=2) + BN + ReLU
        L2:  Conv1d(2→4,  k=7,  s=2) + BN + ReLU
        L3:  Conv1d(4→8,  k=7,  s=2) + BN + ReLU

    Core (7 layers — with residual):
        L6:  Conv1d(8→8,  k=7,  s=2) + BN + ReLU + skip
        L7:  Conv1d(8→8,  k=9,  s=2) + BN + ReLU + skip
        L8:  Conv1d(8→8,  k=9,  s=2) + BN + ReLU + skip
        L9:  Conv1d(8→8,  k=9,  s=2) + BN + ReLU + skip
        L10: Conv1d(8→8,  k=9,  s=1) + BN + ReLU + skip
        L11: Conv1d(8→8,  k=9,  s=1) + BN + ReLU + skip
        L12: Conv1d(8→8,  k=9,  s=1) + BN + ReLU + skip

    Inverse Transform (Decoder, 2 layers — no residual):
        L15: Conv1d(8→4,          k=9,   s=1) + BN + ReLU
        L16: Conv1d(4→num_classes, k=128, s=1) + BN + ReLU

    Total decimation: 128× (2^7).
    Input:  raw 16 kHz mono audio  →  (B, 1, N_samples)
    Output: num_classes channels @ 125 Hz  →  (B, num_classes, T_out)
    """

    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()

        # ── Forward Transform (Encoder) ──
        self.fwd = nn.ModuleList([
            ResCbr1dGen(1, 2, kernel_size=7, stride=2, residual=False),   # L1
            ResCbr1dGen(2, 4, kernel_size=7, stride=2, residual=False),   # L2
            ResCbr1dGen(4, 8, kernel_size=7, stride=2, residual=False),   # L3
        ])

        # ── Core (with residual connections) ──
        self.core = nn.ModuleList([
            ResCbr1dGen(8, 8, kernel_size=7, stride=2, residual=True),    # L6
            ResCbr1dGen(8, 8, kernel_size=9, stride=2, residual=True),    # L7
            ResCbr1dGen(8, 8, kernel_size=9, stride=2, residual=True),    # L8
            ResCbr1dGen(8, 8, kernel_size=9, stride=2, residual=True),    # L9
            ResCbr1dGen(8, 8, kernel_size=9, stride=1, residual=True),    # L10
            ResCbr1dGen(8, 8, kernel_size=9, stride=1, residual=True),    # L11
            ResCbr1dGen(8, 8, kernel_size=9, stride=1, residual=True),    # L12
        ])

        # ── Inverse Transform (Decoder) ──
        self.inv = nn.ModuleList([
            ResCbr1dGen(8, 4,          kernel_size=9,   stride=1,
                        residual=False),                                  # L15
            ResCbr1dGen(4, num_classes, kernel_size=128, stride=1,
                        residual=False),                                  # L16
        ])

    def forward(self, x):
        """
        Args:
            x: (B, 1, N_samples) raw 16 kHz audio waveform
        Returns:
            (B, num_classes, T_out) per-frame logits at 125 Hz
        """
        for layer in self.fwd:
            x = layer(x)
        for layer in self.core:
            x = layer(x)
        for layer in self.inv:
            x = layer(x)
        return x


# ═══════════════════════════════════════════════════════════════════════════════
#  Audio Resampling & GT Alignment
# ═══════════════════════════════════════════════════════════════════════════════

def resample_audio(audio, orig_sr, target_sr=TARGET_SAMPLE_RATE):
    """Resample *audio* from *orig_sr* to *target_sr* via polyphase filtering."""
    if orig_sr == target_sr:
        return audio.astype(np.float32)
    g = gcd(int(target_sr), int(orig_sr))
    up = int(target_sr) // g
    down = int(orig_sr) // g
    return resample_poly(audio, up, down).astype(np.float32)


def gt_samples_to_frames_125hz(gt_samples, orig_sr, n_frames):
    """
    Convert sample-level GT (at *orig_sr*) to 125 Hz frame-level labels.

    Each 125 Hz frame spans DECIMATION_FACTOR samples at 16 kHz, which
    corresponds to (orig_sr * DECIMATION_FACTOR / TARGET_SAMPLE_RATE) samples
    at the original rate.  Label is assigned by majority vote.
    """
    samples_per_frame = int(orig_sr * DECIMATION_FACTOR / TARGET_SAMPLE_RATE)
    # e.g. 48000 * 128 / 16000 = 384

    labels = np.zeros(n_frames, dtype=np.int64)
    for i in range(n_frames):
        start = i * samples_per_frame
        end = min(start + samples_per_frame, len(gt_samples))
        if start >= len(gt_samples):
            break
        chunk = gt_samples[start:end].astype(np.int64)
        counts = np.bincount(chunk, minlength=NUM_CLASSES)
        labels[i] = int(np.argmax(counts))
    return labels


# ═══════════════════════════════════════════════════════════════════════════════
#  Data Loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_split(data_dir, prefix, verbose=True):
    """
    Load all wav + _gt.npy pairs, resample audio to 16 kHz, compute
    frame-level labels at 125 Hz.

    Returns
    -------
    file_data : list of (audio_16k : np.ndarray, frame_labels : np.ndarray)
    total_frames : int
    class_counts : np.ndarray (NUM_CLASSES,)
    """
    data_dir = Path(data_dir)
    wav_files = sorted(data_dir.glob(f'{prefix}_*.wav'))
    wav_files = [f for f in wav_files if '_gt' not in f.stem]

    file_data = []
    total_frames = 0
    class_counts = np.zeros(NUM_CLASSES, dtype=np.int64)

    t0 = time.time()
    for idx, wav_path in enumerate(wav_files):
        gt_path = wav_path.parent / f'{wav_path.stem}_gt.npy'
        if not gt_path.exists():
            continue

        audio, sr = sf.read(str(wav_path), dtype='float64')
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        gt_samples = np.load(str(gt_path)).astype(np.int32)

        # Trim to common length (at original sample rate)
        n = min(len(audio), len(gt_samples))
        audio, gt_samples = audio[:n], gt_samples[:n]

        # Resample audio to 16 kHz
        audio_16k = resample_audio(audio, sr)

        # Frame-level labels at 125 Hz
        n_frames = len(audio_16k) // DECIMATION_FACTOR
        frame_labels = gt_samples_to_frames_125hz(gt_samples, sr, n_frames)

        file_data.append((audio_16k, frame_labels))
        total_frames += n_frames
        for c in range(NUM_CLASSES):
            class_counts[c] += int(np.sum(frame_labels == c))

        if verbose and ((idx + 1) % 50 == 0 or idx + 1 == len(wav_files)):
            print(f"    [{idx + 1:>4d}/{len(wav_files)}]  "
                  f"({time.time() - t0:.1f} s)")

    return file_data, total_frames, class_counts


# ═══════════════════════════════════════════════════════════════════════════════
#  PyTorch Dataset
# ═══════════════════════════════════════════════════════════════════════════════

class MVADChunkDataset(Dataset):
    """
    Extracts fixed-length audio chunks from loaded file data.

    Each sample is a (chunk, labels) pair where:
      - chunk:  (1, chunk_samples) raw audio at 16 kHz
      - labels: (output_len,) frame-level class labels at 125 Hz

    Chunks are extracted with a stride equal to output_len × DECIMATION_FACTOR
    so that output frames are non-overlapping across consecutive chunks.
    Labels are right-aligned with the chunk (causal alignment).
    """

    def __init__(self, file_data, chunk_samples, output_len, augment=False):
        self.file_data = file_data
        self.chunk_samples = chunk_samples
        self.output_len = output_len
        self.augment = augment

        # Stride so that consecutive chunks produce non-overlapping outputs
        self.stride_samples = output_len * DECIMATION_FACTOR

        # Build flat index → (file_idx, start_sample, label_start_frame)
        self.samples = []
        for f_idx, (audio, labels) in enumerate(file_data):
            n_samples = len(audio)
            for start in range(0, n_samples - chunk_samples + 1,
                               self.stride_samples):
                end_sample = start + chunk_samples
                end_frame = end_sample // DECIMATION_FACTOR
                label_start = end_frame - output_len
                if label_start >= 0 and end_frame <= len(labels):
                    self.samples.append((f_idx, start, label_start))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        f_idx, start, label_start = self.samples[idx]
        audio, frame_labels = self.file_data[f_idx]

        chunk = audio[start: start + self.chunk_samples].copy()
        labels = frame_labels[label_start: label_start + self.output_len].copy()

        if self.augment:
            chunk = self._augment(chunk)

        return (torch.from_numpy(chunk).float().unsqueeze(0),   # (1, chunk_samples)
                torch.from_numpy(labels).long())                # (output_len,)

    @staticmethod
    def _augment(chunk):
        """Simple waveform augmentations (gain jitter + additive noise)."""
        gain_db = np.random.uniform(-6, 6)
        chunk = chunk * (10.0 ** (gain_db / 20.0))
        noise_level = np.random.uniform(0, 0.005)
        chunk = chunk + noise_level * np.random.randn(
            *chunk.shape).astype(np.float32)
        return chunk


# ═══════════════════════════════════════════════════════════════════════════════
#  Training & Evaluation Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, criterion, optimizer, device):
    """Run one training epoch.  Returns (avg_loss, accuracy)."""
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for audio_chunks, labels in loader:
        audio_chunks = audio_chunks.to(device)   # (B, 1, chunk_samples)
        labels = labels.to(device)                # (B, output_len)

        optimizer.zero_grad()
        logits = model(audio_chunks)              # (B, C, output_len)

        # CrossEntropyLoss natively handles (B, C, T) vs (B, T)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        B, C, T = logits.shape
        preds = logits.argmax(dim=1)              # (B, T)
        total_loss += loss.item() * B * T
        correct += (preds == labels).sum().item()
        n += B * T

    return total_loss / max(n, 1), correct / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate model.  Returns dict with loss, accuracy, per-class metrics."""
    model.eval()
    total_loss, n = 0.0, 0
    all_preds, all_labels = [], []

    for audio_chunks, labels in loader:
        audio_chunks = audio_chunks.to(device)
        labels = labels.to(device)

        logits = model(audio_chunks)
        loss = criterion(logits, labels)

        B, C, T = logits.shape
        preds = logits.argmax(dim=1)
        total_loss += loss.item() * B * T
        all_preds.append(preds.cpu().numpy().reshape(-1))
        all_labels.append(labels.cpu().numpy().reshape(-1))
        n += B * T

    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)

    per_f1 = f1_score(labels, preds, average=None,
                      labels=[0, 1, 2], zero_division=0)
    per_prec = precision_score(labels, preds, average=None,
                               labels=[0, 1, 2], zero_division=0)
    per_rec = recall_score(labels, preds, average=None,
                           labels=[0, 1, 2], zero_division=0)
    macro_f1 = f1_score(labels, preds, average='macro', zero_division=0)

    return dict(
        loss=total_loss / max(n, 1),
        accuracy=accuracy_score(labels, preds),
        macro_f1=macro_f1,
        silence_f1=float(per_f1[0]),  single_f1=float(per_f1[1]),
        overlap_f1=float(per_f1[2]),
        silence_prec=float(per_prec[0]),  single_prec=float(per_prec[1]),
        overlap_prec=float(per_prec[2]),
        silence_rec=float(per_rec[0]),  single_rec=float(per_rec[1]),
        overlap_rec=float(per_rec[2]),
        preds=preds, labels=labels,
    )


def print_detailed_metrics(labels, preds, title=''):
    """Pretty-print per-class metrics + confusion matrix."""
    print(f"\n{'═' * 68}")
    print(f"  {title}")
    print(f"{'═' * 68}")
    print(f"  Overall Accuracy: {accuracy_score(labels, preds):.4f}")
    print(f"  Macro F1:         "
          f"{f1_score(labels, preds, average='macro', zero_division=0):.4f}")

    cm = confusion_matrix(labels, preds, labels=[0, 1, 2])
    print(f"\n  Confusion Matrix (rows = GT, cols = Predicted):")
    print(f"  {'':>12s}  {'Silence':>8s}  {'Single':>8s}  {'Overlap':>8s}")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name:>12s}  {cm[i, 0]:8d}  {cm[i, 1]:8d}  {cm[i, 2]:8d}")

    report = classification_report(labels, preds, target_names=CLASS_NAMES,
                                    labels=[0, 1, 2], zero_division=0)
    print(f"\n{report}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description='Train VAD_I_C2_L8 for Multi-Voice Activity Detection',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ap.add_argument('--data-dir', type=str, default=DEFAULT_DATA_DIR,
                    help='Root directory with train/ val/ test/ sub-folders')
    ap.add_argument('--chunk-samples', type=int, default=DEFAULT_CHUNK_SAMPLES,
                    help='Audio chunk size in samples at 16 kHz (≥ 21339)')
    ap.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS)
    ap.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE)
    ap.add_argument('--lr', type=float, default=DEFAULT_LR,
                    help='Initial learning rate (Adam)')
    ap.add_argument('--overlap-weight-boost', type=float,
                    default=DEFAULT_OVERLAP_WEIGHT_BOOST,
                    help='Extra multiplicative weight for overlap class')
    ap.add_argument('--augment', action='store_true',
                    help='Enable waveform augmentation (gain jitter + noise)')
    ap.add_argument('--early-stop-patience', type=int,
                    default=DEFAULT_EARLY_STOP_PATIENCE)
    ap.add_argument('--model-path', type=str, default=DEFAULT_MODEL_PATH)
    ap.add_argument('--history-path', type=str, default=DEFAULT_HISTORY_PATH)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--num-workers', type=int, default=0,
                    help='DataLoader workers (0 = main process)')

    args = ap.parse_args()

    # ── Reproducibility ───────────────────────────────────────────────
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_root = Path(args.data_dir)

    # ── Compute output length for the chosen chunk size ───────────────
    output_len = compute_output_length(args.chunk_samples)

    # ── Build model ───────────────────────────────────────────────────
    model = VAD_I_C2_L8(num_classes=NUM_CLASSES).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # ── Banner ────────────────────────────────────────────────────────
    print(f"\n{'═' * 68}")
    print(f"  Multi-Voice VAD — VAD_I_C2_L8 Training")
    print(f"{'═' * 68}")
    print(f"  Device:              {device}")
    print(f"  Data dir:            {data_root}")
    print(f"  Architecture:        VAD_I_C2_L8 (1D Conv Encoder-Decoder)")
    print(f"  Trainable params:    {n_params:,}")
    print(f"  Chunk samples:       {args.chunk_samples}  "
          f"({args.chunk_samples / TARGET_SAMPLE_RATE * 1000:.0f} ms)")
    print(f"  Output frames/chunk: {output_len}")
    print(f"  Decimation:          {DECIMATION_FACTOR}×  "
          f"(16 kHz → {OUTPUT_RATE_HZ:.0f} Hz)")
    print(f"  Frame period:        {FRAME_MS:.1f} ms")
    print(f"  Epochs:              {args.epochs}")
    print(f"  Batch size:          {args.batch_size}")
    print(f"  Learning rate:       {args.lr}")
    print(f"  Overlap weight ×:    {args.overlap_weight_boost}")
    print(f"  Augmentation:        {args.augment}")
    print(f"  Early stop patience: {args.early_stop_patience}")
    print(f"  Seed:                {args.seed}")
    print(f"{'═' * 68}\n")

    # ══════════════════════════════════════════════════════════════════
    #  1. Load data
    # ══════════════════════════════════════════════════════════════════
    print("Step 1: Loading audio, resampling to 16 kHz, computing labels …\n")

    print("  ── Train ──")
    train_data, n_train, train_counts = load_split(
        data_root / 'train', 'train')
    print(f"  {len(train_data)} files, {n_train} frames  |  "
          f"Sil={train_counts[0]}  Sing={train_counts[1]}  "
          f"Ovl={train_counts[2]}\n")

    print("  ── Validation ──")
    val_data, n_val, val_counts = load_split(
        data_root / 'val', 'val')
    print(f"  {len(val_data)} files, {n_val} frames  |  "
          f"Sil={val_counts[0]}  Sing={val_counts[1]}  "
          f"Ovl={val_counts[2]}\n")

    print("  ── Test ──")
    test_data, n_test, test_counts = load_split(
        data_root / 'test', 'test')
    print(f"  {len(test_data)} files, {n_test} frames  |  "
          f"Sil={test_counts[0]}  Sing={test_counts[1]}  "
          f"Ovl={test_counts[2]}\n")

    if not train_data or not val_data:
        print("  ✗  Could not load train/val data. Check --data-dir.")
        return 1

    # ══════════════════════════════════════════════════════════════════
    #  2. Build Datasets & DataLoaders
    # ══════════════════════════════════════════════════════════════════
    print("Step 2: Building chunk datasets …")
    train_ds = MVADChunkDataset(train_data, args.chunk_samples, output_len,
                                augment=args.augment)
    val_ds = MVADChunkDataset(val_data, args.chunk_samples, output_len,
                              augment=False)
    test_ds = MVADChunkDataset(test_data, args.chunk_samples, output_len,
                               augment=False)

    nw = args.num_workers
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=nw, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=nw, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=nw, pin_memory=True)

    print(f"  Train chunks:  {len(train_ds):,}")
    print(f"  Val   chunks:  {len(val_ds):,}")
    print(f"  Test  chunks:  {len(test_ds):,}\n")

    # ══════════════════════════════════════════════════════════════════
    #  3. Class weights
    # ══════════════════════════════════════════════════════════════════
    total_c = train_counts.astype(np.float64)
    total_c[total_c < 1] = 1.0
    weights = total_c.sum() / (NUM_CLASSES * total_c)
    weights[2] *= args.overlap_weight_boost        # boost overlap class
    class_weights = torch.FloatTensor(weights).to(device)
    print(f"  Class weights (after ×{args.overlap_weight_boost} overlap boost):")
    for i, name in enumerate(CLASS_NAMES):
        print(f"    {name:>8s}: {class_weights[i].item():.4f}")
    print()

    # ══════════════════════════════════════════════════════════════════
    #  4. Print model summary
    # ══════════════════════════════════════════════════════════════════
    print("Step 3: Model summary")
    print(model)
    print(f"  Trainable parameters: {n_params:,}\n")

    # ══════════════════════════════════════════════════════════════════
    #  5. Training
    # ══════════════════════════════════════════════════════════════════
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10)

    print(f"Step 4: Training for up to {args.epochs} epochs "
          f"(early-stop patience = {args.early_stop_patience}) …")
    print(f"  Model selection criterion: validation overlap F1\n")

    header = (f"{'Ep':>4s}  {'TrLoss':>7s}  {'TrAcc':>6s}  "
              f"{'VLoss':>7s}  {'VAcc':>6s}  "
              f"{'Sil-F1':>6s}  {'Sng-F1':>6s}  {'Ovl-F1':>6s}  "
              f"{'MacF1':>6s}  {'LR':>9s}")
    print(f"  {'─' * len(header)}")
    print(f"  {header}")
    print(f"  {'─' * len(header)}")

    best_overlap_f1 = 0.0
    best_val_loss = float('inf')
    best_epoch = 0
    patience_ctr = 0
    epoch_history = []
    t_train_start = time.time()

    for epoch in range(1, args.epochs + 1):
        t_loss, t_acc = train_one_epoch(model, train_loader, criterion,
                                        optimizer, device)
        vm = evaluate(model, val_loader, criterion, device)

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(vm['overlap_f1'])

        # Record
        rec = dict(
            epoch=epoch,
            train_loss=round(t_loss, 6), train_acc=round(t_acc, 6),
            val_loss=round(vm['loss'], 6), val_acc=round(vm['accuracy'], 6),
            val_macro_f1=round(vm['macro_f1'], 6),
            val_silence_f1=round(vm['silence_f1'], 6),
            val_single_f1=round(vm['single_f1'], 6),
            val_overlap_f1=round(vm['overlap_f1'], 6),
            lr=current_lr,
        )
        epoch_history.append(rec)

        # Print progress
        print_every = 10
        if epoch % print_every == 0 or epoch == 1 or epoch == args.epochs:
            print(f"  {epoch:4d}  {t_loss:7.4f}  {t_acc:6.4f}  "
                  f"{vm['loss']:7.4f}  {vm['accuracy']:6.4f}  "
                  f"{vm['silence_f1']:6.4f}  {vm['single_f1']:6.4f}  "
                  f"{vm['overlap_f1']:6.4f}  "
                  f"{vm['macro_f1']:6.4f}  {current_lr:9.2e}")

        # ── Model selection (best overlap F1, val loss as tiebreaker) ─
        improved = (vm['overlap_f1'] > best_overlap_f1 or
                    (vm['overlap_f1'] == best_overlap_f1
                     and vm['loss'] < best_val_loss))
        if improved:
            best_overlap_f1 = vm['overlap_f1']
            best_val_loss = vm['loss']
            best_epoch = epoch
            patience_ctr = 0
            torch.save(dict(
                epoch=epoch,
                model_state_dict=model.state_dict(),
                optimizer_state_dict=optimizer.state_dict(),
                val_metrics={k: v for k, v in vm.items()
                             if k not in ('preds', 'labels')},
                config=dict(
                    arch='vad_i_c2_l8',
                    target_sample_rate=TARGET_SAMPLE_RATE,
                    decimation_factor=DECIMATION_FACTOR,
                    chunk_samples=args.chunk_samples,
                    output_len=output_len,
                    num_classes=NUM_CLASSES,
                ),
                class_weights=class_weights.cpu().numpy(),
                args=vars(args),
            ), args.model_path)
        else:
            patience_ctr += 1

        if patience_ctr >= args.early_stop_patience:
            print(f"\n  ⏹  Early stopping at epoch {epoch} "
                  f"(no improvement for {args.early_stop_patience} epochs)")
            break

    t_train = time.time() - t_train_start
    print(f"\n  Training complete in {t_train:.1f} s")
    print(f"  Best overlap F1: {best_overlap_f1:.4f} at epoch {best_epoch}")
    print(f"  Best model saved to: {args.model_path}")

    # ══════════════════════════════════════════════════════════════════
    #  6. Final evaluation on test set
    # ══════════════════════════════════════════════════════════════════
    print(f"\nStep 5: Evaluating best model on TEST set …")
    ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"  Loaded best model from epoch {ckpt['epoch']}")

    test_metrics = evaluate(model, test_loader, criterion, device)
    print_detailed_metrics(test_metrics['labels'], test_metrics['preds'],
                           title='TEST SET — Final Results')

    # ── Also evaluate on val for completeness ─────────────────────
    val_metrics_final = evaluate(model, val_loader, criterion, device)
    print_detailed_metrics(val_metrics_final['labels'],
                           val_metrics_final['preds'],
                           title='VALIDATION SET — Final Results (best model)')

    # ══════════════════════════════════════════════════════════════════
    #  7. Save training history
    # ══════════════════════════════════════════════════════════════════
    cm_test = confusion_matrix(test_metrics['labels'],
                               test_metrics['preds'], labels=[0, 1, 2])
    cm_val = confusion_matrix(val_metrics_final['labels'],
                              val_metrics_final['preds'], labels=[0, 1, 2])

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    history = dict(
        config=dict(
            arch='vad_i_c2_l8',
            target_sample_rate=TARGET_SAMPLE_RATE,
            decimation_factor=DECIMATION_FACTOR,
            output_rate_hz=OUTPUT_RATE_HZ,
            frame_ms=FRAME_MS,
            chunk_samples=args.chunk_samples,
            output_len=output_len,
            num_classes=NUM_CLASSES,
            learning_rate=args.lr,
            batch_size=args.batch_size,
            overlap_weight_boost=args.overlap_weight_boost,
            augment=args.augment,
            n_params=n_params,
            device=str(device),
            seed=args.seed,
        ),
        data=dict(
            train_files=len(train_data), train_frames=int(n_train),
            val_files=len(val_data), val_frames=int(n_val),
            test_files=len(test_data), test_frames=int(n_test),
            train_class_counts=train_counts.tolist(),
            val_class_counts=val_counts.tolist(),
            test_class_counts=test_counts.tolist(),
        ),
        epoch_history=epoch_history,
        best_epoch=best_epoch,
        training_time_s=round(t_train, 1),
        test_results=dict(
            accuracy=round(test_metrics['accuracy'], 4),
            macro_f1=round(test_metrics['macro_f1'], 4),
            silence_f1=round(test_metrics['silence_f1'], 4),
            single_f1=round(test_metrics['single_f1'], 4),
            overlap_f1=round(test_metrics['overlap_f1'], 4),
            overlap_precision=round(test_metrics['overlap_prec'], 4),
            overlap_recall=round(test_metrics['overlap_rec'], 4),
            confusion_matrix=cm_test.tolist(),
        ),
        val_results=dict(
            accuracy=round(val_metrics_final['accuracy'], 4),
            macro_f1=round(val_metrics_final['macro_f1'], 4),
            silence_f1=round(val_metrics_final['silence_f1'], 4),
            single_f1=round(val_metrics_final['single_f1'], 4),
            overlap_f1=round(val_metrics_final['overlap_f1'], 4),
            overlap_precision=round(val_metrics_final['overlap_prec'], 4),
            overlap_recall=round(val_metrics_final['overlap_rec'], 4),
            confusion_matrix=cm_val.tolist(),
        ),
    )

    with open(args.history_path, 'w') as f:
        json.dump(history, f, indent=2, cls=NumpyEncoder)

    print(f"\n  Training history saved to: {args.history_path}")
    print(f"  Best model saved to:       {args.model_path}")
    print(f"\n  ✓ Done!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
