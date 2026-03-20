#!/usr/bin/env python3
"""
train_multivoice_VAD.py — Train a DNN for Multi-Voice Activity Detection

Classifies each 10 ms audio frame as:
    0 = silence / no speech
    1 = single speaker
    2 = overlapping speech (≥ 2 speakers)

Features:
    Log mel-filterbank energies (40 bands, 25 ms analysis window, 10 ms hop)
    fed as a multi-frame context window (default 15 frames = 150 ms).

Architectures (--arch):
    cnn   — 2-D CNN treating the mel context as a small image (default)
    gru   — Bidirectional GRU with centre-frame readout
    mlp   — Fully-connected MLP with BatchNorm

Data layout (multivoice_VAD_data_generation/):
    train/   train_0000.wav  train_0000_gt.npy  …
    val/     val_0000.wav    val_0000_gt.npy    …
    test/    test_0000.wav   test_0000_gt.npy   …

GT files contain **sample-level** labels (0/1/2 at 48 kHz).
They are converted to frame-level by majority vote per analysis window.

Usage:
    python3 train_multivoice_VAD.py
    python3 train_multivoice_VAD.py --arch gru --context-frames 21 --epochs 120
    python3 train_multivoice_VAD.py --overlap-weight-boost 3.0 --batch-size 512
"""

import argparse
import json
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import stft as scipy_stft
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
from torch.utils.data import Dataset, DataLoader


# ═══════════════════════════════════════════════════════════════════════════════
#  Constants & Defaults
# ═══════════════════════════════════════════════════════════════════════════════

SAMPLE_RATE = 48_000
FRAME_MS = 10                                                  # hop = 10 ms
HOP_SAMPLES = int(SAMPLE_RATE * FRAME_MS / 1000)               # 480
ANALYSIS_WINDOW_MS = 25                                        # FFT window
ANALYSIS_WINDOW_SAMPLES = int(SAMPLE_RATE * ANALYSIS_WINDOW_MS / 1000)  # 1200
N_FFT = 2048
N_MELS = 40
FMIN = 80.0
FMAX = 8000.0

NUM_CLASSES = 3
CLASS_NAMES = ['Silence', 'Single', 'Overlap']

DEFAULT_DATA_DIR = 'multivoice_VAD_data_generation'
DEFAULT_CONTEXT_FRAMES = 15      # 7 past + current + 7 future = 150 ms
DEFAULT_EPOCHS = 80
DEFAULT_BATCH_SIZE = 256
DEFAULT_LR = 1e-3
DEFAULT_DROPOUT = 0.3
DEFAULT_HIDDEN_DIMS = [512, 256, 128]
DEFAULT_OVERLAP_WEIGHT_BOOST = 2.0
DEFAULT_EARLY_STOP_PATIENCE = 15
DEFAULT_MODEL_PATH = 'mvad_dnn_model.pt'
DEFAULT_HISTORY_PATH = 'mvad_dnn_training_history.json'


# ═══════════════════════════════════════════════════════════════════════════════
#  Mel-Filterbank Feature Extraction
# ═══════════════════════════════════════════════════════════════════════════════

def _hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + np.asarray(hz, dtype=np.float64) / 700.0)


def _mel_to_hz(mel):
    return 700.0 * (10.0 ** (np.asarray(mel, dtype=np.float64) / 2595.0) - 1.0)


def create_mel_filterbank(sr, n_fft, n_mels, fmin=0.0, fmax=None):
    """Create triangular mel-filterbank matrix → (n_mels, n_fft//2+1)."""
    if fmax is None:
        fmax = sr / 2.0
    n_freqs = n_fft // 2 + 1
    mel_points = np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_mels + 2)
    hz_points = _mel_to_hz(mel_points)
    bins = np.round(hz_points * n_fft / sr).astype(int)
    bins = np.clip(bins, 0, n_freqs - 1)

    fb = np.zeros((n_mels, n_freqs), dtype=np.float64)
    for m in range(n_mels):
        left, centre, right = bins[m], bins[m + 1], bins[m + 2]
        if centre > left:
            fb[m, left:centre + 1] = np.linspace(0.0, 1.0, centre - left + 1)
        if right > centre:
            fb[m, centre:right + 1] = np.linspace(1.0, 0.0, right - centre + 1)
    return fb.astype(np.float32)


# Module-level cache so the filterbank is built only once
_mel_fb_cache = {}


def _get_mel_fb(sr=SAMPLE_RATE):
    if sr not in _mel_fb_cache:
        _mel_fb_cache[sr] = create_mel_filterbank(sr, N_FFT, N_MELS, FMIN, FMAX)
    return _mel_fb_cache[sr]


def compute_log_mel(audio, sr=SAMPLE_RATE):
    """
    Compute log mel-filterbank energies for *audio*.

    Returns
    -------
    log_mel : np.ndarray  (n_frames, N_MELS)   float32
    """
    mel_fb = _get_mel_fb(sr)
    noverlap = ANALYSIS_WINDOW_SAMPLES - HOP_SAMPLES
    _, _, Zxx = scipy_stft(audio, fs=sr, window='hann',
                           nperseg=ANALYSIS_WINDOW_SAMPLES,
                           noverlap=noverlap, nfft=N_FFT)
    power = np.abs(Zxx) ** 2                        # (n_freqs, n_frames)
    mel_energy = mel_fb @ power                      # (n_mels, n_frames)
    log_mel = np.log(np.maximum(mel_energy, 1e-10))
    return log_mel.T.astype(np.float32)              # (n_frames, n_mels)


# ═══════════════════════════════════════════════════════════════════════════════
#  Ground-Truth Alignment
# ═══════════════════════════════════════════════════════════════════════════════

def gt_samples_to_frames(gt_samples, n_frames):
    """Convert sample-level GT → frame-level by majority vote per hop window."""
    labels = np.zeros(n_frames, dtype=np.int64)
    for i in range(n_frames):
        start = i * HOP_SAMPLES
        end = min(start + ANALYSIS_WINDOW_SAMPLES, len(gt_samples))
        if start >= len(gt_samples):
            break
        chunk = gt_samples[start:end]
        counts = np.bincount(chunk.astype(np.int64), minlength=3)
        labels[i] = int(np.argmax(counts))
    return labels


# ═══════════════════════════════════════════════════════════════════════════════
#  Data Loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_split(data_dir, prefix, verbose=True):
    """
    Load all wav + _gt.npy pairs from *data_dir* whose name starts with *prefix*.

    Returns
    -------
    file_data : list of (mel_features, frame_labels)
    total_frames : int
    class_counts : np.ndarray (3,)
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

        n = min(len(audio), len(gt_samples))
        audio, gt_samples = audio[:n], gt_samples[:n]

        mel = compute_log_mel(audio, sr)
        frame_labels = gt_samples_to_frames(gt_samples, len(mel))

        n_f = min(len(mel), len(frame_labels))
        mel, frame_labels = mel[:n_f], frame_labels[:n_f]

        file_data.append((mel, frame_labels))
        total_frames += n_f
        for c in range(NUM_CLASSES):
            class_counts[c] += int(np.sum(frame_labels == c))

        if verbose and ((idx + 1) % 50 == 0 or idx + 1 == len(wav_files)):
            print(f"    [{idx + 1:>4d}/{len(wav_files)}]  "
                  f"({time.time() - t0:.1f} s)")

    return file_data, total_frames, class_counts


# ═══════════════════════════════════════════════════════════════════════════════
#  PyTorch Dataset
# ═══════════════════════════════════════════════════════════════════════════════

class MultivoiceVADDataset(Dataset):
    """
    Wraps a list of (mel, labels) per file.
    Extracts context windows on the fly (memory-efficient).
    Optionally applies SpecAugment during training.
    """

    def __init__(self, file_data, context_frames, feat_mean=None, feat_std=None,
                 augment=False):
        self.context_frames = context_frames
        self.half_ctx = context_frames // 2
        self.file_data = file_data
        self.feat_mean = feat_mean
        self.feat_std = feat_std
        self.augment = augment

        # Build flat index → (file_idx, frame_idx)
        self.samples = []
        for f_idx, (mel, _) in enumerate(file_data):
            n = len(mel)
            for i in range(self.half_ctx, n - self.half_ctx):
                self.samples.append((f_idx, i))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        f_idx, frame_idx = self.samples[idx]
        mel, labels = self.file_data[f_idx]

        ctx = mel[frame_idx - self.half_ctx:
                   frame_idx + self.half_ctx + 1].copy()   # (C, N_MELS)

        # Z-score normalisation (fit on training set)
        if self.feat_mean is not None:
            ctx = (ctx - self.feat_mean) / self.feat_std

        # SpecAugment-style augmentation (training only)
        if self.augment:
            ctx = self._spec_augment(ctx)

        label = labels[frame_idx]
        return torch.from_numpy(ctx), torch.tensor(label, dtype=torch.long)

    @staticmethod
    def _spec_augment(ctx, n_freq_masks=1, freq_mask_width=4,
                      n_time_masks=1, time_mask_width=2):
        """Zero-out random frequency / time bands (SpecAugment)."""
        T, F = ctx.shape
        for _ in range(n_freq_masks):
            f = np.random.randint(0, freq_mask_width + 1)
            f0 = np.random.randint(0, max(1, F - f))
            ctx[:, f0:f0 + f] = 0.0
        for _ in range(n_time_masks):
            t = np.random.randint(0, time_mask_width + 1)
            t0 = np.random.randint(0, max(1, T - t))
            ctx[t0:t0 + t, :] = 0.0
        return ctx


# ═══════════════════════════════════════════════════════════════════════════════
#  Model Architectures
# ═══════════════════════════════════════════════════════════════════════════════

class MultivoiceVAD_CNN(nn.Module):
    """
    2-D CNN that treats the (context_frames × n_mels) mel patch as a small
    single-channel image.

    Architecture:
        Conv-BN-ReLU × 2 → Pool → Conv-BN-ReLU × 2 → Pool
        → Conv-BN-ReLU → AdaptiveAvgPool → FC-BN-ReLU-Drop → FC(3)
    """

    def __init__(self, context_frames, n_mels, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),
            # Block 2
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),
            # Block 3 → global pool
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, NUM_CLASSES),
        )

    def forward(self, x):
        # x: (B, context_frames, n_mels)
        x = x.unsqueeze(1)          # (B, 1, C, M)
        x = self.features(x)
        return self.classifier(x)


class MultivoiceVAD_GRU(nn.Module):
    """
    Bidirectional GRU — reads the full context window and takes the
    centre-frame hidden state as the representation.
    """

    def __init__(self, context_frames, n_mels, hidden_size=128,
                 num_layers=2, dropout=0.3):
        super().__init__()
        self.half_ctx = context_frames // 2
        self.gru = nn.GRU(
            input_size=n_mels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, NUM_CLASSES),
        )

    def forward(self, x):
        # x: (B, context_frames, n_mels)
        out, _ = self.gru(x)                     # (B, C, hidden*2)
        centre = out[:, self.half_ctx, :]         # (B, hidden*2)
        return self.classifier(centre)


class MultivoiceVAD_MLP(nn.Module):
    """Fully-connected MLP with BatchNorm."""

    def __init__(self, context_frames, n_mels, hidden_dims=None, dropout=0.3):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]

        input_dim = context_frames * n_mels
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h),
                       nn.ReLU(inplace=True), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, NUM_CLASSES))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x.flatten(1))


# ═══════════════════════════════════════════════════════════════════════════════
#  Training & Evaluation Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, criterion, optimizer, device):
    """Run one training epoch.  Returns (avg_loss, accuracy)."""
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for features, labels in loader:
        features, labels = features.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(labels)
        correct += (logits.argmax(1) == labels).sum().item()
        n += len(labels)
    return total_loss / n, correct / n


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate model.  Returns dict with loss, accuracy, per-class metrics."""
    model.eval()
    total_loss, n = 0.0, 0
    all_preds, all_labels = [], []

    for features, labels in loader:
        features, labels = features.to(device), labels.to(device)
        logits = model(features)
        loss = criterion(logits, labels)
        total_loss += loss.item() * len(labels)
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        n += len(labels)

    preds = np.array(all_preds)
    labels = np.array(all_labels)

    per_f1 = f1_score(labels, preds, average=None,
                      labels=[0, 1, 2], zero_division=0)
    per_prec = precision_score(labels, preds, average=None,
                               labels=[0, 1, 2], zero_division=0)
    per_rec = recall_score(labels, preds, average=None,
                           labels=[0, 1, 2], zero_division=0)
    macro_f1 = f1_score(labels, preds, average='macro', zero_division=0)

    return dict(
        loss=total_loss / n,
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

    print(f"\n{classification_report(labels, preds, target_names=CLASS_NAMES, labels=[0, 1, 2], zero_division=0)}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description='Train a DNN for Multi-Voice Activity Detection',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ap.add_argument('--data-dir', type=str, default=DEFAULT_DATA_DIR,
                    help='Root directory with train/ val/ test/ sub-folders')
    ap.add_argument('--arch', choices=['cnn', 'gru', 'mlp'], default='cnn',
                    help='Model architecture')
    ap.add_argument('--context-frames', type=int, default=DEFAULT_CONTEXT_FRAMES,
                    help='Number of context frames (must be odd)')
    ap.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS)
    ap.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE)
    ap.add_argument('--lr', type=float, default=DEFAULT_LR,
                    help='Initial learning rate (Adam)')
    ap.add_argument('--hidden-dims', type=int, nargs='+',
                    default=DEFAULT_HIDDEN_DIMS,
                    help='Hidden layer dims for MLP architecture')
    ap.add_argument('--dropout', type=float, default=DEFAULT_DROPOUT)
    ap.add_argument('--overlap-weight-boost', type=float,
                    default=DEFAULT_OVERLAP_WEIGHT_BOOST,
                    help='Extra multiplicative weight for overlap class')
    ap.add_argument('--augment', action='store_true',
                    help='Enable SpecAugment-style data augmentation')
    ap.add_argument('--early-stop-patience', type=int,
                    default=DEFAULT_EARLY_STOP_PATIENCE)
    ap.add_argument('--model-path', type=str, default=DEFAULT_MODEL_PATH)
    ap.add_argument('--history-path', type=str, default=DEFAULT_HISTORY_PATH)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--num-workers', type=int, default=0,
                    help='DataLoader workers (0 = main process)')

    args = ap.parse_args()

    # Ensure context_frames is odd so the window is symmetric
    if args.context_frames % 2 == 0:
        args.context_frames += 1
        print(f"  [INFO] context_frames adjusted to {args.context_frames} (must be odd)")

    # ── Reproducibility ───────────────────────────────────────────────
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_root = Path(args.data_dir)

    # ── Banner ────────────────────────────────────────────────────────
    print(f"\n{'═' * 68}")
    print(f"  Multi-Voice VAD — DNN Training")
    print(f"{'═' * 68}")
    print(f"  Device:              {device}")
    print(f"  Data dir:            {data_root}")
    print(f"  Architecture:        {args.arch}")
    print(f"  Context frames:      {args.context_frames}  "
          f"({args.context_frames * FRAME_MS} ms)")
    print(f"  Mel bands:           {N_MELS}  "
          f"(window {ANALYSIS_WINDOW_MS} ms, hop {FRAME_MS} ms)")
    print(f"  Epochs:              {args.epochs}")
    print(f"  Batch size:          {args.batch_size}")
    print(f"  Learning rate:       {args.lr}")
    print(f"  Dropout:             {args.dropout}")
    print(f"  Overlap weight ×:    {args.overlap_weight_boost}")
    print(f"  Augmentation:        {args.augment}")
    print(f"  Early stop patience: {args.early_stop_patience}")
    print(f"  Seed:                {args.seed}")
    print(f"{'═' * 68}\n")

    # ══════════════════════════════════════════════════════════════════
    #  1. Load data
    # ══════════════════════════════════════════════════════════════════
    print("Step 1: Loading and extracting mel features …\n")

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
    #  2. Feature normalisation (z-score, fit on training set)
    # ══════════════════════════════════════════════════════════════════
    print("Step 2: Computing feature statistics on training set …")
    all_train_mel = np.concatenate([mel for mel, _ in train_data], axis=0)
    feat_mean = all_train_mel.mean(axis=0).astype(np.float32)
    feat_std = all_train_mel.std(axis=0).astype(np.float32)
    feat_std[feat_std < 1e-8] = 1.0
    del all_train_mel
    print(f"  mean range: [{feat_mean.min():.2f}, {feat_mean.max():.2f}]")
    print(f"  std  range: [{feat_std.min():.2f}, {feat_std.max():.2f}]\n")

    # ══════════════════════════════════════════════════════════════════
    #  3. Build Datasets & DataLoaders
    # ══════════════════════════════════════════════════════════════════
    print("Step 3: Building datasets …")
    train_ds = MultivoiceVADDataset(train_data, args.context_frames,
                                    feat_mean, feat_std,
                                    augment=args.augment)
    val_ds = MultivoiceVADDataset(val_data, args.context_frames,
                                  feat_mean, feat_std, augment=False)
    test_ds = MultivoiceVADDataset(test_data, args.context_frames,
                                   feat_mean, feat_std, augment=False)

    nw = args.num_workers
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=nw, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=nw, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=nw, pin_memory=True)

    print(f"  Train samples:  {len(train_ds):,}")
    print(f"  Val   samples:  {len(val_ds):,}")
    print(f"  Test  samples:  {len(test_ds):,}\n")

    # ══════════════════════════════════════════════════════════════════
    #  4. Class weights
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
    #  5. Build Model
    # ══════════════════════════════════════════════════════════════════
    print("Step 4: Building model …")
    if args.arch == 'cnn':
        model = MultivoiceVAD_CNN(args.context_frames, N_MELS, args.dropout)
    elif args.arch == 'gru':
        model = MultivoiceVAD_GRU(args.context_frames, N_MELS,
                                  dropout=args.dropout)
    else:
        model = MultivoiceVAD_MLP(args.context_frames, N_MELS,
                                  args.hidden_dims, args.dropout)
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Architecture: {args.arch}")
    print(model)
    print(f"  Trainable parameters: {n_params:,}\n")

    # ══════════════════════════════════════════════════════════════════
    #  6. Training
    # ══════════════════════════════════════════════════════════════════
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=8)

    print(f"Step 5: Training for up to {args.epochs} epochs "
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
        print_every = 5 if args.epochs <= 100 else 10
        if epoch % print_every == 0 or epoch == 1 or epoch == args.epochs:
            print(f"  {epoch:4d}  {t_loss:7.4f}  {t_acc:6.4f}  "
                  f"{vm['loss']:7.4f}  {vm['accuracy']:6.4f}  "
                  f"{vm['silence_f1']:6.4f}  {vm['single_f1']:6.4f}  "
                  f"{vm['overlap_f1']:6.4f}  "
                  f"{vm['macro_f1']:6.4f}  {current_lr:9.2e}")

        # ── Model selection (best overlap F1, with val loss as tiebreaker) ─
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
                val_metrics=vm | {'preds': None, 'labels': None},
                config=dict(
                    arch=args.arch,
                    context_frames=args.context_frames,
                    n_mels=N_MELS,
                    n_fft=N_FFT,
                    fmin=FMIN, fmax=FMAX,
                    hop_samples=HOP_SAMPLES,
                    analysis_window_samples=ANALYSIS_WINDOW_SAMPLES,
                    sample_rate=SAMPLE_RATE,
                    hidden_dims=args.hidden_dims if args.arch == 'mlp' else None,
                    dropout=args.dropout,
                    num_classes=NUM_CLASSES,
                ),
                standardisation=dict(mean=feat_mean, std=feat_std),
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
    #  7. Final evaluation on test set
    # ══════════════════════════════════════════════════════════════════
    print(f"\nStep 6: Evaluating best model on TEST set …")
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
    #  8. Save training history
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
            arch=args.arch,
            context_frames=args.context_frames,
            n_mels=N_MELS,
            analysis_window_ms=ANALYSIS_WINDOW_MS,
            hop_ms=FRAME_MS,
            sample_rate=SAMPLE_RATE,
            hidden_dims=args.hidden_dims if args.arch == 'mlp' else None,
            dropout=args.dropout,
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
