#!/usr/bin/env python3
"""
train_mvad_dnn_v2.py — Lightweight Streaming DNN for Multi-Voice Activity Detection

Classifies each 10 ms audio frame as:
    0 = silence / no speech
    1 = single speaker
    2 = overlapping speech (≥ 2 speakers)

Architecture:
    MVAD_V2 — 1D Depthwise-Separable CNN operating on mel-filterbank features.
    Processes temporal sequences with center-aligned convolutions along the time axis.
    Designed for streaming with up to 350 ms lookahead (within 1 s budget).

    Key properties:
        ~27 K parameters        (vs ~157 K for v1 2D-CNN → 5.7× smaller)
        ~2.6 MMAC/s             (vs ~1,567 MMAC/s for v1 → ~600× lighter)
        710 ms receptive field  (vs 150 ms for v1 → 4.7× wider context)
        350 ms lookahead        (vs 70 ms for v1; within 1 s budget)
        Mel-filterbank input (40 bands, 25 ms window, 10 ms hop)
        No downsampling — output rate equals input rate (100 Hz)

Features:
    Log mel-filterbank energies (40 bands, 25 ms analysis window, 10 ms hop)
    processed by 5 depthwise-separable 1D conv blocks with residual connections.

Data layout (multivoice_VAD_data_generation/):
    train/   train_0000.wav  train_0000_gt.npy  …
    val/     val_0000.wav    val_0000_gt.npy    …
    test/    test_0000.wav   test_0000_gt.npy   …

GT files contain **sample-level** labels (0/1/2 at 48 kHz).
They are converted to frame-level by majority vote per analysis window.

Usage:
    python3 train_mvad_dnn_v2.py
    python3 train_mvad_dnn_v2.py --epochs 200 --early-stop-patience 25
    python3 train_mvad_dnn_v2.py --hidden-ch 48 --n-blocks 4 --kernel-size 11
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
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
FRAME_RATE_HZ = 1000 // FRAME_MS                              # 100

NUM_CLASSES = 3
CLASS_NAMES = ['Silence', 'Single', 'Overlap']
IGNORE_INDEX = -100          # for padded frames in CrossEntropyLoss

DEFAULT_DATA_DIR = 'multivoice_VAD_data_generation'
DEFAULT_EPOCHS = 200
DEFAULT_BATCH_SIZE = 64
DEFAULT_LR = 1e-3
DEFAULT_DROPOUT = 0.1
DEFAULT_OVERLAP_WEIGHT_BOOST = 2.0
DEFAULT_EARLY_STOP_PATIENCE = 25
DEFAULT_MODEL_PATH = 'mvad_dnn_v2_model.pt'
DEFAULT_HISTORY_PATH = 'mvad_dnn_v2_training_history.json'

# V2 architecture defaults
DEFAULT_HIDDEN_CH = 64
DEFAULT_KERNEL_SIZE = 15
DEFAULT_N_BLOCKS = 5
DEFAULT_CHUNK_FRAMES = 500   # 5 seconds for training chunks


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
#  PyTorch Dataset — Sequence Chunks
# ═══════════════════════════════════════════════════════════════════════════════

class SequenceChunkDataset(Dataset):
    """
    Yields fixed-length chunks of mel frames and labels for
    sequence-to-sequence training of the V2 model.

    Each item is a tuple (mel_chunk, label_chunk) where:
        mel_chunk:   (N_MELS, chunk_frames) float32 — ready for Conv1d
        label_chunk: (chunk_frames,) int64 — per-frame class labels

    Short tail chunks are zero-padded with label = IGNORE_INDEX (-100).
    """

    def __init__(self, file_data, chunk_frames=500, feat_mean=None, feat_std=None,
                 augment=False):
        self.chunk_frames = chunk_frames
        self.file_data = file_data
        self.feat_mean = feat_mean
        self.feat_std = feat_std
        self.augment = augment

        # Build list of (file_idx, start_frame)
        self.chunks = []
        for f_idx, (mel, _) in enumerate(file_data):
            n = len(mel)
            if n <= chunk_frames:
                self.chunks.append((f_idx, 0))
            else:
                starts = list(range(0, n - chunk_frames + 1, chunk_frames))
                if starts[-1] + chunk_frames < n:
                    starts.append(n - chunk_frames)
                for s in starts:
                    self.chunks.append((f_idx, s))

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        f_idx, start = self.chunks[idx]
        mel, labels = self.file_data[f_idx]

        end = min(start + self.chunk_frames, len(mel))
        chunk_mel = mel[start:end].copy()              # (T, N_MELS)
        chunk_labels = labels[start:end].copy()        # (T,)

        # Pad if shorter than chunk_frames (rare — only very short files)
        actual_len = len(chunk_mel)
        if actual_len < self.chunk_frames:
            pad_n = self.chunk_frames - actual_len
            chunk_mel = np.pad(chunk_mel, ((0, pad_n), (0, 0)), mode='constant')
            chunk_labels = np.pad(chunk_labels, (0, pad_n), mode='constant',
                                  constant_values=IGNORE_INDEX)

        # Z-score normalisation (fit on training set)
        if self.feat_mean is not None:
            chunk_mel = (chunk_mel - self.feat_mean) / self.feat_std

        # SpecAugment-style augmentation (training only)
        if self.augment:
            chunk_mel = self._spec_augment(chunk_mel)

        # (chunk_frames, N_MELS) → (N_MELS, chunk_frames) for Conv1d
        mel_tensor = torch.from_numpy(chunk_mel.T.astype(np.float32))
        lbl_tensor = torch.from_numpy(chunk_labels.astype(np.int64))
        return mel_tensor, lbl_tensor

    @staticmethod
    def _spec_augment(mel_chunk, n_freq_masks=2, freq_mask_width=5,
                      n_time_masks=2, time_mask_width=25):
        """Zero-out random frequency / time bands (SpecAugment-style)."""
        T, F = mel_chunk.shape
        for _ in range(n_freq_masks):
            f = np.random.randint(0, freq_mask_width + 1)
            f0 = np.random.randint(0, max(1, F - f))
            mel_chunk[:, f0:f0 + f] = 0.0
        for _ in range(n_time_masks):
            t = np.random.randint(0, min(time_mask_width + 1, max(1, T // 4)))
            t0 = np.random.randint(0, max(1, T - t))
            mel_chunk[t0:t0 + t, :] = 0.0
        return mel_chunk


# ═══════════════════════════════════════════════════════════════════════════════
#  Model Architecture — MVAD_V2 (Depthwise-Separable 1D CNN)
# ═══════════════════════════════════════════════════════════════════════════════

class DSConvBlock(nn.Module):
    """
    Depthwise Separable Conv1d block:
        DW-Conv1d + BN + ReLU → PW-Conv1d + BN + ReLU (+ optional residual).

    The depthwise convolution processes each channel independently (spatial/temporal),
    while the pointwise convolution mixes across channels.

    This factorization reduces MACs from C_in×C_out×K to C_in×K + C_in×C_out
    per time step (~12× savings for C=64, K=15).
    """

    def __init__(self, ch_in, ch_out, kernel_size=15, residual=False, dropout=0.1):
        super().__init__()
        assert kernel_size % 2 == 1, f"kernel_size must be odd, got {kernel_size}"
        self.use_residual = residual and (ch_in == ch_out)
        padding = kernel_size // 2     # center-aligned for odd kernels

        self.dw_conv = nn.Conv1d(ch_in, ch_in, kernel_size, padding=padding,
                                 groups=ch_in, bias=False)
        self.dw_bn = nn.BatchNorm1d(ch_in)
        self.pw_conv = nn.Conv1d(ch_in, ch_out, 1, bias=False)
        self.pw_bn = nn.BatchNorm1d(ch_out)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        identity = x
        x = self.relu(self.dw_bn(self.dw_conv(x)))
        x = self.dropout(self.relu(self.pw_bn(self.pw_conv(x))))
        if self.use_residual:
            x = x + identity
        return x


class MVAD_V2(nn.Module):
    """
    Lightweight streaming Multi-Voice VAD using depthwise-separable 1D convolutions.

    Operates on mel-filterbank features (N_MELS bands, 100 Hz frame rate)
    along the time axis.  Center-aligned convolutions allow a controlled
    amount of lookahead, suitable for streaming with up to 1 s latency budget.

    Default configuration (hidden=64, k=15, 5 blocks):
        Parameters:      ~27,235
        Complexity:      ~2.59 MMAC/s   (vs ~1,567 MMAC/s for v1 CNN → ~600× lighter)
        Lookahead:       350 ms         (35 frames × 10 ms)
        Receptive field: 710 ms         (71 frames × 10 ms)
        Output rate:     100 Hz         (same as input — no decimation)
    """

    def __init__(self, n_mels=N_MELS, n_classes=NUM_CLASSES, hidden_ch=64,
                 kernel_size=15, n_blocks=5, dropout=0.1):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd"
        self.n_mels = n_mels
        self.n_classes = n_classes
        self.hidden_ch = hidden_ch
        self.kernel_size = kernel_size
        self.n_blocks = n_blocks

        # ── Input projection: mel bands → hidden channels ────────────────
        self.input_proj = nn.Sequential(
            nn.Conv1d(n_mels, hidden_ch, 1, bias=False),
            nn.BatchNorm1d(hidden_ch),
            nn.ReLU(inplace=True),
        )

        # ── Depthwise-separable temporal processing blocks ───────────────
        self.blocks = nn.ModuleList()
        for i in range(n_blocks):
            is_last = (i == n_blocks - 1)
            ch_out = hidden_ch // 2 if is_last else hidden_ch
            use_res = (i > 0) and (not is_last)   # middle blocks get residual
            self.blocks.append(DSConvBlock(
                hidden_ch, ch_out, kernel_size,
                residual=use_res, dropout=dropout
            ))

        # ── Output classification head ───────────────────────────────────
        self.output_head = nn.Conv1d(hidden_ch // 2, n_classes, 1)

    def forward(self, x):
        """
        Args:
            x: (batch, n_mels, time) — mel features.
        Returns:
            logits: (batch, n_classes, time) — per-frame class logits.
        """
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        return self.output_head(x)

    @property
    def lookahead_frames(self):
        """Lookahead (future frames) needed for streaming."""
        return self.n_blocks * (self.kernel_size // 2)

    @property
    def receptive_field_frames(self):
        """Total receptive field in frames."""
        return self.n_blocks * (self.kernel_size - 1) + 1


# ═══════════════════════════════════════════════════════════════════════════════
#  Complexity Analysis
# ═══════════════════════════════════════════════════════════════════════════════

def compute_model_complexity(hidden_ch, n_blocks, kernel_size, n_mels=N_MELS,
                             n_classes=NUM_CLASSES, frame_rate=FRAME_RATE_HZ):
    """
    Analytically compute per-layer and total MACs for MVAD_V2.

    Returns
    -------
    layers : list of (name, params, macs_per_frame)
    total_params, total_macs_per_frame, mmacs_per_s : int, int, float
    """
    layers = []

    # Input projection
    p = n_mels * hidden_ch + 2 * hidden_ch   # Conv(no bias) + BN
    m = n_mels * hidden_ch
    layers.append(('Input Conv1d({}→{}, k=1) + BN + ReLU'.format(n_mels, hidden_ch),
                   p, m))

    # DS blocks
    for i in range(n_blocks):
        is_last = (i == n_blocks - 1)
        ch_out = hidden_ch // 2 if is_last else hidden_ch
        use_res = (i > 0) and (not is_last)

        dw_p = hidden_ch * kernel_size + 2 * hidden_ch      # DW conv + BN
        pw_p = hidden_ch * ch_out + 2 * ch_out               # PW conv + BN
        dw_m = hidden_ch * kernel_size
        pw_m = hidden_ch * ch_out

        res_str = ' + Res' if use_res else ''
        name = (f'DS Block {i + 1}: DW-Conv1d({hidden_ch}, k={kernel_size}) '
                f'+ PW-Conv1d({hidden_ch}→{ch_out}){res_str}')
        layers.append((name, dw_p + pw_p, dw_m + pw_m))

    # Output
    out_ch = hidden_ch // 2
    p = out_ch * n_classes + n_classes     # Conv(with bias)
    m = out_ch * n_classes
    layers.append((f'Output Conv1d({out_ch}→{n_classes}, k=1)', p, m))

    total_params = sum(pr for _, pr, _ in layers)
    total_macs = sum(mc for _, _, mc in layers)
    mmacs_per_s = total_macs * frame_rate / 1e6

    return layers, total_params, total_macs, mmacs_per_s


# ═══════════════════════════════════════════════════════════════════════════════
#  Training & Evaluation Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, criterion, optimizer, device):
    """Run one training epoch on chunks.  Returns (avg_loss, accuracy)."""
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for mel_chunks, label_chunks in loader:
        mel_chunks = mel_chunks.to(device)        # (B, N_MELS, T)
        label_chunks = label_chunks.to(device)    # (B, T)

        optimizer.zero_grad()
        logits = model(mel_chunks)                # (B, NUM_CLASSES, T)
        loss = criterion(logits, label_chunks)
        loss.backward()
        optimizer.step()

        valid = (label_chunks != IGNORE_INDEX)
        n_valid = valid.sum().item()
        total_loss += loss.item() * n_valid
        preds = logits.argmax(dim=1)              # (B, T)
        correct += ((preds == label_chunks) & valid).sum().item()
        n += n_valid

    return total_loss / max(n, 1), correct / max(n, 1)


@torch.no_grad()
def evaluate(model, file_data, feat_mean, feat_std, criterion, device):
    """
    Evaluate model on full files (no chunking) for clean, unbiased metrics.

    Returns dict with loss, accuracy, per-class metrics, and raw predictions.
    """
    model.eval()
    total_loss, n = 0.0, 0
    all_preds, all_labels = [], []

    for mel, labels in file_data:
        mel_norm = ((mel - feat_mean) / feat_std).astype(np.float32)

        x = torch.from_numpy(mel_norm.T).unsqueeze(0).to(device)        # (1, N_MELS, T)
        target = torch.from_numpy(labels.astype(np.int64)).unsqueeze(0).to(device)  # (1, T)

        logits = model(x)                              # (1, NUM_CLASSES, T)
        loss = criterion(logits, target)

        n_frames = labels.shape[0]
        total_loss += loss.item() * n_frames

        preds = logits.argmax(dim=1).squeeze(0).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels)
        n += n_frames

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

    print(f"\n{classification_report(labels, preds, target_names=CLASS_NAMES, labels=[0, 1, 2], zero_division=0)}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description='Train a lightweight streaming DNN (V2) for Multi-Voice VAD',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ap.add_argument('--data-dir', type=str, default=DEFAULT_DATA_DIR,
                    help='Root directory with train/ val/ test/ sub-folders')
    ap.add_argument('--hidden-ch', type=int, default=DEFAULT_HIDDEN_CH,
                    help='Hidden channel width for DS-Conv blocks')
    ap.add_argument('--kernel-size', type=int, default=DEFAULT_KERNEL_SIZE,
                    help='Temporal kernel size (must be odd)')
    ap.add_argument('--n-blocks', type=int, default=DEFAULT_N_BLOCKS,
                    help='Number of DS-Conv blocks')
    ap.add_argument('--chunk-frames', type=int, default=DEFAULT_CHUNK_FRAMES,
                    help='Training chunk length in frames (10 ms each)')
    ap.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS)
    ap.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE)
    ap.add_argument('--lr', type=float, default=DEFAULT_LR,
                    help='Initial learning rate (Adam)')
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

    # Ensure kernel_size is odd
    if args.kernel_size % 2 == 0:
        args.kernel_size += 1
        print(f"  [INFO] kernel_size adjusted to {args.kernel_size} (must be odd)")

    # ── Reproducibility ───────────────────────────────────────────────
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_root = Path(args.data_dir)

    # ── Compute architecture properties ───────────────────────────────
    layer_info, total_params_est, total_macs, mmacs_s = compute_model_complexity(
        args.hidden_ch, args.n_blocks, args.kernel_size)

    lookahead_frames = args.n_blocks * (args.kernel_size // 2)
    lookahead_ms = lookahead_frames * FRAME_MS
    receptive_frames = args.n_blocks * (args.kernel_size - 1) + 1
    receptive_ms = receptive_frames * FRAME_MS

    # ── Banner ────────────────────────────────────────────────────────
    print(f"\n{'═' * 68}")
    print(f"  Multi-Voice VAD — V2 Streaming DNN Training")
    print(f"{'═' * 68}")
    print(f"  Device:              {device}")
    print(f"  Data dir:            {data_root}")
    print(f"  Architecture:        MVAD_V2 (DS-Conv1D)")
    print(f"  Hidden channels:     {args.hidden_ch}")
    print(f"  Kernel size:         {args.kernel_size}")
    print(f"  DS-Conv blocks:      {args.n_blocks}")
    print(f"  Mel bands:           {N_MELS}  "
          f"(window {ANALYSIS_WINDOW_MS} ms, hop {FRAME_MS} ms)")
    print(f"  Chunk frames:        {args.chunk_frames}  "
          f"({args.chunk_frames * FRAME_MS} ms)")
    print(f"  Parameters:          ~{total_params_est:,}")
    print(f"  Complexity:          {mmacs_s:.2f} MMAC/s")
    print(f"  Lookahead:           {lookahead_frames} frames = {lookahead_ms} ms")
    print(f"  Receptive field:     {receptive_frames} frames = {receptive_ms} ms")
    print(f"  Epochs:              {args.epochs}")
    print(f"  Batch size:          {args.batch_size}")
    print(f"  Learning rate:       {args.lr}")
    print(f"  Dropout:             {args.dropout}")
    print(f"  Overlap weight ×:    {args.overlap_weight_boost}")
    print(f"  Augmentation:        {args.augment}")
    print(f"  Early stop patience: {args.early_stop_patience}")
    print(f"  Seed:                {args.seed}")
    print(f"{'═' * 68}")

    # ── Per-layer complexity ──────────────────────────────────────────
    print(f"\n  Per-layer complexity:")
    print(f"  {'Layer':<55s}  {'Params':>8s}  {'MACs/fr':>8s}  {'MMAC/s':>7s}")
    print(f"  {'─' * 82}")
    for name, params, macs in layer_info:
        ms = macs * FRAME_RATE_HZ / 1e6
        print(f"  {name:<55s}  {params:>8,}  {macs:>8,}  {ms:>7.3f}")
    print(f"  {'─' * 82}")
    print(f"  {'TOTAL':<55s}  {total_params_est:>8,}  {total_macs:>8,}  {mmacs_s:>7.3f}")
    print()

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
    train_ds = SequenceChunkDataset(train_data, args.chunk_frames,
                                    feat_mean, feat_std,
                                    augment=args.augment)

    nw = args.num_workers
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=nw, pin_memory=True)

    print(f"  Train chunks:   {len(train_ds):,}  "
          f"({args.chunk_frames} frames × {FRAME_MS} ms = "
          f"{args.chunk_frames * FRAME_MS / 1000:.1f} s each)")
    print(f"  Train frames:   {n_train:,}")
    print(f"  Val files:      {len(val_data)}  (evaluated per-file)")
    print(f"  Test files:     {len(test_data)}  (evaluated per-file)\n")

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
    model = MVAD_V2(
        n_mels=N_MELS,
        n_classes=NUM_CLASSES,
        hidden_ch=args.hidden_ch,
        kernel_size=args.kernel_size,
        n_blocks=args.n_blocks,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Architecture: MVAD_V2 (DS-Conv1D)")
    print(model)
    print(f"  Trainable parameters: {n_params:,}\n")

    # ══════════════════════════════════════════════════════════════════
    #  6. Training
    # ══════════════════════════════════════════════════════════════════
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=IGNORE_INDEX)
    # Separate criterion for eval (no ignore_index needed — full files, no padding)
    criterion_eval = nn.CrossEntropyLoss(weight=class_weights)

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
        vm = evaluate(model, val_data, feat_mean, feat_std,
                      criterion_eval, device)

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
                val_metrics={k: v for k, v in vm.items()
                             if k not in ('preds', 'labels')},
                config=dict(
                    arch='mvad_v2_ds_cnn1d',
                    n_mels=N_MELS,
                    n_fft=N_FFT,
                    fmin=FMIN, fmax=FMAX,
                    hop_samples=HOP_SAMPLES,
                    analysis_window_samples=ANALYSIS_WINDOW_SAMPLES,
                    sample_rate=SAMPLE_RATE,
                    frame_rate_hz=FRAME_RATE_HZ,
                    hidden_ch=args.hidden_ch,
                    kernel_size=args.kernel_size,
                    n_blocks=args.n_blocks,
                    dropout=args.dropout,
                    num_classes=NUM_CLASSES,
                    lookahead_ms=lookahead_ms,
                    receptive_field_ms=receptive_ms,
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

    test_metrics = evaluate(model, test_data, feat_mean, feat_std,
                            criterion_eval, device)
    print_detailed_metrics(test_metrics['labels'], test_metrics['preds'],
                           title='TEST SET — Final Results')

    # ── Also evaluate on val for completeness ─────────────────────
    val_metrics_final = evaluate(model, val_data, feat_mean, feat_std,
                                 criterion_eval, device)
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
            arch='mvad_v2_ds_cnn1d',
            n_mels=N_MELS,
            analysis_window_ms=ANALYSIS_WINDOW_MS,
            hop_ms=FRAME_MS,
            sample_rate=SAMPLE_RATE,
            frame_rate_hz=FRAME_RATE_HZ,
            hidden_ch=args.hidden_ch,
            kernel_size=args.kernel_size,
            n_blocks=args.n_blocks,
            chunk_frames=args.chunk_frames,
            dropout=args.dropout,
            learning_rate=args.lr,
            batch_size=args.batch_size,
            overlap_weight_boost=args.overlap_weight_boost,
            augment=args.augment,
            n_params=n_params,
            mmacs_per_s=round(mmacs_s, 3),
            lookahead_ms=lookahead_ms,
            receptive_field_ms=receptive_ms,
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
        complexity=dict(
            total_params=total_params_est,
            total_macs_per_frame=total_macs,
            mmacs_per_s=round(mmacs_s, 3),
            per_layer=[dict(name=n, params=p, macs=m)
                       for n, p, m in layer_info],
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
