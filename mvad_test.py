#!/usr/bin/env python3
"""
multivoice_VAD_test.py — Multi-Voice Activity Detection + Evaluation

Classifies each audio frame as:
    0 = silence / no speech
    1 = single speaker
    2 = overlapping speech (multiple speakers)

Traditional signal-processing approach (v2 — enhanced):
    • YIN pitch tracking (robust F0 + confidence)
    • Harmonic-to-Noise Ratio (HNR) — overlap reduces harmonicity
    • Spectral bandwidth — overlap broadens spectrum
    • Sub-band entropy — overlap distributes energy uniformly
    • Delta energy — detects second speaker entering
    • Multi-feature overlap scoring with adaptive thresholds
    • Temporal context smoothing over ±K frames

Modes:
    1. Single file:   python3 multivoice_VAD_test.py -i inputs/example_1.wav --plot
    2. Evaluation:    python3 multivoice_VAD_test.py --eval-dir multivoice_VAD_data_generation/test
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.ndimage import maximum_filter1d, uniform_filter1d
from scipy.signal import butter, filtfilt, medfilt, stft as scipy_stft

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ===========================================================================
#  Signal pre-processing helpers
# ===========================================================================

def highpass_filter(signal, sr, cutoff=80, order=4):
    """Butterworth high-pass filter to remove DC offset and low-freq rumble."""
    nyq = sr / 2.0
    b, a = butter(order, cutoff / nyq, btype='high')
    return filtfilt(b, a, signal).astype(np.float64)


def pre_emphasis(signal, coeff=0.97):
    """Pre-emphasis filter — boosts high frequencies for better ZCR / energy."""
    return np.append(signal[0], signal[1:] - coeff * signal[:-1])


def frame_signal(signal, frame_len, hop_len):
    """Split *signal* into overlapping frames  →  (n_frames, frame_len)."""
    n_frames = 1 + (len(signal) - frame_len) // hop_len
    idx = np.arange(frame_len)[None, :] + np.arange(n_frames)[:, None] * hop_len
    return signal[idx]


# ===========================================================================
#  Per-frame feature extraction
# ===========================================================================

def compute_energy(frames):
    """Short-Time Energy (mean squared amplitude) per frame."""
    return np.mean(frames ** 2, axis=1)


def compute_zcr(frames):
    """
    Zero-Crossing Rate per frame.

    ZCR = (number of sign changes) / (frame_length − 1)

    • Voiced speech  → moderate ZCR  (quasi-periodic waveform)
    • Unvoiced speech (fricatives, /s/, /f/) → high ZCR
    • Silence / low-energy noise → variable (but energy is low)
    """
    signs = np.sign(frames)
    crossings = np.abs(np.diff(signs, axis=1))
    return np.sum(crossings > 0, axis=1) / (frames.shape[1] - 1)


# ---------------------------------------------------------------------------
# Pitch (F0) estimation via autocorrelation
# ---------------------------------------------------------------------------

def _autocorrelation_fft(frame):
    """
    Fast (FFT-based) unnormalised autocorrelation of *frame*.

    R[τ] = Σ_n frame[n] · frame[n + τ]

    Computed via  IFFT( |FFT(frame)|² )  which is O(N log N).
    """
    n = len(frame)
    fft_size = 1 << int(np.ceil(np.log2(2 * n)))   # next power of 2
    X = np.fft.rfft(frame, n=fft_size)
    acf = np.fft.irfft(X * np.conj(X))
    return np.real(acf[:n])


def _find_acf_peaks(acf, min_lag, max_lag):
    """
    Return list of (lag, normalised_height) for every local maximum of the
    autocorrelation *acf* in the range [min_lag, max_lag].
    """
    r0 = acf[0]
    if r0 < 1e-12:
        return []

    peaks = []
    for i in range(max(1, min_lag), min(max_lag + 1, len(acf) - 1)):
        if acf[i] > acf[i - 1] and acf[i] > acf[i + 1]:
            # Parabolic interpolation for sub-sample accuracy
            alpha = acf[i - 1]
            beta  = acf[i]
            gamma = acf[i + 1]
            denom = alpha - 2 * beta + gamma
            if abs(denom) > 1e-12:
                shift = 0.5 * (alpha - gamma) / denom
            else:
                shift = 0.0
            refined_lag = i + shift
            refined_val = beta - 0.25 * (alpha - gamma) * shift

            # Normalise by the geometric mean of energies at lag 0 and at lag
            # (this compensates for energy decrease with lag)
            energy_at_lag = np.sum(acf[0] if i == 0 else acf[0])  # R(0) proxy
            norm = refined_val / r0
            if norm > 0:
                peaks.append((refined_lag, norm))

    # Sort descending by normalised height
    peaks.sort(key=lambda p: p[1], reverse=True)
    return peaks


def estimate_pitch(frame, sr, f0_min=80, f0_max=400):
    """
    Autocorrelation-based F0 estimator.

    Returns
    -------
    f0 : float          Estimated fundamental frequency (Hz), 0 if unvoiced.
    confidence : float  Normalised autocorrelation peak  (0 … 1).
    peaks : list        All candidate (lag, confidence) pairs found.
    """
    min_lag = int(sr / f0_max)
    max_lag = min(int(sr / f0_min), len(frame) - 1)
    if max_lag <= min_lag:
        return 0.0, 0.0, []

    windowed = frame * np.hamming(len(frame))
    acf = _autocorrelation_fft(windowed)
    peaks = _find_acf_peaks(acf, min_lag, max_lag)

    if not peaks:
        return 0.0, 0.0, []

    best_lag, best_conf = peaks[0]
    f0 = sr / best_lag if best_lag > 0 else 0.0
    return f0, best_conf, peaks


# ---------------------------------------------------------------------------
# Multi-pitch (overlap) detection via harmonic sieve
# ---------------------------------------------------------------------------

def _is_harmonic(lag_a, lag_b, tol=0.08):
    """
    Check whether *lag_b* is a harmonic or sub-harmonic of *lag_a*.

    Two lags are harmonically related if their ratio is close to a small
    integer  n ∈ {1, 2, 3, 4}  (within *tol* relative error).
    """
    if lag_a <= 0 or lag_b <= 0:
        return True
    ratio = lag_b / lag_a if lag_b > lag_a else lag_a / lag_b
    for n in range(1, 5):
        if abs(ratio - n) < tol * n:
            return True
    return False


def detect_multi_pitch(peaks, min_secondary_conf=0.25):
    """
    Given autocorrelation *peaks* (sorted by confidence), return True if an
    independent secondary pitch is present (i.e. not a harmonic of the
    strongest pitch).
    """
    if len(peaks) < 2:
        return False
    primary_lag = peaks[0][0]
    for lag, conf in peaks[1:]:
        if conf < min_secondary_conf:
            break                       # remaining peaks are weaker
        if not _is_harmonic(primary_lag, lag):
            return True                 # independent pitch found!
    return False


# ---------------------------------------------------------------------------
# Spectral flatness (Wiener entropy)
# ---------------------------------------------------------------------------

def compute_spectral_flatness(frame, sr):
    """
    Spectral Flatness  ∈  [0, 1].

    Low  (~0.0)  →  tonal / harmonic (single voice)
    High (~1.0)  →  noise-like / spectrally complex (overlap or noise)
    """
    windowed = frame * np.hamming(len(frame))
    mag = np.abs(np.fft.rfft(windowed))
    power = mag[1:] ** 2                    # skip DC
    power = np.maximum(power, 1e-20)

    log_mean   = np.mean(np.log(power))
    geo_mean   = np.exp(log_mean)
    arith_mean = np.mean(power)
    if arith_mean < 1e-20:
        return 0.0
    return geo_mean / arith_mean


# ---------------------------------------------------------------------------
# YIN pitch estimator (more robust than basic autocorrelation)
# ---------------------------------------------------------------------------

def _yin_difference(frame):
    """FFT-based YIN difference function d(τ) — O(N log N)."""
    N = len(frame)
    W = N // 2
    fft_size = 1 << int(np.ceil(np.log2(2 * N)))
    X = np.fft.rfft(frame, n=fft_size)
    acf = np.fft.irfft(X * np.conj(X))[:W]

    x_sq = frame ** 2
    cum = np.concatenate([[0.0], np.cumsum(x_sq)])

    d = np.zeros(W)
    taus = np.arange(1, W)
    d[1:] = cum[N - taus] + cum[N] - cum[taus] - 2 * acf[1:W]
    return d


def _yin_cmnd(d):
    """Cumulative mean normalised difference d'(τ)."""
    cmnd = np.ones_like(d)
    if len(d) < 2:
        return cmnd
    running = np.cumsum(d[1:])
    taus = np.arange(1, len(d))
    cmnd[1:] = np.where(running > 0, d[1:] * taus / running, 1.0)
    return cmnd


def estimate_pitch_yin(frame, sr, f0_min=80, f0_max=400, threshold=0.15):
    """
    YIN F0 estimator — substantially more robust than basic autocorrelation.

    The cumulative mean normalised difference function suppresses
    sub-harmonic errors, making both the F0 estimate *and* the confidence
    more reliable.

    Returns
    -------
    f0         : float   Fundamental frequency (Hz), 0 if unvoiced.
    confidence : float   1 − d'(τ_best),  range ≈ 0 … 1.
    candidates : list    All (lag, confidence) valley candidates.
    """
    windowed = frame * np.hamming(len(frame))
    min_lag = max(2, int(sr / f0_max))
    max_lag = min(int(sr / f0_min), len(windowed) // 2 - 1)
    if max_lag <= min_lag:
        return 0.0, 0.0, []

    d = _yin_difference(windowed)
    cmnd = _yin_cmnd(d)

    # Step 3: absolute threshold — first valley below *threshold*
    best_tau = 0
    for tau in range(min_lag, max_lag):
        if cmnd[tau] < threshold:
            if tau + 1 >= len(cmnd) or cmnd[tau] <= cmnd[tau + 1]:
                best_tau = tau
                break

    # Fallback: global minimum in range
    if best_tau == 0:
        search = cmnd[min_lag:max_lag + 1]
        if len(search) > 0:
            best_tau = min_lag + int(np.argmin(search))
            if cmnd[best_tau] > 0.5:
                return 0.0, 0.0, []

    # Parabolic interpolation for sub-sample accuracy
    if 1 < best_tau < len(cmnd) - 1:
        a, b, c = cmnd[best_tau - 1], cmnd[best_tau], cmnd[best_tau + 1]
        denom = a - 2 * b + c
        shift = 0.5 * (a - c) / denom if abs(denom) > 1e-12 else 0.0
        refined = best_tau + shift
    else:
        refined = float(best_tau)

    f0 = sr / refined if refined > 0 else 0.0
    confidence = max(0.0, 1.0 - cmnd[best_tau])

    # Collect all candidate valleys (for multi-pitch analysis)
    candidates = []
    for tau in range(min_lag, min(max_lag + 1, len(cmnd) - 1)):
        if cmnd[tau] < 0.5:
            is_valley = (cmnd[tau] <= cmnd[max(0, tau - 1)] and
                         cmnd[tau] <= cmnd[min(len(cmnd) - 1, tau + 1)])
            if is_valley:
                candidates.append((float(tau), max(0.0, 1.0 - cmnd[tau])))
    candidates.sort(key=lambda x: x[1], reverse=True)

    return f0, confidence, candidates


# ---------------------------------------------------------------------------
# Harmonic-to-Noise Ratio (HNR)
# ---------------------------------------------------------------------------

def compute_hnr(frame, sr, f0):
    """
    HNR in dB via autocorrelation at the pitch period.

    High HNR (≥15 dB) → clean single voice (strong harmonics)
    Low  HNR (<5 dB)  → noisy, breathy, or *multiple overlapping voices*

    The key insight: when two independent voices overlap, the periodicity
    of each is disrupted by the other, reducing the autocorrelation peak
    at both pitch periods and thus dropping HNR significantly.
    """
    if f0 <= 0:
        return 0.0

    period = int(round(sr / f0))
    if period < 2 or period >= len(frame) // 2:
        return 0.0

    windowed = frame * np.hamming(len(frame))
    acf = _autocorrelation_fft(windowed)

    r0 = acf[0]
    if r0 < 1e-12:
        return 0.0

    r_tau = acf[period] if period < len(acf) else 0.0

    if r_tau >= r0:
        return 30.0
    if r_tau <= 0:
        return -10.0

    noise_power = r0 - r_tau
    if noise_power < 1e-12:
        return 30.0

    return float(10.0 * np.log10(r_tau / noise_power))


# ---------------------------------------------------------------------------
# Spectral centroid & bandwidth
# ---------------------------------------------------------------------------

def compute_spectral_centroid_bandwidth(frame, sr):
    """
    Returns (centroid_hz, bandwidth_hz).

    Overlapping speakers produce a broader spectral envelope because
    each voice contributes energy at different formant frequencies,
    leading to higher bandwidth.
    """
    windowed = frame * np.hamming(len(frame))
    mag = np.abs(np.fft.rfft(windowed))
    power = mag[1:] ** 2          # skip DC
    freqs = np.fft.rfftfreq(len(frame), d=1.0 / sr)[1:]

    total = np.sum(power)
    if total < 1e-20:
        return 0.0, 0.0

    centroid = float(np.sum(freqs * power) / total)
    bandwidth = float(np.sqrt(np.sum((freqs - centroid) ** 2 * power) / total))
    return centroid, bandwidth


# ---------------------------------------------------------------------------
# Sub-band energy entropy
# ---------------------------------------------------------------------------

def compute_sub_band_entropy(frame, sr, n_bands=6):
    """
    Normalised spectral entropy across *n_bands* equal sub-bands.

    High entropy (→ 1.0) : energy spread uniformly (noise or overlap)
    Low  entropy (→ 0.0) : energy concentrated in few bands (single voice)
    """
    windowed = frame * np.hamming(len(frame))
    power = np.abs(np.fft.rfft(windowed)) ** 2
    power = power[1:]

    band_size = len(power) // n_bands
    if band_size < 1:
        return 0.0

    band_e = np.array([np.sum(power[i * band_size:(i + 1) * band_size])
                        for i in range(n_bands)])
    total = np.sum(band_e)
    if total < 1e-20:
        return 0.0

    ratios = band_e / total
    ratios = ratios[ratios > 1e-10]
    entropy = -np.sum(ratios * np.log(ratios)) / np.log(n_bands)
    return float(np.clip(entropy, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Delta features (regression-based temporal derivative)
# ---------------------------------------------------------------------------

def compute_delta(arr, N=2):
    """
    First-order delta (regression) of a 1-D feature array.

    Δ[n] = Σ_{k=1}^{N} k · (arr[n+k] − arr[n−k])  /  2 · Σ k²
    """
    padded = np.pad(arr, N, mode='edge')
    denom = 2 * sum(k ** 2 for k in range(1, N + 1))
    delta = np.zeros_like(arr, dtype=np.float64)
    for k in range(1, N + 1):
        delta += k * (padded[N + k:N + k + len(arr)]
                      - padded[N - k:N - k + len(arr)])
    return delta / denom


# ---------------------------------------------------------------------------
# Spectral flux (frame-to-frame spectral change)
# ---------------------------------------------------------------------------

def compute_spectral_flux(frames):
    """
    Normalised spectral flux between consecutive frames.

    High flux → rapid spectral change (common in overlap, where two
    independent voices create a combined spectrum that varies quickly).
    """
    n_frames = frames.shape[0]
    flux = np.zeros(n_frames)
    prev_mag = None
    for i in range(n_frames):
        windowed = frames[i] * np.hamming(frames.shape[1])
        mag = np.abs(np.fft.rfft(windowed))
        total = np.sum(mag) + 1e-20
        mag_norm = mag / total          # L1-normalised
        if prev_mag is not None:
            flux[i] = np.sqrt(np.sum((mag_norm - prev_mag) ** 2))
        prev_mag = mag_norm
    return flux


# ---------------------------------------------------------------------------
# Running F0 variance (pitch stability)
# ---------------------------------------------------------------------------

def compute_pitch_variance(f0_track, half_win=5):
    """
    Running standard deviation of F0 over ±half_win frames.

    High variance → unstable pitch → likely overlap (pitch tracker jumps
    between two speakers' F0, or the combined signal confuses the estimator).
    Only voiced frames (f0 > 0) are considered in each window.
    """
    n = len(f0_track)
    var_track = np.zeros(n)
    for i in range(n):
        lo = max(0, i - half_win)
        hi = min(n, i + half_win + 1)
        seg = f0_track[lo:hi]
        voiced = seg[seg > 0]
        if len(voiced) >= 3:
            var_track[i] = np.std(voiced)
    return var_track


# ===========================================================================
#  Main classifier
# ===========================================================================

class MultivoiceVAD:
    """
    Enhanced frame-level multi-voice activity detector (v2).

    Output per frame:
        0 — silence / no speech
        1 — single speaker
        2 — overlapping speech (≥2 speakers)

    Improvements over v1:
        • YIN pitch tracking (more robust F0 + confidence)
        • Harmonic-to-Noise Ratio (HNR) — key overlap indicator
        • Spectral bandwidth — overlap broadens spectrum
        • Spectral flux — rapid spectral change in overlap
        • Pitch variance — F0 instability when tracker confused by overlap
        • Sub-band entropy — overlap distributes energy uniformly
        • Delta energy — detects second speaker entering
        • Multi-feature overlap SCORING with adaptive thresholds
          (instead of hard if/else on a single feature)
        • Max-pool temporal context smoothing
        • Minimum-segment-duration post-processing

    Best-known parameters (2026-03-12, test split 100 files):
    ─────────────────────────────────────────────────────────
      frame_ms               = 30
      hop_ms                 = 10
      energy_threshold_db    = -40
      zcr_speech_range       = (0.02, 0.50)
      pitch_conf_thresh      = 0.25
      yin_threshold           = 0.15
      secondary_pitch_conf   = 0.20
      spectral_flatness_overlap = 0.30
      overlap_threshold      = 0.38
      context_frames         = 3      (max-pool ±3 frames)
      median_filter_size     = 7
      f0_min                 = 80
      f0_max                 = 400
    ─────────────────────────────────────────────────────────
    Results with these defaults:
      3-class accuracy       = 63.4 %
      Macro F1               = 63.7 %
      Overlap F1             = 43.7 %  (prec 47.8 %, rec 40.3 %)
      Single  F1             = 64.8 %
      Silence F1             = 82.5 %
      Binary Speech/Sil F1   = 94.6 %
    ─────────────────────────────────────────────────────────
    """

    def __init__(self, sr=48000,
                 frame_ms=30, hop_ms=10,
                 energy_threshold_db=-40,
                 zcr_speech_range=(0.02, 0.50),
                 pitch_conf_thresh=0.25,
                 yin_threshold=0.15,
                 secondary_pitch_conf=0.20,
                 spectral_flatness_overlap=0.30,
                 overlap_threshold=0.38,
                 context_frames=3,
                 median_filter_size=7,
                 f0_min=80, f0_max=400):
        self.sr = sr
        self.frame_len = int(sr * frame_ms / 1000)
        self.hop_len   = int(sr * hop_ms / 1000)
        self.energy_threshold_db = energy_threshold_db
        self.zcr_lo, self.zcr_hi = zcr_speech_range
        self.pitch_conf_thresh = pitch_conf_thresh
        self.yin_threshold = yin_threshold
        self.secondary_pitch_conf = secondary_pitch_conf
        self.sf_overlap = spectral_flatness_overlap
        self.overlap_thresh = overlap_threshold
        self.ctx_k = context_frames
        self.med_k = median_filter_size if median_filter_size > 1 else 1
        self.f0_min = f0_min
        self.f0_max = f0_max

    # -----------------------------------------------------------------------
    def process(self, signal, verbose=True):
        """
        Enhanced multi-feature analysis pipeline.

        Returns
        -------
        labels   : np.ndarray[int32]   shape (n_frames,)   values in {0,1,2}
        features : dict                per-frame feature arrays
        """
        # ── Phase 1: Pre-processing ───────────────────────────────────────
        sig_hp = highpass_filter(signal, self.sr, cutoff=80)
        sig_pe = pre_emphasis(sig_hp)

        frames_pe   = frame_signal(sig_pe, self.frame_len, self.hop_len)
        frames_orig = frame_signal(sig_hp, self.frame_len, self.hop_len)
        n_frames    = frames_pe.shape[0]

        if verbose:
            print(f"  Analysing {n_frames} frames  "
                  f"(frame={self.frame_len/self.sr*1000:.0f} ms, "
                  f"hop={self.hop_len/self.sr*1000:.0f} ms) …")

        # ── Phase 2: Vectorised features ──────────────────────────────────
        energy    = compute_energy(frames_orig)
        energy_db = 10.0 * np.log10(np.maximum(energy, 1e-20))
        zcr       = compute_zcr(frames_pe)
        delta_e   = compute_delta(energy_db, N=3)
        sp_flux   = compute_spectral_flux(frames_orig)

        # Adaptive silence threshold
        noise_floor = np.percentile(energy_db, 15)
        adapt_thresh = max(self.energy_threshold_db, noise_floor + 8)

        speech_mask = energy_db >= adapt_thresh

        # ── Phase 3: Per-frame features (speech frames only) ──────────────
        f0_track    = np.zeros(n_frames)
        conf_track  = np.zeros(n_frames)
        hnr_track   = np.zeros(n_frames)
        bw_track    = np.zeros(n_frames)
        cent_track  = np.zeros(n_frames)
        sf_track    = np.zeros(n_frames)
        sbe_track   = np.zeros(n_frames)
        mp_flags    = np.zeros(n_frames, dtype=bool)

        for i in range(n_frames):
            if not speech_mask[i]:
                continue

            frame = frames_orig[i]

            # Pitch (YIN)
            f0, conf, candidates = estimate_pitch_yin(
                frame, self.sr, self.f0_min, self.f0_max, self.yin_threshold)
            f0_track[i] = f0
            conf_track[i] = conf

            # Multi-pitch detection (harmonic sieve on YIN candidates)
            if conf > self.pitch_conf_thresh and len(candidates) >= 2:
                mp_flags[i] = detect_multi_pitch(
                    candidates, min_secondary_conf=self.secondary_pitch_conf)

            # Harmonic-to-Noise Ratio
            hnr_track[i] = compute_hnr(frame, self.sr, f0)

            # Spectral features
            cent, bw = compute_spectral_centroid_bandwidth(frame, self.sr)
            cent_track[i] = cent
            bw_track[i] = bw
            sf_track[i] = compute_spectral_flatness(frame, self.sr)
            sbe_track[i] = compute_sub_band_entropy(frame, self.sr, n_bands=6)

        # ── Phase 3b: Temporal features (need full f0_track) ────────────
        f0_var = compute_pitch_variance(f0_track, half_win=5)

        # ── Phase 4: Adaptive thresholds from speech distributions ────────
        #   Use percentiles — HNR: bottom 25%; others: top 25%.
        sp_idx = np.where(speech_mask)[0]
        if len(sp_idx) > 20:
            hnr_sp  = hnr_track[sp_idx]
            hnr_low = np.percentile(hnr_sp, 25)

            bw_sp   = bw_track[sp_idx]
            bw_p75  = np.percentile(bw_sp, 75)

            de_sp   = delta_e[sp_idx]
            de_p75  = np.percentile(de_sp, 75)

            sbe_sp  = sbe_track[sp_idx]
            sbe_p75 = np.percentile(sbe_sp, 75)

            flux_sp = sp_flux[sp_idx]
            flux_p75 = np.percentile(flux_sp, 75)

            f0v_sp  = f0_var[sp_idx]
            f0v_p75 = np.percentile(f0v_sp[f0v_sp > 0], 75) if np.any(f0v_sp > 0) else 30.0
        else:
            hnr_low   = 3.0
            bw_p75    = 2000.0
            de_p75    = 1.0
            sbe_p75   = 0.85
            flux_p75  = 0.05
            f0v_p75   = 30.0

        # ── Phase 5: Overlap scoring ──────────────────────────────────────
        #
        #   Weights are calibrated so that:
        #   • Multi-pitch alone (0.40) triggers at threshold 0.38
        #   • HNR drop + 1 supporting indicator triggers
        #   • 3+ supporting indicators trigger without HNR/MP
        #
        #   Strong indicators:
        #     Multi-pitch (0.40) — independent secondary pitch
        #     HNR drop   (0.25) — overlap destroys harmonicity
        #   Medium indicators:
        #     Pitch var  (0.12) — unstable F0 (tracker confused)
        #     Sp. flux   (0.12) — rapid spectral change
        #   Supporting indicators:
        #     Bandwidth  (0.08) — overlap broadens spectrum
        #     Flatness   (0.06) — spectral noise
        #     ΔEnergy    (0.06) — energy jump
        #     SubBandEnt (0.06) — uniform energy distribution
        #
        overlap_scores = np.zeros(n_frames)

        for i in range(n_frames):
            if not speech_mask[i]:
                continue

            score = 0.0

            # Strong indicators
            if mp_flags[i]:
                score += 0.40
            if hnr_track[i] < hnr_low:
                score += 0.25

            # Medium indicators (temporal dynamics)
            if f0_var[i] > f0v_p75:
                score += 0.12
            if sp_flux[i] > flux_p75:
                score += 0.12

            # Supporting indicators
            if bw_track[i] > bw_p75:
                score += 0.08
            if sf_track[i] > self.sf_overlap:
                score += 0.06
            if delta_e[i] > de_p75:
                score += 0.06
            if sbe_track[i] > sbe_p75:
                score += 0.06

            overlap_scores[i] = score

        # ── Phase 6: Temporal context smoothing ───────────────────────────
        #   Use maximum_filter1d so that a high overlap score propagates to
        #   neighbouring frames (hangover effect), preventing isolated
        #   overlap frames from being diluted away by averaging.
        if self.ctx_k > 0:
            ctx_size = 2 * self.ctx_k + 1
            # Step A: max-pool expands overlap regions
            ovl_max = maximum_filter1d(
                overlap_scores, size=ctx_size, mode='nearest')
            # Step B: light averaging to smooth the edges
            ovl_smooth = uniform_filter1d(ovl_max, size=5, mode='nearest')
        else:
            ovl_smooth = overlap_scores.copy()

        # ── Phase 7: Classification ───────────────────────────────────────
        labels = np.zeros(n_frames, dtype=np.int32)

        for i in range(n_frames):
            if not speech_mask[i]:
                # Soft speech recovery: just below threshold + ZCR hint
                if (energy_db[i] > adapt_thresh - 3
                        and self.zcr_lo <= zcr[i] <= self.zcr_hi):
                    labels[i] = 1
                else:
                    labels[i] = 0
                continue

            # Overlap decision: use smoothed score
            if ovl_smooth[i] >= self.overlap_thresh:
                labels[i] = 2
            elif conf_track[i] < self.pitch_conf_thresh:
                # Weak pitch — unvoiced speech or noise
                if self.zcr_lo <= zcr[i] <= self.zcr_hi:
                    labels[i] = 1
                elif energy_db[i] > adapt_thresh + 10:
                    labels[i] = 1
                else:
                    labels[i] = 0
            else:
                labels[i] = 1

        # ── Phase 8: Post-processing ──────────────────────────────────────
        # 8a. Median filter
        if self.med_k > 1:
            k = self.med_k if self.med_k % 2 == 1 else self.med_k + 1
            labels = medfilt(labels.astype(np.float64),
                             kernel_size=k).astype(np.int32)

        # 8b. Minimum segment duration — remove very short overlap bursts
        #     (< 80 ms are likely false positives)
        min_ovl_frames = max(1, int(0.08 * self.sr / self.hop_len))
        in_ovl = False
        seg_start = 0
        for i in range(n_frames + 1):
            if i < n_frames and labels[i] == 2:
                if not in_ovl:
                    seg_start = i
                    in_ovl = True
            else:
                if in_ovl:
                    if (i - seg_start) < min_ovl_frames:
                        labels[seg_start:i] = 1   # downgrade to single
                    in_ovl = False

        # 8c. Similarly, remove very short speech segments (< 100 ms)
        min_sp_frames = max(1, int(0.10 * self.sr / self.hop_len))
        in_sp = False
        seg_start = 0
        for i in range(n_frames + 1):
            if i < n_frames and labels[i] > 0:
                if not in_sp:
                    seg_start = i
                    in_sp = True
            else:
                if in_sp:
                    if (i - seg_start) < min_sp_frames:
                        labels[seg_start:i] = 0   # downgrade to silence
                    in_sp = False

        features = {
            'energy_db':          energy_db,
            'delta_energy':       delta_e,
            'zcr':                zcr,
            'f0':                 f0_track,
            'pitch_confidence':   conf_track,
            'pitch_variance':     f0_var,
            'hnr':                hnr_track,
            'spectral_centroid':  cent_track,
            'spectral_bandwidth': bw_track,
            'spectral_flatness':  sf_track,
            'spectral_flux':      sp_flux,
            'sub_band_entropy':   sbe_track,
            'multi_pitch':        mp_flags,
            'overlap_score':      overlap_scores,
            'overlap_score_smooth': ovl_smooth,
            'adaptive_threshold': adapt_thresh,
        }
        return labels, features

    # -----------------------------------------------------------------------
    def frame_times(self, n_frames):
        """Return (starts, ends) arrays in seconds."""
        starts = np.arange(n_frames) * self.hop_len / self.sr
        ends   = starts + self.frame_len / self.sr
        return starts, ends


# ===========================================================================
#  Visualisation
# ===========================================================================

def plot_results(signal, sr, labels, features, vad, output_path=None):
    """Eight-panel plot: waveform, energy+Δ, ZCR, F0, HNR, bandwidth+flatness,
    overlap score, labels."""
    if not HAS_MPL:
        print("  ⚠ matplotlib not available — skipping plot.")
        return

    n_frames = len(labels)
    starts, _ = vad.frame_times(n_frames)

    fig, axes = plt.subplots(8, 1, figsize=(18, 20), sharex=True)
    fig.suptitle('Multi-Voice Activity Detection (v2 — enhanced)',
                 fontsize=14, fontweight='bold')

    # Colour backgrounds by label
    for ax in axes:
        for i in range(n_frames - 1):
            if labels[i] == 1:
                ax.axvspan(starts[i], starts[i + 1], alpha=0.12, color='green')
            elif labels[i] == 2:
                ax.axvspan(starts[i], starts[i + 1], alpha=0.22, color='red')

    # 1 — Waveform
    t_sig = np.arange(len(signal)) / sr
    axes[0].plot(t_sig, signal, lw=0.25, color='steelblue')
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title('Waveform')

    # 2 — Energy + Delta Energy
    axes[1].plot(starts, features['energy_db'], lw=0.6, color='orange',
                 label='Energy (dB)')
    axes[1].axhline(features['adaptive_threshold'], color='red', ls='--', lw=1,
                     label=f"threshold = {features['adaptive_threshold']:.1f} dB")
    if 'delta_energy' in features:
        ax1b = axes[1].twinx()
        ax1b.plot(starts, features['delta_energy'], lw=0.4, color='blue',
                  alpha=0.5, label='ΔEnergy')
        ax1b.set_ylabel('ΔE', color='blue')
        ax1b.legend(loc='upper left', fontsize=7)
    axes[1].set_ylabel('Energy (dB)')
    axes[1].set_title('Short-Time Energy + Delta')
    axes[1].legend(loc='upper right', fontsize=8)

    # 3 — ZCR
    axes[2].plot(starts, features['zcr'], lw=0.6, color='purple')
    axes[2].set_ylabel('ZCR')
    axes[2].set_title('Zero-Crossing Rate (ZCR)')

    # 4 — Pitch contour (YIN)
    f0 = features['f0'].copy()
    f0[f0 == 0] = np.nan
    axes[3].scatter(starts, f0, s=1.2, color='teal')
    axes[3].set_ylabel('F0 (Hz)')
    axes[3].set_title('Pitch Track (YIN)')
    axes[3].set_ylim(50, 500)

    # 5 — HNR
    if 'hnr' in features:
        hnr = features['hnr'].copy()
        hnr[hnr == 0] = np.nan
        axes[4].plot(starts, hnr, lw=0.6, color='darkorange')
        axes[4].set_ylabel('HNR (dB)')
        axes[4].set_title('Harmonic-to-Noise Ratio (HNR)')
    else:
        axes[4].plot(starts, features['pitch_confidence'], lw=0.6,
                     color='darkgreen')
        axes[4].set_title('Pitch Confidence')

    # 6 — Spectral bandwidth + flatness + sub-band entropy
    if 'spectral_bandwidth' in features:
        ax6 = axes[5]
        ax6.plot(starts, features['spectral_bandwidth'], lw=0.5,
                 color='royalblue', label='Bandwidth (Hz)')
        ax6.set_ylabel('Bandwidth (Hz)', color='royalblue')
        ax6b = ax6.twinx()
        ax6b.plot(starts, features['spectral_flatness'], lw=0.5,
                  color='crimson', alpha=0.7, label='Sp. Flatness')
        if 'sub_band_entropy' in features:
            ax6b.plot(starts, features['sub_band_entropy'], lw=0.5,
                      color='goldenrod', alpha=0.7, label='SubBand Ent.')
        ax6b.set_ylabel('Flatness / Entropy')
        ax6b.legend(loc='upper left', fontsize=7)
        ax6.legend(loc='upper right', fontsize=7)
        ax6.set_title('Spectral Bandwidth, Flatness & Sub-Band Entropy')
    else:
        axes[5].plot(starts, features['spectral_flatness'], lw=0.6,
                     color='crimson')
        axes[5].set_title('Spectral Flatness')

    # 7 — Overlap score (raw + smoothed)
    if 'overlap_score' in features:
        axes[6].plot(starts, features['overlap_score'], lw=0.4,
                     color='salmon', alpha=0.5, label='Raw score')
        if 'overlap_score_smooth' in features:
            axes[6].plot(starts, features['overlap_score_smooth'], lw=1.0,
                         color='red', label='Smoothed score')
        axes[6].axhline(vad.overlap_thresh, color='darkred', ls='--', lw=1,
                         label=f'threshold = {vad.overlap_thresh:.2f}')
        axes[6].set_ylabel('Overlap Score')
        axes[6].set_title('Multi-Feature Overlap Score')
        axes[6].legend(loc='upper right', fontsize=8)
    else:
        axes[6].set_visible(False)

    # 8 — Labels
    cmap = {0: 'gray', 1: 'green', 2: 'red'}
    nmap = {0: 'Silence', 1: 'Single speaker', 2: 'Overlap'}
    for v in (0, 1, 2):
        m = labels == v
        axes[7].scatter(starts[m], labels[m], s=3, color=cmap[v], label=nmap[v])
    axes[7].set_ylabel('Label')
    axes[7].set_title('Classification')
    axes[7].set_yticks([0, 1, 2])
    axes[7].set_yticklabels(['Silence', 'Single', 'Overlap'])
    axes[7].legend(loc='upper right', fontsize=8)
    axes[7].set_xlabel('Time (s)')

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Plot saved → {output_path}")
    else:
        plt.show()
    plt.close(fig)


# ===========================================================================
#  CSV output
# ===========================================================================

def save_csv(labels, vad, output_path):
    n_frames = len(labels)
    starts, ends = vad.frame_times(n_frames)
    with open(output_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['frame_idx', 'time_start', 'time_end', 'label'])
        for i in range(n_frames):
            w.writerow([i, f'{starts[i]:.6f}', f'{ends[i]:.6f}', int(labels[i])])
    print(f"  CSV saved  → {output_path}")


# ===========================================================================
#  Summary
# ===========================================================================

def print_summary(labels, vad):
    n = len(labels)
    _, ends = vad.frame_times(n)
    total = ends[-1] if n else 0
    hop_s = vad.hop_len / vad.sr

    counts = {0: 0, 1: 0, 2: 0}
    for l in labels:
        counts[int(l)] += 1

    print(f"\n{'=' * 55}")
    print(f"  Multi-Voice VAD Summary")
    print(f"{'=' * 55}")
    print(f"  Total duration : {total:.2f} s   ({n} frames)")
    print(f"  Frame / Hop    : "
          f"{vad.frame_len / vad.sr * 1000:.0f} ms / "
          f"{vad.hop_len / vad.sr * 1000:.0f} ms")
    print(f"  {'─' * 40}")
    for lab, name in [(0, 'Silence / noise'), (1, 'Single speaker'), (2, 'Overlap (≥2)')]:
        dur = counts[lab] * hop_s
        pct = counts[lab] / n * 100 if n else 0
        print(f"  {name:20s} : {dur:7.2f} s  ({pct:5.1f} %)")
    print(f"{'=' * 55}\n")


# ===========================================================================
#  Evaluation: GT alignment + metrics
# ===========================================================================

def gt_to_frame_labels(gt_samples, frame_len, hop_len):
    """
    Convert sample-level GT  →  frame-level GT via majority vote.

    Each frame spans [i*hop .. i*hop + frame_len) samples; the frame label
    is the most frequent GT value within that window.
    """
    n_frames = 1 + (len(gt_samples) - frame_len) // hop_len
    labels = np.zeros(n_frames, dtype=np.int32)
    for i in range(n_frames):
        start = i * hop_len
        end = start + frame_len
        chunk = gt_samples[start:end]
        counts = np.bincount(chunk.astype(np.int64), minlength=3)
        labels[i] = int(np.argmax(counts))
    return labels


def compute_confusion_matrix(pred, gt, n_classes=3):
    """Row = GT, Col = Pred.  Shape: (n_classes, n_classes)."""
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for g, p in zip(gt, pred):
        cm[int(g), int(p)] += 1
    return cm


def metrics_from_cm(cm):
    """
    From a confusion matrix compute per-class and aggregate metrics.

    Returns dict with:
        per_class: [{class, support, TP, FP, FN, precision, recall, f1}, ...]
        accuracy, macro_precision, macro_recall, macro_f1
        weighted_precision, weighted_recall, weighted_f1
    """
    n = cm.shape[0]
    names = {0: 'silence', 1: 'single', 2: 'overlap'}
    total = int(cm.sum())
    correct = int(np.trace(cm))

    per_class = []
    for c in range(n):
        tp = int(cm[c, c])
        fp = int(cm[:, c].sum() - tp)
        fn = int(cm[c, :].sum() - tp)
        support = int(cm[c, :].sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        per_class.append(dict(
            cls=c, name=names.get(c, str(c)),
            support=support, TP=tp, FP=fp, FN=fn,
            precision=prec, recall=rec, f1=f1,
        ))

    supports = np.array([p['support'] for p in per_class], dtype=np.float64)
    w = supports / supports.sum() if supports.sum() > 0 else np.ones(n) / n

    macro_prec = np.mean([p['precision'] for p in per_class])
    macro_rec  = np.mean([p['recall']    for p in per_class])
    macro_f1   = np.mean([p['f1']        for p in per_class])

    weighted_prec = np.sum([p['precision'] * wi for p, wi in zip(per_class, w)])
    weighted_rec  = np.sum([p['recall']    * wi for p, wi in zip(per_class, w)])
    weighted_f1   = np.sum([p['f1']        * wi for p, wi in zip(per_class, w)])

    return dict(
        per_class=per_class,
        accuracy=correct / total if total > 0 else 0.0,
        total_frames=total, correct_frames=correct,
        macro_precision=macro_prec, macro_recall=macro_rec, macro_f1=macro_f1,
        weighted_precision=weighted_prec, weighted_recall=weighted_rec,
        weighted_f1=weighted_f1,
    )


def binary_metrics(pred, gt, positive_labels):
    """
    Collapse classes into binary (positive vs rest) and compute metrics.
    *positive_labels*: set of labels to treat as positive.
    """
    p_bin = np.isin(pred, list(positive_labels)).astype(np.int32)
    g_bin = np.isin(gt, list(positive_labels)).astype(np.int32)
    tp = int(np.sum((p_bin == 1) & (g_bin == 1)))
    fp = int(np.sum((p_bin == 1) & (g_bin == 0)))
    fn = int(np.sum((p_bin == 0) & (g_bin == 1)))
    tn = int(np.sum((p_bin == 0) & (g_bin == 0)))
    total = tp + fp + fn + tn
    acc  = (tp + tn) / total if total > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return dict(accuracy=acc, precision=prec, recall=rec, f1=f1,
                TP=tp, FP=fp, FN=fn, TN=tn)


# ===========================================================================
#  Evaluation runner
# ===========================================================================

def evaluate_dataset(eval_dir, vad_factory_fn, verbose=True):
    """
    Run VAD on every WAV in *eval_dir*, compare with GT (_gt.npy), return
    aggregated metrics.

    Parameters
    ----------
    eval_dir : Path-like
    vad_factory_fn : callable(sr) → MultivoiceVAD
        Returns a configured VAD instance for a given sample rate.
    verbose : bool

    Returns
    -------
    results : dict
    """
    eval_dir = Path(eval_dir)
    wav_files = sorted(eval_dir.glob('*.wav'))
    if not wav_files:
        print(f"  ✗ No WAV files found in {eval_dir}")
        return None

    # Global confusion matrix
    cm_3class = np.zeros((3, 3), dtype=np.int64)
    all_pred, all_gt = [], []

    per_file_results = []
    total_time = 0.0

    n_files = len(wav_files)
    print(f"\n  Evaluating {n_files} files in: {eval_dir}\n")

    for idx, wp in enumerate(wav_files):
        gt_path = wp.parent / (wp.stem + '_gt.npy')
        if not gt_path.exists():
            if verbose:
                print(f"    ⚠ Skipping {wp.name} — no GT file")
            continue

        # Load audio + GT
        signal, sr = sf.read(str(wp), dtype='float64')
        if signal.ndim > 1:
            signal = np.mean(signal, axis=1)
        gt_samples = np.load(str(gt_path)).astype(np.int32)

        # Truncate to same length
        n = min(len(signal), len(gt_samples))
        signal, gt_samples = signal[:n], gt_samples[:n]

        # Run VAD
        vad = vad_factory_fn(sr)
        t0 = time.time()
        pred_frames, _ = vad.process(signal, verbose=False)
        elapsed = time.time() - t0
        total_time += elapsed

        # Align GT to frame level
        gt_frames = gt_to_frame_labels(gt_samples, vad.frame_len, vad.hop_len)

        # Ensure same length
        m = min(len(pred_frames), len(gt_frames))
        pred_frames, gt_frames = pred_frames[:m], gt_frames[:m]

        # Per-file confusion matrix
        file_cm = compute_confusion_matrix(pred_frames, gt_frames)
        cm_3class += file_cm

        all_pred.append(pred_frames)
        all_gt.append(gt_frames)

        # Per-file accuracy
        file_acc = np.sum(pred_frames == gt_frames) / m if m > 0 else 0.0
        per_file_results.append(dict(
            file=wp.name, n_frames=m,
            accuracy=file_acc, time_s=elapsed,
        ))

        if verbose and ((idx + 1) % 10 == 0 or idx + 1 == n_files):
            print(f"    [{idx+1:>4d}/{n_files}]  "
                  f"running acc = {np.trace(cm_3class) / max(1, cm_3class.sum()) * 100:.1f} %")

    if not per_file_results:
        print("  ✗ No files evaluated")
        return None

    # ── Aggregate ─────────────────────────────────────────────────────────
    all_pred = np.concatenate(all_pred)
    all_gt   = np.concatenate(all_gt)

    metrics = metrics_from_cm(cm_3class)
    vad_binary = binary_metrics(all_pred, all_gt, positive_labels={1, 2})
    ovl_binary = binary_metrics(all_pred, all_gt, positive_labels={2})

    # Per-file accuracy stats
    accs = [r['accuracy'] for r in per_file_results]

    results = dict(
        n_files=len(per_file_results),
        total_frames=int(cm_3class.sum()),
        total_processing_time_s=total_time,
        rtf=total_time / (len(all_pred) * 10 / 1000) if len(all_pred) > 0 else 0,
        confusion_matrix=cm_3class.tolist(),
        three_class=metrics,
        binary_vad=vad_binary,
        binary_overlap=ovl_binary,
        per_file_accuracy_mean=float(np.mean(accs)),
        per_file_accuracy_std=float(np.std(accs)),
        per_file_accuracy_min=float(np.min(accs)),
        per_file_accuracy_max=float(np.max(accs)),
        per_file=per_file_results,
    )

    return results


def print_eval_report(results):
    """Pretty-print evaluation results."""
    m = results['three_class']
    cm = np.array(results['confusion_matrix'])

    print(f"\n{'═' * 65}")
    print(f"  MULTI-VOICE VAD — EVALUATION REPORT")
    print(f"{'═' * 65}")
    print(f"  Files evaluated  : {results['n_files']}")
    print(f"  Total frames     : {results['total_frames']:,}")
    print(f"  Processing time  : {results['total_processing_time_s']:.1f} s")
    print()

    # ── Confusion matrix ──────────────────────────────────────────────────
    names = ['Silence', 'Single', 'Overlap']
    print(f"  ┌─── Confusion Matrix (rows=GT, cols=Pred) ───┐")
    print(f"  {'':12s}  {'Silence':>9s}  {'Single':>9s}  {'Overlap':>9s}  {'Total':>9s}")
    print(f"  {'─' * 56}")
    for i in range(3):
        row_total = cm[i, :].sum()
        print(f"  {names[i]:12s}  {cm[i,0]:>9d}  {cm[i,1]:>9d}  {cm[i,2]:>9d}  {row_total:>9d}")
    print(f"  {'─' * 56}")
    col_totals = cm.sum(axis=0)
    print(f"  {'Total':12s}  {col_totals[0]:>9d}  {col_totals[1]:>9d}  {col_totals[2]:>9d}  {cm.sum():>9d}")
    print()

    # ── Per-class metrics ─────────────────────────────────────────────────
    print(f"  ┌─── Per-Class Metrics ───────────────────────────────┐")
    print(f"  {'Class':12s}  {'Support':>8s}  {'Prec':>7s}  {'Recall':>7s}  {'F1':>7s}")
    print(f"  {'─' * 50}")
    for pc in m['per_class']:
        print(f"  {pc['name']:12s}  {pc['support']:>8d}  "
              f"{pc['precision']:>7.1%}  {pc['recall']:>7.1%}  {pc['f1']:>7.1%}")
    print(f"  {'─' * 50}")
    print(f"  {'macro avg':12s}  {m['total_frames']:>8d}  "
          f"{m['macro_precision']:>7.1%}  {m['macro_recall']:>7.1%}  {m['macro_f1']:>7.1%}")
    print(f"  {'weighted avg':12s}  {m['total_frames']:>8d}  "
          f"{m['weighted_precision']:>7.1%}  {m['weighted_recall']:>7.1%}  {m['weighted_f1']:>7.1%}")
    print()

    # ── Overall accuracy ──────────────────────────────────────────────────
    print(f"  ┌─── Overall ───────────────────────────────────┐")
    print(f"  3-class accuracy       : {m['accuracy']:>7.1%}  "
          f"({m['correct_frames']:,} / {m['total_frames']:,})")
    print()

    # ── Binary: speech vs silence ─────────────────────────────────────────
    bv = results['binary_vad']
    print(f"  ┌─── Binary: Speech (1+2) vs Silence (0) ─────┐")
    print(f"  Accuracy  : {bv['accuracy']:>7.1%}")
    print(f"  Precision : {bv['precision']:>7.1%}    (of predicted speech, how much is real)")
    print(f"  Recall    : {bv['recall']:>7.1%}    (of real speech, how much is detected)")
    print(f"  F1        : {bv['f1']:>7.1%}")
    print()

    # ── Binary: overlap vs non-overlap ────────────────────────────────────
    bo = results['binary_overlap']
    print(f"  ┌─── Binary: Overlap (2) vs Non-overlap (0+1) ┐")
    print(f"  Accuracy  : {bo['accuracy']:>7.1%}")
    print(f"  Precision : {bo['precision']:>7.1%}    (of predicted overlap, how much is real)")
    print(f"  Recall    : {bo['recall']:>7.1%}    (of real overlap, how much is detected)")
    print(f"  F1        : {bo['f1']:>7.1%}")
    print()

    # ── Per-file accuracy distribution ────────────────────────────────────
    print(f"  ┌─── Per-File Accuracy Distribution ───────────┐")
    print(f"  Mean   : {results['per_file_accuracy_mean']:>7.1%}")
    print(f"  Std    : {results['per_file_accuracy_std']:>7.1%}")
    print(f"  Min    : {results['per_file_accuracy_min']:>7.1%}")
    print(f"  Max    : {results['per_file_accuracy_max']:>7.1%}")

    print(f"\n{'═' * 65}\n")


def plot_confusion_matrix(cm, output_path):
    """Save a colour-coded confusion matrix as an image."""
    if not HAS_MPL:
        print("  ⚠ matplotlib not available — skipping confusion matrix plot.")
        return

    cm_arr = np.array(cm, dtype=np.float64)
    names = ['Silence (0)', 'Single (1)', 'Overlap (2)']

    # Normalise rows to percentages
    row_sums = cm_arr.sum(axis=1, keepdims=True)
    cm_pct = np.divide(cm_arr, row_sums, where=row_sums > 0,
                       out=np.zeros_like(cm_arr)) * 100

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_pct, cmap='Blues', vmin=0, vmax=100)

    for i in range(3):
        for j in range(3):
            count = int(cm_arr[i, j])
            pct = cm_pct[i, j]
            colour = 'white' if pct > 60 else 'black'
            ax.text(j, i, f'{count:,}\n({pct:.1f}%)',
                    ha='center', va='center', fontsize=10, color=colour)

    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(names)
    ax.set_yticklabels(names)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Ground Truth', fontsize=12)
    ax.set_title('Confusion Matrix (row-normalised %)', fontsize=13,
                 fontweight='bold')
    fig.colorbar(im, ax=ax, shrink=0.8, label='%')
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Confusion matrix plot → {output_path}")


# ===========================================================================
#  DNN Model — Loading, Inference & Comparison (requires PyTorch)
# ===========================================================================

# -- Mel-filterbank feature extraction (must match training script) ---------

def _hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + np.asarray(hz, dtype=np.float64) / 700.0)


def _mel_to_hz(mel):
    return 700.0 * (10.0 ** (np.asarray(mel, dtype=np.float64) / 2595.0) - 1.0)


def _create_mel_filterbank(sr, n_fft, n_mels, fmin=0.0, fmax=None):
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


_dnn_mel_fb_cache = {}


def _compute_log_mel(audio, sr, cfg):
    """Compute log mel-filterbank energies using DNN checkpoint config."""
    n_fft = cfg.get('n_fft', 2048)
    n_mels = cfg.get('n_mels', 40)
    fmin = cfg.get('fmin', 80.0)
    fmax = cfg.get('fmax', 8000.0)
    hop_samples = cfg.get('hop_samples', int(sr * 0.01))
    analysis_win = cfg.get('analysis_window_samples', int(sr * 0.025))

    cache_key = (sr, n_fft, n_mels, fmin, fmax)
    if cache_key not in _dnn_mel_fb_cache:
        _dnn_mel_fb_cache[cache_key] = _create_mel_filterbank(
            sr, n_fft, n_mels, fmin, fmax)
    mel_fb = _dnn_mel_fb_cache[cache_key]

    noverlap = analysis_win - hop_samples
    _, _, Zxx = scipy_stft(audio, fs=sr, window='hann',
                           nperseg=analysis_win, noverlap=noverlap,
                           nfft=n_fft)
    power = np.abs(Zxx) ** 2
    mel_energy = mel_fb @ power
    log_mel = np.log(np.maximum(mel_energy, 1e-10))
    return log_mel.T.astype(np.float32)            # (n_frames, n_mels)


# -- DNN model architectures (must match training script exactly) -----------
#    Class names & layer structure MUST be identical so that state_dict
#    keys produced by the training script can be loaded correctly.

if HAS_TORCH:

    class MultivoiceVAD_CNN(nn.Module):
        """2-D CNN (matches train_mvad_dnn.py architecture)."""

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
                nn.Linear(128, 3),
            )

        def forward(self, x):
            x = x.unsqueeze(1)          # (B, 1, C, M)
            x = self.features(x)
            return self.classifier(x)

    class MultivoiceVAD_GRU(nn.Module):
        """Bidirectional GRU (matches train_mvad_dnn.py architecture)."""

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
                nn.Linear(128, 3),
            )

        def forward(self, x):
            out, _ = self.gru(x)                     # (B, C, hidden*2)
            centre = out[:, self.half_ctx, :]         # (B, hidden*2)
            return self.classifier(centre)

    class MultivoiceVAD_MLP(nn.Module):
        """Fully-connected MLP (matches train_mvad_dnn.py architecture)."""

        def __init__(self, context_frames, n_mels, hidden_dims=None,
                     dropout=0.3):
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
            layers.append(nn.Linear(prev, 3))
            self.network = nn.Sequential(*layers)

        def forward(self, x):
            return self.network(x.flatten(1))


# -- DNN loading & inference ------------------------------------------------

def load_dnn_model(model_path, device=None):
    """
    Load a trained DNN model from a checkpoint file.

    Parameters
    ----------
    model_path : str or Path
    device : torch.device or None  (auto-detect GPU/CPU)

    Returns
    -------
    model, config, feat_mean, feat_std, device
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required for DNN model inference. "
                           "Install with: pip install torch")

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt = torch.load(str(model_path), map_location=device, weights_only=False)
    cfg = ckpt['config']

    arch = cfg['arch']
    ctx = cfg['context_frames']
    n_mels = cfg['n_mels']
    dropout = cfg.get('dropout', 0.3)

    if arch == 'cnn':
        model = MultivoiceVAD_CNN(ctx, n_mels, dropout)
    elif arch == 'gru':
        model = MultivoiceVAD_GRU(ctx, n_mels, dropout=dropout)
    else:
        hidden_dims = cfg.get('hidden_dims', [512, 256, 128])
        model = MultivoiceVAD_MLP(ctx, n_mels, hidden_dims, dropout)

    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    model.eval()

    std_info = ckpt['standardisation']
    feat_mean = np.asarray(std_info['mean'], dtype=np.float32)
    feat_std = np.asarray(std_info['std'], dtype=np.float32)

    print(f"  DNN model loaded: arch={arch}, context={ctx} frames "
          f"({ctx * 10} ms), n_mels={n_mels}, device={device}")
    if 'epoch' in ckpt:
        print(f"  Best epoch: {ckpt['epoch']}")

    return model, cfg, feat_mean, feat_std, device


def dnn_predict_file(audio, sr, model, cfg, feat_mean, feat_std, device,
                     batch_size=1024):
    """
    Run DNN inference on a single audio file.

    Returns
    -------
    predictions : np.ndarray[int32]   shape (n_frames,)  values {0,1,2}
    """
    log_mel = _compute_log_mel(audio, sr, cfg)       # (n_frames, n_mels)
    n_frames = len(log_mel)
    ctx_frames = cfg['context_frames']
    half_ctx = ctx_frames // 2

    # Z-score normalisation (fitted on training set)
    mel_norm = (log_mel - feat_mean) / feat_std

    # Pad for context window extraction
    mel_padded = np.pad(mel_norm, ((half_ctx, half_ctx), (0, 0)), mode='edge')

    predictions = np.zeros(n_frames, dtype=np.int32)

    with torch.no_grad():
        for start in range(0, n_frames, batch_size):
            end = min(start + batch_size, n_frames)
            batch_list = []
            for i in range(start, end):
                ctx = mel_padded[i:i + ctx_frames]
                batch_list.append(ctx)
            batch_t = torch.from_numpy(np.array(batch_list)).to(device)
            logits = model(batch_t)
            predictions[start:end] = logits.argmax(1).cpu().numpy()

    return predictions


# -- DNN dataset evaluation -------------------------------------------------

def evaluate_dnn_dataset(eval_dir, model, cfg, feat_mean, feat_std, device,
                         verbose=True):
    """
    Run DNN on every WAV in *eval_dir*, compare with GT (_gt.npy),
    return aggregated metrics (same format as evaluate_dataset).
    """
    eval_dir = Path(eval_dir)
    wav_files = sorted(eval_dir.glob('*.wav'))
    if not wav_files:
        print(f"  ✗ No WAV files found in {eval_dir}")
        return None

    cm_3class = np.zeros((3, 3), dtype=np.int64)
    all_pred, all_gt = [], []
    per_file_results = []
    total_time = 0.0

    # DNN framing parameters (from checkpoint config)
    dnn_hop = cfg.get('hop_samples', 480)
    dnn_win = cfg.get('analysis_window_samples', 1200)

    n_files = len(wav_files)
    print(f"\n  Evaluating DNN on {n_files} files in: {eval_dir}\n")

    for idx, wp in enumerate(wav_files):
        gt_path = wp.parent / (wp.stem + '_gt.npy')
        if not gt_path.exists():
            if verbose:
                print(f"    ⚠ Skipping {wp.name} — no GT file")
            continue

        signal, sr = sf.read(str(wp), dtype='float64')
        if signal.ndim > 1:
            signal = np.mean(signal, axis=1)
        gt_samples = np.load(str(gt_path)).astype(np.int32)

        n = min(len(signal), len(gt_samples))
        signal, gt_samples = signal[:n], gt_samples[:n]

        t0 = time.time()
        pred_frames = dnn_predict_file(
            signal, sr, model, cfg, feat_mean, feat_std, device)
        elapsed = time.time() - t0
        total_time += elapsed

        # Align GT to DNN framing
        gt_frames = gt_to_frame_labels(gt_samples, dnn_win, dnn_hop)

        m = min(len(pred_frames), len(gt_frames))
        pred_frames, gt_frames = pred_frames[:m], gt_frames[:m]

        file_cm = compute_confusion_matrix(pred_frames, gt_frames)
        cm_3class += file_cm
        all_pred.append(pred_frames)
        all_gt.append(gt_frames)

        file_acc = np.sum(pred_frames == gt_frames) / m if m > 0 else 0.0
        per_file_results.append(dict(
            file=wp.name, n_frames=m, accuracy=file_acc, time_s=elapsed,
        ))

        if verbose and ((idx + 1) % 10 == 0 or idx + 1 == n_files):
            print(f"    [{idx+1:>4d}/{n_files}]  "
                  f"running acc = "
                  f"{np.trace(cm_3class) / max(1, cm_3class.sum()) * 100:.1f} %")

    if not per_file_results:
        print("  ✗ No files evaluated (DNN)")
        return None

    all_pred = np.concatenate(all_pred)
    all_gt = np.concatenate(all_gt)

    metrics = metrics_from_cm(cm_3class)
    vad_binary = binary_metrics(all_pred, all_gt, positive_labels={1, 2})
    ovl_binary = binary_metrics(all_pred, all_gt, positive_labels={2})

    accs = [r['accuracy'] for r in per_file_results]

    return dict(
        n_files=len(per_file_results),
        total_frames=int(cm_3class.sum()),
        total_processing_time_s=total_time,
        rtf=total_time / (len(all_pred) * 10 / 1000) if len(all_pred) > 0 else 0,
        confusion_matrix=cm_3class.tolist(),
        three_class=metrics,
        binary_vad=vad_binary,
        binary_overlap=ovl_binary,
        per_file_accuracy_mean=float(np.mean(accs)),
        per_file_accuracy_std=float(np.std(accs)),
        per_file_accuracy_min=float(np.min(accs)),
        per_file_accuracy_max=float(np.max(accs)),
        per_file=per_file_results,
    )


# -- Comparison report ------------------------------------------------------

def print_comparison_report(trad_results, dnn_results):
    """Print a side-by-side comparison of Traditional vs DNN model metrics."""
    t = trad_results['three_class']
    d = dnn_results['three_class']

    print(f"\n{'═' * 75}")
    print(f"  COMPARISON: Traditional Signal-Processing vs DNN Model")
    print(f"{'═' * 75}")

    # Header
    print(f"\n  {'Metric':<35s}  {'Traditional':>12s}  {'DNN':>12s}  {'Δ (DNN−Trad)':>13s}")
    print(f"  {'─' * 75}")

    def _row(name, tv, dv):
        delta = dv - tv
        sign = '+' if delta >= 0 else ''
        print(f"  {name:<35s}  {tv:>11.1%}  {dv:>11.1%}  {sign}{delta:>11.1%}")

    # ── Overall metrics
    _row('3-class accuracy', t['accuracy'], d['accuracy'])
    _row('Macro F1', t['macro_f1'], d['macro_f1'])
    _row('Macro precision', t['macro_precision'], d['macro_precision'])
    _row('Macro recall', t['macro_recall'], d['macro_recall'])
    print()

    # ── Per-class F1 / Precision / Recall
    class_names = ['silence', 'single', 'overlap']
    for i, name in enumerate(class_names):
        tp = t['per_class'][i]
        dp = d['per_class'][i]
        _row(f'  {name.capitalize()} — F1', tp['f1'], dp['f1'])
        _row(f'  {name.capitalize()} — Precision', tp['precision'], dp['precision'])
        _row(f'  {name.capitalize()} — Recall', tp['recall'], dp['recall'])
        print()

    # ── Binary: Speech vs Silence
    tb, db = trad_results['binary_vad'], dnn_results['binary_vad']
    _row('Binary Speech/Sil — F1', tb['f1'], db['f1'])
    _row('Binary Speech/Sil — Precision', tb['precision'], db['precision'])
    _row('Binary Speech/Sil — Recall', tb['recall'], db['recall'])
    print()

    # ── Binary: Overlap vs Non-overlap
    to, do_ = trad_results['binary_overlap'], dnn_results['binary_overlap']
    _row('Binary Overlap — F1', to['f1'], do_['f1'])
    _row('Binary Overlap — Precision', to['precision'], do_['precision'])
    _row('Binary Overlap — Recall', to['recall'], do_['recall'])
    print()

    # ── Processing time
    t_time = trad_results['total_processing_time_s']
    d_time = dnn_results['total_processing_time_s']
    speedup = t_time / d_time if d_time > 0 else float('inf')
    print(f"  {'Processing time (s)':<35s}  {t_time:>11.1f}s  {d_time:>11.1f}s"
          f"  {'(' + f'{speedup:.1f}×' + ')':>13s}")

    # ── Per-file accuracy stats
    print(f"\n  {'Per-file accuracy — Mean':<35s}  "
          f"{trad_results['per_file_accuracy_mean']:>11.1%}  "
          f"{dnn_results['per_file_accuracy_mean']:>11.1%}")
    print(f"  {'Per-file accuracy — Std':<35s}  "
          f"{trad_results['per_file_accuracy_std']:>11.1%}  "
          f"{dnn_results['per_file_accuracy_std']:>11.1%}")
    print(f"  {'Per-file accuracy — Min':<35s}  "
          f"{trad_results['per_file_accuracy_min']:>11.1%}  "
          f"{dnn_results['per_file_accuracy_min']:>11.1%}")
    print(f"  {'Per-file accuracy — Max':<35s}  "
          f"{trad_results['per_file_accuracy_max']:>11.1%}  "
          f"{dnn_results['per_file_accuracy_max']:>11.1%}")

    print(f"\n{'═' * 75}\n")


def plot_comparison_confusion_matrices(trad_cm, dnn_cm, output_path):
    """Save side-by-side confusion matrices for Traditional and DNN."""
    if not HAS_MPL:
        print("  ⚠ matplotlib not available — skipping comparison plot.")
        return

    names = ['Silence (0)', 'Single (1)', 'Overlap (2)']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Confusion Matrices: Traditional vs DNN',
                 fontsize=14, fontweight='bold')

    for ax, cm_raw, title in [(ax1, trad_cm, 'Traditional'),
                               (ax2, dnn_cm, 'DNN')]:
        cm_arr = np.array(cm_raw, dtype=np.float64)
        row_sums = cm_arr.sum(axis=1, keepdims=True)
        cm_pct = np.divide(cm_arr, row_sums, where=row_sums > 0,
                           out=np.zeros_like(cm_arr)) * 100

        im = ax.imshow(cm_pct, cmap='Blues', vmin=0, vmax=100)
        for i in range(3):
            for j in range(3):
                count = int(cm_arr[i, j])
                pct = cm_pct[i, j]
                colour = 'white' if pct > 60 else 'black'
                ax.text(j, i, f'{count:,}\n({pct:.1f}%)',
                        ha='center', va='center', fontsize=9, color=colour)

        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.set_xticklabels(names, fontsize=9)
        ax.set_yticklabels(names, fontsize=9)
        ax.set_xlabel('Predicted', fontsize=11)
        ax.set_ylabel('Ground Truth', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')

    fig.colorbar(im, ax=[ax1, ax2], shrink=0.8, label='%')
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Comparison confusion matrix plot → {output_path}")


def print_single_file_comparison(trad_labels, dnn_labels, hop_s):
    """Print comparison summary when no GT is available (single-file mode)."""
    m = min(len(trad_labels), len(dnn_labels))
    tl = trad_labels[:m]
    dl = dnn_labels[:m]

    print(f"\n{'═' * 65}")
    print(f"  COMPARISON: Traditional vs DNN (single file)")
    print(f"{'═' * 65}")

    # Per-class distribution
    print(f"\n  {'Class':<20s}  {'Traditional':>14s}  {'DNN':>14s}")
    print(f"  {'─' * 52}")
    for lab, name in [(0, 'Silence / noise'), (1, 'Single speaker'),
                       (2, 'Overlap (≥2)')]:
        t_dur = np.sum(tl == lab) * hop_s
        t_pct = np.sum(tl == lab) / m * 100
        d_dur = np.sum(dl == lab) * hop_s
        d_pct = np.sum(dl == lab) / m * 100
        print(f"  {name:<20s}  {t_dur:6.2f}s ({t_pct:5.1f}%)  "
              f"{d_dur:6.2f}s ({d_pct:5.1f}%)")

    # Agreement
    agree = np.sum(tl == dl)
    print(f"\n  Frame agreement: {agree:,} / {m:,}  ({agree / m * 100:.1f} %)")

    # Per-class IoU (Intersection over Union of label sets)
    print(f"\n  {'Class':<20s}  {'IoU':>8s}  {'Trad→DNN':>10s}  {'DNN→Trad':>10s}")
    print(f"  {'─' * 52}")
    for lab, name in [(0, 'Silence'), (1, 'Single'), (2, 'Overlap')]:
        both = np.sum((tl == lab) & (dl == lab))
        either = np.sum((tl == lab) | (dl == lab))
        iou = both / either * 100 if either > 0 else 0
        # What fraction of Trad's class X is also DNN's class X
        trad_cnt = np.sum(tl == lab)
        dnn_cnt = np.sum(dl == lab)
        t2d = both / trad_cnt * 100 if trad_cnt > 0 else 0
        d2t = both / dnn_cnt * 100 if dnn_cnt > 0 else 0
        print(f"  {name:<20s}  {iou:7.1f}%  {t2d:9.1f}%  {d2t:9.1f}%")

    print(f"\n{'═' * 65}\n")


# ===========================================================================
#  CLI
# ===========================================================================

def _build_vad_factory(args):
    """Return a callable that creates a MultivoiceVAD for a given sr."""
    def factory(sr):
        return MultivoiceVAD(
            sr=sr,
            frame_ms=args.frame_ms,
            hop_ms=args.hop_ms,
            energy_threshold_db=args.energy_threshold_db,
            pitch_conf_thresh=args.pitch_conf,
            yin_threshold=args.yin_threshold,
            secondary_pitch_conf=args.secondary_pitch_conf,
            spectral_flatness_overlap=args.sf_threshold,
            overlap_threshold=args.overlap_threshold,
            context_frames=args.context_frames,
            median_filter_size=(args.median_filter
                                if args.median_filter > 0 else 1),
            f0_min=args.f0_min,
            f0_max=args.f0_max,
        )
    return factory


def main():
    ap = argparse.ArgumentParser(
        description='Multi-Voice VAD — single-file analysis OR dataset evaluation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap())

    mode = ap.add_argument_group('Mode (choose one)')
    mode.add_argument('-i', '--input', default=None,
                      help='Path to a single input WAV file')
    mode.add_argument('--eval-dir', default=None,
                      help='Directory with test WAV + _gt.npy files '
                           '(e.g. multivoice_VAD_data_generation/test)')

    ap.add_argument('-o', '--output-csv', default=None,
                    help='[single-file] Output CSV path')
    ap.add_argument('--plot', nargs='?', const='auto', default=None,
                    help='[single-file] Save plot')
    ap.add_argument('--report-json', default=None,
                    help='[eval] Save evaluation report as JSON')
    ap.add_argument('--confusion-plot', default=None,
                    help='[eval] Save confusion matrix image '
                         '(default: <eval_dir>/confusion_matrix.png)')
    ap.add_argument('--dnn-model', default=None, metavar='PATH',
                    help='Path to a trained DNN model (.pt checkpoint) — '
                         'enables comparison with the traditional VAD')

    # tuning knobs
    g = ap.add_argument_group('Algorithm parameters')
    g.add_argument('--frame-ms',    type=float, default=30,
                   help='Frame duration in ms (default: 30)')
    g.add_argument('--hop-ms',      type=float, default=10,
                   help='Hop duration in ms (default: 10)')
    g.add_argument('--energy-threshold-db', type=float, default=-40,
                   help='Silence energy floor in dB (default: −40)')
    g.add_argument('--pitch-conf',  type=float, default=0.25,
                   help='Min pitch confidence for voiced (default: 0.25)')
    g.add_argument('--yin-threshold', type=float, default=0.15,
                   help='YIN absolute threshold (default: 0.15)')
    g.add_argument('--secondary-pitch-conf', type=float, default=0.20,
                   help='Min confidence for secondary pitch (default: 0.20)')
    g.add_argument('--sf-threshold', type=float, default=0.30,
                   help='Spectral-flatness overlap hint (default: 0.30)')
    g.add_argument('--overlap-threshold', type=float, default=0.38,
                   help='Overlap score threshold for label=2 (default: 0.38)')
    g.add_argument('--context-frames', type=int, default=3,
                   help='±K frames for temporal max-pool smoothing (default: 3)')
    g.add_argument('--median-filter', type=int, default=7,
                   help='Median-filter kernel (odd int; 0 = off; default: 7)')
    g.add_argument('--f0-min', type=float, default=80,
                   help='Minimum F0 in Hz (default: 80)')
    g.add_argument('--f0-max', type=float, default=400,
                   help='Maximum F0 in Hz (default: 400)')

    args = ap.parse_args()

    # ── Evaluation mode ───────────────────────────────────────────────────
    if args.eval_dir is not None:
        eval_dir = Path(args.eval_dir)
        if not eval_dir.is_dir():
            print(f"  ✗ Not a directory: {eval_dir}")
            return 1

        factory = _build_vad_factory(args)
        results = evaluate_dataset(eval_dir, factory, verbose=True)
        if results is None:
            return 1

        print_eval_report(results)

        # ── DNN comparison (optional) ────────────────────────────────────
        dnn_results = None
        if args.dnn_model is not None:
            dnn_path = Path(args.dnn_model)
            if not dnn_path.exists():
                print(f"  ✗ DNN model not found: {dnn_path}")
                return 1

            model, cfg, feat_mean, feat_std, device = load_dnn_model(dnn_path)
            dnn_results = evaluate_dnn_dataset(
                eval_dir, model, cfg, feat_mean, feat_std, device,
                verbose=True,
            )
            if dnn_results is not None:
                print_comparison_report(results, dnn_results)

        # ── Save JSON report
        json_path = args.report_json
        if json_path is None:
            json_path = str(eval_dir / 'eval_report.json')

        # Make JSON-serialisable (remove per_class dicts with lambda)
        report = {k: v for k, v in results.items() if k != 'per_file'}
        report['per_file_summary'] = [
            {k: (round(v, 4) if isinstance(v, float) else v)
             for k, v in f.items()}
            for f in results['per_file']
        ]
        if dnn_results is not None:
            report['dnn'] = {k: v for k, v in dnn_results.items()
                             if k != 'per_file'}
            report['dnn']['per_file_summary'] = [
                {k: (round(v, 4) if isinstance(v, float) else v)
                 for k, v in f.items()}
                for f in dnn_results['per_file']
            ]
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"  Report saved → {json_path}")

        # ── Confusion matrix plot
        cm_path = args.confusion_plot
        if cm_path is None:
            cm_path = str(eval_dir / 'confusion_matrix.png')
        plot_confusion_matrix(results['confusion_matrix'], cm_path)

        # ── Side-by-side confusion matrix plot (if DNN evaluated)
        if dnn_results is not None:
            cmp_cm_path = str(
                Path(cm_path).parent
                / (Path(cm_path).stem + '_comparison.png')
            )
            plot_comparison_confusion_matrices(
                results['confusion_matrix'],
                dnn_results['confusion_matrix'],
                cmp_cm_path,
            )

        return 0

    # ── Single-file mode ──────────────────────────────────────────────────
    if args.input is None:
        ap.print_help()
        print("\n  ✗ Please specify either -i (single file) or --eval-dir (evaluation)")
        return 1

    inp = Path(args.input)
    if not inp.exists():
        print(f"  ✗ File not found: {inp}")
        return 1

    stem   = inp.stem
    parent = inp.parent
    csv_path = Path(args.output_csv) if args.output_csv else parent / f'{stem}_mvad.csv'

    plot_path = None
    if args.plot is not None:
        plot_path = (parent / f'{stem}_mvad.png') if args.plot == 'auto' else Path(args.plot)

    # --- load audio --------------------------------------------------------
    print(f"  Loading : {inp}")
    signal, sr = sf.read(str(inp), dtype='float64')
    if signal.ndim > 1:
        signal = np.mean(signal, axis=1)
    print(f"  Format  : {sr} Hz, mono, {len(signal)/sr:.2f} s")

    # --- run VAD -----------------------------------------------------------
    factory = _build_vad_factory(args)
    vad = factory(sr)

    labels, features = vad.process(signal)

    # --- outputs -----------------------------------------------------------
    print_summary(labels, vad)
    save_csv(labels, vad, csv_path)
    if plot_path:
        plot_results(signal, sr, labels, features, vad, output_path=str(plot_path))

    # --- DNN comparison (optional, single-file) ---------------------------
    if args.dnn_model is not None:
        dnn_path = Path(args.dnn_model)
        if not dnn_path.exists():
            print(f"  ✗ DNN model not found: {dnn_path}")
            return 1

        model, cfg, feat_mean, feat_std, device = load_dnn_model(dnn_path)
        dnn_labels = dnn_predict_file(
            signal, sr, model, cfg, feat_mean, feat_std, device,
        )

        hop_s = args.hop_ms / 1000.0

        # If ground-truth is available, do full evaluation comparison
        gt_path = parent / (stem + '_gt.npy')
        if gt_path.exists():
            gt_samples = np.load(str(gt_path)).astype(np.int32)
            n = min(len(signal), len(gt_samples))
            gt_samples = gt_samples[:n]

            # Traditional GT frames (use same hop as traditional VAD)
            trad_hop = int(sr * args.hop_ms / 1000)
            trad_win = int(sr * args.frame_ms / 1000)
            gt_frames_trad = gt_to_frame_labels(gt_samples, trad_win, trad_hop)

            # DNN GT frames (use DNN framing from loaded checkpoint config)
            dnn_hop = cfg.get('hop_samples', int(sr * 0.01))
            dnn_win = cfg.get('analysis_window_samples', int(sr * 0.025))
            gt_frames_dnn = gt_to_frame_labels(gt_samples, dnn_win, dnn_hop)

            # Compute metrics for both
            m_trad = min(len(labels), len(gt_frames_trad))
            cm_trad = compute_confusion_matrix(labels[:m_trad],
                                               gt_frames_trad[:m_trad])
            m_dnn = min(len(dnn_labels), len(gt_frames_dnn))
            cm_dnn = compute_confusion_matrix(dnn_labels[:m_dnn],
                                              gt_frames_dnn[:m_dnn])

            trad_metrics_3c = metrics_from_cm(cm_trad)
            dnn_metrics_3c = metrics_from_cm(cm_dnn)

            trad_result = dict(
                three_class=trad_metrics_3c,
                binary_vad=binary_metrics(labels[:m_trad],
                                          gt_frames_trad[:m_trad],
                                          positive_labels={1, 2}),
                binary_overlap=binary_metrics(labels[:m_trad],
                                              gt_frames_trad[:m_trad],
                                              positive_labels={2}),
                confusion_matrix=cm_trad.tolist(),
                total_processing_time_s=0,
                per_file_accuracy_mean=trad_metrics_3c['accuracy'],
                per_file_accuracy_std=0.0,
                per_file_accuracy_min=trad_metrics_3c['accuracy'],
                per_file_accuracy_max=trad_metrics_3c['accuracy'],
            )
            dnn_result = dict(
                three_class=dnn_metrics_3c,
                binary_vad=binary_metrics(dnn_labels[:m_dnn],
                                          gt_frames_dnn[:m_dnn],
                                          positive_labels={1, 2}),
                binary_overlap=binary_metrics(dnn_labels[:m_dnn],
                                              gt_frames_dnn[:m_dnn],
                                              positive_labels={2}),
                confusion_matrix=cm_dnn.tolist(),
                total_processing_time_s=0,
                per_file_accuracy_mean=dnn_metrics_3c['accuracy'],
                per_file_accuracy_std=0.0,
                per_file_accuracy_min=dnn_metrics_3c['accuracy'],
                per_file_accuracy_max=dnn_metrics_3c['accuracy'],
            )
            print_comparison_report(trad_result, dnn_result)
        else:
            # No GT — just compare label distributions
            print_single_file_comparison(labels, dnn_labels, hop_s)

    return 0


# ---------------------------------------------------------------------------

def textwrap():
    return """\
Output labels (per frame):
  0  silence / no speech
  1  single speaker
  2  overlapping speech (≥ 2 speakers)

Modes:
  1. Single file (traditional VAD only):
     python3 mvad_test.py -i inputs/example_1.wav --plot

  2. Evaluate traditional VAD on generated test set:
     python3 mvad_test.py --eval-dir multivoice_VAD_data_generation/test

  3. Single file + DNN comparison:
     python3 mvad_test.py -i inputs/example_1.wav --plot --dnn-model mvad_dnn_model_ep47.pt

  4. Evaluate + DNN comparison (full side-by-side report):
     python3 mvad_test.py --eval-dir multivoice_VAD_data_generation/test \\
                          --dnn-model mvad_dnn_model_ep47.pt

Pipeline — Traditional (v2 — enhanced):
  1. High-pass filter (80 Hz) + pre-emphasis
  2. Frame into 30 ms windows (10 ms hop)
  3. Vectorised: STE, ZCR, delta-energy
  4. Per speech frame:
       a) YIN pitch tracking (F0 + confidence)
       b) Harmonic sieve multi-pitch detection
       c) Harmonic-to-Noise Ratio (HNR)
       d) Spectral centroid + bandwidth
       e) Spectral flatness + sub-band entropy
  5. Adaptive thresholds from speech-frame distributions
  6. Multi-feature overlap scoring  (0 … 1)
  7. Temporal context smoothing (±K frames)
  8. Classification: silence / single / overlap
  9. Median-filter post-processing

Pipeline — DNN (requires --dnn-model):
  1. Mel-filterbank feature extraction (log-mel energies)
  2. Z-score standardisation (using training-set stats from checkpoint)
  3. Context-window framing (±K frames)
  4. Forward pass through CNN / GRU / MLP
  5. Argmax → 3-class prediction

Examples:
  python3 mvad_test.py -i inputs/example_1.wav --plot
  python3 mvad_test.py --eval-dir multivoice_VAD_data_generation/test
  python3 mvad_test.py --eval-dir multivoice_VAD_data_generation/test --pitch-conf 0.25
  python3 mvad_test.py -i inputs/example_1.wav --dnn-model mvad_dnn_model_ep47.pt
  python3 mvad_test.py --eval-dir multivoice_VAD_data_generation/test --dnn-model mvad_dnn_model_ep47.pt
"""


if __name__ == '__main__':
    sys.exit(main())
