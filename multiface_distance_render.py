#!/usr/bin/env python3
# coding: utf-8

"""
Multi-Face Distance Video Renderer with Speaker Avatars
Renders an annotated video with face distance dashboard overlay
and speaker identification avatars above detected faces.
Uses pre-computed CSV dumps for fast rendering.
"""

import argparse
import pandas as pd
import numpy as np
import cv2
import imageio
from pathlib import Path
from tqdm import tqdm
from collections import deque, OrderedDict
import subprocess
import os
try:
    import soundfile as sf
except ImportError:
    sf = None
import torch
import torch.nn as nn

from reindex_face_ids import compute_assignment_map, _reindex_df

# Color scheme for different faces (BGR format for OpenCV)
FACE_COLORS_BGR = [
    (228, 119, 31),   # Blue
    (14, 127, 255),   # Orange
    (44, 160, 44),    # Green
    (40, 39, 214),    # Red
    (189, 103, 148),  # Purple
    (75, 86, 140),    # Brown
    (194, 119, 227),  # Pink
    (127, 127, 127),  # Gray
    (34, 189, 188),   # Olive
    (207, 190, 23),   # Cyan
]


def brighten_color(bgr, factor=1.4, offset=40):
    """Return a brightened version of a BGR colour tuple for dark backgrounds."""
    return tuple(min(255, int(c * factor + offset)) for c in bgr)


def calculate_distance(z_value):
    """
    Convert z-coordinate to approximate distance in meters
    z is depth from camera, inverse relationship
    """
    if z_value <= 0:
        return 0
    return 100.0 / z_value


def smooth_binary_signal(signal, pad_frames):
    """
    Morphological dilation of a binary (0/1) signal.

    Extends each contiguous active block by *pad_frames* on both sides.
    Implementation uses a 1-D convolution with a rectangular kernel.

    Args:
        signal:     1-D array-like of 0/1 values.
        pad_frames: Number of frames to add on each side of every active block.

    Returns:
        numpy int array of same length as *signal* (values 0 or 1).
    """
    if pad_frames <= 0:
        return np.array(signal, dtype=int)
    arr = np.array(signal, dtype=float)
    kernel = np.ones(2 * pad_frames + 1)
    smoothed = np.convolve(arr, kernel, mode='same')
    return (smoothed > 0).astype(int)


def precompute_smoothed_audio_vad(df_vad, timestamps, fps, pad_ms=250.0):
    """
    Align Audio VAD decisions to video-frame timestamps and apply smoothing.

    Smoothing = morphological dilation: each contiguous block of 1s is extended
    by *pad_ms* milliseconds on both sides.

    Args:
        df_vad:     DataFrame with columns 'seconds' and 'vadDagcDecFinal'.
        timestamps: 1-D array of video-frame timestamps (seconds).
        fps:        Video frame rate (used to convert ms → frames).
        pad_ms:     Padding in milliseconds added to each side of active blocks.

    Returns:
        numpy int array of 0/1 per video frame (smoothed Audio VAD).
    """
    n = len(timestamps)
    if df_vad is None or df_vad.empty or n == 0:
        return np.zeros(n, dtype=int)

    vad_times = df_vad['seconds'].values
    vad_vals = df_vad['vadDagcDecFinal'].values.astype(int)

    # Nearest-neighbour alignment of Audio VAD to video-frame grid
    aligned = np.zeros(n, dtype=int)
    for i, t in enumerate(timestamps):
        idx = np.argmin(np.abs(vad_times - t))
        aligned[i] = int(vad_vals[idx])

    # Smooth: extend each active block by pad_ms on each side
    if pad_ms > 0 and n > 1:
        dt = 1.0 / fps
        pad_frames = int(np.ceil(pad_ms / 1000.0 / dt))
        aligned = smooth_binary_signal(aligned, pad_frames)

    return aligned


# ---------------------------------------------------------------------------
# Face identity from embeddings (stable identity across frames)
# Use a face recognition/embedding model (e.g. InsightFace, FaceNet, dlib) to
# extract one vector per (frame, face). Pass precomputed embeddings via
# --embeddings path.npz; this module matches faces frame-to-frame by
# embedding similarity (cosine) and assigns stable track IDs.
# Assignment and reindexing logic: reindex_face_ids.compute_assignment_map, _reindex_df.
# See extract_face_embeddings.py for a script that iterates (seconds, face_idx),
# crops using landmarks, and optionally runs your embedding model → .npz.
# ---------------------------------------------------------------------------

def load_embeddings_npz(path):
    """
    Load precomputed face embeddings from a .npz file.
    Expected keys: 'embeddings' (N, D), 'seconds' (N,), 'face_idx' (N,).
    Each row i corresponds to one (seconds[i], face_idx[i]) with embedding embeddings[i].
    Use a face recognition model (e.g. InsightFace, FaceNet, dlib) on each detected face
    crop (aligned using dumps landmarks or bbox) and save in this format for --embeddings.
    Returns: (embeddings, seconds, face_idx).
    """
    data = np.load(path, allow_pickle=False)
    emb = np.asarray(data['embeddings'], dtype=np.float64)
    sec = np.asarray(data['seconds'], dtype=np.float64).ravel()
    fidx = np.asarray(data['face_idx'], dtype=np.int64).ravel()
    if len(sec) != len(fidx) or len(sec) != len(emb):
        raise ValueError("embeddings.npz: 'embeddings', 'seconds', 'face_idx' must have same length")
    return emb, sec, fidx


def draw_line_graph(frame, x, y, width, height, data_points, title, show_value=False, current_value=None, unit=""):
    """
    Draw a line graph on the frame using OpenCV
    
    Args:
        frame: Video frame (numpy array)
        x: X coordinate of graph top-left corner
        y: Y coordinate of graph top-left corner
        width: Width of the graph
        height: Height of the graph
        data_points: List of values to plot (recent history)
        title: Title text to display
        show_value: Whether to show the current value prominently
        current_value: Current value to display prominently (if show_value=True)
        unit: Unit string (e.g., "MS", "%")
    
    Returns:
        Modified frame
    """
    # Ensure frame is contiguous
    if not frame.flags['C_CONTIGUOUS']:
        frame = np.ascontiguousarray(frame)
    
    # Draw dark background for graph area
    cv2.rectangle(frame, (x, y), (x + width, y + height), (30, 30, 30), -1)
    cv2.rectangle(frame, (x, y), (x + width, y + height), (200, 200, 200), 1)
    
    # Draw title
    font = cv2.FONT_HERSHEY_SIMPLEX
    title_font_scale = 0.6
    title_thickness = 2
    
    cv2.putText(frame, title, (x + 5, y + 20), font, title_font_scale, 
                (200, 200, 200), title_thickness)
    
    # Optionally draw current value prominently
    graph_start_y = y + 30
    if show_value and current_value is not None:
        value_font_scale = 1.2
        value_thickness = 2
        value_text = f"{current_value:.0f} {unit}" if isinstance(current_value, (int, float)) else f"{current_value} {unit}"
        cv2.putText(frame, value_text, (x + 5, y + 55), font, value_font_scale, 
                    (255, 255, 255), value_thickness)
        graph_start_y = y + 65
    
    # Draw the line graph
    if len(data_points) > 1:
        # Graph area (below the text)
        graph_y = graph_start_y
        graph_height = height - (graph_start_y - y)
        graph_margin = 5
        
        # Normalize data points
        if len(data_points) > 0:
            min_val = min(data_points)
            max_val = max(data_points)
            val_range = max_val - min_val if max_val > min_val else 1
            
            # Calculate points
            points = []
            for i, val in enumerate(data_points):
                px = x + graph_margin + int((i / (len(data_points) - 1)) * (width - 2 * graph_margin))
                # Invert y-axis (higher values at top)
                normalized = (val - min_val) / val_range
                py = graph_y + graph_height - graph_margin - int(normalized * (graph_height - 2 * graph_margin))
                points.append((px, py))
            
            # Draw dotted line at max value
            max_y = min([p[1] for p in points])
            for dx in range(0, width, 10):
                cv2.line(frame, (x + dx, max_y), (x + dx + 5, max_y), (150, 150, 150), 1)
            
            # Draw the line graph
            for i in range(len(points) - 1):
                cv2.line(frame, points[i], points[i + 1], (200, 200, 200), 2)
    
    return frame


def draw_mvad_timeline(frame, x, y, width, height, history_labels, title, current_label):
    """
    Draw a coloured Multivoice VAD timeline on the dashboard.

    Colours (BGR):
        0 (silence)  → dark gray  (50, 50, 50)
        1 (single)   → dim green  (0, 150, 0)
        2 (overlap)  → bright orange  (0, 200, 255)

    The title text and border glow orange when the current label is overlap.
    """
    if not frame.flags['C_CONTIGUOUS']:
        frame = np.ascontiguousarray(frame)

    # Background
    cv2.rectangle(frame, (x, y), (x + width, y + height), (30, 30, 30), -1)
    cv2.rectangle(frame, (x, y), (x + width, y + height), (200, 200, 200), 1)

    font = cv2.FONT_HERSHEY_SIMPLEX

    # Title — orange when overlap, white otherwise
    title_color = (0, 200, 255) if current_label == 2 else (200, 200, 200)
    cv2.putText(frame, title, (x + 5, y + 20), font, 0.6, title_color, 2)

    # Coloured bar area
    bar_y_start = y + 30
    bar_h = height - 40
    bar_margin = 5
    usable_width = width - 2 * bar_margin

    MVAD_COLORS = {
        0: (50, 50, 50),       # silence — dark
        1: (0, 150, 0),        # single speaker — dim green
        2: (0, 200, 255),      # overlap — bright orange
    }

    n = len(history_labels)
    if n > 0:
        # Max-pool downsample if more data points than pixels (preserves overlap)
        if n > usable_width:
            bin_size = n / usable_width
            pooled = []
            for i in range(usable_width):
                s = int(i * bin_size)
                e = max(s + 1, int((i + 1) * bin_size))
                pooled.append(int(max(history_labels[s:e])))
            history_labels = pooled
            n = len(history_labels)

        for i in range(n):
            px = x + bar_margin + int(i * usable_width / n)
            pw = max(1, int((i + 1) * usable_width / n) - int(i * usable_width / n))
            color = MVAD_COLORS.get(int(history_labels[i]), (50, 50, 50))
            cv2.rectangle(frame, (px, bar_y_start), (px + pw, bar_y_start + bar_h), color, -1)

    # Orange glow border when overlap is active
    if current_label == 2:
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 200, 255), 2)

    return frame


def frontalize_mouth_landmarks(points_3d):
    """
    Frontalize mouth landmark points by estimating and removing head rotation.
    
    Uses stable facial landmarks (eyes, nose bridge, chin) to compute the 3D
    orientation of the face, then applies the inverse rotation to mouth points
    to produce a frontal-view representation of the mouth shape.
    
    Approach (Direct 3D Geometry):
        1. Compute the face coordinate frame from stable (non-deformable) landmarks:
           - X-axis: left eye center → right eye center (horizontal)
           - Y-axis: orthogonalized chin → nose bridge direction (vertical)
           - Z-axis: face normal = cross(X, Y_approx), pointing toward camera
        2. Build rotation matrix R_face = [x_axis | y_axis | z_axis]
        3. Apply R_face^T (inverse rotation) to centered mouth points to "de-rotate" them
        4. Restore the original centroid position so the points stay near the face in the image
    
    This preserves the actual mouth expression (open/closed/asymmetric) while removing
    the geometric distortion caused by head rotation (yaw/pitch/roll).
    
    Args:
        points_3d: dict {pt_idx: (x, y, z)} for all available facial landmarks.
                   Must contain at least: 8 (chin), 27 (nose bridge top),
                   36, 39 (right eye corners), 42, 45 (left eye corners),
                   and all mouth points 48-67.
    
    Returns:
        dict {pt_idx: (x_frontal, y_frontal)} for mouth landmarks (48-67),
        or None if required landmarks are missing.
    """
    # Required stable reference points for face frame estimation
    required_stable = [8, 27, 36, 39, 42, 45]
    mouth_indices = list(range(48, 68))
    
    for idx in required_stable + mouth_indices:
        if idx not in points_3d:
            return None
    
    # Convert to numpy arrays
    p = {k: np.array(v, dtype=np.float64) for k, v in points_3d.items()}
    
    # --- Step 1: Compute face coordinate frame from stable landmarks ---
    
    # Eye centers (structurally rigid - not affected by expression)
    # 68-point model: 36-41 = right eye, 42-47 = left eye (from subject's perspective)
    # In image space: points 36,39 = lower x (image-left), points 42,45 = higher x (image-right)
    left_eye_center = (p[36] + p[39]) / 2.0
    right_eye_center = (p[42] + p[45]) / 2.0
    
    # X-axis: horizontal direction of the face (left to right in image)
    x_axis = right_eye_center - left_eye_center
    x_norm = np.linalg.norm(x_axis)
    if x_norm < 1e-8:
        return None
    x_axis = x_axis / x_norm
    
    # Approximate Y-axis: vertical direction (nose bridge top → chin, downward in image)
    y_approx = p[8] - p[27]
    
    # Z-axis: face normal via cross product (points toward camera for frontal face)
    z_axis = np.cross(x_axis, y_approx)
    z_norm = np.linalg.norm(z_axis)
    if z_norm < 1e-8:
        return None
    z_axis = z_axis / z_norm
    
    # Re-orthogonalize Y-axis to ensure a proper orthonormal frame
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    
    # --- Step 2: Build rotation matrix ---
    # R_face columns = face frame axes in the CSV coordinate system
    # R_face transforms from canonical (frontal) frame → current face frame
    R_face = np.column_stack([x_axis, y_axis, z_axis])
    
    # Inverse rotation: R_face^T transforms face frame → canonical (frontal) frame
    R_inv = R_face.T
    
    # --- Step 3: De-rotate mouth points ---
    mouth_3d = np.array([list(points_3d[i]) for i in mouth_indices], dtype=np.float64)
    
    # Center around the mouth centroid (rotation must be about a local center)
    centroid = mouth_3d.mean(axis=0)
    mouth_centered = mouth_3d - centroid
    
    # Apply inverse rotation to remove head pose
    mouth_derotated = (R_inv @ mouth_centered.T).T
    
    # Restore position: place frontalized points at the original centroid (x, y)
    mouth_frontalized = mouth_derotated + centroid
    
    result = {}
    for i, idx in enumerate(mouth_indices):
        result[idx] = (int(round(mouth_frontalized[i, 0])), int(round(mouth_frontalized[i, 1])))
    
    return result


def frontalize_mouth_solvepnp(points_2d, points_z, frame_width, frame_height):
    """
    Alternative frontalization approach using cv2.solvePnP.
    
    Uses a canonical 3D face model and the detected 2D landmarks to estimate
    the head pose via the Perspective-n-Point algorithm, then de-rotates the
    mouth points in 3D camera space and re-projects them to 2D.
    
    Approach (solvePnP-based):
        1. Define a canonical 3D face model (6 reference points in mm)
        2. solvePnP matches detected 2D points to the canonical model → rvec, tvec
        3. Convert mouth 2D points to approximate 3D camera coordinates using z-depth
        4. Apply R^(-1) to the centered 3D mouth points to remove head rotation
        5. Re-project the de-rotated 3D points to 2D image coordinates
    
    Args:
        points_2d: dict {pt_idx: (x, y)} for all available landmarks (2D image coords)
        points_z: dict {pt_idx: z_value} for corresponding z-depth values from 3DDFA
        frame_width: width of the video frame in pixels
        frame_height: height of the video frame in pixels
    
    Returns:
        dict {pt_idx: (x_frontal, y_frontal)} for mouth landmarks (48-67),
        or None if pose estimation fails.
    """
    # Canonical 3D face model reference points (in mm, nose tip as origin)
    # Widely used with the 68-point iBUG landmark model
    # Reference: https://learnopencv.com/head-pose-estimation-using-opencv-and-dlib/
    canonical_model = {
        30: np.array([0.0, 0.0, 0.0]),             # Nose tip
        8:  np.array([0.0, -330.0, -65.0]),         # Chin
        36: np.array([-225.0, 170.0, -135.0]),      # Left eye outer corner
        45: np.array([225.0, 170.0, -135.0]),       # Right eye outer corner
        48: np.array([-150.0, -150.0, -125.0]),     # Left mouth corner
        54: np.array([150.0, -150.0, -125.0]),      # Right mouth corner
    }
    
    pose_indices = list(canonical_model.keys())
    mouth_indices = list(range(48, 68))
    
    # Verify required points exist
    for idx in pose_indices + mouth_indices:
        if idx not in points_2d:
            return None
    
    # Prepare arrays for solvePnP
    model_points = np.array([canonical_model[i] for i in pose_indices], dtype=np.float64)
    image_points = np.array([[points_2d[i][0], points_2d[i][1]] for i in pose_indices], dtype=np.float64)
    
    # Approximate camera intrinsics (pinhole model)
    focal_length = float(frame_width)
    cx, cy = frame_width / 2.0, frame_height / 2.0
    camera_matrix = np.array([
        [focal_length, 0, cx],
        [0, focal_length, cy],
        [0, 0, 1]
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1))
    
    # Estimate head pose
    success, rvec, tvec = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        return None
    
    # Get rotation matrix from rotation vector
    R, _ = cv2.Rodrigues(rvec)
    
    # --- Frontalize mouth points using their 3D positions ---
    # Construct approximate 3D coordinates for mouth points
    # using (u, v) pixel coords and z-depth from the CSV
    mouth_3d = np.array([
        [points_2d[i][0], points_2d[i][1], points_z[i]]
        for i in mouth_indices
    ], dtype=np.float64)
    
    # Center around mouth centroid
    centroid = mouth_3d.mean(axis=0)
    mouth_centered = mouth_3d - centroid
    
    # De-rotate using the inverse of the solvePnP rotation
    mouth_derotated = (R.T @ mouth_centered.T).T
    
    # Restore position
    mouth_frontalized = mouth_derotated + centroid
    
    result = {}
    for i, idx in enumerate(mouth_indices):
        result[idx] = (int(round(mouth_frontalized[i, 0])), int(round(mouth_frontalized[i, 1])))
    
    return result


def compute_mouth_aspect_ratio(mouth_points):
    """
    Compute the Mouth Aspect Ratio (MAR) from frontalized inner mouth landmarks.
    
    MAR measures how open the mouth is, analogous to EAR (Eye Aspect Ratio).
    Uses inner mouth landmarks (60-67) to compute the ratio of vertical opening
    to horizontal width.
    
    Inner mouth landmarks (68-point model):
        60: left corner, 61: upper-left, 62: upper-center, 63: upper-right,
        64: right corner, 65: lower-right, 66: lower-center, 67: lower-left
    
    Formula:
        MAR = (|p61-p67| + |p62-p66| + |p63-p65|) / (2 * |p60-p64|)
    
    Args:
        mouth_points: dict {pt_idx: (x, y)} containing at least inner mouth points 60-67
    
    Returns:
        float: Mouth Aspect Ratio (0 = fully closed, higher = more open),
        or None if required points are missing.
    """
    inner_indices = list(range(60, 68))
    for idx in inner_indices:
        if idx not in mouth_points:
            return None
    
    p = {k: np.array(v, dtype=np.float64) for k, v in mouth_points.items() if k in inner_indices}
    
    # Vertical distances (3 pairs: top-bottom)
    d1 = np.linalg.norm(p[61] - p[67])  # upper-left  ↔ lower-left
    d2 = np.linalg.norm(p[62] - p[66])  # upper-center ↔ lower-center
    d3 = np.linalg.norm(p[63] - p[65])  # upper-right ↔ lower-right
    
    # Horizontal distance (mouth width)
    d_horiz = np.linalg.norm(p[60] - p[64])  # left corner ↔ right corner
    
    if d_horiz < 1e-8:
        return None
    
    mar = (d1 + d2 + d3) / (2.0 * d_horiz)
    return mar


class VisualVADDetector:
    """
    Visual Voice Activity Detector (V-VAD) based on frontalized mouth landmarks.
    
    Detects whether a speaker is actively talking by analyzing the dynamics of the
    Mouth Aspect Ratio (MAR) over a sliding time window. Since landmarks are
    frontalized (head rotation removed), all MAR changes correspond to actual
    facial muscle movement rather than head pose changes.
    
    Detection logic (Delta-Enhanced with Zero-Crossing Rate):
        1. MAR variance > var_threshold  — mouth is moving dynamically
        2. MAR mean > mar_activity_threshold — mouth is at least partially open
        3. ZCR of MAR derivative >= min_zcr — mouth oscillates rhythmically
           (speech produces ~3-7 Hz open/close cycles, while idle open mouth
            or slow head-movement artifacts have low ZCR)
        All three conditions must be met simultaneously.
    
    Speech Probability:
        In addition to the binary decision, a continuous speech probability [0, 1]
        is computed using sigmoid normalization of each feature relative to its
        threshold. Individual feature scores are multiplied, so probability is high
        only when ALL features indicate speech. A smoothed version with exponential
        decay (matching the hold time) is also available.
    
    Hysteresis (hold-time):
        Once the detector transitions to ACTIVE (speaking), it will maintain that
        state for at least `hold_seconds` even if the MAR signal temporarily drops
        (e.g. between syllables, during plosive consonants where mouth closes briefly).
        This prevents rapid flickering between active/inactive states during continuous
        speech. The hold timer resets each time the raw signal confirms speech, so
        during actual speech the green state is sustained continuously.
    
    Args:
        window_seconds: Length of the sliding analysis window in seconds (default: 0.3)
        fps: Video frame rate, used to convert window to frame count
        var_threshold: MAR variance threshold (default: 0.005)
        mar_activity_threshold: Minimum MAR mean for speech (default: 0.30)
        hold_seconds: Minimum time to sustain ACTIVE state (default: 0.3)
        min_zcr: Minimum zero-crossing rate of MAR derivative (default: 3)
    """
    
    SIGMOID_K = 4.0  # Steepness of sigmoid normalization
    
    def __init__(self, window_seconds=0.3, fps=30.0, var_threshold=0.005,
                 mar_activity_threshold=0.30, hold_seconds=0.3, min_zcr=3):
        self.window_size = max(4, int(window_seconds * fps))
        self.var_threshold = var_threshold
        self.mar_activity_threshold = mar_activity_threshold
        self.hold_frames = max(1, int(hold_seconds * fps))
        self.min_zcr = min_zcr
        self._decay = 1.0 - (1.0 / max(1, self.hold_frames))
        # Per-face MAR history: {face_idx: deque([mar_values])}
        self._history = {}
        # Per-face hold countdown: {face_idx: int} — frames remaining in hold
        self._hold_counter = {}
        # Per-face smoothed probability: {face_idx: float}
        self._smoothed_prob = {}
    
    @staticmethod
    def _sigmoid(x, center, k=4.0):
        """Sigmoid function for soft thresholding. Returns 0.5 at x=center."""
        z = k * (x - center)
        z = max(-20.0, min(20.0, z))
        return 1.0 / (1.0 + np.exp(-z))
    
    def _compute_probability(self, mar_var, mar_mean, zcr):
        """
        Compute continuous speech probability [0, 1] from signal features.
        
        Each feature is normalized via sigmoid relative to its threshold (0.5 at
        the threshold, smooth transition). The combined probability is the product
        of all individual scores — high only when ALL features indicate speech.
        
        Returns:
            float: Speech probability in [0, 1]
        """
        k = self.SIGMOID_K
        vt = self.var_threshold
        mt = self.mar_activity_threshold
        mz = self.min_zcr
        
        s_var = self._sigmoid(mar_var / vt, 1.0, k) if vt > 0 else 0.0
        s_mar = self._sigmoid(mar_mean / mt, 1.0, k) if mt > 0 else 0.0
        s_zcr = self._sigmoid(zcr / mz, 1.0, k) if mz > 0 else 1.0
        
        return float(s_var * s_mar * s_zcr)
    
    def update(self, face_idx, mar_value):
        """
        Add a new MAR observation for a face and return the activity state
        along with speech probability.
        
        Uses hysteresis: once speaking is detected, the ACTIVE state is held for
        at least `hold_frames` even if the signal drops momentarily. Each new
        positive detection resets the hold timer.
        
        Args:
            face_idx: Index of the face
            mar_value: Current Mouth Aspect Ratio (float), or None if unavailable
        
        Returns:
            tuple: (is_active: bool, probability: float, smoothed_probability: float)
                - is_active: True if speaker is detected as active (speaking)
                - probability: Instantaneous speech probability [0, 1]
                - smoothed_probability: Smoothed probability with hold decay [0, 1]
        """
        if face_idx not in self._history:
            self._history[face_idx] = deque(maxlen=self.window_size)
            self._hold_counter[face_idx] = 0
            self._smoothed_prob[face_idx] = 0.0
        
        if mar_value is not None:
            self._history[face_idx].append(mar_value)
        
        history = self._history[face_idx]
        
        # Raw detection from MAR signal (delta-enhanced with ZCR)
        raw_speaking = False
        probability = 0.0
        if len(history) >= 4:
            values = np.array(history)
            mar_var = np.var(values)
            mar_mean = np.mean(values)
            
            # Compute zero-crossing rate of MAR derivative
            # Speech → rhythmic oscillation → high ZCR
            # Idle open mouth → stable MAR → low ZCR
            delta = np.diff(values)
            signs = np.sign(delta)
            zcr = int(np.sum(np.abs(np.diff(signs)) > 0))
            
            # All three conditions: variance + mean level + oscillation pattern
            raw_speaking = ((mar_var > self.var_threshold)
                            and (mar_mean > self.mar_activity_threshold)
                            and (zcr >= self.min_zcr))
            
            # Continuous probability
            probability = self._compute_probability(mar_var, mar_mean, zcr)
        
        # Smoothed probability (exponential decay, mirrors hold-time behavior)
        smoothed = max(probability, self._smoothed_prob[face_idx] * self._decay)
        self._smoothed_prob[face_idx] = smoothed
        
        # Hysteresis logic
        if raw_speaking:
            # Speech detected → reset hold timer to full duration
            self._hold_counter[face_idx] = self.hold_frames
        else:
            # No speech detected → decrement hold timer
            if self._hold_counter[face_idx] > 0:
                self._hold_counter[face_idx] -= 1
        
        # Active as long as hold timer hasn't expired
        is_active = self._hold_counter[face_idx] > 0
        return is_active, probability, smoothed
    
    def reset(self, face_idx=None):
        """Clear history for a specific face or all faces."""
        if face_idx is not None:
            self._history.pop(face_idx, None)
            self._hold_counter.pop(face_idx, None)
            self._smoothed_prob.pop(face_idx, None)
        else:
            self._history.clear()
            self._hold_counter.clear()
            self._smoothed_prob.clear()


# ═══════════════════════════════════════════════════════════════════════════════
# DNN-based Visual Voice Activity Detector
# ═══════════════════════════════════════════════════════════════════════════════

class VVAD_DNN(nn.Module):
    """
    Feedforward DNN for Visual Voice Activity Detection.

    Architecture must match the training script (train_vvad_dnn.py) exactly
    so that checkpoint weights can be loaded.
    """

    def __init__(self, input_dim, hidden_dims=None, dropout=0.3):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128]
        layers = []
        prev_dim = input_dim
        for i, h_dim in enumerate(hidden_dims):
            layers.append((f'fc{i+1}', nn.Linear(prev_dim, h_dim)))
            layers.append((f'bn{i+1}', nn.BatchNorm1d(h_dim)))
            layers.append((f'relu{i+1}', nn.ReLU()))
            layers.append((f'drop{i+1}', nn.Dropout(dropout)))
            prev_dim = h_dim
        layers.append(('fc_out', nn.Linear(prev_dim, 1)))
        self.network = nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        return self.network(x).squeeze(-1)


class DNN_VVAD_Detector:
    """
    DNN-based Visual Voice Activity Detector.

    Uses a trained feedforward neural network that takes frontalized and
    scale-normalized facial landmark coordinates from a sliding window of
    consecutive frames to predict whether a speaker is actively speaking.

    Preprocessing (identical to training pipeline):
        1. Face frontalization (derotation via stable reference landmarks)
        2. Centering (midpoint of eye centers as origin)
        3. Scale normalization (÷ inter-ocular distance)
        4. Z-score standardization (using training-set statistics from checkpoint)

    Features per sample:
        • Absolute positions:   window_size × 68 landmarks × 2 (x, y)
        • Frame-to-frame deltas: (window_size - 1) × 68 × 2

    Hysteresis:
        Once the detector transitions to ACTIVE (speaking), it maintains that
        state for at least `hold_seconds` to prevent flickering.

    Args:
        checkpoint_path: Path to the trained .pt model checkpoint.
        device: torch.device (auto-detected if None).
        hold_seconds: Minimum time to sustain ACTIVE state after last positive detection.
        fps: Video frame rate (used to convert hold_seconds to frames).
    """

    NUM_LANDMARKS = 68

    def __init__(self, checkpoint_path, device=None, hold_seconds=1.0, fps=30.0):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        cfg = checkpoint['config']

        self.window_size = cfg['window_size']
        self.num_landmarks = cfg['num_landmarks']

        # Standardization parameters (fit on training set)
        std_params = checkpoint['standardization']
        self.feat_mean = torch.FloatTensor(np.array(std_params['mean'])).to(self.device)
        self.feat_std = torch.FloatTensor(np.array(std_params['std'])).to(self.device)
        self.feat_std[self.feat_std < 1e-8] = 1.0  # avoid division by zero

        # Build and load model
        self.model = VVAD_DNN(
            input_dim=cfg['input_dim'],
            hidden_dims=cfg['hidden_dims'],
            dropout=cfg['dropout'],
        ).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        # Per-face sliding windows and hysteresis state
        self._landmark_windows = {}   # {face_idx: deque of (68, 2) arrays}
        self._hold_counter = {}       # {face_idx: int}
        self._smoothed_prob = {}      # {face_idx: float}

        self.hold_frames = max(1, int(hold_seconds * fps))
        self._decay = 1.0 - (1.0 / max(1, self.hold_frames))

        print(f"  DNN V-VAD loaded from: {checkpoint_path}")
        print(f"    Window size:   {self.window_size} frames")
        print(f"    Input dim:     {cfg['input_dim']}")
        print(f"    Hidden layers: {cfg['hidden_dims']}")
        print(f"    Model epoch:   {checkpoint.get('epoch', '?')}")
        print(f"    Val F1:        {checkpoint.get('val_f1', '?'):.4f}")

    @staticmethod
    def _frontalize_and_normalize(pts_x, pts_y, pts_z):
        """
        Frontalize and normalize all 68 facial landmarks.
        Identical to the preprocessing in train_vvad_dnn.py.

        Returns (68, 2) numpy array of normalized (x, y) or None on failure.
        """
        pts_3d = np.column_stack([pts_x, pts_y, pts_z])

        left_eye_center = (pts_3d[36] + pts_3d[39]) / 2.0
        right_eye_center = (pts_3d[42] + pts_3d[45]) / 2.0

        x_axis = right_eye_center - left_eye_center
        inter_ocular_dist = np.linalg.norm(x_axis)
        if inter_ocular_dist < 1e-8:
            return None
        x_axis = x_axis / inter_ocular_dist

        y_approx = pts_3d[8] - pts_3d[27]
        z_axis = np.cross(x_axis, y_approx)
        z_norm = np.linalg.norm(z_axis)
        if z_norm < 1e-8:
            return None
        z_axis = z_axis / z_norm

        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)

        R_inv = np.array([x_axis, y_axis, z_axis])
        face_center = (left_eye_center + right_eye_center) / 2.0

        centered = pts_3d - face_center
        derotated = (R_inv @ centered.T).T
        normalized_xy = derotated[:, :2] / inter_ocular_dist

        return normalized_xy  # shape (68, 2)

    def update(self, face_idx, points_3d):
        """
        Process one frame of landmarks and return speech activity prediction.

        Args:
            face_idx: Index of the face.
            points_3d: dict {pt_idx: (x, y, z)} for all available facial landmarks.

        Returns:
            tuple (is_active, probability, smoothed_probability)
        """
        if face_idx not in self._landmark_windows:
            self._landmark_windows[face_idx] = deque(maxlen=self.window_size)
            self._hold_counter[face_idx] = 0
            self._smoothed_prob[face_idx] = 0.0

        # Must have all 68 landmarks
        if len(points_3d) < self.NUM_LANDMARKS:
            return False, 0.0, self._smoothed_prob.get(face_idx, 0.0)

        pts_x = np.array([points_3d[i][0] for i in range(self.NUM_LANDMARKS)], dtype=np.float64)
        pts_y = np.array([points_3d[i][1] for i in range(self.NUM_LANDMARKS)], dtype=np.float64)
        pts_z = np.array([points_3d[i][2] for i in range(self.NUM_LANDMARKS)], dtype=np.float64)

        normalized = self._frontalize_and_normalize(pts_x, pts_y, pts_z)
        if normalized is None:
            return False, 0.0, self._smoothed_prob.get(face_idx, 0.0)

        self._landmark_windows[face_idx].append(normalized)

        # Need a full window for prediction
        window = self._landmark_windows[face_idx]
        probability = 0.0
        raw_speaking = False

        if len(window) == self.window_size:
            frames = list(window)

            # Absolute positions: each frame (68, 2) → flatten
            abs_feats = np.concatenate([f.flatten() for f in frames])

            # Frame-to-frame deltas
            deltas = []
            for j in range(1, len(frames)):
                deltas.append((frames[j] - frames[j - 1]).flatten())
            delta_feats = np.concatenate(deltas)

            feat_vector = np.concatenate([abs_feats, delta_feats]).astype(np.float32)

            # Standardize and run inference
            feat_tensor = torch.FloatTensor(feat_vector).unsqueeze(0).to(self.device)
            feat_tensor = (feat_tensor - self.feat_mean) / self.feat_std

            with torch.no_grad():
                logit = self.model(feat_tensor)
                probability = float(torch.sigmoid(logit).item())
                raw_speaking = probability > 0.5

        # Smoothed probability (exponential decay)
        smoothed = max(probability, self._smoothed_prob[face_idx] * self._decay)
        self._smoothed_prob[face_idx] = smoothed

        # Hysteresis (hold-time)
        if raw_speaking:
            self._hold_counter[face_idx] = self.hold_frames
        else:
            if self._hold_counter[face_idx] > 0:
                self._hold_counter[face_idx] -= 1

        is_active = self._hold_counter[face_idx] > 0
        return is_active, probability, smoothed

    def reset(self, face_idx=None):
        """Clear state for a specific face or all faces."""
        if face_idx is not None:
            self._landmark_windows.pop(face_idx, None)
            self._hold_counter.pop(face_idx, None)
            self._smoothed_prob.pop(face_idx, None)
        else:
            self._landmark_windows.clear()
            self._hold_counter.clear()
            self._smoothed_prob.clear()


def draw_facial_landmarks(frame, landmarks_df, timestamp, face_idx=0,
                          frontalize=True, vvad_detector=None,
                          speaking_override=None):
    """
    Draw facial landmarks on the frame with Visual VAD coloring.
    
    Color scheme:
        - White:  original (raw) mouth landmark positions
        - Red:    frontalized mouth landmarks — speaker INACTIVE (not speaking)
        - Green:  frontalized mouth landmarks — speaker ACTIVE (speaking)
    
    Args:
        frame: Video frame (BGR format)
        landmarks_df: DataFrame with landmark data (columns: seconds, face_idx, point_type, x, y, z)
        timestamp: Current frame timestamp
        face_idx: Index of the face to draw landmarks for
        frontalize: If True, apply frontalization to mouth landmarks
        vvad_detector: VisualVADDetector instance for speech activity detection.
                       If None, frontalized points default to green.
                       Ignored when *speaking_override* is not None.
        speaking_override: If not None, use this boolean instead of calling the
                           VVAD detector (to avoid double state updates when the
                           detector was already called externally).
    
    Returns:
        Modified frame
    """
    # Ensure frame is contiguous
    if not frame.flags['C_CONTIGUOUS']:
        frame = np.ascontiguousarray(frame)
    
    # Get landmarks for this frame and face
    frame_landmarks = landmarks_df[
        (landmarks_df['seconds'].round(6) == round(timestamp, 6)) &
        (landmarks_df['face_idx'] == face_idx)
    ]
    
    if frame_landmarks.empty:
        # Try to find closest timestamp
        idx = (landmarks_df['seconds'] - timestamp).abs().argmin()
        frame_landmarks = landmarks_df[
            (landmarks_df.iloc[idx]['seconds'] == landmarks_df.iloc[idx]['seconds']) &
            (landmarks_df['face_idx'] == face_idx)
        ]
    
    if frame_landmarks.empty:
        return frame
    
    # Convert landmarks to dictionaries (2D for drawing, 3D for frontalization)
    points_2d = {}
    points_3d = {}
    for _, row in frame_landmarks.iterrows():
        pt_idx = int(row['point_type'])
        points_2d[pt_idx] = (int(row['x']), int(row['y']))
        points_3d[pt_idx] = (float(row['x']), float(row['y']), float(row['z']))
    
    mouth_indices = set(range(48, 68))
    
    # --- Frontalization ---
    frontalized_points = None
    if frontalize:
        frontalized_points = frontalize_mouth_landmarks(points_3d)
    
    # --- Visual VAD: determine if speaker is active ---
    is_speaking = False
    speech_prob = 0.0
    speech_prob_smoothed = 0.0
    if speaking_override is not None:
        # Decision was already computed externally (avoids double state update)
        is_speaking = bool(speaking_override)
    elif vvad_detector is not None:
        if isinstance(vvad_detector, DNN_VVAD_Detector):
            # DNN-based V-VAD: pass all 68 landmark 3D coordinates
            if len(points_3d) >= 68:
                is_speaking, speech_prob, speech_prob_smoothed = vvad_detector.update(face_idx, points_3d)
        elif frontalized_points:
            # Heuristic MAR-based V-VAD (legacy fallback)
            mar = compute_mouth_aspect_ratio(frontalized_points)
            is_speaking, speech_prob, speech_prob_smoothed = vvad_detector.update(face_idx, mar)
    
    # --- Drawing ---
    point_radius = 2
    point_thickness = -1  # Filled
    
    # # 1) Original (raw) mouth landmarks — always White
    # original_color = (255, 255, 255)  # White (BGR)
    # for pt_idx, pt_coord in points_2d.items():
    #     if pt_idx in mouth_indices:
    #         cv2.circle(frame, pt_coord, point_radius, original_color, point_thickness, cv2.LINE_AA)
    
    # 2) Frontalized mouth landmarks — Green if speaking, Red if inactive
    #    Only outer mouth (48-59); inner mouth (60-67) disabled.
    if frontalized_points:
        if is_speaking:
            frontal_color = (0, 255, 0)    # Green (BGR) — ACTIVE
        else:
            frontal_color = (0, 0, 255)    # Red (BGR)   — INACTIVE
        
        for pt_idx, pt_coord in frontalized_points.items():
            # if pt_idx >= 60:          # inner mouth landmarks
            #     continue
            if pt_idx in range(60, 68):  # skip inner mouth landmarks
                continue
            cv2.circle(frame, pt_coord, point_radius, frontal_color, point_thickness, cv2.LINE_AA)
    
    return frame


def draw_pie_chart(frame, center_x, center_y, radius, data_dict, title, show_legend=True):
    """
    Draw a pie chart on the frame using OpenCV
    
    Args:
        frame: Video frame (numpy array)
        center_x: X coordinate of pie center
        center_y: Y coordinate of pie center
        radius: Radius of the pie chart
        data_dict: Dictionary with categories and percentages
                   e.g., {'near_talker': 45.2, 'far_talker': 28.5, ...}
        title: Title text to display above chart
        show_legend: Whether to show legend with percentages
    
    Returns:
        Modified frame
    """
    # Color scheme for audio categories (BGR format)
    colors = {
        'near_talker': (228, 119, 31),   # Blue
        'far_talker': (48, 172, 119),    # Green
        'noise': (25, 50, 220),          # Red
        'silence': (208, 224, 64),       # Turquoise
        'unknown': (180, 180, 180),      # Gray
    }
    
    # Category display names
    display_names = {
        'near_talker': 'NT',
        'far_talker': 'FT',
        'noise': 'Noise',
        'silence': 'Silence',
        'unknown': 'Unknown',
    }
    
    # Ensure frame is contiguous
    if not frame.flags['C_CONTIGUOUS']:
        frame = np.ascontiguousarray(frame)
    
    # Draw title
    font = cv2.FONT_HERSHEY_SIMPLEX
    title_font_scale = 0.55
    title_thickness = 2
    cv2.putText(frame, title, 
                (center_x - radius, center_y - radius - 10),
                font, title_font_scale, (255, 255, 255), title_thickness)
    
    # Calculate start angles for each slice
    start_angle = -90  # Start from top (12 o'clock position)
    
    # Draw pie slices
    category_order = ['near_talker', 'far_talker', 'noise', 'silence', 'unknown']
    for category in category_order:
        if category not in data_dict:
            continue
        
        percentage = data_dict[category]
        if percentage <= 0:
            continue
        
        # Calculate sweep angle
        sweep_angle = (percentage / 100.0) * 360
        
        # Draw filled ellipse (pie slice)
        color = colors.get(category, (127, 127, 127))
        cv2.ellipse(frame, 
                   (center_x, center_y), 
                   (radius, radius), 
                   0, 
                   start_angle, 
                   start_angle + sweep_angle, 
                   color, 
                   -1)  # Filled
        
        # Draw outline
        cv2.ellipse(frame, 
                   (center_x, center_y), 
                   (radius, radius), 
                   0, 
                   start_angle, 
                   start_angle + sweep_angle, 
                   (200, 200, 200), 
                   1)  # Outline
        
        start_angle += sweep_angle
    
    # Draw legend if requested
    if show_legend:
        legend_x = center_x + radius + 15
        legend_y = center_y - radius + 20
        legend_line_height = 20
        legend_font_scale = 0.45
        legend_thickness = 1
        
        for i, category in enumerate(category_order):
            if category not in data_dict:
                continue
            
            percentage = data_dict[category]
            color = colors.get(category, (127, 127, 127))
            y_pos = legend_y + (i * legend_line_height)
            
            # Draw color box
            cv2.rectangle(frame, 
                         (legend_x, y_pos - 8), 
                         (legend_x + 12, y_pos + 2), 
                         color, -1)
            cv2.rectangle(frame, 
                         (legend_x, y_pos - 8), 
                         (legend_x + 12, y_pos + 2), 
                         (200, 200, 200), 1)
            
            # Draw text
            text = f"{display_names[category]}: {percentage:.1f}%"
            cv2.putText(frame, text, 
                       (legend_x + 18, y_pos),
                       font, legend_font_scale, (255, 255, 255), legend_thickness)
    
    return frame


def draw_dashboard(frame, frame_idx, face_count, face_distances, timestamp,
                   audio_data=None, vad_data=None,
                   enrollment_collector=None, enrollment_duration=10.0,
                   face_avatars=None, active_targets=None,
                   visible_targets=None, mvad_data=None):
    """
    Draw dashboard overlay on frame
    
    Args:
        frame: Video frame (numpy array)
        frame_idx: Current frame number
        face_count: Number of faces detected
        face_distances: Dict mapping face_idx to distance in meters
        timestamp: Time in seconds
        audio_data: Dict with before/after BNR audio classification data
        vad_data: Dict with VAD history and current value
        enrollment_collector: EnrollmentCollector instance (or None)
        enrollment_duration: Target enrollment duration in seconds
        face_avatars: dict face_idx -> BGR avatar image (or None)
        active_targets: list of face_idx that are active this frame (or None)
        visible_targets: set of face_idx recently visible (current + 10 prev frames)
    """
    # Ensure frame is contiguous for OpenCV operations
    if not frame.flags['C_CONTIGUOUS']:
        frame = np.ascontiguousarray(frame)
    
    height, width = frame.shape[:2]
    
    # Dashboard dimensions - FIXED SIZE for consistent appearance
    panel_width = 260  # Narrower panel
    panel_margin = 20
    panel_x = width - panel_width - panel_margin
    panel_y = panel_margin
    
    # Fixed panel height (always constant for consistent appearance)
    # Calculate based on actual content:
    # Title: 30 + 35 = 65
    # Info lines (Frame, Time, Faces): 3 * 38 = 114
    # Padding + separator: 10 + 28 = 38
    # Face slots (2 faces): 2 * 38 = 76
    # Enrollment section: 10 + separator + 20 + title(25) + N*slot(62)
    # Audio section: 30 (separator) + 250 (pie charts) = 280
    # VAD section: 30 (separator) + 70 (graph) = 100
    # Bottom padding: 20
    line_height = 38
    # Determine how many enrollment slots to show (only recently-visible targets)
    _vis = visible_targets if visible_targets else set()
    if enrollment_collector is not None:
        _enroll_display = sorted(fid for fid in enrollment_collector.ever_seen if fid in _vis)
        _n_enroll_slots = len(_enroll_display)
    else:
        _enroll_display = []
        _n_enroll_slots = 0
    enroll_slot_h = 62  # avatar_dim(56) + 6
    enrollment_section_height = (10 + 22 + 25 + _n_enroll_slots * enroll_slot_h) if _n_enroll_slots > 0 else 0
    mvad_section_height = 100 if mvad_data is not None else 0
    panel_height = 65 + 114 + 38 + 76 + enrollment_section_height + 280 + mvad_section_height + 100 + 20
    
    # Draw solid background panel (more opaque for better visibility)
    overlay = frame.copy()
    cv2.rectangle(overlay, 
                  (panel_x, panel_y), 
                  (panel_x + panel_width, panel_y + panel_height),
                  (20, 20, 20),  # Darker background
                  -1)  # Filled
    
    # Draw panel border (brighter for better visibility)
    cv2.rectangle(overlay, 
                  (panel_x, panel_y), 
                  (panel_x + panel_width, panel_y + panel_height),
                  (220, 220, 220),  # Brighter border
                  3)  # Thicker border
    
    # Blend overlay with original frame (more opaque)
    alpha = 0.92
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    
    # Text parameters - larger for better visibility
    font = cv2.FONT_HERSHEY_SIMPLEX
    title_font_scale = 0.85
    text_font_scale = 0.65
    title_thickness = 2
    text_thickness = 2  # Bolder text
    text_color = (255, 255, 255)  # White
    
    # Current Y position for text
    current_y = panel_y + 30
    
    # Draw title
    cv2.putText(frame, "FACE TRACKING", 
                (panel_x + 10, current_y),
                font, title_font_scale, text_color, title_thickness)
    
    current_y += 35
    
    # Draw separator line
    cv2.line(frame, 
             (panel_x + 10, current_y), 
             (panel_x + panel_width - 10, current_y),
             (200, 200, 200), 1)
    
    current_y += 25
    
    # Draw frame info
    cv2.putText(frame, f"Frame: {frame_idx}", 
                (panel_x + 15, current_y),
                font, text_font_scale, text_color, text_thickness)
    
    current_y += line_height
    
    # Draw timestamp
    cv2.putText(frame, f"Time: {timestamp:.2f}s", 
                (panel_x + 15, current_y),
                font, text_font_scale, text_color, text_thickness)
    
    current_y += line_height
    
    # Draw face count
    count_color = (100, 255, 100) if face_count > 0 else (100, 100, 255)  # Green if faces, blue if none
    cv2.putText(frame, f"Faces: {face_count}", 
                (panel_x + 15, current_y),
                font, text_font_scale, count_color, text_thickness)
    
    current_y += 10
    
    # Draw separator line
    cv2.line(frame, 
             (panel_x + 10, current_y), 
             (panel_x + panel_width - 10, current_y),
             (220, 220, 220), 2)
    
    current_y += 28
    
    # Draw slots for up to 2 faces (fixed layout); face_distances keyed by track ID
    sorted_faces = sorted(face_distances.items())[:2]
    for slot_idx in range(2):
        if slot_idx < len(sorted_faces):
            face_idx, distance = sorted_faces[slot_idx]
            color_idx = face_idx % len(FACE_COLORS_BGR)
            face_color = brighten_color(FACE_COLORS_BGR[color_idx])
            cv2.circle(frame, (panel_x + 22, current_y - 6), 7, face_color, -1)
            dist_text = f"Face {face_idx}: {distance:.1f} m"
            cv2.putText(frame, dist_text, (panel_x + 40, current_y),
                       font, text_font_scale, text_color, text_thickness)
        else:
            gray_color = (100, 100, 100)
            cv2.circle(frame, (panel_x + 22, current_y - 6), 7, gray_color, -1)
            placeholder_text = f"Face {slot_idx}: ---"
            cv2.putText(frame, placeholder_text, (panel_x + 40, current_y),
                       font, text_font_scale, gray_color, text_thickness)
        current_y += line_height
    
    # ── Enrollment section ──────────────────────────────────────────
    if enrollment_collector is not None and _n_enroll_slots > 0:
        current_y += 10
        cv2.line(frame,
                 (panel_x + 10, current_y),
                 (panel_x + panel_width - 10, current_y),
                 (220, 220, 220), 2)
        current_y += 20

        cv2.putText(frame, "Enrollment",
                    (panel_x + 10, current_y),
                    font, title_font_scale, text_color, title_thickness)
        current_y += 25

        # Enrollment slot layout:
        # [10px] [avatar 56x56] [8px gap] [label / bar] [10px]
        avatar_dim = 56                                  # 80px * 0.7 ≈ 56
        enroll_bar_height = 24                           # taller than before
        enroll_label_h = 18
        bar_x_offset = 10 + avatar_dim + 8               # = 74
        enroll_bar_width = panel_width - bar_x_offset - 10  # narrower
        enroll_slot_height = avatar_dim + 6               # = 62

        sorted_targets = _enroll_display  # only recently-visible targets
        if face_avatars is None:
            face_avatars = {}
        for slot_idx in range(len(sorted_targets)):
            if slot_idx < len(sorted_targets):
                fid = sorted_targets[slot_idx]
                progress = enrollment_collector.get_progress(fid)
                completed = enrollment_collector.is_completed(fid)
                dur = enrollment_collector.get_duration(fid)

                # ── Avatar (circular thumbnail) ──────────────────────
                avatar_cx = panel_x + 10 + avatar_dim // 2
                avatar_cy = current_y + avatar_dim // 2
                avatar_radius = avatar_dim // 2

                if fid in face_avatars:
                    av_img = face_avatars[fid]
                    h_orig, w_orig = av_img.shape[:2]
                    av_h = int(avatar_dim * h_orig / w_orig)
                    av_w = avatar_dim
                    if av_h <= 0:
                        av_h = avatar_dim
                    av_resized = cv2.resize(av_img, (av_w, av_h),
                                            interpolation=cv2.INTER_AREA)
                    # Crop to square if needed (center crop)
                    if av_h > avatar_dim:
                        crop_y = (av_h - avatar_dim) // 2
                        av_resized = av_resized[crop_y:crop_y + avatar_dim, :, :]
                        av_h = avatar_dim
                    elif av_h < avatar_dim:
                        pad_top = (avatar_dim - av_h) // 2
                        pad_bot = avatar_dim - av_h - pad_top
                        av_resized = cv2.copyMakeBorder(
                            av_resized, pad_top, pad_bot, 0, 0,
                            cv2.BORDER_CONSTANT, value=(20, 20, 20))
                        av_h = avatar_dim

                    # Circular mask
                    mask = np.zeros((avatar_dim, avatar_dim), dtype=np.uint8)
                    cv2.circle(mask, (avatar_dim // 2, avatar_dim // 2),
                               avatar_radius, 255, -1, cv2.LINE_AA)

                    x1 = avatar_cx - avatar_radius
                    y1 = avatar_cy - avatar_radius
                    # Clamp to frame
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(frame.shape[1], x1 + avatar_dim)
                    y2 = min(frame.shape[0], y1 + avatar_dim)
                    rw, rh = x2 - x1, y2 - y1

                    roi = frame[y1:y2, x1:x2]
                    mask_roi = mask[:rh, :rw]
                    av_roi = av_resized[:rh, :rw]

                    alpha = (mask_roi / 255.0)[:, :, np.newaxis]
                    blended = (av_roi * alpha + roi * (1.0 - alpha)).astype(np.uint8)
                    frame[y1:y2, x1:x2] = blended

                    # Avatar border – colour indicates activity
                    # Yellow override when MVAD signals overlap
                    _at = active_targets if active_targets else []
                    _mvad_overlap = (mvad_data is not None
                                     and mvad_data.get('current_label') == 2)
                    if fid in _at:
                        if _mvad_overlap:
                            border_color = (0, 200, 255)   # orange — MVAD overlap
                        elif len(_at) == 1:
                            border_color = (0, 255, 0)     # green — single active
                        else:
                            border_color = (0, 255, 255)   # yellow — multi active
                        border_thick = 3
                    else:
                        border_color = (200, 200, 200)
                        border_thick = 1
                    cv2.circle(frame, (avatar_cx, avatar_cy),
                               avatar_radius + 1, border_color, border_thick, cv2.LINE_AA)
                else:
                    # Placeholder: empty circle
                    _at = active_targets if active_targets else []
                    _mvad_overlap = (mvad_data is not None
                                     and mvad_data.get('current_label') == 2)
                    if fid in _at:
                        if _mvad_overlap:
                            ph_border = (0, 200, 255)      # orange — MVAD overlap
                        elif len(_at) == 1:
                            ph_border = (0, 255, 0)
                        else:
                            ph_border = (0, 255, 255)
                        ph_thick = 3
                    else:
                        ph_border = (100, 100, 100)
                        ph_thick = 1
                    cv2.circle(frame, (avatar_cx, avatar_cy),
                               avatar_radius, (60, 60, 60), -1, cv2.LINE_AA)
                    cv2.circle(frame, (avatar_cx, avatar_cy),
                               avatar_radius + 1, ph_border, ph_thick, cv2.LINE_AA)
                    cv2.putText(frame, "?",
                                (avatar_cx - 6, avatar_cy + 6),
                                font, 0.55, (100, 100, 100), 1, cv2.LINE_AA)

                # ── Label text (above bar, right of avatar) ──────────
                color_idx = fid % len(FACE_COLORS_BGR)
                label_color = brighten_color(FACE_COLORS_BGR[color_idx])

                label = f"Talker{fid}: {dur:.1f}s / {enrollment_duration:.0f}s"
                text_x = panel_x + bar_x_offset
                # Vertically centre label+bar within avatar height
                label_baseline = current_y + (avatar_dim // 2) - (enroll_bar_height // 2) - 3
                cv2.putText(frame, label,
                            (text_x, label_baseline),
                            font, 0.52, label_color, 1, cv2.LINE_AA)

                # ── Progress bar (below label, right of avatar) ──────
                bar_y = label_baseline + 4
                bar_x = text_x

                # Background
                cv2.rectangle(frame, (bar_x, bar_y),
                              (bar_x + enroll_bar_width, bar_y + enroll_bar_height),
                              (50, 50, 50), -1)
                # Progress fill (same colour as label; green when done)
                fill_w = int(progress * enroll_bar_width)
                if fill_w > 0:
                    fill_color = (0, 200, 0) if completed else label_color
                    cv2.rectangle(frame, (bar_x, bar_y),
                                  (bar_x + fill_w, bar_y + enroll_bar_height),
                                  fill_color, -1)
                # Border
                cv2.rectangle(frame, (bar_x, bar_y),
                              (bar_x + enroll_bar_width, bar_y + enroll_bar_height),
                              (200, 200, 200), 1)

            current_y += enroll_slot_height

    # Draw Audio section separator
    current_y += 10
    cv2.line(frame, 
             (panel_x + 10, current_y), 
             (panel_x + panel_width - 10, current_y),
             (220, 220, 220), 2)
    
    current_y += 28
    
    # Draw AUDIO title
    cv2.putText(frame, "Audio Metrics", 
                (panel_x + 10, current_y),
                font, title_font_scale, text_color, title_thickness)
    
    current_y += 35  # Increased spacing before first pie chart
    
    # Define pie chart dimensions
    pie_radius = 45
    pie_center_x = panel_x + pie_radius + 10
    
    # Draw audio pie charts if data is available
    if audio_data is not None:
        
        # Color scheme for audio categories (BGR format)
        colors = {
            'near_talker': (228, 119, 31),   # Blue
            'far_talker': (48, 172, 119),    # Green
            'noise': (25, 50, 220),          # Red
            'silence': (208, 224, 64),       # Turquoise
            'unknown': (180, 180, 180),      # Gray
        }
        
        # Category display names
        display_names = {
            'near_talker': 'NT',
            'far_talker': 'FT',
            'noise': 'Noise',
            'silence': 'Silence',
            'unknown': 'Unknown',
        }
        
        category_order = ['near_talker', 'far_talker', 'noise', 'silence', 'unknown']
        
        # Draw "BNR Input" pie chart with legend
        pie_center_y = current_y + pie_radius
        frame = draw_pie_chart(frame, pie_center_x, pie_center_y, pie_radius, 
                               audio_data['before_bnr'], "BNR Input", show_legend=False)
        
        # Draw legend for BNR Input - Layout: pie | percentages | legend names
        percentages_x = pie_center_x + pie_radius + 5
        legend_x = percentages_x + 44  # Moved right from 38
        legend_y = pie_center_y - pie_radius + 10
        legend_line_height = 20  # Reduced for smaller font
        legend_font_scale = 0.64  # Reduced by 20% from 0.8
        legend_thickness = 1  # Thinner for smaller font
        
        for i, category in enumerate(category_order):
            color = colors.get(category, (127, 127, 127))
            y_pos = legend_y + (i * legend_line_height)
            percentage = audio_data['before_bnr'].get(category, 0)
            
            # Adjust text baseline to align with box center
            text_y = y_pos + 5  # Move text down to align with box
            
            # Draw percentage (rounded to integer)
            percentage_text = f"{percentage:.0f}%"
            cv2.putText(frame, percentage_text, 
                       (percentages_x, text_y),
                       font, legend_font_scale, (200, 200, 200), legend_thickness)
            
            # Draw color box (smaller to match reduced font)
            box_size = 13  # Reduced from 16
            cv2.rectangle(frame, 
                         (legend_x, y_pos - 6), 
                         (legend_x + box_size, y_pos + 7), 
                         color, -1)
            cv2.rectangle(frame, 
                         (legend_x, y_pos - 6), 
                         (legend_x + box_size, y_pos + 7), 
                         (200, 200, 200), 1)
            
            # Draw legend name
            text = f"{display_names[category]}"
            cv2.putText(frame, text, 
                       (legend_x + box_size + 6, text_y),
                       font, legend_font_scale, (200, 200, 200), legend_thickness)
        
        current_y += pie_radius * 2 + 35
        
        # Draw "System Output" pie chart with legend
        pie_center_y = current_y + pie_radius
        frame = draw_pie_chart(frame, pie_center_x, pie_center_y, pie_radius, 
                               audio_data['after_bnr'], "System Output", show_legend=False)
        
        # Draw legend for System Output - Layout: pie | percentages | legend names
        legend_y = pie_center_y - pie_radius + 10
        
        for i, category in enumerate(category_order):
            color = colors.get(category, (127, 127, 127))
            y_pos = legend_y + (i * legend_line_height)
            percentage = audio_data['after_bnr'].get(category, 0)
            text_y = y_pos + 5
            # Draw percentage (rounded to integer)
            percentage_text = f"{percentage:.0f}%"
            cv2.putText(frame, percentage_text, 
                       (percentages_x, text_y),
                       font, legend_font_scale, (200, 200, 200), legend_thickness)
            
            # Draw color box (smaller to match reduced font)
            box_size = 13  # Reduced from 16
            cv2.rectangle(frame, 
                         (legend_x, y_pos - 8), 
                         (legend_x + box_size, y_pos + 5), 
                         color, -1)
            cv2.rectangle(frame, 
                         (legend_x, y_pos - 8), 
                         (legend_x + box_size, y_pos + 5), 
                         (200, 200, 200), 1)
            
            # Draw legend name
            text = f"{display_names[category]}"
            cv2.putText(frame, text, 
                       (legend_x + box_size + 6, text_y),
                       font, legend_font_scale, (200, 200, 200), legend_thickness)
    
    # Advance past the second pie chart area (if audio was drawn)
    if audio_data is not None:
        current_y += pie_radius * 2 + 40

    # Draw Multivoice VAD (MVAD) overlap indicator section
    if mvad_data is not None:
        cv2.line(frame,
                 (panel_x + 10, current_y),
                 (panel_x + panel_width - 10, current_y),
                 (220, 220, 220), 2)
        current_y += 20

        graph_width = panel_width - 20
        graph_height = 70
        graph_x = panel_x + 10
        graph_y = current_y - 10

        frame = draw_mvad_timeline(
            frame, graph_x, graph_y, graph_width, graph_height,
            mvad_data['history'], "Multivoice VAD",
            mvad_data['current_label']
        )
        current_y += graph_height

    # Draw VAD (Voice Activity Detection) graph section
    if vad_data is not None:
        # If MVAD was not drawn but audio was, still need to advance past pie charts
        if mvad_data is None and audio_data is not None:
            current_y += pie_radius * 2 + 40

        # Draw separator
        cv2.line(frame,
                 (panel_x + 10, current_y),
                 (panel_x + panel_width - 10, current_y),
                 (220, 220, 220), 2)

        current_y += 20

        # Draw VAD graph
        graph_width = panel_width - 20
        graph_height = 70  # Reduced height since we're not showing the value
        graph_x = panel_x + 10
        graph_y = current_y - 10

        frame = draw_line_graph(
            frame,
            graph_x,
            graph_y,
            graph_width,
            graph_height,
            vad_data['history'],
            "Voice Activity",
            show_value=False
        )

    return frame


def draw_face_boxes(frame, face_data, timestamp, debug_mode=False):
    """
    Draw bounding boxes around detected faces
    
    Args:
        face_data: DataFrame with face position data for current timestamp
        timestamp: Current time in seconds
        debug_mode: If True, draw face index labels (F0, F1, ...) next to faces
    """
    # Ensure frame is contiguous for OpenCV operations
    if not frame.flags['C_CONTIGUOUS']:
        frame = np.ascontiguousarray(frame)
    
    # Get unique faces at this timestamp
    faces_at_time = face_data[face_data['seconds'] == timestamp]
    
    if faces_at_time.empty:
        return frame
    
    # For each face, we need to estimate position from mouth point 63
    # This is approximate since we don't have bbox from CSV
    # We'll draw a marker at the face position instead
    
    for face_idx in faces_at_time['face_idx'].unique():
        face_subset = faces_at_time[faces_at_time['face_idx'] == face_idx]
        
        # Get brightened color for this face (consistent with dashboard)
        color_idx = int(face_idx) % len(FACE_COLORS_BGR)
        color = brighten_color(FACE_COLORS_BGR[color_idx])
        
        # If we have x,y coordinates, draw a marker
        if not face_subset.empty and 'x' in face_subset.columns:
            x = int(face_subset['x'].values[0])
            y = int(face_subset['y'].values[0])
            
            # Draw circle marker
            # cv2.circle(frame, (x, y), 8, color, 2)
            # cv2.circle(frame, (x, y), 3, color, -1)
            
            # Draw face label above marker (debug only)
            if debug_mode:
                label = f"F{int(face_idx)}"
                cv2.putText(frame, label, 
                           (x + 70, y - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    return frame


def load_speaker_identification(csv_path, reference_seconds=None):
    """
    Load speaker identification data and build a fast lookup dictionary.

    The speaker_identification.csv may use a slightly different time base than
    the other dumps (face_count.csv, mouth_position.csv).  When
    *reference_seconds* is provided (sorted array of canonical timestamps from
    face_count.csv), each speaker timestamp is snapped to the nearest
    reference timestamp so that downstream exact-match lookups work.

    Args:
        csv_path: path to speaker_identification.csv
        reference_seconds: optional 1-D sorted array of canonical timestamps

    Returns:
        dict mapping (rounded_seconds, face_idx) -> speaker_name
    """
    df = pd.read_csv(csv_path)

    ref = None
    if reference_seconds is not None:
        ref = np.asarray(reference_seconds, dtype=np.float64)

    lookup = {}
    for _, row in df.iterrows():
        sec = float(row['seconds'])
        if ref is not None and len(ref) > 0:
            idx = np.searchsorted(ref, sec)
            candidates = []
            if idx > 0:
                candidates.append(ref[idx - 1])
            if idx < len(ref):
                candidates.append(ref[idx])
            nearest = min(candidates, key=lambda x: abs(x - sec))
            sec = float(nearest)
        key = (round(sec, 6), int(row['face_idx']))
        lookup[key] = str(row['speaker'])
    return lookup


def load_avatars(avatars_dir):
    """
    Load avatar images from a directory, keyed by speaker name (filename stem).

    Returns:
        dict mapping speaker_name -> BGR numpy array
    """
    avatars = {}
    avatars_path = Path(avatars_dir)
    if not avatars_path.exists():
        print(f"  Warning: avatars directory not found: {avatars_path}")
        return avatars
    for img_path in avatars_path.iterdir():
        if img_path.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp'):
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img is not None:
                avatars[img_path.stem] = img
    return avatars


def reindex_speaker_lookup(speaker_lookup, assignment_map):
    """
    Remap face_idx in the speaker lookup using the embedding-based assignment map,
    so that speaker identities align with the stable track IDs.
    """
    remapped = {}
    for (sec, face_idx), speaker in speaker_lookup.items():
        new_face_idx = assignment_map.get((sec, face_idx), face_idx)
        remapped[(sec, int(new_face_idx))] = speaker
    return remapped


def draw_speaker_avatar(frame, avatar_img, face_landmarks_df,
                        avatar_size=80, height_scale=1.1):
    """
    Overlay a circular miniature avatar image above a detected face.

    Positioning: compute |chin→nose_bridge| * height_scale + avatar_radius,
    then place the avatar centre that many pixels *straight up* from the
    nose bridge.  If the avatar would be cropped at the top of the frame,
    it is clamped so it stays fully visible.

    Args:
        frame: Video frame (BGR, must be contiguous)
        avatar_img: Speaker avatar image (BGR)
        face_landmarks_df: DataFrame rows for all landmarks of this face/timestamp
        avatar_size: Fixed avatar width in pixels (independent of face size)
        height_scale: Multiplier on |chin→nose_bridge| for the upward offset
    """
    pt = {}
    for _, row in face_landmarks_df.iterrows():
        pt[int(row['point_type'])] = (float(row['x']), float(row['y']))

    if 8 not in pt or 27 not in pt:
        return frame

    chin = np.array(pt[8], dtype=np.float64)
    nose_bridge = np.array(pt[27], dtype=np.float64)

    chin_to_bridge = nose_bridge - chin
    vec_len = np.linalg.norm(chin_to_bridge)
    if vec_len < 1.0:
        return frame

    avatar_w = max(20, avatar_size)
    h_orig, w_orig = avatar_img.shape[:2]
    avatar_h = int(avatar_w * h_orig / w_orig)
    if avatar_h <= 0:
        return frame

    avatar_resized = cv2.resize(avatar_img, (avatar_w, avatar_h),
                                interpolation=cv2.INTER_AREA)

    mask = np.zeros((avatar_h, avatar_w), dtype=np.uint8)
    cx, cy = avatar_w // 2, avatar_h // 2
    radius = min(avatar_w, avatar_h) // 2
    cv2.circle(mask, (cx, cy), radius, 255, -1, cv2.LINE_AA)

    # Offset = |chin→bridge| * height_scale + radius, applied straight up
    offset = vec_len * height_scale + radius
    avatar_cx = int(round(nose_bridge[0]))
    avatar_cy = int(round(nose_bridge[1] - offset))

    frame_h, frame_w = frame.shape[:2]

    # Clamp so the avatar stays fully inside the frame
    avatar_cy = max(avatar_h // 2, min(avatar_cy, frame_h - 1 - avatar_h // 2))
    avatar_cx = max(avatar_w // 2, min(avatar_cx, frame_w - 1 - avatar_w // 2))

    x1 = avatar_cx - avatar_w // 2
    y1 = avatar_cy - avatar_h // 2

    roi = frame[y1:y1 + avatar_h, x1:x1 + avatar_w]
    mask_roi = mask[:roi.shape[0], :roi.shape[1]]
    avatar_roi = avatar_resized[:roi.shape[0], :roi.shape[1]]

    alpha = (mask_roi / 255.0)[:, :, np.newaxis]
    blended = (avatar_roi * alpha + roi * (1.0 - alpha)).astype(np.uint8)
    frame[y1:y1 + avatar_h, x1:x1 + avatar_w] = blended

    cv2.circle(frame, (avatar_cx, avatar_cy), radius + 2,
               (255, 255, 255), 2, cv2.LINE_AA)

    return frame


class EnrollmentCollector:
    """
    Accumulates enrollment audio for each target talker independently.

    For every target talker (face within *target_distance* of the camera),
    audio frames are appended when both the smoothed Audio VAD and the
    Video VAD indicate speech — unless multiple target talkers are
    simultaneously active (overlap), in which case no audio is collected
    for any of them.

    Once *target_duration_s* seconds of speech have been collected for a
    given face, that face is marked as completed.
    """

    def __init__(self, target_duration_s=10.0, sample_rate=48000):
        self.target_duration = target_duration_s
        self.sample_rate = sample_rate
        self.target_samples = int(target_duration_s * sample_rate)
        self._collected = {}        # face_idx -> list[np.ndarray]
        self._total_samples = {}    # face_idx -> int
        self._completed = {}        # face_idx -> bool
        self.ever_seen = set()      # all face_idx ever registered

    def register_target(self, face_idx):
        """Register *face_idx* as a target talker (idempotent)."""
        self.ever_seen.add(face_idx)
        if face_idx not in self._collected:
            self._collected[face_idx] = []
            self._total_samples[face_idx] = 0
            self._completed[face_idx] = False

    def add_audio(self, face_idx, samples):
        """Append audio *samples* for *face_idx*. Truncates at target duration."""
        if face_idx not in self._collected:
            self.register_target(face_idx)
        if self._completed[face_idx]:
            return

        remaining = self.target_samples - self._total_samples[face_idx]
        if remaining <= 0:
            self._completed[face_idx] = True
            return

        chunk = samples[:remaining]
        self._collected[face_idx].append(chunk.copy())
        self._total_samples[face_idx] += len(chunk)

        if self._total_samples[face_idx] >= self.target_samples:
            self._completed[face_idx] = True

    def is_completed(self, face_idx):
        """Return True if enrollment target has been reached."""
        return self._completed.get(face_idx, False)

    def get_progress(self, face_idx):
        """Return enrollment progress as a fraction in [0, 1]."""
        if face_idx not in self._total_samples:
            return 0.0
        return min(1.0, self._total_samples[face_idx] / self.target_samples)

    def get_duration(self, face_idx):
        """Return collected duration in seconds."""
        return self._total_samples.get(face_idx, 0) / self.sample_rate

    def get_audio(self, face_idx):
        """Return concatenated audio array, or None if nothing was collected."""
        if face_idx not in self._collected or not self._collected[face_idx]:
            return None
        return np.concatenate(self._collected[face_idx])


def draw_enrollment_progress_bars(frame, enrollment_collector, target_duration,
                                  x_start=20, y_start=20):
    """
    Draw an enrollment progress bar for every registered target talker.

    Each bar shows the amount of collected enrollment audio relative to
    *target_duration*.  Bars are drawn in the top-left area of the frame.
    """
    if not frame.flags['C_CONTIGUOUS']:
        frame = np.ascontiguousarray(frame)

    font = cv2.FONT_HERSHEY_SIMPLEX
    bar_width = 200
    bar_height = 18
    label_height = 18
    spacing = bar_height + label_height + 8

    sorted_targets = sorted(enrollment_collector.ever_seen)

    for i, face_idx in enumerate(sorted_targets):
        y = y_start + i * spacing
        progress = enrollment_collector.get_progress(face_idx)
        completed = enrollment_collector.is_completed(face_idx)
        duration = enrollment_collector.get_duration(face_idx)

        # Label above bar
        label = f"TT{face_idx}: {duration:.1f}s / {target_duration:.0f}s"
        if completed:
            label += " Done"
        color_idx = face_idx % len(FACE_COLORS_BGR)
        label_color = FACE_COLORS_BGR[color_idx]
        cv2.putText(frame, label, (x_start, y + label_height - 4),
                    font, 0.5, label_color, 1, cv2.LINE_AA)

        bar_y = y + label_height

        # Dark background
        cv2.rectangle(frame, (x_start, bar_y),
                      (x_start + bar_width, bar_y + bar_height),
                      (50, 50, 50), -1)

        # Progress fill
        fill_width = int(progress * bar_width)
        if fill_width > 0:
            fill_color = (0, 200, 0) if completed else FACE_COLORS_BGR[color_idx]
            cv2.rectangle(frame, (x_start, bar_y),
                          (x_start + fill_width, bar_y + bar_height),
                          fill_color, -1)

        # Border
        cv2.rectangle(frame, (x_start, bar_y),
                      (x_start + bar_width, bar_y + bar_height),
                      (200, 200, 200), 1)

    return frame


def copy_audio_with_ffmpeg(input_video, temp_video, final_output):
    """
    Copy audio from input video to rendered video using ffmpeg
    
    Args:
        input_video: Original video with audio
        temp_video: Rendered video without audio
        final_output: Output path for final video with audio
    """
    try:
        # Check if ffmpeg is available
        subprocess.run(['ffmpeg', '-version'], 
                      stdout=subprocess.DEVNULL, 
                      stderr=subprocess.DEVNULL, 
                      check=True)
        
        print("\nCopying audio from original video...")
        
        # Use ffmpeg to merge video and audio
        # -i temp_video: input video (no audio)
        # -i input_video: input video (with audio)
        # -map 0:v:0: take video from first input
        # -map 1:a:0?: take audio from second input (if exists)
        # -c:v copy: copy video without re-encoding
        # -c:a aac: encode audio as AAC
        # -shortest: match duration of shortest stream
        cmd = [
            'ffmpeg',
            '-i', str(temp_video),      # Video without audio
            '-i', str(input_video),      # Original with audio
            '-map', '0:v:0',             # Video from first input
            '-map', '1:a:0?',            # Audio from second input (optional)
            '-c:v', 'copy',              # Copy video stream
            '-c:a', 'aac',               # Encode audio as AAC
            '-b:a', '192k',              # Audio bitrate
            '-shortest',                  # Match shortest stream
            '-y',                        # Overwrite output
            str(final_output)
        ]
        
        result = subprocess.run(cmd, 
                               stdout=subprocess.DEVNULL, 
                               stderr=subprocess.PIPE,
                               text=True)
        
        if result.returncode == 0:
            print("✓ Audio copied successfully")
            # Remove temporary file
            os.remove(temp_video)
            return True
        else:
            print("⚠ Warning: Could not copy audio (video may not have audio track)")
            # If ffmpeg failed, just rename temp to final
            os.rename(temp_video, final_output)
            return False
            
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⚠ Warning: ffmpeg not found. Video will have no audio.")
        print("  Install ffmpeg to enable audio copying: apt install ffmpeg / brew install ffmpeg")
        # Rename temp file to final output
        os.rename(temp_video, final_output)
        return False


def render_video_with_dashboard(video_path, df_face_count, df_mouth, df_mqe, df_vad, output_path,
                                show_markers=True, speaker_lookup=None, avatars=None, vvad_model_path=None,
                                audio_vad_pad_ms=250.0, vvad_pad_ms=0.0,
                                enrollment_duration=10.0, target_distance=2.0,
                                debug_mode=False, enrollment_wav_path=None,
                                mvad_labels=None, mvad_hop_sec=None):
    """
    Render annotated video with dashboard overlay and speaker avatars.
    
    Args:
        video_path: Path to input video
        df_face_count: DataFrame with face count data
        df_mouth: DataFrame with facial landmark data (all 68 points including x,y,z coordinates)
        df_mqe: DataFrame with MQE metrics (5-second segments)
        df_vad: DataFrame with VAD (Voice Activity Detection) data
        output_path: Path to output video
        show_markers: Whether to show facial landmark markers
        speaker_lookup: dict (seconds, face_idx) -> speaker_name (from speaker_identification.csv)
        avatars: dict speaker_name -> BGR image (loaded from enrollment-avatars/)
        vvad_model_path: Path to trained V-VAD DNN model (.pt). If None or missing, use MAR heuristic.
        audio_vad_pad_ms: Audio VAD smoothing — milliseconds of padding added on
                          each side of every active block (default 250 ms).
        vvad_pad_ms: Video VAD smoothing — milliseconds of padding added on each
                     side of every active block (default 0 ms = inactive).
                     NOTE: non-zero values would require a two-pass approach;
                     currently the DNN hold-time provides the only temporal
                     smoothing for Video VAD.
        enrollment_duration: Target enrollment audio length in seconds (default 10).
        target_distance: Maximum distance (meters) for a face to be considered
                         a target talker (default 2.0).
        enrollment_wav_path: Path to WAV file used as source audio for enrollment
                             (e.g. the *_txfeOut.wav from dumps/). If None,
                             enrollment audio collection is disabled.
        mvad_labels: 1-D int32 array of Multivoice VAD labels (0=silence,
                     1=single, 2=overlap) loaded from dumps/mvad_dnn.npz.
                     When label==2, enrollment audio collection is blocked.
        mvad_hop_sec: Time step in seconds between consecutive MVAD labels
                      (typically 0.01 for 10 ms hop).
    """
    print(f"\nReading video: {video_path}")
    reader = imageio.get_reader(video_path)
    metadata = reader.get_meta_data()
    fps = metadata['fps']
    
    print(f"Video FPS: {fps}")
    print(f"Total frames: {len(df_face_count)}")
    print(f"Output: {output_path}")
    
    # Create temporary output path for video without audio
    output_path = Path(output_path)
    temp_output = output_path.parent / (output_path.stem + '_temp' + output_path.suffix)
    
    # Initialize video writer (without audio)
    writer = imageio.get_writer(
        str(temp_output),
        fps=fps,
        codec='libx264',
        quality=8,
        pixelformat='yuv420p',
        macro_block_size=1
    )
    
    print("\nRendering annotated video...")
    
    # Initialize Visual VAD detector (DNN if model exists, else MAR heuristic)
    if vvad_model_path and os.path.exists(vvad_model_path):
        print(f"  Loading DNN-based V-VAD model...")
        vvad_detector = DNN_VVAD_Detector(vvad_model_path, hold_seconds=1.0, fps=fps)
    else:
        if vvad_model_path:
            print(f"  ⚠ DNN model not found at '{vvad_model_path}', falling back to MAR heuristic")
        print(f"  Using heuristic MAR-based V-VAD")
        vvad_detector = VisualVADDetector(window_seconds=0.3, fps=fps,
                                          var_threshold=0.002, mar_activity_threshold=0.15,
                                          hold_seconds=1.0, min_zcr=3)
    
    # ── Enrollment: load source audio and precompute smoothed Audio VAD ──
    audio_samples = None
    audio_sr = None
    enrollment_collector = None
    smoothed_audio_vad = None

    wav_path = Path(enrollment_wav_path) if enrollment_wav_path else None
    if sf is not None and wav_path is not None and wav_path.exists():
        audio_samples, audio_sr = sf.read(str(wav_path), dtype='float32')
        if audio_samples.ndim > 1:
            audio_samples = audio_samples[:, 0]  # mono
        print(f"  Source audio loaded: {wav_path.name} "
              f"({len(audio_samples)/audio_sr:.1f}s, {audio_sr} Hz)")

        # Video-frame timestamps used for alignment
        video_timestamps = np.array([i / fps for i in range(len(df_face_count))])

        # Smoothed Audio VAD (morphological dilation by audio_vad_pad_ms)
        smoothed_audio_vad = precompute_smoothed_audio_vad(
            df_vad, video_timestamps, fps, pad_ms=audio_vad_pad_ms
        )
        print(f"  Audio VAD smoothed (pad={audio_vad_pad_ms:.0f} ms): "
              f"{int(smoothed_audio_vad.sum())} / {len(smoothed_audio_vad)} frames active")

        enrollment_collector = EnrollmentCollector(
            target_duration_s=enrollment_duration,
            sample_rate=audio_sr,
        )
        print(f"  Enrollment collector ready: target {enrollment_duration:.0f}s per talker, "
              f"max distance {target_distance:.1f}m")
    elif sf is None:
        print(f"  ⚠ 'soundfile' module not installed; enrollment audio disabled")
    else:
        if wav_path is not None:
            print(f"  ⚠ Enrollment WAV not found ({wav_path}); enrollment audio disabled")
        else:
            print(f"  ⚠ No enrollment WAV path provided; enrollment audio disabled")

    # Store last valid audio data to persist after MQE data ends
    last_audio_data = None

    # Persistent mapping face_idx → avatar image (accumulated over frames)
    persistent_face_avatars = {}
    # Majority-vote tracker: {face_idx: {speaker_name: frame_count}}
    # Prevents avatar flickering when speaker identification is noisy.
    face_speaker_votes = {}

    # Rolling window of target-talker face indices visible per frame
    # (current frame + up to 10 previous).  Used to hide enrollment
    # slots for faces that haven't been seen recently.
    recent_target_window = deque(maxlen=11)

    # Process each frame
    for frame_idx, frame in enumerate(tqdm(reader, total=len(df_face_count))):
        # RGB to BGR for OpenCV - ensure contiguous array
        frame_bgr = np.ascontiguousarray(frame[..., ::-1])
        
        # Get timestamp for this frame
        timestamp = frame_idx / fps
        
        # Get face count for this frame
        face_count_row = df_face_count[df_face_count['seconds'].round(6) == round(timestamp, 6)]
        if not face_count_row.empty:
            face_count = int(face_count_row['face_count'].values[0])
        else:
            # Find closest timestamp
            idx = (df_face_count['seconds'] - timestamp).abs().argmin()
            face_count = int(df_face_count.iloc[idx]['face_count'])
        
        # Get face distances for this frame
        face_distances = {}
        
        # Get mouth data for current timestamp (point 63)
        mouth_data = df_mouth[
            (df_mouth['seconds'].round(6) == round(timestamp, 6)) & 
            (df_mouth['point_type'] == 63)
        ]
        
        if mouth_data.empty:
            # Find closest timestamp
            closest_idx = (df_mouth['seconds'] - timestamp).abs().argmin()
            closest_time = df_mouth.iloc[closest_idx]['seconds']
            mouth_data = df_mouth[
                (df_mouth['seconds'] == closest_time) & 
                (df_mouth['point_type'] == 63)
            ]
        
        for _, row in mouth_data.iterrows():
            face_idx = int(row['face_idx'])
            z_value = row['z']
            distance = calculate_distance(z_value)
            face_distances[face_idx] = distance
        
        # Draw face position markers (if enabled and data available)
        if show_markers and not mouth_data.empty:
            # Create a temporary dataframe with position data
            frame_bgr = draw_face_boxes(frame_bgr, mouth_data, timestamp, debug_mode=debug_mode)
        
        # Get MQE metrics data - cumulative from start to current time
        audio_data = None
        if df_mqe is not None and not df_mqe.empty:
            # Calculate which 5-second segment this timestamp belongs to
            segment_idx = int(timestamp / 5)
            
            # Accumulate all segments from start up to current segment
            if segment_idx < len(df_mqe):
                # Sum up all segments from 0 to segment_idx (inclusive)
                segments = df_mqe.iloc[:segment_idx + 1]
                
                # Calculate total accumulated durations
                total_nt_in = segments['nearTalkerIn'].sum()
                total_ft_in = segments['farTalkerIn'].sum()
                total_noise_in = segments['noiseIn'].sum()
                total_silence_in = segments['silenceIn'].sum()
                
                total_nt_out = segments['nearTalkerOut'].sum()
                total_ft_out = segments['farTalkerOut'].sum()
                total_noise_out = segments['noiseOut'].sum()
                total_silence_out = segments['silenceOut'].sum()
                
                # Total elapsed time (all segments have same duration)
                total_duration = len(segments) * segments.iloc[0]['segmentSizeSec']
                
                # Convert accumulated durations to percentages
                before_bnr = {
                    'near_talker': (total_nt_in / total_duration) * 100,
                    'far_talker': (total_ft_in / total_duration) * 100,
                    'noise': (total_noise_in / total_duration) * 100,
                    'silence': (total_silence_in / total_duration) * 100,
                }
                after_bnr = {
                    'near_talker': (total_nt_out / total_duration) * 100,
                    'far_talker': (total_ft_out / total_duration) * 100,
                    'noise': (total_noise_out / total_duration) * 100,
                    'silence': (total_silence_out / total_duration) * 100,
                }
                
                # Calculate "unknown" as the remainder to 100%
                sum_before = sum(before_bnr.values())
                sum_after = sum(after_bnr.values())
                before_bnr['unknown'] = max(0, 100 - sum_before)
                after_bnr['unknown'] = max(0, 100 - sum_after)
                
                audio_data = {
                    'before_bnr': before_bnr,
                    'after_bnr': after_bnr
                }
                
                # Store this as the last valid data
                last_audio_data = audio_data
            else:
                # Use last valid data if we're past the MQE segments
                audio_data = last_audio_data
        
        # Get VAD data for visualization (show last 5 seconds of history)
        vad_data = None
        if df_vad is not None and not df_vad.empty:
            # Get VAD history for last 5 seconds
            history_window = 5.0  # seconds
            history_start = max(0, timestamp - history_window)
            
            vad_history = df_vad[
                (df_vad['seconds'] >= history_start) & 
                (df_vad['seconds'] <= timestamp)
            ]
            
            if not vad_history.empty:
                # Get VAD values for plotting
                vad_values = vad_history['vadDagcDecFinal'].tolist()
                
                # Calculate percentage of voice activity in recent window
                vad_percentage = (sum(vad_values) / len(vad_values)) * 100 if vad_values else 0
                
                vad_data = {
                    'history': vad_values,
                    'percentage': vad_percentage
                }
        
        # Update persistent face_idx → avatar mapping (majority-vote)
        # Each frame casts a "vote" for the recognised speaker; the avatar
        # always corresponds to the speaker with the most cumulative votes.
        # This prevents flickering when identification is noisy.
        if speaker_lookup and avatars and face_count > 0:
            ts_key = round(timestamp, 6)
            for face_idx in face_distances.keys():
                speaker = speaker_lookup.get((ts_key, face_idx))
                if speaker and speaker != 'unknown':
                    if face_idx not in face_speaker_votes:
                        face_speaker_votes[face_idx] = {}
                    face_speaker_votes[face_idx][speaker] = (
                        face_speaker_votes[face_idx].get(speaker, 0) + 1
                    )
                    # Pick the speaker with the highest vote count
                    best_speaker = max(
                        face_speaker_votes[face_idx],
                        key=face_speaker_votes[face_idx].get,
                    )
                    if best_speaker in avatars:
                        persistent_face_avatars[face_idx] = avatars[best_speaker]

        # ── Compute V-VAD decisions for ALL faces this frame ──────────────
        # Done once here so (a) the detector state advances exactly once per
        # face per frame, (b) the decision can be reused for landmark
        # colouring AND enrollment logic.
        vvad_decisions_frame = {}
        if face_count > 0 and vvad_detector is not None:
            ts_rounded = round(timestamp, 6)
            for face_idx in sorted(face_distances.keys()):
                face_lm = df_mouth[
                    (df_mouth['seconds'].round(6) == ts_rounded) &
                    (df_mouth['face_idx'] == face_idx)
                ]
                if face_lm.empty:
                    continue
                pts3d = {}
                for _, row in face_lm.iterrows():
                    pts3d[int(row['point_type'])] = (
                        float(row['x']), float(row['y']), float(row['z']))

                if isinstance(vvad_detector, DNN_VVAD_Detector):
                    if len(pts3d) >= 68:
                        is_act, _, _ = vvad_detector.update(face_idx, pts3d)
                        vvad_decisions_frame[face_idx] = is_act
                else:
                    fp = frontalize_mouth_landmarks(pts3d)
                    if fp:
                        mar = compute_mouth_aspect_ratio(fp)
                        is_act, _, _ = vvad_detector.update(face_idx, mar)
                        vvad_decisions_frame[face_idx] = is_act

        # ── Determine which target talkers are active this frame ─────────
        active_targets = []
        if enrollment_collector is not None and audio_samples is not None:
            avad_active = bool(
                smoothed_audio_vad[frame_idx]
            ) if frame_idx < len(smoothed_audio_vad) else False

            target_active = {}
            for fid, dist in face_distances.items():
                if dist <= target_distance:
                    enrollment_collector.register_target(fid)
                    vvad_active = vvad_decisions_frame.get(fid, False)
                    target_active[fid] = avad_active and vvad_active

            active_targets = [fid for fid, active in target_active.items()
                              if active]

        # Track which target talkers are visible in this frame (within distance)
        frame_visible_targets = set(
            fid for fid, dist in face_distances.items() if dist <= target_distance
        ) if enrollment_collector is not None else set()
        recent_target_window.append(frame_visible_targets)
        # Union of all target talkers seen in the last 11 frames (current + 10 prev)
        visible_targets = set().union(*recent_target_window)

        # ── Multivoice VAD (MVAD) — overlap detection ─────────────────
        mvad_current_label = 0
        mvad_data_dict = None
        if mvad_labels is not None and mvad_hop_sec is not None and mvad_hop_sec > 0:
            mvad_fidx = min(int(timestamp / mvad_hop_sec), len(mvad_labels) - 1)
            if mvad_fidx >= 0:
                mvad_current_label = int(mvad_labels[mvad_fidx])
            # Build history for last 5 seconds
            mvad_hist_start = max(0, int((timestamp - 5.0) / mvad_hop_sec))
            mvad_hist_end = min(len(mvad_labels), mvad_fidx + 1)
            mvad_history = list(mvad_labels[mvad_hist_start:mvad_hist_end])
            mvad_data_dict = {
                'history': mvad_history,
                'current_label': mvad_current_label,
            }

        # Draw dashboard overlay
        frame_bgr = draw_dashboard(frame_bgr, frame_idx, face_count, face_distances, timestamp,
                                   audio_data, vad_data,
                                   enrollment_collector=enrollment_collector,
                                   enrollment_duration=enrollment_duration,
                                   face_avatars=persistent_face_avatars,
                                   active_targets=active_targets,
                                   visible_targets=visible_targets,
                                   mvad_data=mvad_data_dict)

        # Draw facial landmarks for all detected faces (with V-VAD coloring)
        # Uses speaking_override to avoid double state updates on the detector.
        # Only shown when debug_mode is enabled.
        if debug_mode and show_markers and face_count > 0:
            for face_idx in sorted(face_distances.keys()):
                frame_bgr = draw_facial_landmarks(
                    frame_bgr, df_mouth, timestamp, face_idx,
                    frontalize=True, vvad_detector=None,
                    speaking_override=vvad_decisions_frame.get(face_idx, False)
                )

        # ── Enrollment audio collection ────────────────────────────────────
        # Block enrollment when MVAD indicates overlap (label == 2)
        mvad_is_overlap = (mvad_current_label == 2)
        if enrollment_collector is not None and audio_samples is not None:
            if len(active_targets) == 1 and not mvad_is_overlap:
                fid = active_targets[0]
                if not enrollment_collector.is_completed(fid):
                    s_start = int(frame_idx / fps * audio_sr)
                    s_end = int((frame_idx + 1) / fps * audio_sr)
                    s_end = min(s_end, len(audio_samples))
                    if s_end > s_start:
                        enrollment_collector.add_audio(
                            fid, audio_samples[s_start:s_end])

        # Speaker avatars are now drawn inside the dashboard Enrollment section
        
        # Convert back to RGB for imageio
        frame_rgb = frame_bgr[..., ::-1]
        
        # Write frame
        writer.append_data(frame_rgb)
    
    # Clean up
    writer.close()
    reader.close()
    
    print(f"\n✓ Video rendering complete")

    # ── Save enrollment WAVs ──────────────────────────────────────────
    if enrollment_collector is not None and enrollment_collector.ever_seen:
        video_stem = Path(video_path).stem
        out_dir = output_path.parent
        print(f"\nEnrollment audio results:")
        for fid in sorted(enrollment_collector.ever_seen):
            audio_data = enrollment_collector.get_audio(fid)
            dur = enrollment_collector.get_duration(fid)
            if audio_data is not None and len(audio_data) > 0:
                wav_name = f"{video_stem}_enrollment_tt{fid}.wav"
                wav_out = out_dir / wav_name
                sf.write(str(wav_out), audio_data, audio_sr, subtype='FLOAT')
                print(f"  ✓ Target Talker {fid}: {dur:.2f}s → {wav_out}")
            else:
                print(f"  ✗ Target Talker {fid}: no speech collected (0.0s)")

    # Copy audio from original video
    copy_audio_with_ffmpeg(video_path, temp_output, output_path)
    
    print(f"\n✓ Final video saved to: {output_path}")


def main(args):
    # Setup paths
    dumps_dir = Path(args.dumps_dir)
    video_path = Path(args.video)
    output_path = Path(args.output)
    
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        return
    
    print("="*60)
    print("Multi-Face Distance Video Renderer (with Speaker Avatars)")
    print("="*60)
    
    # Read CSV files
    print("\nLoading CSV data...")
    df_face_count = pd.read_csv(dumps_dir / 'face_count.csv')
    df_mouth = pd.read_csv(dumps_dir / 'mouth_position.csv')
    
    # Try to load MQE metrics data
    df_mqe = None
    mqe_path = dumps_dir / 'MQE_Metrics.csv'
    if mqe_path.exists():
        df_mqe = pd.read_csv(mqe_path)
        print(f"  Loaded MQE metrics: {len(df_mqe)} segments ({len(df_mqe) * 5}s)")
    else:
        print(f"  No MQE metrics data found (optional)")
    
    # Try to load VAD data
    df_vad = None
    vad_path = dumps_dir / 'vad.csv'
    if vad_path.exists():
        df_vad = pd.read_csv(vad_path)
        print(f"  Loaded VAD data: {len(df_vad)} entries")
    else:
        print(f"  No VAD data found (optional)")

    # Try to load Multivoice VAD DNN predictions
    mvad_labels = None
    mvad_hop_sec = None
    mvad_path = dumps_dir / 'mvad_dnn.npz'
    if mvad_path.exists():
        mvad_npz = np.load(str(mvad_path))
        mvad_labels = mvad_npz['labels']
        mvad_hop_sec = float(mvad_npz['hop_sec'])
        n_ovl = int(np.sum(mvad_labels == 2))
        n_sng = int(np.sum(mvad_labels == 1))
        print(f"  Loaded MVAD DNN: {len(mvad_labels)} frames "
              f"({len(mvad_labels) * mvad_hop_sec:.1f}s), "
              f"single={n_sng * mvad_hop_sec:.1f}s, "
              f"overlap={n_ovl * mvad_hop_sec:.1f}s")
    else:
        print(f"  No MVAD DNN data found ({mvad_path}) — overlap blocking disabled")
    
    print(f"  Loaded {len(df_face_count)} frames")

    # Load speaker identification data
    speaker_lookup = None
    speaker_id_path = dumps_dir / 'speaker_identification.csv'
    if speaker_id_path.exists():
        ref_seconds = np.sort(df_face_count['seconds'].values)
        speaker_lookup = load_speaker_identification(speaker_id_path,
                                                     reference_seconds=ref_seconds)
        speakers_found = set(v for v in speaker_lookup.values() if v != 'unknown')
        print(f"  Loaded speaker identification: {len(speaker_lookup)} entries, "
              f"speakers: {sorted(speakers_found)}")
    else:
        print(f"  No speaker identification data found (optional)")

    # Load avatar images
    avatars_dir = Path(args.avatars_dir)
    avatars = load_avatars(avatars_dir)
    if avatars:
        print(f"  Loaded {len(avatars)} avatar(s): {sorted(avatars.keys())}")
    else:
        print(f"  No avatar images loaded from {avatars_dir}")

    # Locate enrollment WAV (*_txfeOut.wav) in dumps directory
    enrollment_wav_path = None
    video_stem = video_path.stem  # e.g. "example_6"
    txfe_candidates = sorted(dumps_dir.glob(f"{video_stem}*_txfeOut.wav"))
    if txfe_candidates:
        enrollment_wav_path = txfe_candidates[0]
        print(f"  Enrollment WAV: {enrollment_wav_path}")
    else:
        print(f"  No *_txfeOut.wav found in {dumps_dir} for '{video_stem}' (enrollment audio disabled)")

    # Optionally assign stable track IDs from face recognition embeddings
    if getattr(args, 'embeddings', None):
        emb_path = Path(args.embeddings)
        if not emb_path.exists():
            print(f"  Error: Embeddings file not found: {emb_path}")
            return
        print(f"  Loading face embeddings from {emb_path}...")
        embeddings, seconds_arr, face_idx_arr = load_embeddings_npz(emb_path)
        print(f"  Computing face track IDs from embeddings (similarity matching)...")
        assignment_map = compute_assignment_map(
            embeddings, seconds_arr, face_idx_arr, df_face_count,
            similarity_threshold=getattr(args, 'embedding_threshold', 0.3),
            drop_after_frames=getattr(args, 'embedding_drop_frames', 60)
        )
        df_mouth = _reindex_df(df_mouth, assignment_map)
        if speaker_lookup:
            speaker_lookup = reindex_speaker_lookup(speaker_lookup, assignment_map)
        unique_faces = sorted(df_mouth['face_idx'].unique())
        print(f"  Found {len(unique_faces)} stable track IDs: {unique_faces}")
    else:
        unique_faces = sorted(df_mouth['face_idx'].unique())
        print(f"  Using raw face indices (no embeddings provided)")

    # Render video
    render_video_with_dashboard(
        video_path, 
        df_face_count, 
        df_mouth, 
        df_mqe,
        df_vad,
        output_path,
        show_markers=args.show_markers,
        speaker_lookup=speaker_lookup,
        avatars=avatars,
        vvad_model_path=getattr(args, 'vvad_model', None),
        audio_vad_pad_ms=getattr(args, 'audio_vad_pad_ms', 250.0),
        vvad_pad_ms=getattr(args, 'vvad_pad_ms', 0.0),
        enrollment_duration=getattr(args, 'enrollment_duration', 10.0),
        target_distance=getattr(args, 'target_distance', 2.0),
        debug_mode=getattr(args, 'debug_mode', False),
        enrollment_wav_path=enrollment_wav_path,
        mvad_labels=mvad_labels,
        mvad_hop_sec=mvad_hop_sec,
    )
    
    print("\n" + "="*60)
    print("Done!")
    print("="*60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Render video with face distance dashboard overlay and speaker avatars',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (avatars loaded automatically from enrollment-avatars/)
  python3 multiface_avatar_render.py -i inputs/video.mp4 -o output_annotated.mp4

  # Specify custom dumps directory and avatars directory
  python3 multiface_avatar_render.py -i video.mp4 -o output.mp4 --dumps_dir=dumps --avatars-dir=enrollment-avatars

  # Without face position markers
  python3 multiface_avatar_render.py -i video.mp4 -o output.mp4 --no-markers

  # With face identity from precomputed embeddings
  python3 multiface_avatar_render.py -i video.mp4 -o output.mp4 --embeddings dumps/face_embeddings.npz

Output:
  Creates an MP4 video with:
  - Dashboard panel showing frame info, face count, distances
  - Speaker avatar overlays above identified faces
  - Optional face position markers
  - Color-coded per face
  - Distance in cm (calculated as 100/z)
        """
    )
    
    parser.add_argument('-i', '--video', type=str, required=True,
                       help='Input video file path')
    parser.add_argument('-o', '--output', type=str, required=True,
                       help='Output video file path')
    parser.add_argument('--dumps_dir', type=str, default='dumps',
                       help='Directory containing CSV dump files (default: dumps)')
    parser.add_argument('--avatars-dir', type=str, default='enrollment-avatars',
                       help='Directory containing speaker avatar images (default: enrollment-avatars)')
    parser.add_argument('--show-markers', dest='show_markers', action='store_true',
                       help='Show face position markers on video (default)')
    parser.add_argument('--no-markers', dest='show_markers', action='store_false',
                       help='Hide face position markers')
    parser.add_argument('--embeddings', type=str, default=None,
                       help='Path to .npz with precomputed face embeddings (keys: embeddings, seconds, face_idx) for stable identity')
    parser.add_argument('--embedding-threshold', type=float, default=0.3,
                       help='Min cosine similarity to match face to existing track (default: 0.3)')
    parser.add_argument('--embedding-drop-frames', type=int, default=60,
                       help='Drop track after this many frames unseen (default: 60)')
    parser.add_argument('--vvad_model', type=str, default='vvad_dnn_model.pt',
                       help='Path to trained V-VAD DNN model checkpoint '
                            '(default: vvad_dnn_model.pt). '
                            'Falls back to MAR heuristic if file not found.')
    parser.add_argument('--audio-vad-pad-ms', dest='audio_vad_pad_ms',
                       type=float, default=250.0,
                       help='Audio VAD smoothing: ms of padding added on each '
                            'side of every active block (default: 250)')
    parser.add_argument('--vvad-pad-ms', dest='vvad_pad_ms',
                       type=float, default=0.0,
                       help='Video VAD smoothing: ms of padding added on each '
                            'side of every active block (default: 0 = off)')
    parser.add_argument('--enrollment-duration', dest='enrollment_duration',
                       type=float, default=10.0,
                       help='Target enrollment audio duration in seconds '
                            '(default: 10)')
    parser.add_argument('--target-distance', dest='target_distance',
                       type=float, default=2.0,
                       help='Maximum distance in meters for a face to be '
                            'considered a target talker (default: 2.0)')
    parser.add_argument('--debug-mode', dest='debug_mode',
                       action='store_true', default=False,
                       help='Enable debug overlays: outer mouth landmarks '
                            'and face index labels (F0, F1, ...) on video '
                            '(default: False)')
    parser.set_defaults(show_markers=True)

    args = parser.parse_args()
    main(args)
