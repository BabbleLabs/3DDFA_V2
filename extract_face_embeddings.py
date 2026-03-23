#!/usr/bin/env python3
# coding: utf-8

"""
Extract face embeddings for stable identity in multiface_distance_render.

You do NOT need to change your dump data. This script uses the existing dumps:
  - mouth_position.csv has all 68 landmark (x, y, z) per (seconds, face_idx).
  - From those landmarks we derive a face bounding box (min/max x,y + padding)
    and crop the face from the video at that timestamp.

Pipeline:
  1. Load video and dumps (face_count.csv, mouth_position.csv).
  2. For each (seconds, face_idx) present in the dumps:
     - Get the video frame at that timestamp.
     - Get the 68 (x, y) points from mouth_position for that (seconds, face_idx).
     - Compute bbox = (min_x, min_y, max_x, max_y) over landmarks, add padding.
     - Crop the face from the frame (optionally resize/align for your model).
  3. Run your face recognition model on each crop → one vector per face.
  4. Save embeddings, seconds, face_idx to a .npz for --embeddings.

Usage:
  # With a custom embedding function (you implement get_embedding(crop_bgr) -> np.array):
  python3 extract_face_embeddings.py -i video.mp4 --dumps_dir dumps -o face_embeddings.npz

  # Without a model: only writes crops to a directory (for debugging or external embedding):
  python3 extract_face_embeddings.py -i video.mp4 --dumps_dir dumps --crops_dir crops

To plug in InsightFace / FaceNet / dlib:
  - Implement get_embedding(crop_bgr) that returns a 1D numpy array (e.g. 512-d).
  - Pass it via --embedding_callback module.function or set EMBEDDING_CALLBACK in script.
"""

import argparse
import os
import numpy as np
import pandas as pd
import cv2
import imageio
from pathlib import Path
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Crop from existing dumps (no extra dump preparation)
# ---------------------------------------------------------------------------

def get_face_bbox_from_landmarks(df_mouth, seconds, face_idx, padding_ratio=0.35, tol=0.001):
    """
    Get face bounding box from mouth_position.csv landmarks for one (seconds, face_idx).
    Uses all 68 points: bbox = min/max of (x, y) with padding.
    padding_ratio: add padding_ratio * bbox_size on each side (default 0.35).
    Returns (x1, y1, x2, y2) in image coordinates, or None if no data.
    """
    rows = df_mouth[
        (np.abs(df_mouth['seconds'].astype(float) - seconds) < tol) &
        (df_mouth['face_idx'] == face_idx)
    ]
    if rows.empty:
        return None
    x = rows['x'].values.astype(float)
    y = rows['y'].values.astype(float)
    x1, x2 = float(x.min()), float(x.max())
    y1, y2 = float(y.min()), float(y.max())
    w = x2 - x1
    h = y2 - y1
    pad_w = w * padding_ratio
    pad_h = h * padding_ratio
    x1 = max(0, x1 - pad_w)
    y1 = max(0, y1 - pad_h)
    x2 = x2 + pad_w
    y2 = y2 + pad_h
    return (x1, y1, x2, y2)


def iter_face_crops(video_path, df_face_count, df_mouth, padding_ratio=0.35):
    """
    Yield (crop_bgr, seconds, face_idx) for each (seconds, face_idx) in the dumps.
    crop_bgr: BGR image (numpy array) cropped using landmark-derived bbox.
    No embedding model is called here; you run your model on crop_bgr.
    """
    video_path = Path(video_path)
    reader = imageio.get_reader(video_path)
    fps = reader.get_meta_data().get('fps', 30.0)

    frames_sorted = df_face_count.sort_values('seconds')[['seconds', 'face_count']].values
    current_frame_idx = None
    current_bgr = None

    try:
        for seconds, face_count in tqdm(frames_sorted, desc="Face crops"):
            frame_idx = int(round(seconds * fps))
            if frame_idx != current_frame_idx:
                try:
                    frame = reader.get_data(frame_idx)
                except (IndexError, Exception):
                    continue
                current_bgr = np.ascontiguousarray(frame[..., ::-1])
                current_frame_idx = frame_idx

            if current_bgr is None:
                continue

            h_img, w_img = current_bgr.shape[:2]
            for face_idx in range(int(face_count)):
                bbox = get_face_bbox_from_landmarks(df_mouth, seconds, face_idx, padding_ratio=padding_ratio)
                if bbox is None:
                    continue
                x1, y1, x2, y2 = bbox
                x1, y1 = int(max(0, x1)), int(max(0, y1))
                x2, y2 = int(min(w_img, x2)), int(min(h_img, y2))
                if x2 <= x1 or y2 <= y1:
                    continue
                crop = current_bgr[y1:y2, x1:x2].copy()
                yield crop, float(seconds), int(face_idx)
    finally:
        reader.close()


def extract_and_save(video_path, dumps_dir, output_npz, get_embedding=None, padding_ratio=0.35,
                    crops_dir=None):
    """
    Iterate all (seconds, face_idx), crop face, optionally run get_embedding(crop_bgr),
    and save embeddings (and optionally crops) to disk.
    get_embedding: callable(crop_bgr) -> np.array 1D, or None to skip embeddings.
    """
    dumps_dir = Path(dumps_dir)
    df_face_count = pd.read_csv(dumps_dir / 'face_count.csv')
    df_mouth = pd.read_csv(dumps_dir / 'mouth_position.csv')

    embeddings_list = []
    seconds_list = []
    face_idx_list = []
    if crops_dir:
        Path(crops_dir).mkdir(parents=True, exist_ok=True)

    for crop_bgr, seconds, face_idx in iter_face_crops(video_path, df_face_count, df_mouth, padding_ratio=padding_ratio):
        sec_key = round(seconds, 6)
        if crops_dir:
            out_name = Path(crops_dir) / f"sec_{sec_key}_face_{face_idx}.jpg"
            cv2.imwrite(str(out_name), crop_bgr)
        if get_embedding is not None:
            emb = get_embedding(crop_bgr)
            if emb is not None:
                embeddings_list.append(np.asarray(emb, dtype=np.float64).ravel())
                seconds_list.append(sec_key)
                face_idx_list.append(face_idx)

    if get_embedding is not None and embeddings_list:
        embeddings = np.stack(embeddings_list, axis=0)
        seconds_arr = np.array(seconds_list, dtype=np.float64)
        face_idx_arr = np.array(face_idx_list, dtype=np.int64)
        np.savez(output_npz, embeddings=embeddings, seconds=seconds_arr, face_idx=face_idx_arr)
        print(f"Saved {len(seconds_list)} embeddings to {output_npz}")
    elif get_embedding is not None:
        print("No embeddings produced (get_embedding returned None for all crops).")
    if crops_dir:
        print(f"Crops written to {crops_dir}")


def get_embedding_placeholder(crop_bgr):
    """
    Placeholder: returns a dummy vector so the script runs end-to-end.
    Replace this with your model (InsightFace, FaceNet, dlib) to get real embeddings.
    Example: return your_model.get_embedding(crop_bgr)  # 1D numpy array
    """
    return np.zeros(128, dtype=np.float64)


class FaceNetEmbedder:
    """Loads a FaceNet frozen-graph model and computes 512-d embeddings."""

    FACENET_INPUT_SIZE = 160

    def __init__(self, model_dir):
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        import tensorflow as tf
        if hasattr(tf, 'compat') and hasattr(tf.compat, 'v1'):
            tf = tf.compat.v1
            tf.disable_v2_behavior()

        self._graph = tf.Graph()
        with self._graph.as_default():
            gpu_opts = tf.GPUOptions(allow_growth=True)
            self._sess = tf.Session(
                config=tf.ConfigProto(gpu_options=gpu_opts,
                                      log_device_placement=False))
            model_exp = os.path.expanduser(model_dir)
            pb_files = [f for f in os.listdir(model_exp) if f.endswith('.pb')]
            if pb_files:
                pb_path = os.path.join(model_exp, pb_files[0])
                with tf.io.gfile.GFile(pb_path, 'rb') as f:
                    graph_def = tf.GraphDef()
                    graph_def.ParseFromString(f.read())
                    tf.import_graph_def(graph_def, name='')
            else:
                files = os.listdir(model_exp)
                meta_files = [s for s in files if s.endswith('.meta')]
                if not meta_files:
                    raise ValueError(
                        f'No .pb or .meta file in model directory: {model_exp}')
                meta_file = meta_files[0]
                ckpt = tf.train.get_checkpoint_state(model_exp)
                ckpt_file = os.path.basename(ckpt.model_checkpoint_path)
                saver = tf.train.import_meta_graph(
                    os.path.join(model_exp, meta_file))
                saver.restore(self._sess,
                              os.path.join(model_exp, ckpt_file))

        self._input_ph = self._graph.get_tensor_by_name('input:0')
        self._embeddings_t = self._graph.get_tensor_by_name('embeddings:0')
        self._phase_ph = self._graph.get_tensor_by_name('phase_train:0')

    @staticmethod
    def _prewhiten(x):
        mean = np.mean(x)
        std = np.std(x)
        std_adj = max(std, 1.0 / np.sqrt(x.size))
        return (x - mean) / std_adj

    def __call__(self, crop_bgr):
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.FACENET_INPUT_SIZE,
                                   self.FACENET_INPUT_SIZE))
        prewhitened = self._prewhiten(resized.astype(np.float64))
        feed = {
            self._input_ph: np.expand_dims(prewhitened, axis=0),
            self._phase_ph: False,
        }
        emb = self._sess.run(self._embeddings_t, feed_dict=feed)
        return emb[0]


def main():
    parser = argparse.ArgumentParser(
        description="Extract face crops from video using dumps and optionally compute embeddings for stable identity."
    )
    parser.add_argument("-i", "--video", type=str, required=True, help="Input video file")
    parser.add_argument("--dumps_dir", type=str, default="dumps", help="Directory with face_count.csv, mouth_position.csv")
    parser.add_argument("-o", "--output", type=str, default="face_embeddings.npz", help="Output .npz path for embeddings")
    parser.add_argument("--crops_dir", type=str, default=None, help="If set, save crop images here (for debugging)")
    parser.add_argument("--padding", type=float, default=0.35, help="Padding around landmark bbox (ratio of width/height)")
    parser.add_argument("--no_embedding", action="store_true", help="Only save crops; do not compute embeddings (no model)")
    parser.add_argument("--model_dir", type=str, default=None,
                        help="Path to FaceNet model directory (frozen .pb or "
                             ".meta/.ckpt). Produces real 512-d embeddings "
                             "instead of placeholder zeros.")
    args = parser.parse_args()

    if args.no_embedding:
        get_embedding = None
    elif args.model_dir:
        print(f"Loading FaceNet model from {args.model_dir} ...")
        get_embedding = FaceNetEmbedder(args.model_dir)
        print("FaceNet model loaded — will produce 512-d embeddings")
    else:
        get_embedding = get_embedding_placeholder

    extract_and_save(
        args.video,
        args.dumps_dir,
        args.output,
        get_embedding=get_embedding,
        padding_ratio=args.padding,
        crops_dir=args.crops_dir,
    )


if __name__ == "__main__":
    main()
