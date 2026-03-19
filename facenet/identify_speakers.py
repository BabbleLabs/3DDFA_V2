#!/usr/bin/env python3
"""
Identify speakers in video frames by comparing FaceNet embeddings against
enrollment avatars.

For every frame (or every Nth frame) of a given video this script:
  1. Detects all faces using MTCNN (with bounding boxes used for cropping).
  2. Optionally matches detected faces to face IDs from mouth_position.csv by
     comparing point_type 27 (x, y) from the CSV to the crop/bbox positions;
     each CSV face_idx is assigned to the detected face whose bbox center
     is closest to that (x, y).
  3. Computes a 512-d FaceNet embedding for each detected face.
  4. Compares each embedding against pre-enrolled avatar embeddings via
     cosine similarity.
  5. Writes a CSV file whose rows match the (seconds, face_idx) layout
     used by mouth_position.csv / face_position.csv, augmented with the
     identified speaker name and similarity score.

Usage:
  export PYTHONPATH=./src
  python identify_speakers.py <model_dir> <video_path> <enrollment_dir> [options]

Example:
  python identify_speakers.py models/20180402-114759 myvideo.mp4 \\
      enrollment-avatars/ --threshold 0.7 -o speaker_id.csv
  # Match face IDs to mouth_position.csv (point_type 27):
  python identify_speakers.py ... -m mouth_position.csv -o speaker_id.csv
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import csv
import os
import sys
import time

import cv2
import numpy as np
from PIL import Image

os.environ['CUDA_VISIBLE_DEVICES'] = ''

import tensorflow as tf
if hasattr(tf, 'compat') and hasattr(tf.compat, 'v1'):
    tf = tf.compat.v1
    tf.disable_v2_behavior()

import facenet
import align.detect_face

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
MTCNN_MINSIZE = 20
MTCNN_THRESHOLD = [0.6, 0.7, 0.7]
MTCNN_FACTOR = 0.709


def detect_and_align_faces(img, pnet, rnet, onet, image_size, margin,
                           sort_left_to_right=True):
    """Detect faces with MTCNN, crop, align, and prewhiten each one.

    Returns
    -------
    faces : list[np.ndarray]
        Prewhitened face crops ready for embedding.
    bboxes : np.ndarray, shape (n_faces, 5)
        Bounding boxes [x1, y1, x2, y2, conf]; same order as faces.
        If sort_left_to_right, bboxes are sorted by x1 for consistent ordering.
    """
    bboxes, _ = align.detect_face.detect_face(
        img, MTCNN_MINSIZE, pnet, rnet, onet, MTCNN_THRESHOLD, MTCNN_FACTOR
    )
    if bboxes.size == 0:
        return [], np.empty((0, 5))

    if sort_left_to_right:
        order = np.argsort(bboxes[:, 0])
        bboxes = bboxes[order]
    img_h, img_w = img.shape[:2]

    faces = []
    valid_indices = []
    for i in range(bboxes.shape[0]):
        det = bboxes[i, :4]
        bb = [
            max(int(det[0] - margin / 2), 0),
            max(int(det[1] - margin / 2), 0),
            min(int(det[2] + margin / 2), img_w),
            min(int(det[3] + margin / 2), img_h),
        ]
        cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
        if cropped.size == 0:
            continue
        aligned = np.array(
            Image.fromarray(cropped).resize((image_size, image_size), Image.BILINEAR)
        )
        faces.append(facenet.prewhiten(aligned))
        valid_indices.append(i)

    return faces, bboxes[valid_indices] if valid_indices else np.empty((0, 5))


def load_mouth_position_refs(csv_path, point_type=27):
    """Load (seconds, face_idx, x, y) for point_type from mouth_position.csv.

    Returns
    -------
    ref_by_time : dict[float, list of (face_idx, x, y)]
        Keys are seconds; values are lists of (face_idx, x, y) for that
        timestamp (one per face from the CSV with point_type).
    """
    ref_by_time = {}
    path = os.path.expanduser(csv_path)
    with open(path, newline='', encoding='utf-8') as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if int(row['point_type']) != point_type:
                continue
            t = float(row['seconds'])
            face_idx = int(row['face_idx'])
            x, y = float(row['x']), float(row['y'])
            ref_by_time.setdefault(t, []).append((face_idx, x, y))
    # Sort each list by face_idx for deterministic matching
    for t in ref_by_time:
        ref_by_time[t] = sorted(ref_by_time[t], key=lambda r: r[0])
    return ref_by_time


def get_refs_for_timestamp(ref_by_time, timestamp):
    """Return list of (face_idx, x, y) for the timestamp closest to *timestamp*."""
    if not ref_by_time:
        return []
    times = np.array(list(ref_by_time.keys()))
    best_t = float(times[np.argmin(np.abs(times - timestamp))])
    return ref_by_time[best_t]


def bbox_center(bbox):
    """Return (cx, cy) for bbox [x1, y1, x2, y2, ...]."""
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def match_faces_to_refs(bboxes, refs):
    """Assign each CSV (face_idx, x, y) to the closest detected face by position.

    Uses the same crop/bbox positions as in detect_and_align_faces (bbox
    center). Each ref is assigned to the detected face whose bbox center is
    closest to (x, y); each detected face is assigned at most one ref.

    Returns
    -------
    detected_to_face_idx : dict[int, int]
        Maps detected face index -> mouth_position.csv face_idx.
    """
    if not bboxes.size or not refs:
        return {}
    n_det = bboxes.shape[0]
    centers = np.array([bbox_center(bboxes[i]) for i in range(n_det)])
    # refs are (face_idx, x, y), sorted by face_idx
    assigned_det = set()
    detected_to_face_idx = {}
    for face_idx, rx, ry in refs:
        best_det = None
        best_d = float('inf')
        for i in range(n_det):
            if i in assigned_det:
                continue
            cx, cy = centers[i]
            d = (cx - rx) ** 2 + (cy - ry) ** 2
            if d < best_d:
                best_d = d
                best_det = i
        if best_det is not None:
            assigned_det.add(best_det)
            detected_to_face_idx[best_det] = face_idx
    return detected_to_face_idx


def compute_embeddings(faces, sess, img_placeholder, emb_tensor, phase_placeholder):
    """Run FaceNet forward pass on a batch of prewhitened face crops."""
    if not faces:
        return np.zeros((0, 512))
    feed = {img_placeholder: np.stack(faces), phase_placeholder: False}
    return sess.run(emb_tensor, feed_dict=feed)


def cosine_similarity(a, b):
    """Cosine similarity between two 1-D vectors."""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def identify_speaker(face_emb, enrollments, threshold):
    """Find the best-matching enrolled speaker for *face_emb*.

    Returns (speaker_name, similarity).  If no enrolled speaker exceeds
    *threshold*, speaker_name is ``"unknown"``.
    """
    best_name = 'unknown'
    best_sim = -1.0
    for name, enrolled_emb in enrollments.items():
        sim = cosine_similarity(face_emb, enrolled_emb)
        if sim > best_sim:
            best_sim = sim
            best_name = name
    if best_sim < threshold:
        return 'unknown', best_sim
    return best_name, best_sim


def enroll_avatars(avatar_dir, pnet, rnet, onet,
                   sess, img_placeholder, emb_tensor, phase_placeholder,
                   image_size, margin):
    """Compute reference embeddings for every avatar image.

    Each image file name (without extension) becomes the speaker label.
    Only the most prominent face in each avatar image is used.

    Returns dict[str, np.ndarray] mapping speaker name -> embedding.
    """
    avatar_dir = os.path.expanduser(avatar_dir)
    if not os.path.isdir(avatar_dir):
        raise ValueError('Enrollment directory not found: %s' % avatar_dir)

    enrollments = {}
    for filename in sorted(os.listdir(avatar_dir)):
        ext = os.path.splitext(filename)[1].lower()
        if ext not in IMAGE_EXTENSIONS:
            continue
        path = os.path.join(avatar_dir, filename)
        img = np.array(Image.open(path).convert('RGB'))
        faces, _ = detect_and_align_faces(img, pnet, rnet, onet, image_size, margin)
        if not faces:
            print('Warning: no face detected in avatar %s, skipping' % filename,
                  file=sys.stderr)
            continue
        emb = compute_embeddings(
            [faces[0]], sess, img_placeholder, emb_tensor, phase_placeholder
        )
        speaker_name = os.path.splitext(filename)[0]
        enrollments[speaker_name] = emb[0]
        print('  Enrolled: %s' % speaker_name)

    if not enrollments:
        raise ValueError('No faces found in any avatar image in %s' % avatar_dir)
    return enrollments


def main(args):
    video_path = os.path.expanduser(args.video_path)
    if not os.path.isfile(video_path):
        print('Error: video file not found: %s' % video_path, file=sys.stderr)
        sys.exit(1)

    # ---- MTCNN (face detection) in its own TF graph ----
    print('Initializing MTCNN face detector...')
    with tf.Graph().as_default():
        gpu_opts = tf.GPUOptions(allow_growth=True)
        mtcnn_sess = tf.Session(
            config=tf.ConfigProto(gpu_options=gpu_opts, log_device_placement=False)
        )
        with mtcnn_sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(mtcnn_sess, None)

    # ---- FaceNet (embedding model) in its own TF graph ----
    print('Loading FaceNet model from %s ...' % args.model_dir)
    fn_graph = tf.Graph()
    with fn_graph.as_default():
        gpu_opts = tf.GPUOptions(allow_growth=True)
        fn_sess = tf.Session(
            config=tf.ConfigProto(gpu_options=gpu_opts, log_device_placement=False)
        )
        with fn_sess.as_default():
            facenet.load_model(args.model_dir)
    img_ph = fn_graph.get_tensor_by_name('input:0')
    emb_t = fn_graph.get_tensor_by_name('embeddings:0')
    phase_ph = fn_graph.get_tensor_by_name('phase_train:0')

    # ---- Enroll speakers from avatar images ----
    print('Enrolling avatars from %s ...' % args.enrollment_dir)
    enrollments = enroll_avatars(
        args.enrollment_dir, pnet, rnet, onet,
        fn_sess, img_ph, emb_t, phase_ph,
        args.image_size, args.margin,
    )
    print('Enrolled %d speaker(s): %s' % (len(enrollments), ', '.join(enrollments)))

    # ---- Optional: load mouth_position.csv for face ID matching ----
    ref_by_time = None
    if args.mouth_position:
        mp_path = os.path.expanduser(args.mouth_position)
        if not os.path.isfile(mp_path):
            print('Error: mouth_position file not found: %s' % mp_path,
                  file=sys.stderr)
            sys.exit(1)
        print('Loading face reference positions (point_type %d) from %s ...'
              % (args.mouth_point_type, mp_path))
        ref_by_time = load_mouth_position_refs(mp_path, args.mouth_point_type)
        n_ts = len(ref_by_time)
        n_refs = sum(len(v) for v in ref_by_time.values())
        print('  %d timestamps, %d face reference points' % (n_ts, n_refs))

    # ---- Open video ----
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print('Error: cannot open video: %s' % video_path, file=sys.stderr)
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print('Video: %d frames @ %.2f fps' % (total_frames, fps))
    print('Processing every %d frame(s), similarity threshold=%.3f'
          % (args.step, args.threshold))
    if ref_by_time is not None:
        print('Face IDs matched to mouth_position.csv (point_type %d)'
              % args.mouth_point_type)

    # ---- Frame-by-frame processing ----
    rows = []
    frame_idx = 0
    processed = 0
    t0 = time.time()
    use_mouth_refs = ref_by_time is not None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % args.step == 0:
                timestamp = frame_idx / fps
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                faces, bboxes = detect_and_align_faces(
                    rgb, pnet, rnet, onet, args.image_size, args.margin,
                    sort_left_to_right=(not use_mouth_refs),
                )
                if faces:
                    if use_mouth_refs:
                        refs = get_refs_for_timestamp(ref_by_time, timestamp)
                        det_to_face_idx = match_faces_to_refs(bboxes, refs)
                        if not det_to_face_idx:
                            processed += 1
                            frame_idx += 1
                            continue
                    else:
                        det_to_face_idx = {i: i for i in range(len(faces))}

                    embs = compute_embeddings(
                        faces, fn_sess, img_ph, emb_t, phase_ph
                    )
                    for det_i, face_idx in det_to_face_idx.items():
                        speaker, sim = identify_speaker(
                            embs[det_i], enrollments, args.threshold
                        )
                        rows.append({
                            'seconds': timestamp,
                            'face_idx': face_idx,
                            'speaker': speaker,
                            'similarity': round(sim, 6),
                        })

                processed += 1
                if processed % 100 == 0:
                    elapsed = time.time() - t0
                    print('  %d frames processed (%d/%d), %.1fs elapsed'
                          % (processed, frame_idx, total_frames, elapsed))

            frame_idx += 1
    finally:
        cap.release()

    elapsed = time.time() - t0
    print('Processed %d frames in %.1fs' % (processed, elapsed))

    # ---- Write output CSV ----
    out_path = args.output
    print('Writing %d rows to %s' % (len(rows), out_path))
    with open(out_path, 'w', newline='') as fh:
        writer = csv.DictWriter(
            fh, fieldnames=['seconds', 'face_idx', 'speaker', 'similarity']
        )
        writer.writeheader()
        writer.writerows(rows)

    print('Done.')


def parse_arguments(argv):
    parser = argparse.ArgumentParser(
        description='Identify speakers in video frames by comparing face '
                    'embeddings against enrollment avatars.'
    )
    parser.add_argument(
        'model_dir', type=str,
        help='Path to FaceNet model directory (contains .pb or .meta/.ckpt)',
    )
    parser.add_argument(
        'video_path', type=str,
        help='Path to the input video file',
    )
    parser.add_argument(
        'enrollment_dir', type=str,
        help='Directory with enrollment avatar images (one face per image, '
             'file name = speaker label)',
    )
    parser.add_argument(
        '--threshold', type=float, default=0.7,
        help='Cosine-similarity threshold for positive identification '
             '(default: 0.7)',
    )
    parser.add_argument(
        '--output', '-o', type=str, default='speaker_identification.csv',
        help='Output CSV path (default: speaker_identification.csv)',
    )
    parser.add_argument(
        '--step', '-s', type=int, default=1,
        help='Process every Nth frame (default: 1 = every frame)',
    )
    parser.add_argument(
        '--image_size', type=int, default=160,
        help='Aligned face crop size in pixels (default: 160)',
    )
    parser.add_argument(
        '--margin', type=int, default=44,
        help='Margin around the bounding box in pixels (default: 44)',
    )
    parser.add_argument(
        '--gpu_memory_fraction', type=float, default=1.0,
        help='Fraction of GPU memory to allocate (default: 1.0)',
    )
    parser.add_argument(
        '--mouth_position', '-m', type=str, default=None,
        help='Path to mouth_position.csv; if set, face_idx in the output is '
             'matched to this file by comparing point_type 27 (x,y) to bbox '
             'centers of detected faces.',
    )
    parser.add_argument(
        '--mouth_point_type', type=int, default=27,
        help='Point type in mouth_position.csv to use for (x,y) matching '
             '(default: 27).',
    )
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
