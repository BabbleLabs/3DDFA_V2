# coding: utf-8

import argparse
import sys
import imageio
import cv2
import numpy as np
from tqdm import tqdm
import yaml
from collections import deque
from utils.tddfa_util import str2bool
from utils.pose import calc_pose
import os
from scipy.optimize import linear_sum_assignment

from FaceBoxes import FaceBoxes
from TDDFA import TDDFA
from utils.render import render
# from utils.render_ctypes import render
from utils.functions import cv_draw_landmark

face_count_dump_file_path = 'dumps/face_count.csv'
face_orientation_dump_file_path = 'dumps/face_position.csv'
mouth_position_dump_file_path = 'dumps/mouth_position.csv'


class FaceTracker:
    """
    Spatial face tracker that maintains consistent face_idx across frames.

    Problem: FaceBoxes returns detected faces in an order that may change
    frame-to-frame (e.g. sorted by confidence or area).  When two speakers
    are at a similar distance from the camera, their detection order can
    flip, causing face_idx 0 and 1 to swap randomly.

    Solution: Track the spatial position (centroid) of each face across
    frames using an exponential moving average (EMA).  On each new frame,
    use the Hungarian algorithm to find the optimal assignment between
    detected face centroids and tracked positions, then reorder the
    detection list so that face_idx remains consistent.

    Parameters:
        ema_alpha: Smoothing factor for centroid EMA (0–1).
                   Higher → quicker response to movement; lower → more stable.
        max_distance: Maximum pixel distance to allow matching a detection
                      to an existing track.  Detections farther than this
                      create a new track.
        drop_after_frames: Number of consecutive unseen frames before a
                           track is considered lost and freed.
    """

    def __init__(self, ema_alpha=0.3, max_distance=300, drop_after_frames=30):
        self.ema_alpha = ema_alpha
        self.max_distance = max_distance
        self.drop_after_frames = drop_after_frames
        # Each entry: {'cx': float, 'cy': float, 'last_frame': int}
        # Index in this list == stable face_idx
        self._tracks = []

    @staticmethod
    def _box_center(box):
        """Compute centroid from a bounding box [x1, y1, x2, y2, ...]."""
        return ((float(box[0]) + float(box[2])) / 2.0,
                (float(box[1]) + float(box[3])) / 2.0)

    def stabilize(self, boxes, frame_idx):
        """
        Reorder *boxes* so that the i-th element corresponds to the
        spatially consistent face_idx = i.

        Args:
            boxes: list/array of detected face bounding boxes
                   (each element: [x1, y1, x2, y2, ...])
            frame_idx: current frame number (for track age management)

        Returns:
            list of boxes in stabilized order (same length as input).
        """
        n_det = len(boxes)
        if n_det == 0:
            return []

        centers = [self._box_center(b) for b in boxes]

        # ---- First frame or all tracks lost → initialise ----------------
        active = [(i, t) for i, t in enumerate(self._tracks)
                  if frame_idx - t['last_frame'] <= self.drop_after_frames]

        if not active:
            self._tracks = [
                {'cx': cx, 'cy': cy, 'last_frame': frame_idx}
                for cx, cy in centers
            ]
            return list(boxes)

        n_active = len(active)

        # ---- Build cost matrix (Euclidean distance) ---------------------
        cost = np.empty((n_det, n_active), dtype=np.float64)
        for i, (dcx, dcy) in enumerate(centers):
            for j, (_, t) in enumerate(active):
                cost[i, j] = np.sqrt((dcx - t['cx'])**2 +
                                     (dcy - t['cy'])**2)

        # Pad to square so linear_sum_assignment works for any shape
        dim = max(n_det, n_active)
        pad = np.full((dim, dim), self.max_distance * 2, dtype=np.float64)
        pad[:n_det, :n_active] = cost
        rows, cols = linear_sum_assignment(pad)

        # ---- Interpret assignment ---------------------------------------
        # track_orig_idx → detection_idx
        assignments = {}
        matched_dets = set()

        for r, c in zip(rows, cols):
            if r < n_det and c < n_active and cost[r, c] <= self.max_distance:
                track_orig_idx = active[c][0]  # original index in self._tracks
                assignments[track_orig_idx] = r
                matched_dets.add(r)
                # Update centroid with EMA
                cx, cy = centers[r]
                old = self._tracks[track_orig_idx]
                old['cx'] = self.ema_alpha * cx + (1 - self.ema_alpha) * old['cx']
                old['cy'] = self.ema_alpha * cy + (1 - self.ema_alpha) * old['cy']
                old['last_frame'] = frame_idx

        # ---- New tracks for unmatched detections ------------------------
        for det_i in range(n_det):
            if det_i not in matched_dets:
                cx, cy = centers[det_i]
                new_idx = len(self._tracks)
                self._tracks.append(
                    {'cx': cx, 'cy': cy, 'last_frame': frame_idx})
                assignments[new_idx] = det_i

        # ---- Build reordered output (sorted by track index = face_idx) --
        reordered = []
        for track_idx in sorted(assignments.keys()):
            reordered.append(boxes[assignments[track_idx]])

        return reordered

def dump_mouth_coordinates(ver, i, face_idx, mouth_position_dump_file):
    """
        Dump all facial landmark positions into the file (68 points)
    """
    coordinates_of_points_of_interest = dict()
    # Save all 68 facial landmark points (0-67)
    # 0-16: Jaw line, 17-21: Right eyebrow, 22-26: Left eyebrow
    # 27-35: Nose, 36-41: Right eye, 42-47: Left eye, 48-67: Mouth
    all_points_index_range = range(0, 68)  # all 68 facial landmarks
    for point_index in all_points_index_range:
        coordinates_of_points_of_interest[point_index] = dict()
        for coordinate_index in range(3): # coordinates (x, y, z)
            coordinate_numeric_value = ver[coordinate_index][point_index]
            coordinates_of_points_of_interest[point_index][coordinate_index] = str(coordinate_numeric_value)
        point_of_interest_coordinate_strings = coordinates_of_points_of_interest[point_index]
        position_information_string = ",".join([str(i), str(face_idx), str(point_index), *point_of_interest_coordinate_strings.values()]) + '\n'
        mouth_position_dump_file.write(position_information_string) # write into the dumps file

def dump_face_orientation(param, i, face_idx, face_orientation_dump_file):
    """
        Dump face orientation into the file
    """
    P, pose = calc_pose(param) # P is a rotation matrix, pose are Euler angles
    pose_list_string = list(map(str, pose)) # create a list with all values converted to string types
    pose_string_stripped = ','.join([str(i), str(face_idx), *pose_list_string]) + '\n' # join with relative timestamp to form one string
    face_orientation_dump_file.write(pose_string_stripped) # Write the string into the dump file

def main(args):
    cfg = yaml.load(open(args.config), Loader=yaml.SafeLoader)

    # Init FaceBoxes and TDDFA, recommend using onnx flag
    if args.onnx:
        import os
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
        os.environ['OMP_NUM_THREADS'] = '4'

        from FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX
        from TDDFA_ONNX import TDDFA_ONNX

        face_boxes = FaceBoxes_ONNX()
        tddfa = TDDFA_ONNX(**cfg)
    else:
        gpu_mode = args.mode == 'gpu'
        tddfa = TDDFA(gpu_mode=gpu_mode, **cfg)
        face_boxes = FaceBoxes()

    # before run this line, make sure you have installed `imageio-ffmpeg`
    reader = imageio.get_reader(args.video_fp) # offline video reader
    metadata = reader.get_meta_data()
    video_framerate = metadata['fps']
    
    # Count total frames for audio processing
    total_frames = metadata.get('nframes', 0)
    if total_frames == 0:
        # Fallback: count frames manually
        print("Counting frames...")
        for _ in reader:
            total_frames += 1
        reader.close()
        reader = imageio.get_reader(args.video_fp)
    
    print(f"Video info: {total_frames} frames at {video_framerate:.2f} fps")

    # the simple implementation of average smoothing by looking ahead by n_next frames
    # assert the frames of the video >= n
    n_pre, n_next = args.n_pre, args.n_next
    n = n_pre + n_next + 1
    queue_ver = deque()
    queue_frame = deque()

    # run
    dense_flag = args.opt in ('2d_dense', '3d')
    pre_ver = None

    # Spatial face tracker: reorders detected boxes each frame so that
    # face_idx stays consistent even when FaceBoxes changes detection order.
    tracker = FaceTracker(ema_alpha=0.3, max_distance=300, drop_after_frames=30)

    with open(face_count_dump_file_path, mode='w', buffering=1) as face_count_dump_file, \
         open(face_orientation_dump_file_path, mode='w', buffering=1) as face_orientation_dump_file, \
         open(mouth_position_dump_file_path, mode='w') as mouth_position_dump_file:
        # Dump headers of the messages into dump files
        face_count_dump_file.write(",".join(["seconds","face_count\n"]))
        face_orientation_dump_file.write(",".join(["seconds","face_idx","roll","pitch","yaw\n"]))
        mouth_position_dump_file.write(",".join(["seconds","face_idx","point_type","x","y","z\n"]))
        
        for i, frame in tqdm(enumerate(reader)):
            relative_timestamp = i / video_framerate
            ver = None # this is in case we do not recognise a face, otherwise this value is overriden
            frame_bgr = frame[..., ::-1]  # RGB->BGR

            if i == 0:
                # the first frame, detect face, here we only use the first face, you can change depending on your need
                boxes = face_boxes(frame_bgr)
                #boxes = [boxes[0]]
                # Stabilize face order using spatial tracker
                boxes = tracker.stabilize(boxes, i)
                # Detect faces, get 3DMM params and roi boxes
                # If dumps are enabled, save the number of faces into a .txt file
                face_count = len(boxes) # count how many faces were detected

                param_lst, roi_box_lst = tddfa(frame_bgr, boxes)
                if face_count > 0:
                    ver = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)[0] # obtain locations of points of interest

                    # refine
                    param_lst, roi_box_lst = tddfa(frame_bgr, [ver], crop_policy='landmark')
                    ver = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)[0]


                    # padding queue
                    for _ in range(n_pre):
                        queue_ver.append(ver.copy())
                    queue_ver.append(ver.copy())

                    for _ in range(n_pre):
                        queue_frame.append(frame_bgr.copy())
                    queue_frame.append(frame_bgr.copy())
            else:
                if pre_ver is not None:
                    param_lst, roi_box_lst = tddfa(frame_bgr, [pre_ver], crop_policy='landmark')

            #    roi_box = roi_box_lst[0]
                # todo: add confidence threshold to judge the tracking is failed
            #    if abs(roi_box[2] - roi_box[0]) * abs(roi_box[3] - roi_box[1]) < 2020:
                boxes = face_boxes(frame_bgr)
            #        boxes = [boxes[0]]
                # Stabilize face order using spatial tracker
                boxes = tracker.stabilize(boxes, i)
                face_count = len(boxes)

                param_lst, roi_box_lst = tddfa(frame_bgr, boxes)

                if face_count > 0:
                    ver = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)[0]

                    queue_ver.append(ver.copy())
                    queue_frame.append(frame_bgr.copy())

            if args.dump_results is True: # dump the information about the number of faces into a text file
                face_count_dump_file.write(str(relative_timestamp) + ',' + str(face_count) + '\n') # format: one number per iteration, each in new line
                if face_count > 0: # perform dumping of mouth coordinates and face orientation for all detected faces
                    # Get reconstructed vertices for all faces
                    all_vers = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)
                    # Dump data for each detected face
                    for face_idx, (param, ver_face) in enumerate(zip(param_lst, all_vers)):
                        dump_mouth_coordinates(ver_face, relative_timestamp, face_idx, mouth_position_dump_file)
                        dump_face_orientation(param, relative_timestamp, face_idx, face_orientation_dump_file)

            pre_ver = ver  # for tracking

            # smoothing: enqueue and dequeue ops
            if len(queue_ver) >= n:
                ver_ave = np.mean(queue_ver, axis=0)

                if args.opt == '2d_sparse':
                    img_draw = cv_draw_landmark(queue_frame[n_pre], ver_ave)  # since we use padding
                elif args.opt == '2d_dense':
                    img_draw = cv_draw_landmark(queue_frame[n_pre], ver_ave, size=1)
                elif args.opt == '3d':
                    img_draw = render(queue_frame[n_pre], [ver_ave], tddfa.tri, alpha=0.7)
                else:
                    raise ValueError(f'Unknown opt {args.opt}')

                # cv2.imshow('image', img_draw)
                # k = cv2.waitKey(20)
                # if (k & 0xff == ord('q')):
                #     break

                queue_ver.popleft()
                queue_frame.popleft()
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Processes a video file with 3DDFA_V2')
    parser.add_argument('-c', '--config', type=str, default='configs/mb1_120x120.yml')
    parser.add_argument('-m', '--mode', default='cpu', type=str, help='gpu or cpu mode')
    parser.add_argument('-o', '--opt', type=str, default='2d_sparse', choices=['2d_sparse', '2d_dense', '3d'])
    parser.add_argument('-n_pre', default=1, type=int, help='the pre frames of smoothing')
    parser.add_argument('-n_next', default=1, type=int, help='the next frames of smoothing')
    parser.add_argument('--onnx', action='store_true', default=False)
    parser.add_argument('--dump_results', type=str2bool, default='false', help='whether to dump the results into a file')
    parser.add_argument('-f', '--video_fp', type=str)

    args = parser.parse_args()
    main(args)
