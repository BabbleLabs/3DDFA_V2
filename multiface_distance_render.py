#!/usr/bin/env python3
# coding: utf-8

"""
Multi-Face Distance Video Renderer
Renders an annotated video with face distance dashboard overlay
Uses pre-computed CSV dumps for fast rendering
"""

import argparse
import pandas as pd
import numpy as np
import cv2
import imageio
from pathlib import Path
from tqdm import tqdm
import subprocess
import os

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

def calculate_distance(z_value):
    """
    Convert z-coordinate to approximate distance in meters
    z is depth from camera, inverse relationship
    """
    if z_value <= 0:
        return 0
    return 100.0 / z_value


def draw_dashboard(frame, frame_idx, face_count, face_distances, timestamp):
    """
    Draw dashboard overlay on frame
    
    Args:
        frame: Video frame (numpy array)
        frame_idx: Current frame number
        face_count: Number of faces detected
        face_distances: Dict mapping face_idx to distance in cm
        timestamp: Time in seconds
    """
    # Ensure frame is contiguous for OpenCV operations
    if not frame.flags['C_CONTIGUOUS']:
        frame = np.ascontiguousarray(frame)
    
    height, width = frame.shape[:2]
    
    # Dashboard dimensions - FIXED SIZE for consistent appearance
    panel_width = 300
    panel_margin = 20
    panel_x = width - panel_width - panel_margin
    panel_y = panel_margin
    
    # Fixed panel height for 2 faces (always constant)
    # Calculate based on actual content:
    # Title: 30 + 35 = 65
    # Info lines (Frame, Time, Faces): 3 * 38 = 114
    # Padding + separator: 10 + 28 = 38
    # Face slots (2 faces): 2 * 38 = 76
    # Bottom padding: 20
    line_height = 38
    panel_height = 65 + 114 + 38 + 76 + 20  # = 313 pixels
    
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
    
    # Always draw slots for 2 faces (fixed layout)
    for slot_idx in range(2):
        # Check if we have data for this face
        if slot_idx in face_distances:
            # Face detected - show with color
            face_idx = slot_idx
            distance = face_distances[face_idx]
            color_idx = face_idx % len(FACE_COLORS_BGR)
            face_color = FACE_COLORS_BGR[color_idx]
            
            # Draw colored indicator dot
            cv2.circle(frame, 
                      (panel_x + 22, current_y - 6), 
                      7, face_color, -1)
            
            # Draw face distance
            dist_text = f"Face {face_idx}: {distance:.1f} m"
            cv2.putText(frame, dist_text, 
                       (panel_x + 40, current_y),
                       font, text_font_scale, text_color, text_thickness)
        else:
            # No face in this slot - show placeholder
            gray_color = (100, 100, 100)
            
            # Draw gray indicator dot
            cv2.circle(frame, 
                      (panel_x + 22, current_y - 6), 
                      7, gray_color, -1)
            
            # Draw placeholder text
            placeholder_text = f"Face {slot_idx}: ---"
            cv2.putText(frame, placeholder_text, 
                       (panel_x + 40, current_y),
                       font, text_font_scale, gray_color, text_thickness)
        
        current_y += line_height
    
    return frame


def draw_face_boxes(frame, face_data, timestamp):
    """
    Draw bounding boxes around detected faces
    
    Args:
        face_data: DataFrame with face position data for current timestamp
        timestamp: Current time in seconds
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
        
        # Get color for this face
        color_idx = int(face_idx) % len(FACE_COLORS_BGR)
        color = FACE_COLORS_BGR[color_idx]
        
        # If we have x,y coordinates, draw a marker
        if not face_subset.empty and 'x' in face_subset.columns:
            x = int(face_subset['x'].values[0])
            y = int(face_subset['y'].values[0])
            
            # Draw circle marker
            cv2.circle(frame, (x, y), 8, color, 2)
            cv2.circle(frame, (x, y), 3, color, -1)
            
            # Draw face label above marker
            label = f"F{int(face_idx)}"
            cv2.putText(frame, label, 
                       (x - 15, y - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
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


def render_video_with_dashboard(video_path, df_face_count, df_mouth, output_path, show_markers=True):
    """
    Render annotated video with dashboard overlay
    
    Args:
        video_path: Path to input video
        df_face_count: DataFrame with face count data
        df_mouth: DataFrame with mouth position data (contains z coordinate)
        output_path: Path to output video
        show_markers: Whether to show face position markers
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
            frame_bgr = draw_face_boxes(frame_bgr, mouth_data, timestamp)
        
        # Draw dashboard overlay
        frame_bgr = draw_dashboard(frame_bgr, frame_idx, face_count, face_distances, timestamp)
        
        # Convert back to RGB for imageio
        frame_rgb = frame_bgr[..., ::-1]
        
        # Write frame
        writer.append_data(frame_rgb)
    
    # Clean up
    writer.close()
    reader.close()
    
    print(f"\n✓ Video rendering complete")
    
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
    print("Multi-Face Distance Video Renderer")
    print("="*60)
    
    # Read CSV files
    print("\nLoading CSV data...")
    df_face_count = pd.read_csv(dumps_dir / 'face_count.csv')
    df_mouth = pd.read_csv(dumps_dir / 'mouth_position.csv')
    
    print(f"  Loaded {len(df_face_count)} frames")
    
    unique_faces = sorted(df_mouth['face_idx'].unique())
    print(f"  Found {len(unique_faces)} unique face indices: {unique_faces}")
    
    # Render video
    render_video_with_dashboard(
        video_path, 
        df_face_count, 
        df_mouth, 
        output_path,
        show_markers=args.show_markers
    )
    
    print("\n" + "="*60)
    print("Done!")
    print("="*60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Render video with face distance dashboard overlay',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python3 multiface_distance_render.py -i inputs/video.mp4 -o output_annotated.mp4

  # Specify custom dumps directory
  python3 multiface_distance_render.py -i video.mp4 -o output.mp4 --dumps_dir=dumps

  # Without face position markers
  python3 multiface_distance_render.py -i video.mp4 -o output.mp4 --no-markers

Output:
  Creates an MP4 video with:
  - Dashboard panel showing frame info, face count, distances
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
    parser.add_argument('--show-markers', dest='show_markers', action='store_true',
                       help='Show face position markers on video (default)')
    parser.add_argument('--no-markers', dest='show_markers', action='store_false',
                       help='Hide face position markers')
    parser.set_defaults(show_markers=True)
    
    args = parser.parse_args()
    main(args)
