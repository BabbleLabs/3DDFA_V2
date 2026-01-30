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


def draw_facial_landmarks(frame, landmarks_df, timestamp, face_idx=0):
    """
    Draw facial landmarks on the frame
    
    Args:
        frame: Video frame (BGR format)
        landmarks_df: DataFrame with landmark data (columns: seconds, face_idx, point_type, x, y, z)
        timestamp: Current frame timestamp
        face_idx: Index of the face to draw landmarks for
    
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
    
    # Convert landmarks to dictionary for easy access
    points = {}
    for _, row in frame_landmarks.iterrows():
        pt_idx = int(row['point_type'])
        points[pt_idx] = (int(row['x']), int(row['y']))
    
    # Define connections between landmarks (68-point facial landmark model)
    connections = [
        # Jaw line: 0-16
        list(zip(range(0, 16), range(1, 17))),
        # Right eyebrow: 17-21
        list(zip(range(17, 21), range(18, 22))),
        # Left eyebrow: 22-26
        list(zip(range(22, 26), range(23, 27))),
        # Nose bridge: 27-30
        list(zip(range(27, 30), range(28, 31))),
        # Nose bottom: 31-35
        list(zip(range(31, 35), range(32, 36))),
        # Right eye: 36-41 (loop)
        list(zip(range(36, 41), range(37, 42))) + [(41, 36)],
        # Left eye: 42-47 (loop)
        list(zip(range(42, 47), range(43, 48))) + [(47, 42)],
        # Outer mouth: 48-59 (loop)
        list(zip(range(48, 59), range(49, 60))) + [(59, 48)],
        # Inner mouth: 60-67 (loop)
        list(zip(range(60, 67), range(61, 68))) + [(67, 60)],
    ]
    
    # Draw connecting lines first (so points appear on top)
    line_color = (255, 255, 255)  # White
    line_thickness = 1
    
    for connection_group in connections:
        for pt1_idx, pt2_idx in connection_group:
            if pt1_idx in points and pt2_idx in points:
                cv2.line(frame, points[pt1_idx], points[pt2_idx], line_color, line_thickness, cv2.LINE_AA)
    
    # Draw landmark points
    point_color = (255, 255, 255)  # White
    point_radius = 2
    point_thickness = -1  # Filled
    
    for pt_idx, pt_coord in points.items():
        cv2.circle(frame, pt_coord, point_radius, point_color, point_thickness, cv2.LINE_AA)
    
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


def draw_dashboard(frame, frame_idx, face_count, face_distances, timestamp, audio_data=None, vad_data=None):
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
    
    # Fixed panel height for 2 faces + audio section + VAD graph (always constant)
    # Calculate based on actual content:
    # Title: 30 + 35 = 65
    # Info lines (Frame, Time, Faces): 3 * 38 = 114
    # Padding + separator: 10 + 28 = 38
    # Face slots (2 faces): 2 * 38 = 76
    # Audio section: 30 (separator) + 250 (pie charts) = 280
    # VAD section: 30 (separator) + 70 (graph) = 100
    # Bottom padding: 20
    line_height = 38
    panel_height = 65 + 114 + 38 + 76 + 280 + 100 + 20  # = 693 pixels
    
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
    
    # Draw VAD (Voice Activity Detection) graph section
    if vad_data is not None:
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


def render_video_with_dashboard(video_path, df_face_count, df_mouth, df_mqe, df_vad, output_path, show_markers=True):
    """
    Render annotated video with dashboard overlay
    
    Args:
        video_path: Path to input video
        df_face_count: DataFrame with face count data
        df_mouth: DataFrame with facial landmark data (all 68 points including x,y,z coordinates)
        df_mqe: DataFrame with MQE metrics (5-second segments)
        df_vad: DataFrame with VAD (Voice Activity Detection) data
        output_path: Path to output video
        show_markers: Whether to show facial landmark markers
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
    
    # Store last valid audio data to persist after MQE data ends
    last_audio_data = None
    
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
        
        # Draw dashboard overlay
        frame_bgr = draw_dashboard(frame_bgr, frame_idx, face_count, face_distances, timestamp, audio_data, vad_data)
        
        # Draw facial landmarks for all detected faces
        if show_markers and face_count > 0:
            for face_idx in range(face_count):
                frame_bgr = draw_facial_landmarks(frame_bgr, df_mouth, timestamp, face_idx)
        
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
    
    print(f"  Loaded {len(df_face_count)} frames")
    
    unique_faces = sorted(df_mouth['face_idx'].unique())
    print(f"  Found {len(unique_faces)} unique face indices: {unique_faces}")
    
    # Render video
    render_video_with_dashboard(
        video_path, 
        df_face_count, 
        df_mouth, 
        df_mqe,
        df_vad,
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
