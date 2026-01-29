import os
import subprocess
import argparse
from pathlib import Path

def quick_extract(args):
    output_path = os.path.splitext(args.input)[0] + ".wav"
    
    # Command: ffmpeg -i input_video -vn -acodec pcm_s16le output_audio.wav
    # -vn: Disable video recording
    # -acodec pcm_s16le: Use standard WAV encoding
    command = [
        'ffmpeg', '-i', args.input, 
        '-vn', 
        '-acodec', 'pcm_f32le', 
        '-ar', '48000', # Sample rate
        '-ac', '1',     # Channels (stereo)
        output_path
    ]
    
    try:
        subprocess.run(command, check=True)
        print(f"Done! Created {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract WAV from video')
    parser.add_argument('-i', '--input', type=str, required=True, help='Input video file path')
    args = parser.parse_args()

    quick_extract(args)