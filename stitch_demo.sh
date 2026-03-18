#!/bin/bash

# Before running the script, you need to:
# 1. Set the paths below (make sure they are absolute paths)
# 2. Prepare all the repositories, including audio processing pipeline
# 3. Activate the virtual environment:  source .videmo-venv/bin/activate

# Set up variables (absolute paths)
TDDFA_V2_DIRECTORY_PATH="$(pwd)/"
VIDEO_PATH="$(pwd)/inputs/example_3.mp4"
AUDIO_PIPELINE_DIRECTORY_PATH="/work/user_data/msulewsk/data/repos/leapFox/trunk_main188/"
FACENET_DIRECTORY_PATH="$(pwd)/facenet/"

# Obtain audio from video and save to the same directory
python3 extract_wav_from_video.py -i "$VIDEO_PATH"

# Run audio pipeline
echo "Please run the required library for the audio pipeline now. Type 'done' and press Enter when finished to continue."
while true; do
    read -p "Type 'done' when complete: " user_input
    if [ "$user_input" = "done" ]; then
        break
    else
        echo "Waiting for confirmation... (type 'done' to continue)"
    fi
done

# Copy the dumps obtained from the audio pipeline
cp "$AUDIO_PIPELINE_DIRECTORY_PATH/"{vad,MQE_Metrics}".csv" dumps/
BASENAME="$(basename "${VIDEO_PATH%.*}")"
cp "$AUDIO_PIPELINE_DIRECTORY_PATH/${BASENAME}"*_txfeOut.wav dumps/

# Obtain video information from 3DDFA_V2
python3 process_video.py -f "$VIDEO_PATH" --dump_results=true

# Obtain face embeddings from the video
cd "$FACENET_DIRECTORY_PATH"
./run_identify_speakers.sh models/20180402-114759 "$VIDEO_PATH" "$TDDFA_V2_DIRECTORY_PATH/enrollment-avatars/" --threshold 0.5 -o "$TDDFA_V2_DIRECTORY_PATH/dumps/speaker_identification.csv" -m "$TDDFA_V2_DIRECTORY_PATH/dumps/mouth_position.csv"

# Render the video with all the information
cd "$TDDFA_V2_DIRECTORY_PATH"
# Extract face embeddings pre-render
python3 extract_face_embeddings.py -i "$VIDEO_PATH" --dumps_dir dumps -o face_embeddings.npz

# Reindex face IDs in dump CSVs so they are persistent across frames
python3 reindex_face_ids.py --dumps_dir dumps --embeddings face_embeddings.npz

OUTPUT_NAME="outputs/$(basename "${VIDEO_PATH%.*}")_output.${VIDEO_PATH##*.}"
DEBUG_OUTPUT_NAME="outputs/$(basename "${VIDEO_PATH%.*}")_output_debug.${VIDEO_PATH##*.}"
python3 multiface_distance_render.py -i "$VIDEO_PATH" -o "$OUTPUT_NAME" --dumps_dir dumps --vvad_model=vvad_dnn_model.pt
# python3 multiface_distance_render.py -i "$VIDEO_PATH" -o "$DEBUG_OUTPUT_NAME" --dumps_dir dumps --vvad_model=vvad_dnn_model.pt --debug-mode
echo "Done!"
