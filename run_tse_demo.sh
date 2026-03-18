#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

usage() {
    cat <<'USAGE'
Usage: run_tse_demo.sh --model MODEL --mixture FILE --enrollment FILE [FILE ...] [--output FILE|DIR] [extra args...]

Target Speaker Extraction demo.  Extracts a target speaker from a
mixture audio file using the specified model.

Required arguments:
  --model MODEL        One of: solospeech, speechbrain, wesep
  --mixture FILE       Path to the mixture .wav file
  --enrollment FILE    One or more enrollment .wav files for the target speaker

Optional arguments:
  --output PATH        Output path.  For speechbrain this is a file path;
                       for solospeech/wesep this is a directory (default: ./outputs/)

Any additional arguments are forwarded to the underlying Python script.

Examples:
  ./run_tse_demo.sh --model solospeech  --mixture mix.wav --enrollment ref.wav
  ./run_tse_demo.sh --model speechbrain --mixture mix.wav --enrollment ref.wav --output out.wav
  ./run_tse_demo.sh --model wesep       --mixture mix.wav --enrollment ref1.wav ref2.wav
USAGE
    exit 1
}

MODEL=""
MIXTURE=""
OUTPUT=""
ENROLLMENT=()
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            MODEL="$2"; shift 2 ;;
        --mixture)
            MIXTURE="$2"; shift 2 ;;
        --enrollment)
            shift
            while [[ $# -gt 0 && "$1" != --* ]]; do
                ENROLLMENT+=("$1"); shift
            done
            ;;
        --output)
            OUTPUT="$2"; shift 2 ;;
        -h|--help)
            usage ;;
        *)
            EXTRA_ARGS+=("$1"); shift ;;
    esac
done

if [[ -z "$MODEL" || -z "$MIXTURE" || ${#ENROLLMENT[@]} -eq 0 ]]; then
    echo "Error: --model, --mixture, and --enrollment are required." >&2
    usage
fi

case "$MODEL" in
    solospeech)
        OUTPUT_ARGS=()
        if [[ -n "$OUTPUT" ]]; then
            OUTPUT_ARGS=(--output_dir "$OUTPUT")
        fi
        python3 "$SCRIPT_DIR/run_solospeech_tse.py" \
            --mixture "$MIXTURE" \
            --enrollment "${ENROLLMENT[@]}" \
            "${OUTPUT_ARGS[@]}" \
            "${EXTRA_ARGS[@]}"
        ;;
    speechbrain)
        OUTPUT_ARGS=()
        if [[ -n "$OUTPUT" ]]; then
            OUTPUT_ARGS=(--output "$OUTPUT")
        fi
        python3 "$SCRIPT_DIR/extract_target_speaker.py" \
            --input "$MIXTURE" \
            --targets "${ENROLLMENT[@]}" \
            "${OUTPUT_ARGS[@]}" \
            "${EXTRA_ARGS[@]}"
        ;;
    wesep)
        OUTPUT_ARGS=()
        if [[ -n "$OUTPUT" ]]; then
            OUTPUT_ARGS=(--output_dir "$OUTPUT")
        fi
        python3 "$SCRIPT_DIR/run_wesep_tse.py" \
            --mixture "$MIXTURE" \
            --enrollment "${ENROLLMENT[@]}" \
            "${OUTPUT_ARGS[@]}" \
            "${EXTRA_ARGS[@]}"
        ;;
    *)
        echo "Error: unknown model '$MODEL'. Choose from: solospeech, speechbrain, wesep" >&2
        exit 1
        ;;
esac
