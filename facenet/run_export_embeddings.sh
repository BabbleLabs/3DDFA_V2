#!/usr/bin/env bash
# Run export_embeddings.py with correct PYTHONPATH.
# Requires the .videmo-venv virtual environment to be activated.
# Usage: ./run_export_embeddings.sh <model_dir> <data_dir> [options...]
# Example: ./run_export_embeddings.sh ~/models/facenet/20170216-091149 ~/datasets/lfw/mylfw

set -e
cd "$(dirname "$0")"
export PYTHONPATH=./src
exec python3 contributed/export_embeddings.py "$@"
