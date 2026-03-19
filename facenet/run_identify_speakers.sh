#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
export PYTHONPATH=./src
exec python3 identify_speakers.py "$@"
