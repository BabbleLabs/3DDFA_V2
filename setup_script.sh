#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Initializing git submodules ==="
git submodule update --init --recursive

echo "=== Creating virtual environment (.videmo-venv) ==="
python3 -m venv .videmo-venv
source .videmo-venv/bin/activate

mkdir -p .cache/pip
PIP_CACHE="$SCRIPT_DIR/.cache/pip"

echo "=== Installing Python dependencies ==="
pip install --cache-dir "$PIP_CACHE" --upgrade pip
pip install --cache-dir "$PIP_CACHE" -r requirements.txt

echo "=== Installing SoloSpeech (editable, no-deps) ==="
pip install --cache-dir "$PIP_CACHE" --no-deps -e SoloSpeech

echo "=== Installing WeSep (editable, no-deps) ==="
pip install --cache-dir "$PIP_CACHE" --no-deps -e wesep

echo "=== Installing descript-audiotools (no-deps, protobuf conflict) ==="
pip install --cache-dir "$PIP_CACHE" --no-deps 'git+https://github.com/descriptinc/audiotools'

echo "=== Installing WeSpeaker (no-deps) ==="
pip install --cache-dir "$PIP_CACHE" --no-deps 'git+https://github.com/wenet-e2e/wespeaker.git'

# WeSpeaker's __init__.py eagerly imports its full CLI which pulls in
# dozens of optional frontends (whisper, s3prl, peft, ...).  WeSep only
# uses wespeaker.models.speaker_model, so we replace the __init__ with
# lazy wrappers to avoid unnecessary import errors.
echo "=== Patching WeSpeaker __init__.py for lazy imports ==="
python3 - <<'PYEOF'
import site, os, pathlib
for sp in site.getsitepackages():
    init = pathlib.Path(sp) / "wespeaker" / "__init__.py"
    if init.exists():
        init.write_text(
            "def load_model(*a, **kw):\n"
            "    from wespeaker.cli.speaker import load_model as _f\n"
            "    return _f(*a, **kw)\n"
            "\n"
            "def load_model_pt(*a, **kw):\n"
            "    from wespeaker.cli.speaker import load_model_pt as _f\n"
            "    return _f(*a, **kw)\n"
        )
        print(f"Patched {init}")
        break
else:
    print("WARNING: wespeaker __init__.py not found, skipping patch")
PYEOF

echo "=== Running Cython build script for 3DDFA_v2 ==="
sh ./build.sh

deactivate

echo ""
echo "=== Setup complete ==="
echo "Activate the environment with:  source .videmo-venv/bin/activate"
