#!/usr/bin/env python3
"""Target Speaker Extraction using WeSep.

Uses the WeSep BSRNN + ECAPA model for target speaker extraction.
Accepts a mixture .wav and one or more enrollment .wav files,
outputs the extracted target speaker audio.
"""

import argparse
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
LOCAL_CACHE = SCRIPT_DIR / ".cache"

os.environ.setdefault("HF_HOME", str(LOCAL_CACHE / "huggingface"))
os.environ.setdefault("PIP_CACHE_DIR", str(LOCAL_CACHE / "pip"))

import glob

import soundfile as sf
import torch
import torchaudio

import wesep.cli.hub as wesep_hub

_original_get_model = wesep_hub.Hub.get_model.__func__


@staticmethod
def _patched_get_model(lang: str) -> str:
    """Redirect WeSep model cache from ~/.wesep/ to .cache/wesep/."""
    model_dir = os.path.join(str(LOCAL_CACHE / "wesep"), lang)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if {"avg_model.pt", "config.yaml"}.issubset(set(os.listdir(model_dir))):
        return model_dir
    model_name = wesep_hub.Hub.Assets.get(lang)
    if model_name is None:
        print(f"ERROR: Unsupported lang {lang} !!!")
        sys.exit(1)
    if model_name in wesep_hub.Hub.ModelURLs:
        model_url = wesep_hub.Hub.ModelURLs[model_name]
        wesep_hub.download(model_url, model_dir)
        return model_dir
    else:
        print(f"ERROR: No URL found for model {model_name}")
        return None


wesep_hub.Hub.get_model = _patched_get_model

import wesep


def load_and_resample(path, target_sr=16000):
    wav, sr = torchaudio.load(path)
    if sr != target_sr:
        wav = torchaudio.transforms.Resample(sr, target_sr)(wav)
        sr = target_sr
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    return wav, sr


def parse_args():
    parser = argparse.ArgumentParser(
        description="Target Speaker Extraction using WeSep",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Example usage:
  python run_wesep_tse.py \\
    --mixture inputs/mix.wav \\
    --enrollment inputs/ref.wav \\
    --output_dir outputs/
        """,
    )
    parser.add_argument(
        "--mixture", required=True,
        help="Path to the mixture audio file (WAV)",
    )
    parser.add_argument(
        "--enrollment", required=True, nargs="+",
        help="Path(s) to enrollment audio file(s) of the target speaker",
    )
    parser.add_argument(
        "--output_dir", default="./outputs/",
        help="Directory to save the extracted output (default: ./outputs/)",
    )
    parser.add_argument(
        "--lang", default="english",
        help='Model key for wesep.load_model() (default: "english")',
    )
    return parser.parse_args()


def build_output_filename(mixture_path, enrollment_paths):
    mix_stem = Path(mixture_path).stem
    enroll_stems = "_".join(Path(ep).stem for ep in enrollment_paths)
    return f"{mix_stem}_enrolled_{enroll_stems}_wesep.wav"


def main():
    args = parse_args()

    if not os.path.isfile(args.mixture):
        print(f"Error: mixture file not found: {args.mixture}", file=sys.stderr)
        sys.exit(1)
    for ep in args.enrollment:
        if not os.path.isfile(ep):
            print(f"Error: enrollment file not found: {ep}", file=sys.stderr)
            sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading WeSep model (lang={args.lang})...")
    model = wesep.load_model(args.lang)

    tmp_dir = os.path.join(args.output_dir, "_wesep_tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    mix_wav, _ = load_and_resample(args.mixture, 16000)
    mix_tmp = os.path.join(tmp_dir, "_mix_tmp.wav")
    sf.write(mix_tmp, mix_wav.squeeze(0).numpy(), 16000)

    tmp_ref_paths = []
    for i, ref_path in enumerate(args.enrollment, 1):
        wav, sr = load_and_resample(ref_path, 16000)
        tmp_path = os.path.join(tmp_dir, f"_ref_{i:02d}.wav")
        sf.write(tmp_path, wav.squeeze(0).numpy(), 16000)
        tmp_ref_paths.append(tmp_path)

    print(f"\nInput mixture  : {args.mixture}")
    print(f"Enrollment     : {args.enrollment}")
    print(f"Output dir     : {args.output_dir}")
    print()

    for i, ref_tmp in enumerate(tmp_ref_paths, 1):
        print(f"Extracting with enrollment {i}/{len(tmp_ref_paths)}...")
        speech = model.extract_speech(mix_tmp, ref_tmp)
        audio = speech[0]

        out_name = build_output_filename(args.mixture, [args.enrollment[i - 1]])
        out_path = os.path.join(args.output_dir, out_name)
        sf.write(out_path, audio, 16000)
        print(f"  Saved: {out_path}")

    for f in glob.glob(os.path.join(tmp_dir, "_*")):
        os.remove(f)
    try:
        os.rmdir(tmp_dir)
    except OSError:
        pass

    print("\nDone.")


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()
