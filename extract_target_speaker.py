#!/usr/bin/env python3
"""Target speaker extraction using SpeechBrain.

Uses a two-stage pipeline:
  1. Blind source separation via SepFormer.
  2. Post-hoc speaker selection via ECAPA-TDNN embeddings and cosine
     similarity between separated sources and target enrollment audio.

Sources that best match the target speaker(s) are summed to produce
the output waveform.
"""

import argparse
import os
import sys

import torch
import torch.nn.functional as F
import torchaudio


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract target speaker(s) from a mixture using SpeechBrain.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Path to the mixture .wav audio file.",
    )
    parser.add_argument(
        "-t", "--targets",
        nargs="+",
        required=True,
        help="One or more target speaker enrollment .wav files.",
    )
    parser.add_argument(
        "-o", "--output",
        default="./outputs/extracted_output.wav",
        help="Path for the output .wav file.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Compute device ('cpu' or 'cuda').",
    )
    parser.add_argument(
        "--sep-model",
        default="speechbrain/sepformer-whamr16k",
        help="HuggingFace source for the SepFormer separation model.",
    )
    parser.add_argument(
        "--spk-model",
        default="speechbrain/spkrec-ecapa-voxceleb",
        help="HuggingFace source for the speaker embedding model.",
    )
    return parser.parse_args()


def load_audio_mono(path, target_sr=None):
    """Load an audio file, mix down to mono, and optionally resample.

    Returns
    -------
    waveform : torch.Tensor  – shape [1, time]
    sample_rate : int
    """
    waveform, sr = torchaudio.load(path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if target_sr is not None and sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
        sr = target_sr
    return waveform, sr


def compute_embedding(encoder, waveform):
    """Compute a speaker embedding for a waveform tensor [1, time]."""
    emb = encoder.encode_batch(waveform)
    return F.normalize(emb.squeeze(), dim=0)


def main():
    args = parse_args()

    if not os.path.isfile(args.input):
        print(f"Error: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    for t in args.targets:
        if not os.path.isfile(t):
            print(f"Error: target file not found: {t}", file=sys.stderr)
            sys.exit(1)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    from speechbrain.inference.separation import SepformerSeparation
    from speechbrain.inference.speaker import EncoderClassifier

    print("Loading SepFormer separation model ...")
    separator = SepformerSeparation.from_hparams(
        source=args.sep_model,
        savedir=f"pretrained_models/{args.sep_model.split('/')[-1]}",
        run_opts={"device": args.device},
    )
    sep_sr = separator.hparams.sample_rate

    print("Loading ECAPA-TDNN speaker encoder ...")
    spk_encoder = EncoderClassifier.from_hparams(
        source=args.spk_model,
        savedir=f"pretrained_models/{args.spk_model.split('/')[-1]}",
        run_opts={"device": args.device},
    )
    spk_sr = 16000

    print(f"\nInput mixture  : {args.input}")
    print(f"Target files   : {args.targets}")
    print(f"Output         : {args.output}")
    print(f"Separation SR  : {sep_sr} Hz")
    print()

    mix_wav, _ = load_audio_mono(args.input, target_sr=sep_sr)
    mix_wav = mix_wav.to(args.device)

    print("Running blind source separation ...")
    with torch.no_grad():
        est_sources = separator.separate_batch(mix_wav)
    est_sources = est_sources / est_sources.abs().max(dim=1, keepdim=True)[0]

    num_sources = est_sources.shape[-1]
    print(f"  Separated into {num_sources} source(s).")

    mix_for_spk, _ = load_audio_mono(args.input, target_sr=spk_sr)
    mix_for_spk = mix_for_spk.to(args.device)

    print("Computing speaker embeddings for separated sources ...")
    source_embeddings = []
    with torch.no_grad():
        for s in range(num_sources):
            src = est_sources[:, :, s]
            if sep_sr != spk_sr:
                src = torchaudio.functional.resample(src, sep_sr, spk_sr)
            emb = compute_embedding(spk_encoder, src)
            source_embeddings.append(emb)

    print("Computing speaker embeddings for target enrollment(s) ...")
    target_embeddings = []
    with torch.no_grad():
        for tpath in args.targets:
            twav, _ = load_audio_mono(tpath, target_sr=spk_sr)
            twav = twav.to(args.device)
            emb = compute_embedding(spk_encoder, twav)
            target_embeddings.append(emb)

    print("Matching targets to separated sources ...")
    selected_indices = set()
    for ti, temb in enumerate(target_embeddings):
        best_idx = -1
        best_sim = -1.0
        for si, semb in enumerate(source_embeddings):
            sim = F.cosine_similarity(temb.unsqueeze(0), semb.unsqueeze(0)).item()
            print(f"  target[{ti}] vs source[{si}]: cosine={sim:.4f}")
            if sim > best_sim:
                best_sim = sim
                best_idx = si
        print(f"  -> target[{ti}] best match: source[{best_idx}] (cosine={best_sim:.4f})")
        if best_sim < 0.1:
            print(f"  WARNING: low similarity for target[{ti}]; match may be unreliable.")
        selected_indices.add(best_idx)

    print(f"\nSelected source indices: {sorted(selected_indices)}")

    output_wav = torch.zeros_like(est_sources[:, :, 0])
    for idx in selected_indices:
        output_wav += est_sources[:, :, idx]

    peak = output_wav.abs().max()
    if peak > 0:
        output_wav = output_wav / peak

    output_wav = output_wav.detach().cpu()
    torchaudio.save(args.output, output_wav, sep_sr)
    print(f"\nSaved output to: {args.output}")
    print("Done.")


if __name__ == "__main__":
    main()
