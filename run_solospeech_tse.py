#!/usr/bin/env python3
"""
Target Speaker Extraction using SoloSpeech.

Extracts a target speaker from a mixture audio file given one or more
enrollment audio samples of that speaker.

Based on SoloSpeech: Enhancing Intelligibility and Quality in Target Speech
Extraction through a Cascaded Generative Pipeline (arXiv 2505.19314).
"""

import argparse
import os
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
LOCAL_CACHE = SCRIPT_DIR / ".cache"
os.environ.setdefault("HF_HOME", str(LOCAL_CACHE / "huggingface"))
os.environ.setdefault("PIP_CACHE_DIR", str(LOCAL_CACHE / "pip"))

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import yaml
from diffusers import DDIMScheduler

import huggingface_hub
_orig_hf_hub_download = huggingface_hub.hf_hub_download
def _patched_hf_hub_download(*args, **kwargs):
    kwargs.pop("use_auth_token", None)
    return _orig_hf_hub_download(*args, **kwargs)
huggingface_hub.hf_hub_download = _patched_hf_hub_download

from speechbrain.pretrained.interfaces import Pretrained

from solospeech.corrector.fastgeco.model import ScoreModel
from solospeech.corrector.geco.util.other import pad_spec
from solospeech.model.solospeech.conditioners import SoloSpeech_TSE
from solospeech.vae_modules.autoencoder_wrapper import Autoencoder


SAMPLE_RATE = 16000
MODEL_DIR = SCRIPT_DIR / ".cache" / "solospeech-models"


class SpeakerEncoder(Pretrained):
    """ECAPA-TDNN speaker encoder for candidate selection."""

    MODULES_NEEDED = [
        "compute_features",
        "mean_var_norm",
        "embedding_model",
    ]

    def encode_batch(self, wavs, wav_lens=None, normalize=False):
        if len(wavs.shape) == 1:
            wavs = wavs.unsqueeze(0)
        if wav_lens is None:
            wav_lens = torch.ones(wavs.shape[0], device=self.device)
        wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)
        wavs = wavs.float()
        feats = self.mods.compute_features(wavs)
        feats = self.mods.mean_var_norm(feats, wav_lens)
        embeddings = self.mods.embedding_model(feats, wav_lens)
        if normalize:
            embeddings = self.hparams.mean_var_norm_emb(
                embeddings,
                torch.ones(embeddings.shape[0], device=self.device),
            )
        return embeddings


def load_models(device):
    """Load pretrained checkpoints from the local .cache/solospeech-models dir."""
    model_dir = MODEL_DIR
    if not model_dir.exists():
        print(
            f"Error: model directory not found: {model_dir}\n"
            "Download the models first (see README or plan).",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Loading models from {model_dir}")
    tse_config_path = os.path.join(model_dir, "config_extractor.yaml")
    vae_config_path = os.path.join(model_dir, "config_compressor.json")
    autoencoder_ckpt = os.path.join(model_dir, "compressor.ckpt")
    tse_ckpt = os.path.join(model_dir, "extractor.pt")
    geco_ckpt = os.path.join(model_dir, "corrector.ckpt")

    with open(tse_config_path, "r") as fp:
        tse_config = yaml.safe_load(fp)

    print("Loading compressor (VAE)...")
    autoencoder = Autoencoder(
        autoencoder_ckpt, vae_config_path, "stft_vae", quantization_first=True
    )
    autoencoder.eval().to(device)

    print("Loading extractor (diffusion)...")
    tse_model = SoloSpeech_TSE(
        tse_config["diffwrap"]["UDiT"],
        tse_config["diffwrap"]["ViT"],
    ).to(device)
    tse_model.load_state_dict(torch.load(tse_ckpt, map_location=device)["model"])
    tse_model.eval()

    print("Loading corrector (GeCo)...")
    geco_model = ScoreModel.load_from_checkpoint(
        geco_ckpt, batch_size=1, num_workers=0, kwargs=dict(gpu=False)
    )
    geco_model.eval(no_ema=False)
    geco_model.to(device)

    print("Loading speaker encoder (ECAPA-TDNN)...")
    ecapa_dir = str(LOCAL_CACHE / "ecapa-tdnn")
    speaker_encoder = SpeakerEncoder.from_hparams(
        source=ecapa_dir,
        savedir=ecapa_dir,
    )

    noise_scheduler = DDIMScheduler(**tse_config["ddim"]["diffusers"])
    latents = torch.randn((1, 128, 128), device=device)
    noise = torch.randn(latents.shape).to(device)
    timesteps = torch.randint(
        0,
        noise_scheduler.config.num_train_timesteps,
        (noise.shape[0],),
        device=device,
    ).long()
    _ = noise_scheduler.add_noise(latents, noise, timesteps)

    return autoencoder, tse_model, geco_model, speaker_encoder, noise_scheduler


def run_diffusion(
    tse_model, autoencoder, std, scheduler, device, mixture, reference,
    lengths, reference_lengths, ddim_steps=200, eta=0, seed=42,
):
    """Run the diffusion extraction process."""
    with torch.no_grad():
        generator = torch.Generator(device=device).manual_seed(seed)
        scheduler.set_timesteps(ddim_steps)
        tse_pred = torch.randn(mixture.shape, generator=generator, device=device)

        for t in scheduler.timesteps:
            tse_pred = scheduler.scale_model_input(tse_pred, t)
            model_output, _ = tse_model(
                x=tse_pred,
                timesteps=t,
                mixture=mixture,
                reference=reference,
                x_len=lengths,
                ref_len=reference_lengths,
            )
            tse_pred = scheduler.step(
                model_output=model_output,
                timestep=t,
                sample=tse_pred,
                eta=eta,
                generator=generator,
            ).prev_sample

        tse_pred = autoencoder(
            embedding=tse_pred.transpose(2, 1), std=std
        ).squeeze(1)

    return tse_pred


def run_corrector(geco_model, pred, mixture_wav):
    """Apply GeCo corrector to refine the extraction."""
    min_len = min(pred.shape[-1], mixture_wav.shape[-1])
    x = pred[..., :min_len]
    m = mixture_wav[..., :min_len]
    norm_factor = m.abs().max()
    x = x / norm_factor
    m = m / norm_factor

    X = torch.unsqueeze(
        geco_model._forward_transform(geco_model._stft(x.cuda())), 0
    )
    X = pad_spec(X)
    M = torch.unsqueeze(
        geco_model._forward_transform(geco_model._stft(m.cuda())), 0
    )
    M = pad_spec(M)

    timesteps = torch.linspace(0.5, 0.03, 1, device=M.device)
    std = geco_model.sde._std(
        0.5 * torch.ones((M.shape[0],), device=M.device)
    )
    z = torch.randn_like(M)
    X_t = M + z * std[:, None, None, None]

    for idx in range(len(timesteps)):
        t = timesteps[idx]
        dt = (
            t - timesteps[idx + 1]
            if idx != len(timesteps) - 1
            else timesteps[-1]
        )
        with torch.no_grad():
            f, g = geco_model.sde.sde(X_t, t, M)
            vec_t = torch.ones(M.shape[0], device=M.device) * t
            score = geco_model.forward(
                X_t, vec_t, M, X, vec_t[:, None, None, None]
            )
            mean_x_tm1 = X_t - (f - g**2 * score) * dt
            if idx == len(timesteps) - 1:
                X_t = mean_x_tm1
                break
            z = torch.randn_like(X)
            X_t = mean_x_tm1 + z * g * torch.sqrt(dt)

    sample = X_t.squeeze()
    x_hat = geco_model.to_audio(sample.squeeze(), min_len)
    x_hat = x_hat * norm_factor / x_hat.abs().max()
    return x_hat.detach().cpu().squeeze().numpy()


def extract_speaker(
    mixture_path,
    enrollment_paths,
    autoencoder,
    tse_model,
    geco_model,
    speaker_encoder,
    noise_scheduler,
    device,
    num_candidates=4,
    num_infer_steps=200,
    seed=42,
):
    """Full TSE pipeline: load audio, extract, correct, return waveform.

    Candidates are processed one at a time to avoid GPU OOM on large inputs.
    """
    print(f"\nLoading mixture: {mixture_path}")
    mixture, _ = librosa.load(mixture_path, sr=SAMPLE_RATE)

    enrollment_segments = []
    for ep in enrollment_paths:
        print(f"Loading enrollment: {ep}")
        audio, _ = librosa.load(ep, sr=SAMPLE_RATE)
        enrollment_segments.append(audio)

    reference = np.concatenate(enrollment_segments)
    print(
        f"Combined enrollment length: {len(reference) / SAMPLE_RATE:.2f}s "
        f"({len(enrollment_paths)} file(s))"
    )

    reference_wav = reference
    reference_t = torch.tensor(reference).unsqueeze(0).to(device)

    with torch.no_grad():
        reference_latent, _ = autoencoder(audio=reference_t.unsqueeze(1))
        ref_len = torch.LongTensor([reference_latent.shape[-1]]).to(device)

        mixture_t = torch.tensor(mixture).unsqueeze(0).to(device)
        mixture_wav = mixture_t
        mixture_latent, std = autoencoder(audio=mixture_t.unsqueeze(1))
        mix_len = torch.LongTensor([mixture_latent.shape[-1]]).to(device)

    print(
        f"Running diffusion extraction ({num_infer_steps} steps, "
        f"{num_candidates} candidates, sequential)..."
    )
    start = time.time()

    with torch.no_grad():
        emb_ref = speaker_encoder.encode_batch(
            torch.tensor(reference_wav)
        ).squeeze()

    best_score = -float("inf")
    best_pred = None

    for k in range(num_candidates):
        cand_seed = seed + k
        print(f"  Candidate {k}/{num_candidates} (seed={cand_seed})...", end=" ")

        pred_wav = run_diffusion(
            tse_model, autoencoder, std, noise_scheduler, device,
            mixture_latent.transpose(2, 1), reference_latent.transpose(2, 1),
            mix_len, ref_len,
            ddim_steps=num_infer_steps, eta=0, seed=cand_seed,
        )

        with torch.no_grad():
            emb_pred = speaker_encoder.encode_batch(pred_wav).squeeze()
            score = F.cosine_similarity(
                emb_pred.unsqueeze(0), emb_ref.unsqueeze(0), dim=1
            ).item()

        print(f"cosine sim = {score:.4f}")

        if score > best_score:
            best_score = score
            best_pred = pred_wav

    print(f"Best candidate score: {best_score:.4f}")
    print("Running corrector...")
    output = run_corrector(geco_model, best_pred, mixture_wav)

    elapsed = time.time() - start
    audio_len = len(output) / SAMPLE_RATE
    print(f"Done. Audio length: {audio_len:.2f}s, RTF: {elapsed / audio_len:.4f}")

    return output


def build_output_filename(mixture_path, enrollment_paths):
    """Build a descriptive output filename from inputs."""
    mix_stem = Path(mixture_path).stem
    enroll_stems = "_".join(Path(ep).stem for ep in enrollment_paths)
    return f"{mix_stem}_enrolled_{enroll_stems}_solospeech.wav"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Target Speaker Extraction using SoloSpeech",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Example usage:
  python run_solospeech_tse.py \\
    --mixture inputs/example_4.wav \\
    --enrollment inputs/example_4_first10s.wav \\
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
        "--num_candidates", type=int, default=4,
        help="Number of diffusion candidates to generate (default: 4)",
    )
    parser.add_argument(
        "--num_infer_steps", type=int, default=200,
        help="Number of DDIM diffusion steps (default: 200)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--gpu", type=int, default=0,
        help="GPU device index to use (default: 0)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    for ep in args.enrollment:
        if not os.path.isfile(ep):
            print(f"Error: enrollment file not found: {ep}", file=sys.stderr)
            sys.exit(1)
    if not os.path.isfile(args.mixture):
        print(f"Error: mixture file not found: {args.mixture}", file=sys.stderr)
        sys.exit(1)

    device = (
        f"cuda:{args.gpu}"
        if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    autoencoder, tse_model, geco_model, speaker_encoder, noise_scheduler = (
        load_models(device)
    )

    output = extract_speaker(
        args.mixture,
        args.enrollment,
        autoencoder,
        tse_model,
        geco_model,
        speaker_encoder,
        noise_scheduler,
        device,
        num_candidates=args.num_candidates,
        num_infer_steps=args.num_infer_steps,
        seed=args.seed,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    out_name = build_output_filename(args.mixture, args.enrollment)
    out_path = os.path.join(args.output_dir, out_name)
    sf.write(out_path, output, SAMPLE_RATE)
    print(f"\nSaved extracted audio to: {out_path}")


if __name__ == "__main__":
    main()
