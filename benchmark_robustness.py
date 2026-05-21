"""Robustness benchmark: evaluate Baseline vs Proposed under noisy conditions.

Validation set: Fold-5 originals only (no _aug_ files) — identical to train.py split.
Conditions: Clean (no noise), SNR 20 dB, 10 dB, 0 dB.
Results saved to benchmark_results.json.
"""
import glob
import json
import os
import random

import torch
from sklearn.metrics import f1_score, precision_score, recall_score

from models.sed_model import CRNN
from utils.audio_utils import (
    FIXED_FRAMES,
    N_MELS,
    build_mel_transform,
    load_audio,
    mix_noise_with_snr,
    waveform_to_logmel,
)

CLASS_DIRS = {"coughing": 0, "sneezing": 1, "infant_crying": 2}
NOISE_DIRS = [
    "data/noise/vehicle_engine",
    "data/noise/environmental",
    "data/noise/hvac",
]
MODELS = {
    "Baseline": "models/sed_baseline.pth",
    "Proposed": "models/sed_robust_best.pth",
}
SNR_CONDITIONS = [None, 20.0, 10.0, 0.0]  # None = clean


def build_val_samples(data_root="data"):
    """Fold 5 originals strictly — no _aug_ files."""
    samples = []
    for name, label in CLASS_DIRS.items():
        for wav in sorted(glob.glob(os.path.join(data_root, name, "*.wav"))):
            basename = os.path.basename(wav)
            if basename.startswith("5-") and "_aug_" not in basename:
                samples.append((wav, label))
    return samples


def load_noise_files():
    files = []
    for d in NOISE_DIRS:
        files.extend(glob.glob(os.path.join(d, "*.wav")))
    return files


def pad_or_truncate(logmel):
    t = logmel.shape[-1]
    if t >= FIXED_FRAMES:
        return logmel[:, :, :FIXED_FRAMES]
    return torch.nn.functional.pad(logmel, (0, FIXED_FRAMES - t))


def evaluate_condition(model, val_samples, noise_files, snr_db, mel_transform, device):
    model.eval()
    all_preds, all_labels = [], []
    rng = random.Random(0)

    with torch.no_grad():
        for path, label in val_samples:
            waveform = load_audio(path)
            if snr_db is not None and noise_files:
                noise = load_audio(rng.choice(noise_files))
                waveform = mix_noise_with_snr(waveform, noise, snr_db)
            logmel = pad_or_truncate(waveform_to_logmel(waveform, mel_transform))
            logits = model(logmel.unsqueeze(0).to(device))
            all_preds.append(logits.argmax(dim=1).item())
            all_labels.append(label)

    p = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    r = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return p, r, f


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    val_samples = build_val_samples()
    noise_files = load_noise_files()
    mel_transform = build_mel_transform()

    print(f"Val samples (Fold-5 originals): {len(val_samples)}")
    print(f"Noise files: {len(noise_files)}")

    results = {}
    for model_name, model_path in MODELS.items():
        model = CRNN(num_classes=3, n_mels=N_MELS).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        results[model_name] = {}
        print(f"\n=== {model_name} ===")
        print(f"{'Condition':<12} {'Precision':>10} {'Recall':>8} {'F1':>8}")
        print("-" * 42)

        for snr in SNR_CONDITIONS:
            cond_label = "Clean" if snr is None else f"SNR {int(snr)}dB"
            p, r, f = evaluate_condition(
                model, val_samples, noise_files, snr, mel_transform, device
            )
            results[model_name][cond_label] = {
                "precision": round(p, 4),
                "recall":    round(r, 4),
                "f1":        round(f, 4),
            }
            print(f"  {cond_label:<10} {p:>10.4f} {r:>8.4f} {f:>8.4f}")

    with open("benchmark_results.json", "w") as fh:
        json.dump(results, fh, indent=2)
    print("\nResults saved to benchmark_results.json")

    print("\n--- Robustness Comparison (F1-Score) ---")
    conditions = ["Clean", "SNR 20dB", "SNR 10dB", "SNR 0dB"]
    print(f"{'Condition':<12} {'Baseline':>10} {'Proposed':>10} {'Delta':>8}")
    print("-" * 44)
    for cond in conditions:
        b = results["Baseline"][cond]["f1"]
        p = results["Proposed"][cond]["f1"]
        print(f"  {cond:<10} {b:>10.4f} {p:>10.4f} {p-b:>+8.4f}")


if __name__ == "__main__":
    main()
