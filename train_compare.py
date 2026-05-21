"""Multi-model comparison training for the AURA ablation study.

Trains all five architectures under identical conditions (Proposed regime:
SNR noise mixing + SpecAugment, Fold-5 validation) and saves a comparison
summary to comparison_results.json.

Models
------
VGG          — pure CNN, no temporal modelling
CRNN         — CNN + Bi-GRU (original AURA)
MobileNet    — lightweight CNN (inverted residuals)
CNN-Mamba    — CNN + Mamba SSM  [proposed]
CNN-Transformer — CNN + Self-Attention
"""
import glob
import json
import logging
import os
import random
import time
from collections import Counter

import torch
import torch.nn as nn
import torchaudio.transforms as T
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset

from models.ast_model import CRNNTransformer
from models.mamba_model import CRNNMamba
from models.mobilenet_model import MobileNetSED
from models.sed_model import CRNN
from models.vgg_model import VGG
from utils.audio_utils import (
    FIXED_FRAMES,
    N_MELS,
    build_mel_transform,
    load_audio,
    mix_noise_with_snr,
    waveform_to_logmel,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    handlers=[
        logging.FileHandler("compare_training.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CLASS_DIRS = {"coughing": 0, "sneezing": 1, "infant_crying": 2}
NUM_CLASSES = 3

NOISE_DIRS = [
    "data/noise/vehicle_engine",
    "data/noise/environmental",
    "data/noise/hvac",
]
SNR_OPTIONS = [0.0, 10.0, 20.0]

BATCH_SIZE = 16
EPOCHS     = 50
LR         = 1e-3

SNR_EVAL_CONDITIONS = [None, 20.0, 10.0, 0.0]   # None = clean

# Model registry: name → (constructor_fn, save_path)
MODEL_REGISTRY = {
    "VGG":             (lambda: VGG(NUM_CLASSES, N_MELS),              "models/vgg_sed.pth"),
    "CRNN":            (lambda: CRNN(NUM_CLASSES, N_MELS),             "models/crnn_compare.pth"),
    "MobileNet":       (lambda: MobileNetSED(NUM_CLASSES),             "models/mobilenet_sed.pth"),
    "CNN-Mamba":       (lambda: CRNNMamba(NUM_CLASSES, N_MELS),        "models/mamba_sed.pth"),
    "CNN-Transformer": (lambda: CRNNTransformer(NUM_CLASSES, N_MELS),  "models/transformer_sed.pth"),
}

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Fold-based data split (same as train.py)
# ---------------------------------------------------------------------------
def _is_train_file(path: str) -> bool:
    return os.path.basename(path)[0] in "1234"

def _is_val_file(path: str) -> bool:
    name = os.path.basename(path)
    return name.startswith("5-") and "_aug_" not in name

def build_split(file_filter):
    samples = []
    for cls, lbl in CLASS_DIRS.items():
        for wav in sorted(glob.glob(os.path.join("data", cls, "*.wav"))):
            if file_filter(wav):
                samples.append((wav, lbl))
    return samples


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class AudioDataset(Dataset):
    def __init__(self, samples, noise_aug=True, spec_aug=True, online_aug=True):
        self.samples    = samples
        self.noise_aug  = noise_aug
        self.online_aug = online_aug
        self.mel_tf     = build_mel_transform()
        self.spec_aug   = nn.Sequential(
            T.FrequencyMasking(16),
            T.TimeMasking(50),
        ) if spec_aug else nn.Identity()
        self.noise_files = []
        for d in NOISE_DIRS:
            self.noise_files.extend(glob.glob(os.path.join(d, "*.wav")))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        wf = load_audio(path)
        if self.online_aug:
            wf = (wf * random.uniform(0.71, 1.41)).clamp(-1.0, 1.0)
            if random.random() < 0.5:
                wf = -wf
        if self.noise_aug and self.noise_files:
            noise = load_audio(random.choice(self.noise_files))
            wf = mix_noise_with_snr(wf, noise, random.choice(SNR_OPTIONS))
        lm = waveform_to_logmel(wf, self.mel_tf)
        lm = self.spec_aug(lm)
        return _pad_or_trunc(lm), label


def _pad_or_trunc(lm):
    t = lm.shape[-1]
    if t >= FIXED_FRAMES:
        return lm[:, :, :FIXED_FRAMES]
    return torch.nn.functional.pad(lm, (0, FIXED_FRAMES - t))


# ---------------------------------------------------------------------------
# Latency helper
# ---------------------------------------------------------------------------
def measure_latency(model, device, n_runs=50):
    model.eval()
    dummy = torch.randn(1, 1, N_MELS, FIXED_FRAMES).to(device)
    with torch.no_grad():
        for _ in range(10):
            model(dummy)
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_runs):
            model(dummy)
    return (time.perf_counter() - t0) / n_runs * 1000


# ---------------------------------------------------------------------------
# Robustness evaluation
# ---------------------------------------------------------------------------
def evaluate_snr(model, val_samples, snr_db, device):
    """Evaluate model on val_samples at a given SNR (None = clean)."""
    noise_files = []
    for d in NOISE_DIRS:
        noise_files.extend(glob.glob(os.path.join(d, "*.wav")))
    mel_tf = build_mel_transform()
    rng = random.Random(0)

    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for path, label in val_samples:
            wf = load_audio(path)
            if snr_db is not None and noise_files:
                wf = mix_noise_with_snr(wf, load_audio(rng.choice(noise_files)), snr_db)
            lm = _pad_or_trunc(waveform_to_logmel(wf, mel_tf))
            pred = model(lm.unsqueeze(0).to(device)).argmax(dim=1).item()
            preds.append(pred)
            labels.append(label)

    return {
        "precision": round(precision_score(labels, preds, average="macro", zero_division=0), 4),
        "recall":    round(recall_score(labels,    preds, average="macro", zero_division=0), 4),
        "f1":        round(f1_score(labels,        preds, average="macro", zero_division=0), 4),
    }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train_model(name: str, model: nn.Module, save_path: str, device: torch.device) -> dict:
    log.info(f"\n{'='*60}")
    log.info(f"  Training: {name}  ({model.count_parameters():,} params)")
    log.info(f"{'='*60}")

    train_samples = build_split(_is_train_file)
    val_samples   = build_split(_is_val_file)

    log.info(f"Train: {len(train_samples)} | Val: {len(val_samples)} (Fold-5 orig only)")

    train_set = AudioDataset(train_samples, noise_aug=True,  spec_aug=True,  online_aug=True)
    val_set   = AudioDataset(val_samples,   noise_aug=False, spec_aug=False, online_aug=False)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()

    best_f1, best_loss = 0.0, float("inf")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for x, y in val_loader:
                all_preds.extend(model(x.to(device)).argmax(1).cpu().numpy())
                all_labels.extend(y.numpy())

        f1  = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        avg_loss = total_loss / len(train_loader)

        log.info(
            f"[{name}] Epoch {epoch:3d}/{EPOCHS} | "
            f"Loss: {avg_loss:.4f} | Val F1: {f1:.4f}"
        )

        if f1 >= best_f1:
            best_f1 = f1
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), save_path)

    # --- Robustness evaluation on saved best model ---
    model.load_state_dict(torch.load(save_path, map_location=device))
    latency_ms = measure_latency(model, device)

    robustness = {}
    for snr in SNR_EVAL_CONDITIONS:
        cond = "Clean" if snr is None else f"SNR {int(snr)}dB"
        robustness[cond] = evaluate_snr(model, val_samples, snr, device)

    result = {
        "params":     model.count_parameters(),
        "best_val_f1": round(best_f1, 4),
        "latency_ms":  round(latency_ms, 2),
        "robustness":  robustness,
    }
    log.info(f"[{name}] Done — best F1={best_f1:.4f}, latency={latency_ms:.2f}ms")
    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    device = get_device()
    log.info(f"Device: {device}")

    results = {}
    for name, (ctor, save_path) in MODEL_REGISTRY.items():
        model = ctor()
        results[name] = train_model(name, model, save_path, device)

    # Save results
    with open("comparison_results.json", "w") as f:
        json.dump(results, f, indent=2)
    log.info("\nResults saved to comparison_results.json")

    # Print comparison table
    print("\n" + "=" * 72)
    print("  Model Comparison — ESC-50 Fold-5, Proposed Training Regime")
    print("=" * 72)
    print(f"{'Model':<18} {'Params':>9} {'Clean F1':>9} {'20dB F1':>9} {'10dB F1':>9} {'0dB F1':>9} {'Lat(ms)':>8}")
    print("-" * 72)
    for name, r in results.items():
        rob = r["robustness"]
        marker = " ◄" if name == "CNN-Mamba" else ""
        print(
            f"  {name:<16} {r['params']:>9,} "
            f"{rob['Clean']['f1']:>9.4f} "
            f"{rob['SNR 20dB']['f1']:>9.4f} "
            f"{rob['SNR 10dB']['f1']:>9.4f} "
            f"{rob['SNR 0dB']['f1']:>9.4f} "
            f"{r['latency_ms']:>8.2f}"
            f"{marker}"
        )
    print("=" * 72)
    print("◄ Proposed model")


if __name__ == "__main__":
    main()
