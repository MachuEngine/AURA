"""Train Baseline and Proposed CRNN SED models on ESC-50 data.

Data split (data leakage prevention — ESC-50 official Fold protocol):
  Train : Folds 1-4 originals + their offline augmentations (_aug_)
  Val   : Fold 5 originals ONLY — zero augmented files permitted

Baseline  — no noise mixing, no SpecAugment → models/sed_baseline.pth
Proposed  — on-the-fly SNR noise mixing + SpecAugment → models/sed_robust_best.pth
"""
import glob
import logging
import os
import random
import time
from collections import Counter

import torch
import torch.nn as nn
import torchaudio.transforms as T
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset

from models.sed_model import CRNN
from utils.audio_utils import (
    FIXED_FRAMES,
    N_MELS,
    build_mel_transform,
    load_audio,
    mix_noise_with_snr,
    waveform_to_logmel,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

CLASS_DIRS = {
    "coughing":      0,
    "sneezing":      1,
    "infant_crying": 2,
}
NUM_CLASSES = len(CLASS_DIRS)

NOISE_DIRS = [
    "data/noise/vehicle_engine",
    "data/noise/environmental",
    "data/noise/hvac",
]
SNR_OPTIONS = [0.0, 10.0, 20.0]  # dB

BATCH_SIZE = 16
EPOCHS = 50
LR = 1e-3


# ---------------------------------------------------------------------------
# Fold-based sample builders — the ONLY place train/val membership is decided
# ---------------------------------------------------------------------------

def _is_train_file(path: str) -> bool:
    """Folds 1–4: originals and their offline augmented variants."""
    name = os.path.basename(path)
    return name[0] in "1234"


def _is_val_file(path: str) -> bool:
    """Fold 5 originals strictly — no _aug_ files allowed."""
    name = os.path.basename(path)
    return name.startswith("5-") and "_aug_" not in name


def build_split(data_root: str, file_filter) -> list[tuple[str, int]]:
    samples = []
    for class_name, label in CLASS_DIRS.items():
        class_dir = os.path.join(data_root, class_name)
        for wav in sorted(glob.glob(os.path.join(class_dir, "*.wav"))):
            if file_filter(wav):
                samples.append((wav, label))
    return samples


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class AudioDataset(Dataset):
    def __init__(
        self,
        samples: list[tuple[str, int]],
        noise_aug: bool = False,
        spec_aug: bool = False,
        online_aug: bool = True,
    ):
        self.samples = samples
        self.noise_aug = noise_aug
        self.online_aug = online_aug
        self.mel_transform = build_mel_transform()
        # Slightly more aggressive SpecAugment for better regularisation
        self.spec_augment = nn.Sequential(
            T.FrequencyMasking(freq_mask_param=16),
            T.TimeMasking(time_mask_param=50),
        ) if spec_aug else nn.Identity()

        self.noise_files: list[str] = []
        for d in NOISE_DIRS:
            self.noise_files.extend(glob.glob(os.path.join(d, "*.wav")))

        if noise_aug and not self.noise_files:
            log.warning("noise_aug=True but no noise files found in data/noise/")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        waveform = load_audio(path)

        if self.online_aug:
            # Random gain jitter ±3 dB  (both models — general-purpose regulariser)
            gain = random.uniform(0.71, 1.41)
            waveform = (waveform * gain).clamp(-1.0, 1.0)
            # Random polarity inversion
            if random.random() < 0.5:
                waveform = -waveform

        if self.noise_aug and self.noise_files:
            noise = load_audio(random.choice(self.noise_files))
            waveform = mix_noise_with_snr(waveform, noise, random.choice(SNR_OPTIONS))

        logmel = waveform_to_logmel(waveform, self.mel_transform)
        logmel = self.spec_augment(logmel)
        return self._pad_or_truncate(logmel), label

    def _pad_or_truncate(self, logmel: torch.Tensor) -> torch.Tensor:
        t = logmel.shape[-1]
        if t >= FIXED_FRAMES:
            return logmel[:, :, :FIXED_FRAMES]
        return torch.nn.functional.pad(logmel, (0, FIXED_FRAMES - t))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def measure_latency(model: nn.Module, device: torch.device, n_runs: int = 50) -> float:
    model.eval()
    dummy = torch.randn(1, 1, N_MELS, FIXED_FRAMES).to(device)
    with torch.no_grad():
        for _ in range(10):
            model(dummy)
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_runs):
            model(dummy)
    return (time.perf_counter() - start) / n_runs * 1000


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def run_training(noise_aug: bool, save_path: str, label: str, device: torch.device) -> float:
    log.info(f"{'='*60}")
    log.info(f"Training: {label}")
    log.info(f"  noise_aug={noise_aug}  save → {save_path}")
    log.info(f"{'='*60}")

    train_samples = build_split("data", _is_train_file)
    val_samples   = build_split("data", _is_val_file)

    val_dist = Counter(lbl for _, lbl in val_samples)
    log.info(f"Train samples : {len(train_samples)} (Folds 1-4, orig+aug)")
    log.info(f"Val samples   : {len(val_samples)} (Fold 5, orig only) | dist={dict(val_dist)}")

    train_set = AudioDataset(train_samples, noise_aug=noise_aug, spec_aug=noise_aug, online_aug=True)
    val_set   = AudioDataset(val_samples,   noise_aug=False,     spec_aug=False,     online_aug=False)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = CRNN(num_classes=NUM_CLASSES, n_mels=N_MELS).to(device)
    log.info(f"Model parameters: {model.count_parameters():,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()

    best_f1 = 0.0

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
                preds = model(x.to(device)).argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(y.numpy())

        f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        latency_ms = measure_latency(model, device)

        log.info(
            f"[{label}] Epoch {epoch:3d}/{EPOCHS} | "
            f"Loss: {total_loss / len(train_loader):.4f} | "
            f"Val F1: {f1:.4f} | Latency: {latency_ms:.2f}ms"
        )

        if f1 >= best_f1:
            best_f1 = f1
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), save_path)

    log.info(f"[{label}] Best Val F1: {best_f1:.4f}  →  {save_path}")
    return best_f1


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    baseline_f1 = run_training(
        noise_aug=False,
        save_path="models/sed_baseline.pth",
        label="Baseline",
        device=device,
    )

    proposed_f1 = run_training(
        noise_aug=True,
        save_path="models/sed_robust_best.pth",
        label="Proposed",
        device=device,
    )

    log.info("")
    log.info("=" * 60)
    log.info("  Ablation Study Results  (Fold-5, leak-free validation)")
    log.info("=" * 60)
    log.info(f"  Baseline (no noise aug) : Val F1 = {baseline_f1:.4f}")
    log.info(f"  Proposed (SNR noise aug): Val F1 = {proposed_f1:.4f}")
    log.info(f"  Delta                   : {proposed_f1 - baseline_f1:+.4f}")
    log.info("=" * 60)
