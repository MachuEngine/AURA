"""Train the CRNN SED model on ESC-50 data with SNR-based noise augmentation."""
import glob
import logging
import os
import random
import time

import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset, random_split

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
EPOCHS = 10
LR = 1e-3


class AudioDataset(Dataset):
    def __init__(self, data_root: str = "data", augment: bool = False):
        self.augment = augment
        self.mel_transform = build_mel_transform()
        self.samples: list[tuple[str, int]] = []

        for class_name, label in CLASS_DIRS.items():
            class_dir = os.path.join(data_root, class_name)
            for wav in glob.glob(os.path.join(class_dir, "*.wav")):
                self.samples.append((wav, label))

        # Pre-collect noise file paths for augmentation
        self.noise_files: list[str] = []
        for d in NOISE_DIRS:
            self.noise_files.extend(glob.glob(os.path.join(d, "*.wav")))

        if augment and not self.noise_files:
            log.warning("Augmentation requested but no noise files found in data/noise/")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        waveform = load_audio(path)

        if self.augment and self.noise_files:
            noise_path = random.choice(self.noise_files)
            snr_db = random.choice(SNR_OPTIONS)
            noise = load_audio(noise_path)
            waveform = mix_noise_with_snr(waveform, noise, snr_db)

        logmel = waveform_to_logmel(waveform, self.mel_transform)
        logmel = self._pad_or_truncate(logmel)
        return logmel, label

    def _pad_or_truncate(self, logmel: torch.Tensor) -> torch.Tensor:
        T = logmel.shape[-1]
        if T >= FIXED_FRAMES:
            return logmel[:, :, :FIXED_FRAMES]
        return torch.nn.functional.pad(logmel, (0, FIXED_FRAMES - T))


def measure_latency(model: nn.Module, device: torch.device, n_runs: int = 50) -> float:
    model.eval()
    dummy = torch.randn(1, 1, N_MELS, FIXED_FRAMES).to(device)
    with torch.no_grad():
        for _ in range(10):  # warmup
            model(dummy)
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_runs):
            model(dummy)
    return (time.perf_counter() - start) / n_runs * 1000


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    full_dataset = AudioDataset("data", augment=False)
    log.info(f"Total samples: {len(full_dataset)}")

    n_val = max(1, int(0.2 * len(full_dataset)))
    n_train = len(full_dataset) - n_val
    train_indices, val_indices = random_split(
        range(len(full_dataset)),
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    # Train set uses augmentation; val set does not
    train_set = AudioDataset("data", augment=True)
    train_set.samples = [full_dataset.samples[i] for i in train_indices]
    val_set = AudioDataset("data", augment=False)
    val_set.samples = [full_dataset.samples[i] for i in val_indices]

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

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
        avg_loss = total_loss / len(train_loader)

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
            f"Epoch {epoch:2d}/{EPOCHS} | Loss: {avg_loss:.4f} | "
            f"Val F1: {f1:.4f} | Latency: {latency_ms:.2f}ms"
        )

        if f1 >= best_f1:
            best_f1 = f1
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), "models/sed_model_best.pth")

    torch.save(model.state_dict(), "models/sed_model_final.pth")
    log.info(f"Training complete. Best Val F1: {best_f1:.4f}")
    log.info("Weights saved → models/sed_model_best.pth  models/sed_model_final.pth")


if __name__ == "__main__":
    train()
