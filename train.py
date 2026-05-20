"""Train the CRNN SED model on the dummy audio dataset."""
import os
import time
import logging
import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import f1_score
import numpy as np

from models.sed_model import CRNN
from utils.audio_utils import load_audio, waveform_to_logmel, build_mel_transform, N_MELS

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
    "background_noise": 0,
    "coughing": 1,
    "yawning": 2,
    "infant_crying": 3,
}
NUM_CLASSES = len(CLASS_DIRS)
BATCH_SIZE = 8
EPOCHS = 5
LR = 1e-3
FIXED_FRAMES = 128  # fixed time dimension for padding/truncation


class AudioDataset(Dataset):
    def __init__(self, data_root="data"):
        self.samples = []
        self.mel_transform = build_mel_transform()
        for class_name, label in CLASS_DIRS.items():
            class_dir = os.path.join(data_root, class_name)
            for wav_path in glob.glob(os.path.join(class_dir, "*.wav")):
                self.samples.append((wav_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        waveform = load_audio(path)
        logmel = waveform_to_logmel(waveform, self.mel_transform)  # (1, n_mels, T)
        logmel = self._pad_or_truncate(logmel)
        return logmel, label

    def _pad_or_truncate(self, logmel: torch.Tensor) -> torch.Tensor:
        T = logmel.shape[-1]
        if T >= FIXED_FRAMES:
            return logmel[:, :, :FIXED_FRAMES]
        pad = FIXED_FRAMES - T
        return torch.nn.functional.pad(logmel, (0, pad))


def measure_latency(model, device, n_runs=50):
    model.eval()
    dummy = torch.randn(1, 1, N_MELS, FIXED_FRAMES).to(device)
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            model(dummy)
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_runs):
            model(dummy)
    elapsed_ms = (time.perf_counter() - start) / n_runs * 1000
    return elapsed_ms


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    dataset = AudioDataset("data")
    log.info(f"Dataset size: {len(dataset)} samples")

    n_val = max(1, int(0.2 * len(dataset)))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )
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
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        avg_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                preds = model(x).argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(y.numpy())

        f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        latency_ms = measure_latency(model, device)

        log.info(
            f"Epoch {epoch}/{EPOCHS} | Loss: {avg_loss:.4f} | "
            f"Val F1: {f1:.4f} | Latency: {latency_ms:.2f}ms"
        )

        if f1 >= best_f1:
            best_f1 = f1
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), "models/sed_model_best.pth")

    # Save final weights
    torch.save(model.state_dict(), "models/sed_model_final.pth")
    log.info(f"Training complete. Best Val F1: {best_f1:.4f}")
    log.info("Weights saved to models/sed_model_best.pth and models/sed_model_final.pth")


if __name__ == "__main__":
    train()
