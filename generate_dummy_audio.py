"""Generate dummy audio dataset for AURA pipeline testing."""
import os
import numpy as np
import torchaudio
import torch

SAMPLE_RATE = 16000
DURATION = 2  # seconds
NUM_SAMPLES_PER_CLASS = 20

CLASSES = {
    0: "background_noise",
    1: "coughing",
    2: "yawning",
    3: "infant_crying",
}


def generate_background_noise(duration, sr):
    samples = int(duration * sr)
    return torch.FloatTensor(np.random.normal(0, 0.02, samples))


def generate_cough(duration, sr):
    samples = int(duration * sr)
    t = np.linspace(0, duration, samples)
    # Short burst of noise with amplitude envelope
    envelope = np.zeros(samples)
    burst_start = int(0.2 * sr)
    burst_len = int(0.3 * sr)
    envelope[burst_start:burst_start + burst_len] = np.hanning(burst_len)
    noise = np.random.normal(0, 0.3, samples)
    tone = 0.2 * np.sin(2 * np.pi * 400 * t)
    signal = (noise + tone) * envelope
    return torch.FloatTensor(signal.astype(np.float32))


def generate_yawn(duration, sr):
    samples = int(duration * sr)
    t = np.linspace(0, duration, samples)
    # Slow frequency sweep simulating yawn
    freq = np.linspace(200, 500, samples)
    envelope = np.sin(np.pi * t / duration)
    signal = 0.3 * np.sin(2 * np.pi * np.cumsum(freq) / sr) * envelope
    noise = np.random.normal(0, 0.02, samples)
    return torch.FloatTensor((signal + noise).astype(np.float32))


def generate_infant_cry(duration, sr):
    samples = int(duration * sr)
    t = np.linspace(0, duration, samples)
    # High frequency oscillating signal
    base_freq = 440
    vibrato = 20 * np.sin(2 * np.pi * 5 * t)
    signal = 0.4 * np.sin(2 * np.pi * (base_freq + vibrato) * t)
    harmonics = 0.2 * np.sin(2 * np.pi * 2 * (base_freq + vibrato) * t)
    envelope = 0.5 * (1 + np.sin(2 * np.pi * 2 * t))
    noise = np.random.normal(0, 0.01, samples)
    return torch.FloatTensor(((signal + harmonics) * envelope + noise).astype(np.float32))


GENERATORS = {
    0: generate_background_noise,
    1: generate_cough,
    2: generate_yawn,
    3: generate_infant_cry,
}


def main():
    for class_id, class_name in CLASSES.items():
        class_dir = os.path.join("data", class_name)
        os.makedirs(class_dir, exist_ok=True)

        for i in range(NUM_SAMPLES_PER_CLASS):
            audio = GENERATORS[class_id](DURATION, SAMPLE_RATE)
            audio = audio.unsqueeze(0)  # (1, samples)
            filename = os.path.join(class_dir, f"{class_name}_{i:03d}.wav")
            torchaudio.save(filename, audio, SAMPLE_RATE)

        print(f"Generated {NUM_SAMPLES_PER_CLASS} samples for class '{class_name}'")

    print(f"\nDataset ready in data/ ({len(CLASSES)} classes x {NUM_SAMPLES_PER_CLASS} samples)")


if __name__ == "__main__":
    main()
