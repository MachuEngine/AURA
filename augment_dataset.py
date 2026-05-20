"""Offline augmentation to expand ESC-50 target classes 7× (40 → 280 per class).

Augmentations per original file:
  speed_085, speed_093, speed_107, speed_115  — speed perturbation (resampling trick)
  gain_low  (×0.35)                           — quieter / distant mic
  gain_high (×2.00, clamped)                  — louder / close mic

Result: 40 originals × 7 = 280 files per class → 840 total training samples.

Note: augmented files are saved alongside originals.
Re-running is idempotent (existing aug files are skipped).
"""
import glob
import os

import torch
import torchaudio

from utils.audio_utils import SAMPLE_RATE, load_audio

CLASS_DIRS = ["data/coughing", "data/sneezing", "data/infant_crying"]

SPEED_FACTORS = [0.85, 0.93, 1.07, 1.15]   # <1 = slower+lower, >1 = faster+higher
GAIN_FACTORS  = [("low", 0.35), ("high", 2.0)]


def speed_perturb(waveform: torch.Tensor, sr: int, factor: float) -> torch.Tensor:
    """Resample trick: treat waveform as recorded at sr*factor, then resample to sr.

    factor < 1 → stretched (slower, lower pitch)
    factor > 1 → compressed (faster, higher pitch)
    """
    fake_sr = int(sr * factor)
    return torchaudio.functional.resample(waveform, fake_sr, sr)


def main():
    total_added = 0

    for class_dir in CLASS_DIRS:
        # Only operate on originals (files without _aug_ suffix)
        originals = sorted(
            f for f in glob.glob(os.path.join(class_dir, "*.wav"))
            if "_aug_" not in os.path.basename(f)
        )

        added = 0
        for wav_path in originals:
            waveform = load_audio(wav_path)
            stem = os.path.splitext(wav_path)[0]

            for factor in SPEED_FACTORS:
                tag = f"speed_{int(factor * 100):03d}"
                out = f"{stem}_aug_{tag}.wav"
                if not os.path.exists(out):
                    aug = speed_perturb(waveform, SAMPLE_RATE, factor)
                    torchaudio.save(out, aug, SAMPLE_RATE)
                    added += 1

            for name, gain in GAIN_FACTORS:
                out = f"{stem}_aug_gain_{name}.wav"
                if not os.path.exists(out):
                    aug = (waveform * gain).clamp(-1.0, 1.0)
                    torchaudio.save(out, aug, SAMPLE_RATE)
                    added += 1

        total = len(glob.glob(os.path.join(class_dir, "*.wav")))
        print(f"{class_dir}: +{added} new files  →  {total} total")
        total_added += added

    grand_total = sum(
        len(glob.glob(os.path.join(d, "*.wav"))) for d in CLASS_DIRS
    )
    print(f"\nAugmented files added : {total_added}")
    print(f"Total training samples: {grand_total}")


if __name__ == "__main__":
    main()
