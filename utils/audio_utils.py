"""Audio preprocessing utilities."""
import torch
import torchaudio
import torchaudio.transforms as T

SAMPLE_RATE = 16000
DURATION = 5          # ESC-50: 5-second clips
N_MELS = 64
N_FFT = 512
HOP_LENGTH = 256
F_MIN = 20
F_MAX = 8000

# Frames produced by MelSpectrogram with center=True for a 5-second clip at 16 kHz:
#   padded_samples = 80000 + N_FFT = 80512
#   frames = 1 + (80512 - N_FFT) // HOP_LENGTH = 1 + 80000//256 = 313
FIXED_FRAMES = 313


def build_mel_transform():
    return T.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        f_min=F_MIN,
        f_max=F_MAX,
        power=2.0,
    )


def waveform_to_logmel(waveform: torch.Tensor, mel_transform=None) -> torch.Tensor:
    """Convert (1, T) waveform to (1, N_MELS, time_frames) log-mel spectrogram."""
    if mel_transform is None:
        mel_transform = build_mel_transform()
    mel = mel_transform(waveform)
    log_mel = torch.log(mel + 1e-9)
    log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-6)
    return log_mel


def load_audio(path: str, target_sr: int = SAMPLE_RATE) -> torch.Tensor:
    """Load an audio file and return a mono (1, T) tensor at target_sr."""
    waveform, sr = torchaudio.load(path)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform


def mix_noise_with_snr(
    clean: torch.Tensor,
    noise: torch.Tensor,
    snr_db: float,
) -> torch.Tensor:
    """Mix clean signal with noise at a specified SNR (dB).

    Both inputs are (1, T) mono tensors. The noise is cropped / tiled to match
    the length of the clean signal before mixing.

    SNR(dB) = 20 * log10(RMS_clean / RMS_noise_scaled)
    → scale = RMS_clean / (RMS_noise * 10^(snr_db/20))
    """
    target_len = clean.shape[-1]

    # Tile or crop noise to match clean length
    if noise.shape[-1] < target_len:
        repeats = (target_len // noise.shape[-1]) + 1
        noise = noise.repeat(1, repeats)
    start = torch.randint(0, noise.shape[-1] - target_len + 1, (1,)).item()
    noise = noise[:, start: start + target_len]

    rms_clean = clean.pow(2).mean().sqrt()
    rms_noise = noise.pow(2).mean().sqrt()

    if rms_noise < 1e-9:
        return clean  # silent noise — skip mixing

    scale = rms_clean / (rms_noise * (10 ** (snr_db / 20)))
    mixed = clean + scale * noise

    # Prevent clipping
    peak = mixed.abs().max()
    if peak > 1.0:
        mixed = mixed / peak

    return mixed
