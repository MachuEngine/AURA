"""Audio preprocessing utilities."""
import torch
import torchaudio
import torchaudio.transforms as T


N_MELS = 64
SAMPLE_RATE = 16000
N_FFT = 512
HOP_LENGTH = 256
F_MIN = 20
F_MAX = 8000


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
    mel = mel_transform(waveform)              # (1, n_mels, T)
    log_mel = torch.log(mel + 1e-9)
    # Normalize per sample
    log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-6)
    return log_mel


def load_audio(path: str, target_sr: int = SAMPLE_RATE) -> torch.Tensor:
    waveform, sr = torchaudio.load(path)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform
