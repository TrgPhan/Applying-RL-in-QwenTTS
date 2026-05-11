"""
gwen_rl/utils/audio.py
Audio loading, resampling, SNR filtering utilities.
SR_TARGET = 24000 (gwen-tts-0.6B native output rate).
"""

import torch
import torchaudio
import numpy as np

SR_TARGET = 24000   # gwen-tts-0.6B output sample rate
SR_WHISPER = 16000  # Whisper input sample rate


def load_audio(path: str, target_sr: int = SR_TARGET) -> torch.Tensor:
    """Load audio file → mono float32 tensor [T] at target_sr."""
    wav, sr = torchaudio.load(path)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)  # stereo → mono
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav.squeeze(0)   # [T]


def resample(wav: torch.Tensor, from_sr: int, to_sr: int) -> torch.Tensor:
    """Resample [T] tensor from from_sr to to_sr."""
    if from_sr == to_sr:
        return wav
    return torchaudio.functional.resample(wav.unsqueeze(0), from_sr, to_sr).squeeze(0)


def compute_snr_db(wav: torch.Tensor, frame_len: int = 1600) -> float:
    """
    Rough SNR estimate: ratio of RMS signal to RMS of silence frames.
    Silence = frames with RMS below 10th percentile.
    Returns SNR in dB. Returns 99.0 if wav is too short.
    """
    if wav.numel() < frame_len * 2:
        return 99.0
    frames = wav.unfold(0, frame_len, frame_len // 2)   # [N, frame_len]
    rms = frames.pow(2).mean(dim=-1).sqrt()              # [N]
    threshold = float(rms.quantile(0.10))
    silence = rms[rms <= threshold]
    signal = rms[rms > threshold]
    if silence.numel() == 0 or signal.numel() == 0:
        return 99.0
    noise_rms = float(silence.mean()) + 1e-9
    signal_rms = float(signal.mean())
    return 20.0 * np.log10(signal_rms / noise_rms)


def is_valid_audio(
    wav: torch.Tensor,
    sr: int,
    min_dur: float = 2.0,
    max_dur: float = 15.0,
    min_snr_db: float = 15.0,
) -> bool:
    """Return True if wav passes duration and SNR filters."""
    dur = wav.size(0) / sr
    if dur < min_dur or dur > max_dur:
        return False
    if compute_snr_db(wav) < min_snr_db:
        return False
    return True
