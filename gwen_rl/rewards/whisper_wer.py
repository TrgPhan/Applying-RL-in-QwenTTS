"""
gwen_rl/rewards/whisper_wer.py
WER reward via faster-whisper (CTranslate2 backend, 4x faster than HF Whisper).

Non-differentiable black box reward. Returns [0,1] float per sample.
  reward = max(0, 1 - WER)
  WER = Levenshtein(hypothesis, reference) / len(reference_words)

Model: tiny for 4GB GPU (150MB), medium for better accuracy (600MB).
Force language="vi" for Vietnamese.
"""

import numpy as np
import torch
import jiwer
from faster_whisper import WhisperModel


class WhisperWERReward:
    def __init__(self, model_size: str = "tiny", device: str = "cuda"):
        """
        Args:
            model_size: "tiny"|"base"|"small"|"medium"|"large-v3-turbo"
            device: "cuda" or "cpu" (use "cpu" if VRAM is tight)
        """
        compute_type = "float16" if device == "cuda" else "int8"
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self.device = device
        print("[whisper] Loaded whisper-" + model_size + " on " + device)

    @torch.no_grad()
    def compute_rewards(
        self,
        wavs: list,          # list of np.ndarray [T] @ 16kHz or torch.Tensor
        ref_texts: list,     # list of str (ground truth)
        language: str = "vi",
    ) -> torch.Tensor:
        """
        Args:
            wavs: list of mono audio arrays/tensors @ 16kHz
            ref_texts: list of reference transcripts
        Returns:
            rewards: [N] tensor of float in [0, 1]
        """
        rewards = []
        for wav, ref in zip(wavs, ref_texts):
            if isinstance(wav, torch.Tensor):
                wav_np = wav.float().cpu().numpy()
            else:
                wav_np = np.asarray(wav, dtype=np.float32)

            # Transcribe
            try:
                segs, _ = self.model.transcribe(
                    wav_np, language=language, beam_size=1, best_of=1,
                    without_timestamps=True,
                )
                hypothesis = " ".join(s.text.strip() for s in segs)
            except Exception:
                hypothesis = ""

            # WER
            ref_clean = ref.strip().lower()
            hyp_clean = hypothesis.strip().lower()
            if len(ref_clean) == 0:
                rewards.append(0.0)
                continue
            try:
                wer = jiwer.wer(ref_clean, hyp_clean)
            except Exception:
                wer = 1.0
            rewards.append(max(0.0, 1.0 - wer))

        return torch.tensor(rewards, dtype=torch.float32)
