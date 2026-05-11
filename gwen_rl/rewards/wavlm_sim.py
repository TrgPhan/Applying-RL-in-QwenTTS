"""
gwen_rl/rewards/wavlm_sim.py
Speaker similarity reward via WavLM-base-plus speaker verification.

Non-differentiable. CPU offloaded between steps to save VRAM.
Returns cosine similarity in [0, 1] (clipped from [-1, 1]).

Call .to_cpu() after computing to free GPU memory.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoFeatureExtractor


class WavLMSimReward:
    def __init__(self, model_name: str = "microsoft/wavlm-base-plus-sv"):
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        # Load on CPU initially to save VRAM during training
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self._on_gpu = False
        print("[wavlm] Loaded " + model_name + " (CPU mode, offloaded)")

    def to_gpu(self):
        if not self._on_gpu:
            self.model = self.model.cuda()
            self._on_gpu = True

    def to_cpu(self):
        if self._on_gpu:
            self.model = self.model.cpu()
            torch.cuda.empty_cache()
            self._on_gpu = False

    @torch.no_grad()
    def _embed(self, wav_16k: torch.Tensor) -> torch.Tensor:
        """Extract speaker embedding [D] from mono 16kHz tensor."""
        inputs = self.feature_extractor(
            wav_16k.cpu().numpy(), sampling_rate=16000, return_tensors="pt", padding=True
        )
        device = next(self.model.parameters()).device
        hidden = self.model(inputs["input_values"].to(device)).last_hidden_state
        return hidden.mean(dim=1).squeeze(0)  # [D]

    @torch.no_grad()
    def compute_rewards(
        self,
        gen_wavs_16k: list,   # list of [T] tensors @ 16kHz
        ref_wavs_16k: list,   # list of [T] tensors @ 16kHz
    ) -> torch.Tensor:
        """
        Returns cosine similarity [N] in [0, 1].
        Moves model to GPU for computation, then back to CPU.
        """
        self.to_gpu()
        sims = []
        for gen, ref in zip(gen_wavs_16k, ref_wavs_16k):
            e_gen = self._embed(gen)
            e_ref = self._embed(ref)
            sim = F.cosine_similarity(e_gen.unsqueeze(0), e_ref.unsqueeze(0)).item()
            sims.append(max(0.0, sim))  # clip negative to 0
        self.to_cpu()
        return torch.tensor(sims, dtype=torch.float32)
