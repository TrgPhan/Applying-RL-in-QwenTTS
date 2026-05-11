"""
gwen_rl/utils/model.py
Load and wrap gwen-tts-0.6B talker for RL training.

Architecture (verified from weight shapes):
  talker.model.text_embedding:   [151936, 2048]  — Qwen3 BPE text tokens
  talker.text_projection.linear_fc1: [1024, 2048] — project text emb → hidden
  talker.text_projection.linear_fc2: [1024, 1024]
  talker.model.codec_embedding:  [3072, 1024]    — codec tokens (codebook 0+special)
  talker.model.layers.*          28×Qwen3 layers  — TRAINABLE via LoRA
  talker.model.norm              RMSNorm          — frozen
  talker.codec_head.weight       [3072, 1024]    — output head for codec tokens

Input sequence (for SFT/RL):
  [text_tokens | codec_tokens]
  Text tokens go through text_embedding + projection.
  Codec tokens go through codec_embedding.
  Combined → Qwen3 layers → codec_head → logits over codec vocab (3072)

For SFT loss: CE on codec_tokens (teacher forcing on codebook 0).
For GRPO: generate codec tokens, compute log_probs.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, TaskType
from transformers import Qwen3Config, Qwen3Model, AutoTokenizer

MODEL_ID = "g-group-ai-lab/gwen-tts-0.6B"
MODEL_CACHE = "./model_cache/gwen-tts-0.6B"

# Special token IDs in codec vocab (from config.json)
CODEC_BOS_ID = 2149
CODEC_EOS_ID = 2150
CODEC_PAD_ID = 2148
CODEC_LANG_VI = 2068      # Vietnamese language token


def _download_if_needed(model_id: str, cache_dir: str) -> str:
    if not os.path.exists(os.path.join(cache_dir, "model.safetensors")):
        from huggingface_hub import snapshot_download
        print(f"[model] Downloading {model_id} → {cache_dir}")
        snapshot_download(model_id, ignore_patterns=["data/*"], local_dir=cache_dir)
    return cache_dir


class GwenTalker(nn.Module):
    """
    Minimal wrapper around gwen-tts-0.6B talker.
    Handles the heterogeneous text+codec embedding correctly.
    LoRA is applied to the inner Qwen3Model's attention layers.
    """

    def __init__(self, weights: dict, lora_r: int = 16, lora_alpha: int = 32,
                 lora_dropout: float = 0.05, dtype=torch.float16):
        super().__init__()

        # --- Embeddings (frozen) ---
        self.text_embedding = nn.Embedding(151936, 2048)
        self.codec_embedding = nn.Embedding(3072, 1024)

        # Text projection: 2048 → 1024 (2-layer MLP from config)
        self.text_proj_fc1 = nn.Linear(2048, 1024, bias=True)
        self.text_proj_fc2 = nn.Linear(1024, 1024, bias=True)

        # --- Qwen3 transformer (trainable via LoRA) ---
        qwen_cfg = Qwen3Config(
            hidden_size=1024,
            num_hidden_layers=28,
            num_attention_heads=16,
            num_key_value_heads=8,
            head_dim=128,
            intermediate_size=3072,
            vocab_size=3072,
            max_position_embeddings=32768,
            rms_norm_eps=1e-6,
            rope_theta=1_000_000,
            attention_bias=False,
            attention_dropout=0.0,
            use_sliding_window=False,
            use_cache=False,
        )
        self.transformer = Qwen3Model(qwen_cfg)

        # --- Output head (frozen) ---
        self.codec_head = nn.Linear(1024, 3072, bias=False)

        # --- Load weights ---
        self._load_weights(weights, dtype)

        # --- Apply LoRA to transformer attention ---
        lora_cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
        )
        # Freeze everything first
        for p in self.parameters():
            p.requires_grad = False
        # Apply LoRA (only transformer params get LoRA adapters)
        self.transformer = get_peft_model(self.transformer, lora_cfg)
        # Re-enable grads for LoRA params
        for n, p in self.transformer.named_parameters():
            if "lora_" in n:
                p.requires_grad = True

        # Gradient checkpointing on transformer
        self.transformer.enable_input_require_grads()
        self.transformer.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    def _load_weights(self, weights: dict, dtype):
        """Load talker weights from state dict. Handles prefix remapping."""
        def _get(key, default=None):
            return weights.get(key, default)

        # Text embedding
        w = _get("talker.model.text_embedding.weight")
        if w is not None:
            self.text_embedding.weight = nn.Parameter(w.to(dtype), requires_grad=False)

        # Codec embedding
        w = _get("talker.model.codec_embedding.weight")
        if w is not None:
            self.codec_embedding.weight = nn.Parameter(w.to(dtype), requires_grad=False)

        # Text projection
        w = _get("talker.text_projection.linear_fc1.weight")
        if w is not None:
            self.text_proj_fc1.weight = nn.Parameter(w.to(dtype), requires_grad=False)
        b = _get("talker.text_projection.linear_fc1.bias")
        if b is not None:
            self.text_proj_fc1.bias = nn.Parameter(b.to(dtype), requires_grad=False)
            
        w = _get("talker.text_projection.linear_fc2.weight")
        if w is not None:
            self.text_proj_fc2.weight = nn.Parameter(w.to(dtype), requires_grad=False)
        b = _get("talker.text_projection.linear_fc2.bias")
        if b is not None:
            self.text_proj_fc2.bias = nn.Parameter(b.to(dtype), requires_grad=False)

        # Codec head
        w = _get("talker.codec_head.weight")
        if w is not None:
            self.codec_head.weight = nn.Parameter(w.to(dtype), requires_grad=False)

        # Transformer layers: remap talker.model.* → *
        transformer_sd = {}
        for k, v in weights.items():
            if k.startswith("talker.model.layers.") or k.startswith("talker.model.norm"):
                new_k = k[len("talker.model."):]    # → layers.*/norm
                transformer_sd[new_k] = v.to(dtype)
        missing, unexpected = self.transformer.load_state_dict(transformer_sd, strict=False)
        n = len(transformer_sd)
        print(f"[model] Loaded {n} transformer weights | "
              f"missing: {len(missing)} | unexpected: {len(unexpected)}")

    def forward(self, text_ids, codec_ids, attention_mask=None):
        """
        Forward pass for SFT (teacher forcing).
        Args:
            text_ids:  [B, T_text] — text token IDs (Qwen3 BPE, vocab=151936)
            codec_ids: [B, T_codec] — codec token IDs (vocab=3072)
            attention_mask: optional [B, T_text + T_codec]
        Returns:
            logits: [B, T_codec, 3072] — logits over codec vocab for each position
        """
        B = text_ids.size(0)
        device = text_ids.device

        # Text embeddings: [B, T_text, 2048] → project → [B, T_text, 1024]
        t_emb = self.text_embedding(text_ids).to(self.text_proj_fc1.weight.dtype)
        t_emb = F.silu(self.text_proj_fc1(t_emb))
        t_emb = self.text_proj_fc2(t_emb)           # [B, T_text, 1024]

        # Codec embeddings: [B, T_codec, 1024]
        c_emb = self.codec_embedding(codec_ids)      # [B, T_codec, 1024]

        # Concatenate: [B, T_text + T_codec, 1024]
        inputs_embeds = torch.cat([t_emb, c_emb], dim=1)

        # Build attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones(B, inputs_embeds.size(1),
                                        device=device, dtype=torch.long)

        # Transformer forward
        out = self.transformer(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )
        hidden = out.last_hidden_state        # [B, T_text+T_codec, 1024]

        # Only take codec positions for logits
        T_text = text_ids.size(1)
        codec_hidden = hidden[:, T_text:, :]  # [B, T_codec, 1024]

        logits = self.codec_head(codec_hidden.to(self.codec_head.weight.dtype)).float()  # [B, T_codec, 3072]
        return logits

    def generate(self, text_ids, max_new_tokens=256, temperature=0.3,
                 top_k=20, top_p=0.9, repetition_penalty=2.0):
        """
        Autoregressive generation of codec tokens from text input.
        Returns: (codec_ids [T], log_probs [T])
        """
        device = text_ids.device
        B = text_ids.size(0)
        assert B == 1, "generate() supports batch_size=1 only"

        # Text embedding
        with torch.no_grad() if not self.training else torch.enable_grad():
            t_emb = self.text_embedding(text_ids).to(self.text_proj_fc1.weight.dtype)
            t_emb = F.silu(self.text_proj_fc1(t_emb))
            t_emb = self.text_proj_fc2(t_emb)

        # Start with BOS codec token
        generated = [CODEC_BOS_ID]
        log_probs = []
        past_key_values = None

        for step in range(max_new_tokens):
            codec_so_far = torch.tensor([generated], dtype=torch.long, device=device)
            c_emb = self.codec_embedding(codec_so_far)
            inputs_embeds = torch.cat([t_emb, c_emb], dim=1) if step == 0 else c_emb[:, -1:]

            with torch.no_grad():
                out = self.transformer(
                    inputs_embeds=inputs_embeds,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
            past_key_values = out.past_key_values
            hidden = out.last_hidden_state[:, -1:, :]       # [1, 1, 1024]
            logits = self.codec_head(hidden.to(self.codec_head.weight.dtype)).float()[0, 0]  # [3072]

            # Repetition penalty
            if repetition_penalty != 1.0:
                for tid in set(generated[-50:]):
                    logits[tid] = logits[tid] / repetition_penalty

            # Temperature + top-k/p sampling
            logits = logits / max(temperature, 1e-8)
            if top_k > 0:
                topk_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < topk_vals[-1]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            token = torch.multinomial(probs, 1).item()
            lp = float(torch.log(probs[token] + 1e-9))

            if token == CODEC_EOS_ID:
                break
            generated.append(token)
            log_probs.append(lp)

        codec_ids = torch.tensor(generated[1:], dtype=torch.long)   # exclude BOS
        lp_tensor = torch.tensor(log_probs, dtype=torch.float32)
        return codec_ids, lp_tensor

    def get_ref_logits(self, text_ids, codec_ids, attention_mask=None):
        """Forward pass without LoRA = reference model."""
        with self.transformer.disable_adapter():
            with torch.no_grad():
                return self.forward(text_ids, codec_ids, attention_mask)


def build_model(
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    load_in_4bit: bool = True,   # reserved for future; currently uses fp16
    model_cache: str = MODEL_CACHE,
):
    """
    Load gwen-tts-0.6B talker → GwenTalker with LoRA.
    Returns (model, tokenizer).
    """
    model_dir = _download_if_needed(MODEL_ID, model_cache)
    safetensors_path = os.path.join(model_dir, "model.safetensors")

    import safetensors.torch as st
    print("[model] Loading safetensors weights...")
    weights = st.load_file(safetensors_path, device="cpu")
    print(f"[model] Loaded {len(weights)} weight tensors")

    model = GwenTalker(weights, lora_r=lora_r, lora_alpha=lora_alpha,
                       lora_dropout=lora_dropout, dtype=torch.float16)
    model = model.cuda()

    vram = torch.cuda.memory_allocated() / 1024**3
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[model] VRAM: {vram:.2f}GB | "
          f"Trainable: {trainable/1e6:.2f}M / {total/1e6:.0f}M ({100*trainable/total:.2f}%)")

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, fix_mistral_regex=True)
    return model, tokenizer


# Backward-compatible aliases for training scripts
def build_model_4bit(lora_r=16, lora_alpha=32, lora_dropout=0.05,
                     load_in_4bit=True, model_cache=MODEL_CACHE):
    return build_model(lora_r, lora_alpha, lora_dropout, load_in_4bit, model_cache)


def get_ref_logits(model, input_ids, attention_mask=None):
    """Compatibility shim; actual logic is in GwenTalker.get_ref_logits."""
    raise NotImplementedError(
        "Use model.get_ref_logits(text_ids, codec_ids) instead."
    )
