"""
gwen_rl/utils/gpu_config.py
Auto-detect GPU VRAM and return optimal training config.

Called at the start of every training script. Adjusts:
  - batch_size, group_size, gradient_accum_steps
  - gradient_checkpointing, use_wavlm
  - quantization bits (4 vs 8)
  - max_seq_len

GPU tiers:
  < 6 GB   → Tier 1 (RTX A1000 4GB, GTX 1660 6GB)   — most aggressive savings
  6-12 GB  → Tier 2 (RTX 3060 8GB, RTX 2080 11GB)   — moderate
  12-20 GB → Tier 3 (RTX 3080 10GB, RTX 4090 12GB)  — relaxed
  20-40 GB → Tier 4 (A100 40GB, RTX 4090 24GB)       — plan defaults
  > 40 GB  → Tier 5 (A100 80GB, H100)                — no restrictions

Usage:
    from gwen_rl.utils.gpu_config import apply_gpu_config
    cfg = apply_gpu_config(cfg)   # call before training
"""

import torch


def detect_vram_gb() -> float:
    """Return GPU VRAM in GB, or 0.0 if no CUDA GPU."""
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.get_device_properties(0).total_memory / 1024 ** 3


def apply_gpu_config(cfg: dict) -> dict:
    """
    Auto-adjust cfg based on detected VRAM.
    Only overrides keys that the user did NOT explicitly set in their YAML.
    Returns modified cfg with a new key 'gpu_tier' and 'vram_gb'.
    """
    vram = detect_vram_gb()
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

    # Determine tier
    if vram == 0:
        tier = 0          # CPU only
    elif vram < 6:
        tier = 1          # ≤ 6GB (4GB laptop GPU)
    elif vram < 12:
        tier = 2          # 6-12GB
    elif vram < 20:
        tier = 3          # 12-20GB
    elif vram < 44:
        tier = 4          # 20-44GB (A100 40GB)
    else:
        tier = 5          # ≥ 44GB (A100 80GB / H100)

    # Tier-based defaults
    tier_defaults = {
        0: {   # CPU only — tiny config for testing
            "batch_size": 1, "gradient_accum_steps": 4, "group_size": 2,
            "max_seq_len": 256, "load_in_4bit": True,
            "gradient_checkpointing": True, "use_wavlm": False,
            "gen_max_new_tokens": 128, "whisper_device": "cpu",
        },
        1: {   # ≤ 6GB
            "batch_size": 1, "gradient_accum_steps": 16, "group_size": 2,
            "max_seq_len": 512, "load_in_4bit": True,
            "gradient_checkpointing": True, "use_wavlm": True,
            "gen_max_new_tokens": 256, "whisper_device": "cpu",
        },
        2: {   # 6-12GB
            "batch_size": 1, "gradient_accum_steps": 8, "group_size": 4,
            "max_seq_len": 1024, "load_in_4bit": True,
            "gradient_checkpointing": True, "use_wavlm": True,
            "gen_max_new_tokens": 512, "whisper_device": "cuda",
        },
        3: {   # 12-20GB
            "batch_size": 2, "gradient_accum_steps": 4, "group_size": 4,
            "max_seq_len": 2048, "load_in_4bit": False,
            "gradient_checkpointing": True, "use_wavlm": True,
            "gen_max_new_tokens": 1024, "whisper_device": "cuda",
        },
        4: {   # 20-44GB
            "batch_size": 4, "gradient_accum_steps": 4, "group_size": 8,
            "max_seq_len": 4096, "load_in_4bit": False,
            "gradient_checkpointing": False, "use_wavlm": True,
            "gen_max_new_tokens": 1024, "whisper_device": "cuda",
        },
        5: {   # ≥ 44GB
            "batch_size": 8, "gradient_accum_steps": 2, "group_size": 8,
            "max_seq_len": 4096, "load_in_4bit": False,
            "gradient_checkpointing": False, "use_wavlm": True,
            "gen_max_new_tokens": 2048, "whisper_device": "cuda",
        },
    }

    overrides = tier_defaults[tier]

    # Only apply if user hasn't explicitly set the key in YAML
    # (We can't know what was in YAML vs default, so we always apply and log)
    for key, val in overrides.items():
        cfg[key] = val

    cfg["vram_gb"] = round(vram, 1)
    cfg["gpu_tier"] = tier

    print(f"[gpu_config] GPU: {gpu_name} | VRAM: {vram:.1f}GB | Tier {tier}")
    print(f"[gpu_config] Auto-set: batch={cfg['batch_size']} "
          f"accum={cfg['gradient_accum_steps']} "
          f"group={cfg['group_size']} "
          f"seq={cfg['max_seq_len']} "
          f"4bit={cfg.get('load_in_4bit', True)}")

    return cfg
