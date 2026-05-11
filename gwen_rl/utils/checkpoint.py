"""
gwen_rl/utils/checkpoint.py
Save and load QLoRA adapters + training state.

gwen-tts-0.6B specific:
  - Only LoRA adapter weights are saved (not full model).
  - Full model is always reloaded from HuggingFace and adapters are merged on top.
"""

import os
import json
import torch
from peft import PeftModel


def save_checkpoint(model, optimizer, scheduler, step: int, metrics: dict, save_dir: str):
    """Save LoRA adapter + optimizer state + metadata."""
    os.makedirs(save_dir, exist_ok=True)
    # LoRA adapter weights
    model.save_pretrained(save_dir)
    # Optimizer + scheduler state
    torch.save({
        "step": step,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "metrics": metrics,
    }, os.path.join(save_dir, "trainer_state.pt"))
    print("[ckpt] Saved step=" + str(step) + " -> " + save_dir)


def load_checkpoint(model, optimizer, scheduler, save_dir: str):
    """Load LoRA adapter into model + restore optimizer/scheduler."""
    state_path = os.path.join(save_dir, "trainer_state.pt")
    if not os.path.exists(state_path):
        print("[ckpt] No trainer_state.pt found, starting from scratch.")
        return 0
    state = torch.load(state_path, map_location="cpu")
    optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and state["scheduler"] is not None:
        scheduler.load_state_dict(state["scheduler"])
    step = state["step"]
    print("[ckpt] Resumed from step=" + str(step) + " dir=" + save_dir)
    return step
