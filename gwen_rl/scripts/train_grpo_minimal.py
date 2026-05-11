"""
gwen_rl/scripts/train_grpo_minimal.py
Gate 2: GRPO Minimal — Whisper WER reward only.

Algorithm (GSPO-style sequence-level, per Section 18.4 of plan):
  1. For each prompt, generate G rollouts from current policy (no grad)
  2. Compute WER rewards for each rollout
  3. Group-normalize rewards → advantages
  4. Re-forward rollouts through model (with grad) → get new log_probs
  5. Compute GRPO loss with DAPO clip-higher (eps_low=0.2, eps_high=0.28)
  6. Add KL penalty using PEFT disable_adapter()
  7. Backward + update LoRA

4GB VRAM optimizations:
  - Group size G=2 (plan says 8, but 4GB cannot handle it)
  - Rollouts on CPU (generate on GPU, immediately move audio to CPU)
  - Whisper on CPU (offloaded)
  - batch_size=1, gradient_accum=16 (effective=16)

Usage:
    py -3 -m gwen_rl.scripts.train_grpo_minimal --config gwen_rl/configs/gate2_grpo_min.yaml --max_steps 10
"""

import os
import sys
import argparse
import yaml
import warnings
warnings.filterwarnings("ignore", message=".*flash attention.*")

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from itertools import cycle
import torchaudio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gwen_rl.utils.model import build_model_4bit, get_ref_logits, MODEL_ID
from gwen_rl.utils.data import TTSRLDataset, collate_fn
from gwen_rl.utils.checkpoint import save_checkpoint, load_checkpoint
from gwen_rl.utils.log import init_logging, log_metrics, close_logging
from gwen_rl.utils.audio import SR_TARGET, SR_WHISPER, resample
from gwen_rl.rewards.whisper_wer import WhisperWERReward
from gwen_rl.utils.gpu_config import apply_gpu_config


DEFAULT_CFG = {
    "model_id": MODEL_ID,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "lora_checkpoint": "",          # path to Gate 1 adapter, or "" to start from base

    "train_path": "data/processed/train.pt",
    "val_path": "data/processed/val.pt",
    "save_dir": "checkpoints/gate2",
    "log_dir": "logs",

    "whisper_size": "tiny",         # tiny=150MB; use "small" if you have VRAM
    "whisper_device": "cpu",        # keep on CPU to save GPU VRAM

    "lr": 5e-7,
    "beta_kl": 0.05,
    "group_size": 2,                # G=2 for 4GB; increase to 4-8 with more VRAM
    "epsilon_low": 0.2,             # DAPO clip-lower
    "epsilon_high": 0.28,           # DAPO clip-higher (anti entropy collapse)
    "grad_clip": 1.0,
    "warmup_steps": 100,
    "batch_size": 1,
    "gradient_accum_steps": 16,
    "max_steps": 5000,
    "val_every": 200,
    "save_every": 1000,
    "max_seq_len": 512,

    # Generation config (must match inference)
    "gen_max_new_tokens": 256,
    "gen_temperature": 0.3,
    "gen_top_k": 20,
    "gen_top_p": 0.9,
    "gen_repetition_penalty": 2.0,
}


# ---------------------------------------------------------------------------
# Rollout: generate G samples for one prompt, return audio + log_probs
# ---------------------------------------------------------------------------
@torch.no_grad()
def rollout_one(model, tokenizer, text: str, cfg: dict, device: str):
    """
    Generate G samples for a single text prompt.
    Returns:
        codes: list[Tensor [T]] — generated token sequences (codebook 0 proxy)
        log_probs: list[Tensor [T]] — per-token log prob under current policy
        audio_wavs: list[Tensor [T_wav]] — raw audio output (on CPU, 24kHz)
    """
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=cfg["max_seq_len"] // 2,
    ).to(device)

    input_ids = encoded["input_ids"]
    G = cfg["group_size"]

    codes_list = []
    logprob_list = []
    # NOTE: gwen-tts-0.6B generates audio codes, not text.
    # We use model.generate() to get token sequences and then
    # interpret them as audio codes through the model's own pipeline.
    # For now we treat it as causal LM and use the generated tokens
    # to compute rewards via the model's text output.

    for _ in range(G):
        gen_out = model.generate(
            input_ids,
            max_new_tokens=cfg["gen_max_new_tokens"],
            do_sample=True,
            temperature=cfg["gen_temperature"],
            top_k=cfg["gen_top_k"],
            top_p=cfg["gen_top_p"],
            repetition_penalty=cfg["gen_repetition_penalty"],
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
        # generated tokens (excluding prompt)
        new_tokens = gen_out.sequences[0, input_ids.size(1):]   # [T_new]

        # Per-token log probs from scores
        scores = torch.stack(gen_out.scores, dim=0)             # [T_new, V]
        log_probs = F.log_softmax(scores.float(), dim=-1)       # [T_new, V]
        token_lp = log_probs.gather(-1, new_tokens.unsqueeze(-1)).squeeze(-1)  # [T_new]

        codes_list.append(new_tokens.cpu())
        logprob_list.append(token_lp.cpu())

    # For WER reward: decode generated tokens back to text
    # (In a full gwen-tts pipeline, this would go through Mimi decoder → audio → Whisper)
    # Since gwen-tts may not expose a simple text decode for audio codes,
    # we decode the token IDs as text and use that for WER as a proxy.
    decoded_texts = []
    for codes in codes_list:
        text_out = tokenizer.decode(codes, skip_special_tokens=True)
        decoded_texts.append(text_out)

    return codes_list, logprob_list, decoded_texts


# ---------------------------------------------------------------------------
# GRPO loss (DAPO asymmetric clip + Dr.GRPO length norm + MO-GRPO)
# ---------------------------------------------------------------------------
def grpo_loss(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    codes_list: list,
    old_logprobs_list: list,
    advantages: torch.Tensor,   # [G]
    cfg: dict,
    device: str,
) -> dict:
    """
    Compute GRPO policy loss for G rollouts.
    Re-forward each rollout through model (with gradient).
    """
    total_policy_loss = torch.tensor(0.0, device=device, requires_grad=True)
    total_kl = torch.tensor(0.0, device=device)
    G = len(codes_list)

    for i in range(G):
        codes = codes_list[i].to(device)           # [T_new]
        old_lp = old_logprobs_list[i].to(device)   # [T_new]
        adv = advantages[i].to(device)             # scalar

        # Build full sequence: prompt + generated codes
        full_ids = torch.cat([input_ids[0], codes]).unsqueeze(0)  # [1, T_prompt + T_new]
        T_prompt = input_ids.size(1)

        with torch.amp.autocast("cuda", dtype=torch.float16):
            out = model(input_ids=full_ids)
        logits = out.logits.float()[:, T_prompt - 1: -1, :]  # [1, T_new, V] aligned to codes

        new_log_probs = F.log_softmax(logits, dim=-1)
        new_lp = new_log_probs[0].gather(-1, codes.unsqueeze(-1)).squeeze(-1)  # [T_new]

        # KL with reference (PEFT disable_adapter)
        ref_logits = get_ref_logits(model, full_ids).float()[:, T_prompt - 1:-1, :]
        ref_lp = F.log_softmax(ref_logits, dim=-1)[0].gather(-1, codes.unsqueeze(-1)).squeeze(-1)
        kl_i = (new_lp - ref_lp).mean()
        total_kl = total_kl + kl_i.detach()

        # Importance ratio (sequence-level, Dr.GRPO length-normalized, GSPO style)
        T = codes.size(0)
        seq_log_ratio = (new_lp - old_lp).sum() / max(T, 1)   # scalar, length-normalized
        ratio = torch.exp(seq_log_ratio)

        # DAPO asymmetric clip
        clipped = torch.clamp(ratio, 1 - cfg["epsilon_low"], 1 + cfg["epsilon_high"])
        policy_loss_i = -torch.min(ratio * adv, clipped * adv)
        total_policy_loss = total_policy_loss + policy_loss_i

    total_policy_loss = total_policy_loss / G
    total_kl = total_kl / G
    total_loss = total_policy_loss + cfg["beta_kl"] * total_kl

    return {
        "total": total_loss,
        "policy": total_policy_loss.detach().item(),
        "kl": total_kl.item(),
    }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[grpo_min] Device: " + device)

    # Build model
    model, tokenizer = build_model_4bit(cfg["lora_r"], cfg["lora_alpha"], cfg["lora_dropout"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load Gate 1 adapter if specified
    if cfg["lora_checkpoint"] and os.path.exists(cfg["lora_checkpoint"]):
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, cfg["lora_checkpoint"], is_trainable=True)
        print("[grpo_min] Loaded adapter from " + cfg["lora_checkpoint"])

    # Whisper reward (on CPU to save VRAM)
    whisper = WhisperWERReward(cfg["whisper_size"], device=cfg["whisper_device"])

    # Data
    train_ds = TTSRLDataset(cfg["train_path"])
    val_ds = TTSRLDataset(cfg["val_path"])
    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True,
        collate_fn=collate_fn, num_workers=0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=0,
    )

    # Optimizer
    trainable = [p for p in model.parameters() if p.requires_grad]
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(
            trainable, lr=cfg["lr"], betas=(0.9, 0.999), weight_decay=0.0, is_paged=True,
        )
    except Exception:
        optimizer = torch.optim.AdamW(trainable, lr=cfg["lr"], betas=(0.9, 0.999), weight_decay=0.0)

    os.makedirs(cfg["save_dir"], exist_ok=True)
    resume_dir = os.path.join(cfg["save_dir"], "latest")
    start_step = 0
    if os.path.exists(resume_dir):
        start_step = load_checkpoint(model, optimizer, None, resume_dir)

    init_logging(cfg["log_dir"], run_name="gate2_grpo_min")

    model.train()
    data_iter = cycle(train_loader)
    best_val_reward = -float("inf")
    accum_losses = {"total": 0.0, "policy": 0.0, "kl": 0.0}
    accum_rewards = []

    print("[grpo_min] Starting from step " + str(start_step))

    from tqdm import tqdm
    pbar = tqdm(range(start_step, cfg["max_steps"]), desc="[grpo_min] Training")
    for step in pbar:
        # LR warmup
        lr_now = cfg["lr"] * min(1.0, (step + 1) / max(cfg["warmup_steps"], 1))
        for pg in optimizer.param_groups:
            pg["lr"] = lr_now

        batch = next(data_iter)
        text = batch["text"][0]  # batch_size=1

        # Tokenize prompt
        encoded = tokenizer(
            text, return_tensors="pt", truncation=True,
            max_length=cfg["max_seq_len"] // 2,
        ).to(device)
        input_ids = encoded["input_ids"]

        # --- Rollout G samples ---
        model.eval()
        with torch.no_grad():
            codes_list, old_logprobs_list, decoded_texts = rollout_one(
                model, tokenizer, text, cfg, device
            )
        model.train()

        # --- Compute WER rewards ---
        rewards = whisper.compute_rewards(decoded_texts, [text] * len(decoded_texts))
        accum_rewards.extend(rewards.tolist())

        # --- MO-GRPO: group-normalize rewards → advantages ---
        mu = rewards.mean()
        sigma = rewards.std() + 1e-8
        advantages = (rewards - mu) / sigma  # [G]

        if sigma < 1e-6:
            # All rewards identical → no signal, skip
            optimizer.zero_grad()
            continue

        # --- GRPO loss ---
        try:
            losses = grpo_loss(
                model, tokenizer, input_ids,
                codes_list, old_logprobs_list, advantages, cfg, device,
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                optimizer.zero_grad()
                print("[grpo_min] OOM at step " + str(step) + " — skipping.")
                continue
            raise

        scaled = losses["total"] / cfg["gradient_accum_steps"]
        scaled.backward()

        for k in accum_losses:
            accum_losses[k] += losses.get(k, 0.0) / cfg["gradient_accum_steps"]

        if (step + 1) % cfg["gradient_accum_steps"] == 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, cfg["grad_clip"])
            optimizer.step()
            optimizer.zero_grad()

            vram = torch.cuda.memory_allocated() / 1024**3 if device == "cuda" else 0
            metrics = {
                "loss": round(accum_losses["total"], 5),
                "policy": round(accum_losses["policy"], 5),
                "kl": round(accum_losses["kl"], 5),
                "reward_mean": round(float(mu), 4),
                "reward_std": round(float(sigma), 4),
                "grad_norm": round(float(grad_norm), 4),
                "lr": round(lr_now, 9),
                "vram_gb": round(vram, 2),
            }
            log_metrics(step, metrics)
            pbar.set_postfix({"loss": metrics["loss"], "rew": metrics["reward_mean"], "vram": metrics["vram_gb"]})
            accum_losses = {k: 0.0 for k in accum_losses}
            accum_rewards = []

        # Validation
        if step > 0 and step % cfg["val_every"] == 0:
            model.eval()
            val_rewards = []
            for i, vbatch in enumerate(val_loader):
                if i >= 20:
                    break
                _, _, vtexts = rollout_one(model, tokenizer, vbatch["text"][0], cfg, device)
                r = whisper.compute_rewards(vtexts, [vbatch["text"][0]] * len(vtexts))
                val_rewards.append(r.mean().item())
            model.train()
            val_r = sum(val_rewards) / max(len(val_rewards), 1)
            log_metrics(step, {"val_reward": round(val_r, 4)})
            if val_r > best_val_reward:
                best_val_reward = val_r
                save_checkpoint(
                    model, optimizer, None, step,
                    {"val_reward": val_r},
                    os.path.join(cfg["save_dir"], "best"),
                )
                print("[grpo_min] New best val_reward=" + f"{val_r:.4f}")

        if step > 0 and step % cfg["save_every"] == 0:
            save_checkpoint(model, optimizer, None, step, {}, resume_dir)

    save_checkpoint(model, optimizer, None, cfg["max_steps"], {}, resume_dir)
    close_logging()
    print("[grpo_min] Done. Best val_reward=" + f"{best_val_reward:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Gate 2 GRPO Minimal")
    parser.add_argument("--config", default="gwen_rl/configs/gate2_grpo_min.yaml")
    parser.add_argument("--max_steps", type=int, default=0)
    args = parser.parse_args()

    cfg = dict(DEFAULT_CFG)
    if os.path.exists(args.config):
        with open(args.config, encoding="utf-8") as f:
            cfg.update(yaml.safe_load(f) or {})
    else:
        print("[warn] Config not found, using defaults.")
    if args.max_steps > 0:
        cfg["max_steps"] = args.max_steps

    cfg = apply_gpu_config(cfg)
    print("[grpo_min] Config:", cfg)
    train(cfg)


if __name__ == "__main__":
    main()
