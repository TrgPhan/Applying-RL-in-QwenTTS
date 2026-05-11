"""
gwen_rl/scripts/train_sft.py
Gate 1: SFT Continuation for gwen-tts-0.6B.

Loss: L = CE(codec_tokens)
      (KL anchor removed — counterproductive for small-dataset speaker adaptation)

Input to model:
  text_ids  [B, T_text]  — Qwen3 BPE tokens from metadata.csv "text"
  codec_ids [B, T_codec] — real Mimi codec tokens (pre-encoded by encode_codec.py)

Prerequisites:
  1. Build dataset:  python -m gwen_rl.scripts.build_manifest --csv data/metadata.csv ...
  2. Encode codecs:  python -m gwen_rl.scripts.encode_codec
  3. Train:          python -m gwen_rl.scripts.train_sft --config gwen_rl/configs/gate1_sft.yaml
"""

import os
import sys
import math
import argparse
import yaml
import warnings
warnings.filterwarnings("ignore", message=".*flash attention.*")

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from itertools import cycle
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gwen_rl.utils.model import build_model, CODEC_BOS_ID
from gwen_rl.utils.data import TTSRLDataset, collate_fn
from gwen_rl.utils.checkpoint import save_checkpoint, load_checkpoint
from gwen_rl.utils.log import init_logging, log_metrics, close_logging
from gwen_rl.utils.gpu_config import apply_gpu_config


DEFAULT_CFG = {
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,

    "train_path": "data/processed/train.pt",
    "val_path":   "data/processed/val.pt",
    "save_dir":   "checkpoints/gate1",
    "log_dir":    "logs",

    "lr": 5e-5,
    "beta_kl": 0.0,
    "grad_clip": 1.0,
    "warmup_steps": 200,
    "batch_size": 1,
    "gradient_accum_steps": 8,
    "max_seq_len": 128,        # max codec tokens per sample
    "max_steps": 5000,
    "val_every": 200,
    "save_every": 1000,

    # Early stopping: stop if val_ce does not improve for this many val checks
    "early_stop_patience": 10,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _has_real_codec(dataset: TTSRLDataset) -> bool:
    """Check whether the dataset contains real codec tokens."""
    if len(dataset) == 0:
        return False
    sample = dataset[0]
    return "codec_ids" in sample and isinstance(sample["codec_ids"], torch.Tensor)


def tokenize_batch(batch, tokenizer, max_seq_len, device):
    """
    Tokenize text → text_ids [B, T_text].
    Codec_ids are taken directly from the batch (aligned by DataLoader).
    Returns (text_ids, codec_ids) both on `device`.
    """
    # Apply Qwen chat template format (critical for gwen-tts base model)
    formatted_texts = [
        f"<|im_start|>user\n{t}<|im_end|>\n<|im_start|>assistant\n"
        for t in batch["text"]
    ]
    encoded = tokenizer(
        formatted_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128,
    )
    text_ids = encoded["input_ids"].to(device)

    # Get codec_ids from the batch (correctly aligned by collate_fn)
    codec_ids_batch = batch["codec_ids"]  # list of [T_i] LongTensors

    # Pad/truncate codec tokens to max_seq_len, prepend CODEC_BOS_ID
    B = len(codec_ids_batch)
    T = min(max_seq_len, max(c.size(0) + 1 for c in codec_ids_batch))
    codec_tensor = torch.zeros(B, T, dtype=torch.long, device=device)
    for i, c in enumerate(codec_ids_batch):
        c_with_bos = torch.cat([
            torch.tensor([CODEC_BOS_ID], dtype=c.dtype, device=c.device), c
        ])
        length = min(c_with_bos.size(0), T)
        codec_tensor[i, :length] = c_with_bos[:length].to(device)

    return text_ids, codec_tensor


def sft_step(model, tokenizer, batch, cfg, device):
    """Single SFT step: CE loss only (right now i will use no KL anchor — small dataset)."""
    text_ids, codec_ids = tokenize_batch(
        batch, tokenizer, cfg["max_seq_len"], device
    )

    # Forward: logits [B, T_codec, 3072]
    with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(device == "cuda")):
        logits = model(text_ids, codec_ids)
    logits = logits.float()

    # Causal CE: predict token[t+1] from token[t]
    shift_logits = logits[:, :-1, :].contiguous()          # [B, T-1, 3072]
    shift_labels = codec_ids[:, 1:].contiguous()           # [B, T-1]

    ce_loss = F.cross_entropy(
        shift_logits.view(-1, 3072),
        shift_labels.view(-1),
        ignore_index=0,      # ignore padding token
        reduction="mean",
    )

    # Optional KL divergence with frozen reference (LoRA disabled)
    beta_kl = cfg.get("beta_kl", 0.0)
    kl_val = 0.0
    total = ce_loss

    if beta_kl > 0:
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(device == "cuda")):
            ref_logits = model.get_ref_logits(text_ids, codec_ids).float()
        ref_shift = ref_logits[:, :-1, :].contiguous()

        kl = F.kl_div(
            F.log_softmax(shift_logits, dim=-1),
            F.log_softmax(ref_shift, dim=-1),
            reduction="batchmean",
            log_target=True,
        )
        kl_val = kl.detach().item()
        total = ce_loss + beta_kl * kl

    return {"total": total, "ce": ce_loss.detach().item(), "kl": kl_val}


@torch.no_grad()
def validate(model, tokenizer, val_loader, cfg, device, max_batches=None):
    model.eval()
    total_ce, n = 0.0, 0
    limit = max_batches or len(val_loader)
    for i, batch in enumerate(val_loader):
        if i >= limit:
            break
        # codec_ids come from the batch directly (correctly aligned)
        text_ids, codec_ids = tokenize_batch(
            batch, tokenizer, cfg["max_seq_len"], device
        )
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(device == "cuda")):
            logits = model(text_ids, codec_ids).float()
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = codec_ids[:, 1:].contiguous()
        ce = F.cross_entropy(
            shift_logits.view(-1, 3072),
            shift_labels.view(-1),
            ignore_index=0,
            reduction="mean",
        )
        total_ce += ce.item()
        n += 1
    model.train()
    avg_ce = total_ce / max(n, 1)
    return {"val_ce": round(avg_ce, 4), "val_ppl": round(math.exp(min(avg_ce, 20)), 4)}


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[sft] Device: {device} | max_steps: {cfg['max_steps']}")

    model, tokenizer = build_model(
        lora_r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
    )

    train_ds = TTSRLDataset(cfg["train_path"])
    val_ds   = TTSRLDataset(cfg["val_path"])

    # Check real codec availability
    has_codec = _has_real_codec(train_ds)
    if has_codec:
        print("[sft] ✓ Real Mimi codec tokens found — training with actual audio targets")
    else:
        print("[sft] ⚠  No codec_ids in dataset. Run encode_codec.py first for meaningful loss.")
        print("[sft] ⚠  Falling back to random codec tokens (for sanity-check only).")

    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True,
        collate_fn=collate_fn, num_workers=0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )

    trainable = [p for p in model.parameters() if p.requires_grad]
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(
            trainable, lr=cfg["lr"], betas=(0.9, 0.95), weight_decay=0.01,
        )
        print("[sft] Optimizer: 8-bit AdamW")
    except Exception:
        optimizer = torch.optim.AdamW(
            trainable, lr=cfg["lr"], betas=(0.9, 0.95), weight_decay=0.01
        )
        print("[sft] Optimizer: AdamW fp32")

    os.makedirs(cfg["save_dir"], exist_ok=True)
    resume_dir = os.path.join(cfg["save_dir"], "latest")
    start_step = 0
    best_val_ppl = float("inf")
    if os.path.exists(resume_dir):
        start_step, best_val_ppl = load_checkpoint(model.transformer, optimizer, None, resume_dir)

    init_logging(cfg["log_dir"], run_name="gate1_sft")
    model.train()

    data_iter = cycle(iter(train_loader))
    accum = {"total": 0.0, "ce": 0.0, "kl": 0.0}
    patience_counter = 0
    patience = cfg.get("early_stop_patience", 10)

    pbar = tqdm(range(start_step, cfg["max_steps"]), desc="[sft]", dynamic_ncols=True)
    for step in pbar:
        # LR schedule: Linear warmup + Cosine decay
        warmup = max(cfg["warmup_steps"], 1)
        if step < warmup:
            lr_now = cfg["lr"] * ((step + 1) / warmup)
        else:
            progress = (step - warmup) / max(1, cfg["max_steps"] - warmup)
            lr_now = cfg["lr"] * 0.5 * (1.0 + math.cos(math.pi * progress))
            # Keep a minimum LR floor
            lr_now = max(lr_now, cfg["lr"] * 0.05)
            
        for pg in optimizer.param_groups:
            pg["lr"] = lr_now

        batch = next(data_iter)

        try:
            losses = sft_step(model, tokenizer, batch, cfg, device)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                optimizer.zero_grad()
                print(f"\n[sft] OOM at step {step} — skipping")
                continue
            raise

        scaled = losses["total"] / cfg["gradient_accum_steps"]
        scaled.backward()
        for k in accum:
            accum[k] += losses.get(k, 0.0) / cfg["gradient_accum_steps"]

        if (step + 1) % cfg["gradient_accum_steps"] == 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, cfg["grad_clip"])
            optimizer.step()
            optimizer.zero_grad()

            if step % 20 == 0:
                vram = torch.cuda.memory_allocated() / 1024**3 if device == "cuda" else 0
                metrics = {
                    "loss": round(accum["total"], 4),
                    "ce":   round(accum["ce"], 4),
                    "kl":   round(accum["kl"], 4),
                    "gnorm": round(float(grad_norm), 3),
                    "lr":   round(lr_now, 7),
                    "vram": round(vram, 2),
                }
                log_metrics(step, metrics)
                pbar.set_postfix(
                    loss=metrics["loss"], ce=metrics["ce"], vram=metrics["vram"]
                )
            accum = {k: 0.0 for k in accum}

        # Validation + Early Stopping
        if step > 0 and step % cfg["val_every"] == 0:
            val_m = validate(model, tokenizer, val_loader, cfg, device)
            log_metrics(step, val_m)
            ppl = val_m["val_ppl"]
            if ppl < best_val_ppl:
                best_val_ppl = ppl
                patience_counter = 0
                save_checkpoint(
                    model.transformer, optimizer, None, step, val_m,
                    os.path.join(cfg["save_dir"], "best"),
                )
                tqdm.write(f"[sft] ✓ step={step}  val_ce={val_m['val_ce']}  val_ppl={ppl}  (new best → saved)")
            else:
                patience_counter += 1
                tqdm.write(
                    f"[sft]   step={step}  val_ce={val_m['val_ce']}  val_ppl={ppl}"
                    f"  (no improvement {patience_counter}/{patience})"
                )
                if patience_counter >= patience:
                    tqdm.write(f"[sft] Early stopping triggered at step {step}.")
                    break

        if step > 0 and step % cfg["save_every"] == 0:
            save_checkpoint(model.transformer, optimizer, None, step, {}, resume_dir)

    save_checkpoint(model.transformer, optimizer, None, step, {}, resume_dir)
    close_logging()
    print(f"[sft] Done. Best val PPL={best_val_ppl:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",    default="gwen_rl/configs/gate1_sft.yaml")
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
    print("[sft] Config:", {k: v for k, v in cfg.items() if k != "model_id"})
    train(cfg)


if __name__ == "__main__":
    main()
