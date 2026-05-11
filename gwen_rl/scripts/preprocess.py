"""
gwen_rl/scripts/preprocess.py
Build preprocessed dataset for RL training from raw audio files.

Input: a folder of .wav/.flac files with a matching TSV/CSV/JSONL manifest.
Output: data/processed/train.pt, val.pt, test.pt

Manifest format (one line per sample, tab-separated):
    audio_path<TAB>text<TAB>ref_audio_path<TAB>ref_text

If ref_audio_path == audio_path (self-reference), that is fine.
If you don't have ref_audio, set ref_audio_path = audio_path and ref_text = text.

Usage:
    py -3 -m gwen_rl.scripts.preprocess \\
        --manifest data/raw/manifest.tsv \\
        --out_dir  data/processed \\
        --val_ratio 0.05 \\
        --test_ratio 0.05
"""

import os
import sys
import json
import random
import argparse
import torch
import torchaudio

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from gwen_rl.utils.audio import load_audio, is_valid_audio, SR_TARGET


def read_manifest(manifest_path: str) -> list:
    """
    Support TSV (tab-separated) and JSONL formats.
    TSV columns: audio_path  text  ref_audio_path  ref_text
    JSONL keys: audio_path, text, ref_audio_path, ref_text
    """
    items = []
    with open(manifest_path, encoding="utf-8") as f:
        first = f.readline().strip()
        f.seek(0)

        if first.startswith("{"):
            # JSONL
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
        else:
            # TSV (skip header if it contains "audio_path")
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 2:
                    continue
                if parts[0] == "audio_path":
                    continue  # header row
                audio_path = parts[0]
                text = parts[1] if len(parts) > 1 else ""
                ref_audio = parts[2] if len(parts) > 2 else audio_path
                ref_text = parts[3] if len(parts) > 3 else text
                items.append({
                    "audio_path": audio_path,
                    "text": text,
                    "ref_audio_path": ref_audio,
                    "ref_text": ref_text,
                })
    return items


def process_item(item: dict) -> dict | None:
    """Load, validate, return dict ready for training. Returns None if filtered."""
    try:
        wav = load_audio(item["audio_path"], SR_TARGET)
    except Exception as e:
        print("[skip] Load error: " + item["audio_path"] + " -> " + str(e))
        return None

    if not is_valid_audio(wav, SR_TARGET, min_dur=2.0, max_dur=15.0, min_snr_db=10.0):
        return None

    # Verify ref audio is loadable (don't store full wav to save disk space — load at train time)
    if item["ref_audio_path"] != item["audio_path"]:
        try:
            load_audio(item["ref_audio_path"], SR_TARGET)
        except Exception:
            return None

    return {
        "text": item["text"].strip(),
        "ref_text": item["ref_text"].strip(),
        "ref_audio_path": item["ref_audio_path"],
        "target_wav": wav,                          # [T] @ 24kHz
        "duration": wav.size(0) / SR_TARGET,
    }


def main():
    parser = argparse.ArgumentParser(description="Preprocess TTS data for RL training")
    parser.add_argument("--manifest", required=True, help="Path to manifest TSV/JSONL")
    parser.add_argument("--out_dir", default="data/processed", help="Output directory")
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--test_ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=0, help="0 = no limit")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    raw_items = read_manifest(args.manifest)
    print("[preprocess] Found " + str(len(raw_items)) + " raw items in manifest.")

    random.seed(args.seed)
    random.shuffle(raw_items)

    if args.max_samples > 0:
        raw_items = raw_items[:args.max_samples]

    processed = []
    for i, item in enumerate(raw_items):
        if i % 500 == 0:
            print(f"  Processing {i}/{len(raw_items)} ...")
        result = process_item(item)
        if result is not None:
            processed.append(result)

    print("[preprocess] Kept " + str(len(processed)) + " / " + str(len(raw_items)) + " items.")

    # Split
    n = len(processed)
    n_test = max(1, int(n * args.test_ratio))
    n_val = max(1, int(n * args.val_ratio))
    n_train = n - n_val - n_test

    train = processed[:n_train]
    val = processed[n_train:n_train + n_val]
    test = processed[n_train + n_val:]

    torch.save(train, os.path.join(args.out_dir, "train.pt"))
    torch.save(val, os.path.join(args.out_dir, "val.pt"))
    torch.save(test, os.path.join(args.out_dir, "test.pt"))

    print("[preprocess] Saved:")
    print("  train: " + str(len(train)) + " -> " + os.path.join(args.out_dir, "train.pt"))
    print("  val:   " + str(len(val)) + " -> " + os.path.join(args.out_dir, "val.pt"))
    print("  test:  " + str(len(test)) + " -> " + os.path.join(args.out_dir, "test.pt"))


if __name__ == "__main__":
    main()
