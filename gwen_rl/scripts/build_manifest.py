"""
gwen_rl/scripts/build_manifest.py
Convert data/metadata.csv (audio, text, source) → processed dataset.
"""

import os
import sys
import csv
import argparse
import random
import torch
import torchaudio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from gwen_rl.utils.audio import load_audio, is_valid_audio, compute_snr_db, SR_TARGET


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",       default="data/metadata.csv")
    parser.add_argument("--audio_dir", default="data/audio")
    parser.add_argument("--out_dir",   default="data/processed")
    parser.add_argument("--val_ratio",  type=float, default=0.05)
    parser.add_argument("--test_ratio", type=float, default=0.05)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--min_snr",    type=float, default=0.0,  help="SNR threshold dB")
    parser.add_argument("--min_dur",    type=float, default=2.0)
    parser.add_argument("--max_dur",    type=float, default=25.0)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Read CSV
    with open(args.csv, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    print(f"[build_manifest] Found {len(rows)} rows in {args.csv}")

    random.seed(args.seed)
    random.shuffle(rows)

    processed = []
    skip_reasons = {"not_found": 0, "too_short": 0, "too_long": 0, "low_snr": 0, "error": 0}

    for i, row in enumerate(rows):
        audio_file = row["audio"].strip()
        text = row["text"].strip()
        audio_path = os.path.join(args.audio_dir, audio_file)

        if not os.path.exists(audio_path):
            skip_reasons["not_found"] += 1
            continue

        try:
            wav = load_audio(audio_path, SR_TARGET)
        except Exception:
            skip_reasons["error"] += 1
            continue

        dur = wav.size(0) / SR_TARGET
        if dur < args.min_dur:
            skip_reasons["too_short"] += 1
            continue
        if dur > args.max_dur:
            skip_reasons["too_long"] += 1
            continue
        
        snr = compute_snr_db(wav)
        if snr < args.min_snr:
            skip_reasons["low_snr"] += 1
            continue

        processed.append({
            "text":           text,
            "ref_text":       text,          # self-reference
            "ref_audio_path": audio_path,    # same file as target
            "target_wav":     wav,           # [T] float32 @ 24kHz
            "duration":       dur,
            "audio_file":     audio_file,
        })

        if (i + 1) % 200 == 0:
            kept = len(processed)
            skipped = sum(skip_reasons.values())
            print(f"  Processed {i+1}/{len(rows)}, kept {kept}, skipped {skipped}")

    print(f"[build_manifest] Done processing.")
    print(f"  - Kept: {len(processed)}")
    for reason, count in skip_reasons.items():
        if count > 0:
            print(f"  - Skipped ({reason}): {count}")

    # Split
    n = len(processed)
    if n == 0:
        print("[error] No samples kept! Check your audio paths and duration filters.")
        return

    n_test = max(1, int(n * args.test_ratio))
    n_val  = max(1, int(n * args.val_ratio))
    n_train = n - n_val - n_test

    train_data = processed[:n_train]
    val_data   = processed[n_train:n_train + n_val]
    test_data  = processed[n_train + n_val:]

    torch.save(train_data, os.path.join(args.out_dir, "train.pt"))
    torch.save(val_data,   os.path.join(args.out_dir, "val.pt"))
    torch.save(test_data,  os.path.join(args.out_dir, "test.pt"))

    print(f"[build_manifest] Saved to {args.out_dir}:")
    print(f"  train: {len(train_data)}")
    print(f"  val:   {len(val_data)}")
    print(f"  test:  {len(test_data)}")


if __name__ == "__main__":
    main()
