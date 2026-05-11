"""
gwen_rl/scripts/build_manifest.py
Convert data/metadata.csv (audio, text, source) → processed dataset.

Format of metadata.csv:
    audio,text,source
    00000001.wav,"text here",0

Since this is self-reference voice cloning:
    ref_audio = target_audio (same file)
    ref_text  = text (same transcript)

Usage:
    py -3 -m gwen_rl.scripts.build_manifest \\
        --csv     data/metadata.csv \\
        --audio_dir data/audio \\
        --out_dir data/processed
"""

import os
import sys
import csv
import argparse
import random
import torch
import torchaudio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from gwen_rl.utils.audio import load_audio, is_valid_audio, SR_TARGET


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",       default="data/metadata.csv")
    parser.add_argument("--audio_dir", default="data/audio")
    parser.add_argument("--out_dir",   default="data/processed")
    parser.add_argument("--val_ratio",  type=float, default=0.05)
    parser.add_argument("--test_ratio", type=float, default=0.05)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--min_snr",    type=float, default=5.0,  help="SNR threshold dB (lower=keep more)")
    parser.add_argument("--min_dur",    type=float, default=1.5)
    parser.add_argument("--max_dur",    type=float, default=20.0)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Read CSV
    with open(args.csv, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    print(f"[build_manifest] Found {len(rows)} rows in {args.csv}")

    random.seed(args.seed)
    random.shuffle(rows)

    processed, skipped = [], 0
    for i, row in enumerate(rows):
        audio_file = row["audio"].strip()
        text = row["text"].strip()
        audio_path = os.path.join(args.audio_dir, audio_file)

        if not os.path.exists(audio_path):
            skipped += 1
            continue

        try:
            wav = load_audio(audio_path, SR_TARGET)
        except Exception as e:
            skipped += 1
            continue

        if not is_valid_audio(wav, SR_TARGET, min_dur=args.min_dur,
                              max_dur=args.max_dur, min_snr_db=args.min_snr):
            skipped += 1
            continue

        processed.append({
            "text":           text,
            "ref_text":       text,          # self-reference
            "ref_audio_path": audio_path,    # same file as target
            "target_wav":     wav,           # [T] float32 @ 24kHz
            "duration":       wav.size(0) / SR_TARGET,
            "audio_file":     audio_file,
        })

        if (i + 1) % 200 == 0:
            print(f"  Processed {i+1}/{len(rows)}, kept {len(processed)}, skipped {skipped}")

    print(f"[build_manifest] Kept {len(processed)} / {len(rows)} (skipped {skipped})")

    # Split
    n = len(processed)
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
