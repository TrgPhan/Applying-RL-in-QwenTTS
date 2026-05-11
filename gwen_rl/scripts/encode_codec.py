"""
gwen_rl/scripts/encode_codec.py
Re-process existing train/val/test .pt files to add real Mimi codec tokens.

The speech tokenizer at model_cache/gwen-tts-0.6B/speech_tokenizer/ encodes
24kHz audio → codec token IDs at 12.5 fps (frame_rate).
codebook_size = 2048, num_quantizers = 32 (we only need quantizer-0 → vocab [0, 2048)).

Usage:
    python -m gwen_rl.scripts.encode_codec --model_cache model_cache/gwen-tts-0.6B
"""

import os
import sys
import argparse
import warnings
warnings.filterwarnings("ignore")

import torch
import torchaudio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from gwen_rl.utils.audio import load_audio, SR_TARGET

SPEECH_TOKENIZER_PATH = "model_cache/gwen-tts-0.6B/speech_tokenizer"
MAX_CODEC_LEN = 256   # cap at 256 frames @ 12.5fps ≈ 20s, keeps VRAM safe


def load_speech_tokenizer(tokenizer_path: str, device: str = "cuda"):
    """Load Qwen3TTSTokenizer wrapper."""
    print(f"[encode_codec] Loading Qwen3TTSTokenizer from {tokenizer_path}")
    from qwen_tts import Qwen3TTSTokenizer
    
    # Qwen3TTSTokenizer handles registration and loading
    tokenizer = Qwen3TTSTokenizer.from_pretrained(
        tokenizer_path,
        device_map=device,
        torch_dtype=torch.float16 if "cuda" in device else torch.float32
    )
    return tokenizer


def encode_wav(tokenizer, wav: torch.Tensor, sr_target: int, device: str):
    """Use the official tokenizer wrapper to encode."""
    # wav: [T] float32 on CPU
    wav_np = wav.cpu().numpy()
    
    # Qwen3TTSTokenizer.encode expects numpy array or path
    # Returns an object with .audio_codes as list of [T_frames, num_quantizers]
    with torch.no_grad():
        enc = tokenizer.encode(wav_np, sr=sr_target)
    
    # Use only quantizer-0 (semantic codebook)
    # audio_codes[0] shape is [T_frames, Q]
    codes = enc.audio_codes[0]
    codec_ids = codes[:, 0] # [T_frames] main codebook
    
    # Cap length
    codec_ids = codec_ids[:MAX_CODEC_LEN]
    return codec_ids.cpu()


def encode_split(speech_tokenizer, data_path: str, out_path: str, device: str):
    """Load a split .pt, add 'codec_ids' field, save back."""
    if not os.path.exists(data_path):
        print(f"[encode_codec] Skipping (not found): {data_path}")
        return

    from tqdm import tqdm
    items = torch.load(data_path, weights_only=False)
    pbar = tqdm(items, desc=f"[encode_codec] {os.path.basename(data_path)}")

    encoded, failed = 0, 0
    for i, item in enumerate(pbar):
        if "codec_ids" in item:
            encoded += 1
            continue  # already done

        wav: torch.Tensor = item["target_wav"]   # [T] float32 @ 24kHz
        try:
            codec_ids = encode_wav(speech_tokenizer, wav, SR_TARGET, device)
            item["codec_ids"] = codec_ids        # [T_frames] LongTensor
            encoded += 1
        except Exception as e:
            # Fallback: uniform codec tokens (better than random)
            dur_frames = min(int(item["duration"] * 12.5), MAX_CODEC_LEN)
            item["codec_ids"] = torch.zeros(max(dur_frames, 4), dtype=torch.long)
            failed += 1

        if (i + 1) % 10 == 0:
            pbar.set_postfix(encoded=encoded, failed=failed)

    torch.save(items, out_path)
    print(f"[encode_codec] Saved {out_path} | encoded={encoded} failed={failed}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_cache", default="model_cache/gwen-tts-0.6B")
    parser.add_argument("--data_dir",    default="data/processed")
    parser.add_argument("--device",      default="cpu",
                        help="cpu recommended (speech tokenizer is small, keeps GPU for training)")
    args = parser.parse_args()

    tokenizer_path = os.path.join(args.model_cache, "speech_tokenizer")
    speech_tok = load_speech_tokenizer(tokenizer_path, device=args.device)

    for split in ["train", "val", "test"]:
        src = os.path.join(args.data_dir, f"{split}.pt")
        dst = os.path.join(args.data_dir, f"{split}.pt")   # overwrite in-place
        encode_split(speech_tok, src, dst, device=args.device)

    print("[encode_codec] Done. Re-run train_sft.py — loss will now be meaningful.")


if __name__ == "__main__":
    main()
