"""
gwen_rl/utils/data.py
Dataset and DataLoader for gwen-tts-0.6B RL training.

Expected data format (output of scripts/preprocess.py):
  torch.save(list_of_dicts, "data/processed/train.pt")

Each dict has keys:
  text          : str   — Vietnamese utterance (normalized)
  ref_text      : str   — reference text (normalized)
  ref_audio_path: str   — path to reference wav (24kHz)
  target_wav    : Tensor [T] float32 @ 24kHz  — target waveform for SIM reward
  duration      : float — seconds
"""

import torch
import random
from torch.utils.data import Dataset
from typing import List, Dict


class TTSRLDataset(Dataset):
    def __init__(self, data_path: str, max_dur: float = 12.0):
        self.items: List[Dict] = torch.load(data_path, weights_only=False)
        # Secondary filter in case preprocess missed some
        self.items = [x for x in self.items if x["duration"] <= max_dur]
        print("[data] Loaded " + str(len(self.items)) + " samples from " + data_path)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def collate_fn(batch: List[Dict]) -> Dict:
    """
    Collate a batch of TTS samples.
    target_wav is padded to max length.
    Strings are kept as lists (not tensored).
    """
    texts = [x["text"] for x in batch]
    ref_texts = [x["ref_text"] for x in batch]
    ref_audio_paths = [x["ref_audio_path"] for x in batch]
    durations = torch.tensor([x["duration"] for x in batch])

    # Pad target_wav to same length
    target_wavs = [x["target_wav"] for x in batch]
    max_len = max(w.size(0) for w in target_wavs)
    padded = torch.zeros(len(batch), max_len)
    wav_lens = torch.zeros(len(batch), dtype=torch.long)
    for i, w in enumerate(target_wavs):
        padded[i, : w.size(0)] = w
        wav_lens[i] = w.size(0)

    return {
        "text": texts,
        "ref_text": ref_texts,
        "ref_audio_path": ref_audio_paths,
        "target_wav": padded,           # [B, T_max]
        "wav_lens": wav_lens,           # [B]
        "duration": durations,          # [B]
    }
