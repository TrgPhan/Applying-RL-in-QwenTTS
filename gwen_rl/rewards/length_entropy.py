"""
gwen_rl/rewards/length_entropy.py
Length penalty and entropy rewards (no model needed, cheap).

length_reward: 1.0 if duration ratio within [0.85, 1.15] vs reference, else 0.0
entropy_reward: penalize if avg token entropy < h_target (collapse prevention)
"""

import torch
import math


def length_reward(
    gen_wavs: list,         # list of [T] tensors @ 24kHz
    ref_texts: list,        # list of str (word count for speaking rate)
    ref_durations: list,    # list of float (ref audio duration in seconds)
    sr: int = 24000,
    tol: float = 0.15,      # tolerance: [1-tol, 1+tol]
) -> torch.Tensor:
    """
    Reward 1.0 if speaking rate ratio (gen/ref) is within tolerance.
    """
    rewards = []
    for gen_wav, ref_text, ref_dur in zip(gen_wavs, ref_texts, ref_durations):
        gen_dur = gen_wav.size(0) / sr
        ref_words = max(len(ref_text.split()), 1)
        gen_rate = gen_dur / ref_words         # sec/word for generated
        ref_rate = ref_dur / ref_words         # sec/word for reference
        ratio = gen_rate / (ref_rate + 1e-9)
        rewards.append(1.0 if (1 - tol) <= ratio <= (1 + tol) else 0.0)
    return torch.tensor(rewards, dtype=torch.float32)


def entropy_reward(
    log_probs_list: list,   # list of [T] tensors of token log-probs
    h_target: float = 1.5,  # target entropy in nats; penalize if below
    lambda_ent: float = 0.5,
) -> torch.Tensor:
    """
    Penalize low entropy (mode collapse):
      r_ent = -lambda_ent * max(0, h_target - avg_entropy)

    Average entropy approximated as -mean(log_prob_of_chosen_token).
    (This is a lower bound; actual entropy is higher.)
    """
    rewards = []
    for lp in log_probs_list:
        avg_h = float(-lp.mean())           # approximate entropy per token
        penalty = max(0.0, h_target - avg_h)
        rewards.append(-lambda_ent * penalty)
    return torch.tensor(rewards, dtype=torch.float32)
