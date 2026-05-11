"""
gwen_rl/smoke_test.py
End-to-end import + logic check (does NOT load gwen-tts-0.6B model — needs internet).
Tests all utility functions with dummy data to verify code runs before real training.

Usage:
    py -3 gwen_rl/smoke_test.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np


def test_imports():
    print("=== Testing imports ===")
    from gwen_rl.utils.audio import load_audio, compute_snr_db, is_valid_audio, resample
    from gwen_rl.utils.log import init_logging, log_metrics, close_logging
    from gwen_rl.utils.checkpoint import save_checkpoint, load_checkpoint
    from gwen_rl.utils.data import TTSRLDataset, collate_fn
    from gwen_rl.utils.model import MODEL_ID
    from gwen_rl.rewards.whisper_wer import WhisperWERReward
    from gwen_rl.rewards.wavlm_sim import WavLMSimReward
    from gwen_rl.rewards.length_entropy import length_reward, entropy_reward
    print("  [OK] All imports passed")


def test_audio_utils():
    print("=== Testing audio utilities ===")
    from gwen_rl.utils.audio import compute_snr_db, is_valid_audio, resample, SR_TARGET

    # Synthetic 3s audio at 24kHz
    wav = torch.sin(torch.linspace(0, 300, SR_TARGET * 3)) * 0.5
    snr = compute_snr_db(wav)
    print(f"  SNR estimate: {snr:.1f} dB")

    valid = is_valid_audio(wav, SR_TARGET, min_dur=2.0, max_dur=15.0, min_snr_db=0.0)
    assert valid, "3s audio should pass filter"

    wav_16k = resample(wav, SR_TARGET, 16000)
    expected = round(wav.size(0) * 16000 / SR_TARGET)
    assert abs(wav_16k.size(0) - expected) <= 2, f"Resample wrong length: {wav_16k.size(0)} vs {expected}"
    print("  [OK] Audio utils passed")


def test_length_entropy():
    print("=== Testing length/entropy rewards ===")
    from gwen_rl.rewards.length_entropy import length_reward, entropy_reward
    from gwen_rl.utils.audio import SR_TARGET

    wavs = [torch.zeros(SR_TARGET * 3), torch.zeros(SR_TARGET * 3)]
    r_len = length_reward(wavs, ref_texts=["xin chao ban oi"] * 2, ref_durations=[3.0, 3.0])
    assert r_len.shape == (2,), "Wrong shape"
    print(f"  Length rewards: {r_len.tolist()}")

    log_probs = [torch.full((50,), -2.0), torch.full((50,), -0.5)]
    r_ent = entropy_reward(log_probs, h_target=1.5, lambda_ent=0.5)
    assert r_ent.shape == (2,)
    print(f"  Entropy rewards: {r_ent.tolist()}")
    print("  [OK] Length/entropy rewards passed")


def test_configs():
    print("=== Testing YAML configs ===")
    import yaml
    config_dir = os.path.join(os.path.dirname(__file__), "configs")
    for fname in ["gate1_sft.yaml", "gate2_grpo_min.yaml", "gate3_grpo_full.yaml"]:
        path = os.path.join(config_dir, fname)
        assert os.path.exists(path), "Config not found: " + fname
        with open(path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        assert isinstance(cfg, dict), fname + " is not a valid dict"
        assert "lr" in cfg, fname + " missing 'lr' key"
        print("  [OK] " + fname)
    print("  [OK] All configs valid")


def test_whisper_reward():
    print("=== Testing Whisper WER reward (CPU, tiny) ===")
    from gwen_rl.rewards.whisper_wer import WhisperWERReward

    whisper = WhisperWERReward(model_size="tiny", device="cpu")

    # Synthetic silence → Whisper will produce empty or garbage transcript → reward ≈ 0
    dummy_wav = np.zeros(16000 * 2, dtype=np.float32)  # 2s silence @ 16kHz
    rewards = whisper.compute_rewards([dummy_wav], ref_texts=["xin chao"])
    assert rewards.shape == (1,)
    assert 0.0 <= rewards[0].item() <= 1.0
    print(f"  Reward for silence: {rewards[0].item():.3f} (expected near 0)")
    print("  [OK] Whisper reward passed")


def main():
    test_imports()
    test_audio_utils()
    test_length_entropy()
    test_configs()
    test_whisper_reward()

    print()
    print("=" * 50)
    print("ALL SMOKE TESTS PASSED")
    print("=" * 50)
    print()
    print("Next steps:")
    print("  1. Prepare data: py -3 -m gwen_rl.scripts.preprocess --manifest data/raw/manifest.tsv")
    print("  2. Gate 1 test:  py -3 -m gwen_rl.scripts.train_sft --config gwen_rl/configs/gate1_sft.yaml --max_steps 10")
    print("  3. Gate 1 full:  py -3 -m gwen_rl.scripts.train_sft --config gwen_rl/configs/gate1_sft.yaml")
    print("  4. Gate 2 test:  py -3 -m gwen_rl.scripts.train_grpo_minimal --config gwen_rl/configs/gate2_grpo_min.yaml --max_steps 5")
    print("  5. Gate 2 full:  py -3 -m gwen_rl.scripts.train_grpo_minimal --config gwen_rl/configs/gate2_grpo_min.yaml")
    print("  6. Gate 3 full:  py -3 -m gwen_rl.scripts.train_grpo_full --config gwen_rl/configs/gate3_grpo_full.yaml")


if __name__ == "__main__":
    main()
