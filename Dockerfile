# Dockerfile for gwen_rl GRPO training
# Compatible with: torch 2.3 + CUDA 11.8 / 12.x
#
# === BUILD ===
#   docker build -t gwen-rl:latest .
#
# === RUN (cloud GPU VM) ===
#   docker run --gpus all --rm -it \
#     -v /path/to/your/data:/workspace/data \
#     -v /path/to/checkpoints:/workspace/checkpoints \
#     -v /path/to/model_cache:/workspace/model_cache \
#     gwen-rl:latest \
#     python -m gwen_rl.scripts.train_sft --config gwen_rl/configs/gate1_sft.yaml
#
# === DATA FORMAT ===
#   Mount your data/audio/ and data/processed/ (pre-built via build_manifest.py)
#   Or run preprocessing inside container with:
#     python -m gwen_rl.scripts.build_manifest --csv data/metadata.csv ...

FROM pytorch/pytorch:2.3.0-cuda11.8-cudnn8-runtime

WORKDIR /workspace

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir \
    transformers>=4.57.0 \
    peft==0.10.0 \
    accelerate>=1.12.0 \
    bitsandbytes \
    faster-whisper \
    jiwer \
    soundfile \
    safetensors \
    huggingface_hub \
    pyyaml \
    tqdm \
    numpy \
    || pip install --no-cache-dir -r requirements.txt

# Copy project code (excludes data/checkpoints via .dockerignore)
COPY gwen_rl/ ./gwen_rl/

# Create runtime dirs
RUN mkdir -p data/audio data/processed checkpoints/gate1 checkpoints/gate2 checkpoints/gate3 logs model_cache

# Environment
ENV HF_HOME=/workspace/model_cache
ENV TRANSFORMERS_CACHE=/workspace/model_cache
ENV PYTHONPATH=/workspace
ENV PYTHONUNBUFFERED=1
# Suppress transformers warning about qwen3_tts architecture mismatch
ENV TRANSFORMERS_VERBOSITY=error

# Verify imports at build time (fast check, no GPU needed)
RUN python -c "
from gwen_rl.utils.audio import load_audio, compute_snr_db
from gwen_rl.utils.log import init_logging
from gwen_rl.utils.gpu_config import apply_gpu_config
from gwen_rl.rewards.whisper_wer import WhisperWERReward
from gwen_rl.rewards.length_entropy import length_reward, entropy_reward
import yaml
print('All imports OK')
for f in ['gwen_rl/configs/gate1_sft.yaml', 'gwen_rl/configs/gate2_grpo_min.yaml']:
    with open(f) as fh:
        cfg = yaml.safe_load(fh)
    assert 'lr' in cfg
print('All configs OK')
print('Docker image verified successfully')
"

# Default command: Gate 1 SFT (auto-detects GPU, adjusts config)
CMD ["python", "-m", "gwen_rl.scripts.train_sft", "--config", "gwen_rl/configs/gate1_sft.yaml"]
