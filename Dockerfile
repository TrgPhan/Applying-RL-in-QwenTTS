# Dockerfile for gwen_rl GRPO training
# Updated for reproducible environments

FROM pytorch/pytorch:2.3.0-cuda11.8-cudnn8-runtime

WORKDIR /workspace

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
# Install from requirements.txt to ensure git-based transformers and qwen-tts are included
RUN pip install --no-cache-dir -r requirements.txt

# Copy project code
COPY gwen_rl/ ./gwen_rl/

# Create runtime dirs
RUN mkdir -p data/audio data/processed checkpoints/gate1 checkpoints/gate2 checkpoints/gate3 logs model_cache

# Environment
ENV HF_HOME=/workspace/model_cache
ENV TRANSFORMERS_CACHE=/workspace/model_cache
ENV PYTHONPATH=/workspace
ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_VERBOSITY=error

# Verify imports
RUN python -c "
import transformers
from gwen_rl.utils.audio import load_audio
from gwen_rl.utils.log import init_logging
print('Transformers version:', transformers.__version__)
print('All imports OK')
"

# Default command
CMD ["python", "-m", "gwen_rl.scripts.train_sft", "--config", "gwen_rl/configs/gate1_sft.yaml"]
