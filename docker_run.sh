#!/bin/bash
# docker_run.sh — Run gwen-rl training on cloud GPU VM
# Usage: bash docker_run.sh [gate1|gate2|gate3]
# Assumes Docker with NVIDIA runtime installed.

GATE=${1:-gate1}
DATA_DIR=${DATA_DIR:-$(pwd)/data}
CKPT_DIR=${CKPT_DIR:-$(pwd)/checkpoints}
CACHE_DIR=${CACHE_DIR:-$(pwd)/model_cache}

echo "=== gwen-rl GRPO Training ==="
echo "Gate:       $GATE"
echo "Data dir:   $DATA_DIR"
echo "Checkpoint: $CKPT_DIR"
echo "Model cache: $CACHE_DIR"
echo ""

case $GATE in
  gate1)
    CMD="python -m gwen_rl.scripts.train_sft --config gwen_rl/configs/gate1_sft.yaml"
    ;;
  gate2)
    CMD="python -m gwen_rl.scripts.train_grpo_minimal --config gwen_rl/configs/gate2_grpo_min.yaml"
    ;;
  gate3)
    CMD="python -m gwen_rl.scripts.train_grpo_full --config gwen_rl/configs/gate3_grpo_full.yaml"
    ;;
  preprocess)
    CMD="python -m gwen_rl.scripts.build_manifest --csv data/metadata.csv --audio_dir data/audio --out_dir data/processed"
    ;;
  *)
    echo "Unknown gate: $GATE. Use gate1|gate2|gate3|preprocess"
    exit 1
    ;;
esac

docker run --gpus all --rm -it \
  -v "${DATA_DIR}:/workspace/data" \
  -v "${CKPT_DIR}:/workspace/checkpoints" \
  -v "${CACHE_DIR}:/workspace/model_cache" \
  -v "$(pwd)/logs:/workspace/logs" \
  --shm-size=2g \
  gwen-rl:latest \
  $CMD
