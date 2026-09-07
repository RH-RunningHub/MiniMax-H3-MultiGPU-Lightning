#!/bin/bash
# MiniMax-H3 serving instance — fl2va variant (text-to-video / keyframes)
# Mirrors the validated 8x RTX 6000D deployment: TP2 + Ulysses4, sage_attn, torch.compile, turbo LoRA.
#
# For a ref2va instance (reference images/video/audio):
#   1. change --model-variant to ref2va
#   2. mount a ref2v LoRA, e.g. minimax_h3_ref2v_turbo_8step_v1.0_bf16.safetensors
# fl2va and ref2va variants are mutually exclusive: run one instance per variant.

set -e
INSTALL_ROOT=${INSTALL_ROOT:-/data/sglang-h3}
MODEL_PATH=${MODEL_PATH:-/data/models/MiniMax-H3}
PORT=${PORT:-30010}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
# Cache-DiT (server default settings; quality-validated)
export SGLANG_CACHE_DIT_ENABLED=true
# Persist torch.compile / triton caches so restarts do not recompile
export TORCHINDUCTOR_CACHE_DIR=$INSTALL_ROOT/compile-cache/torchinductor
export TRITON_CACHE_DIR=$INSTALL_ROOT/compile-cache/triton

exec "$INSTALL_ROOT/venv/bin/sglang" serve \
  --model-path "$MODEL_PATH" \
  --model-variant fl2va \
  --num-gpus 8 --tp-size 2 --ulysses-degree 4 \
  --attention-backend sage_attn \
  --enable-torch-compile \
  --performance-mode speed --host 0.0.0.0 --port "$PORT" \
  --lora-path "$INSTALL_ROOT/models-lora" \
  --lora-weight-name minimax_h3_turbo_v4_step600_ema.safetensors \
  --lora-nickname turbo --lora-scale 1.0 --lora-merge-mode auto
