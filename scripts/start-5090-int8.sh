#!/usr/bin/env bash
# MiniMax-H3 INT8 API service for 8x RTX 5090.
# Default profile: TP1 + Ulysses8, layerwise DiT, component-offloaded TE,
# resident FP16 video VAE / FP32 audio VAE, torch.compile, no Cache-DiT/LoRA.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
INSTALL_ROOT="${INSTALL_ROOT:-/data/sglang-h3}"
MODEL_PATH="${MODEL_PATH:-$INSTALL_ROOT/models/MiniMax-H3}"
QUANT_ROOT="${QUANT_ROOT:-$INSTALL_ROOT/models/minimax-h3-comfy}"
DIT_WEIGHTS="${DIT_WEIGHTS:-$QUANT_ROOT/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors}"
TE_WEIGHTS="${TE_WEIGHTS:-$QUANT_ROOT/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors}"
SGLANG_BIN="${SGLANG_BIN:-$INSTALL_ROOT/venv/bin/sglang}"
PROFILE="${PLACEMENT_PROFILE:-balanced}"
NUM_GPUS="${NUM_GPUS:-8}"
TP_SIZE="${TP_SIZE:-1}"
ULYSSES_DEGREE="${ULYSSES_DEGREE:-8}"

for path in "$SGLANG_BIN" "$MODEL_PATH/FL2VA/model_index.json" "$DIT_WEIGHTS" "$TE_WEIGHTS"; do
  if [[ ! -e "$path" ]]; then
    echo "required path not found: $path" >&2
    exit 1
  fi
done
if (( TP_SIZE * ULYSSES_DEGREE != NUM_GPUS )); then
  echo "TP_SIZE * ULYSSES_DEGREE must equal NUM_GPUS" >&2
  exit 1
fi

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.0}"
export PATH="$CUDA_HOME/bin:$PATH"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export SGLANG_CACHE_DIT_ENABLED="${SGLANG_CACHE_DIT_ENABLED:-false}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$INSTALL_ROOT/compile-cache/torchinductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$INSTALL_ROOT/compile-cache/triton}"
OUTPUT_PATH="${OUTPUT_PATH:-$INSTALL_ROOT/outputs}"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$OUTPUT_PATH"

args=(serve
  --model-path "$MODEL_PATH"
  --model-variant "${MODEL_VARIANT:-fl2va}"
  --num-gpus "$NUM_GPUS"
  --tp-size "$TP_SIZE"
  --ulysses-degree "$ULYSSES_DEGREE"
  --attention-backend "${ATTENTION_BACKEND:-sage_attn}"
  --component-attention-backends.transformer="${DIT_ATTENTION_BACKEND:-ck_int8_attn}"
  --component-weights-paths.transformer="$DIT_WEIGHTS"
  --component-weights-paths.text_encoder="$TE_WEIGHTS"
  --component-precisions.video_vae=fp16
  --component-precisions.audio_vae=fp32
  --enable-torch-compile
  --performance-mode speed
  --dit-cpu-offload false
  --image-encoder-cpu-offload false
  --host "${HOST:-0.0.0.0}"
  --port "${PORT:-30010}"
  --output-path "$OUTPUT_PATH")

case "$PROFILE" in
  balanced)
    # Recommended: 113.38 s steady 15 s generation, 26.06 GiB request peak.
    args+=(--component-residency text_encoder=component-offload
      --dit-layerwise-offload
      --layerwise-offload-components dit
      --dit-offload-prefetch-size 1
      --dit-layerwise-resident-layers 0
      --vae-cpu-offload false
      --warmup-resolutions "${WARMUP_RESOLUTIONS:-1344x768}"
      --warmup-num-frames "${WARMUP_NUM_FRAMES:-362}"
      --warmup-steps "${WARMUP_STEPS:-20}")
    ;;
  low-host-memory)
    # VAE resident; TE and the complete DiT alternate on the GPU.
    # No 19.55 GiB/GPU pinned DiT copy. 115.82 s steady, 30.62 GiB peak.
    args+=(--component-residency
      transformer=component-offload text_encoder=component-offload
      --dit-layerwise-offload false
      --vae-cpu-offload false
      --warmup-mode off)
    ;;
  low-gpu-memory)
    # TE, complete DiT and VAE alternate. Lowest idle VRAM and no pinned DiT.
    # 116.92 s steady, 26.52 GiB peak.
    args+=(--component-residency
      transformer=component-offload text_encoder=component-offload vae=component-offload
      --dit-layerwise-offload false
      --warmup-mode off)
    ;;
  fastest-risky)
    # TE and VAEs resident; only DiT streams layerwise. First-shape peak was
    # 31.88 GiB on a 32,607 MiB card, so use only on exclusive fixed-shape GPUs.
    args+=(--dit-layerwise-offload
      --layerwise-offload-components dit
      --dit-offload-prefetch-size 1
      --dit-layerwise-resident-layers 0
      --vae-cpu-offload false
      --warmup-resolutions "${WARMUP_RESOLUTIONS:-1344x768}"
      --warmup-num-frames "${WARMUP_NUM_FRAMES:-362}"
      --warmup-steps "${WARMUP_STEPS:-20}")
    ;;
  *)
    echo "unknown PLACEMENT_PROFILE=$PROFILE" >&2
    echo "valid profiles: balanced, low-host-memory, low-gpu-memory, fastest-risky" >&2
    exit 2
    ;;
esac

if [[ -n "${EXTRA_SERVE_ARGS:-}" ]]; then
  read -r -a extra_args <<<"$EXTRA_SERVE_ARGS"
  args+=("${extra_args[@]}")
fi

echo "Starting MiniMax-H3 profile=$PROFILE TP=$TP_SIZE Ulysses=$ULYSSES_DEGREE"
printf 'Command:'
printf ' %q' "$SGLANG_BIN" "${args[@]}"
printf '\n'
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  exit 0
fi
exec "$SGLANG_BIN" "${args[@]}"
