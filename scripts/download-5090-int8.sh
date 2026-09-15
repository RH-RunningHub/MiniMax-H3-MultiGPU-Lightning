#!/usr/bin/env bash
# Download only the components required by the validated FL2VA INT8 service.
set -euo pipefail

INSTALL_ROOT="${INSTALL_ROOT:-/data/sglang-h3}"
MODEL_PATH="${MODEL_PATH:-$INSTALL_ROOT/models/MiniMax-H3}"
QUANT_ROOT="${QUANT_ROOT:-$INSTALL_ROOT/models/minimax-h3-comfy}"
BASE_REVISION="${BASE_REVISION:-42ed227ee7df40d41602854ae760620d6eb651fe}"
COMFY_REVISION="${COMFY_REVISION:-a98869194787969724c7425d95d0ed73ce9202af}"

# shellcheck disable=SC1091
source "$INSTALL_ROOT/venv/bin/activate"
command -v hf >/dev/null || python -m pip install 'huggingface-hub[cli]==1.31.0'
export HF_HOME="${HF_HOME:-$INSTALL_ROOT/hf-cache}"

hf download MiniMaxAI/MiniMax-H3 \
  --revision "$BASE_REVISION" --local-dir "$MODEL_PATH" \
  --include 'FL2VA/video_vae/**' \
  --include 'FL2VA/audio_vae/**' \
  --include 'FL2VA/tokenizer/**' \
  --include 'FL2VA/processor/**' \
  --include 'FL2VA/model_index.json' \
  --include 'FL2VA/text_encoder/*.json' \
  --include 'FL2VA/transformer/*.json' \
  --include 'model_index.json'

hf download Comfy-Org/MiniMax-H3 \
  --revision "$COMFY_REVISION" --local-dir "$QUANT_ROOT" \
  --include 'diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors' \
  --include 'text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors'

DIT="$QUANT_ROOT/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors"
TE="$QUANT_ROOT/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
echo 'e889202c41dafb67b10d67b97f0d8541508036a6090af23425a5c2615d03c47a' " $DIT" | sha256sum -c -
echo '35a88d51044231fe332301d7a62aa81e3f2cba62febeb446e2c1e3e0ef76f2c6' " $TE" | sha256sum -c -

printf 'base=%s\ncomfy=%s\n' "$BASE_REVISION" "$COMFY_REVISION" \
  > "$INSTALL_ROOT/model-revisions-5090-int8.txt"
du -sh "$MODEL_PATH" "$QUANT_ROOT"
echo "Download complete. Start with scripts/start-5090-int8.sh."
