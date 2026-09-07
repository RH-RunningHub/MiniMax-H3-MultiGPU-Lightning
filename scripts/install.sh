#!/bin/bash
# MiniMax-H3 multi-GPU inference acceleration — one-shot install
# Tested on: Ubuntu 22.04, Python 3.10, 8x NVIDIA RTX 6000D (sm_120), driver 580+
set -e

INSTALL_ROOT=${INSTALL_ROOT:-/data/sglang-h3}
PINNED_SGLANG=f8cbf000f4a5bfd86d3fb7c1e2d6c8fb12339d0e

echo "== [1/5] system deps (ffmpeg is mandatory for MiniMax-H3) =="
apt-get update
apt-get install -y ffmpeg git python3.10 python3.10-venv

echo "== [2/5] python venv =="
python3.10 -m venv "$INSTALL_ROOT/venv"
# shellcheck disable=SC1091
source "$INSTALL_ROOT/venv/bin/activate"
pip install -U pip wheel packaging ninja

echo "== [3/5] sglang @ $PINNED_SGLANG (bundled in this repo) =="
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SGLANG_SRC="$SCRIPT_DIR/../sglang"
if [ ! -d "$SGLANG_SRC/python" ]; then
  echo "bundled sglang/ not found in this checkout; falling back to upstream clone"
  if [ ! -d "$INSTALL_ROOT/sglang" ]; then
    git clone https://github.com/sgl-project/sglang.git "$INSTALL_ROOT/sglang"
  fi
  cd "$INSTALL_ROOT/sglang"
  git checkout "$PINNED_SGLANG"
  SGLANG_SRC="$INSTALL_ROOT/sglang"
fi
cd "$SGLANG_SRC"
SGLANG_BUILD_RUST_EXTS=no pip install --no-build-isolation -e python
python -c "import sglang; print('sglang import OK')"

echo "== [4/5] SageAttention2 (needs nvcc/CUDA toolkit, e.g. CUDA_HOME=/usr/local/cuda) =="
if [ -z "$CUDA_HOME" ]; then
  echo "WARN: CUDA_HOME not set; install CUDA toolkit first, then re-run this step:"
  echo "  CUDA_HOME=/usr/local/cuda pip install --no-build-isolation ./SageAttention"
else
  git clone https://github.com/thu-ml/SageAttention.git /tmp/SageAttention || true
  (cd /tmp/SageAttention && CUDA_HOME="$CUDA_HOME" pip install --no-build-isolation .)
fi

echo "== [5/5] directories for LoRA & persistent compile cache =="
mkdir -p "$INSTALL_ROOT/models-lora"
mkdir -p "$INSTALL_ROOT/compile-cache/torchinductor" "$INSTALL_ROOT/compile-cache/triton"

echo "INSTALL-DONE"
echo "next: download models (see README), then start via scripts/start.sh + scripts/rh-h3.service"
