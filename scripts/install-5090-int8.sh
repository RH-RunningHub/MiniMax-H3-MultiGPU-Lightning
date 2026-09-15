#!/usr/bin/env bash
# Install the validated RTX 5090 INT8 serving environment.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
INSTALL_ROOT="${INSTALL_ROOT:-/data/sglang-h3}"
PYTHON="${PYTHON:-python3.10}"
TORCH_VERSION="${TORCH_VERSION:-2.13.0}"
SAGE_COMMIT="${SAGE_COMMIT:-d9704247a5139ab4c03bf7fc6b35cc0e2cbb5ea4}"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.0}"
export PATH="$CUDA_HOME/bin:$PATH"
if [[ ! -x "$CUDA_HOME/bin/nvcc" ]]; then
  echo "nvcc not found under CUDA_HOME=$CUDA_HOME" >&2
  exit 1
fi

apt-get update
apt-get install -y git curl ffmpeg build-essential "$PYTHON" "$PYTHON-dev" "$PYTHON-venv"
mkdir -p "$INSTALL_ROOT"
"$PYTHON" -m venv "$INSTALL_ROOT/venv"
# shellcheck disable=SC1091
source "$INSTALL_ROOT/venv/bin/activate"
python -m pip install --upgrade pip wheel packaging ninja \
  'setuptools>=77' 'setuptools-rust>=1.11' 'setuptools-scm>=8'
python -m pip install "torch==$TORCH_VERSION"
SGLANG_BUILD_RUST_EXTS=none python -m pip install --no-build-isolation \
  -e "$REPO_ROOT/sglang/python[diffusion]"
python -m pip install 'comfy-kitchen==0.2.33'

SAGE_DIR="$INSTALL_ROOT/SageAttention"
if [[ ! -d "$SAGE_DIR/.git" ]]; then
  git clone https://github.com/thu-ml/SageAttention.git "$SAGE_DIR"
fi
git -C "$SAGE_DIR" fetch --depth 1 origin "$SAGE_COMMIT"
git -C "$SAGE_DIR" checkout --detach "$SAGE_COMMIT"
MAX_JOBS="${MAX_JOBS:-8}" python -m pip install --no-build-isolation "$SAGE_DIR"

mkdir -p "$INSTALL_ROOT/models" "$INSTALL_ROOT/compile-cache/torchinductor" \
  "$INSTALL_ROOT/compile-cache/triton" "$INSTALL_ROOT/outputs"
python -m pip check
python - <<'PY'
import importlib.metadata as metadata
import torch
import comfy_kitchen

assert torch.cuda.is_available(), "CUDA is unavailable"
assert comfy_kitchen.int8_attention_is_available(), "comfy-kitchen INT8 kernel is unavailable"
print("torch", torch.__version__, "CUDA", torch.version.cuda)
print("comfy-kitchen", metadata.version("comfy-kitchen"))
print("GPU", torch.cuda.get_device_name(0), "count", torch.cuda.device_count())
PY

echo "Install complete. Run scripts/download-5090-int8.sh next."
