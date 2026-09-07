# MiniMax-H3 Multi-GPU Inference Acceleration

[![RunningHub China](https://img.shields.io/badge/RunningHub-China-2F80ED)](https://www.runninghub.cn/?inviteCode=rh-v1367)
[![RunningHub International](https://img.shields.io/badge/RunningHub-International-7B61FF)](https://www.runninghub.ai/?inviteCode=rh-v1367)
[![English](https://img.shields.io/badge/Language-English-2563EB)](./README.md)
[![简体中文](https://img.shields.io/badge/Language-%E7%AE%80%E4%BD%93%E4%B8%AD%E6%96%87-EF4444)](./README_CN.md)

![License](https://img.shields.io/badge/License-Apache%202.0-green)

A complete recipe that speeds up MiniMax-H3 video generation by ~12× on 8× RTX 6000D: RH post-training acceleration model (step distillation) + SageAttention2 + Cache-DiT + torch.compile, served by the sglang `multimodal_gen` engine with TP2+Ulysses4 multi-GPU parallelism. **The pinned sglang source is bundled directly in this repo under `sglang/`** — clone this repo and you have the whole solution; no separate upstream clone needed. The goal is that teams with 8× RTX 6000D (or similar cards) can reproduce it end to end.

---

## Introduction

Turning text or images into video takes many iterative compute rounds — more steps, longer waits. In an A/B benchmark on 5-second video generation, we cut MiniMax-H3 generation time from **348.8 s to 28.7 s**, roughly **12×** faster than the baseline.

Faster feedback, faster iteration — that is what drove this work.

### The recipe and measured numbers

#### 5-second t2va A/B (1344×768 · 4× RTX 6000D · TP2+Ulysses2)

All measured configurations, sorted by latency (descending):

| Approach | Steps | Latency | vs. baseline* | Notes |
|---|---:|---:|---:|---|
| BF16 baseline (original weights) | 50 | 348.8 s | 1× | speedup baseline |
| INT8-ConvRot quantization | 50 | 316.2 s | 1.10× | |
| NVFP4 quantization | 50 | 283.8 s | 1.23× | failed quality acceptance, dropped |
| BF16 + turbo LoRA (step distillation) | 9 | 60.0 s | 5.8× | open-source LoRAs reach the same tier |
| BF16 + turbo LoRA + Cache-DiT | 9 | 57.3 s | 6.1× | Cache-DiT alone adds 4.5% |
| PulpCut fused INT8+turbo + flashinfer RoPE/LN | 8 | 48.8 s | 7.1× | on par with PulpCut, no gain, not adopted |
| PulpCut fused INT8+turbo | 8 | 48.7 s | 7.2× | validated on fl2va only, review quality yourself |
| BF16 + turbo LoRA + SageAttention2 | 9 | 36.5 s | 9.6× | |
| BF16 + turbo LoRA + SageAttention2 + Cache-DiT | 9 | 33.1 s | 10.5× | |
| **BF16 + turbo LoRA + SageAttention2 + Cache-DiT + torch.compile (final serving config)** | 9 | **28.7 s** | **12.2×** | |

\* Baseline = original BF16 weights at 50 steps (348.8 s). The A/B group ran on a single 4-GPU instance (TP2+U2) of an 8-GPU machine, while the other half served ref2va at the time. All numbers are post-warmup net latencies, excluding queueing and model loading.

**Do less computation (step distillation), then make the remaining computation faster (kernel & compile optimizations).** The overall speedup combines fewer steps and faster steps; per-component gains cannot be separated.

#### 8-GPU comparison (15s · 768×1344, production parallel TP2+U4)

For multi-GPU parallelism on PCIe-only machines (no NVLink), we benchmarked split strategies and settled on **TP2 + Ulysses4** on 8 cards: ~**12%** faster than TP4 + Ulysses2 while saving ~**14 GiB** of GPU memory (independent comparison, not multiplied with the 12×).

Net latency, sorted descending; no same-condition baseline, so no multipliers:

| Generation mode | Parallel / GPUs | Steps | Latency | Notes |
|---|---|---:|---:|---|
| Two reference images (ref2va) | TP2+U4 · 8× | 8 | 134.3 s | 8-step tier for high motion |
| Two reference images (ref2va, landscape 1344×768) | TP2+U4 · 8× | 4 | 93.4 s | |
| Text-to-video (t2va) | TP2+U4 · 8× | 8 | 89.3 s | 8-step tier for high motion |
| Two reference images (ref2va) | TP2+U4 · 8× | 4 | 73.0 s | |
| Text-to-video (t2va, TP4 split A/B) | TP4+U2 · 8× | 4 | 54.0 s | 12% slower than TP2+U4, saves ~14GiB |
| Text-to-video (t2va, production shape) | TP2+U4 · 8× | 4 | 48.2 s | ~40% faster than the 4-GPU setup |

\* The 15s group is measured on single 8-GPU instances (post-warmup net latency); t2va and ref2va were validated on two identical 8-GPU machines.

**Step count guidance: 4 steps by default; 8 steps for fast motion / large-amplitude actions**, trading speed for fidelity.

## 🛠️ Installation

### Requirements

- Ubuntu 22.04, Python 3.10
- NVIDIA driver 580+; **ffmpeg / ffprobe are mandatory** on the host (H3 hard-validates them at startup)
- Tested hardware: 8× NVIDIA RTX 6000D (sm_120, PCIe, no NVLink), 85GB each
- sglang: upstream main `f8cbf000f4a5` (2026-09-02), **source bundled in this repo under `sglang/`** (pin notes in `sglang/RH-PIN.md`); the install script uses the bundled copy by default

### One-shot install

```bash
bash scripts/install.sh
```

The script (expand to run manually):

```bash
#!/bin/bash
set -e

# 1. System deps
apt-get update && apt-get install -y ffmpeg git python3.10 python3.10-venv

# 2. Virtual env
python3.10 -m venv /data/sglang-h3/venv
source /data/sglang-h3/venv/bin/activate
pip install -U pip wheel

# 3. sglang (bundled in this repo at sglang/, = upstream main @ f8cbf000f4a5)
cd <this repo>/sglang
SGLANG_BUILD_RUST_EXTS=no pip install --no-build-isolation -e python

# 4. SageAttention2 (needs nvcc/CUDA toolkit to build)
pip install packaging ninja
git clone https://github.com/thu-ml/SageAttention.git /tmp/SageAttention
cd /tmp/SageAttention && CUDA_HOME=/usr/local/cuda pip install --no-build-isolation .

echo "INSTALL-DONE"
```

### Start script & systemd

`scripts/start.sh` launches the serving instance:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export SGLANG_CACHE_DIT_ENABLED=true

exec /data/sglang-h3/venv/bin/sglang serve \
  --model-path /data/models/MiniMax-H3 \
  --model-variant fl2va \
  --num-gpus 8 --tp-size 2 --ulysses-degree 4 \
  --attention-backend sage_attn \
  --enable-torch-compile \
  --performance-mode speed --host 0.0.0.0 --port 30010 \
  --lora-path /data/sglang-h3/models-lora \
  --lora-weight-name minimax_h3_turbo_v4_step600_ema.safetensors \
  --lora-nickname turbo --lora-scale 1.0 --lora-merge-mode auto
```

Key points:

- `--model-variant`: `fl2va` (text / keyframes) and `ref2va` (reference images / video / audio) are **mutually exclusive — run one instance per variant**. For a ref2va instance, switch the variant and mount a ref2v LoRA; everything else stays the same.
- `--tp-size 2 --ulysses-degree 4`: best combo measured on PCIe-only boxes (12% faster than TP4+U2, saves 14GiB).
- Manage with `scripts/rh-h3.service` (systemd, `Restart=on-failure`) and persist the compile caches:

```ini
[Service]
Environment=TORCHINDUCTOR_CACHE_DIR=/data/sglang-h3/compile-cache/torchinductor
Environment=TRITON_CACHE_DIR=/data/sglang-h3/compile-cache/triton
```

- The first large-shape request after (re)start pays 1–3 minutes of torch.compile; with the cache persisted, restarts do not recompile. After the first successful run, warm up your common shapes (e.g. 15s / 9:16 and 16:9).

## 📦 Model Download & Installation

### Base model (required)

[MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) (~354GB, BF16, ships FL2VA and Ref2VA component sets):

```bash
pip install -U "huggingface_hub[cli]"
HF_HUB_DISABLE_XET=1 hf download MiniMaxAI/MiniMax-H3 --local-dir /data/models/MiniMax-H3
```

> Keep `HF_HUB_DISABLE_XET=1`; the Xet channel 401-loops against mirrors. Avoid multi-connection downloaders (they corrupt xet-backed files).

### Acceleration LoRAs (open-source substitutes, ~80% of the effect)

> **About the RH post-trained acceleration model**: RunningHub's in-house acceleration model is **not yet available for download** — compatibility work is still ongoing. In the meantime, open-source acceleration LoRAs are a drop-in substitute that reaches roughly **80%** of the effect; the rest of the pipeline stays identical.

| Model | Link | Notes |
|---|---|---|
| larryvrh acceleration LoRA | [larryvrh/MiniMax-H3-Turbo-Lora](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora) | `minimax_h3_turbo_v4_step600_ema.safetensors` (743MB, works at 4–9 steps; **recommended in sglang's official cookbook**, used in our 5.8× benchmark) |
| lightx2v acceleration LoRA | [lightx2v/Minimax-h3-Turbo](https://huggingface.co/lightx2v/Minimax-h3-Turbo) | fl2v / ref2v, 4-step and 8-step variants (`minimax_h3_fl2v_turbo_8step_v1.0_bf16.safetensors` etc.), task-specific distillation |

```bash
hf download larryvrh/MiniMax-H3-Turbo-Lora --local-dir /data/sglang-h3/models-lora
```

### Layout & memory reference

```text
/data/sglang-h3/
├── venv/                 # Python environment
├── models-lora/          # acceleration LoRA safetensors
└── compile-cache/        # torch.compile / triton caches (persisted)
```

- Measured peak memory on 8× TP2+U4: **60–66 GiB per card** (BF16 + turbo LoRA + compile).
- For 8-GPU boxes with <80GB cards, validate with lower resolution/duration first, or evaluate quantization (validate image quality yourself).

## 🚀 Usage

OpenAI-style video API (full example in `scripts/api_example.sh`):

```bash
curl -s -X POST http://127.0.0.1:30010/v1/videos -H 'Content-Type: application/json' -d '{
  "model": "MiniMaxAI/MiniMax-H3",
  "prompt": "a dancer performing on a live stream",
  "seconds": 15,
  "task": "t2va",
  "conditions": [],
  "target": {"short_edge": 768, "aspect_ratio": "9:16", "duration_seconds": 15.0},
  "num_inference_steps": 4,
  "flow_shift": 12.0, "audio_flow_shift": 3.0, "seed": 20260904
}'
# Poll GET /v1/videos/{id}, download via GET /v1/videos/{id}/content
```

- `task`: `t2va` (text) / `fl2va` (keyframes, conditions carry `role=keyframe`) / `ref2va` (references, conditions carry `role=reference`).
- Durations snap up to the 17n+5 frame lattice at fixed 24fps: 15s→362 frames, 8s→192 frames exact.
- ref2va requests must go to a `--model-variant ref2va` instance.

## 📄 License

- Code in this repo: Apache 2.0
- The MiniMax-H3 model weights follow the [MiniMax-H3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3)
- Referenced acceleration LoRAs follow their respective licenses

## 🔗 Links

[![RunningHub China](https://img.shields.io/badge/RunningHub-China-2F80ED)](https://www.runninghub.cn/?inviteCode=rh-v1367)
[![RunningHub International](https://img.shields.io/badge/RunningHub-International-7B61FF)](https://www.runninghub.ai/?inviteCode=rh-v1367)

- [sgl-project/sglang](https://github.com/sgl-project/sglang) (pinned at `f8cbf000f4a5` for this solution)
- [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3)
- [larryvrh/MiniMax-H3-Turbo-Lora](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora) / [lightx2v/Minimax-h3-Turbo](https://huggingface.co/lightx2v/Minimax-h3-Turbo)
- [SageAttention](https://github.com/thu-ml/SageAttention)

## 🙏 Acknowledgements

- [sgl-project/sglang](https://github.com/sgl-project/sglang) — multimodal_gen engine and multi-GPU parallelism
- [MiniMax](https://huggingface.co/MiniMaxAI) — the open-source MiniMax-H3 model
- [lightx2v](https://huggingface.co/lightx2v) / [larryvrh](https://huggingface.co/larryvrh) — open-source acceleration LoRAs
- [thu-ml/SageAttention](https://github.com/thu-ml/SageAttention) — attention kernel acceleration

---

## Closing notes

This recipe boils down to two moves: **do less computation, then make the remaining computation faster.** Step distillation decides "how much"; SageAttention2, Cache-DiT and torch.compile decide "how fast"; TP2+Ulysses4 decides how PCIe-only cards cooperate. The pieces are independent — adopt them step by step based on your hardware and quality bar.

We will keep refining the reproduction notes toward "checkable configs with matching results". Numbers vary across GPUs, drivers and model versions — trust your own A/B tests. Issues and ideas are welcome.

**Faster feedback for creators, clearer methods for developers.**

---

*Test notes: data from technical tests in September 2026. The ~12× speedup refers to a 4× RTX 6000D, 5-second, 1344×768 text-to-video benchmark against a BF16 50-step baseline and includes the step-count change; the 15-second portrait numbers are 8× RTX 6000D at 4 steps. All latencies are post-warmup generation times and exclude queueing; results vary with task and settings.*
