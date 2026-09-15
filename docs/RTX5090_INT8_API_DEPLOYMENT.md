# MiniMax-H3 INT8 API deployment on 8× RTX 5090

This deployment profile serves MiniMax-H3 FL2VA through SGLang's asynchronous
`/v1/videos` API. It is the exact 15-second configuration validated on eight
GeForce RTX 5090 GPUs:

- pruned ConvRot INT8 DiT;
- NVFP4-AWQ Qwen3-VL text encoder;
- FP16 video VAE and FP32 audio VAE;
- TP1 + Ulysses8;
- comfy-kitchen `ck_int8_attn` for the DiT;
- `torch.compile` enabled;
- 20 full denoising forwards, Cache-DiT disabled and no LoRA.

The detailed measurements and diagnosis are in
[the performance report](./rtx5090-int8-20step-performance-report.md).

## Validated environment

| Component | Version |
|---|---|
| GPU | 8× NVIDIA GeForce RTX 5090, 32,607 MiB/GPU |
| Driver / CUDA toolkit | 595.91.07 / CUDA 13.0 |
| Python / PyTorch | 3.10.12 / 2.13.0+cu130 |
| Diffusers / Transformers | 0.37.0 / 5.12.1 |
| Cache-DiT / comfy-kitchen | 1.3.0 / 0.2.33 |

Other Blackwell systems may work, but re-run warmup and peak-memory validation
before placing the service under load.

## Quick start

Install CUDA 13 first, then clone the repository to the path used by the
systemd template:

```bash
git clone https://github.com/RH-RunningHub/MiniMax-H3-MultiGPU-Lightning.git \
  /opt/MiniMax-H3-MultiGPU-Lightning
cd /opt/MiniMax-H3-MultiGPU-Lightning

CUDA_HOME=/usr/local/cuda-13.0 \
INSTALL_ROOT=/data/sglang-h3 \
  sudo -E bash scripts/install-5090-int8.sh

# HF_TOKEN may be required by the model repositories.
HF_TOKEN="$HF_TOKEN" INSTALL_ROOT=/data/sglang-h3 \
  bash scripts/download-5090-int8.sh
```

Start the recommended profile in the foreground:

```bash
INSTALL_ROOT=/data/sglang-h3 \
PLACEMENT_PROFILE=balanced \
HOST=0.0.0.0 PORT=30010 \
  scripts/start-5090-int8.sh
```

From another shell:

```bash
BASE=http://127.0.0.1:30010 scripts/api-example-5090-int8.sh
```

Do not expose the model-management API directly to the public internet. Put it
behind an authenticated gateway and restrict model loading, memory-management
and local-file input routes.

## Required model files

The download script pins these repository revisions:

```text
MiniMaxAI/MiniMax-H3       42ed227ee7df40d41602854ae760620d6eb651fe
Comfy-Org/MiniMax-H3       a98869194787969724c7425d95d0ed73ce9202af
```

It verifies the two quantized weight files:

```text
e889202c41dafb67b10d67b97f0d8541508036a6090af23425a5c2615d03c47a  minimax_h3_fl2va_pruned_int8_convrot.safetensors
35a88d51044231fe332301d7a62aa81e3f2cba62febeb446e2c1e3e0ef76f2c6  qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors
```

Weights are never stored in Git. Override `MODEL_PATH`, `QUANT_ROOT`,
`DIT_WEIGHTS` or `TE_WEIGHTS` when cluster storage uses different mount paths.

## Placement profiles

`PLACEMENT_PROFILE` selects one of four measured strategies. Times are steady
15-second, 1344×768, 362-frame, 20-step requests.

| Profile | Placement | Time | `nvidia-smi` peak | Idle VRAM | Pinned DiT host memory |
|---|---|---:|---:|---:|---:|
| `balanced` | TE component-offload; VAE resident; DiT layerwise | **113.38 s** | **26.06 GiB** | 8.56 GiB | 19.55 GiB/GPU |
| `low-host-memory` | VAE resident; TE ↔ complete DiT | 115.82 s | 30.62 GiB | 6.69 GiB | ≈0 |
| `low-gpu-memory` | TE → complete DiT → VAE | 116.92 s | 26.52 GiB | **1.13 GiB** | ≈0 |
| `fastest-risky` | TE/VAE resident; DiT layerwise | **111.24 s** | 30.81 GiB steady; 31.88 GiB first shape | 23.58 GiB | 19.55 GiB/GPU |

Use `balanced` by default. `fastest-risky` had only 725 MiB headroom during the
first new-shape request and is unsuitable for concurrency or mixed shapes.
The complete-DiT profiles use `--warmup-mode off`: their first request performs
compilation and can take substantially longer than steady state.

“Complete DiT resident” means resident only during the denoising stage. A TP1
DiT (19.61 GiB) and TE (14.73 GiB) cannot remain on the same 32 GiB GPU at the
same time, so the complete-DiT profiles swap those components at every request.

## API lifecycle

Submit a job:

```bash
curl --fail --show-error -X POST http://127.0.0.1:30010/v1/videos \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"MiniMaxAI/MiniMax-H3",
    "task":"t2va",
    "prompt":"A cinematic tracking shot through a rainy neon street.",
    "conditions":[],
    "target":{"short_edge":768,"aspect_ratio":"16:9","duration_seconds":15.0},
    "num_inference_steps":20,
    "flow_shift":12.0,
    "audio_flow_shift":3.0,
    "seed":42
  }'
```

Poll `GET /v1/videos/<id>` until `status` is `completed`, then download
`GET /v1/videos/<id>/content`. POST requests should not be retried blindly: a
client timeout does not prove the server failed to create the job.

## systemd

Copy and edit the environment file, then install the unit:

```bash
sudo cp deploy/rh-h3-5090.env.example /etc/rh-h3-5090.env
sudo cp deploy/rh-h3-5090.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now rh-h3-5090
sudo journalctl -u rh-h3-5090 -f
```

The unit assumes the repository is at
`/opt/MiniMax-H3-MultiGPU-Lightning`. Change `WorkingDirectory` and `ExecStart`
when using a different path.

## Runtime fixes included by this repository

- Registers the comfy-kitchen INT8 attention backend and keeps causal paths on
  SageAttention/SDPA.
- Prevents the quantized TP text encoder from taking a numerically incorrect
  row-parallel path.
- Prevents compile warmup from silently disabling an explicitly requested DiT
  layerwise placement and materializing the full DiT.
- Marks text encoding as memory-intensive so component offload releases its
  CUDA allocator reserve before denoising. Without this, TE tensors are gone
  but `nvidia-smi` can still report approximately 15–19 GiB of stale reserve.

`peak_memory_mb` and `nvidia-smi` primarily reflect allocator reserve, not just
live tensors. During the diagnosed 15-second layerwise request, live allocation
at the DiT block boundary was about 7.1 GiB while stale TE reserve raised the
old reading to about 27.2 GiB. The included cache-release fix reduced the DiT
stage to about 12.8 GiB without measurable latency regression.

## Operations checklist

1. Keep compile caches on local persistent storage and do not share one cache
   directory across incompatible GPU/PyTorch builds.
2. Warm every production duration, orientation and resolution before admitting
   traffic when using `balanced` or `fastest-risky`.
3. Treat a new shape as a cold request and monitor all eight GPUs.
4. Run one worker group per eight exclusive GPUs; do not oversubscribe the
   `fastest-risky` profile.
5. Preserve the launch environment, model revisions, server log and request
   JSON for incident reproduction.
6. Use graceful process termination and verify all eight GPU processes exit
   before restarting with a different profile.
