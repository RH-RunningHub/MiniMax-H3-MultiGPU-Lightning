# Public Turbo v4 on eight RTX PRO 5000 72GB GPUs

Measured by wuyaole on 2026-09-11; retained metrics, request, server log and media-probe records were read back on 2026-09-14. This is one completed post-warmup T2VA request using the public LoRA, not an RH-weight speedup comparison or a latency distribution. No quality non-regression claim is made.

## Configuration

| Item | Recorded value |
|---|---|
| Repository | `5d1401228680ca635dac463825f71c3fbd681ef7` |
| Hardware | 8 x NVIDIA RTX PRO 5000 72GB Blackwell |
| Topology | Two NUMA groups; within-group NODE, cross-group SYS; no NVLink |
| OS / Python | Ubuntu 22.04 / Python 3.10 |
| Driver / toolkit | 580.126.20 / CUDA 13.0, nvcc 13.0.88 |
| Runtime | Bundled SGLang source; torch 2.13.0; Diffusers 0.37.0; Cache-DiT 1.3.0; FlashInfer 0.6.18; Triton 3.7.1; Transformers 5.12.1 |
| SageAttention | `d9704247a5139ab4c03bf7fc6b35cc0e2cbb5ea4` |
| Base / variant | MiniMaxAI/MiniMax-H3 / fl2va, used for T2VA with no conditions |
| Public LoRA | larryvrh/MiniMax-H3-Turbo-Lora, `minimax_h3_turbo_v4_step600_ema.safetensors`, scale 1.0 |
| LoRA SHA256 | `5f3a626cd72c93a8b9318d6760c510bc5092d2ab13aaba1f932c5bab07a416d3` |
| Execution | TP2 + Ulysses4, sage_attn, speed mode, Cache-DiT, torch.compile |
| Request | 4 steps, video flow shift 12, audio flow shift 3, seed 1101 |

## Recorded result

- API `inference_time_s`: **135.6366 s**.
- Client submit-to-completed: **138.2960 s**, including client polling.
- Server stage log: denoising **105.7153 s**, decoding **23.8297 s**. These are stage timers, not interchangeable with the API total.
- Output: H.264 1344 x 768, 24 fps, **362 frames / 15.083333 s**; AAC 32 kHz stereo, 15.075 s. The request asked for 15 seconds; frame alignment means the file is longer than exactly 15.000 seconds.
- API-reported peak memory: **62246 MB**. Separately sampled device peaks were **63911–66475 MiB** across GPUs. These scopes differ: use the device observations for capacity planning rather than treating the API field as a fleet-wide peak.

Machine-readable [request](request.json) and [selected result fields](result.json) are included. The full video and private machine paths are not included. Model startup and initial compilation are excluded; repeat representative requests before drawing throughput or quality conclusions.

## Reproduction

Follow the root README installation and model-download instructions. Use the pinned repository and SageAttention revisions above and verify the LoRA hash. The retained environment was a Python 3.10 installation with the listed dependencies; this is not a lockfile guarantee for a fresh installation.

With the README's `H3_HOME`, model and LoRA directories configured, start a dedicated server on available GPUs:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export SGLANG_CACHE_DIT_ENABLED=true
export TORCHINDUCTOR_CACHE_DIR="$H3_HOME/compile-cache/torchinductor"
export TRITON_CACHE_DIR="$H3_HOME/compile-cache/triton"

"$H3_HOME/venv/bin/sglang" serve \
  --model-path "$H3_HOME/models/MiniMax-H3" --model-variant fl2va \
  --num-gpus 8 --tp-size 2 --ulysses-degree 4 \
  --attention-backend sage_attn --performance-mode speed \
  --host 127.0.0.1 --port 30010 --output-path "$H3_HOME/outputs" \
  --enable-torch-compile --lora-path "$H3_HOME/models-lora" \
  --lora-weight-name minimax_h3_turbo_v4_step600_ema.safetensors \
  --lora-nickname turbo --lora-scale 1.0 --lora-merge-mode auto
```

Wait for startup/warmup to finish, then submit `request.json` to `POST /v1/videos` using the root README's asynchronous generation workflow. Record the completed job's `inference_time_s`, client elapsed time and a media probe separately. The historical server log confirmed adapter loading, distributed Cache-DiT and `max-autotune-no-cudagraphs` compilation.
