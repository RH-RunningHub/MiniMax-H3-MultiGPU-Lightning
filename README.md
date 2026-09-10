# RunningHub H3 Lightning

**Multi-GPU inference acceleration for MiniMax H3 · 5-second video generation from 348.8 s to 28.7 s**

**English** | [简体中文](./README_CN.md)

[![RunningHub China](https://img.shields.io/badge/RunningHub-China-2F80ED)](https://www.runninghub.cn/?inviteCode=rh-v1367)
[![RunningHub Global](https://img.shields.io/badge/RunningHub-Global-7B61FF)](https://www.runninghub.ai/?inviteCode=rh-v1367)
[![License](https://img.shields.io/badge/Code-Apache%202.0-green)](./LICENSE)

H3 Lightning is RunningHub's inference acceleration recipe for MiniMax H3. In a comparison on **8× NVIDIA RTX 6000D**, generating a 5-second video took **28.7 seconds**, down from **348.8 seconds**: approximately **12.2× faster, with 91.8% lower generation latency**.

This repository publishes the acceleration approach, a pinned SGLang source snapshot, and instructions covering installation, model downloads, video generation and measurement so developers can deploy and evaluate the recipe on their own multi-GPU systems.

## Performance results

### 5-second text-to-video: eight-GPU comparison

| Configuration | Requested steps | Generation latency | Speedup over baseline |
|---|---:|---:|---:|
| MiniMax H3 BF16 baseline | 50 | 348.8 s | 1.0× |
| With RH post-trained acceleration model | 4 | 43.0 s | 8.1× |
| **RH acceleration model + SageAttention2 + Cache-DiT + torch.compile** | **4** | **28.7 s** | **12.2×** |

Test conditions: 8× RTX 6000D, 5 seconds, 1344×768, text-to-video (`t2va`), measured by RunningHub. These are generation latencies after warmup; model loading, initial compilation, queueing and downloads are excluded. **The 12.2× result combines step distillation and execution optimizations, including a change in generation step count.**

Speedup is `348.8 / 28.7 ≈ 12.15`; latency reduction is `1 − 28.7 / 348.8 ≈ 91.77%`. Five seconds is the requested duration; H3 aligns the delivered frame count to its temporal buckets.

### 15-second video: text and two reference images

| Task | Resolution | GPUs / parallelism | Requested steps | Generation latency |
|---|---|---|---:|---:|
| Text-to-video `t2va` | 768×1344 | 8 / TP2+Ulysses4 | 4 | 48.2 s |
| Two references `ref2va` | 768×1344 | 8 / TP2+Ulysses4 | 4 | 73.0 s |
| Text-to-video `t2va`, high motion | 768×1344 | 8 / TP2+Ulysses4 | 8 | 89.3 s |
| Two references `ref2va`, high motion | 768×1344 | 8 / TP2+Ulysses4 | 8 | 134.3 s |

A separate eight-GPU parallelism comparison reduced 15-second text-to-video latency from 54.0 s with TP4+Ulysses2 to 48.2 s with TP2+Ulysses4: approximately 12% higher speed and 14 GiB less GPU memory usage. This result is independent of the 12.2× comparison above.

These measurements describe specific test configurations. Public LoRAs, prompts and reference images in the instructions below affect both speed and output; retain generated videos and the full configuration when comparing runs.

## Acceleration approach and release scope

| Layer | Method | Purpose |
|---|---|---|
| Less generation work | Post-training / step distillation | Generate video and audio with fewer steps |
| Faster execution | SageAttention2, Cache-DiT, torch.compile | Optimize attention, reuse parts of the computation, and compile execution graphs |
| Multi-GPU execution | TP2+Ulysses4 | Combine tensor and sequence parallelism on PCIe systems without NVLink |

RunningHub's work covers component integration, parameter selection, parallelism configuration and workload validation. The runtime is **SGLang `multimodal_gen`**. Its source is bundled at `f8cbf000f4a5bfd86d3fb7c1e2d6c8fb12339d0e`; see the [pin record](./sglang/RH-PIN.md).

The backbone uses **BF16 weights**, without INT8/NVFP4 weight quantization; components such as the VAEs retain their upstream precision settings. SageAttention uses quantized attention internally, while distillation and Cache-DiT also change the computation. BF16 weights therefore do not imply identical arithmetic or lossless output. Evaluate subject consistency, detail, motion continuity, prompt adherence and audio/video synchronization.

**Weight availability:** RH's in-house acceleration weights are not currently public. Developers can substitute community acceleration LoRAs to reproduce the complete inference and acceleration workflow. The instructions below pin and demonstrate two public LoRAs. Evaluate speed and quality for each set of weights; the 12.2× result above was measured with RH weights.

## Requirements

| Item | Configuration used here |
|---|---|
| OS / Python | Ubuntu 22.04 / Python 3.10 |
| GPUs | 8× RTX 6000D, PCIe, no NVLink; approximately 85 GB per GPU in the measured environment |
| Driver / toolkit | NVIDIA 580+; a CUDA Toolkit matching PyTorch's CUDA version; this snapshot uses CUDA 13 dependencies |
| Key Python dependencies | PyTorch 2.13.0, Diffusers 0.37.0, Cache-DiT 1.3.0; full declarations in [pyproject.toml](./sglang/python/pyproject.toml) |
| System tools | `nvcc`, a C++ compiler, `ffmpeg`, `ffprobe` |
| Disk | Approximately 354 GB for the base model, plus space for LoRAs, compilation caches and generated videos |

### Measured hardware parameters (from `nvidia-smi`)

The following are measured records from the validation environment (Ubuntu 22.04.5, driver 580.95.05 / CUDA 13.0) — not an official spec sheet; values may vary slightly across batches/drivers, so save your own `nvidia-smi -q` output before deploying:

| Item | Measured value |
|---|---|
| GPU model | NVIDIA RTX 6000D (Blackwell, `sm_120`) |
| Per-GPU memory (`nvidia-smi` usable) | 85,651 MiB |
| GPU power limit | 600 W (default 600 W, adjustable minimum 200 W) |
| Interface / interconnect | PCIe Gen5 x16, no NVLink |
| Max clocks | SM 2430 MHz / memory 12,481 MHz |
| Inter-GPU topology | 8 GPUs across two NUMA domains (peer GPUs `NODE`, cross-domain `SYS`) |
| CPU | 2× AMD EPYC 9354 (32 cores, 128 threads total) |
| Host memory | 1007 GiB |

### Measured software versions

| Component | Version |
|---|---|
| SGLang | bundled snapshot `f8cbf000f4a5` (see [pin notes](./sglang/RH-PIN.md)) |
| PyTorch | 2.13.0+cu130 |
| Diffusers | 0.37.0 |
| Cache-DiT | 1.3.0 |
| SageAttention | 2.2.0 |
| FlashInfer | 0.6.18 |
| Triton | 3.7.1 |
| Transformers | 5.12.1 |

Existing eight-GPU TP2+U4 measurements report approximately 60–66 GiB peak memory per GPU; the 4-GPU reference environment recorded: BF16 50-step 57.1 GiB/card, INT8-ConvRot 36.8 GiB/card, NVFP4 33.2 GiB/card, and the final serving configuration (turbo LoRA + SageAttention2 + Cache-DiT + torch.compile) 37.3 GiB/card. Usage varies with task, shape and compilation state; validate capacity and performance separately on other hardware.

## 1. Install the inference environment

Use the commands below as the reproduction entry point. Run as a regular user except for the `sudo` system-package commands, and keep the same Bash session for sections 1–3. Install a CUDA Toolkit first and point `CUDA_HOME` to the directory containing `bin/nvcc`; Python packages do not replace the local compiler toolchain.

```bash
sudo apt-get update
sudo apt-get install -y git curl ffmpeg build-essential python3.10 python3.10-dev python3.10-venv

git clone https://github.com/RH-RunningHub/MiniMax-H3-MultiGPU-Lightning.git
cd MiniMax-H3-MultiGPU-Lightning
export REPO_ROOT="$PWD"
export H3_HOME="$HOME/h3-lightning"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"

test -x "$CUDA_HOME/bin/nvcc" || { echo "Set CUDA_HOME to your CUDA 13 toolkit directory"; exit 1; }
mkdir -p "$H3_HOME" "$H3_HOME/models-lora" "$H3_HOME/outputs" "$H3_HOME/runs"
python3.10 -m venv "$H3_HOME/venv"
source "$H3_HOME/venv/bin/activate"

python -m pip install --upgrade pip wheel packaging ninja \
  "setuptools>=77" "setuptools-rust>=1.11" "setuptools-scm>=8"
python -m pip install "torch==2.13.0"
SGLANG_BUILD_RUST_EXTS=none python -m pip install --no-build-isolation \
  -e "$REPO_ROOT/sglang/python[diffusion]"

git clone https://github.com/thu-ml/SageAttention.git "$H3_HOME/SageAttention"
git -C "$H3_HOME/SageAttention" checkout d9704247a5139ab4c03bf7fc6b35cc0e2cbb5ea4
MAX_JOBS=8 python -m pip install --no-build-isolation "$H3_HOME/SageAttention"
python -m pip check
```

`python[diffusion]` installs video-generation dependencies. `SGLANG_BUILD_RUST_EXTS=none` skips Rust extensions not needed here. SageAttention is pinned to the [compatibility commit](https://github.com/thu-ml/SageAttention/tree/d9704247a5139ab4c03bf7fc6b35cc0e2cbb5ea4) referenced by the bundled backend.

Check the actual CUDA operator and save environment records:

```bash
python - <<'PY'
import torch, diffusers, cache_dit
from sageattention import sageattn

assert torch.cuda.is_available(), "CUDA is not available"
assert torch.cuda.device_count() == 8, "Expose exactly 8 GPUs for this example"
print("torch:", torch.__version__, "CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))
q = torch.randn(1, 128, 4, 128, device="cuda", dtype=torch.bfloat16)
out = sageattn(q, q, q, tensor_layout="NHD", is_causal=False)
torch.cuda.synchronize()
assert out.shape == q.shape and torch.isfinite(out).all().item()
print("SageAttention operator check: OK")
PY

python -m pip freeze --all > "$H3_HOME/environment.freeze.txt"
git -C "$REPO_ROOT" rev-parse HEAD > "$H3_HOME/repository-commit.txt"
nvidia-smi > "$H3_HOME/nvidia-smi.txt"
nvidia-smi topo -m > "$H3_HOME/gpu-topology.txt"
"$CUDA_HOME/bin/nvcc" --version > "$H3_HOME/nvcc-version.txt"
```

Retain these records and the source checkout. `environment.freeze.txt` records the dependency resolution for your installation; it is not a lock file from the historical benchmark. Reuse your validated environment records for subsequent deployments.

## 2. Download the base model and a public acceleration LoRA

Use [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) with [larryvrh's v4 EMA LoRA](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora) for the text/keyframe example. The revisions below fix the public download contents; the corresponding partitions and filenames have been checked.

```bash
export H3_MODEL_REV=42ed227ee7df40d41602854ae760620d6eb651fe
export H3_LORA_REV=43a74557ac3f6539db8e0f2a959d03feb7a81480

hf download MiniMaxAI/MiniMax-H3 \
  --revision "$H3_MODEL_REV" --local-dir "$H3_HOME/models/MiniMax-H3"
hf download larryvrh/MiniMax-H3-Turbo-Lora \
  minimax_h3_turbo_v4_step600_ema.safetensors \
  --revision "$H3_LORA_REV" --local-dir "$H3_HOME/models-lora"

printf 'base=%s\nlora=%s\n' "$H3_MODEL_REV" "$H3_LORA_REV" > "$H3_HOME/model-revisions.txt"
sha256sum "$H3_HOME/models-lora/minimax_h3_turbo_v4_step600_ema.safetensors" \
  > "$H3_HOME/lora.sha256"
```

Set `HF_ENDPOINT` if your network requires a Hugging Face mirror. Set `HF_HUB_DISABLE_XET=1` when that mirror does not support Xet. Use the Hugging Face client and retain model revisions to avoid mixing components from different snapshots.

## 3. Start the eight-GPU service

Create a reusable launcher with the public LoRA, Cache-DiT, torch.compile and TP2+Ulysses4 enabled by default. Compilation caches live under `H3_HOME` and can be reused with compatible software, hardware and shapes; new shapes or dependency changes may trigger compilation again.

```bash
cat > "$H3_HOME/serve-h3.sh" <<'BASH'
#!/usr/bin/env bash
set -euo pipefail
H3_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export SGLANG_CACHE_DIT_ENABLED="${SGLANG_CACHE_DIT_ENABLED:-true}"
export TORCHINDUCTOR_CACHE_DIR="$H3_HOME/compile-cache/torchinductor"
export TRITON_CACHE_DIR="$H3_HOME/compile-cache/triton"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$H3_HOME/outputs"

args=(serve --model-path "$H3_HOME/models/MiniMax-H3"
  --model-variant "${MODEL_VARIANT:-fl2va}"
  --num-gpus 8 --tp-size 2 --ulysses-degree 4
  --attention-backend "${ATTENTION_BACKEND:-sage_attn}"
  --performance-mode speed --host 127.0.0.1 --port "${PORT:-30010}"
  --output-path "$H3_HOME/outputs")
if [[ "${ENABLE_COMPILE:-1}" == 1 ]]; then
  args+=(--enable-torch-compile)
else
  args+=(--enable-torch-compile false)
fi
if [[ "${USE_LORA:-1}" == 1 ]]; then
  args+=(--lora-path "$H3_HOME/models-lora"
    --lora-weight-name "${LORA_FILE:-minimax_h3_turbo_v4_step600_ema.safetensors}"
    --lora-nickname turbo --lora-scale 1.0 --lora-merge-mode auto)
fi
exec "$H3_HOME/venv/bin/sglang" "${args[@]}"
BASH
chmod +x "$H3_HOME/serve-h3.sh"
"$H3_HOME/serve-h3.sh" 2>&1 | tee "$H3_HOME/server.log"
```

Initial requests may spend several minutes compiling. Check the logs for successful model/LoRA loading and actual SageAttention selection. After the health check succeeds, warm up your common durations and landscape/portrait shapes. The service listens on loopback; remote access should use an authenticated gateway with model-management routes restricted.

## 4. Submit, measure and download a video

In a second terminal, restore the environment and create a five-second landscape request with four inference steps after the service is ready:

```bash
export H3_HOME="$HOME/h3-lightning"
source "$H3_HOME/venv/bin/activate"
curl --fail --show-error http://127.0.0.1:30010/health

cat > "$H3_HOME/request.json" <<'JSON'
{
  "model": "MiniMaxAI/MiniMax-H3",
  "task": "t2va",
  "prompt": "A dancer performs a flowing routine in a bright studio, full body shot, smooth camera movement.",
  "conditions": [],
  "target": {"short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 5.0},
  "num_inference_steps": 4,
  "flow_shift": 12.0,
  "audio_flow_shift": 3.0,
  "seed": 20260904
}
JSON
```

For a 15-second portrait video, set `target.duration_seconds` to `15.0` and `target.aspect_ratio` to `"9:16"`. Start with 4 steps; try 8 for fast or large motions and follow the selected LoRA's recommendations. Set duration through `target.duration_seconds`; do not also pass `fps` or `num_frames`. H3 uses fixed 24 fps and `17n+5` frame buckets: 5 seconds aligns to 124 frames, and 15 seconds to 362 frames.

This client submits a job, polls with a deadline, checks HTTP errors and validates that the downloaded file contains both video and audio. Each run saves its request, final response, metrics and ffprobe output:

```bash
cat > "$H3_HOME/generate.py" <<'PY'
import json, os, shutil, subprocess, sys, time, uuid
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

base = os.environ.get("BASE", "http://127.0.0.1:30010").rstrip("/")
payload = json.loads(Path(sys.argv[1]).read_text())
run_dir = Path(__file__).resolve().parent / "runs" / uuid.uuid4().hex
run_dir.mkdir(parents=True)
(run_dir / "request.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))

def api(path, data=None):
    body = None if data is None else json.dumps(data).encode()
    request = Request(base + path, data=body, headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=600 if data is not None else 30) as response:
        return json.load(response)

started = time.monotonic()
job = api("/v1/videos", payload)  # Do not retry POST: a timeout may still have created a job.
job_id = job["id"]
deadline = started + 1800
while True:
    (run_dir / "job.json").write_text(json.dumps(job, ensure_ascii=False, indent=2))
    if job["status"] == "completed":
        break
    if job["status"] in {"failed", "error", "cancelled", "deleted"}:
        raise RuntimeError(f"Job {job_id}: {job}")
    if time.monotonic() >= deadline:
        raise TimeoutError(f"Job {job_id}; inspect {base}/v1/videos/{job_id}")
    time.sleep(2)
    try:
        job = api(f"/v1/videos/{job_id}")
    except HTTPError as error:
        if error.code not in {429, 500, 502, 503, 504}:
            raise
    except URLError:
        pass  # GET failures retry within the overall deadline.

metrics = {"inference_time_s": job.get("inference_time_s"),
           "peak_memory_mb": job.get("peak_memory_mb"),
           "submit_to_completed_s": time.monotonic() - started}
(run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
partial = run_dir / "video.partial.mp4"
try:
    with urlopen(base + f"/v1/videos/{job_id}/content", timeout=120) as response:
        with partial.open("wb") as output:
            shutil.copyfileobj(response, output)
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-show_format",
                            "-of", "json", str(partial)], check=True, capture_output=True, text=True)
    media = json.loads(probe.stdout)
    kinds = {stream.get("codec_type") for stream in media["streams"]}
    if not {"video", "audio"} <= kinds:
        raise RuntimeError("Expected both video and audio streams")
    (run_dir / "ffprobe.json").write_text(probe.stdout)
    partial.replace(run_dir / "video.mp4")
finally:
    partial.unlink(missing_ok=True)
print(json.dumps({"id": job_id, "output": str(run_dir), **metrics}, indent=2))
PY
python "$H3_HOME/generate.py" "$H3_HOME/request.json"
```

The video is saved to `$H3_HOME/runs/<run-id>/video.mp4`. `inference_time_s` is the server-reported generation time; `submit_to_completed_s` includes submission, queueing and polling delay. Report them separately. `peak_memory_mb` is a server-reported metric; a missing value is not zero. Collect GPU-level monitoring separately for per-device peak memory.

## 5. Generate from reference images

The `fl2va` partition serves `t2va` and keyframe tasks. `ref2va` requires its own partition and matching LoRA; a loaded instance cannot switch partitions per request. Stop the original service and release all eight GPUs before starting this reference-image example:

```bash
hf download lightx2v/Minimax-h3-Turbo \
  minimax_h3_ref2v_turbo_8step_v1.0_768p_bf16.safetensors \
  --revision 2f015e66b37c585cea9dc4ae6f1850ea8788e742 \
  --local-dir "$H3_HOME/models-lora"

MODEL_VARIANT=ref2va \
LORA_FILE=minimax_h3_ref2v_turbo_8step_v1.0_768p_bf16.safetensors \
  "$H3_HOME/serve-h3.sh" 2>&1 | tee "$H3_HOME/server-ref2va.log"
```

Save the following as `$H3_HOME/request-ref2va.json`, replace both URIs with **absolute paths to existing reference images on the server**, then run `python "$H3_HOME/generate.py" "$H3_HOME/request-ref2va.json"`. This public example uses an eight-step Ref2V LoRA; its configuration differs from the four-step internal measurement above.

```json
{
  "model": "MiniMaxAI/MiniMax-H3",
  "task": "ref2va",
  "prompt": "The character in reference image 1 walks through the setting in reference image 2, cinematic tracking shot.",
  "conditions": [
    {"type": "image", "uri": "file:///absolute/server/path/reference-1.png", "role": "reference"},
    {"type": "image", "uri": "file:///absolute/server/path/reference-2.png", "role": "reference"}
  ],
  "target": {"short_edge": 768, "aspect_ratio": "9:16", "duration_seconds": 15.0},
  "num_inference_steps": 8,
  "flow_shift": 12.0,
  "audio_flow_shift": 3.0,
  "seed": 20260904
}
```

For first/last-frame generation, use `task: "fl2va"`, `role: "keyframe"`, and `frame_index: 0` for the first frame or `-1` for the last. Set the target `aspect_ratio` to `"auto"` so geometry is derived from keyframes. See the [MiniMax H3 model documentation](https://huggingface.co/MiniMaxAI/MiniMax-H3) for native input specifications.

## 6. Measure on your hardware

1. Fix the base/LoRA revisions, prompt, seed, duration, resolution and reference assets. The prompt above is a runnable example, not the historical benchmark prompt.
2. Start each configuration separately and wait for loading and compilation to finish. Warm up the same shape, then run at least five measurements; retain individual results, generated videos and the median.
3. Save launch environment variables, server logs, request JSON, model hashes, dependency records and GPU topology. Calculate speedup using the same timing definition for both configurations.
4. Compare detail, subjects, motion, audio and synchronization before accepting a faster configuration for your workload.

For example, after stopping the accelerated service, launch the following BF16/Flash Attention baseline without LoRA, Cache-DiT or compile, and set `num_inference_steps` to 50 in the request. Compare against the default launcher at your LoRA's recommended step count. This defines a local public-LoRA comparison; it is not automatically the same experiment as the RH-weight measurements above.

```bash
USE_LORA=0 ENABLE_COMPILE=0 SGLANG_CACHE_DIT_ENABLED=false ATTENTION_BACKEND=fa \
  "$H3_HOME/serve-h3.sh" 2>&1 | tee "$H3_HOME/server-baseline.log"
```

To measure the LoRA's contribution alone, change `USE_LORA=0` above to `USE_LORA=1` and use four request steps. For the full configuration, use the default launcher from section 3 with four steps. Stop the previous instance before each configuration change and keep the inputs identical.

Keep the actual attention backend reported in the server logs and check for dependency-related fallbacks. If `nvcc` is missing, check `CUDA_HOME`; if `cache_dit` is missing, check the diffusion extra; if a task is rejected for its partition, check `MODEL_VARIANT`. Separate loading and compilation time from steady generation latency.

## License and acknowledgements

Repository code is licensed under [Apache 2.0](./LICENSE). MiniMax H3 base weights follow their [Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3). Refer to each LoRA repository for its usage terms.

Thanks to [MiniMax](https://huggingface.co/MiniMaxAI/MiniMax-H3), [SGLang](https://github.com/sgl-project/sglang), [SageAttention](https://github.com/thu-ml/SageAttention), [Cache-DiT](https://github.com/vipshop/cache-dit), [larryvrh](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora) and [LightX2V](https://huggingface.co/lightx2v/Minimax-h3-Turbo) for their open-source work.

Try RunningHub: [China](https://www.runninghub.cn/?inviteCode=rh-v1367) · [Global](https://www.runninghub.ai/?inviteCode=rh-v1367). For reproduction issues, include hardware, version records, the launch command, a sanitized request and error logs.
