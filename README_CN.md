# RunningHub H3 Lightning

**MiniMax H3 多卡推理加速方案 · 5 秒视频生成耗时从 348.8 秒降至 28.7 秒**

[English](./README.md) | **简体中文**

[![RunningHub China](https://img.shields.io/badge/RunningHub-China-2F80ED)](https://www.runninghub.cn/?inviteCode=rh-v1367)
[![RunningHub Global](https://img.shields.io/badge/RunningHub-Global-7B61FF)](https://www.runninghub.ai/?inviteCode=rh-v1367)
[![License](https://img.shields.io/badge/Code-Apache%202.0-green)](./LICENSE)

H3 Lightning 是 RunningHub 面向 MiniMax H3 的推理加速方案。在 **8× NVIDIA RTX 6000D** 的对照测试中，生成一段 5 秒视频的耗时从 **348.8 秒降至 28.7 秒**，约 **12.2 倍加速、91.8% 耗时降低**。

本仓库公开加速技术路线、固定版本的 SGLang 推理源码，以及从环境安装、权重下载到视频生成和性能记录的操作步骤，帮助开发者在自己的多卡环境中部署和评估 H3 Lightning。

## 性能结果

### 5 秒文生视频：8 卡对照

| 配置 | 请求步数 | 生成耗时 | 相对基线 |
|---|---:|---:|---:|
| MiniMax H3 BF16 基础方案 | 50 | 348.8 秒 | 1.0× |
| 加入 RH 后训练加速模型 | 4 | 43.0 秒 | 8.1× |
| **RH 加速模型 + SageAttention2 + Cache-DiT + torch.compile** | **4** | **28.7 秒** | **12.2×** |

测试口径：8× RTX 6000D，5 秒、1344×768、文生视频（`t2va`），数据由 RunningHub 实测提供。耗时为服务预热后的生成耗时，不包含模型加载、首次编译、排队和文件下载。**12.2× 是步数蒸馏与执行优化的整体收益，包含生成步数变化。**

加速比为 `348.8 / 28.7 ≈ 12.15`；耗时降低比例为 `1 − 28.7 / 348.8 ≈ 91.77%`。这里的“5 秒”指请求时长；实际帧数按 H3 的时间桶对齐。

### 15 秒视频：文字与双参考图

| 任务 | 分辨率 | GPU / 并行 | 请求步数 | 生成耗时 |
|---|---|---|---:|---:|
| 文生视频 `t2va` | 768×1344 | 8 卡 / TP2+Ulysses4 | 4 | 48.2 秒 |
| 双参考图 `ref2va` | 768×1344 | 8 卡 / TP2+Ulysses4 | 4 | 73.0 秒 |
| 文生视频 `t2va`，高动态档 | 768×1344 | 8 卡 / TP2+Ulysses4 | 8 | 89.3 秒 |
| 双参考图 `ref2va`，高动态档 | 768×1344 | 8 卡 / TP2+Ulysses4 | 8 | 134.3 秒 |

在独立的 8 卡并行对照中，15 秒文生视频由 TP4+Ulysses2 的 54.0 秒降至 TP2+Ulysses4 的 48.2 秒，速度约提高 12%，显存占用约降低 14 GiB。这项收益不与上面的 12.2× 相乘。

以上是特定测试配置的测量结果。下方示例中的公开 LoRA、提示词和参考图会影响耗时与画面，复测时应同时记录生成结果与完整配置。

## 加速方法与开源范围

| 层次 | 方法 | 作用 |
|---|---|---|
| 减少生成计算 | 后训练加速 / 步数蒸馏 | 用更少的生成步骤完成视频与音频生成 |
| 提高执行效率 | SageAttention2、Cache-DiT、torch.compile | 优化注意力计算，复用部分计算结果，编译执行图 |
| 优化多卡协作 | TP2+Ulysses4 | 在 PCIe、无 NVLink 环境下组合张量并行与序列并行 |

RunningHub 的工作覆盖加速组件集成、参数选择、多卡配置和任务验证；运行时基于 **SGLang `multimodal_gen`**。完整 SGLang 源码已内嵌于本仓库，版本为 `f8cbf000f4a5bfd86d3fb7c1e2d6c8fb12339d0e`，见 [版本记录](./sglang/RH-PIN.md)。

本方案的主干网络使用 **BF16 权重**，没有使用 INT8/NVFP4 权重量化；VAE 等组件沿用上游的精度配置。SageAttention 内部采用量化注意力计算，蒸馏与 Cache-DiT 也会改变计算路径，因此 BF16 权重本身不代表逐算子等价或画质无损。实际验收应覆盖主体一致性、细节、运动连贯性、提示词遵循及音画同步。

**权重发布状态：**RH 自训加速权重暂未公开。开发者可使用社区加速 LoRA 替代，配合本仓库复现完整推理与加速流程。下方给出两种公开 LoRA 的固定版本下载和使用示例；不同权重的速度与画质需分别评测，上述 12.2× 数据对应 RH 权重。

## 环境要求

| 项目 | 本文配置 |
|---|---|
| 系统 / Python | Ubuntu 22.04 / Python 3.10 |
| GPU | 8× RTX 6000D，PCIe 互联，无 NVLink；实测环境单卡约 85 GB 显存 |
| 驱动 / 工具链 | NVIDIA 580+；安装与 PyTorch CUDA 版本匹配的 CUDA Toolkit，本快照使用 CUDA 13 系列依赖 |
| 关键 Python 依赖 | PyTorch 2.13.0、Diffusers 0.37.0、Cache-DiT 1.3.0，完整声明见 [pyproject.toml](./sglang/python/pyproject.toml) |
| 系统工具 | `nvcc`、C++ 编译器、`ffmpeg`、`ffprobe` |
| 磁盘 | 基座模型约 354 GB，另外预留 LoRA、编译缓存及输出视频空间 |

8 卡 TP2+U4 配置的已有峰值显存记录约为 60–66 GiB/卡，实际值随任务、shape 与编译状态变化。其他硬件应单独验证容量和性能。

## 1. 安装推理环境

下面的命令是本文的复现入口。除 `sudo` 系统依赖安装外，以普通用户执行；在同一 Bash 会话中完成第 1–3 节。提前安装 CUDA Toolkit，并将 `CUDA_HOME` 指向包含 `bin/nvcc` 的目录；下载 Python 包不会替代本机编译工具链。

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

`python[diffusion]` 提供视频生成依赖；`SGLANG_BUILD_RUST_EXTS=none` 跳过此流程不需要的 Rust 扩展构建。SageAttention 固定到内嵌 SGLang backend 指向的 [兼容提交](https://github.com/thu-ml/SageAttention/tree/d9704247a5139ab4c03bf7fc6b35cc0e2cbb5ea4)。

安装后验证实际 CUDA 算子，并保存环境记录：

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

保留上述记录和源码 checkout。`environment.freeze.txt` 用于记录本次解析后的依赖；它不等同于历史性能测试的环境锁文件。重复部署时应沿用已验收的记录。

## 2. 下载基座与公开加速 LoRA

基座使用 [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3)，文生/首尾帧示例使用 [larryvrh 的 v4 EMA LoRA](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora)。以下 revision 固定公开下载内容，并已核对相应模型分区和文件名。

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

网络需要 Hugging Face 镜像时可配置 `HF_ENDPOINT`；若所用镜像不支持 Xet，再设置 `HF_HUB_DISABLE_XET=1`。使用 Hugging Face 客户端下载并保留 revision，避免混用不同版本的模型组件。

## 3. 启动 8 卡服务

下面生成可复用的启动文件，默认加载公开 LoRA、开启 Cache-DiT 和 torch.compile，使用 TP2+Ulysses4。编译缓存保存在 `H3_HOME`，可在相同软件、硬件及 shape 条件下复用；新 shape 或依赖变更仍可能触发编译。

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

首次请求可能需要数分钟编译。检查日志确认模型和 LoRA 加载成功、实际使用 SageAttention；完成健康检查后，先预热常用时长与横竖屏 shape。服务默认只监听本机；跨机访问应通过带认证的网关，并限制模型管理接口。

## 4. 提交任务、保存指标并下载视频

在第二个终端设置环境，待服务就绪后创建一个 5 秒横屏、4 步推理请求：

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

生成 15 秒竖屏视频时，将 `target.duration_seconds` 改为 `15.0`，`target.aspect_ratio` 改为 `"9:16"`。默认从 4 步开始；快速运动或大幅动作可尝试 8 步，并使用对应 LoRA 的推荐档位。`target.duration_seconds` 是时长入口，不要同时传入 `fps` 或 `num_frames`。固定 24 fps，时间桶为 `17n+5`：5 秒对齐为 124 帧，15 秒对齐为 362 帧。

以下客户端完成提交、带截止时间的轮询、HTTP 错误检查和音视频流校验。每次运行都保存请求、最终任务响应、性能指标和 ffprobe 结果：

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

输出位于 `$H3_HOME/runs/<run-id>/video.mp4`。`inference_time_s` 是服务端报告的生成时间，`submit_to_completed_s` 包含提交、排队和轮询等待，二者应分开统计；`peak_memory_mb` 是服务端返回的指标，缺失时不能当作 0。记录每卡显存峰值时需额外采集各 GPU 的监控数据。

## 5. 使用参考图生成

`fl2va` 模型分区处理 `t2va` 和首尾帧任务；`ref2va` 使用独立分区及匹配 LoRA。两种分区不能由同一已加载实例随请求切换。以下示例在停止原服务、释放 8 张 GPU 后启动参考图实例：

```bash
hf download lightx2v/Minimax-h3-Turbo \
  minimax_h3_ref2v_turbo_8step_v1.0_768p_bf16.safetensors \
  --revision 2f015e66b37c585cea9dc4ae6f1850ea8788e742 \
  --local-dir "$H3_HOME/models-lora"

MODEL_VARIANT=ref2va \
LORA_FILE=minimax_h3_ref2v_turbo_8step_v1.0_768p_bf16.safetensors \
  "$H3_HOME/serve-h3.sh" 2>&1 | tee "$H3_HOME/server-ref2va.log"
```

将下列请求保存为 `$H3_HOME/request-ref2va.json`，把两个 URI 替换为**服务器上实际存在的参考图绝对路径**，然后运行 `python "$H3_HOME/generate.py" "$H3_HOME/request-ref2va.json"`。该公开示例使用 8 步 Ref2V LoRA，与上表 4 步内部测试配置不同。

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

首尾帧任务使用 `task: "fl2va"`，条件设置 `role: "keyframe"`，第一帧 `frame_index: 0`、最后一帧 `frame_index: -1`；目标画幅使用 `aspect_ratio: "auto"`，由关键帧解析。更多原生输入说明见 [MiniMax H3 模型文档](https://huggingface.co/MiniMaxAI/MiniMax-H3)。

## 6. 在自己的硬件上复测

1. 固定基座和 LoRA revision、prompt、seed、时长、分辨率以及参考素材。上面的 prompt 是可执行示例，不是历史测试素材。
2. 分别启动每档配置，等待模型加载和编译完成；每档对同一 shape 先预热，再至少运行 5 次，保留所有单次结果、生成视频与中位数。
3. 同时保存启动环境变量、服务日志、请求 JSON、模型哈希、依赖记录及 GPU 拓扑。使用相同的时间口径计算加速比。
4. 对照检查画面细节、主体与运动、音频及同步，确认减少步数和缓存符合业务质量要求后再采用该档配置。

例如，停止加速服务后，可用以下命令启动无 LoRA、无 Cache-DiT、无 compile 的 BF16/Flash Attention 基础配置，将请求中的 `num_inference_steps` 改为 50；再切回默认启动命令和 LoRA 推荐步数进行对比。这定义的是本机公开 LoRA 对照实验，不能仅凭配置名称等同于上表的 RH 权重实测。

```bash
USE_LORA=0 ENABLE_COMPILE=0 SGLANG_CACHE_DIT_ENABLED=false ATTENTION_BACKEND=fa \
  "$H3_HOME/serve-h3.sh" 2>&1 | tee "$H3_HOME/server-baseline.log"
```

要单独测量 LoRA 的加速收益，将上述 `USE_LORA=0` 改为 `USE_LORA=1`，并把请求步数改为 4；要测量全部优化，则使用第 3 节的默认启动命令和 4 步请求。每次切换配置前停止上一实例，并使用同一组输入。

保留服务日志中的实际 attention backend，确认没有发生缺少依赖后的回退。常见问题：`nvcc` 找不到时检查 `CUDA_HOME`；缺少 `cache_dit` 时核对是否安装了 diffusion extra；请求分区不匹配时检查服务的 `MODEL_VARIANT`；首次运行较慢时先区分模型加载、编译与稳定生成耗时。

## License 与致谢

本仓库代码使用 [Apache 2.0](./LICENSE)。MiniMax H3 基座权重遵循其 [Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3)，公开 LoRA 的使用条件以各模型仓库为准。

感谢 [MiniMax](https://huggingface.co/MiniMaxAI/MiniMax-H3)、[SGLang](https://github.com/sgl-project/sglang)、[SageAttention](https://github.com/thu-ml/SageAttention)、[Cache-DiT](https://github.com/vipshop/cache-dit)、[larryvrh](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora) 和 [LightX2V](https://huggingface.co/lightx2v/Minimax-h3-Turbo) 的开源工作。

体验 RunningHub：[中国站](https://www.runninghub.cn/?inviteCode=rh-v1367) · [国际站](https://www.runninghub.ai/?inviteCode=rh-v1367)。提交复现问题时，请附硬件、版本记录、启动命令、去除敏感信息后的请求和错误日志。
