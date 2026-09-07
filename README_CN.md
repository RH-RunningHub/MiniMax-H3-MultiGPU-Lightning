# MiniMax-H3 多卡推理加速方案

[![RunningHub China](https://img.shields.io/badge/RunningHub-China-2F80ED)](https://www.runninghub.cn/?inviteCode=rh-v1367)
[![RunningHub International](https://img.shields.io/badge/RunningHub-International-7B61FF)](https://www.runninghub.ai/?inviteCode=rh-v1367)
[![English](https://img.shields.io/badge/Language-English-2563EB)](./README.md)
[![简体中文](https://img.shields.io/badge/Language-%E7%AE%80%E4%BD%93%E4%B8%AD%E6%96%87-EF4444)](./README_CN.md)

![License](https://img.shields.io/badge/License-Apache%202.0-green)

在 8× RTX 6000D 上把 MiniMax-H3 视频生成推理提速约 12 倍的完整方案：RH 后训练加速模型（步数蒸馏）+ SageAttention2 + Cache-DiT + torch.compile，配合 sglang `multimodal_gen` 推理引擎与 TP2+Ulysses4 多卡并行。**sglang 源码（验证过的固定版本）已直接内嵌在本仓库 `sglang/` 目录**，克隆本仓库即可获得整套方案，无需再单独克隆上游。目标是让手里同样有 8 卡 RTX 6000D（或类似卡型）的团队，能把整套方案复现出来。

---

## 前言

AI 把文字或图片变成视频，需要多轮计算逐步生成画面，"生成步数"越多等待越久。我们在一组 5 秒视频生成的对照测试中，将 MiniMax-H3 的生成耗时从 **348.8 秒缩短到 28.7 秒**，整体速度约为基线的 **12 倍**。

更快看到结果，更快调整创意，更快尝试下一版——这是我们优化 H3 推理速度的出发点。

### 加速路线与实测数据

#### 5 秒文生视频对照（1344×768 · t2va · 4× RTX 6000D · TP2+Ulysses2）

全部实测方案，按生成耗时降序：

| 方案 | 步数 | 生成耗时 | 相对基线* | 备注 |
|---|---:|---:|---:|---|
| BF16 基础方案（原版权重） | 50 | 348.8 秒 | 1× | 加速比基线 |
| INT8-ConvRot 量化 | 50 | 316.2 秒 | 1.10× | |
| NVFP4 量化 | 50 | 283.8 秒 | 1.23× | 画质验收不通过，弃用 |
| BF16 + turbo LoRA（步数蒸馏） | 9 | 60.0 秒 | 5.8× | 开源 LoRA 同档可达 |
| BF16 + turbo LoRA + Cache-DiT | 9 | 57.3 秒 | 6.1× | Cache-DiT 单独 +4.5% |
| PulpCut INT8+turbo 预融合 + flashinfer RoPE/LN | 8 | 48.8 秒 | 7.1× | 与 PulpCut 持平，无收益，未采用 |
| PulpCut INT8+turbo 预融合 | 8 | 48.7 秒 | 7.2× | 仅 fl2va 验证过，自行验收画质 |
| BF16 + turbo LoRA + SageAttention2 | 9 | 36.5 秒 | 9.6× | |
| BF16 + turbo LoRA + SageAttention2 + Cache-DiT | 9 | 33.1 秒 | 10.5× | |
| **BF16 + turbo LoRA + SageAttention2 + Cache-DiT + torch.compile（服务端定型配置）** | 9 | **28.7 秒** | **12.2×** | |

\* 以 BF16 原版权重 50 步（348.8 秒）为基线。对照组在 8 卡机的单个 4 卡实例（TP2+U2）上测得，另一半卡当时同时服务 ref2va 实例；数据为服务预热后的净速，不含排队与模型加载。

**减少要做的计算（步数蒸馏），再让剩下的计算执行得更快（算子与编译优化）。**整体加速包含步数变化与计算优化的综合收益，各组件收益不能单独拆分。

#### 8 卡对照（15s · 768×1344，定型并行 TP2+U4）

多卡并行方式上，我们针对 PCIe 互联（无 NVLink）的环境对比了不同切分方式，在 8 卡上定型 **TP2 + Ulysses4**：比 TP4 + Ulysses2 再快约 **12%**，同时省约 **14 GiB** 显存（独立对比数据，不与 12× 叠乘）。

净速，按生成耗时降序；无同条件基线，不列倍数：

| 生成方式 | 并行 / 卡数 | 步数 | 生成耗时 | 备注 |
|---|---|---:|---:|---|
| 双参考图生成（ref2va） | TP2+U4 · 8 卡 | 8 | 134.3 秒 | 高动态 8 步档 |
| 双参考图生成（ref2va，横屏 1344×768） | TP2+U4 · 8 卡 | 4 | 93.4 秒 | |
| 文字生成视频（t2va） | TP2+U4 · 8 卡 | 8 | 89.3 秒 | 高动态 8 步档 |
| 双参考图生成（ref2va） | TP2+U4 · 8 卡 | 4 | 73.0 秒 | |
| 文字生成视频（t2va，TP4 切分对照） | TP4+U2 · 8 卡 | 4 | 54.0 秒 | 比 TP2+U4 慢 12%，省约 14GiB |
| 文字生成视频（t2va，定型形态） | TP2+U4 · 8 卡 | 4 | 48.2 秒 | 比 4 卡部署快约 40% |

\* 15 秒组为 8 卡单实例实测（预热后净速）；t2va 与 ref2va 分别在两台同型 8 卡机上验证。

**步数建议：默认 4 步即可；快速运动、大幅动作等高动态内容推荐 8 步**，在速度与画面表现之间取舍。

## 🛠️ 部署安装

### 环境要求

- Ubuntu 22.04（其他版本自行调整），Python 3.10
- NVIDIA 驱动 580+；宿主机**必须安装 ffmpeg / ffprobe**（H3 启动强校验，缺失会 worker 反复 crash）
- 实测硬件：8× NVIDIA RTX 6000D（sm_120，PCIe 互联，无 NVLink），单卡 85GB 显存
- sglang 版本：上游 main `f8cbf000f4a5`（2026-09-02），**源码已内嵌在本仓库 `sglang/` 目录**（版本说明见 `sglang/RH-PIN.md`），安装脚本默认直接使用内嵌副本

### 一键安装脚本

```bash
bash scripts/install.sh
```

脚本内容（可拆开手动执行）：

```bash
#!/bin/bash
set -e

# 1. 系统依赖
apt-get update && apt-get install -y ffmpeg git python3.10 python3.10-venv

# 2. 虚拟环境
python3.10 -m venv /data/sglang-h3/venv
source /data/sglang-h3/venv/bin/activate
pip install -U pip wheel

# 3. sglang（源码内嵌在本仓库 sglang/ 目录，= 上游 main @ f8cbf000f4a5）
cd <本仓库目录>/sglang
SGLANG_BUILD_RUST_EXTS=no pip install --no-build-isolation -e python

# 4. SageAttention2（需 nvcc/CUDA toolkit 编译）
pip install packaging ninja
git clone https://github.com/thu-ml/SageAttention.git /tmp/SageAttention
cd /tmp/SageAttention && CUDA_HOME=/usr/local/cuda pip install --no-build-isolation .

echo "INSTALL-DONE"
```

### 启动脚本与 systemd

模型与服务参数写在 `scripts/start.sh`（对应 61 机 fl2va 实例）：

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

要点：

- `--model-variant`：`fl2va`（文生/首尾帧）与 `ref2va`（参考图/视频/音频）**互不兼容，需分别起实例**；ref2va 实例把 variant 换成 `ref2va` 并挂 ref2v 系 LoRA 即可，其余参数相同。
- `--tp-size 2 --ulysses-degree 4`：PCIe 机型实测最优组合（比 TP4+U2 快 12%、省 14GiB）。
- 建议按 `scripts/rh-h3.service` 用 systemd 托管（`Restart=on-failure`），并持久化编译缓存：

```ini
[Service]
Environment=TORCHINDUCTOR_CACHE_DIR=/data/sglang-h3/compile-cache/torchinductor
Environment=TRITON_CACHE_DIR=/data/sglang-h3/compile-cache/triton
```

- torch.compile 首个大 shape 请求需付 1–3 分钟编译；持久化缓存后，**重启服务不再重复编译**。首次跑通后建议按实际业务预热常用 shape（如 15s / 9:16 与 16:9）。

## 📦 模型下载与安装

### 基座模型（必下）

[MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3)（约 354GB，BF16，含 FL2VA 与 Ref2VA 两组组件）：

```bash
pip install -U "huggingface_hub[cli]"
# 国内网络可加 export HF_ENDPOINT=https://hf-mirror.com
HF_HUB_DISABLE_XET=1 hf download MiniMaxAI/MiniMax-H3 --local-dir /data/models/MiniMax-H3
```

> 提示：`HF_HUB_DISABLE_XET=1` 必加，否则走 Xet 通道在镜像源上会 401 卡死；不要用 aria2 多连接下载（会拼坏 xet 文件）。

### 加速 LoRA（开源替代，可达约 80% 效果）

> **关于 RH 后训练加速模型**：RunningHub 自训的加速模型目前**兼容性还没有处理完，暂时不开放下载**。在开放之前，可以用市面上已经开源的加速 LoRA 替代，实测能达到约 **80%** 的效果（对照：RH 版 9 步 60.0s，开源 LoRA 同档速度与画质接近，流程完全一致）。

| 模型 | 地址 | 说明 |
|---|---|---|
| larryvrh 加速 LoRA | [larryvrh/MiniMax-H3-Turbo-Lora](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora) | `minimax_h3_turbo_v4_step600_ema.safetensors`（743MB，4–9 步可用；**sglang 官方 cookbook 的推荐档**，本文 5.8× 基准即用它） |
| lightx2v 加速 LoRA | [lightx2v/Minimax-h3-Turbo](https://huggingface.co/lightx2v/Minimax-h3-Turbo) | fl2v / ref2v 各 4 步、8 步档位（`minimax_h3_fl2v_turbo_8step_v1.0_bf16.safetensors` 等），任务针对性蒸馏，速度与质量同档或更好 |

```bash
hf download larryvrh/MiniMax-H3-Turbo-Lora --local-dir /data/sglang-h3/models-lora
```

### 模型放置与显存参考

```text
/data/sglang-h3/
├── venv/                 # Python 环境
├── models-lora/          # 加速 LoRA safetensors
└── compile-cache/        # torch.compile / triton 缓存（持久化）
```

- 8 卡 TP2+U4 实测峰值显存约 **60–66 GiB/卡**（BF16 + turbo LoRA + compile）。
- 单卡显存不足 80GB 的 8 卡机型，请先降低分辨率/时长验证，或评估量化方案（注意自行验收画质）。

## 🚀 使用方法

服务就绪后调用 OpenAI 风格视频接口（完整示例见 `scripts/api_example.sh`）：

```bash
curl -s -X POST http://127.0.0.1:30010/v1/videos -H 'Content-Type: application/json' -d '{
  "model": "MiniMaxAI/MiniMax-H3",
  "prompt": "美女主播在直播中跳舞，跳大摆锤",
  "seconds": 15,
  "task": "t2va",
  "conditions": [],
  "target": {"short_edge": 768, "aspect_ratio": "9:16", "duration_seconds": 15.0},
  "num_inference_steps": 4,
  "flow_shift": 12.0, "audio_flow_shift": 3.0, "seed": 20260904
}'
# 轮询 GET /v1/videos/{id}，完成后 GET /v1/videos/{id}/content 下载
```

- `task`：`t2va`（文生）/ `fl2va`（首尾帧，conditions 带 `role=keyframe`）/ `ref2va`（参考素材，conditions 带 `role=reference`）。
- 时长会向上对齐 17n+5 帧桶（24fps 固定）：15s→362 帧、8s→192 帧正好。
- ref2va 任务必须发往 `--model-variant ref2va` 的实例。

## 📄 License

- 本仓库代码：Apache 2.0
- MiniMax-H3 模型权重遵循 [MiniMax-H3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3)，使用前请阅读其许可条款
- 引用的加速 LoRA 遵循各自仓库的许可

## 🔗 Links

[![RunningHub China](https://img.shields.io/badge/RunningHub-China-2F80ED)](https://www.runninghub.cn/?inviteCode=rh-v1367)
[![RunningHub International](https://img.shields.io/badge/RunningHub-International-7B61FF)](https://www.runninghub.ai/?inviteCode=rh-v1367)

- [sgl-project/sglang](https://github.com/sgl-project/sglang)（本方案固定版本 `f8cbf000f4a5`）
- [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3)
- [larryvrh/MiniMax-H3-Turbo-Lora](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora) / [lightx2v/Minimax-h3-Turbo](https://huggingface.co/lightx2v/Minimax-h3-Turbo)
- [SageAttention](https://github.com/thu-ml/SageAttention)

## 🙏 Acknowledgements

- [sgl-project/sglang](https://github.com/sgl-project/sglang) — multimodal_gen 推理引擎与多卡并行
- [MiniMax](https://huggingface.co/MiniMaxAI) — MiniMax-H3 开源模型
- [lightx2v](https://huggingface.co/lightx2v) / [larryvrh](https://huggingface.co/larryvrh) — 开源加速 LoRA
- [thu-ml/SageAttention](https://github.com/thu-ml/SageAttention) — 注意力算子加速

---

## 后言

这套方案的本质是两句话：**减少要做的计算，再让剩下的计算执行得更快。**步数蒸馏解决"做多少"，SageAttention2、Cache-DiT 与 torch.compile 解决"做多快"，TP2+Ulysses4 解决"多张 PCIe 卡怎么配合"。三者彼此独立，可按自己的硬件与画质要求分步引入。

我们计划持续把复现说明细化到"可检查的配置与对应结果"。不同卡型、驱动与模型版本上数字会有出入，请以自己的对照测试为准；如果复现中遇到问题或有改进思路，欢迎提 Issue 交流。

**让创作者更快看到想法，让开发者更容易理解方法。**

---

*测试说明：数据来自 2026 年 9 月技术测试记录。约 12 倍加速对应 4 张 RTX 6000D、5 秒、1344×768 文字生成视频测试，以 BF16、50 步方案为基线，是包含生成步数变化的整套方案对比；15 秒竖屏数据对应 8 张 RTX 6000D、4 步设置。以上为服务预热后的生成耗时，不代表包含排队等环节的完整等待时间，效果随任务和设置变化。*
