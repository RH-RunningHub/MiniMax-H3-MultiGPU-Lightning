# MiniMax-H3 文生视频 · 并行度与显存放置性能分析

**机器**：8 × RTX 5090 32GB（PCIe 拓扑、**无 NVLink**，2 路 NUMA；1TB 主机内存）
**代码**：本仓库 `sglang/`（sglang multimodal_gen / `sglang serve`）
**日期**：2026-09-13
**驱动脚本**：`$H3_HOME/bench-h3-int8-20step.py`
**原始结果**：`$H3_HOME/bench-int8-20step/results.json`
**逐配置完整日志**：`$H3_HOME/bench-int8-20step/logs/<case>.log`

---

## 1. 固定条件（全组共享，不可变）

| 项 | 取值 |
|---|---|
| DiT | `minimax_h3_fl2va_pruned_int8_convrot.safetensors`（剪枝 INT8，21 GB 磁盘） |
| 文本编码器 | `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`（NVFP4-AWQ，15.7 GB） |
| video VAE | **fp16**（`--component-precisions.video_vae=fp16`，权重 4.85 GB/卡） |
| audio VAE | **fp32**（`--component-precisions.audio_vae=fp32`，权重 0.56 GB/卡） |
| 注意力后端 | 去噪主干 = **comfy-kitchen INT8**（`ck_int8_attn`）；因果路径（TE / audio VAE）= SageAttention |
| torch.compile | **开启**，`offload_during_compile` 默认 on |
| 去噪步数 | **20** |
| LoRA | **不加载**（`lora_path=null`） |
| cache-dit | **关闭**（`SGLANG_CACHE_DIT_ENABLED=0`）——否则步数会被跳步缓存改写，无法代表「20 步」 |
| 分辨率 / 时长 | **1344×768（768p, 16:9）**，15.0 s = **362 帧 @ 24fps** |
| prompt / seed | 固定同一条 cyberpunk saxophone prompt，seed=42 |
| 显存分配器 | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |

自变量只有两个：**TP / Ulysses 组合** 与 **各组件的常驻/卸载方式（子配置）**。

## 2. 测量方法（如何保证「不是首加载 + 首次编译」的时间）

1. 启动时用 `--warmup-resolutions 1344x768 --warmup-steps 20` 让服务端在启动阶段就跑一次目标分辨率、
   目标步数的 synthetic warmup，把 `torch.compile` 的一次性开销压在启动阶段（启动总耗时 215–260 s）。
2. 每个配置在正式计次前**再发 1 次同样的 15 s 请求并丢弃**——因为启动 warmup 的帧数会被
   `SERVER_WARMUP_MAX_VIDEO_FRAMES` 截到 124 帧，362 帧的 shape 仍会触发一次重编译
   （实测这一发白跑要多花 **+17 s**，正好等于重编译成本）。
3. 主配置计 **3 次**，子配置计 **2 次**，取均值。逐次计时来自 API 的 `inference_time_s`，
   峰值显存同时记录 API 的 **torch reserved 峰值**与**独立的 nvidia-smi 1 Hz 采样峰值**（全 8 卡取最大）。
4. 阶段耗时来自 `SGLANG_DIFFUSION_STAGE_LOGGING=1` + `SGLANG_DIFFUSION_SYNC_STAGE_PROFILING=1`。

重复性极好：同一配置的 3 次结果离散度 < 0.1 %（例：156.57 / 156.54 / 156.59 s）。

## 3. 主结果

### 3.1 三种主配置（各自「能跑完 15 s / 20 步」的最优放置）

| 排名 | 主配置 | 子配置（显存放置） | **端到端时延** | 单卡峰值(torch) | 单卡峰值(nvidia-smi) | 每步去噪 |
|---|---|---|---|---|---|---|
| ★1 | **TP2 + U4** | DiT+VAE 常驻，TE 逐层卸载 | **127.61 s**（127.55/127.62/127.67） | **27.02 GB** | 26.8–28.1 GB | **6.04 s** |
| ★2 | **TP4 + U2** | DiT+VAE 常驻，TE 逐层卸载 | **156.57 s**（156.57/156.54/156.59） | **21.45 GB** | 22.5 GB | 7.50 s |
| ✗3 | **TP1 + U8** | 5 种放置组合全部尝试 | **全部 OOM** | — | 30.6–32.1 GB | — |

> TP2+U4 比 TP4+U2 快 **19 %**，但每卡峰值高 **5.6 GB**（27.0 vs 21.5 GB，即 86 % vs 68 % 的 31.36 GB）。
> TP1+U8 在本机上不可用（见 §4）。

### 3.2 全部可跑通的子配置

| 主配置 | 子配置 | 端到端 (s) | 去噪 (s) | 解码 (s) | 文本编码 (s) | 峰值 torch (GB) | 峰值 smi (GB) |
|---|---|---|---|---|---|---|---|
| TP4+U2 | **TE 逐层卸载**（DiT+VAE 常驻） | **156.57** | 150.0 | 5.1 | 0.9 | **21.45** | 22.5 |
| TP4+U2 | DiT 半常驻 26/52 + TE 逐层卸载 | 156.47 | 149.9 | 5.0 | 0.8 | 21.46 | **18.8–19.7** |
| TP4+U2 | 全常驻（TE 也驻留显存） | 156.40 | 149.9 | 5.0 | 0.8 | 28.06 | 25.5–26.6 |
| TP4+U2 | TE 整块卸载（component-offload） | 157.89 | 150.5 | 5.1 | 0.9 | 21.38 | 21.6–22.8 |
| TP4+U2 | DiT 逐层卸载 + TE 整块卸载 | 157.77 | 150.5 | 5.0 | 0.8 | 21.30 | 22.6 |
| TP2+U4 | **TE 逐层卸载**（DiT+VAE 常驻） | **127.61** | 120.8 | 5.1 | 1.0 | **27.02** | 26.8–28.1 |
| TP2+U4 | TE 整块卸载（DiT 常驻） | 129.01 | 121.4 | 5.0 | 1.0 | 27.78 | 29.06 |
| TP2+U4 | DiT 逐层卸载 + TE 整块卸载 | 129.04 | 121.4 | 5.0 | 1.0 | 27.82 | 29.10 |

**阶段拆解**：端到端 ≈ 去噪 + 解码(≈5.0 s) + 文本编码(≈0.9–1.0 s) + Latent 准备(0.1 s)。
**去噪占了 94–95 %**，其余阶段（含 fp16 video VAE 解码 362 帧）合计只有 ~6 s，所以优化空间全部在去噪循环里。

**权重上限（实测 `Loaded ... model size`）**

| | DiT/卡 | TE/卡 | video VAE | audio VAE | 合计 |
|---|---|---|---|---|---|
| TP4+U2 | 4.91 GB | 7.19 GB | 4.85 GB | 0.56 GB | 17.51 GB |
| TP2+U4 | 9.81 GB | 9.70 GB | 4.85 GB | 0.56 GB | 24.92 GB |
| TP1+U8 | 19.61 GB（每卡全量复制） | 14.73 GB | 4.85 GB | 0.56 GB | 39.75 GB |

## 4. TP1+U8：确定不可用

在 TP=1 下 DiT（19.6 GB）与 TE（14.7 GB）都**无法被张量并行切分**，只能整份驻留每张卡；
两者相加 34.3 GB 已经超过 31.36 GB 的物理显存。逐一尝试的 4 种卸载组合全部在 15 s 请求上 OOM：

| 尝试 | 放置 | 结果 |
|---|---|---|
| 1 | DiT 逐层卸载 + TE 整块卸载 + VAE 常驻 | OOM（解码阶段，已分配 28.10 GB）→ 重试仍 OOM（30.01 GB，1.38 MiB 空闲） |
| 2 | 同上，DiT prefetch=2 | OOM（峰值 32.08 GB） |
| 3 | DiT 逐层卸载 + **VAE 也逐层卸载** + TE 整块卸载 | OOM（30.0 GB，连 124 帧的启动 warmup 都过不了） |
| 4 | DiT 逐层卸载 + **VAE 整块卸载** + TE 整块卸载 | OOM（32.10 GB） |
| 5 | **TE 逐层卸载** + DiT 逐层卸载 | 直接崩：`comfy_nvfp4.py:90 embedding` → `Expected all tensors to be on the same device, but got index is on cpu`（TP1 下 `encoder parallel folding (mode=world)` 与 NVFP4 embedding 的 `weight_scale` 逐层搬运不兼容，属确定性 bug） |

值得注意：DiT 逐层卸载**并没有**把峰值降下来（TP2 上 DiT 逐层卸载 27.82 GB 反而比 DiT 常驻 27.78 GB 略高），
因为 host→device 的 staging buffer + 编译产物一起把省下的显存又吃回去了。
唯一真正有效的省显存手段是 **把 TE 做逐层卸载**（TP4 上省 6.6 GB 且不掉速）。
而在 TP1 下这条路正好被上面的 bug 堵死。

## 5. 结论与建议

1. **本机（8×5090、无 NVLink）跑 15 s / 768p / 20 步 / 无 LoRA，正确选择是 `TP2 + Ulysses4`。**
   - 端到端 **127.6 s**，每步 6.04 s，峰值 **27.0 GB/卡**。
2. **TP4+U2 更省显存但更慢**（156.6 s，21.5 GB/卡）。慢的原因是双重的：
   TP 4 路在 PCIe（无 NVLink）上做 all-reduce 的通信量是 TP2 的两倍；
   同时 Ulysses 只切 2 份，每卡注意力序列长度是 U4 的两倍。长序列场景下 Ulysses 的收益远大于 TP。
3. **TE 的放置是显存的第一杠杆，而且几乎免费**：把文本编码器从「整块搬入/搬出」换成
   「逐层卸载」后，TP4 峰值从 21.38 → 21.45 GB 基本不变、时间从 157.89 → 156.57 s（反而略快），
   TP2 峰值 27.78 → 27.02 GB、时间 129.01 → 127.61 s。**推荐所有配置都用 TE 逐层卸载。**
4. **DiT 逐层卸载不值得**：在 TP2/TP4 上既不省显存也不提速（PDT 差 < 0.2 %），
   却把 9.8–19.6 GB 的主机内存钉成 pinned（TP1 下每卡 19.55 GB）。
   只有在 TP4 用 `resident=26/52` 的**半常驻**时，nvidia-smi 峰值能压到 18.8 GB（省 ~3.7 GB）而时间不变，
   适合需要给别的进程留显存的场景。
5. **TP1+U8 不要用**（本机的 32 GB 装不下复制后的权重），除非：
   - 把 TE 换成可切分的实现（或放到 CPU/独立进程），或
   - 用显存更大的卡，或
   - 补齐 TP1 下 TE 逐层卸载 + NVFP4 embedding 的设备不匹配 bug。
6. **运维提醒**：
   - 峰值 27.8–29.1 GB（TP2+U4 系列）距离 31.36 GB 上限只有 2–3 GB，
    同样的配置在少数几次尝试中确实出现过 OOM/进程组坍塌（本项目早期证据里也有同样现象）。
     建议 TP2+U4 方案至少留 4 GB 余量，或降低分辨率/时长，或把 VAE 纳入卸载。
   - 首次真实请求会因 362 帧 shape 未被启动 warmup 覆盖而多花 ~17 s；
     若要首帧即稳态，需要提高 `SERVER_WARMUP_MAX_VIDEO_FRAMES`（当前把 362 帧截到 124 帧）。
   - 启动（加载 21 GB DiT + 15.7 GB TE + 10 GB VAE + 编译 + warmup）约 **215–260 s**，
     已排除在计次之外。

## 6. 复现方式

```bash
cd "$H3_HOME"
python3 bench-h3-int8-20step.py            # 全部 case，断点续跑
python3 bench-h3-int8-20step.py --only T2U4-te-lw --force
python3 /tmp/summarize.py                  # 汇总表
python3 /tmp/audit_args.py                 # 逐 case 回读 server_args，核对固定条件
```

`serve-h3.sh` 新增了三个透传开关（向后兼容）：`COMPONENT_PRECISIONS`（按组件精度）、
`WARMUP_RESOLUTIONS/WARMUP_NUM_FRAMES/WARMUP_STEPS`（启动期 warmup 形状）、
`EXTRA_SERVE_ARGS`（逐字追加，用于把组件常驻语义写死，避免 `performance-mode=speed`
的 auto-tuner 把「未显式声明」的组件默认改成卸载）。原文件已备份为 `serve-h3.sh.bak-20260913`。

---

# 附录 A：TP1+U8 的 DiT 逐层卸载到底能不能跑？（消融实验）

**结论：能跑，但前提是 cache-dit 打开。**

主表里 TP1+U8 报 OOM，是因为我为「干净的 20 步」把 cache-dit 关掉了。
补做单变量消融（其余全同：TP=1/U=8、剪枝 INT8 DiT 逐层卸载 prefetch=1 resident=0、
TE 整块卸载、VAE 常驻、video VAE fp16、audio VAE fp32、torch.compile on、
20 步、无 LoRA、1344×768×362 帧）：

| 实验 | 注意力后端 | cache-dit | 其它 | 结果 |
|---|---|---|---|---|
| A1 | SageAttention | 关 | — | **OOM**（31.26 GB） |
| A5 | comfy-kitchen INT8 | 关 | + `--regional-compile` | **OOM**（31.24 GB） |
| A6 | SageAttention | 关 | + `--regional-compile` | **OOM**（31.26 GB） |
| A4 | SageAttention | **开** | — | **59.4 / 59.7 s，峰值 27.5 GB** |
| A2 | comfy-kitchen INT8 | **开** | — | **60.0 / 60.1 s，峰值 27.5 GB** |

即：**注意力后端不是变量**（A2 vs A4 差 < 1 %），**cache-dit 才是**。

## A.1 死因定位

A1 的 OOM 栈（不是死在去噪）：

```
component_manager.py:296            begin_use
component_manager.py:405            _prepare_forward_use
component_residency_strategies.py:133  prepare_for_use
component_residency_strategies.py:33   _module_to_local_device     ← OOM 在这里
torch/nn/modules/module.py:1383     module.to(...)
```

也就是**把 14.73 GB 的 TE 整块搬上卡的那一刻**。此时卡上已经有 ~15.3 GB 常驻：

| 项 | TP1 下大小 |
|---|---|
| DiT（逐层卸载后常驻部分） | 1.14 GB |
| video VAE fp16 | 4.85 GB |
| audio VAE fp32 | 0.56 GB |
| 编译图池 / 激活 | ~8.7 GB |
| **小计** | **~15.3 GB** |
| + TE 整块搬入 | 14.73 GB |
| **合计** | **~30.0 GB** ← 撞上 ~30 GB 可用上限（31.36 GB 卡 − 显存上下文/NCCL） |

cache-dit 走 `CachedBlocks_Pattern_3_4_5` 的**按块编译**（`DBCache_F1B0_W4I1M0MC3_R0.24_N19`），
编译图池比整图编译小 4–5 GB，正好把 TE 搬入所需的余量腾出来。
`--regional-compile` 虽然把 DiT 编译成 52 个子模块（日志确认 `regional torch.compile for 52 submodules`，
124 帧 warmup 只要 1.09 s/it），但它省的是**编译期**峰值，省不了运行期图池，所以仍然 OOM。

## A.2 统一口径后的完整矩阵

| 主配置（TE 逐层卸载 + VAE 常驻） | cache-dit **关**（真跑满 20 次前向） | cache-dit **开**（20 步，含跳步） |
|---|---|---|
| **TP1 + U8** | **OOM**（4 种组合全挂） | **59.4–60.1 s / 27.5 GB** ★最快 |
| **TP2 + U4** | **127.6 s / 27.0 GB** ★cache-dit 关时最优 | 64.3–64.4 s / 29.8–30.0 GB |
| **TP4 + U2** | 156.6 s / 21.5 GB | 78.6–78.7 s / 25.0 GB |

**控制实验**：复刻本项目早期配置 H（sage_attn + cache-dit 开 + 4 步 turbo LoRA + compile on）
得 **26.8 s / 29.7 GB**，早期记录为 26.6 s / 27.7 GB → 测量口径与既有工作一致，
证明 TP1+U8 的 OOM 不是本次 harness 的假象。

## A.3 结论修正

- 原结论「TP2+U4 最优」**只在 cache-dit 关闭时成立**（即真正跑满 20 次 DiT 前向时，
  TP1+U8 因显存装不下而退赛）。
- 一旦允许 cache-dit，并行度排序恢复为**单调的 U8 > U4 > U2**（59.4 / 64.3 / 78.6 s），
  TP1+U8 反而最快 —— 因为 Ulysses 切得越细、每卡序列越短，且 TP=1 完全没有 all-reduce。
- **但 cache-dit 是有损加速**：`DBCache_F1B0_W4I1M0MC3_R0.24_N19` 表示 warmup 4 步、
  最多连续缓存 3 步、残差阈值 0.24，因此 59.4 s 并不等于 20 次完整 DiT 前向。
  需要「干净 20 步」就用 cache-dit 关 + TP2+U4；需要最低延迟且能接受近似，就用 cache-dit 开 + TP1+U8。
- 若必须「cache-dit 关 + TP1+U8」，唯一可行方向是**降低 TE 的瞬时占用**（TP1 下 TE 无法切分，14.73 GB
  只能整块搬入）：把 TE 放到 CPU/独立进程执行，或修掉 TP1 下 TE 逐层卸载与 NVFP4 embedding
  的 `weight_scale` 设备不匹配 bug（见主报告 §4）。

**复现**：
```bash
cd "$H3_HOME"
python3 bench-t1u8-ablation.py A1 A2 A4 A3   # 主消融
python3 bench-t1u8-ablation.py A5 A6         # regional-compile 对照
python3 bench-t1u8-ablation.py A7 A8         # cache-dit 开下的 TP2/TP4 对照
python3 /tmp/abl_summary.py
# 原始数据：bench-int8-20step/ablation/results.json，日志同目录 logs/
```

---

# 附录 B：组件级整块互斥（TE→DiT→VAE 各自加载/计算/卸载）

## B.1 逐层卸载的语义澄清

DiT 逐层卸载 = 整份权重放在主机 pinned 内存，按计算顺序流式搬运。日志：`layers=52`
（`token_refiner.blocks` 2 层 + `blocks` 50 层），19.61 GB ÷ 52 ≈ 0.377 GB/层，
`prefetch/group=1, resident=0/52` → 显存里只保留当前层 + 预取的 1 层，合计约 1.14 GB
（`Layerwise offload summary: transformer (vram: 1.14 GB, host pinned: 19.55 GB)`）。
即任意时刻 52 层里只有约 3 层在卡上，且这 19.61 GB 在一次请求的 20 步里会被搬运 20 遍。

## B.2 修正后实测：性能结论有效，旧峰值被 TE allocator 缓存污染

cache-dit 关，真跑满 20 次前向；整块 DiT 配置均使用 `--warmup-mode off`，并采用
附录 C 的 TE `memory_intensive=True` 清缓存修复。除了三组件全互斥，还补测了用户提出的
VAE 常驻、仅 TE↔DiT 整块互斥方案。

| DiT 放置方案 | 空转单卡显存 | 端到端 | 去噪 | 峰值(torch) | 峰值(nvidia-smi) | 每卡 pinned 主机内存 |
|---|---|---|---|---|---|---|
| DiT 逐层卸载 + TE 清缓存 | 8.56 GB | **113.38 s** | **104.5 s** | **24.74 GB** | **26.06 GB** | 19.55 GB |
| **VAE 常驻，TE↔DiT 整块互斥** | 6.69 GB | 115.63 / 116.01 s | 105.3 / 105.5 s | 29.22–29.25 GB | 30.56–30.62 GB | ≈ 0 |
| TE→DiT→VAE 三组件整块互斥 | **1.13 GB** | 116.54 / 117.29 s | 105.4 / 106.0 s | 25.18 GB | 26.52 GB | ≈ 0 |

旧三组件互斥实验为 116.57 / 116.83 s，修复后平均 116.92 s，只差约 0.2%，属于波动；
因此旧性能结论有效。旧峰值 30.3–31.2 GB 被 TE allocator 缓存污染，清缓存后降到 26.52 GB。

- VAE 常驻中间方案可以工作，平均 **115.82 s**，比 VAE 也卸载快 1.10 s；代价是峰值
  升到约 30.62 GB，只剩约 1.98 GB 物理余量。
- 它仍比 DiT 逐层卸载慢约 2.44 s（2.2%）。U8 下逐层 H2D 被 pinned host + 预取有效隐藏，
  而整块 DiT 的 19.61 GB H2D 直接落在关键路径。
- 三组件全互斥最省空转显存且峰值更安全；VAE 常驻节省一次 VAE 搬运；两种整块方案都
  消除了每卡 19.55 GB pinned 主机内存。

## B.3 必须知道的前提：启动 warmup 会把 DiT 钉在显存里

直接开这个方案（保留启动 warmup）仍会 OOM，原因不在方案本身：

- `denoising.py:1813` 把 transformer 标成 `preferred_ready_after_request=True`；
- `ComponentOffloadStrategy.finish_request` 遇到 `preferred and state.batch_is_warmup` 时
  调用 `prepare_for_use` + `wait_for_use`，**把它再搬回显存并留在那里而不是卸载**。

于是 warmup 后 DiT 的 19.61 GB 常驻（实测空转 23.07 GB），第一个真实请求再搬入
14.73 GB 的 TE 就爆了。绕过办法是 `--warmup-mode off`（让 `batch_is_warmup` 永远为假），
代价是首次真实请求自行承担编译，实测 272 s。

该方案走整块卸载路径，因此与 §4 的逐层卸载修复互不依赖。

## B.4 选型建议

- 要最低延迟与安全峰值 → **DiT 逐层卸载 + TE 清缓存**（113.38 s、26.06 GB）。
- 要消除 pinned 主机内存，愿意用约 2 GB 显存余量换速度 → **VAE 常驻、TE↔DiT 互斥**
  （平均 115.82 s、30.62 GB）。
- 要消除 pinned 主机内存且优先显存余量/低空转 → **三组件全互斥**
  （平均 116.92 s、26.52 GB、空转 1.13 GB）。

注意所谓“DiT 常驻”是**去噪阶段整块驻留**，不是跨请求永久驻留；TP1 下 DiT
19.61 GB + TE 14.73 GB 仅权重就超过单卡容量，不可能让两者永久同时驻留。

**复现**：`python3 probe-mem.py R3`（注意 `--warmup-mode off` 时服务端不会再打印
`ready to roll`，就绪判定要只依赖 `/health`）。

---

# 附录 C：逐层卸载下峰值为何仍有 20 多 GB

## C.1 进程内诊断：20 多 GB 主要是 reserved，不是 live tensor

在 layerwise pre/post hook 内记录首轮每层的 CUDA allocator 状态。启动 warmup 的主 DiT 第 0 层结果：

```text
第 0 层计算前：allocated=1293.3 MiB, reserved=25414.0 MiB
第 0 层计算中：allocated=1712.8 MiB, reserved=25794.0 MiB（含预取的下一层）
第 0 层释放后：allocated=1341.7 MiB, reserved=25794.0 MiB
第 49 层计算中：allocated=1761.2 MiB, reserved=25794.0 MiB
第 49 层释放后：allocated=1390.1 MiB, reserved=25794.0 MiB
```

真正 15 秒请求中，VAE、audio VAE 和请求 tensor 也常驻，DiT 每步入口约为
`allocated=7.10 GB, reserved=25.84 GB`。因此 `nvidia-smi` 看到的 27.2 GB 中，
真正存活的 tensor 只有约 7.1 GB；其余约 18.7 GB 是 PyTorch CUDA caching allocator
已经没有 tensor 使用、但仍向 CUDA driver 保留的缓存块。单层计算没有占 20 多 GB。

## C.2 根因：TE 卸载了 tensor，却没有清 CUDA allocator 缓存

调用链已经定位：

1. `TextEncodingStage.component_uses()` 创建 text encoder 的 `ComponentUse`，但没有设置
   `memory_intensive=True`；
2. 文本编码结束后，`ComponentOffloadStrategy.finish_use()` 执行 `module.to("cpu")`，
   TE tensor 确实离开 GPU；
3. 随后的 `ComponentResidencyManager._empty_cache_after_large_release()` 首行判断
   `if not use.memory_intensive: return`，所以没有调用 `torch.cuda.empty_cache()`；
4. TE 阶段留下约 15–19 GB allocator reserve。DiT 启动后只分配约 7 GB，
   但 `nvidia-smi` 和 `max_memory_reserved()` 把两者一起显示为约 27 GB。

这也解释了为什么 5 秒与 15 秒原先峰值相同：固定大小的是上一阶段 TE 留下的缓存池，
不是视频序列激活。

## C.3 A/B 修复验证

临时在 `text_encoding.py` 的 `ComponentUse` 加上 `memory_intensive=True`，让 TE 整块卸载后
进入已有的 `empty_cache()` 路径。其他参数完全不变。

| 目标规格 | 修复前 DiT 阶段 | 修复后 DiT 阶段 | 修复后 allocated / reserved | 整请求峰值（修复后） |
|---|---:|---:|---:|---:|
| 5 秒、122 帧、20 步 | 27162 MiB | **10426 MiB** | 6.87 / 9.10 GB | 26042 MiB |
| 15 秒、362 帧、20 步 | 27184 MiB | **12806 MiB** | 7.10 / 11.47 GB | 26064 MiB |

15 秒稳态总时延 113.38 s、去噪 104.5 s，与修复前 113.90 s / 105.3 s 基本一致。
DiT 阶段峰值下降约 **14.4 GB**，证明诊断成立。

整请求峰值只从约 27.2 GB 降到 26.1 GB，是因为修复后最高点转移到了 TE 自己运行的阶段：
请求开始后 1.3–2.5 秒约 26.1 GB；TE 结束并清缓存后，15 秒 DiT 平台为 12.8 GB。
如果要进一步降低「整请求峰值」，必须降低 TE 阶段本身的占用，例如修复 NVFP4 TE
逐层卸载、对 TE 做 TP，或在 TE 阶段卸载 VAE/audio VAE；继续优化 DiT 权重放置无济于事。

## C.4 指标解释与正式修复建议

- `allocated`：仍被活跃 tensor 占用，才接近实际工作集；
- `reserved`：PyTorch 向 CUDA driver 申请后暂存以供复用的池；
- `nvidia-smi`：看不到池内哪些块已空闲，因而接近 reserved；
- reserved 不是泄漏，同一 PyTorch 进程通常可以复用，但其他进程在 `empty_cache()` 前用不到，
  而且它会污染峰值统计并加剧碎片化边界下的 OOM 风险。

建议把 `TextEncodingStage.component_uses()` 的 text encoder 标记为
`memory_intensive=True`，复用现有的「组件确实从设备卸载后才 empty_cache」保护。
该修复的 15 秒实测没有性能回退。正式合入前应跑其他模型的组件驻留测试，确认
`memory_intensive` 带来的下一组件预取顺序变化没有副作用。

诊断插桩和 TE 临时修复测试后均已撤销；机器上仍只保留 §4 的 `denoising.py` 补丁。
复现日志：`profile.log`、`profile_tefix.log`、`profile_tefix_15s.log`。

## C.5 TE/VAE 全常驻、仅 DiT 逐层卸载

按用户提出的放置方式实测：text encoder、video VAE、audio VAE 全部常驻，只有 DiT
使用 `prefetch=1, resident=0/52` 逐层卸载。

| 指标 | 首个 15 秒请求 | 第二个稳态请求 |
|---|---:|---:|
| 空转单卡显存 | 23.58 GB | 23.58 GB |
| 总时延 | 124.36 s | **111.24 s** |
| 去噪 | — | 103.8 s |
| torch reserved 峰值 | 29.50 GB | 29.52 GB |
| nvidia-smi 峰值 | **31.88 GB** | **30.81 GB** |
| 相对 32607 MiB 物理上限的余量 | **725 MiB** | **1801 MiB** |
| 去噪阶段平台 | — | 27.85 GB |

配置可以跑通，并比 TE 整块卸载的原稳态 113.63 s 快约 2.1%；但首个新形状请求只有
725 MiB 显存余量，容易因 prompt 长度、allocator 碎片、不同输入形状、并发或后端波动 OOM。
因此它适合独占卡、固定形状下的激进低延迟模式，不建议作为稳健默认值。

更均衡的默认方案仍是：TE 组件卸载 + 本附录的 `memory_intensive=True` 清缓存修复、
VAE/audio VAE 常驻、DiT 逐层卸载。该方案 15 秒总时延 113.38 s、整请求峰值 26.06 GB，
比全常驻只慢约 2.14 s，却多出约 5.8 GB 的首请求显存余量。

## C.6 VAE 常驻、TE↔完整 DiT 互斥

旧 `R2` 因保留启动 warmup，使完整 DiT 在 warmup 后留在 GPU（空转 28.56 GB），随后
TE 上卡 OOM，不能用于否定该方案。按正确条件重测：VAE/audio VAE 常驻，transformer 与
text encoder 为 component-offload，`--warmup-mode off`，TE 卸载后清 allocator 缓存。

| 指标 | 首个编译请求 | 稳态 run 1 | 稳态 run 2 |
|---|---:|---:|---:|
| 空转显存 | 6.69 GB | 6.69 GB | 6.69 GB |
| 总时延 | 177.22 s | **115.63 s** | **116.01 s** |
| 去噪 | — | 105.3 s | 105.5 s |
| torch 峰值 | 29.29 GB | 29.25 GB | 29.22 GB |
| nvidia-smi 峰值 | 30.63 GB | 30.62 GB | 30.56 GB |

方案可行，稳态平均 115.82 s；比三组件全互斥快 1.10 s，说明省掉 VAE 搬运确实有效。
但峰值比三组件全互斥高约 4.1 GB，仅余约 1.98 GB；它是“低主机 pinned 内存”方案里的
性能档，而三组件全互斥是显存安全档。

复现：`profile_r2fixed.py`；日志：`profile_r2fixed.log`、`profile_r2fixed.json`。
