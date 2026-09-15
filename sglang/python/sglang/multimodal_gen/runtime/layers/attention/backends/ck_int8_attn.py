# SPDX-License-Identifier: Apache-2.0
"""comfy-kitchen INT8 注意力后端（Comfy Kitchen Attention）。

原理：把注意力的 Q/K/V 在线量化为 INT8，用 tensor core 做 INT8 matmul，
累加仍为 FP32，算完反量化回输入精度。相对 SageAttention 的区别在于来源与
调度：comfy-kitchen 由 ComfyUI 官方维护、随主线同步，并在内部提供
CUDA 原生 / ROCm / Triton / eager 多后端自动调度；本机已在 INT8-ConvRot DiT
上使用同一依赖，此处复用它是为了不再引入第三个需要按 commit 编译的第三方库。

布局约定：SGLang 传入的是 (batch, seq, heads, head_dim)，
comfy-kitchen 要求 (batch, heads, seq, head_dim)，故此处做两次 transpose。
转置后的张量最后一维 stride 仍为 1，满足内核要求，无需 contiguous 拷贝。

MiniMax-H3 的实际取值是 56 头 × head_dim 128，正好落在内核原生宽度上
（内核按 64/128/256 分档，128 档无需 padding）。
"""

from __future__ import annotations

import torch

from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

try:  # 允许在未安装 comfy-kitchen 的环境里导入本模块（resolver 会回退到 FA）
    import comfy_kitchen as _ck
except ImportError:  # pragma: no cover
    _ck = None

try:  # 因果路径的备选内核（见 forward 中的说明）
    from sageattention import sageattn as _sageattn
except ImportError:  # pragma: no cover
    _sageattn = None


def _make_opaque_int8_attention():
    """把 comfy-kitchen 的注意力内核包装成 Dynamo 的**不透明调用**。

    `comfy_kitchen.int8_attention` 是一个包含 dataclass 构造、按 head_dim 动态
    padding 与 DLPack 包装的纯 Python 函数。Dynamo 会尝试内联它，而在按形状
    重新编译时生成的代码会引用未定义变量——实测在 8 卡 Ulysses 下报
    `name 's78' is not defined`（单卡侥幸通过，因为重编译路径不同）。

    标记为不透明后，Dynamo 在调用点断图，注意力本身以 eager 方式执行，
    前后其余算子仍然被编译。这同时让不同序列长度不再触发注意力子图重编译。
    """
    disable = getattr(torch.compiler, "disable", None) or torch._dynamo.disable

    @disable
    def _opaque(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float | None):
        return _ck.int8_attention(q, k, v, scale=scale)

    return _opaque


if _ck is not None:
    _ck_int8_attention = _make_opaque_int8_attention()
else:  # pragma: no cover
    _ck_int8_attention = None

logger = init_logger(__name__)


def _trailing_padding_used_len(
    *,
    total_tokens: int,
    max_seqlen: int,
    bounds: tuple[int, ...],
) -> int | None:
    """Return live token count for H3-style [0, used, total] trailing padding."""
    if len(bounds) != 3:
        return None
    start, used, total = bounds
    if start != 0 or used >= total or total != total_tokens or used != max_seqlen:
        return None
    return used


class CkInt8AttentionBackend(AttentionBackend):

    accept_output_buffer: bool = True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        # 内核把 head_dim 归到 64/128/256 三档，其一维上界为 256。
        return [64, 128, 256]

    @staticmethod
    def get_enum() -> AttentionBackendEnum:
        return AttentionBackendEnum.CK_INT8_ATTN

    @staticmethod
    def get_impl_cls() -> type["CkInt8AttentionImpl"]:
        return CkInt8AttentionImpl


class CkInt8AttentionImpl(AttentionImpl):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        causal: bool,
        softmax_scale: float,
        num_kv_heads: int | None = None,
        prefix: str = "",
        **extra_impl_args,
    ) -> None:
        if _ck is None:
            raise ImportError(
                "comfy-kitchen is required for the ck_int8_attn backend: "
                "pip install 'comfy-kitchen>=0.2.30'"
            )
        if head_size > 256:
            raise ValueError(
                f"comfy-kitchen int8 attention supports head_dim <= 256, got {head_size}"
            )
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.dropout = extra_impl_args.get("dropout_p", 0.0)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata,
        *,
        return_softmax_lse: bool = False,
    ) -> torch.Tensor:
        if return_softmax_lse:
            raise NotImplementedError(
                "comfy-kitchen int8 attention does not expose the softmax LSE"
            )
        # (B, S, H, D) -> (B, H, S, D)
        q = query.transpose(1, 2)
        k = key.transpose(1, 2)
        v = value.transpose(1, 2)

        if self.causal:
            # 因果注意力出现在文本编码器（Qwen3-VL，带 GQA）与音频 VAE 上，
            # comfy-kitchen 的内核没有 causal 掩码实现。这里不自行拼掩码，而是：
            #   1) 本机装有 SageAttention 时复用它已编译的因果内核；
            #   2) 否则回退到 PyTorch SDPA（GQA 需要 enable_gqa）。
            # 这样切到本后端不会让非主干路径退化，两组对照的差异也被隔离在
            # 非因果（DiT 主干）注意力上。
            if _sageattn is not None:
                return _sageattn(
                    query,
                    key,
                    value,
                    tensor_layout="NHD",
                    is_causal=True,
                    sm_scale=self.softmax_scale,
                )
            output = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                is_causal=True,
                scale=self.softmax_scale,
                enable_gqa=q.shape[1] != k.shape[1],
            )
            return output.transpose(1, 2)

        output = _ck_int8_attention(q, k, v, self.softmax_scale)
        return output.transpose(1, 2)

    def forward_varlen(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        cu_seqlens_host: tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        bounds = (
            cu_seqlens_host
            if cu_seqlens_host is not None
            else tuple(int(x) for x in cu_seqlens.tolist())
        )
        return self._packed(
            query.contiguous(),
            key.contiguous(),
            value.contiguous(),
            bounds=bounds,
            max_seqlen=max_seqlen,
        )

    def _packed(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        bounds: tuple[int, ...],
        max_seqlen: int,
    ) -> torch.Tensor:
        # MiniMax-H3 packs one live document as bounds=(0, used, total):
        # [0, used) are real tokens; [used, total) is 64-aligned tail padding.
        used = _trailing_padding_used_len(
            total_tokens=query.shape[0],
            max_seqlen=max_seqlen,
            bounds=bounds,
        )
        if used is not None:
            live_out = self.forward(
                query[:used].unsqueeze(0),
                key[:used].unsqueeze(0),
                value[:used].unsqueeze(0),
                None,
            )[0]
            if used == query.shape[0]:
                return live_out
            # Keep padded tail at zero so downstream masked rows stay inactive.
            output = torch.zeros_like(query)
            output[:used] = live_out
            return output

        output = torch.empty_like(query)
        for start, stop in zip(bounds[:-1], bounds[1:]):
            if start == stop:
                continue
            output[start:stop] = self.forward(
                query[start:stop].unsqueeze(0),
                key[start:stop].unsqueeze(0),
                value[start:stop].unsqueeze(0),
                None,
            )[0]
        return output
