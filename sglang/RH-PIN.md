# RH Pin Info

This directory is a verbatim snapshot of
[sgl-project/sglang](https://github.com/sgl-project/sglang) `main` @
`1f6ce4b0686232fce5feb0d0d841df7d9bcabc02` (2026-09-26).

Additions on top of the upstream tree are limited to: this pin file, and a
clearly-marked re-include block appended to `.gitignore` (upstream ignore
patterns such as `*.png` / `*.mod` / `*.jsonl` would otherwise hide tracked
snapshot files like `docs/cards/*.png`, `bindings/golang/go.mod` and
`benchmark/llava_bench/questions.jsonl`). No source file was modified.

Notes for this pin:

- Bumped from the previous pin `f8cbf000f4a5bfd86d3fb7c1e2d6c8fb12339d0e`
  (2026-09-02). The published performance tables in this README were measured
  on that earlier pin and remain tied to its software versions.
- This pin carries the H3-relevant upstream changes merged since then,
  notably: SM120 FP8 attention backend (`fp8_fa_sm120`, #40175), fused
  SwiGLU for quantized MiniMax-H3 MLPs (#40378), MiniMax-H3 PDD (Parallel
  Decoding Distillation) inference support (#40568), a multimodal feature
  offload race fix (#40621), and a 400 response for corrupt image inputs
  (#28131). Mamba2 `selective_state_update` speedup for B200 (#41196-class)
  is included but its 6000D benefit is untested.
