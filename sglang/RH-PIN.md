# RH Pin Info

This directory is a verbatim snapshot of
[sgl-project/sglang](https://github.com/sgl-project/sglang) `main` @
`f8cbf000f4a5bfd86d3fb7c1e2d6c8fb12339d0e` (2026-09-02) — the exact
version used and validated in this multi-GPU acceleration solution.

Content verified byte-for-byte against the wheel installed on the test
machines (sha256 `fbe0ff43...`). Do not upgrade in place casually;
upgrades should re-run the solution's A/B benchmarks.

Additions on top of the upstream tree are limited to: this pin file, and a
clearly-marked re-include block appended to `.gitignore` (upstream ignore
patterns such as `*.png` / `*.mod` / `*.jsonl` would otherwise hide tracked
snapshot files like `docs/cards/*.png`, `bindings/golang/go.mod` and
`benchmark/llava_bench/questions.jsonl`). No source file was modified.
