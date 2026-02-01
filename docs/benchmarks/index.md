# Benchmarks

*Tracking cost and latency improvements version-wise.*

## v0.1.0 (Current)

| Metric | Value | Notes |
|Str|Val|Notes|
|---|---|---|
| **Latency (Rust)** | < 1ms | Overhead added by Rust wrapper |
| **Token Savings** | ~15% | Avg savings via whitespace minification |
| **Cache Lookup** | < 0.1ms | In-Memory Hash Map |

## Roadmap

- [ ] Compare `SambaNovaLLM` (Rust) vs `langchain.llms.SambaNova` (Python) latency.
- [ ] Measure memory footprint reduction.
