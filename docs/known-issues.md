# Known Issues

All original bugs (issues 1-5) are **fixed**. This document now tracks remaining optimization barriers.

---

## Resolved Issues

1. **K-Quant dequantization** (Q6_K scale, Q5_K nibble/bit layout, Q3_K type size) — Fixed
2. **Qwen3.5-9B conv buffer overflow** — Fixed
3. **Qwen3 HasGatedQ detection** — Fixed
4. **CausalConv1d buffer overflow** — Fixed
5. **Qwen3.5-9B Q/K head expansion** (repeat vs repeat_interleave) — Fixed

---

## Open: Vulkan DeltaNet Batching Crash

### Symptom

DeltaNet layers crash with `ErrorDeviceLost` when multiple dispatches are batched into a single Vulkan command buffer. Standard attention layers batch correctly.

### What Works
- Standard attention layers: fully batched per-layer (gives +46% for 8B models)
- Individual DeltaNet operations tested in isolation pass
- 1-2 compute dispatches from any pipeline in a batch work fine

### What Crashes
- A full DeltaNet layer (~12 dispatches) in a single command buffer
- Specific crashing shader(s): CausalConv1d, SplitQKV, L2NormGroups, ComputeDecayBeta, DeltaNetStep, or SiLUGate (not yet isolated)

### Workaround
DeltaNet layers use per-dispatch SubmitAndWait (no batching). Standard attention layers are batched. This gives full performance for standard attention models (Qwen3-8B, DeepSeek R1) and partial performance for hybrid DeltaNet models (Qwen3.5-0.8B, 9B).

### How to Debug
Need to test each DeltaNet composite operation in isolation within a batch to find which one crashes. Then inspect that shader's buffer access patterns for out-of-bounds reads/writes.

---

## Open: dp4a Precision Loss

### Symptom
The `__dp4a` integer dot product approach (CUDA) quantizes the activation to int8, which loses precision that compounds across 36 layers, producing garbage for 8B+ models.

### Status
dp4a path disabled. llama.cpp handles this with a more sophisticated Q8_1 format that preserves per-block activation sums for error compensation. A proper implementation would require matching their exact precision handling.

---

## Architecture Reference: Qwen3.5-9B DeltaNet

See previous documentation for tensor mapping, layer schedule, and architecture details.
