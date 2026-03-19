# DeltaNet & Hybrid Architecture

> Gated DeltaNet linear attention as used in Qwen 3.5's hybrid architecture.
> [Definitions](definitions.md) | [Architecture](architecture.md) | [Phase 7 Roadmap](roadmap/phase-07-deltanet.md)

---

## Overview

Qwen 3.5 uses a **hybrid** architecture: most layers use standard softmax attention, but select layers use **Gated DeltaNet** — a linear attention variant with O(1) per-token memory and compute during decode. This enables efficient long-context inference without the quadratic KV cache growth of full attention.

DeltaNet is implemented as a drop-in replacement for the standard attention block. The rest of the layer (RMSNorm, SwiGLU FFN, residuals) is identical.

---

## Standard Attention vs DeltaNet

```mermaid
flowchart TD
    subgraph Standard["Standard Softmax Attention"]
        direction TB
        S_QKV["Project Q, K, V"]
        S_ROPE["Apply RoPE to Q, K"]
        S_KV["Store K, V in KV cache"]
        S_SCORE["scores = Q × K_cache^T / √d"]
        S_MASK["Apply causal mask"]
        S_SOFT["softmax(scores)"]
        S_OUT["output = attn_weights × V_cache"]

        S_QKV --> S_ROPE --> S_KV --> S_SCORE --> S_MASK --> S_SOFT --> S_OUT
    end

    subgraph DeltaNet["DeltaNet Linear Attention"]
        direction TB
        D_QKV["Project Q, K, V"]
        D_BETA["Compute β (gate scalar)"]
        D_NORM["Normalize Q, K"]
        D_STATE["Update state:\nS ← S + β × (V - K^T × S)^T × K"]
        D_OUT["output = Q × S"]

        D_QKV --> D_BETA --> D_NORM --> D_STATE --> D_OUT
    end

    style Standard fill:#e8f0fe,stroke:#4285f4
    style DeltaNet fill:#fce8e6,stroke:#ea4335
```

### Key differences

| Property | Standard Attention | DeltaNet |
|----------|-------------------|----------|
| **Per-token decode cost** | O(context_length) | O(1) |
| **Memory per layer** | KV cache grows with context | Fixed-size state matrix |
| **Training** | Straightforward backprop | Requires delta rule update |
| **Quality** | Best for precise retrieval | Better for smooth reasoning |
| **Causal masking** | Explicit mask matrix | Inherent (recurrent update) |
| **Position encoding** | RoPE (sinusoidal rotation) | None needed (recurrent) |

---

## DeltaNet State Update

DeltaNet maintains a **state matrix** `S` of shape `[d_head × d_head]` per head per layer. This replaces the KV cache for that layer.

### Update equation

At each position t:

```
β_t = sigmoid(W_β × x_t)                    # gating scalar
e_t = V_t - K_t^T × S_{t-1}                 # error: what V should be minus what S predicts
S_t = S_{t-1} + β_t × (e_t^T × K_t)         # delta update to state
o_t = Q_t × S_t                              # output query against updated state
```

### Step-by-step state evolution

```mermaid
flowchart TD
    subgraph Step1["Position t=0"]
        S0["S_0 = 0\n(zero matrix)"]
        K0["K_0, V_0"]
        B0["β_0 = sigmoid(W_β x_0)"]
        E0["e_0 = V_0 - K_0^T × 0 = V_0"]
        S1["S_1 = 0 + β_0 (V_0^T × K_0)\n= β_0 V_0^T K_0"]
        O0["o_0 = Q_0 × S_1"]

        S0 --> E0
        K0 --> E0
        B0 --> S1
        E0 --> S1
        S1 --> O0
    end

    subgraph Step2["Position t=1"]
        S1_in["S_1 (from previous)"]
        K1["K_1, V_1"]
        B1["β_1 = sigmoid(W_β x_1)"]
        E1["e_1 = V_1 - K_1^T × S_1"]
        S2["S_2 = S_1 + β_1 (e_1^T × K_1)"]
        O1["o_1 = Q_1 × S_2"]

        S1_in --> E1
        K1 --> E1
        B1 --> S2
        E1 --> S2
        S2 --> O1
    end

    subgraph StepN["Position t=N"]
        SN["S_N (accumulated)"]
        KN["K_N, V_N"]
        BN["β_N"]
        EN["e_N = V_N - K_N^T × S_N"]
        SN1["S_{N+1} = S_N + β_N (e_N^T × K_N)"]
        ON["o_N = Q_N × S_{N+1}"]

        SN --> EN
        KN --> EN
        BN --> SN1
        EN --> SN1
        SN1 --> ON
    end

    Step1 --> Step2
    Step2 -->|"..."| StepN
```

### Intuition

- The state `S` acts as an **associative memory** that maps keys to values
- The **delta update** adjusts S to better predict V from K — it's an online learning rule
- The **gate β** controls how much each position updates the memory (higher β = stronger write)
- The **error term** `e_t = V_t - K_t^T S` measures how well the current state predicts the desired value — the update corrects this error
- During decode, each step is **O(d_head²)** regardless of context length

---

## Qwen 3.5 Layer Schedule

Qwen 3.5 0.8B has 28 transformer layers. The model uses a **hybrid schedule** where specific layers use DeltaNet and the rest use standard attention.

```mermaid
flowchart TD
    subgraph Legend
        STD_L["■ Standard Attention"]
        DN_L["■ DeltaNet"]
    end

    subgraph Layers["28 Layers"]
        L0["Layer 0\nStandard"]
        L1["Layer 1\nStandard"]
        L2["Layer 2\nDeltaNet"]
        L3["Layer 3\nStandard"]
        L4["..."]
        LN["Layer 27\nStandard"]
    end

    style L2 fill:#ea4335,color:#fff
    style L0 fill:#4285f4,color:#fff
    style L1 fill:#4285f4,color:#fff
    style L3 fill:#4285f4,color:#fff
    style LN fill:#4285f4,color:#fff
```

> **Note:** The exact layer schedule (which layers are DeltaNet vs standard) is specified in the model's GGUF metadata. The implementation reads this schedule dynamically rather than hardcoding it.

The hybrid approach gets the best of both worlds:
- **Standard attention layers** provide precise token-level retrieval (good for factual recall, copying)
- **DeltaNet layers** provide efficient long-range reasoning (good for summarization, sustained context)

---

## Memory Comparison

### Per-layer memory during decode

| Component | Standard Attention | DeltaNet |
|-----------|-------------------|----------|
| **State** | KV cache: `2 × seq_len × kv_heads × head_dim × sizeof(dtype)` | State matrix: `heads × head_dim × head_dim × sizeof(float)` |
| **At 1K context** (FP16) | 2 × 1024 × 8 × 64 × 2 = **2 MB** | 16 × 64 × 64 × 4 = **1 MB** |
| **At 8K context** (FP16) | 2 × 8192 × 8 × 64 × 2 = **16 MB** | **1 MB** (constant) |
| **At 32K context** (FP16) | 2 × 32768 × 8 × 64 × 2 = **64 MB** | **1 MB** (constant) |
| **Growth** | Linear with context length | **Constant** |

```mermaid
---
config:
    xyChart:
        width: 600
        height: 400
---
xychart-beta
    title "Per-Layer Memory vs Context Length"
    x-axis "Context Length (K tokens)" [1, 2, 4, 8, 16, 32]
    y-axis "Memory (MB)" 0 --> 70
    line "Standard Attention" [2, 4, 8, 16, 32, 64]
    line "DeltaNet" [1, 1, 1, 1, 1, 1]
```

This is why the hybrid approach is powerful at long contexts: DeltaNet layers contribute zero KV cache growth, dramatically reducing total memory for a 28-layer model where (for example) 8 layers use DeltaNet.

---

## Implementation Approach

### Weight tensors for DeltaNet layers

DeltaNet layers have different weight tensors than standard attention layers:

| Standard Attention | DeltaNet |
|-------------------|----------|
| `attn_q.weight` | `attn_q.weight` |
| `attn_k.weight` | `attn_k.weight` |
| `attn_v.weight` | `attn_v.weight` |
| `attn_output.weight` | `attn_output.weight` |
| — | `attn_beta.weight` (gate projection) |
| `attn_norm.weight` | `attn_norm.weight` |

### Backend operations needed

DeltaNet requires these additional backend operations beyond standard attention:

| Operation | Description | Shape |
|-----------|-------------|-------|
| **OuterProduct** | `e^T × K` to form the state update | `[d_head] × [d_head] → [d_head × d_head]` |
| **MatVecBatched** | `Q × S` for output, batched over heads | `[d_head] × [d_head × d_head] → [d_head]` per head |
| **Sigmoid** | For computing β gate | Element-wise |
| **StateUpdate** | Fused `S += β × (e^T × K)` | In-place on state matrix |

These may be added to `IComputeBackend` or implemented as a DeltaNet-specific extension.

### Forward pass integration

```mermaid
flowchart TD
    CHECK{"Layer type?"}
    STD["Standard attention\n(Q/K/V → RoPE → KV cache → softmax attn)"]
    DN["DeltaNet attention\n(Q/K/V → β gate → state update → query state)"]
    MERGE["Output projection + residual"]

    CHECK -->|"standard"| STD --> MERGE
    CHECK -->|"deltanet"| DN --> MERGE
```

The inference engine checks each layer's type (from model metadata) and dispatches to the appropriate attention implementation. Everything else in the layer (norms, FFN, residuals) is shared.
