# Inference Pipeline

> Complete walkthrough of how daisi-llama transforms input text into generated output.
> [Definitions](definitions.md) | [Architecture](architecture.md) | [Roadmap](../README.md#roadmap)

---

## Pipeline Overview

```mermaid
flowchart LR
    TEXT["Input text"]
    TOK["Tokenize\n(BPE)"]
    EMB["Embed\n(token → vector)"]
    LAYERS["Transformer Stack\n(N layers)"]
    NORM["Final RMSNorm"]
    HEAD["LM Head\n(hidden → logits)"]
    SAMP["Sample\n(logits → token)"]
    DETOK["Detokenize"]
    OUT["Output text"]

    TEXT --> TOK --> EMB --> LAYERS --> NORM --> HEAD --> SAMP
    SAMP -->|"loop"| EMB
    SAMP --> DETOK --> OUT
```

---

## Full Token Generation Sequence

```mermaid
sequenceDiagram
    participant T as Tokenizer
    participant E as Embedding
    participant L as Transformer Layer ×N
    participant N as Final RMSNorm
    participant H as LM Head
    participant S as Sampler
    participant K as KV Cache

    Note over T,S: === Prefill Phase (all prompt tokens at once) ===
    T->>E: token_ids[0..P-1]
    E->>L: hidden_states [P × D]
    loop Each layer (0..N-1)
        L->>K: Write K, V for positions 0..P-1
        L->>L: Attention + FFN
    end
    L->>N: hidden_states[P-1] (last position)
    N->>H: normalized vector
    H->>S: logits [vocab_size]
    S->>S: Apply sampling strategy

    Note over T,S: === Decode Phase (one token at a time) ===
    loop Until EOS or max_tokens
        S->>E: next_token_id
        E->>L: hidden_state [1 × D]
        loop Each layer (0..N-1)
            L->>K: Append K, V for new position
            L->>K: Read all K, V for attention
            L->>L: Attention + FFN
        end
        L->>N: hidden_state
        N->>H: normalized vector
        H->>S: logits [vocab_size]
        S->>S: Apply sampling strategy
    end
```

---

## Per-Layer Detail

Each transformer layer performs the same sequence of operations. The input is a hidden state vector (or matrix during prefill) and the output is an updated hidden state of the same shape.

```mermaid
flowchart TD
    IN["Input hidden state\n[seq_len × hidden_dim]"]

    subgraph Attention["Self-Attention Block"]
        AN["RMSNorm\n(attn_norm.weight)"]
        QP["Q Projection\nattn_q.weight × normalized"]
        KP["K Projection\nattn_k.weight × normalized"]
        VP["V Projection\nattn_v.weight × normalized"]
        ROPE["RoPE\n(rotate Q and K by position)"]
        KVC_W["KV Cache Write\n(store K, V)"]
        KVC_R["KV Cache Read\n(retrieve all K, V)"]
        SDPA["Scaled Dot-Product Attention\nscore = (Q × K^T) / √d_head\nattn = softmax(score) × V"]
        OP["Output Projection\nattn_output.weight × attn"]
    end

    ADD1["Residual Add\nhidden + attn_output"]

    subgraph FFN["Feed-Forward Block (SwiGLU)"]
        FN["RMSNorm\n(ffn_norm.weight)"]
        GATE["Gate Projection\nffn_gate.weight × normalized"]
        UP["Up Projection\nffn_up.weight × normalized"]
        SILU["SiLU(gate)"]
        MUL["Element Multiply\nSiLU(gate) ⊙ up"]
        DOWN["Down Projection\nffn_down.weight × mul"]
    end

    ADD2["Residual Add\nresidual + ffn_output"]
    OUT["Output hidden state"]

    IN --> AN --> QP & KP & VP
    QP & KP --> ROPE
    ROPE --> KVC_W
    KVC_W --> KVC_R
    VP --> KVC_W
    KVC_R --> SDPA
    ROPE --> SDPA
    SDPA --> OP --> ADD1
    IN --> ADD1
    ADD1 --> FN --> GATE & UP
    GATE --> SILU --> MUL
    UP --> MUL
    MUL --> DOWN --> ADD2
    ADD1 --> ADD2
    ADD2 --> OUT
```

---

## Attention Mechanism

### Multi-Head Attention with Grouped Query Attention (GQA)

Qwen 3.5 0.8B uses GQA: 16 query heads share 8 KV heads (ratio 2:1). Each pair of query heads shares the same K and V head.

```mermaid
flowchart TD
    subgraph Projections["Linear Projections"]
        INPUT["normalized hidden state\n[seq × 1024]"]
        Q["Q = W_q × input\n[seq × 16 × 64]"]
        K["K = W_k × input\n[seq × 8 × 64]"]
        V["V = W_v × input\n[seq × 8 × 64]"]
    end

    subgraph PositionEncoding["RoPE"]
        QR["Q_rot = RoPE(Q, pos)"]
        KR["K_rot = RoPE(K, pos)"]
    end

    subgraph Cache["KV Cache"]
        KC["K_cache[layer][0:pos] = K_rot"]
        VC["V_cache[layer][0:pos] = V"]
        KA["All K = K_cache[layer][0:pos]"]
        VA["All V = V_cache[layer][0:pos]"]
    end

    subgraph ScaledDotProduct["Per-Head Attention (×16 heads)"]
        SCORE["scores = Q_rot × All_K^T / √64"]
        MASK["Apply causal mask\n(future positions = -∞)"]
        SM["attention_weights = softmax(scores)"]
        WEIGHTED["context = attention_weights × All_V"]
    end

    CONCAT["Concatenate 16 head outputs\n[seq × 1024]"]
    OUT_PROJ["output = W_o × concat\n[seq × 1024]"]

    INPUT --> Q & K & V
    Q --> QR
    K --> KR
    QR --> SCORE
    KR --> KC --> KA --> SCORE
    V --> VC --> VA --> WEIGHTED
    SCORE --> MASK --> SM --> WEIGHTED
    WEIGHTED --> CONCAT --> OUT_PROJ
```

### RoPE (Rotary Position Embedding)

RoPE encodes position by rotating pairs of dimensions using sinusoidal functions:

```
For dimension pair (2i, 2i+1) at position p:
    θ_i = rope_base^(-2i / d_head)
    q_rot[2i]   = q[2i] × cos(p × θ_i) - q[2i+1] × sin(p × θ_i)
    q_rot[2i+1] = q[2i] × sin(p × θ_i) + q[2i+1] × cos(p × θ_i)
```

The `rope_base` (theta) for Qwen 3.5 is 1,000,000 — a high value that extends effective context length.

---

## Feed-Forward Network (SwiGLU)

The SwiGLU FFN uses three weight matrices and a gated activation:

```
gate  = W_gate × input       [hidden_dim → intermediate_dim]
up    = W_up × input          [hidden_dim → intermediate_dim]
fused = SiLU(gate) ⊙ up       [element-wise multiply]
output = W_down × fused       [intermediate_dim → hidden_dim]
```

Where `SiLU(x) = x × σ(x)` and `σ` is the sigmoid function.

```mermaid
flowchart LR
    IN["input\n[1024]"]
    G["W_gate\n[1024 → 3072]"]
    U["W_up\n[1024 → 3072]"]
    SILU["SiLU"]
    MUL["⊙"]
    D["W_down\n[3072 → 1024]"]
    OUT["output\n[1024]"]

    IN --> G --> SILU --> MUL
    IN --> U --> MUL
    MUL --> D --> OUT
```

For Qwen 3.5 0.8B: hidden_dim = 1024, intermediate_dim = 3072. The FFN has 3× the parameters of a single attention layer (three matrices of 1024×3072 vs four matrices but with smaller KV projections).

---

## KV Cache

The KV cache avoids recomputing keys and values for previously processed tokens.

### Lifecycle

```mermaid
stateDiagram-v2
    [*] --> Allocated: Model load
    Allocated --> Prefilled: Prefill (write positions 0..P-1)
    Prefilled --> Growing: Decode (append one position per step)
    Growing --> Growing: Each decode step
    Growing --> [*]: Generation complete / dispose

    state Allocated {
        [*] --> Empty
        note right of Empty
            Pre-allocated for max_context_length
            or grown dynamically
        end note
    }
```

### Memory layout

```
Layer L, KV Cache:
┌──────────────────────────────────────────────────┐
│ K_cache[L]: [max_seq_len × kv_heads × head_dim] │  ← Written sequentially
│ V_cache[L]: [max_seq_len × kv_heads × head_dim] │  ← Read entirely each step
└──────────────────────────────────────────────────┘
```

**Memory cost per layer** (FP16, Qwen 3.5 0.8B):
- KV heads = 8, head_dim = 64, per position = 8 × 64 × 2 bytes × 2 (K+V) = **2 KB**
- At 32K context: 2 KB × 32,768 = **64 MB per layer**, × 28 layers = **1.75 GB total**
- KV cache quantization (Q8_0) roughly halves this to ~900 MB.

---

## Prefill vs Decode

| Aspect | Prefill | Decode |
|--------|---------|--------|
| **Tokens processed** | All prompt tokens at once | One new token per step |
| **Computation** | Large matrix multiplications (compute-bound) | Small vector-matrix operations (memory-bound) |
| **KV cache** | Written for all positions | Appended one position, read entirely |
| **Bottleneck** | GPU compute (TFLOPS) | Memory bandwidth (GB/s) |
| **Batch dimension** | seq_len (hundreds to thousands) | 1 |
| **Backend optimization** | Large GEMMs, high occupancy | Fused kernels, memory coalescing |

During prefill, the attention score computation is a `[P × D] × [D × P]` matmul — compute-intensive but highly parallelizable. During decode, it's a `[1 × D] × [D × S]` matmul where S grows with context — dominated by reading KV cache from memory.

---

## Sampling Strategies

After the forward pass produces logits (one float per vocabulary token), sampling selects the next token:

```mermaid
flowchart TD
    LOGITS["Raw logits\n[vocab_size]"]
    REP["Repetition penalty\n(divide seen token logits by penalty)"]
    TEMP["Temperature scaling\n(logits / temperature)"]
    TOPK["Top-k filtering\n(keep k highest, zero rest)"]
    TOPP["Top-p filtering\n(keep smallest set ≥ p cumulative prob)"]
    SM["Softmax\n(logits → probabilities)"]
    SAMPLE["Sample from distribution\n(or argmax if temperature = 0)"]
    TOKEN["Selected token ID"]

    LOGITS --> REP --> TEMP --> TOPK --> TOPP --> SM --> SAMPLE --> TOKEN
```

### Parameter effects

| Parameter | Low value | High value |
|-----------|-----------|------------|
| **Temperature** (0.0-2.0) | Deterministic, repetitive | Creative, chaotic |
| **Top-k** (1-100) | Very focused (k=1 = greedy) | Wide candidate set |
| **Top-p** (0.0-1.0) | Narrow (few candidates) | Broad (many candidates) |
| **Repetition penalty** (1.0-2.0) | No effect | Strongly discourages repeats |
