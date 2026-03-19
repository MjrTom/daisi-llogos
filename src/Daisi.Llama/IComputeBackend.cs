using Daisi.Llama.Gguf;

namespace Daisi.Llama;

/// <summary>
/// Abstraction for a compute provider that creates tensors and executes tensor operations.
/// Each backend targets specific hardware (CPU/SIMD, CUDA, Vulkan, Metal).
/// The inference engine works exclusively through this interface.
/// </summary>
public interface IComputeBackend : IDisposable
{
    /// <summary>
    /// Human-readable backend name (e.g. "CPU", "CUDA").
    /// </summary>
    string Name { get; }

    /// <summary>
    /// Allocate an empty tensor with the given shape and type.
    /// </summary>
    ITensor CreateTensor(string name, GgmlType type, ReadOnlySpan<long> dimensions);

    /// <summary>
    /// Allocate a tensor and populate it with raw data from a GGUF file.
    /// The data span must match the expected byte size for the given type and dimensions.
    /// </summary>
    ITensor LoadTensor(string name, GgmlType type, ReadOnlySpan<long> dimensions, ReadOnlySpan<byte> data);

    // ── Math Operations ──────────────────────────────────────────────────────

    /// <summary>
    /// Matrix multiplication: output = a × b.
    /// a is [M × K], b is [K × N], output is [M × N]. All stored as 1D row-major.
    /// b may be quantized — the backend fuses dequantization into the multiply.
    /// </summary>
    void MatMul(ITensor output, ITensor a, ITensor b, int M, int K, int N);

    /// <summary>
    /// RMS normalization: output[i] = (input[i] / rms) * weight[i]
    /// where rms = sqrt(mean(input²) + eps).
    /// All tensors are 1D with the same element count.
    /// </summary>
    void RmsNorm(ITensor output, ITensor input, ITensor weight, float eps);

    /// <summary>
    /// Numerically stable softmax over a 1D float tensor.
    /// output[i] = exp(input[i] - max) / sum(exp(input - max)).
    /// </summary>
    void Softmax(ITensor output, ITensor input);

    /// <summary>
    /// SiLU activation: output[i] = input[i] * sigmoid(input[i]).
    /// </summary>
    void SiLU(ITensor output, ITensor input);

    /// <summary>
    /// Rotary position embedding applied in-place to q and k tensors.
    /// Rotates pairs of dimensions (2i, 2i+1) by position-dependent angles.
    /// Only the first <paramref name="ropeDim"/> dimensions of each head are rotated.
    /// </summary>
    /// <param name="q">Query tensor [nHeads × headDim], modified in-place.</param>
    /// <param name="k">Key tensor [nKvHeads × headDim], modified in-place.</param>
    /// <param name="headDim">Dimension of each attention head.</param>
    /// <param name="ropeDim">Number of dimensions to apply rotation to (0 = all).</param>
    /// <param name="positionOffset">Starting position index for the rotation.</param>
    /// <param name="ropeTheta">RoPE frequency base (e.g. 1000000.0).</param>
    void RoPE(ITensor q, ITensor k, int headDim, int ropeDim, int positionOffset, float ropeTheta);

    /// <summary>
    /// Element-wise multiply: output[i] = a[i] * b[i].
    /// </summary>
    void ElementMul(ITensor output, ITensor a, ITensor b);

    /// <summary>
    /// Element-wise add: output[i] = a[i] + b[i].
    /// </summary>
    void ElementAdd(ITensor output, ITensor a, ITensor b);

    /// <summary>
    /// Copy a single row from a (possibly quantized) embedding table into an FP32 output tensor.
    /// Dequantizes the row if the table is quantized.
    /// </summary>
    void EmbeddingLookup(ITensor output, ITensor table, int tokenId);

    // ── Composite Operations (used by forward pass) ─────────────────────────

    /// <summary>
    /// Copy tensor data from src to dst. Both must be same type and size.
    /// </summary>
    void CopyTensor(ITensor dst, ITensor src);

    /// <summary>
    /// In-place SiLU activation: data[i] = data[i] * sigmoid(data[i]).
    /// </summary>
    void SiLUInPlace(ITensor data);

    /// <summary>
    /// L2-normalize groups of elements in-place.
    /// For each group g of size groupDim: normalize data[g*groupDim .. (g+1)*groupDim-1].
    /// </summary>
    void L2NormGroups(ITensor data, int numGroups, int groupDim);

    /// <summary>
    /// Per-head RMSNorm in-place: apply RMSNorm independently to each head-sized slice.
    /// weight has headDim elements, shared across all heads.
    /// </summary>
    void PerHeadRmsNorm(ITensor data, ITensor weight, int numHeads, int headDim, float eps);

    /// <summary>
    /// De-interleave Q projection output from [q_h0, gate_h0, q_h1, gate_h1, ...]
    /// into separate contiguous Q_attn and Q_gate buffers.
    /// </summary>
    void DeInterleaveQ(ITensor qAttn, ITensor qGate, ITensor qFull, int numHeads, int headDim);

    /// <summary>
    /// Write K and V vectors for a single position into the KV cache tensors.
    /// k: [nKvHeads × keyLength], kCache: [nKvHeads × maxSeqLen × keyLength].
    /// </summary>
    void KvCacheWrite(ITensor kCache, ITensor vCache, ITensor k, ITensor v,
        int nKvHeads, int keyLength, int valueLength, int maxSeqLen, int position);

    /// <summary>
    /// Compute multi-head attention: scores, softmax, weighted V sum, sigmoid gating.
    /// output = sigmoid(qGate) * softmax(qAttn @ kCache^T / scale) @ vCache.
    /// </summary>
    void GatedAttention(ITensor output, ITensor qAttn, ITensor qGate,
        ITensor kCache, ITensor vCache,
        int numHeads, int numKvHeads, int keyLength, int valueLength,
        int maxSeqLen, int seqLen, float scale);

    /// <summary>
    /// Apply depthwise causal conv1d with shift buffer.
    /// Modifies qkv in-place and updates convBuffer.
    /// </summary>
    void CausalConv1d(ITensor qkv, ITensor convBuffer, ITensor convWeight, int channels, int kernelSize);

    /// <summary>
    /// Compute DeltaNet decay and beta from alpha projection, ssmA, dt_bias.
    /// decay[g] = exp(ssmA[g] * softplus(alpha[g] + dtBias[g]))
    /// beta[g] = sigmoid(betaProj[g])
    /// </summary>
    void ComputeDecayBeta(ITensor decay, ITensor beta, ITensor alphaProj, ITensor betaProj,
        ITensor ssmA, ITensor dtBias, int groupCount);

    /// <summary>
    /// Fused DeltaNet state update + output computation:
    /// For each group: sk = S^T*k, error = (v - d*sk)*beta, S = d*S + k*error, o = S^T*q * scale.
    /// Then per-head RMSNorm on output using normWeight.
    /// </summary>
    void DeltaNetStep(ITensor output, ITensor q, ITensor k, ITensor v,
        ITensor state, ITensor decay, ITensor beta,
        ITensor normWeight, int groupCount, int headDim, float scale, float normEps);

    /// <summary>
    /// Element-wise: output[i] = data[i] * silu(gate[i]).
    /// </summary>
    void SiLUGate(ITensor output, ITensor data, ITensor gate);

    /// <summary>
    /// Split a concatenated [Q|K|V] buffer into separate tensors.
    /// qkv has 3*innerSize elements; q, k, v each have innerSize elements.
    /// </summary>
    void SplitQKV(ITensor q, ITensor k, ITensor v, ITensor qkv, int innerSize);

    /// <summary>
    /// Zero all elements of a tensor.
    /// </summary>
    void ZeroTensor(ITensor tensor);

    /// <summary>
    /// Copy the first byteCount bytes from src to dst. dst may be larger than src.
    /// Used for growing cache reallocation.
    /// </summary>
    void CopyTensorBytes(ITensor dst, ITensor src, long byteCount);

    /// <summary>
    /// Create a tensor in host-accessible memory (pinned for GPU, regular for CPU).
    /// GPU kernels can access this tensor via mapped device pointers, but at reduced bandwidth.
    /// </summary>
    ITensor CreateHostTensor(string name, GgmlType type, ReadOnlySpan<long> dimensions);
}
