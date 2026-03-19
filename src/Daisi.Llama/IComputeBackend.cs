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
    /// <param name="output">FP32 tensor to write the embedding into.</param>
    /// <param name="table">Embedding table tensor with dimensions [hiddenDim, vocabSize] (GGUF order).</param>
    /// <param name="tokenId">Row index (token ID) to look up.</param>
    void EmbeddingLookup(ITensor output, ITensor table, int tokenId);
}
