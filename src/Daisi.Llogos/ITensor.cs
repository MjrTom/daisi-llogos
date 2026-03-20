using Daisi.Llogos.Gguf;

namespace Daisi.Llogos;

/// <summary>
/// A multidimensional tensor managed by a compute backend.
/// Tensors are created and owned by an <see cref="IComputeBackend"/> — they cannot
/// be shared across backends. The backend controls where the data lives (managed
/// array, device memory, etc.).
/// </summary>
public interface ITensor : IDisposable
{
    /// <summary>
    /// Tensor name (e.g. "blk.0.attn_q.weight").
    /// </summary>
    string Name { get; }

    /// <summary>
    /// The quantization / numeric type of the stored data.
    /// </summary>
    GgmlType Type { get; }

    /// <summary>
    /// Shape of the tensor. Length is the number of dimensions.
    /// </summary>
    ReadOnlySpan<long> Dimensions { get; }

    /// <summary>
    /// Total number of logical elements (product of all dimensions).
    /// </summary>
    long ElementCount { get; }

    /// <summary>
    /// Total size of the raw data in bytes.
    /// </summary>
    long ByteSize { get; }

    /// <summary>
    /// Copy raw data into this tensor from a host buffer.
    /// The span length must match <see cref="ByteSize"/>.
    /// </summary>
    void CopyFrom(ReadOnlySpan<byte> data);

    /// <summary>
    /// Dequantize (or copy if already FP32) the tensor contents into a float span.
    /// The destination length must be at least <see cref="ElementCount"/>.
    /// </summary>
    void DequantizeTo(Span<float> destination);

    /// <summary>
    /// Get a writable float span over the tensor data. Only valid for F32 tensors.
    /// </summary>
    Span<float> AsFloatSpan();
}
