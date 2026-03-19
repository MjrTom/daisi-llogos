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
}
