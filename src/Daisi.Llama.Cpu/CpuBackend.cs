using Daisi.Llama.Gguf;

namespace Daisi.Llama.Cpu;

/// <summary>
/// CPU compute backend with SIMD-optimized tensor operations.
/// </summary>
public sealed class CpuBackend : IComputeBackend
{
    /// <inheritdoc />
    public string Name => "CPU";

    /// <inheritdoc />
    public ITensor CreateTensor(string name, GgmlType type, ReadOnlySpan<long> dimensions)
    {
        return new CpuTensor(name, type, dimensions);
    }

    /// <inheritdoc />
    public ITensor LoadTensor(string name, GgmlType type, ReadOnlySpan<long> dimensions, ReadOnlySpan<byte> data)
    {
        return new CpuTensor(name, type, dimensions, data);
    }

    /// <inheritdoc />
    public void Dispose()
    {
        // No unmanaged resources to clean up.
    }
}
