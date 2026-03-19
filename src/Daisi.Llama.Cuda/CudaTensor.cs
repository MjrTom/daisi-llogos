using System.Runtime.InteropServices;
using Daisi.Llama.Gguf;

namespace Daisi.Llama.Cuda;

/// <summary>
/// GPU-backed tensor. Data lives in CUDA device memory.
/// Supports H2D upload on creation and D2H download for reading results.
/// </summary>
public sealed class CudaTensor : ITensor
{
    private readonly long[] _dimensions;
    private CudaDeviceMemory? _memory;
    private CudaPinnedMemory? _pinnedMemory;
    private bool _disposed;

    internal CudaTensor(string name, GgmlType type, ReadOnlySpan<long> dimensions, bool pinned = false)
    {
        Name = name;
        Type = type;
        IsPinned = pinned;
        _dimensions = dimensions.ToArray();
        ElementCount = ComputeElementCount(dimensions);
        ByteSize = (long)GgmlTypeInfo.ByteSize(type, (ulong)ElementCount);
        if (pinned)
            _pinnedMemory = new CudaPinnedMemory((ulong)ByteSize);
        else
            _memory = new CudaDeviceMemory((ulong)ByteSize);
    }

    internal CudaTensor(string name, GgmlType type, ReadOnlySpan<long> dimensions, ReadOnlySpan<byte> data)
        : this(name, type, dimensions, pinned: false)
    {
        _memory!.CopyFromHost(data);
    }

    /// <inheritdoc />
    public string Name { get; }

    /// <inheritdoc />
    public GgmlType Type { get; }

    /// <inheritdoc />
    public ReadOnlySpan<long> Dimensions => _dimensions;

    /// <inheritdoc />
    public long ElementCount { get; }

    /// <inheritdoc />
    public long ByteSize { get; }

    /// <summary>Whether this tensor is backed by pinned host memory.</summary>
    internal bool IsPinned { get; }

    /// <summary>
    /// Device pointer for kernel parameters. Works for both device and pinned memory.
    /// </summary>
    internal ulong DevicePtr => IsPinned
        ? (_pinnedMemory?.DevicePtr ?? 0)
        : (_memory?.DevicePtr ?? 0);

    /// <summary>
    /// The underlying device memory allocation. Only valid for non-pinned tensors.
    /// </summary>
    internal CudaDeviceMemory Memory => _memory ?? throw new ObjectDisposedException(Name);

    /// <inheritdoc />
    public void CopyFrom(ReadOnlySpan<byte> data)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (data.Length != ByteSize)
            throw new ArgumentException($"Data length {data.Length} does not match tensor byte size {ByteSize}.");
        if (IsPinned)
            _pinnedMemory!.CopyFromHost(data);
        else
            _memory!.CopyFromHost(data);
    }

    /// <inheritdoc />
    public void DequantizeTo(Span<float> destination)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (destination.Length < ElementCount)
            throw new ArgumentException($"Destination too small.");

        // Sync any active CUDA stream before D2H transfer
        ActiveStream?.Synchronize();

        if (Type == GgmlType.F32)
        {
            if (IsPinned)
                _pinnedMemory!.CopyToHost(destination.Slice(0, (int)ElementCount));
            else
                _memory!.CopyToHost(destination.Slice(0, (int)ElementCount));
        }
        else
        {
            // Download raw bytes then dequantize on CPU (fallback)
            var raw = new byte[ByteSize];
            if (IsPinned)
                _pinnedMemory!.CopyToHost(raw.AsSpan());
            else
                _memory!.CopyToHost(raw.AsSpan());
            var cpuTensor = new Cpu.CpuTensor(Name + "_tmp", Type, _dimensions, raw);
            cpuTensor.DequantizeTo(destination);
            cpuTensor.Dispose();
        }
    }

    /// <inheritdoc />
    public Span<float> AsFloatSpan()
    {
        // For GPU tensors, we need to download to a managed buffer.
        // This is intentionally not supported for performance reasons —
        // the ForwardPass should use kernel launches instead of reading spans.
        throw new InvalidOperationException(
            "Cannot get float span for GPU tensor. Use kernel operations or DequantizeTo instead.");
    }

    /// <summary>
    /// Set the stream to synchronize before D2H transfers.
    /// </summary>
    internal static CudaStream? ActiveStream { get; set; }

    /// <summary>
    /// Download F32 tensor data to a host buffer.
    /// </summary>
    internal void DownloadTo(Span<float> destination)
    {
        if (Type != GgmlType.F32)
            throw new InvalidOperationException($"DownloadTo only valid for F32 tensors, got {Type}.");
        _memory!.CopyToHost(destination);
    }

    /// <summary>
    /// Upload F32 data from a host buffer.
    /// </summary>
    internal unsafe void UploadFrom(ReadOnlySpan<float> data)
    {
        if (Type != GgmlType.F32)
            throw new InvalidOperationException($"UploadFrom only valid for F32 tensors, got {Type}.");
        fixed (float* ptr = data)
            CudaApi.Check(
                CudaApi.MemcpyHtoD(DevicePtr, ptr, (ulong)(data.Length * sizeof(float))),
                "cuMemcpyHtoD");
    }

    /// <inheritdoc />
    public void Dispose()
    {
        if (!_disposed)
        {
            _memory?.Dispose();
            _memory = null;
            _pinnedMemory?.Dispose();
            _pinnedMemory = null;
            _disposed = true;
        }
    }

    private static long ComputeElementCount(ReadOnlySpan<long> dims)
    {
        long count = 1;
        for (int i = 0; i < dims.Length; i++)
            count *= dims[i];
        return count;
    }
}
