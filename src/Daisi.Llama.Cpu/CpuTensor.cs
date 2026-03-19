using System.Runtime.InteropServices;
using Daisi.Llama.Gguf;

namespace Daisi.Llama.Cpu;

/// <summary>
/// CPU-backed tensor. Stores data as a raw byte array and supports
/// dequantization to FP32 for quantized types.
/// </summary>
public sealed class CpuTensor : ITensor
{
    private readonly long[] _dimensions;
    private byte[] _data;
    private bool _disposed;

    internal CpuTensor(string name, GgmlType type, ReadOnlySpan<long> dimensions)
    {
        Name = name;
        Type = type;
        _dimensions = dimensions.ToArray();
        ElementCount = ComputeElementCount(dimensions);
        ByteSize = (long)GgmlTypeInfo.ByteSize(type, (ulong)ElementCount);
        _data = new byte[ByteSize];
    }

    internal CpuTensor(string name, GgmlType type, ReadOnlySpan<long> dimensions, ReadOnlySpan<byte> data)
        : this(name, type, dimensions)
    {
        data.CopyTo(_data);
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

    /// <summary>
    /// Direct read access to the underlying raw byte buffer.
    /// </summary>
    internal ReadOnlySpan<byte> RawData => _data;

    /// <summary>
    /// Direct read/write access to the data interpreted as FP32.
    /// Only valid when <see cref="Type"/> is <see cref="GgmlType.F32"/>.
    /// </summary>
    internal Span<float> AsFloatSpan()
    {
        if (Type != GgmlType.F32)
            throw new InvalidOperationException($"Cannot get float span for tensor of type {Type}.");
        return MemoryMarshal.Cast<byte, float>((Span<byte>)_data);
    }

    /// <inheritdoc />
    public void CopyFrom(ReadOnlySpan<byte> data)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (data.Length != ByteSize)
            throw new ArgumentException($"Data length {data.Length} does not match tensor byte size {ByteSize}.");
        data.CopyTo(_data);
    }

    /// <inheritdoc />
    public void DequantizeTo(Span<float> destination)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (destination.Length < ElementCount)
            throw new ArgumentException($"Destination length {destination.Length} is less than element count {ElementCount}.");

        switch (Type)
        {
            case GgmlType.F32:
                MemoryMarshal.Cast<byte, float>(_data).CopyTo(destination);
                break;
            case GgmlType.Q8_0:
                Dequantize.DequantizeQ8_0(_data, destination);
                break;
            case GgmlType.Q4_0:
                Dequantize.DequantizeQ4_0(_data, destination);
                break;
            case GgmlType.Q4_K:
                Dequantize.DequantizeQ4_K(_data, destination);
                break;
            default:
                throw new NotSupportedException($"Dequantization not implemented for {Type}.");
        }
    }

    /// <inheritdoc />
    public void Dispose()
    {
        if (!_disposed)
        {
            _data = null!;
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
