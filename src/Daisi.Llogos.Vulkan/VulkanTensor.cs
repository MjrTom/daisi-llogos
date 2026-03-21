using System.Runtime.InteropServices;
using Daisi.Llogos.Gguf;

namespace Daisi.Llogos.Vulkan;

/// <summary>
/// GPU-backed tensor. Data lives in Vulkan device memory with a host-visible staging buffer for transfers.
/// </summary>
public sealed class VulkanTensor : ITensor
{
    private readonly long[] _dimensions;
    private readonly VulkanDevice _vkDevice;
    internal VulkanBuffer DeviceBuffer { get; private set; }
    internal VulkanBuffer StagingBuffer { get; private set; }
    private bool _disposed;

    internal VulkanTensor(VulkanDevice vkDevice, string name, GgmlType type, ReadOnlySpan<long> dimensions)
    {
        _vkDevice = vkDevice;
        Name = name;
        Type = type;
        _dimensions = dimensions.ToArray();
        ElementCount = ComputeElementCount(dimensions);
        ByteSize = (long)GgmlTypeInfo.ByteSize(type, (ulong)ElementCount);

        ulong bufSize = (ulong)Math.Max(ByteSize, 4); // Vulkan requires non-zero buffers
        DeviceBuffer = new VulkanBuffer(vkDevice, bufSize, hostVisible: false, transferSrc: true, transferDst: true);
        StagingBuffer = new VulkanBuffer(vkDevice, bufSize, hostVisible: true, transferSrc: true, transferDst: true);
    }

    internal VulkanTensor(VulkanDevice vkDevice, string name, GgmlType type, ReadOnlySpan<long> dimensions, ReadOnlySpan<byte> data,
        bool isAlignedQ8_0 = false)
    {
        _vkDevice = vkDevice;
        Name = name;
        Type = type;
        IsAlignedQ8_0 = isAlignedQ8_0;
        _dimensions = dimensions.ToArray();
        ElementCount = ComputeElementCount(dimensions);
        ByteSize = isAlignedQ8_0 ? (ElementCount / 32) * 36 : (long)GgmlTypeInfo.ByteSize(type, (ulong)ElementCount);

        ulong bufSize = (ulong)Math.Max(ByteSize, 4);
        DeviceBuffer = new VulkanBuffer(vkDevice, bufSize, hostVisible: false, transferSrc: true, transferDst: true);
        StagingBuffer = new VulkanBuffer(vkDevice, bufSize, hostVisible: true, transferSrc: true, transferDst: true);
        UploadFromHost(data);
    }

    internal bool IsAlignedQ8_0 { get; }

    public string Name { get; }
    public GgmlType Type { get; }
    public ReadOnlySpan<long> Dimensions => _dimensions;
    public long ElementCount { get; }
    public long ByteSize { get; }

    /// <summary>
    /// Upload raw bytes from host to device via staging buffer.
    /// </summary>
    internal void UploadFromHost(ReadOnlySpan<byte> data)
    {
        StagingBuffer.Upload(data);
        ulong size = (ulong)data.Length;
        CopyStagingToDevice(size);
    }

    /// <summary>
    /// Download raw bytes from device to host via staging buffer.
    /// </summary>
    internal void DownloadToHost(Span<byte> destination)
    {
        ulong size = (ulong)destination.Length;
        CopyDeviceToStaging(size);
        StagingBuffer.Download(destination);
    }

    private void CopyStagingToDevice(ulong size)
    {
        _vkDevice.SubmitAndWait(cmd =>
        {
            unsafe
            {
                var region = new VkBufferCopy { size = size };
                VulkanApi.CmdCopyBuffer(cmd, StagingBuffer.Buffer, DeviceBuffer.Buffer, 1, &region);
            }
        });
    }

    private void CopyDeviceToStaging(ulong size)
    {
        _vkDevice.SubmitAndWait(cmd =>
        {
            unsafe
            {
                var region = new VkBufferCopy { size = size };
                VulkanApi.CmdCopyBuffer(cmd, DeviceBuffer.Buffer, StagingBuffer.Buffer, 1, &region);
            }
        });
    }

    public void CopyFrom(ReadOnlySpan<byte> data)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (data.Length != ByteSize)
            throw new ArgumentException($"Data length {data.Length} does not match tensor byte size {ByteSize}.");
        UploadFromHost(data);
    }

    public void DequantizeTo(Span<float> destination)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (destination.Length < ElementCount)
            throw new ArgumentException("Destination too small.");

        if (Type == GgmlType.F32)
        {
            var bytes = new byte[ByteSize];
            DownloadToHost(bytes);
            MemoryMarshal.Cast<byte, float>(bytes).Slice(0, (int)ElementCount).CopyTo(destination);
        }
        else
        {
            // Download raw bytes then dequantize on CPU (fallback)
            var raw = new byte[ByteSize];
            DownloadToHost(raw);
            using var cpuTensor = new Cpu.CpuTensor(Name + "_tmp", Type, _dimensions, raw.AsSpan());
            cpuTensor.DequantizeTo(destination);
        }
    }

    public Span<float> AsFloatSpan() =>
        throw new InvalidOperationException(
            "Cannot get float span for GPU tensor. Use kernel operations or DequantizeTo instead.");

    public void Dispose()
    {
        if (!_disposed)
        {
            DeviceBuffer?.Dispose();
            StagingBuffer?.Dispose();
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
