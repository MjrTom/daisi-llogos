namespace Daisi.Llogos.Cuda;

/// <summary>
/// Wraps a CUDA pinned host memory allocation. Accessible by GPU kernels
/// via a mapped device pointer (reads go over PCIe).
/// </summary>
internal sealed class CudaPinnedMemory : IDisposable
{
    private nint _hostPtr;
    private ulong _devicePtr;
    private bool _disposed;

    public ulong ByteSize { get; }

    public CudaPinnedMemory(ulong byteSize)
    {
        ByteSize = byteSize;
        CudaApi.Check(CudaApi.MemAllocHost(out _hostPtr, byteSize), "cuMemAllocHost");
        CudaApi.Check(CudaApi.MemHostGetDevicePointer(out _devicePtr, _hostPtr, 0), "cuMemHostGetDevicePointer");
    }

    /// <summary>
    /// Device pointer for use in kernel parameters (mapped to host memory).
    /// </summary>
    public ulong DevicePtr => _devicePtr;

    /// <summary>
    /// Host pointer for direct CPU access.
    /// </summary>
    public nint HostPtr => _hostPtr;

    /// <summary>
    /// Copy data from a managed buffer into pinned host memory.
    /// </summary>
    public unsafe void CopyFromHost(ReadOnlySpan<byte> data)
    {
        if ((ulong)data.Length > ByteSize)
            throw new ArgumentException($"Data length {data.Length} exceeds allocation size {ByteSize}");
        fixed (byte* src = data)
            Buffer.MemoryCopy(src, (void*)_hostPtr, (long)ByteSize, data.Length);
    }

    /// <summary>
    /// Copy data from pinned host memory to a managed buffer.
    /// </summary>
    public unsafe void CopyToHost(Span<byte> destination)
    {
        if ((ulong)destination.Length > ByteSize)
            throw new ArgumentException($"Destination length {destination.Length} exceeds allocation size {ByteSize}");
        fixed (byte* dst = destination)
            Buffer.MemoryCopy((void*)_hostPtr, dst, destination.Length, destination.Length);
    }

    /// <summary>
    /// Copy data from pinned host memory to a float span.
    /// </summary>
    public unsafe void CopyToHost(Span<float> destination)
    {
        ulong byteCount = (ulong)(destination.Length * sizeof(float));
        if (byteCount > ByteSize)
            throw new ArgumentException($"Destination byte size {byteCount} exceeds allocation size {ByteSize}");
        fixed (float* dst = destination)
            Buffer.MemoryCopy((void*)_hostPtr, dst, (long)byteCount, (long)byteCount);
    }

    public void Dispose()
    {
        if (!_disposed && _hostPtr != 0)
        {
            CudaApi.MemFreeHost(_hostPtr);
            _hostPtr = 0;
            _devicePtr = 0;
            _disposed = true;
        }
    }
}
