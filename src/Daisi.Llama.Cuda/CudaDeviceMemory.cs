namespace Daisi.Llama.Cuda;

/// <summary>
/// Wraps a CUDA device memory allocation. Provides H2D and D2H transfers.
/// </summary>
internal sealed class CudaDeviceMemory : IDisposable
{
    private ulong _devicePtr;
    private bool _disposed;

    public ulong ByteSize { get; }

    public CudaDeviceMemory(ulong byteSize)
    {
        ByteSize = byteSize;
        CudaApi.Check(CudaApi.MemAlloc(out _devicePtr, byteSize), "cuMemAlloc");
    }

    /// <summary>
    /// Device pointer for use in kernel parameters.
    /// </summary>
    public ulong DevicePtr => _devicePtr;

    /// <summary>
    /// Copy data from host to device.
    /// </summary>
    public unsafe void CopyFromHost(ReadOnlySpan<byte> data)
    {
        if ((ulong)data.Length > ByteSize)
            throw new ArgumentException($"Data length {data.Length} exceeds allocation size {ByteSize}");
        fixed (byte* ptr = data)
            CudaApi.Check(CudaApi.MemcpyHtoD(_devicePtr, ptr, (ulong)data.Length), "cuMemcpyHtoD");
    }

    /// <summary>
    /// Copy data from device to host.
    /// </summary>
    public unsafe void CopyToHost(Span<byte> destination)
    {
        if ((ulong)destination.Length > ByteSize)
            throw new ArgumentException($"Destination length {destination.Length} exceeds allocation size {ByteSize}");
        fixed (byte* ptr = destination)
            CudaApi.Check(CudaApi.MemcpyDtoH(ptr, _devicePtr, (ulong)destination.Length), "cuMemcpyDtoH");
    }

    /// <summary>
    /// Copy data from device to host (float span convenience).
    /// </summary>
    public unsafe void CopyToHost(Span<float> destination)
    {
        ulong byteCount = (ulong)(destination.Length * sizeof(float));
        if (byteCount > ByteSize)
            throw new ArgumentException($"Destination byte size {byteCount} exceeds allocation size {ByteSize}");
        fixed (float* ptr = destination)
            CudaApi.Check(CudaApi.MemcpyDtoH(ptr, _devicePtr, byteCount), "cuMemcpyDtoH");
    }

    public void Dispose()
    {
        if (!_disposed && _devicePtr != 0)
        {
            CudaApi.MemFree(_devicePtr);
            _devicePtr = 0;
            _disposed = true;
        }
    }
}
