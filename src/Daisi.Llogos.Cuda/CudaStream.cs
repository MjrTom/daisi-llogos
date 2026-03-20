namespace Daisi.Llogos.Cuda;

/// <summary>
/// Wraps a CUDA stream for asynchronous kernel execution.
/// </summary>
internal sealed class CudaStream : IDisposable
{
    private nint _handle;
    private bool _disposed;

    public CudaStream()
    {
        CudaApi.Check(CudaApi.StreamCreate(out _handle, 0), "cuStreamCreate");
    }

    public nint Handle => _handle;

    /// <summary>
    /// Launch a kernel on this stream.
    /// </summary>
    public unsafe void Launch(nint function, uint gridX, uint gridY, uint gridZ,
                              uint blockX, uint blockY, uint blockZ,
                              uint sharedMem, nint* kernelParams)
    {
        CudaApi.Check(
            CudaApi.LaunchKernel(function, gridX, gridY, gridZ,
                                 blockX, blockY, blockZ,
                                 sharedMem, _handle, kernelParams, null),
            "cuLaunchKernel");
    }

    public void Synchronize()
    {
        CudaApi.Check(CudaApi.StreamSynchronize(_handle), "cuStreamSynchronize");
    }

    public void Dispose()
    {
        if (!_disposed && _handle != 0)
        {
            CudaApi.StreamDestroy(_handle);
            _handle = 0;
            _disposed = true;
        }
    }
}
