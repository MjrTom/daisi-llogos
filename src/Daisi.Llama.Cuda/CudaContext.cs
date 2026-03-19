namespace Daisi.Llama.Cuda;

/// <summary>
/// Manages a CUDA context on a specific GPU device.
/// </summary>
internal sealed class CudaContext : IDisposable
{
    private nint _handle;
    private bool _disposed;

    public int DeviceOrdinal { get; }
    public string DeviceName { get; }

    public CudaContext(int deviceOrdinal = 0)
    {
        DeviceOrdinal = deviceOrdinal;

        CudaApi.Check(CudaApi.Init(0), "cuInit");
        CudaApi.Check(CudaApi.DeviceGet(out int device, deviceOrdinal), "cuDeviceGet");

        var nameBuffer = new byte[256];
        CudaApi.Check(CudaApi.DeviceGetName(nameBuffer, 256, device), "cuDeviceGetName");
        DeviceName = System.Text.Encoding.ASCII.GetString(nameBuffer).TrimEnd('\0');

        CudaApi.Check(CudaApi.CtxCreate(out _handle, 0, device), "cuCtxCreate");
    }

    public nint Handle => _handle;

    public void MakeCurrent()
    {
        CudaApi.Check(CudaApi.CtxSetCurrent(_handle), "cuCtxSetCurrent");
    }

    public void Dispose()
    {
        if (!_disposed && _handle != 0)
        {
            CudaApi.CtxDestroy(_handle);
            _handle = 0;
            _disposed = true;
        }
    }
}
