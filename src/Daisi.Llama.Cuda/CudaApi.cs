using System.Runtime.InteropServices;

namespace Daisi.Llama.Cuda;

/// <summary>
/// Raw P/Invoke bindings to the CUDA Driver API (nvcuda.dll).
/// </summary>
internal static partial class CudaApi
{
    private const string Lib = "nvcuda";

    // ── Initialization ───────────────────────────────────────────────────────

    [LibraryImport(Lib, EntryPoint = "cuInit")]
    internal static partial CuResult Init(uint flags);

    // ── Device ───────────────────────────────────────────────────────────────

    [LibraryImport(Lib, EntryPoint = "cuDeviceGet")]
    internal static partial CuResult DeviceGet(out int device, int ordinal);

    [LibraryImport(Lib, EntryPoint = "cuDeviceGetName")]
    internal static partial CuResult DeviceGetName(
        [MarshalAs(UnmanagedType.LPArray, SizeParamIndex = 1)] byte[] name,
        int len, int device);

    [LibraryImport(Lib, EntryPoint = "cuDeviceGetAttribute")]
    internal static partial CuResult DeviceGetAttribute(out int value, int attribute, int device);

    // ── Context ──────────────────────────────────────────────────────────────

    [LibraryImport(Lib, EntryPoint = "cuCtxCreate_v2")]
    internal static partial CuResult CtxCreate(out nint ctx, uint flags, int device);

    [LibraryImport(Lib, EntryPoint = "cuCtxDestroy_v2")]
    internal static partial CuResult CtxDestroy(nint ctx);

    [LibraryImport(Lib, EntryPoint = "cuCtxSetCurrent")]
    internal static partial CuResult CtxSetCurrent(nint ctx);

    // ── Module ───────────────────────────────────────────────────────────────

    [LibraryImport(Lib, EntryPoint = "cuModuleLoadDataEx")]
    internal static partial CuResult ModuleLoadDataEx(
        out nint module, [MarshalAs(UnmanagedType.LPArray)] byte[] image,
        uint numOptions, nint options, nint optionValues);

    [LibraryImport(Lib, EntryPoint = "cuModuleUnload")]
    internal static partial CuResult ModuleUnload(nint module);

    [LibraryImport(Lib, EntryPoint = "cuModuleGetFunction")]
    internal static partial CuResult ModuleGetFunction(
        out nint function, nint module,
        [MarshalAs(UnmanagedType.LPStr)] string name);

    // ── Memory ───────────────────────────────────────────────────────────────

    [LibraryImport(Lib, EntryPoint = "cuMemAlloc_v2")]
    internal static partial CuResult MemAlloc(out ulong dptr, ulong bytesize);

    [LibraryImport(Lib, EntryPoint = "cuMemFree_v2")]
    internal static partial CuResult MemFree(ulong dptr);

    [LibraryImport(Lib, EntryPoint = "cuMemcpyHtoD_v2")]
    internal static unsafe partial CuResult MemcpyHtoD(ulong dstDevice, void* srcHost, ulong byteCount);

    [LibraryImport(Lib, EntryPoint = "cuMemcpyDtoH_v2")]
    internal static unsafe partial CuResult MemcpyDtoH(void* dstHost, ulong srcDevice, ulong byteCount);

    // ── Stream ───────────────────────────────────────────────────────────────

    [LibraryImport(Lib, EntryPoint = "cuStreamCreate")]
    internal static partial CuResult StreamCreate(out nint stream, uint flags);

    [LibraryImport(Lib, EntryPoint = "cuStreamDestroy_v2")]
    internal static partial CuResult StreamDestroy(nint stream);

    [LibraryImport(Lib, EntryPoint = "cuStreamSynchronize")]
    internal static partial CuResult StreamSynchronize(nint stream);

    // ── Kernel launch ────────────────────────────────────────────────────────

    [LibraryImport(Lib, EntryPoint = "cuLaunchKernel")]
    internal static unsafe partial CuResult LaunchKernel(
        nint function,
        uint gridDimX, uint gridDimY, uint gridDimZ,
        uint blockDimX, uint blockDimY, uint blockDimZ,
        uint sharedMemBytes, nint stream,
        nint* kernelParams, nint* extra);

    // ── Helpers ──────────────────────────────────────────────────────────────

    internal static void Check(CuResult result, string operation)
    {
        if (result != CuResult.Success)
            throw new CudaException($"CUDA {operation} failed: {result} ({(int)result})");
    }
}

internal enum CuResult
{
    Success = 0,
    ErrorInvalidValue = 1,
    ErrorOutOfMemory = 2,
    ErrorNotInitialized = 3,
    ErrorDeinitialized = 4,
    ErrorNoDevice = 100,
    ErrorInvalidDevice = 101,
    ErrorInvalidContext = 201,
    ErrorNotFound = 500,
}

public sealed class CudaException : Exception
{
    public CudaException(string message) : base(message) { }
}
