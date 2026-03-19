using System.Runtime.InteropServices;

namespace Daisi.Llama.Cuda;

/// <summary>
/// P/Invoke bindings for NVRTC (NVIDIA Runtime Compilation).
/// Compiles CUDA C++ source to PTX at runtime.
/// </summary>
internal static partial class NvrtcApi
{
    // NVRTC 12.x uses nvrtc64_120_0.dll
    private const string Lib = "nvrtc64_120_0";

    [LibraryImport(Lib, EntryPoint = "nvrtcCreateProgram")]
    internal static partial NvrtcResult CreateProgram(
        out nint prog,
        [MarshalAs(UnmanagedType.LPStr)] string src,
        [MarshalAs(UnmanagedType.LPStr)] string? name,
        int numHeaders,
        nint headers,    // const char**
        nint includeNames); // const char**

    [LibraryImport(Lib, EntryPoint = "nvrtcCompileProgram")]
    internal static partial NvrtcResult CompileProgram(
        nint prog, int numOptions,
        [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.LPStr)] string[]? options);

    [LibraryImport(Lib, EntryPoint = "nvrtcGetPTXSize")]
    internal static partial NvrtcResult GetPTXSize(nint prog, out nuint ptxSize);

    [LibraryImport(Lib, EntryPoint = "nvrtcGetPTX")]
    internal static partial NvrtcResult GetPTX(
        nint prog,
        [MarshalAs(UnmanagedType.LPArray, SizeParamIndex = 0)] byte[] ptx);

    [LibraryImport(Lib, EntryPoint = "nvrtcGetProgramLogSize")]
    internal static partial NvrtcResult GetProgramLogSize(nint prog, out nuint logSize);

    [LibraryImport(Lib, EntryPoint = "nvrtcGetProgramLog")]
    internal static partial NvrtcResult GetProgramLog(
        nint prog,
        [MarshalAs(UnmanagedType.LPArray, SizeParamIndex = 0)] byte[] log);

    [LibraryImport(Lib, EntryPoint = "nvrtcDestroyProgram")]
    internal static partial NvrtcResult DestroyProgram(ref nint prog);

    internal static void Check(NvrtcResult result, string operation)
    {
        if (result != NvrtcResult.Success)
            throw new CudaException($"NVRTC {operation} failed: {result} ({(int)result})");
    }
}

internal enum NvrtcResult
{
    Success = 0,
    OutOfMemory = 1,
    ProgramCreationFailure = 2,
    InvalidInput = 3,
    InvalidProgram = 4,
    CompilationError = 5,
    BuiltinOperationFailure = 6,
    NoNameExpressionsAfterCompilation = 7,
    NoLoweredNamesBeforeCompilation = 8,
    NameExpressionNotValid = 9,
    InternalError = 10,
}
