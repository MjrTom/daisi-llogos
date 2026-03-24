using System.Runtime.InteropServices;

namespace Daisi.Llogos.Cuda;

/// <summary>
/// P/Invoke bindings to cuBLASLt (cublasLt64_12.dll).
/// cuBLASLt provides better algorithm selection and workspace optimization
/// than cuBLAS for specific matrix dimensions.
/// </summary>
internal static partial class CublasLtApi
{
    private const string Lib = "cublasLt64_12";

    [LibraryImport(Lib, EntryPoint = "cublasLtCreate")]
    internal static partial int Create(out nint handle);

    [LibraryImport(Lib, EntryPoint = "cublasLtDestroy")]
    internal static partial int Destroy(nint handle);

    // ── Matrix layout ─────────────────────────────────────────────────────

    [LibraryImport(Lib, EntryPoint = "cublasLtMatrixLayoutCreate")]
    internal static partial int MatrixLayoutCreate(out nint layout, int dataType, ulong rows, ulong cols, long ld);

    [LibraryImport(Lib, EntryPoint = "cublasLtMatrixLayoutDestroy")]
    internal static partial int MatrixLayoutDestroy(nint layout);

    // ── Matmul descriptor ─────────────────────────────────────────────────

    [LibraryImport(Lib, EntryPoint = "cublasLtMatmulDescCreate")]
    internal static partial int MatmulDescCreate(out nint desc, int computeType, int scaleType);

    [LibraryImport(Lib, EntryPoint = "cublasLtMatmulDescDestroy")]
    internal static partial int MatmulDescDestroy(nint desc);

    [LibraryImport(Lib, EntryPoint = "cublasLtMatmulDescSetAttribute")]
    internal static unsafe partial int MatmulDescSetAttribute(nint desc, int attr, void* value, ulong sizeInBytes);

    // ── Matmul preference (algorithm selection) ───────────────────────────

    [LibraryImport(Lib, EntryPoint = "cublasLtMatmulPreferenceCreate")]
    internal static partial int MatmulPreferenceCreate(out nint pref);

    [LibraryImport(Lib, EntryPoint = "cublasLtMatmulPreferenceDestroy")]
    internal static partial int MatmulPreferenceDestroy(nint pref);

    [LibraryImport(Lib, EntryPoint = "cublasLtMatmulPreferenceSetAttribute")]
    internal static unsafe partial int MatmulPreferenceSetAttribute(nint pref, int attr, void* value, ulong sizeInBytes);

    // ── Algorithm selection ───────────────────────────────────────────────

    /// <summary>
    /// Find best algorithm for the given matmul configuration.
    /// heuristicResult is an array of CublasLtMatmulHeuristicResult structs.
    /// </summary>
    [LibraryImport(Lib, EntryPoint = "cublasLtMatmulAlgoGetHeuristic")]
    internal static unsafe partial int MatmulAlgoGetHeuristic(
        nint handle, nint matmulDesc,
        nint layoutA, nint layoutB, nint layoutC, nint layoutD,
        nint preference, int requestedAlgoCount,
        nint heuristicResults, out int returnedAlgoCount);

    // ── Matmul execution ──────────────────────────────────────────────────

    [LibraryImport(Lib, EntryPoint = "cublasLtMatmul")]
    internal static unsafe partial int Matmul(
        nint handle, nint matmulDesc,
        void* alpha,
        ulong A, nint layoutA,
        ulong B, nint layoutB,
        void* beta,
        ulong C, nint layoutC,
        ulong D, nint layoutD,
        void* algo,          // from heuristic result, or null for default
        ulong workspace, ulong workspaceSize,
        nint stream);

    // ── Attribute enums ───────────────────────────────────────────────────

    internal const int CUBLASLT_MATMUL_DESC_TRANSA = 0;
    internal const int CUBLASLT_MATMUL_DESC_TRANSB = 1;

    internal const int CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES = 1;

    // Heuristic result struct (used with MatmulAlgoGetHeuristic)
    // The full struct is 1056 bytes but we only need the algo (first 512 bytes) and wavesCount
    [StructLayout(LayoutKind.Sequential)]
    internal struct MatmulHeuristicResult
    {
        // cublasLtMatmulAlgo_t is 512 bytes
        internal unsafe fixed byte algo[512];
        internal ulong workspaceSize;
        internal int status; // CUBLAS_STATUS_SUCCESS = 0
        internal float wavesCount;
        internal unsafe fixed int reserved[4];
    }

    internal static void Check(int status, string op)
    {
        if (status != 0)
            throw new CudaException($"cuBLASLt {op} failed: status {status}");
    }
}
