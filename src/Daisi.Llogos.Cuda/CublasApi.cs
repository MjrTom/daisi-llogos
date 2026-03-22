using System.Runtime.InteropServices;

namespace Daisi.Llogos.Cuda;

/// <summary>
/// P/Invoke bindings to cuBLAS (cublas64_*.dll, part of CUDA Toolkit).
/// Used for FP16/FP32 GEMV operations that benefit from tensor cores.
/// </summary>
internal static partial class CublasApi
{
    private const string Lib = "cublas64_12";

    [LibraryImport(Lib, EntryPoint = "cublasCreate_v2")]
    internal static partial int Create(out nint handle);

    [LibraryImport(Lib, EntryPoint = "cublasDestroy_v2")]
    internal static partial int Destroy(nint handle);

    [LibraryImport(Lib, EntryPoint = "cublasSetStream_v2")]
    internal static partial int SetStream(nint handle, nint stream);

    /// <summary>
    /// General matrix multiply: C = alpha * op(A) * op(B) + beta * C
    /// Supports mixed precision via datatype parameters.
    /// </summary>
    [LibraryImport(Lib, EntryPoint = "cublasGemmEx")]
    internal static unsafe partial int GemmEx(
        nint handle,
        int transa, int transb,
        int m, int n, int k,
        void* alpha,
        ulong A, int Atype, int lda,
        ulong B, int Btype, int ldb,
        void* beta,
        ulong C, int Ctype, int ldc,
        int computeType, int algo);

    /// <summary>
    /// Single-precision GEMV: y = alpha * op(A) * x + beta * y
    /// </summary>
    [LibraryImport(Lib, EntryPoint = "cublasSgemv_v2")]
    internal static unsafe partial int Sgemv(
        nint handle,
        int trans,
        int m, int n,
        float* alpha,
        ulong A, int lda,
        ulong x, int incx,
        float* beta,
        ulong y, int incy);

    /// <summary>
    /// Single-precision GEMM: C = alpha * op(A) * op(B) + beta * C
    /// </summary>
    [LibraryImport(Lib, EntryPoint = "cublasSgemm_v2")]
    internal static unsafe partial int Sgemm(
        nint handle,
        int transa, int transb,
        int m, int n, int k,
        float* alpha,
        ulong A, int lda,
        ulong B, int ldb,
        float* beta,
        ulong C, int ldc);

    // cuBLAS enums
    internal const int CUBLAS_OP_N = 0;
    internal const int CUBLAS_OP_T = 1;
    internal const int CUDA_R_16F = 2;
    internal const int CUDA_R_32F = 0;
    internal const int CUBLAS_GEMM_DEFAULT = -1;
    internal const int CUBLAS_COMPUTE_32F = 68;

    [LibraryImport(Lib, EntryPoint = "cublasSetMathMode")]
    internal static partial int SetMathMode(nint handle, int mode);

    internal const int CUBLAS_DEFAULT_MATH = 0;
    internal const int CUBLAS_TF32_TENSOR_OP_MATH = 3;

    internal static void Check(int status, string op)
    {
        if (status != 0)
            throw new CudaException($"cuBLAS {op} failed: status {status}");
    }
}
