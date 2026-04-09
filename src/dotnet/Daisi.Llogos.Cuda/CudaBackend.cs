using Daisi.Llogos.Gguf;

namespace Daisi.Llogos.Cuda;

/// <summary>
/// CUDA compute backend. Implements IComputeBackend using GPU kernels via P/Invoke
/// to the CUDA Driver API. Kernels are JIT-compiled from embedded PTX at initialization.
/// </summary>
public sealed class CudaBackend : IComputeBackend
{
    private readonly CudaContext _context;
    private readonly CudaModule _elementwiseModule;
    private readonly CudaModule _matmulModule;
    private readonly CudaModule _compositeModule;
    private readonly CudaStream _stream;
    private readonly nint _cublasHandle;
    private CudaDeviceMemory? _q8_1Scratch; // scratch for quantized activation
    private int _q8_1ScratchK; // K dimension of current scratch
    private ulong _q8_1CachedInputPtr; // device ptr of last quantized activation (for reuse)
    private int _q8_1CacheGeneration; // incremented by non-matmul ops to invalidate cache
    private int _q8_1CachedGeneration; // generation when cache was last written
    private bool _q8_1FusedReady; // true when Q8_1 was pre-computed by fused RmsNorm
    private bool _hasQ4_0Weights; // true when model contains Q4_0 weight tensors
    private bool _hasQ8_0Weights; // true when model contains Q8_0 weight tensors (enables dp4a path)
    private bool _disposed;

    private const int BlockSize = 256;

    // ── CUDA Graph capture state ─────────────────────────────────────────────
    private bool _capturing;          // true while stream is in capture mode
    private nint _graphExec;          // reusable graph executable (0 = none)
    private bool _graphEnabled = true;

    /// <summary>Disable CUDA graph capture (needed for TurboQuant which uses different kernel topology).</summary>
    public void DisableGraphCapture() { _graphEnabled = false; }

    /// <summary>
    /// Invalidate all cached weight-derived state. Must be called after overwriting
    /// GPU weight tensor data (e.g. pipeline layer swap) so that fused RmsNorm Q8_1,
    /// FP16 dequant caches, and activation quantization caches are not reused with stale data.
    /// </summary>
    public void InvalidateWeightCache()
    {
        _q8_1CacheGeneration++;
        _q8_1CachedInputPtr = 0;
        _q8_1CachedGeneration = 0;
        _q8_1FusedReady = false;

        // Clear FP16 weight cache — shared tensor DevicePtrs get reused with different data
        foreach (var buf in _f16WeightCache.Values) buf.Dispose();
        _f16WeightCache.Clear();
        _f16CacheBytes = 0;
        _f16WeightCacheEnabled = true;

        // Clear activation caches
        _f16ActCachedSrcPtr = 0;
        _f16ActCachedSize = 0;
    }

    public CudaBackend(int deviceOrdinal = 0)
    {
        _context = new CudaContext(deviceOrdinal);
        _stream = new CudaStream();

        // Initialize cuBLAS
        CublasApi.Check(CublasApi.Create(out _cublasHandle), "cublasCreate");
        CublasApi.Check(CublasApi.SetStream(_cublasHandle, _stream.Handle), "cublasSetStream");

        // Load kernels with architecture-specific compilation for best codegen
        var archOpts = new[] { $"--gpu-architecture=compute_{_context.ComputeCapabilityMajor}{_context.ComputeCapabilityMinor}" };
        _elementwiseModule = CudaModule.FromEmbeddedResource("elementwise.cu", archOpts);
        _matmulModule = CudaModule.FromEmbeddedResource("dequant_matmul.cu", archOpts);
        _compositeModule = CudaModule.FromEmbeddedResource("composite_ops.cu", archOpts);

        // Set the active stream so D2H transfers sync properly
        CudaTensor.ActiveStream = _stream;
    }

    /// <inheritdoc />
    public string Name => $"CUDA ({_context.DeviceName})";
    public bool SupportsBatchedOps => true;

    /// <summary>GPU compute capability major version (e.g. 8 for Ampere, 12 for Blackwell).</summary>
    public int ComputeCapabilityMajor => _context.ComputeCapabilityMajor;

    /// <summary>GPU compute capability minor version.</summary>
    public int ComputeCapabilityMinor => _context.ComputeCapabilityMinor;

    /// <summary>
    /// Launch a CUDA kernel on the backend's main stream (inside graph capture).
    /// </summary>
    public unsafe void LaunchKernel(nint function, uint gridX, uint gridY, uint gridZ,
        uint blockX, uint blockY, uint blockZ, uint sharedMem, nint* args)
    {
        _stream.Launch(function, gridX, gridY, gridZ, blockX, blockY, blockZ, sharedMem, args);
    }

    /// <summary>Begin recording operations into a CUDA graph (if enabled).</summary>
    public void BeginCommands()
    {
        EnsureContext();
        _q8_1CacheGeneration++; // invalidate Q8_1 cache for new token
        if (!_graphEnabled) return;
        // Start stream capture — all subsequent launches are recorded, not executed
        var result = CudaApi.StreamBeginCapture(_stream.Handle, 0); // 0 = CU_STREAM_CAPTURE_MODE_GLOBAL
        if (result == CuResult.Success)
            _capturing = true;
        // If capture fails (e.g., stream busy), fall through to normal execution
    }

    /// <summary>End recording, instantiate/update graph, and launch.</summary>
    public unsafe void FlushCommands()
    {
        if (!_capturing) return;
        _capturing = false;

        // End capture — get the recorded graph
        CudaApi.Check(CudaApi.StreamEndCapture(_stream.Handle, out nint graph), "cuStreamEndCapture");

        if (_graphExec != 0)
        {
            // Try to update existing executable with new graph (fast path — same topology, new params)
            var updateResult = CudaApi.GraphExecUpdate(_graphExec, graph, null);
            if (updateResult != CuResult.Success)
            {
                // Topology changed — re-instantiate
                CudaApi.GraphExecDestroy(_graphExec);
                CudaApi.Check(CudaApi.GraphInstantiate(out _graphExec, graph, 0, 0, 0), "cuGraphInstantiate");
            }
        }
        else
        {
            // First capture — instantiate
            CudaApi.Check(CudaApi.GraphInstantiate(out _graphExec, graph, 0, 0, 0), "cuGraphInstantiate");
        }

        CudaApi.GraphDestroy(graph); // graph object no longer needed after instantiation

        // Launch the graph — executes all captured operations in one submission
        CudaApi.Check(CudaApi.GraphLaunch(_graphExec, _stream.Handle), "cuGraphLaunch");
    }

    /// <summary>
    /// Ensure the CUDA context is current on the calling thread.
    /// Required because minion tasks run on threadpool threads that may not
    /// have the context bound. The GpuInferenceGate serializes access, so
    /// this is safe — only one thread calls CUDA at a time.
    /// </summary>
    private void EnsureContext() => _context.MakeCurrent();

    /// <summary>Bind the CUDA context to the calling thread. Required before cuMemFree from non-owner threads.</summary>
    public void EnsureCudaContext() => _context.MakeCurrent();

    /// <summary>
    /// Batch MatMul for M > 1 (prefill, speculative decode verification).
    /// Dequantizes quantized weights to FP32 temp buffer, then uses cuBLAS SGEMM.
    /// </summary>
    private unsafe void BatchMatMul(CudaTensor outT, CudaTensor aT, CudaTensor bT, int M, int K, int N)
    {
        ulong outPtr = outT.DevicePtr;
        ulong aPtr = aT.DevicePtr;
        ulong bPtr = bT.DevicePtr;

        // ── Quantized → FP16 dequant + Tensor Core GemmEx ──
        if (bT.Type != Gguf.GgmlType.F32 && bT.Type != Gguf.GgmlType.F16)
        {
            long f16WBytes = (long)K * N * 2;
            ulong f16WPtr;

            // Check persistent FP16 weight cache (avoids re-dequant on repeat prefills)
            if (_f16WeightCache.TryGetValue(bPtr, out var cached))
            {
                f16WPtr = cached.DevicePtr;
            }
            else
            {
                // Dequantize weight → FP16 (temp buffer or cached)
                CudaDeviceMemory? newBuf = null;
                bool canCache = _f16WeightCacheEnabled && TryAllocF16Cache(f16WBytes, out newBuf);
                if (canCache)
                {
                    f16WPtr = newBuf!.DevicePtr;
                    _f16WeightCache[bPtr] = newBuf;
                    _f16CacheBytes += f16WBytes;
                }
                else
                {
                    // Fallback: use temp buffer
                    if (_batchF16WtBuf == null || _batchF16WtBufSize < f16WBytes)
                    {
                        _batchF16WtBuf?.Dispose();
                        _batchF16WtBuf = new CudaDeviceMemory((ulong)f16WBytes);
                        _batchF16WtBufSize = f16WBytes;
                    }
                    f16WPtr = _batchF16WtBuf.DevicePtr;
                    _f16CacheMisses++;
                    // Disable further cache attempts after first failure (avoid repeated MemGetInfo calls)
                    _f16WeightCacheEnabled = false;
                }

                var dqFunc = _elementwiseModule.GetFunction("dequant_to_f16");
                int totalElements = K * N;
                int blockSizeQ = Gguf.GgmlTypeInfo.BlockSize(bT.Type);
                int typeTag = (int)bT.Type;
                int isAligned = (bT.IsAlignedQ8_0 || bT.IsAlignedQ4_0) ? 1 : 0;
                uint dqGrid = (uint)((totalElements + BlockSize - 1) / BlockSize);
                nint* dArgs = stackalloc nint[6];
                dArgs[0] = (nint)(&f16WPtr);
                dArgs[1] = (nint)(&bPtr);
                dArgs[2] = (nint)(&totalElements);
                dArgs[3] = (nint)(&typeTag);
                dArgs[4] = (nint)(&blockSizeQ);
                dArgs[5] = (nint)(&isAligned);
                _stream.Launch(dqFunc, dqGrid, 1, 1, (uint)BlockSize, 1, 1, 0, dArgs);
            }

            // Convert activation FP32 → FP16 (cached: same normOut feeds Q/K/V and gate/up)
            long f16ABytes = (long)M * K * 2;
            ulong f16APtr;
            if (_f16ActCachedSrcPtr == aPtr && _f16ActCachedSize == f16ABytes && _batchF16ActBuf != null)
            {
                f16APtr = _batchF16ActBuf.DevicePtr;
            }
            else
            {
                if (_batchF16ActBuf == null || _batchF16ActBufSize < f16ABytes)
                {
                    _batchF16ActBuf?.Dispose();
                    _batchF16ActBuf = new CudaDeviceMemory((ulong)f16ABytes);
                    _batchF16ActBufSize = f16ABytes;
                }
                f16APtr = _batchF16ActBuf.DevicePtr;

                var cvtFunc = _elementwiseModule.GetFunction("convert_f32_to_f16");
                int aElements = M * K;
                uint cvtGrid = (uint)((aElements + BlockSize - 1) / BlockSize);
                nint* cArgs = stackalloc nint[3];
                cArgs[0] = (nint)(&f16APtr);
                cArgs[1] = (nint)(&aPtr);
                cArgs[2] = (nint)(&aElements);
                _stream.Launch(cvtFunc, cvtGrid, 1, 1, (uint)BlockSize, 1, 1, 0, cArgs);
                _f16ActCachedSrcPtr = aPtr;
                _f16ActCachedSize = f16ABytes;
            }

            // GemmEx: C(FP32) = B^T(FP16) × A(FP16) using tensor cores
            // Column-major: C(N×M) = B^T(N×K) × A(K×M)
            {
            float a16 = 1.0f, b16 = 0.0f;
            CublasApi.Check(CublasApi.GemmEx(_cublasHandle,
                CublasApi.CUBLAS_OP_T,  // B^T
                CublasApi.CUBLAS_OP_N,  // A as-is
                N, M, K,               // column-major dims
                &a16,
                f16WPtr, CublasApi.CUDA_R_16F, K,   // B: [N×K] → ldb = K
                f16APtr, CublasApi.CUDA_R_16F, K,   // A: [M×K] → lda = K
                &b16,
                outPtr, CublasApi.CUDA_R_32F, N,     // C: [M×N] → ldc = N
                CublasApi.CUBLAS_COMPUTE_32F, CublasApi.CUBLAS_GEMM_DEFAULT),
                "cublasGemmEx");
            }
            return;
        }

        ulong bF32Ptr;
        bool needsDequant = bT.Type != Gguf.GgmlType.F32;

        if (needsDequant)
        {
            // Dequantize weight to temporary FP32 buffer
            long f32Bytes = (long)K * N * sizeof(float);
            if (_batchDequantBuf == null || _batchDequantBufSize < f32Bytes)
            {
                _batchDequantBuf?.Dispose();
                _batchDequantBuf = new CudaDeviceMemory((ulong)f32Bytes);
                _batchDequantBufSize = f32Bytes;
            }
            bF32Ptr = _batchDequantBuf.DevicePtr;

            // GPU dequant kernel: expand quantized weight to FP32
            var func = _elementwiseModule.GetFunction("dequant_to_f32");
            int totalElements = K * N;
            int blockSize = Gguf.GgmlTypeInfo.BlockSize(bT.Type);
            int typeTag = (int)bT.Type;
            int isAligned = (bT.IsAlignedQ8_0 || bT.IsAlignedQ4_0) ? 1 : 0;
            uint grid = (uint)((totalElements + BlockSize - 1) / BlockSize);
            nint* dArgs = stackalloc nint[6];
            dArgs[0] = (nint)(&bF32Ptr);
            dArgs[1] = (nint)(&bPtr);
            dArgs[2] = (nint)(&totalElements);
            dArgs[3] = (nint)(&typeTag);
            dArgs[4] = (nint)(&blockSize);
            dArgs[5] = (nint)(&isAligned);
            _stream.Launch(func, grid, 1, 1, (uint)BlockSize, 1, 1, 0, dArgs);
        }
        else
        {
            bF32Ptr = bPtr;
        }

        // cuBLAS SGEMM: output[M×N] = activation[M×K] × weight^T[K×N]
        // Weight is GGUF layout: Dimensions[0]=K (inner), Dimensions[1]=N (outer)
        // Stored in memory as [N rows × K cols] row-major
        // cuBLAS is column-major: sees weight as [K×N] col-major, need OP_T → [N×K]
        float alpha = 1.0f, beta = 0.0f;
        CublasApi.Check(CublasApi.Sgemm(_cublasHandle,
            CublasApi.CUBLAS_OP_T,  // weight: [K×N] col-major → transpose → [N×K]
            CublasApi.CUBLAS_OP_N,  // activation: [K×M] col-major = [M×K] row-major
            N, M, K,
            &alpha,
            bF32Ptr, K,            // weight lda=K (column stride in col-major)
            aPtr, K,               // activation ldb=K
            &beta,
            outPtr, N),            // output ldc=N
            "cublasSgemm");
    }

    private CudaDeviceMemory? _batchDequantBuf;
    private long _batchDequantBufSize;
    private CudaDeviceMemory? _batchF16WtBuf;
    private long _batchF16WtBufSize;
    private CudaDeviceMemory? _batchF16ActBuf;
    private long _batchF16ActBufSize;
    private ulong _f16ActCachedSrcPtr;  // DevicePtr of last converted activation
    private long _f16ActCachedSize;     // byte size of last conversion

    // Persistent FP16 weight cache: avoids re-dequanting on subsequent prefills
    private readonly Dictionary<ulong, CudaDeviceMemory> _f16WeightCache = new();
    private bool _f16WeightCacheEnabled = true;
    private long _f16CacheBytes;
    private int _f16CacheMisses;
    private const long F16CacheVramHeadroom = 512L * 1024 * 1024; // keep 512 MB free

    /// <summary>Log FP16 weight cache stats. Called once after first prefill.</summary>
    internal void LogF16CacheStats()
    {
        if (_f16WeightCache.Count > 0 || _f16CacheMisses > 0)
        {
            Console.Error.WriteLine($"  FP16 weight cache: {_f16WeightCache.Count} tensors cached " +
                $"({_f16CacheBytes / (1024 * 1024)} MB), {_f16CacheMisses} uncached");
        }
    }

    private bool TryAllocF16Cache(long bytes, out CudaDeviceMemory? buf)
    {
        buf = null;
        EnsureContext();
        if (CudaApi.MemGetInfo(out ulong free, out _) != CuResult.Success)
            return false;
        if ((long)free - bytes < F16CacheVramHeadroom)
            return false;
        try
        {
            buf = new CudaDeviceMemory((ulong)bytes);
            return true;
        }
        catch
        {
            return false;
        }
    }

    /// <summary>Launch a standard 6-arg matmul kernel: (output, a, b, M, K, N).</summary>
    private unsafe void LaunchMatMul(string kernelName, ulong outPtr, ulong aPtr, ulong bPtr,
        int M, int K, int N, uint grid, uint threads, uint smem)
    {
        var func = _matmulModule.GetFunction(kernelName);
        int nVal = N;
        nint* kArgs = stackalloc nint[6];
        kArgs[0] = (nint)(&outPtr); kArgs[1] = (nint)(&aPtr); kArgs[2] = (nint)(&bPtr);
        kArgs[3] = (nint)(&M); kArgs[4] = (nint)(&K); kArgs[5] = (nint)(&nVal);
        _stream.Launch(func, grid, 1, 1, threads, 1, 1, smem, kArgs);
    }

    /// <summary>Launch Q1_0 matmul kernel with extra blockSizeQ parameter.</summary>
    private unsafe void LaunchMatMulQ1_0(ulong outPtr, ulong aPtr, ulong bPtr,
        int M, int K, int N, int blockSizeQ, uint grid, uint threads, uint smem)
    {
        var func = _matmulModule.GetFunction("dequant_matmul_q1_0");
        int nVal = N;
        nint* kArgs = stackalloc nint[7];
        kArgs[0] = (nint)(&outPtr); kArgs[1] = (nint)(&aPtr); kArgs[2] = (nint)(&bPtr);
        kArgs[3] = (nint)(&M); kArgs[4] = (nint)(&K); kArgs[5] = (nint)(&nVal);
        kArgs[6] = (nint)(&blockSizeQ);
        _stream.Launch(func, grid, 1, 1, threads, 1, 1, smem, kArgs);
    }

    /// <summary>
    /// Ensure the Q8_1 scratch buffer is large enough for the given K dimension.
    /// Must be called outside batch recording (no cuMemAlloc during batching).
    /// </summary>
    internal ulong EnsureQ8_1Scratch(int K)
    {
        int numBlocks = K / 32;
        int q8_1Bytes = numBlocks * 36; // [d(f16,2b) + sum(f16,2b) + quants(32b)] = 36 bytes
        if (_q8_1Scratch == null || _q8_1ScratchK < K)
        {
            var old = _q8_1Scratch;
            _q8_1Scratch = new CudaDeviceMemory((ulong)q8_1Bytes);
            _q8_1ScratchK = K;
            _q8_1CachedInputPtr = 0;
            old?.Dispose();
        }
        return _q8_1Scratch.DevicePtr;
    }

    /// <inheritdoc />
    public ITensor CreateTensor(string name, GgmlType type, ReadOnlySpan<long> dimensions)
    {
        EnsureContext();
        return new CudaTensor(name, type, dimensions);
    }

    /// <inheritdoc />
    public ITensor LoadTensor(string name, GgmlType type, ReadOnlySpan<long> dimensions, ReadOnlySpan<byte> data)
    {
        EnsureContext();
        // Repack Q8_0 to aligned layout: 34-byte blocks → 36-byte blocks
        // Old: [scale(2b) | quants(32b)] = 34 bytes, quants NOT 4-byte aligned
        // New: [scale(2b) | pad(2b) | quants(32b)] = 36 bytes, quants 4-byte aligned
        // Required for dp4a matmul kernel which assumes aligned block stride.
        if (type == GgmlType.Q8_0 && dimensions.Length >= 2)
        {
            _hasQ8_0Weights = true;
            // Pre-allocate Q8_1 scratch for dp4a matmul (must happen outside stream capture)
            EnsureQ8_1Scratch((int)dimensions[0]);
            int blockCount = data.Length / 34;
            var aligned = new byte[blockCount * 36];
            for (int i = 0; i < blockCount; i++)
            {
                int srcOff = i * 34;
                int dstOff = i * 36;
                // Copy scale (2 bytes)
                aligned[dstOff] = data[srcOff];
                aligned[dstOff + 1] = data[srcOff + 1];
                // Skip 2 bytes padding (already zero)
                // Copy quants (32 bytes)
                data.Slice(srcOff + 2, 32).CopyTo(aligned.AsSpan(dstOff + 4, 32));
            }
            // Create tensor with aligned layout — mark as Q8_0 but with aligned stride
            var tensor = new CudaTensor(name, type, dimensions, pinned: false, alignedQ8_0: true);
            tensor.Memory.CopyFromHost(aligned);
            return tensor;
        }

        // Repack Q4_0 to aligned layout: 18-byte blocks → 20-byte blocks
        // [scale(2b) | pad(2b) | nibbles(16b)] = 20 bytes, 4-byte aligned
        if (type == GgmlType.Q4_0 && dimensions.Length >= 2)
        {
            _hasQ4_0Weights = true;
            // Pre-allocate Q8_1 scratch for dp4a (must happen outside stream capture)
            EnsureQ8_1Scratch((int)dimensions[0]);
            int blockCount = data.Length / 18;
            var aligned = new byte[blockCount * 20];
            for (int i = 0; i < blockCount; i++)
            {
                int srcOff = i * 18;
                int dstOff = i * 20;
                aligned[dstOff] = data[srcOff];
                aligned[dstOff + 1] = data[srcOff + 1];
                data.Slice(srcOff + 2, 16).CopyTo(aligned.AsSpan(dstOff + 4, 16));
            }
            var tensor = new CudaTensor(name, type, dimensions, pinned: false, alignedQ4_0: true);
            tensor.Memory.CopyFromHost(aligned);
            return tensor;
        }

        if (type == GgmlType.Q4_K && dimensions.Length >= 2)
        {
            EnsureQ8_1Scratch((int)dimensions[0]);
        }

        return new CudaTensor(name, type, dimensions, data);
    }


    /// <inheritdoc />
    public unsafe void MatMul(ITensor output, ITensor a, ITensor b, int M, int K, int N)
    {
        var outT = (CudaTensor)output;
        var aT = (CudaTensor)a;
        var bT = (CudaTensor)b;

        ulong outPtr = outT.DevicePtr;
        ulong aPtr = aT.DevicePtr;
        ulong bPtr = bT.DevicePtr;

        // ── Batch MatMul (M > 1) — cuBLAS SGEMM with dequantized weights ──
        if (M > 1)
        {
            BatchMatMul(outT, aT, bT, M, K, N);
            return;
        }

        // ── Single-token MatMul (M = 1) — optimized per-quant kernels below ──

        // Helper: compute adaptive thread count from work items per row
        static (uint grid, uint threads, uint smem) AdaptiveLaunch(int N, int rows, int workItems)
        {
            uint grid = ((uint)N + (uint)rows - 1) / (uint)rows;
            int threads = Math.Clamp((workItems + 31) & ~31, 32, 256);
            uint smem = (uint)((threads / 32) * sizeof(float));
            return (grid, (uint)threads, smem);
        }

        if (b.Type == GgmlType.F32)
        {
            // cuBLAS SGEMV: y = alpha * A^T * x + beta * y
            // A is [K × N] row-major = [N × K] column-major, so transpose to get [K × N]
            float alpha = 1.0f, beta = 0.0f;
            CublasApi.Check(CublasApi.Sgemv(_cublasHandle,
                CublasApi.CUBLAS_OP_T,     // transpose: b is [N×K] row-major
                K, N,                       // m=K, n=N (column-major dimensions)
                &alpha,
                bPtr, K,                    // A = weights, lda = K
                aPtr, 1,                    // x = activation, incx = 1
                &beta,
                outPtr, 1), "cublasSgemv"); // y = output, incy = 1
        }
        else if (b.Type == GgmlType.Q8_0)
        {
            // Use dp4a (integer dot product) for precision matching llama.cpp.
            // Step 1: Quantize FP32 activation to Q8_1 (if not already cached)
            ulong q8_1Ptr;
            if (_q8_1CachedInputPtr == aPtr && _q8_1CachedGeneration == _q8_1CacheGeneration)
            {
                // Cache hit: same activation, already quantized
                q8_1Ptr = _q8_1Scratch!.DevicePtr;
            }
            else
            {
                // Quantize activation to Q8_1 and cache for subsequent matmuls with same input
                q8_1Ptr = _q8_1Scratch!.DevicePtr;
                var qFunc = _matmulModule.GetFunction("quantize_f32_q8_1");
                int numBlocksQ = K / 32;
                uint qGrid = (uint)((numBlocksQ + BlockSize - 1) / BlockSize);
                int kVal = K;
                nint* qArgs = stackalloc nint[3];
                qArgs[0] = (nint)(&q8_1Ptr);
                qArgs[1] = (nint)(&aPtr);
                qArgs[2] = (nint)(&kVal);
                _stream.Launch(qFunc, qGrid, 1, 1, (uint)BlockSize, 1, 1, 0, qArgs);
                _q8_1CachedInputPtr = aPtr;
                _q8_1CachedGeneration = _q8_1CacheGeneration;
            }

            // Step 2: dp4a matmul (Q8_0 weights × Q8_1 activation → int32 → float)
            var func = _matmulModule.GetFunction("dequant_matmul_q8_0_q8_1_aligned");
            int nVal = N;
            uint dp4aSmem = (uint)((256 / 32) * sizeof(float));
            nint* kArgs = stackalloc nint[6];
            kArgs[0] = (nint)(&outPtr);
            kArgs[1] = (nint)(&q8_1Ptr);
            kArgs[2] = (nint)(&bPtr);
            kArgs[3] = (nint)(&M);
            kArgs[4] = (nint)(&K);
            kArgs[5] = (nint)(&nVal);
            _stream.Launch(func, (uint)N, 1, 1, 256, 1, 1, dp4aSmem, kArgs);
        }
        else if (b.Type == GgmlType.Q4_0 && _q8_1FusedReady
                 && _q8_1CachedInputPtr == aPtr && _q8_1CachedGeneration == _q8_1CacheGeneration)
        {
            // Q8_1 pre-computed by fused RmsNorm — use dp4a (zero quantization overhead)
            ulong q8_1Ptr = _q8_1Scratch!.DevicePtr;
            var func = _matmulModule.GetFunction("dequant_matmul_q4_0_q8_1");
            int nVal = N;
            uint dp4aGrid = ((uint)N + 7) / 8; // Q4_0_DP4A_ROWS = 8
            uint dp4aSmem = (256 / 32) * 8 * sizeof(float); // smem[8][8]
            nint* kArgs = stackalloc nint[6];
            kArgs[0] = (nint)(&outPtr);
            kArgs[1] = (nint)(&q8_1Ptr);
            kArgs[2] = (nint)(&bPtr);
            kArgs[3] = (nint)(&M);
            kArgs[4] = (nint)(&K);
            kArgs[5] = (nint)(&nVal);
            _stream.Launch(func, dp4aGrid, 1, 1, 256, 1, 1, dp4aSmem, kArgs);
        }
        else if (b.Type == GgmlType.Q4_0)
        {
            // dp4a path: use fused Q8_1 from RmsNorm, or quantize on demand
            ulong q8_1Ptr = EnsureQ8_1Scratch(K);

            if (_q8_1CachedInputPtr != aPtr || _q8_1CachedGeneration != _q8_1CacheGeneration)
            {
                // Activation changed since last Q8_1 quantize — must re-quantize.
                // This catches cases where _q8_1FusedReady is stale from a previous RmsNorm
                // but the current activation (e.g. attn output, SiLU output) is different.
                _q8_1CachedInputPtr = aPtr;
                _q8_1CachedGeneration = _q8_1CacheGeneration;
                var quantFunc = _matmulModule.GetFunction("quantize_f32_q8_1");
                int numBlocks = K / 32;
                uint quantGrid = (uint)((numBlocks + BlockSize - 1) / BlockSize);
                int kVal = K;
                nint* qArgs = stackalloc nint[3];
                qArgs[0] = (nint)(&q8_1Ptr);
                qArgs[1] = (nint)(&aPtr);
                qArgs[2] = (nint)(&kVal);
                _stream.Launch(quantFunc, quantGrid, 1, 1, (uint)BlockSize, 1, 1, 0, qArgs);
            }

            var func = _matmulModule.GetFunction("dequant_matmul_q4_0_q8_1");
            int nVal = N;
            uint dp4aGrid = ((uint)N + 7) / 8; // Q4_0_DP4A_ROWS = 8
            uint dp4aSmem = (256 / 32) * 8 * sizeof(float); // smem[8][8]
            nint* kArgs = stackalloc nint[6];
            kArgs[0] = (nint)(&outPtr);
            kArgs[1] = (nint)(&q8_1Ptr);
            kArgs[2] = (nint)(&bPtr);
            kArgs[3] = (nint)(&M);
            kArgs[4] = (nint)(&K);
            kArgs[5] = (nint)(&nVal);
            _stream.Launch(func, dp4aGrid, 1, 1, 256, 1, 1, dp4aSmem, kArgs);
        }
        else if (b.Type == GgmlType.Q4_1)
        {
            var (grid, threads, smem) = AdaptiveLaunch(N, 8, K / 32); // Q4_1_ROWS_PER_BLOCK=8
            LaunchMatMul("dequant_matmul_q4_1", outPtr, aPtr, bPtr, M, K, N, grid, threads, smem);
        }
        else if (b.Type == GgmlType.I2_S)
        {
            var (grid, threads, smem) = AdaptiveLaunch(N, 1, K / 8);
            LaunchMatMul("dequant_matmul_i2s", outPtr, aPtr, bPtr, M, K, N, grid, threads, smem);
        }
        else if (b.Type == GgmlType.TQ1_0)
        {
            var (grid, threads, smem) = AdaptiveLaunch(N, 1, K / 8);
            LaunchMatMul("dequant_matmul_tq1_0", outPtr, aPtr, bPtr, M, K, N, grid, threads, smem);
        }
        else if (b.Type == GgmlType.Q1_0 || b.Type == GgmlType.Q1_0_g128)
        {
            int blockSizeQ = b.Type == GgmlType.Q1_0 ? 32 : 128;
            // Multi-row: 4 output neurons per block, 256 threads
            uint q1Rows = 16;
            uint q1Grid = ((uint)N + q1Rows - 1) / q1Rows;
            uint q1Threads = 32;
            uint q1Smem = (q1Threads / 32) * sizeof(float);
            LaunchMatMulQ1_0(outPtr, aPtr, bPtr, M, K, N, blockSizeQ, q1Grid, q1Threads, q1Smem);
        }
        else if (b.Type == GgmlType.F16)
        {
            var (grid, threads, smem) = AdaptiveLaunch(N, 1, K / 8);
            LaunchMatMul("dequant_matmul_f16", outPtr, aPtr, bPtr, M, K, N, grid, threads, smem);
        }
        else if (b.Type == GgmlType.BF16)
        {
            var (grid, threads, smem) = AdaptiveLaunch(N, 1, K / 8);
            LaunchMatMul("dequant_matmul_bf16", outPtr, aPtr, bPtr, M, K, N, grid, threads, smem);
        }
        else if (b.Type == GgmlType.Q4_K && _q8_1FusedReady
                 && _q8_1CachedInputPtr == aPtr && _q8_1CachedGeneration == _q8_1CacheGeneration)
        {
            // Cooperative v2 kernel: 128 threads, 16 per super-block, dp4a
            // Q8_1 pre-computed by fused RmsNorm — zero quantization overhead
            ulong q8_1Ptr = _q8_1Scratch!.DevicePtr;
            var func = _matmulModule.GetFunction("dequant_matmul_q4_k_v2");
            int nVal = N;
            uint v2Smem = 8 * sizeof(float); // shared_sums[8]
            nint* kArgs = stackalloc nint[6];
            kArgs[0] = (nint)(&outPtr);
            kArgs[1] = (nint)(&q8_1Ptr);
            kArgs[2] = (nint)(&bPtr);
            kArgs[3] = (nint)(&M);
            kArgs[4] = (nint)(&K);
            kArgs[5] = (nint)(&nVal);
            _stream.Launch(func, (uint)N, 1, 1, 128, 1, 1, v2Smem, kArgs);
        }
        else if (b.Type == GgmlType.Q4_K && _q8_1Scratch != null)
        {
            // Cooperative v2 kernel with on-demand Q8_1 quantization
            ulong q8_1Ptr = EnsureQ8_1Scratch(K);

            if (_q8_1CachedInputPtr != aPtr || _q8_1CachedGeneration != _q8_1CacheGeneration)
            {
                _q8_1CachedInputPtr = aPtr;
                _q8_1CachedGeneration = _q8_1CacheGeneration;
                var quantFunc = _matmulModule.GetFunction("quantize_f32_q8_1");
                int numBlocks = K / 32;
                uint quantGrid = (uint)((numBlocks + BlockSize - 1) / BlockSize);
                int kVal = K;
                nint* qArgs = stackalloc nint[3];
                qArgs[0] = (nint)(&q8_1Ptr);
                qArgs[1] = (nint)(&aPtr);
                qArgs[2] = (nint)(&kVal);
                _stream.Launch(quantFunc, quantGrid, 1, 1, (uint)BlockSize, 1, 1, 0, qArgs);
            }

            var func = _matmulModule.GetFunction("dequant_matmul_q4_k_v2");
            int nVal = N;
            uint v2Smem = 8 * sizeof(float);
            nint* kArgs = stackalloc nint[6];
            kArgs[0] = (nint)(&outPtr);
            kArgs[1] = (nint)(&q8_1Ptr);
            kArgs[2] = (nint)(&bPtr);
            kArgs[3] = (nint)(&M);
            kArgs[4] = (nint)(&K);
            kArgs[5] = (nint)(&nVal);
            _stream.Launch(func, (uint)N, 1, 1, 128, 1, 1, v2Smem, kArgs);
        }
        else if (b.Type == GgmlType.Q4_K && _q8_1FusedReady
                 && _q8_1CachedInputPtr == aPtr && _q8_1CachedGeneration == _q8_1CacheGeneration)
        {
            // Pre-Blackwell: Q8_1 pre-computed by fused RmsNorm — use dp4a
            ulong q8_1Ptr = _q8_1Scratch!.DevicePtr;
            var func = _matmulModule.GetFunction("dequant_matmul_q4_k_q8_1");
            int nVal = N;
            uint dp4aGrid = ((uint)N + 3) / 4; // Q4K_DP4A_ROWS = 4
            uint dp4aSmem = (256 / 32) * 4 * sizeof(float); // smem[nwarps][Q4K_DP4A_ROWS]
            nint* kArgs = stackalloc nint[6];
            kArgs[0] = (nint)(&outPtr);
            kArgs[1] = (nint)(&q8_1Ptr);
            kArgs[2] = (nint)(&bPtr);
            kArgs[3] = (nint)(&M);
            kArgs[4] = (nint)(&K);
            kArgs[5] = (nint)(&nVal);
            _stream.Launch(func, dp4aGrid, 1, 1, 256, 1, 1, dp4aSmem, kArgs);
        }
        else if (b.Type == GgmlType.Q4_K)
        {
            // Pre-Blackwell fallback: quantize activation on demand, then use dp4a
            ulong q8_1Ptr = EnsureQ8_1Scratch(K);

            if (_q8_1CachedInputPtr != aPtr || _q8_1CachedGeneration != _q8_1CacheGeneration)
            {
                _q8_1CachedInputPtr = aPtr;
                _q8_1CachedGeneration = _q8_1CacheGeneration;
                var quantFunc = _matmulModule.GetFunction("quantize_f32_q8_1");
                int numBlocks = K / 32;
                uint quantGrid = (uint)((numBlocks + BlockSize - 1) / BlockSize);
                int kVal = K;
                nint* qArgs = stackalloc nint[3];
                qArgs[0] = (nint)(&q8_1Ptr);
                qArgs[1] = (nint)(&aPtr);
                qArgs[2] = (nint)(&kVal);
                _stream.Launch(quantFunc, quantGrid, 1, 1, (uint)BlockSize, 1, 1, 0, qArgs);
            }

            var func = _matmulModule.GetFunction("dequant_matmul_q4_k_q8_1");
            int nVal = N;
            uint dp4aGrid = ((uint)N + 3) / 4; // Q4K_DP4A_ROWS = 4
            uint dp4aSmem = (256 / 32) * 4 * sizeof(float); // smem[nwarps][Q4K_DP4A_ROWS]
            nint* kArgs = stackalloc nint[6];
            kArgs[0] = (nint)(&outPtr);
            kArgs[1] = (nint)(&q8_1Ptr);
            kArgs[2] = (nint)(&bPtr);
            kArgs[3] = (nint)(&M);
            kArgs[4] = (nint)(&K);
            kArgs[5] = (nint)(&nVal);
            _stream.Launch(func, dp4aGrid, 1, 1, 256, 1, 1, dp4aSmem, kArgs);
        }
        else if (b.Type == GgmlType.Q5_K)
        {
            var (grid, threads, smem) = AdaptiveLaunch(N, 1, K / 256 * 8); // Q5K_ROWS_PER_BLOCK=1
            LaunchMatMul("dequant_matmul_q5_k", outPtr, aPtr, bPtr, M, K, N, grid, threads, smem);
        }
        else if (b.Type == GgmlType.Q6_K)
        {
            var (grid, threads, smem) = AdaptiveLaunch(N, 3, K / 256 * 8); // Q6K_ROWS_PER_BLOCK=3
            LaunchMatMul("dequant_matmul_q6_k", outPtr, aPtr, bPtr, M, K, N, grid, threads, smem);
        }
        else
        {
            // Generic fallback: download data, dequantize on CPU, compute on CPU, upload result
            GenericCpuFallbackMatMul(outT, aT, bT, M, K, N);
        }


    }

    /// <inheritdoc />
    public unsafe void MatMulSwiGLU(ITensor output, ITensor a, ITensor gateWeights, ITensor upWeights, int M, int K, int N)
    {
        // Fused path only supports Q4_K single-token with Q8_1 scratch available
        if (M != 1 || gateWeights.Type != GgmlType.Q4_K || _q8_1Scratch == null)
            throw new InvalidOperationException(
                $"MatMulSwiGLU fused kernel requires M=1, Q4_K weights, and Q8_1 scratch. " +
                $"Got M={M}, type={gateWeights.Type}. Use separate MatMul+SwiGLU for other types.");

        var outT = (CudaTensor)output;
        var aT = (CudaTensor)a;
        var gateT = (CudaTensor)gateWeights;
        var upT = (CudaTensor)upWeights;
        ulong outPtr = outT.DevicePtr;
        ulong aPtr = aT.DevicePtr;
        ulong gatePtr = gateT.DevicePtr;
        ulong upPtr = upT.DevicePtr;

        // Ensure Q8_1 activation is available (fused or on-demand)
        ulong q8_1Ptr;
        if (_q8_1FusedReady && _q8_1CachedInputPtr == aPtr && _q8_1CachedGeneration == _q8_1CacheGeneration)
        {
            q8_1Ptr = _q8_1Scratch!.DevicePtr;
        }
        else
        {
            q8_1Ptr = EnsureQ8_1Scratch(K);
            if (_q8_1CachedInputPtr != aPtr || _q8_1CachedGeneration != _q8_1CacheGeneration)
            {
                _q8_1CachedInputPtr = aPtr;
                _q8_1CachedGeneration = _q8_1CacheGeneration;
                var quantFunc = _matmulModule.GetFunction("quantize_f32_q8_1");
                int numBlocks = K / 32;
                uint quantGrid = (uint)((numBlocks + BlockSize - 1) / BlockSize);
                int kVal = K;
                nint* qArgs = stackalloc nint[3];
                qArgs[0] = (nint)(&q8_1Ptr);
                qArgs[1] = (nint)(&aPtr);
                qArgs[2] = (nint)(&kVal);
                _stream.Launch(quantFunc, quantGrid, 1, 1, (uint)BlockSize, 1, 1, 0, qArgs);
            }
        }

        var func = _matmulModule.GetFunction("dequant_matmul_swiGLU_q4_k");
        int nVal = N;
        uint smem = 2 * 8 * sizeof(float); // shared_gate[8] + shared_up[8]
        nint* kArgs = stackalloc nint[7];
        kArgs[0] = (nint)(&outPtr);
        kArgs[1] = (nint)(&q8_1Ptr);
        kArgs[2] = (nint)(&gatePtr);
        kArgs[3] = (nint)(&upPtr);
        kArgs[4] = (nint)(&M);
        kArgs[5] = (nint)(&K);
        kArgs[6] = (nint)(&nVal);
        _stream.Launch(func, (uint)N, 1, 1, 128, 1, 1, smem, kArgs);
    }

    /// <inheritdoc />
    public unsafe void BatchedEmbeddingLookup(ITensor output, ITensor table, int[] tokenIds)
    {
        EnsureContext();
        var outT = (CudaTensor)output;
        var tblT = (CudaTensor)table;
        ulong outPtr = outT.DevicePtr;
        ulong tblPtr = tblT.DevicePtr;
        int hiddenDim = (int)table.Dimensions[0];
        int M = tokenIds.Length;

        if (table.Type == GgmlType.Q8_0)
        {
            int blockStride = tblT.IsAlignedQ8_0 ? 36 : 34;
            int quantOffset = blockStride == 36 ? 4 : 2;

            // Upload token IDs to GPU
            if (_batchTokenIdsBuf == null || _batchTokenIdsBufSize < M)
            {
                _batchTokenIdsBuf?.Dispose();
                _batchTokenIdsBuf = new CudaDeviceMemory((ulong)(M * sizeof(int)));
                _batchTokenIdsBufSize = M;
            }
            fixed (int* p = tokenIds)
                CudaApi.Check(CudaApi.MemcpyHtoDAsync(_batchTokenIdsBuf.DevicePtr, p,
                    (ulong)(M * sizeof(int)), _stream.Handle), "upload tokenIds");
            ulong idsPtr = _batchTokenIdsBuf.DevicePtr;

            var func = _elementwiseModule.GetFunction("batched_embedding_q8_0");
            int total = M * hiddenDim;
            uint grid = (uint)((total + BlockSize - 1) / BlockSize);
            nint* kArgs = stackalloc nint[7];
            kArgs[0] = (nint)(&outPtr);
            kArgs[1] = (nint)(&tblPtr);
            kArgs[2] = (nint)(&idsPtr);
            kArgs[3] = (nint)(&hiddenDim);
            kArgs[4] = (nint)(&M);
            kArgs[5] = (nint)(&blockStride);
            kArgs[6] = (nint)(&quantOffset);
            _stream.Launch(func, grid, 1, 1, (uint)BlockSize, 1, 1, 0, kArgs);
        }
        else
        {
            // Fallback: sequential
            for (int i = 0; i < M; i++)
            {
                EmbeddingLookup(output, table, tokenIds[i]);
                // This only works for single-token output buffer — not ideal for batch
            }
        }
    }
    private CudaDeviceMemory? _batchTokenIdsBuf;
    private int _batchTokenIdsBufSize;

    /// <inheritdoc />
    public unsafe void RmsNorm(ITensor output, ITensor input, ITensor weight, float eps)
    {
        _q8_1CacheGeneration++;
        var outT = (CudaTensor)output;
        var inT = (CudaTensor)input;
        var wT = (CudaTensor)weight;

        ulong outPtr = outT.DevicePtr;
        ulong inPtr = inT.DevicePtr;
        ulong wPtr = wT.DevicePtr;
        int n = (int)weight.ElementCount; // hidden dim
        int totalElements = (int)input.ElementCount;
        int M = totalElements / n; // number of rows (1 for decode, >1 for prefill)

        uint sharedMem = (uint)(BlockSize * sizeof(float));

        if (M > 1)
        {
            var func = _elementwiseModule.GetFunction("batched_rms_norm");
            nint* kArgs = stackalloc nint[6];
            kArgs[0] = (nint)(&outPtr);
            kArgs[1] = (nint)(&inPtr);
            kArgs[2] = (nint)(&wPtr);
            kArgs[3] = (nint)(&n);
            kArgs[4] = (nint)(&M);
            kArgs[5] = (nint)(&eps);
            _stream.Launch(func, (uint)M, 1, 1, (uint)BlockSize, 1, 1, sharedMem, kArgs);
        }
        else
        {
            var func = _elementwiseModule.GetFunction("rms_norm");
            nint* kArgs = stackalloc nint[5];
            kArgs[0] = (nint)(&outPtr);
            kArgs[1] = (nint)(&inPtr);
            kArgs[2] = (nint)(&wPtr);
            kArgs[3] = (nint)(&n);
            kArgs[4] = (nint)(&eps);
            _stream.Launch(func, 1, 1, 1, (uint)BlockSize, 1, 1, sharedMem, kArgs);
        }
    }

    /// <inheritdoc />
    public unsafe void Softmax(ITensor output, ITensor input)
    {
        var outT = (CudaTensor)output;
        var inT = (CudaTensor)input;

        ulong outPtr = outT.DevicePtr;
        ulong inPtr = inT.DevicePtr;
        int n = (int)input.ElementCount;

        var func = _elementwiseModule.GetFunction("softmax");
        nint* kArgs = stackalloc nint[3];
        kArgs[0] = (nint)(&outPtr);
        kArgs[1] = (nint)(&inPtr);
        kArgs[2] = (nint)(&n);

        uint sharedMem = (uint)(BlockSize * sizeof(float));
        _stream.Launch(func, 1, 1, 1, (uint)BlockSize, 1, 1, sharedMem, kArgs);

    }

    /// <inheritdoc />
    public unsafe void SiLU(ITensor output, ITensor input)
    {
        var outT = (CudaTensor)output;
        var inT = (CudaTensor)input;

        ulong outPtr = outT.DevicePtr;
        ulong inPtr = inT.DevicePtr;
        int n = (int)input.ElementCount;

        var func = _elementwiseModule.GetFunction("silu");
        uint grid = (uint)((n + BlockSize - 1) / BlockSize);
        nint* kArgs = stackalloc nint[3];
        kArgs[0] = (nint)(&outPtr);
        kArgs[1] = (nint)(&inPtr);
        kArgs[2] = (nint)(&n);

        _stream.Launch(func, grid, 1, 1, (uint)BlockSize, 1, 1, 0, kArgs);

    }

    /// <inheritdoc />
    public unsafe void RoPE(ITensor q, ITensor k, int headDim, int ropeDim, int positionOffset, float ropeTheta)
    {
        var qT = (CudaTensor)q;
        var kT = (CudaTensor)k;

        ulong qPtr = qT.DevicePtr;
        ulong kPtr = kT.DevicePtr;
        int qTotal = (int)q.ElementCount;
        int kTotal = (int)k.ElementCount;
        int effectiveRopeDim = ropeDim > 0 ? ropeDim : headDim;

        var func = _elementwiseModule.GetFunction("rope");
        int maxPairs = Math.Max(qTotal, kTotal) / 2;
        uint grid = (uint)((maxPairs + BlockSize - 1) / BlockSize);

        nint* kArgs = stackalloc nint[7];
        kArgs[0] = (nint)(&qPtr);
        kArgs[1] = (nint)(&kPtr);
        kArgs[2] = (nint)(&qTotal);
        kArgs[3] = (nint)(&kTotal);
        kArgs[4] = (nint)(&headDim);
        kArgs[5] = (nint)(&effectiveRopeDim);
        kArgs[6] = (nint)(&positionOffset);
        // ropeTheta is the 8th arg
        nint* allArgs = stackalloc nint[8];
        allArgs[0] = (nint)(&qPtr);
        allArgs[1] = (nint)(&kPtr);
        allArgs[2] = (nint)(&qTotal);
        allArgs[3] = (nint)(&kTotal);
        allArgs[4] = (nint)(&headDim);
        allArgs[5] = (nint)(&effectiveRopeDim);
        allArgs[6] = (nint)(&positionOffset);
        allArgs[7] = (nint)(&ropeTheta);

        _stream.Launch(func, grid, 1, 1, (uint)BlockSize, 1, 1, 0, allArgs);

    }

    /// <inheritdoc />
    public unsafe void ElementMul(ITensor output, ITensor a, ITensor b)
    {
        var outT = (CudaTensor)output;
        var aT = (CudaTensor)a;
        var bT = (CudaTensor)b;

        ulong outPtr = outT.DevicePtr;
        ulong aPtr = aT.DevicePtr;
        ulong bPtr = bT.DevicePtr;
        int n = (int)a.ElementCount;

        var func = _elementwiseModule.GetFunction("element_mul");
        uint grid = (uint)((n + BlockSize - 1) / BlockSize);
        nint* kArgs = stackalloc nint[4];
        kArgs[0] = (nint)(&outPtr);
        kArgs[1] = (nint)(&aPtr);
        kArgs[2] = (nint)(&bPtr);
        kArgs[3] = (nint)(&n);

        _stream.Launch(func, grid, 1, 1, (uint)BlockSize, 1, 1, 0, kArgs);

    }

    /// <inheritdoc />
    public unsafe void ElementAdd(ITensor output, ITensor a, ITensor b)
    {
        var outT = (CudaTensor)output;
        var aT = (CudaTensor)a;
        var bT = (CudaTensor)b;

        ulong outPtr = outT.DevicePtr;
        ulong aPtr = aT.DevicePtr;
        ulong bPtr = bT.DevicePtr;
        int n = (int)a.ElementCount;

        var func = _elementwiseModule.GetFunction("element_add");
        uint grid = (uint)((n + BlockSize - 1) / BlockSize);
        nint* kArgs = stackalloc nint[4];
        kArgs[0] = (nint)(&outPtr);
        kArgs[1] = (nint)(&aPtr);
        kArgs[2] = (nint)(&bPtr);
        kArgs[3] = (nint)(&n);

        _stream.Launch(func, grid, 1, 1, (uint)BlockSize, 1, 1, 0, kArgs);

    }

    /// <inheritdoc />
    public unsafe void EmbeddingLookup(ITensor output, ITensor table, int tokenId)
    {
        var outT = (CudaTensor)output;
        var tableT = (CudaTensor)table;

        ulong outPtr = outT.DevicePtr;
        ulong tablePtr = tableT.DevicePtr;
        int hiddenDim = (int)table.Dimensions[0];

        uint grid = (uint)((hiddenDim + BlockSize - 1) / BlockSize);

        if (table.Type == GgmlType.F32)
        {
            var func = _elementwiseModule.GetFunction("embedding_lookup_f32");
            nint* kArgs = stackalloc nint[4];
            kArgs[0] = (nint)(&outPtr);
            kArgs[1] = (nint)(&tablePtr);
            kArgs[2] = (nint)(&hiddenDim);
            kArgs[3] = (nint)(&tokenId);
            _stream.Launch(func, grid, 1, 1, (uint)BlockSize, 1, 1, 0, kArgs);
        }
        else if (table.Type == GgmlType.Q8_0)
        {
            // Use block stride based on alignment (34 original, 36 repacked)
            int blockStride = (tableT is CudaTensor ct && ct.IsAlignedQ8_0) ? 36 : 34;
            int quantOffset = blockStride == 36 ? 4 : 2; // quants start after scale+pad or scale
            var func = _elementwiseModule.GetFunction("embedding_lookup_q8_0_v2");
            nint* kArgs = stackalloc nint[6];
            kArgs[0] = (nint)(&outPtr);
            kArgs[1] = (nint)(&tablePtr);
            kArgs[2] = (nint)(&hiddenDim);
            kArgs[3] = (nint)(&tokenId);
            kArgs[4] = (nint)(&blockStride);
            kArgs[5] = (nint)(&quantOffset);
            _stream.Launch(func, grid, 1, 1, (uint)BlockSize, 1, 1, 0, kArgs);
        }
        else if (table.Type == GgmlType.F16)
        {
            var func = _elementwiseModule.GetFunction("embedding_lookup_f16");
            nint* kArgs = stackalloc nint[4];
            kArgs[0] = (nint)(&outPtr);
            kArgs[1] = (nint)(&tablePtr);
            kArgs[2] = (nint)(&hiddenDim);
            kArgs[3] = (nint)(&tokenId);
            _stream.Launch(func, grid, 1, 1, (uint)BlockSize, 1, 1, 0, kArgs);
        }
        else if (table.Type == GgmlType.Q4_K)
        {
            var func = _elementwiseModule.GetFunction("embedding_lookup_q4_k");
            nint* kArgs = stackalloc nint[4];
            kArgs[0] = (nint)(&outPtr);
            kArgs[1] = (nint)(&tablePtr);
            kArgs[2] = (nint)(&hiddenDim);
            kArgs[3] = (nint)(&tokenId);
            _stream.Launch(func, grid, 1, 1, (uint)BlockSize, 1, 1, 0, kArgs);
        }
        else if (table.Type == GgmlType.Q4_0)
        {
            bool isAlignedQ4 = tableT is CudaTensor ct3 && ct3.IsAlignedQ4_0;
            int blockStride = isAlignedQ4 ? 20 : 18;
            int nibbleOffset = isAlignedQ4 ? 4 : 2;
            var func = _elementwiseModule.GetFunction("embedding_lookup_q4_0_v2");
            nint* kArgs = stackalloc nint[6];
            kArgs[0] = (nint)(&outPtr);
            kArgs[1] = (nint)(&tablePtr);
            kArgs[2] = (nint)(&hiddenDim);
            kArgs[3] = (nint)(&tokenId);
            kArgs[4] = (nint)(&blockStride);
            kArgs[5] = (nint)(&nibbleOffset);
            _stream.Launch(func, grid, 1, 1, (uint)BlockSize, 1, 1, 0, kArgs);
        }
        else if (table.Type == GgmlType.Q4_1)
        {
            var func = _elementwiseModule.GetFunction("embedding_lookup_q4_1");
            nint* kArgs = stackalloc nint[4];
            kArgs[0] = (nint)(&outPtr);
            kArgs[1] = (nint)(&tablePtr);
            kArgs[2] = (nint)(&hiddenDim);
            kArgs[3] = (nint)(&tokenId);
            _stream.Launch(func, grid, 1, 1, (uint)BlockSize, 1, 1, 0, kArgs);
        }
        else if (table.Type == GgmlType.BF16)
        {
            var func = _elementwiseModule.GetFunction("embedding_lookup_bf16");
            nint* kArgs = stackalloc nint[4];
            kArgs[0] = (nint)(&outPtr);
            kArgs[1] = (nint)(&tablePtr);
            kArgs[2] = (nint)(&hiddenDim);
            kArgs[3] = (nint)(&tokenId);
            _stream.Launch(func, grid, 1, 1, (uint)BlockSize, 1, 1, 0, kArgs);
        }
        else if (table.Type == GgmlType.Q1_0 || table.Type == GgmlType.Q1_0_g128)
        {
            int blockSizeQ = table.Type == GgmlType.Q1_0 ? 32 : 128;
            int bytesPerBlock = table.Type == GgmlType.Q1_0 ? 6 : 18;
            var func = _elementwiseModule.GetFunction("embedding_lookup_q1_0");
            nint* kArgs = stackalloc nint[6];
            kArgs[0] = (nint)(&outPtr);
            kArgs[1] = (nint)(&tablePtr);
            kArgs[2] = (nint)(&hiddenDim);
            kArgs[3] = (nint)(&tokenId);
            kArgs[4] = (nint)(&blockSizeQ);
            kArgs[5] = (nint)(&bytesPerBlock);
            _stream.Launch(func, grid, 1, 1, (uint)BlockSize, 1, 1, 0, kArgs);
        }
        else
        {
            GenericCpuFallbackEmbeddingLookup(outT, tableT, hiddenDim, tokenId);
        }

    }

    // ── Composite Operations ───────────────────────────────────────────────────

    /// <inheritdoc />
    public void CopyTensor(ITensor dst, ITensor src)
    {
        EnsureContext();
        var dstT = (CudaTensor)dst;
        var srcT = (CudaTensor)src;
        CudaApi.Check(CudaApi.MemcpyDtoDAsync(dstT.DevicePtr, srcT.DevicePtr, (ulong)src.ByteSize, _stream.Handle),
            "cuMemcpyDtoDAsync");
    }

    /// <inheritdoc />
    public unsafe void SiLUInPlace(ITensor data)
    {
        var dT = (CudaTensor)data;
        ulong dPtr = dT.DevicePtr;
        int n = (int)data.ElementCount;

        var func = _compositeModule.GetFunction("silu_inplace");
        uint grid = (uint)((n + BlockSize - 1) / BlockSize);
        nint* kArgs = stackalloc nint[2];
        kArgs[0] = (nint)(&dPtr);
        kArgs[1] = (nint)(&n);
        _stream.Launch(func, grid, 1, 1, (uint)BlockSize, 1, 1, 0, kArgs);

    }

    /// <inheritdoc />
    public unsafe void L2NormGroups(ITensor data, int numGroups, int groupDim)
    {
        var dT = (CudaTensor)data;
        ulong dPtr = dT.DevicePtr;

        var func = _compositeModule.GetFunction("l2_norm_groups");
        uint sharedMem = (uint)(BlockSize * sizeof(float));
        nint* kArgs = stackalloc nint[3];
        kArgs[0] = (nint)(&dPtr);
        kArgs[1] = (nint)(&numGroups);
        kArgs[2] = (nint)(&groupDim);
        _stream.Launch(func, (uint)numGroups, 1, 1, (uint)BlockSize, 1, 1, sharedMem, kArgs);

    }

    /// <inheritdoc />
    public unsafe void PerHeadRmsNorm(ITensor data, ITensor weight, int numHeads, int headDim, float eps)
    {
        var dT = (CudaTensor)data;
        var wT = (CudaTensor)weight;
        ulong dPtr = dT.DevicePtr;
        ulong wPtr = wT.DevicePtr;

        var func = _compositeModule.GetFunction("per_head_rms_norm");
        uint sharedMem = (uint)(BlockSize * sizeof(float));
        nint* kArgs = stackalloc nint[5];
        kArgs[0] = (nint)(&dPtr);
        kArgs[1] = (nint)(&wPtr);
        kArgs[2] = (nint)(&numHeads);
        kArgs[3] = (nint)(&headDim);
        kArgs[4] = (nint)(&eps);
        _stream.Launch(func, (uint)numHeads, 1, 1, (uint)BlockSize, 1, 1, sharedMem, kArgs);

    }

    /// <inheritdoc />
    public unsafe void DeInterleaveQ(ITensor qAttn, ITensor qGate, ITensor qFull, int numHeads, int headDim)
    {
        var qaT = (CudaTensor)qAttn;
        var qgT = (CudaTensor)qGate;
        var qfT = (CudaTensor)qFull;
        ulong qaPtr = qaT.DevicePtr;
        ulong qgPtr = qgT.DevicePtr;
        ulong qfPtr = qfT.DevicePtr;
        int total = numHeads * headDim;

        var func = _compositeModule.GetFunction("deinterleave_q");
        uint grid = (uint)((total + BlockSize - 1) / BlockSize);
        nint* kArgs = stackalloc nint[5];
        kArgs[0] = (nint)(&qaPtr);
        kArgs[1] = (nint)(&qgPtr);
        kArgs[2] = (nint)(&qfPtr);
        kArgs[3] = (nint)(&numHeads);
        kArgs[4] = (nint)(&headDim);
        _stream.Launch(func, grid, 1, 1, (uint)BlockSize, 1, 1, 0, kArgs);

    }

    /// <inheritdoc />
    public unsafe void KvCacheWrite(ITensor kCache, ITensor vCache, ITensor k, ITensor v,
        int nKvHeads, int keyLength, int valueLength, int maxSeqLen, int position)
    {
        var kcT = (CudaTensor)kCache;
        var vcT = (CudaTensor)vCache;
        var kT = (CudaTensor)k;
        var vT = (CudaTensor)v;
        ulong kcPtr = kcT.DevicePtr;
        ulong vcPtr = vcT.DevicePtr;
        ulong kPtr = kT.DevicePtr;
        ulong vPtr = vT.DevicePtr;
        int cacheIsFp16 = kCache.Type == GgmlType.F16 ? 1 : 0;

        int maxElems = Math.Max(nKvHeads * keyLength, nKvHeads * valueLength);
        var func = _compositeModule.GetFunction("kv_cache_write");
        uint grid = (uint)((maxElems + BlockSize - 1) / BlockSize);
        nint* kArgs = stackalloc nint[10];
        kArgs[0] = (nint)(&kcPtr);
        kArgs[1] = (nint)(&vcPtr);
        kArgs[2] = (nint)(&kPtr);
        kArgs[3] = (nint)(&vPtr);
        kArgs[4] = (nint)(&nKvHeads);
        kArgs[5] = (nint)(&keyLength);
        kArgs[6] = (nint)(&valueLength);
        kArgs[7] = (nint)(&maxSeqLen);
        kArgs[8] = (nint)(&position);
        kArgs[9] = (nint)(&cacheIsFp16);
        _stream.Launch(func, grid, 1, 1, (uint)BlockSize, 1, 1, 0, kArgs);

    }

    /// <inheritdoc />
    public unsafe void GatedAttention(ITensor output, ITensor qAttn, ITensor qGate,
        ITensor kCache, ITensor vCache,
        int numHeads, int numKvHeads, int keyLength, int valueLength,
        int maxSeqLen, int seqLen, float scale)
    {
        var outT = (CudaTensor)output;
        var qaT = (CudaTensor)qAttn;
        var qgT = (CudaTensor)qGate;
        var kcT = (CudaTensor)kCache;
        var vcT = (CudaTensor)vCache;
        ulong outPtr = outT.DevicePtr;
        ulong qaPtr = qaT.DevicePtr;
        ulong qgPtr = qgT.DevicePtr;
        ulong kcPtr = kcT.DevicePtr;
        ulong vcPtr = vcT.DevicePtr;
        int cacheIsFp16 = kCache.Type == GgmlType.F16 ? 1 : 0;

        // Tiled attention: shared memory for tile scores + reduction temp (constant size)
        const int tileSize = 256;
        uint sharedMem = (uint)((tileSize + BlockSize) * sizeof(float));
        var func = _compositeModule.GetFunction("gated_attention");
        nint* kArgs = stackalloc nint[13];
        kArgs[0] = (nint)(&outPtr);
        kArgs[1] = (nint)(&qaPtr);
        kArgs[2] = (nint)(&qgPtr);
        kArgs[3] = (nint)(&kcPtr);
        kArgs[4] = (nint)(&vcPtr);
        kArgs[5] = (nint)(&numHeads);
        kArgs[6] = (nint)(&numKvHeads);
        kArgs[7] = (nint)(&keyLength);
        kArgs[8] = (nint)(&valueLength);
        kArgs[9] = (nint)(&maxSeqLen);
        kArgs[10] = (nint)(&seqLen);
        kArgs[11] = (nint)(&scale);
        kArgs[12] = (nint)(&cacheIsFp16);
        _stream.Launch(func, (uint)numHeads, 1, 1, (uint)BlockSize, 1, 1, sharedMem, kArgs);

    }

    /// <inheritdoc />
    public unsafe void BatchedRoPE(ITensor q, ITensor k, int headDim, int ropeDim,
        int startPosition, float ropeTheta, int numHeads, int numKvHeads)
    {
        var qT = (CudaTensor)q;
        var kT = (CudaTensor)k;
        ulong qPtr = qT.DevicePtr;
        ulong kPtr = kT.DevicePtr;
        int qTotal = (int)q.ElementCount;
        int kTotal = (int)k.ElementCount;
        int effectiveRopeDim = ropeDim > 0 ? ropeDim : headDim;

        var func = _elementwiseModule.GetFunction("batched_rope");
        int maxPairs = Math.Max(qTotal, kTotal) / 2;
        uint grid = (uint)((maxPairs + BlockSize - 1) / BlockSize);

        nint* kArgs = stackalloc nint[10];
        kArgs[0] = (nint)(&qPtr);
        kArgs[1] = (nint)(&kPtr);
        kArgs[2] = (nint)(&qTotal);
        kArgs[3] = (nint)(&kTotal);
        kArgs[4] = (nint)(&headDim);
        kArgs[5] = (nint)(&effectiveRopeDim);
        kArgs[6] = (nint)(&startPosition);
        kArgs[7] = (nint)(&ropeTheta);
        kArgs[8] = (nint)(&numHeads);
        kArgs[9] = (nint)(&numKvHeads);
        _stream.Launch(func, grid, 1, 1, (uint)BlockSize, 1, 1, 0, kArgs);
    }

    /// <inheritdoc />
    public unsafe void BatchedKvCacheWrite(ITensor kCache, ITensor vCache, ITensor k, ITensor v,
        int nKvHeads, int keyLength, int valueLength, int maxSeqLen, int startPosition, int M)
    {
        var kcT = (CudaTensor)kCache;
        var vcT = (CudaTensor)vCache;
        var kT = (CudaTensor)k;
        var vT = (CudaTensor)v;
        ulong kcPtr = kcT.DevicePtr;
        ulong vcPtr = vcT.DevicePtr;
        ulong kPtr = kT.DevicePtr;
        ulong vPtr = vT.DevicePtr;
        int cacheIsFp16 = kCache.Type == GgmlType.F16 ? 1 : 0;

        int totalElements = Math.Max(M * nKvHeads * keyLength, M * nKvHeads * valueLength);
        uint grid = (uint)((totalElements + BlockSize - 1) / BlockSize);

        var func = _compositeModule.GetFunction("batched_kv_cache_write");
        nint* kArgs = stackalloc nint[11];
        kArgs[0] = (nint)(&kcPtr);
        kArgs[1] = (nint)(&vcPtr);
        kArgs[2] = (nint)(&kPtr);
        kArgs[3] = (nint)(&vPtr);
        kArgs[4] = (nint)(&nKvHeads);
        kArgs[5] = (nint)(&keyLength);
        kArgs[6] = (nint)(&valueLength);
        kArgs[7] = (nint)(&maxSeqLen);
        kArgs[8] = (nint)(&startPosition);
        kArgs[9] = (nint)(&M);
        kArgs[10] = (nint)(&cacheIsFp16);
        _stream.Launch(func, grid, 1, 1, (uint)BlockSize, 1, 1, 0, kArgs);
    }

    /// <inheritdoc />
    public unsafe void BatchedGatedAttention(ITensor output, ITensor qAttn, ITensor qGate,
        ITensor kCache, ITensor vCache,
        int numHeads, int numKvHeads, int keyLength, int valueLength,
        int maxSeqLen, int startPosition, int M, float scale)
    {
        var outT = (CudaTensor)output;
        var qaT = (CudaTensor)qAttn;
        var qgT = (CudaTensor)qGate;
        var kcT = (CudaTensor)kCache;
        var vcT = (CudaTensor)vCache;
        ulong outPtr = outT.DevicePtr;
        ulong qaPtr = qaT.DevicePtr;
        ulong qgPtr = qgT.DevicePtr;
        ulong kcPtr = kcT.DevicePtr;
        ulong vcPtr = vcT.DevicePtr;
        int cacheIsFp16 = kCache.Type == GgmlType.F16 ? 1 : 0;

        // ── cuBLAS matmul path for FP32 cache (avoids O(M²) redundant KV reads) ──
        if (cacheIsFp16 == 0 && M > 32)
        {
            int headsPerKv = numHeads / numKvHeads;
            int seqLen = startPosition + M;  // total KV entries

            // Allocate scores buffer: [numHeads × M × seqLen]
            long scoreBytes = (long)numHeads * M * seqLen * sizeof(float);
            if (_attnScoreBuf == null || _attnScoreBufSize < scoreBytes)
            {
                _attnScoreBuf?.Dispose();
                _attnScoreBuf = new CudaDeviceMemory((ulong)scoreBytes);
                _attnScoreBufSize = scoreBytes;
            }
            ulong scorePtr = _attnScoreBuf.DevicePtr;

            // Step 1: QK^T via cuBLAS — one strided batched GEMM per KV head group
            float alpha = 1.0f, beta = 0.0f;
            for (int kh = 0; kh < numKvHeads; kh++)
            {
                ulong kPtr = kcPtr + (ulong)((long)kh * maxSeqLen * keyLength * sizeof(float));
                int firstHead = kh * headsPerKv;
                ulong qGroupPtr = qaPtr + (ulong)(firstHead * keyLength * sizeof(float));
                ulong sGroupPtr = scorePtr + (ulong)((long)firstHead * M * seqLen * sizeof(float));

                CublasApi.Check(CublasApi.GemmStridedBatchedEx(_cublasHandle,
                    CublasApi.CUBLAS_OP_T, CublasApi.CUBLAS_OP_N,
                    seqLen, M, keyLength,
                    &alpha,
                    kPtr, CublasApi.CUDA_R_32F, keyLength, 0,
                    qGroupPtr, CublasApi.CUDA_R_32F, numHeads * keyLength, keyLength,
                    &beta,
                    sGroupPtr, CublasApi.CUDA_R_32F, seqLen, (long)M * seqLen,
                    headsPerKv,
                    CublasApi.CUBLAS_COMPUTE_32F, CublasApi.CUBLAS_GEMM_DEFAULT),
                    "QK^T");
            }

            // Step 2: Causal softmax in-place on scores
            var softmaxFunc = _compositeModule.GetFunction("causal_softmax_inplace");
            uint softmaxGrid = (uint)(numHeads * M);
            uint softmaxSmem = (uint)(BlockSize * sizeof(float));
            int startPos = startPosition;
            nint* smArgs = stackalloc nint[4];
            smArgs[0] = (nint)(&scorePtr);
            smArgs[1] = (nint)(&seqLen);  // row width = seqLen
            smArgs[2] = (nint)(&startPos);
            smArgs[3] = (nint)(&scale);
            // Note: scale already applied in GEMM (alpha=1), softmax just needs to apply scale
            // Actually, we need scale in softmax. Let me set alpha=scale in GEMM instead.
            // For now, pass scale=1 since we already factored it as GEMM alpha.
            // Wait — I set alpha=1.0 in GEMM. Need to apply scale in softmax.
            _stream.Launch(softmaxFunc, softmaxGrid, 1, 1, (uint)BlockSize, 1, 1, softmaxSmem, smArgs);

            // Step 3: Score × V via strided batched GEMM per KV head group
            for (int kh = 0; kh < numKvHeads; kh++)
            {
                ulong vPtr = vcPtr + (ulong)((long)kh * maxSeqLen * valueLength * sizeof(float));
                int firstHead = kh * headsPerKv;
                ulong sGroupPtr = scorePtr + (ulong)((long)firstHead * M * seqLen * sizeof(float));
                ulong oGroupPtr = outPtr + (ulong)(firstHead * valueLength * sizeof(float));

                CublasApi.Check(CublasApi.GemmStridedBatchedEx(_cublasHandle,
                    CublasApi.CUBLAS_OP_N, CublasApi.CUBLAS_OP_N,
                    valueLength, M, seqLen,
                    &alpha,
                    vPtr, CublasApi.CUDA_R_32F, valueLength, 0,
                    sGroupPtr, CublasApi.CUDA_R_32F, seqLen, (long)M * seqLen,
                    &beta,
                    oGroupPtr, CublasApi.CUDA_R_32F, numHeads * valueLength, valueLength,
                    headsPerKv,
                    CublasApi.CUBLAS_COMPUTE_32F, CublasApi.CUBLAS_GEMM_DEFAULT),
                    "ScoreV");
            }

            // Step 4: Apply sigmoid gating
            var gateFunc = _compositeModule.GetFunction("sigmoid_gate_inplace");
            int gateN = M * numHeads * valueLength;
            uint gateGrid = (uint)((gateN + BlockSize - 1) / BlockSize);
            nint* gArgs = stackalloc nint[3];
            gArgs[0] = (nint)(&outPtr);
            gArgs[1] = (nint)(&qgPtr);
            gArgs[2] = (nint)(&gateN);
            _stream.Launch(gateFunc, gateGrid, 1, 1, (uint)BlockSize, 1, 1, 0, gArgs);
            return;
        }

        // ── Fallback: custom kernel (for FP16 cache or small M) ──
        const int tileSize = 256;
        uint sharedMem = (uint)((tileSize + BlockSize) * sizeof(float));
        var func = _compositeModule.GetFunction("batched_gated_attention");
        uint gridX = (uint)(M * numHeads);

        nint* kArgs = stackalloc nint[14];
        kArgs[0] = (nint)(&outPtr);
        kArgs[1] = (nint)(&qaPtr);
        kArgs[2] = (nint)(&qgPtr);
        kArgs[3] = (nint)(&kcPtr);
        kArgs[4] = (nint)(&vcPtr);
        kArgs[5] = (nint)(&numHeads);
        kArgs[6] = (nint)(&numKvHeads);
        kArgs[7] = (nint)(&keyLength);
        kArgs[8] = (nint)(&valueLength);
        kArgs[9] = (nint)(&maxSeqLen);
        kArgs[10] = (nint)(&startPosition);
        kArgs[11] = (nint)(&M);
        kArgs[12] = (nint)(&scale);
        kArgs[13] = (nint)(&cacheIsFp16);
        _stream.Launch(func, gridX, 1, 1, (uint)BlockSize, 1, 1, sharedMem, kArgs);
    }
    private CudaDeviceMemory? _attnScoreBuf;
    private long _attnScoreBufSize;

    /// <inheritdoc />
    public unsafe void CausalConv1d(ITensor qkv, ITensor convBuffer, ITensor convWeight, int channels, int kernelSize)
    {
        var qkvT = (CudaTensor)qkv;
        var cbT = (CudaTensor)convBuffer;
        var cwT = (CudaTensor)convWeight;
        ulong qkvPtr = qkvT.DevicePtr;
        ulong cbPtr = cbT.DevicePtr;
        ulong cwPtr = cwT.DevicePtr;

        var func = _compositeModule.GetFunction("causal_conv1d");
        uint grid = (uint)((channels + BlockSize - 1) / BlockSize);
        nint* kArgs = stackalloc nint[5];
        kArgs[0] = (nint)(&qkvPtr);
        kArgs[1] = (nint)(&cbPtr);
        kArgs[2] = (nint)(&cwPtr);
        kArgs[3] = (nint)(&channels);
        kArgs[4] = (nint)(&kernelSize);
        _stream.Launch(func, grid, 1, 1, (uint)BlockSize, 1, 1, 0, kArgs);

    }

    /// <inheritdoc />
    public unsafe void ComputeDecayBeta(ITensor decay, ITensor beta, ITensor alphaProj, ITensor betaProj,
        ITensor ssmA, ITensor dtBias, int groupCount)
    {
        var decT = (CudaTensor)decay;
        var betT = (CudaTensor)beta;
        var apT = (CudaTensor)alphaProj;
        var bpT = (CudaTensor)betaProj;
        var aT = (CudaTensor)ssmA;
        var dbT = (CudaTensor)dtBias;
        ulong decPtr = decT.DevicePtr;
        ulong betPtr = betT.DevicePtr;
        ulong apPtr = apT.DevicePtr;
        ulong bpPtr = bpT.DevicePtr;
        ulong aPtr = aT.DevicePtr;
        ulong dbPtr = dbT.DevicePtr;

        var func = _compositeModule.GetFunction("compute_decay_beta");
        uint grid = (uint)((groupCount + BlockSize - 1) / BlockSize);
        nint* kArgs = stackalloc nint[7];
        kArgs[0] = (nint)(&decPtr);
        kArgs[1] = (nint)(&betPtr);
        kArgs[2] = (nint)(&apPtr);
        kArgs[3] = (nint)(&bpPtr);
        kArgs[4] = (nint)(&aPtr);
        kArgs[5] = (nint)(&dbPtr);
        kArgs[6] = (nint)(&groupCount);
        _stream.Launch(func, grid, 1, 1, (uint)BlockSize, 1, 1, 0, kArgs);

    }

    /// <inheritdoc />
    public unsafe void DeltaNetStep(ITensor output, ITensor q, ITensor k, ITensor v,
        ITensor state, ITensor decay, ITensor beta,
        ITensor normWeight, int groupCount, int headDim, float scale, float normEps)
    {
        var outT = (CudaTensor)output;
        var qT = (CudaTensor)q;
        var kT = (CudaTensor)k;
        var vT = (CudaTensor)v;
        var sT = (CudaTensor)state;
        var decT = (CudaTensor)decay;
        var betT = (CudaTensor)beta;
        var nwT = (CudaTensor)normWeight;
        ulong outPtr = outT.DevicePtr;
        ulong qPtr = qT.DevicePtr;
        ulong kPtr = kT.DevicePtr;
        ulong vPtr = vT.DevicePtr;
        ulong sPtr = sT.DevicePtr;
        ulong decPtr = decT.DevicePtr;
        ulong betPtr = betT.DevicePtr;
        ulong nwPtr = nwT.DevicePtr;

        var func = _compositeModule.GetFunction("deltanet_step");
        uint sharedMem = (uint)(BlockSize * sizeof(float));
        nint* kArgs = stackalloc nint[12];
        kArgs[0] = (nint)(&outPtr);
        kArgs[1] = (nint)(&qPtr);
        kArgs[2] = (nint)(&kPtr);
        kArgs[3] = (nint)(&vPtr);
        kArgs[4] = (nint)(&sPtr);
        kArgs[5] = (nint)(&decPtr);
        kArgs[6] = (nint)(&betPtr);
        kArgs[7] = (nint)(&nwPtr);
        kArgs[8] = (nint)(&groupCount);
        kArgs[9] = (nint)(&headDim);
        kArgs[10] = (nint)(&scale);
        kArgs[11] = (nint)(&normEps);
        _stream.Launch(func, (uint)groupCount, 1, 1, (uint)BlockSize, 1, 1, sharedMem, kArgs);

    }

    /// <inheritdoc />
    public unsafe void SplitQKV(ITensor q, ITensor k, ITensor v, ITensor qkv, int innerSize)
    {
        var qT = (CudaTensor)q;
        var kT = (CudaTensor)k;
        var vT = (CudaTensor)v;
        var qkvT = (CudaTensor)qkv;
        ulong qPtr = qT.DevicePtr;
        ulong kPtr = kT.DevicePtr;
        ulong vPtr = vT.DevicePtr;
        ulong qkvPtr = qkvT.DevicePtr;

        var func = _compositeModule.GetFunction("split_qkv");
        uint grid = (uint)((innerSize + BlockSize - 1) / BlockSize);
        nint* kArgs = stackalloc nint[5];
        kArgs[0] = (nint)(&qPtr);
        kArgs[1] = (nint)(&kPtr);
        kArgs[2] = (nint)(&vPtr);
        kArgs[3] = (nint)(&qkvPtr);
        kArgs[4] = (nint)(&innerSize);
        _stream.Launch(func, grid, 1, 1, (uint)BlockSize, 1, 1, 0, kArgs);

    }

    /// <inheritdoc />
    public unsafe void SiLUGate(ITensor output, ITensor data, ITensor gate)
    {
        var outT = (CudaTensor)output;
        var dT = (CudaTensor)data;
        var gT = (CudaTensor)gate;
        ulong outPtr = outT.DevicePtr;
        ulong dPtr = dT.DevicePtr;
        ulong gPtr = gT.DevicePtr;
        int n = (int)data.ElementCount;

        var func = _compositeModule.GetFunction("silu_gate");
        uint grid = (uint)((n + BlockSize - 1) / BlockSize);
        nint* kArgs = stackalloc nint[4];
        kArgs[0] = (nint)(&outPtr);
        kArgs[1] = (nint)(&dPtr);
        kArgs[2] = (nint)(&gPtr);
        kArgs[3] = (nint)(&n);
        _stream.Launch(func, grid, 1, 1, (uint)BlockSize, 1, 1, 0, kArgs);

    }

    /// <inheritdoc />
    public void ZeroTensor(ITensor tensor)
    {
        EnsureContext();
        var t = (CudaTensor)tensor;
        CudaApi.Check(CudaApi.MemsetD8(t.DevicePtr, 0, (ulong)t.ByteSize), "cuMemsetD8");
    }

    /// <inheritdoc />
    public void CopyTensorBytes(ITensor dst, ITensor src, long byteCount)
    {
        EnsureContext();
        var dstT = (CudaTensor)dst;
        var srcT = (CudaTensor)src;
        CudaApi.Check(CudaApi.MemcpyDtoD(dstT.DevicePtr, srcT.DevicePtr, (ulong)byteCount), "cuMemcpyDtoD");
    }

    /// <inheritdoc />
    public ITensor CreateHostTensor(string name, GgmlType type, ReadOnlySpan<long> dimensions)
    {
        EnsureContext();
        return new CudaTensor(name, type, dimensions, pinned: true);
    }

    /// <inheritdoc />
    public unsafe void FillTensor(ITensor tensor, float value)
    {
        var t = (CudaTensor)tensor;
        ulong ptr = t.DevicePtr;
        int n = (int)tensor.ElementCount;

        var func = _elementwiseModule.GetFunction("fill_tensor");
        uint grid = (uint)((n + BlockSize - 1) / BlockSize);
        nint* kArgs = stackalloc nint[3];
        kArgs[0] = (nint)(&ptr);
        kArgs[1] = (nint)(&n);
        kArgs[2] = (nint)(&value);
        _stream.Launch(func, grid, 1, 1, (uint)BlockSize, 1, 1, 0, kArgs);
    }

    /// <inheritdoc />
    public unsafe void SquaredReLU(ITensor data)
    {
        var dT = (CudaTensor)data;
        ulong dPtr = dT.DevicePtr;
        int n = (int)data.ElementCount;

        var func = _elementwiseModule.GetFunction("squared_relu");
        uint grid = (uint)((n + BlockSize - 1) / BlockSize);
        nint* kArgs = stackalloc nint[2];
        kArgs[0] = (nint)(&dPtr);
        kArgs[1] = (nint)(&n);
        _stream.Launch(func, grid, 1, 1, (uint)BlockSize, 1, 1, 0, kArgs);
    }

    // ── Generic CPU Fallback ──────────────────────────────────────────────────

    private unsafe void GenericCpuFallbackMatMul(CudaTensor outT, CudaTensor aT, CudaTensor bT, int M, int K, int N)
    {
        _stream.Synchronize();

        // Download A (F32) and B (quantized) to CPU
        var aData = new float[M * K];
        aT.DownloadTo(aData);

        var bBytes = new byte[bT.ByteSize];
        bT.Memory.CopyToHost(bBytes);

        // Dequant+matmul on CPU
        using var cpuBackend = new Cpu.CpuBackend();
        var aBytes = new byte[aData.Length * 4];
        Buffer.BlockCopy(aData, 0, aBytes, 0, aBytes.Length);
        using var cpuA = cpuBackend.LoadTensor("a", GgmlType.F32, [M * K], aBytes);
        using var cpuB = cpuBackend.LoadTensor("b", bT.Type, bT.Dimensions, bBytes);
        using var cpuOut = cpuBackend.CreateTensor("out", GgmlType.F32, [M * N]);
        cpuBackend.MatMul(cpuOut, cpuA, cpuB, M, K, N);

        // Upload result
        var result = cpuOut.AsFloatSpan();
        outT.UploadFrom(result);
    }

    private unsafe void GenericCpuFallbackEmbeddingLookup(CudaTensor outT, CudaTensor tableT, int hiddenDim, int tokenId)
    {
        _stream.Synchronize();

        var tableBytes = new byte[tableT.ByteSize];
        tableT.Memory.CopyToHost(tableBytes);

        using var cpuBackend = new Cpu.CpuBackend();
        using var cpuTable = cpuBackend.LoadTensor("t", tableT.Type, tableT.Dimensions, tableBytes);
        using var cpuOut = cpuBackend.CreateTensor("out", GgmlType.F32, [hiddenDim]);
        cpuBackend.EmbeddingLookup(cpuOut, cpuTable, tokenId);

        var result = cpuOut.AsFloatSpan();
        outT.UploadFrom(result);
    }

    // Persistent argmax result buffer (avoid per-token CUDA alloc/dealloc)
    private CudaTensor? _argmaxResult;
    private readonly float[] _argmaxHostBuf = new float[1];

    /// <inheritdoc />
    public unsafe int ArgMax(ITensor tensor) => ArgMax(tensor, (int)tensor.ElementCount);

    public unsafe int ArgMax(ITensor tensor, int count)
    {
        EnsureContext();
        var t = (CudaTensor)tensor;
        ulong tPtr = t.DevicePtr;
        int n = Math.Min(count, (int)tensor.ElementCount);

        _argmaxResult ??= new CudaTensor("argmax_result", Gguf.GgmlType.F32, [1]);

        ulong rPtr = _argmaxResult.DevicePtr;
        var func = _elementwiseModule.GetFunction("argmax");
        uint sharedMem = (uint)(BlockSize * 2 * sizeof(float));
        nint* kArgs = stackalloc nint[3];
        kArgs[0] = (nint)(&rPtr);
        kArgs[1] = (nint)(&tPtr);
        kArgs[2] = (nint)(&n);
        _stream.Launch(func, 1, 1, 1, (uint)BlockSize, 1, 1, sharedMem, kArgs);

        // Download just 1 float (4 bytes) instead of full vocab tensor
        _stream.Synchronize();
        _argmaxResult.DownloadTo(_argmaxHostBuf);
        return (int)_argmaxHostBuf[0];
    }

    /// <inheritdoc />
    public void CopyTensorRegion(ITensor dst, ITensor src, int srcOffset, int count)
    {
        EnsureContext();
        var dstT = (CudaTensor)dst;
        var srcT = (CudaTensor)src;
        ulong srcPtr = srcT.DevicePtr + (ulong)(srcOffset * sizeof(float));
        CudaApi.Check(CudaApi.MemcpyDtoDAsync(dstT.DevicePtr, srcPtr,
            (ulong)(count * sizeof(float)), _stream.Handle), "cuMemcpyDtoDAsync");
    }

    /// <inheritdoc />
    public void CopyTensorSlice(ITensor dst, int dstOffset, ITensor src, int srcOffset, int count)
    {
        _q8_1CacheGeneration++; // destination data changes — invalidate Q8_1 cache
        EnsureContext();
        var dstT = (CudaTensor)dst;
        var srcT = (CudaTensor)src;
        ulong dstPtr = dstT.DevicePtr + (ulong)(dstOffset * sizeof(float));
        ulong srcPtr = srcT.DevicePtr + (ulong)(srcOffset * sizeof(float));
        CudaApi.Check(CudaApi.MemcpyDtoDAsync(dstPtr, srcPtr,
            (ulong)(count * sizeof(float)), _stream.Handle), "cuMemcpyDtoDAsync");
    }

    // ── GPU-Native DeltaNet Operations ──────────────────────────────────────

    /// <inheritdoc />
    public unsafe void SplitUnequalQKV(ITensor q, ITensor k, ITensor v, ITensor qkv, int keyDim, int valueDim)
    {
        var qT = (CudaTensor)q;
        var kT = (CudaTensor)k;
        var vT = (CudaTensor)v;
        var qkvT = (CudaTensor)qkv;
        ulong qPtr = qT.DevicePtr;
        ulong kPtr = kT.DevicePtr;
        ulong vPtr = vT.DevicePtr;
        ulong qkvPtr = qkvT.DevicePtr;

        var func = _compositeModule.GetFunction("split_unequal_qkv");
        uint grid = (uint)((valueDim + BlockSize - 1) / BlockSize);
        nint* kArgs = stackalloc nint[6];
        kArgs[0] = (nint)(&qPtr);
        kArgs[1] = (nint)(&kPtr);
        kArgs[2] = (nint)(&vPtr);
        kArgs[3] = (nint)(&qkvPtr);
        kArgs[4] = (nint)(&keyDim);
        kArgs[5] = (nint)(&valueDim);
        _stream.Launch(func, grid, 1, 1, (uint)BlockSize, 1, 1, 0, kArgs);
    }

    /// <inheritdoc />
    public unsafe void RepeatTile(ITensor tensor, int numHeads, int headDim, int factor)
    {
        var t = (CudaTensor)tensor;
        ulong ptr = t.DevicePtr;
        int srcSize = numHeads * headDim;
        int dstSize = numHeads * factor * headDim;

        var func = _compositeModule.GetFunction("repeat_tile");
        uint grid = (uint)((dstSize + BlockSize - 1) / BlockSize);
        nint* kArgs = stackalloc nint[3];
        kArgs[0] = (nint)(&ptr);
        kArgs[1] = (nint)(&srcSize);
        kArgs[2] = (nint)(&dstSize);
        _stream.Launch(func, grid, 1, 1, (uint)BlockSize, 1, 1, 0, kArgs);
    }

    /// <inheritdoc />
    public unsafe void SplitSwiGLU(ITensor output, ITensor fusedInput, int N)
    {
        var outT = (CudaTensor)output;
        var inT = (CudaTensor)fusedInput;
        ulong outPtr = outT.DevicePtr;
        ulong inPtr = inT.DevicePtr;

        var func = _elementwiseModule.GetFunction("split_swiglu");
        uint grid = (uint)((N + BlockSize - 1) / BlockSize);
        nint* kArgs = stackalloc nint[3];
        kArgs[0] = (nint)(&outPtr);
        kArgs[1] = (nint)(&inPtr);
        kArgs[2] = (nint)(&N);
        _stream.Launch(func, grid, 1, 1, (uint)BlockSize, 1, 1, 0, kArgs);
    }

    /// <inheritdoc />
    public unsafe void PostQkvNormRopeCache(ITensor qOut, ITensor kOut, ITensor vOut,
        ITensor fusedQkv, ITensor kCache, ITensor vCache,
        int qDim, int kDim, int vDim,
        int numHeads, int numKvHeads, int headDim, int ropeDim,
        int position, float ropeTheta, float normEps,
        int maxSeqLen, int seqLen, ITensor? qNormWeight, ITensor? kNormWeight)
    {
        var qT = (CudaTensor)qOut; var kT = (CudaTensor)kOut; var vT = (CudaTensor)vOut;
        var fT = (CudaTensor)fusedQkv;
        var kcT = (CudaTensor)kCache; var vcT = (CudaTensor)vCache;

        ulong qPtr = qT.DevicePtr, kPtr = kT.DevicePtr, vPtr = vT.DevicePtr;
        ulong fPtr = fT.DevicePtr;
        ulong kcPtr = kcT.DevicePtr, vcPtr = vcT.DevicePtr;
        ulong qnPtr = qNormWeight != null ? ((CudaTensor)qNormWeight).DevicePtr : 0;
        ulong knPtr = kNormWeight != null ? ((CudaTensor)kNormWeight).DevicePtr : 0;
        int cacheIsFp16 = kCache.Type == Gguf.GgmlType.F16 ? 1 : 0;

        var func = _elementwiseModule.GetFunction("post_qkv_norm_rope_cache");
        uint sharedMem = (uint)(BlockSize * sizeof(float));
        nint* kArgs = stackalloc nint[18];
        kArgs[0] = (nint)(&qPtr);
        kArgs[1] = (nint)(&kPtr);
        kArgs[2] = (nint)(&vPtr);
        kArgs[3] = (nint)(&fPtr);
        kArgs[4] = (nint)(&kcPtr);
        kArgs[5] = (nint)(&vcPtr);
        kArgs[6] = (nint)(&qDim);
        kArgs[7] = (nint)(&kDim);
        kArgs[8] = (nint)(&vDim);
        kArgs[9] = (nint)(&numHeads);
        kArgs[10] = (nint)(&numKvHeads);
        kArgs[11] = (nint)(&headDim);
        kArgs[12] = (nint)(&ropeDim);
        kArgs[13] = (nint)(&position);
        kArgs[14] = (nint)(&ropeTheta);
        kArgs[15] = (nint)(&normEps);
        kArgs[16] = (nint)(&maxSeqLen);
        kArgs[17] = (nint)(&cacheIsFp16);
        // qNormWeight and kNormWeight as additional args
        nint* allArgs = stackalloc nint[20];
        for (int i = 0; i < 18; i++) allArgs[i] = kArgs[i];
        allArgs[18] = (nint)(&qnPtr);
        allArgs[19] = (nint)(&knPtr);
        _stream.Launch(func, (uint)numHeads, 1, 1, (uint)BlockSize, 1, 1, sharedMem, allArgs);
    }

    // ── Fused Operations ─────────────────────────────────────────────────────

    /// <inheritdoc />
    public unsafe void RmsNormResidual(ITensor output, ITensor residual, ITensor input, ITensor weight, float eps)
    {
        _q8_1CacheGeneration++;
        _q8_1FusedReady = false;
        var outT = (CudaTensor)output;
        var resT = (CudaTensor)residual;
        var inT = (CudaTensor)input;
        var wT = (CudaTensor)weight;
        ulong outPtr = outT.DevicePtr;
        ulong resPtr = resT.DevicePtr;
        ulong inPtr = inT.DevicePtr;
        ulong wPtr = wT.DevicePtr;
        int n = (int)weight.ElementCount; // hidden dim
        int totalElements = (int)input.ElementCount;
        int M = totalElements / n;
        uint sharedMem = (uint)(BlockSize * sizeof(float));

        if (M > 1)
        {
            var func = _elementwiseModule.GetFunction("batched_rms_norm_residual");
            nint* kArgs = stackalloc nint[7];
            kArgs[0] = (nint)(&outPtr); kArgs[1] = (nint)(&resPtr); kArgs[2] = (nint)(&inPtr);
            kArgs[3] = (nint)(&wPtr); kArgs[4] = (nint)(&n); kArgs[5] = (nint)(&M); kArgs[6] = (nint)(&eps);
            _stream.Launch(func, (uint)M, 1, 1, (uint)BlockSize, 1, 1, sharedMem, kArgs);
            return;
        }

        if ((_hasQ4_0Weights || _hasQ8_0Weights) && _q8_1Scratch != null)
        {
            ulong q8Ptr = _q8_1Scratch.DevicePtr;
            var func = _elementwiseModule.GetFunction("rms_norm_residual_q8_1");
            nint* kArgs = stackalloc nint[7];
            kArgs[0] = (nint)(&outPtr);
            kArgs[1] = (nint)(&resPtr);
            kArgs[2] = (nint)(&inPtr);
            kArgs[3] = (nint)(&wPtr);
            kArgs[4] = (nint)(&q8Ptr);
            kArgs[5] = (nint)(&n);
            kArgs[6] = (nint)(&eps);
            _stream.Launch(func, 1, 1, 1, (uint)BlockSize, 1, 1, sharedMem, kArgs);
            _q8_1FusedReady = true;
            _q8_1CachedInputPtr = outPtr;
            _q8_1CachedGeneration = _q8_1CacheGeneration;
        }
        else
        {
            var func = _elementwiseModule.GetFunction("rms_norm_residual");
            nint* kArgs = stackalloc nint[6];
            kArgs[0] = (nint)(&outPtr);
            kArgs[1] = (nint)(&resPtr);
            kArgs[2] = (nint)(&inPtr);
            kArgs[3] = (nint)(&wPtr);
            kArgs[4] = (nint)(&n);
            kArgs[5] = (nint)(&eps);
            _stream.Launch(func, 1, 1, 1, (uint)BlockSize, 1, 1, sharedMem, kArgs);
        }
    }

    /// <inheritdoc />
    public unsafe void SwiGLU(ITensor output, ITensor gate, ITensor up)
    {
        var outT = (CudaTensor)output;
        var gT = (CudaTensor)gate;
        var uT = (CudaTensor)up;
        ulong outPtr = outT.DevicePtr;
        ulong gPtr = gT.DevicePtr;
        ulong uPtr = uT.DevicePtr;
        int n = (int)gate.ElementCount;

        var func = _elementwiseModule.GetFunction("swiglu");
        uint grid = (uint)((n + BlockSize - 1) / BlockSize);
        nint* kArgs = stackalloc nint[4];
        kArgs[0] = (nint)(&outPtr);
        kArgs[1] = (nint)(&gPtr);
        kArgs[2] = (nint)(&uPtr);
        kArgs[3] = (nint)(&n);

        // Fused path: SwiGLU + Q8_1 quantization for dp4a matmul (M=1 only)
        // Skip for batched (M>1): Q8_1 scratch is sized for single-token K, not M*N.
        if ((_hasQ4_0Weights || _hasQ8_0Weights) && _q8_1Scratch != null && n <= _q8_1ScratchK)
        {
            ulong q8Ptr = _q8_1Scratch.DevicePtr;
            var qFunc = _elementwiseModule.GetFunction("swiglu_q8_1");
            nint* qArgs = stackalloc nint[5];
            qArgs[0] = (nint)(&outPtr);
            qArgs[1] = (nint)(&q8Ptr);
            qArgs[2] = (nint)(&gPtr);
            qArgs[3] = (nint)(&uPtr);
            qArgs[4] = (nint)(&n);
            _stream.Launch(qFunc, grid, 1, 1, (uint)BlockSize, 1, 1, 0, qArgs);
            _q8_1CachedInputPtr = outPtr;
            _q8_1CachedGeneration = _q8_1CacheGeneration;
            _q8_1FusedReady = true;
        }
        else
        {
            _stream.Launch(func, grid, 1, 1, (uint)BlockSize, 1, 1, 0, kArgs);
        }
    }

    /// <inheritdoc />
    public unsafe void AddRmsNormResidual(ITensor output, ITensor hidden, ITensor residual, ITensor b, ITensor weight, float eps)
    {
        _q8_1CacheGeneration++;
        _q8_1FusedReady = false;
        var outT = (CudaTensor)output;
        var hidT = (CudaTensor)hidden;
        var resT = (CudaTensor)residual;
        var bT = (CudaTensor)b;
        var wT = (CudaTensor)weight;
        ulong outPtr = outT.DevicePtr, hidPtr = hidT.DevicePtr, resPtr = resT.DevicePtr;
        ulong bPtr = bT.DevicePtr, wPtr = wT.DevicePtr;
        int n = (int)weight.ElementCount;
        int totalElements = (int)hidden.ElementCount;
        int M = totalElements / n;
        uint sharedMem = (uint)(BlockSize * sizeof(float));

        if (M > 1)
        {
            var func = _elementwiseModule.GetFunction("batched_add_rms_norm_residual");
            nint* kArgs = stackalloc nint[8];
            kArgs[0] = (nint)(&outPtr); kArgs[1] = (nint)(&hidPtr); kArgs[2] = (nint)(&resPtr);
            kArgs[3] = (nint)(&bPtr); kArgs[4] = (nint)(&wPtr); kArgs[5] = (nint)(&n);
            kArgs[6] = (nint)(&M); kArgs[7] = (nint)(&eps);
            _stream.Launch(func, (uint)M, 1, 1, (uint)BlockSize, 1, 1, sharedMem, kArgs);
            return;
        }

        if ((_hasQ4_0Weights || _hasQ8_0Weights) && _q8_1Scratch != null)
        {
            ulong q8Ptr = _q8_1Scratch.DevicePtr;
            var func = _elementwiseModule.GetFunction("add_rms_norm_residual_q8_1");
            nint* kArgs = stackalloc nint[8];
            kArgs[0] = (nint)(&outPtr);
            kArgs[1] = (nint)(&hidPtr);
            kArgs[2] = (nint)(&resPtr);
            kArgs[3] = (nint)(&bPtr);
            kArgs[4] = (nint)(&wPtr);
            kArgs[5] = (nint)(&q8Ptr);
            kArgs[6] = (nint)(&n);
            kArgs[7] = (nint)(&eps);
            _stream.Launch(func, 1, 1, 1, (uint)BlockSize, 1, 1, sharedMem, kArgs);
            _q8_1FusedReady = true;
            _q8_1CachedInputPtr = outPtr;
            _q8_1CachedGeneration = _q8_1CacheGeneration;
        }
        else
        {
            var func = _elementwiseModule.GetFunction("add_rms_norm_residual");
            nint* kArgs = stackalloc nint[7];
            kArgs[0] = (nint)(&outPtr);
            kArgs[1] = (nint)(&hidPtr);
            kArgs[2] = (nint)(&resPtr);
            kArgs[3] = (nint)(&bPtr);
            kArgs[4] = (nint)(&wPtr);
            kArgs[5] = (nint)(&n);
            kArgs[6] = (nint)(&eps);
            _stream.Launch(func, 1, 1, 1, (uint)BlockSize, 1, 1, sharedMem, kArgs);
        }
    }

    public unsafe void AddRmsNorm(ITensor output, ITensor hidden, ITensor a, ITensor b, ITensor weight, float eps)
    {
        _q8_1CacheGeneration++;
        _q8_1FusedReady = false;
        var outT = (CudaTensor)output;
        var hidT = (CudaTensor)hidden;
        var aT = (CudaTensor)a;
        var bT = (CudaTensor)b;
        var wT = (CudaTensor)weight;
        ulong outPtr = outT.DevicePtr;
        ulong hidPtr = hidT.DevicePtr;
        ulong aPtr = aT.DevicePtr;
        ulong bPtr = bT.DevicePtr;
        ulong wPtr = wT.DevicePtr;
        int n = (int)weight.ElementCount;
        int totalElements = (int)a.ElementCount;
        int M = totalElements / n;
        uint sharedMem = (uint)(BlockSize * sizeof(float));

        if (M > 1)
        {
            var func = _elementwiseModule.GetFunction("batched_add_rms_norm");
            nint* kArgs = stackalloc nint[8];
            kArgs[0] = (nint)(&outPtr); kArgs[1] = (nint)(&hidPtr); kArgs[2] = (nint)(&aPtr);
            kArgs[3] = (nint)(&bPtr); kArgs[4] = (nint)(&wPtr); kArgs[5] = (nint)(&n);
            kArgs[6] = (nint)(&M); kArgs[7] = (nint)(&eps);
            _stream.Launch(func, (uint)M, 1, 1, (uint)BlockSize, 1, 1, sharedMem, kArgs);
            return;
        }

        if ((_hasQ4_0Weights || _hasQ8_0Weights) && _q8_1Scratch != null)
        {
            ulong q8Ptr = _q8_1Scratch.DevicePtr;
            var func = _elementwiseModule.GetFunction("add_rms_norm_q8_1");
            nint* kArgs = stackalloc nint[8];
            kArgs[0] = (nint)(&outPtr);
            kArgs[1] = (nint)(&hidPtr);
            kArgs[2] = (nint)(&aPtr);
            kArgs[3] = (nint)(&bPtr);
            kArgs[4] = (nint)(&wPtr);
            kArgs[5] = (nint)(&q8Ptr);
            kArgs[6] = (nint)(&n);
            kArgs[7] = (nint)(&eps);
            _stream.Launch(func, 1, 1, 1, (uint)BlockSize, 1, 1, sharedMem, kArgs);
            _q8_1FusedReady = true;
            _q8_1CachedInputPtr = outPtr;
            _q8_1CachedGeneration = _q8_1CacheGeneration;
        }
        else
        {
            var func = _elementwiseModule.GetFunction("add_rms_norm");
            nint* kArgs = stackalloc nint[7];
            kArgs[0] = (nint)(&outPtr);
            kArgs[1] = (nint)(&hidPtr);
            kArgs[2] = (nint)(&aPtr);
            kArgs[3] = (nint)(&bPtr);
            kArgs[4] = (nint)(&wPtr);
            kArgs[5] = (nint)(&n);
            kArgs[6] = (nint)(&eps);
            _stream.Launch(func, 1, 1, 1, (uint)BlockSize, 1, 1, sharedMem, kArgs);
        }
    }

    // ── Training Operations (backward pass) ────────────────────────────────────
    // These methods are used by GpuTrainingForwardPass. They do NOT modify any
    // existing inference code paths.

    private CudaModule? _backwardModule;

    private CudaModule BackwardModule
    {
        get
        {
            if (_backwardModule == null)
            {
                var archOpts = new[] { $"--gpu-architecture=compute_{_context.ComputeCapabilityMajor}{_context.ComputeCapabilityMinor}" };
                _backwardModule = CudaModule.FromEmbeddedResource("backward_ops.cu", archOpts);
            }
            return _backwardModule;
        }
    }

    /// <summary>C[M×N] = A[M×K] × B[K×N], both F32.</summary>
    public unsafe void SgemmF32(ITensor c, ITensor a, ITensor b, int M, int K, int N)
    {
        ulong aPtr = ((CudaTensor)a).DevicePtr;
        ulong bPtr = ((CudaTensor)b).DevicePtr;
        ulong cPtr = ((CudaTensor)c).DevicePtr;
        float alpha = 1.0f, beta = 0.0f;
        CublasApi.Check(CublasApi.Sgemm(_cublasHandle,
            CublasApi.CUBLAS_OP_N, CublasApi.CUBLAS_OP_N,
            N, M, K, &alpha, bPtr, N, aPtr, K, &beta, cPtr, N), "SgemmF32");
    }

    /// <summary>C[M×N] = A[M×K] × B^T[K×N], B stored as [N×K], both F32.</summary>
    public unsafe void SgemmTransB(ITensor c, ITensor a, ITensor b, int M, int K, int N)
    {
        ulong aPtr = ((CudaTensor)a).DevicePtr;
        ulong bPtr = ((CudaTensor)b).DevicePtr;
        ulong cPtr = ((CudaTensor)c).DevicePtr;
        float alpha = 1.0f, beta = 0.0f;
        CublasApi.Check(CublasApi.Sgemm(_cublasHandle,
            CublasApi.CUBLAS_OP_T, CublasApi.CUBLAS_OP_N,
            N, M, K, &alpha, bPtr, K, aPtr, K, &beta, cPtr, N), "SgemmTransB");
    }

    /// <summary>C[M×N] = A^T[M×K] × B[K×N], A stored as [K×M], both F32.</summary>
    public unsafe void SgemmTransA(ITensor c, ITensor a, ITensor b, int M, int K, int N)
    {
        ulong aPtr = ((CudaTensor)a).DevicePtr;
        ulong bPtr = ((CudaTensor)b).DevicePtr;
        ulong cPtr = ((CudaTensor)c).DevicePtr;
        float alpha = 1.0f, beta = 0.0f;
        CublasApi.Check(CublasApi.Sgemm(_cublasHandle,
            CublasApi.CUBLAS_OP_N, CublasApi.CUBLAS_OP_T,
            N, M, K, &alpha, bPtr, N, aPtr, M, &beta, cPtr, N), "SgemmTransA");
    }

    /// <summary>C[M×N] += A[M×K] × B[K×N], both F32 (accumulate, beta=1).</summary>
    public unsafe void SgemmF32Accumulate(ITensor c, ITensor a, ITensor b, int M, int K, int N)
    {
        ulong aPtr = ((CudaTensor)a).DevicePtr;
        ulong bPtr = ((CudaTensor)b).DevicePtr;
        ulong cPtr = ((CudaTensor)c).DevicePtr;
        float alpha = 1.0f, beta = 1.0f;
        CublasApi.Check(CublasApi.Sgemm(_cublasHandle,
            CublasApi.CUBLAS_OP_N, CublasApi.CUBLAS_OP_N,
            N, M, K, &alpha, bPtr, N, aPtr, K, &beta, cPtr, N), "SgemmF32Acc");
    }

    /// <summary>C[M×N] += A^T[M×K] × B[K×N], A stored as [K×M] (accumulate).</summary>
    public unsafe void SgemmTransAAccumulate(ITensor c, ITensor a, ITensor b, int M, int K, int N)
    {
        ulong aPtr = ((CudaTensor)a).DevicePtr;
        ulong bPtr = ((CudaTensor)b).DevicePtr;
        ulong cPtr = ((CudaTensor)c).DevicePtr;
        float alpha = 1.0f, beta = 1.0f;
        CublasApi.Check(CublasApi.Sgemm(_cublasHandle,
            CublasApi.CUBLAS_OP_N, CublasApi.CUBLAS_OP_T,
            N, M, K, &alpha, bPtr, N, aPtr, M, &beta, cPtr, N), "SgemmTransAAcc");
    }

    // ── Backward kernel launchers ───────────────────────────────────────────

    public unsafe void SiLUBackward(ITensor dInput, ITensor dOutput, ITensor input)
    {
        ulong diPtr = ((CudaTensor)dInput).DevicePtr;
        ulong doPtr = ((CudaTensor)dOutput).DevicePtr;
        ulong iPtr = ((CudaTensor)input).DevicePtr;
        int n = (int)input.ElementCount;
        var func = BackwardModule.GetFunction("silu_backward");
        uint grid = (uint)((n + BlockSize - 1) / BlockSize);
        nint* k = stackalloc nint[4];
        k[0] = (nint)(&diPtr); k[1] = (nint)(&doPtr); k[2] = (nint)(&iPtr); k[3] = (nint)(&n);
        _stream.Launch(func, grid, 1, 1, (uint)BlockSize, 1, 1, 0, k);
    }

    public unsafe void SwiGLUBackward(ITensor dGate, ITensor dUp, ITensor dOutput,
        ITensor gate, ITensor up)
    {
        ulong dgPtr = ((CudaTensor)dGate).DevicePtr;
        ulong duPtr = ((CudaTensor)dUp).DevicePtr;
        ulong doPtr = ((CudaTensor)dOutput).DevicePtr;
        ulong gPtr = ((CudaTensor)gate).DevicePtr;
        ulong uPtr = ((CudaTensor)up).DevicePtr;
        int n = (int)gate.ElementCount;
        var func = BackwardModule.GetFunction("swiglu_backward");
        uint grid = (uint)((n + BlockSize - 1) / BlockSize);
        nint* k = stackalloc nint[6];
        k[0] = (nint)(&dgPtr); k[1] = (nint)(&duPtr); k[2] = (nint)(&doPtr);
        k[3] = (nint)(&gPtr); k[4] = (nint)(&uPtr); k[5] = (nint)(&n);
        _stream.Launch(func, grid, 1, 1, (uint)BlockSize, 1, 1, 0, k);
    }

    public unsafe void BatchedRmsNormBackward(ITensor dInput, ITensor dOutput,
        ITensor input, ITensor weight, float eps, int dim, int M)
    {
        ulong diPtr = ((CudaTensor)dInput).DevicePtr;
        ulong doPtr = ((CudaTensor)dOutput).DevicePtr;
        ulong iPtr = ((CudaTensor)input).DevicePtr;
        ulong wPtr = ((CudaTensor)weight).DevicePtr;
        var func = BackwardModule.GetFunction("batched_rms_norm_backward");
        uint sharedMem = (uint)(BlockSize * sizeof(float));
        nint* k = stackalloc nint[7];
        k[0] = (nint)(&diPtr); k[1] = (nint)(&doPtr); k[2] = (nint)(&iPtr);
        k[3] = (nint)(&wPtr); k[4] = (nint)(&eps); k[5] = (nint)(&dim); k[6] = (nint)(&M);
        _stream.Launch(func, (uint)M, 1, 1, (uint)BlockSize, 1, 1, sharedMem, k);
    }

    public unsafe void BatchedRoPEBackward(ITensor dData, int T, int numHeads, int headDim,
        int ropeDim, int startPos, float theta)
    {
        ulong ptr = ((CudaTensor)dData).DevicePtr;
        int totalPairs = T * numHeads * (ropeDim / 2);
        uint grid = (uint)((totalPairs + BlockSize - 1) / BlockSize);
        var func = BackwardModule.GetFunction("batched_rope_backward");
        nint* k = stackalloc nint[7];
        k[0] = (nint)(&ptr); k[1] = (nint)(&T); k[2] = (nint)(&numHeads);
        k[3] = (nint)(&headDim); k[4] = (nint)(&ropeDim); k[5] = (nint)(&startPos);
        k[6] = (nint)(&theta);
        _stream.Launch(func, grid, 1, 1, (uint)BlockSize, 1, 1, 0, k);
    }

    public unsafe float CrossEntropyLoss(ITensor dLogits, ITensor logits, ITensor targets,
        int T, int V)
    {
        ulong dlPtr = ((CudaTensor)dLogits).DevicePtr;
        ulong lPtr = ((CudaTensor)logits).DevicePtr;
        ulong tPtr = ((CudaTensor)targets).DevicePtr;
        // Allocate scalar for loss result
        using var lossMem = new CudaDeviceMemory(sizeof(float));
        CudaApi.MemsetD8(lossMem.DevicePtr, 0, sizeof(float));
        ulong lossPtr = lossMem.DevicePtr;
        var func = BackwardModule.GetFunction("cross_entropy_loss");
        uint sharedMem = (uint)(BlockSize * sizeof(float));
        nint* k = stackalloc nint[6];
        k[0] = (nint)(&dlPtr); k[1] = (nint)(&lossPtr); k[2] = (nint)(&lPtr);
        k[3] = (nint)(&tPtr); k[4] = (nint)(&T); k[5] = (nint)(&V);
        _stream.Launch(func, (uint)T, 1, 1, (uint)BlockSize, 1, 1, sharedMem, k);
        // Download scalar loss
        float loss = 0;
        _stream.Synchronize();
        lossMem.CopyToHost(new Span<float>(ref loss));
        return loss;
    }

    public unsafe void CausalGatedAttentionBackward(
        ITensor dQAttn, ITensor dQGate, ITensor dK, ITensor dV,
        ITensor dOutput, ITensor qAttn, ITensor qGate,
        ITensor kData, ITensor vData, ITensor savedProbs, ITensor attnOutput,
        int T, int numHeads, int numKvHeads, int keyDim, int valDim, float scale)
    {
        ulong dqPtr = ((CudaTensor)dQAttn).DevicePtr;
        ulong dgPtr = ((CudaTensor)dQGate).DevicePtr;
        ulong dkPtr = ((CudaTensor)dK).DevicePtr;
        ulong dvPtr = ((CudaTensor)dV).DevicePtr;
        ulong doPtr = ((CudaTensor)dOutput).DevicePtr;
        ulong qaPtr = ((CudaTensor)qAttn).DevicePtr;
        ulong qgPtr = ((CudaTensor)qGate).DevicePtr;
        ulong kdPtr = ((CudaTensor)kData).DevicePtr;
        ulong vdPtr = ((CudaTensor)vData).DevicePtr;
        ulong spPtr = ((CudaTensor)savedProbs).DevicePtr;
        ulong aoPtr = ((CudaTensor)attnOutput).DevicePtr;
        var func = BackwardModule.GetFunction("causal_gated_attention_backward");
        int blocks = numHeads * T;
        uint threads = (uint)Math.Min(T, BlockSize);
        nint* k = stackalloc nint[16];
        k[0] = (nint)(&dqPtr); k[1] = (nint)(&dgPtr); k[2] = (nint)(&dkPtr); k[3] = (nint)(&dvPtr);
        k[4] = (nint)(&doPtr); k[5] = (nint)(&qaPtr); k[6] = (nint)(&qgPtr);
        k[7] = (nint)(&kdPtr); k[8] = (nint)(&vdPtr); k[9] = (nint)(&spPtr); k[10] = (nint)(&aoPtr);
        k[11] = (nint)(&T); k[12] = (nint)(&numHeads); k[13] = (nint)(&numKvHeads);
        k[14] = (nint)(&keyDim); k[15] = (nint)(&valDim);
        // scale passed via shared memory trick? No — add as kernel arg
        // Actually the kernel signature has scale as last param, need 17 args
        // Let's use a larger array
        nint* k2 = stackalloc nint[17];
        for (int i = 0; i < 16; i++) k2[i] = k[i];
        k2[16] = (nint)(&scale);
        _stream.Launch(func, (uint)blocks, 1, 1, threads, 1, 1, 0, k2);
    }

    public unsafe void ElementMulBackward(ITensor dA, ITensor dB, ITensor dC,
        ITensor a, ITensor b)
    {
        ulong daPtr = ((CudaTensor)dA).DevicePtr;
        ulong dbPtr = ((CudaTensor)dB).DevicePtr;
        ulong dcPtr = ((CudaTensor)dC).DevicePtr;
        ulong aPtr = ((CudaTensor)a).DevicePtr;
        ulong bPtr = ((CudaTensor)b).DevicePtr;
        int n = (int)a.ElementCount;
        var func = BackwardModule.GetFunction("element_mul_backward");
        uint grid = (uint)((n + BlockSize - 1) / BlockSize);
        nint* k = stackalloc nint[6];
        k[0] = (nint)(&daPtr); k[1] = (nint)(&dbPtr); k[2] = (nint)(&dcPtr);
        k[3] = (nint)(&aPtr); k[4] = (nint)(&bPtr); k[5] = (nint)(&n);
        _stream.Launch(func, grid, 1, 1, (uint)BlockSize, 1, 1, 0, k);
    }

    public unsafe void AddInPlace(ITensor dst, ITensor src)
    {
        ulong dPtr = ((CudaTensor)dst).DevicePtr;
        ulong sPtr = ((CudaTensor)src).DevicePtr;
        int n = (int)dst.ElementCount;
        var func = BackwardModule.GetFunction("add_inplace");
        uint grid = (uint)((n + BlockSize - 1) / BlockSize);
        nint* k = stackalloc nint[3];
        k[0] = (nint)(&dPtr); k[1] = (nint)(&sPtr); k[2] = (nint)(&n);
        _stream.Launch(func, grid, 1, 1, (uint)BlockSize, 1, 1, 0, k);
    }

    public unsafe void ScaleInPlace(ITensor data, float scale)
    {
        ulong ptr = ((CudaTensor)data).DevicePtr;
        int n = (int)data.ElementCount;
        var func = BackwardModule.GetFunction("scale_inplace");
        uint grid = (uint)((n + BlockSize - 1) / BlockSize);
        nint* k = stackalloc nint[3];
        k[0] = (nint)(&ptr); k[1] = (nint)(&scale); k[2] = (nint)(&n);
        _stream.Launch(func, grid, 1, 1, (uint)BlockSize, 1, 1, 0, k);
    }

    public unsafe void AdamWStep(ITensor param, ITensor grad, ITensor m, ITensor v,
        float lr, float beta1, float beta2, float eps, float weightDecay, float bc1, float bc2)
    {
        ulong pPtr = ((CudaTensor)param).DevicePtr;
        ulong gPtr = ((CudaTensor)grad).DevicePtr;
        ulong mPtr = ((CudaTensor)m).DevicePtr;
        ulong vPtr = ((CudaTensor)v).DevicePtr;
        int n = (int)param.ElementCount;
        var func = BackwardModule.GetFunction("adamw_step");
        uint grid = (uint)((n + BlockSize - 1) / BlockSize);
        nint* k = stackalloc nint[11];
        k[0] = (nint)(&pPtr); k[1] = (nint)(&gPtr); k[2] = (nint)(&mPtr); k[3] = (nint)(&vPtr);
        k[4] = (nint)(&lr); k[5] = (nint)(&beta1); k[6] = (nint)(&beta2); k[7] = (nint)(&eps);
        k[8] = (nint)(&weightDecay); k[9] = (nint)(&bc1); k[10] = (nint)(&bc2);
        // n is the 12th arg
        nint* k2 = stackalloc nint[12];
        for (int i = 0; i < 11; i++) k2[i] = k[i];
        k2[11] = (nint)(&n);
        _stream.Launch(func, grid, 1, 1, (uint)BlockSize, 1, 1, 0, k2);
    }

    public unsafe float GradNormSq(ITensor grad)
    {
        ulong gPtr = ((CudaTensor)grad).DevicePtr;
        int n = (int)grad.ElementCount;
        using var result = new CudaDeviceMemory(sizeof(float));
        CudaApi.MemsetD8(result.DevicePtr, 0, sizeof(float));
        ulong rPtr = result.DevicePtr;
        var func = BackwardModule.GetFunction("grad_norm_sq");
        uint sharedMem = (uint)(BlockSize * sizeof(float));
        uint grid = (uint)((n + BlockSize - 1) / BlockSize);
        nint* k = stackalloc nint[3];
        k[0] = (nint)(&rPtr); k[1] = (nint)(&gPtr); k[2] = (nint)(&n);
        _stream.Launch(func, grid, 1, 1, (uint)BlockSize, 1, 1, sharedMem, k);
        float normSq = 0;
        _stream.Synchronize();
        result.CopyToHost(new Span<float>(ref normSq));
        return normSq;
    }

    public unsafe void TrainingCausalGatedAttention(
        ITensor output, ITensor savedProbs,
        ITensor qAttn, ITensor qGate, ITensor k, ITensor v,
        int T, int numHeads, int numKvHeads, int keyDim, int valDim, float scale)
    {
        ulong oPtr = ((CudaTensor)output).DevicePtr;
        ulong spPtr = ((CudaTensor)savedProbs).DevicePtr;
        ulong qaPtr = ((CudaTensor)qAttn).DevicePtr;
        ulong qgPtr = ((CudaTensor)qGate).DevicePtr;
        ulong kPtr = ((CudaTensor)k).DevicePtr;
        ulong vPtr = ((CudaTensor)v).DevicePtr;
        var func = BackwardModule.GetFunction("training_causal_gated_attention");
        int blocks = numHeads * T;
        uint threads = (uint)Math.Min(Math.Max(T, 32), BlockSize);
        uint sharedMem = (uint)(threads * sizeof(float));
        nint* args = stackalloc nint[12];
        args[0] = (nint)(&oPtr); args[1] = (nint)(&spPtr);
        args[2] = (nint)(&qaPtr); args[3] = (nint)(&qgPtr);
        args[4] = (nint)(&kPtr); args[5] = (nint)(&vPtr);
        args[6] = (nint)(&T); args[7] = (nint)(&numHeads);
        args[8] = (nint)(&numKvHeads); args[9] = (nint)(&keyDim);
        args[10] = (nint)(&valDim); args[11] = (nint)(&scale);
        _stream.Launch(func, (uint)blocks, 1, 1, threads, 1, 1, sharedMem, args);
    }

    public unsafe void TruncatedElementMul(ITensor c, ITensor a, ITensor b,
        int T, int cDim, int aDim, int bDim)
    {
        ulong cPtr = ((CudaTensor)c).DevicePtr;
        ulong aPtr = ((CudaTensor)a).DevicePtr;
        ulong bPtr = ((CudaTensor)b).DevicePtr;
        int total = T * cDim;
        var func = BackwardModule.GetFunction("truncated_element_mul");
        uint grid = (uint)((total + BlockSize - 1) / BlockSize);
        nint* k = stackalloc nint[7];
        k[0] = (nint)(&cPtr); k[1] = (nint)(&aPtr); k[2] = (nint)(&bPtr);
        k[3] = (nint)(&T); k[4] = (nint)(&cDim); k[5] = (nint)(&aDim); k[6] = (nint)(&bDim);
        _stream.Launch(func, grid, 1, 1, (uint)BlockSize, 1, 1, 0, k);
    }

    public unsafe void TruncatedElementMulBackward(ITensor dA, ITensor dB, ITensor dC,
        ITensor a, ITensor b, int T, int cDim, int aDim, int bDim)
    {
        ulong daPtr = ((CudaTensor)dA).DevicePtr;
        ulong dbPtr = ((CudaTensor)dB).DevicePtr;
        ulong dcPtr = ((CudaTensor)dC).DevicePtr;
        ulong aPtr = ((CudaTensor)a).DevicePtr;
        ulong bPtr = ((CudaTensor)b).DevicePtr;
        int total = T * cDim;
        var func = BackwardModule.GetFunction("truncated_element_mul_backward");
        uint grid = (uint)((total + BlockSize - 1) / BlockSize);
        nint* k = stackalloc nint[9];
        k[0] = (nint)(&daPtr); k[1] = (nint)(&dbPtr); k[2] = (nint)(&dcPtr);
        k[3] = (nint)(&aPtr); k[4] = (nint)(&bPtr);
        k[5] = (nint)(&T); k[6] = (nint)(&cDim); k[7] = (nint)(&aDim); k[8] = (nint)(&bDim);
        _stream.Launch(func, grid, 1, 1, (uint)BlockSize, 1, 1, 0, k);
    }

    public unsafe void BatchedEmbeddingLookup(ITensor output, ITensor table, ITensor tokenIds,
        int T, int dim)
    {
        ulong oPtr = ((CudaTensor)output).DevicePtr;
        ulong tPtr = ((CudaTensor)table).DevicePtr;
        ulong idPtr = ((CudaTensor)tokenIds).DevicePtr;
        int total = T * dim;
        var func = BackwardModule.GetFunction("batched_embedding_lookup");
        uint grid = (uint)((total + BlockSize - 1) / BlockSize);
        nint* k = stackalloc nint[5];
        k[0] = (nint)(&oPtr); k[1] = (nint)(&tPtr); k[2] = (nint)(&idPtr);
        k[3] = (nint)(&T); k[4] = (nint)(&dim);
        _stream.Launch(func, grid, 1, 1, (uint)BlockSize, 1, 1, 0, k);
    }

    /// <summary>Synchronize the CUDA stream (wait for all queued operations to complete).</summary>
    public void Synchronize() => _stream.Synchronize();

    /// <summary>Raw handle to the compute stream (for pipeline H2D transfers).</summary>
    internal nint ComputeStreamHandle => _stream.Handle;

    /// <inheritdoc />
    public void Dispose()
    {
        if (!_disposed)
        {
            _context.MakeCurrent();
            if (_graphExec != 0) CudaApi.GraphExecDestroy(_graphExec);
            _argmaxResult?.Dispose();
            _q8_1Scratch?.Dispose();
            _backwardModule?.Dispose();
            if (_cublasHandle != 0) CublasApi.Destroy(_cublasHandle);
            _stream.Dispose();
            _matmulModule.Dispose();
            _elementwiseModule.Dispose();
            _compositeModule.Dispose();
            _context.Dispose();
            _disposed = true;
        }
    }
}
