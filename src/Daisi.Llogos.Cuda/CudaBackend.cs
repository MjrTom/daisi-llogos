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
    private bool _disposed;

    private const int BlockSize = 256;

    public CudaBackend(int deviceOrdinal = 0)
    {
        _context = new CudaContext(deviceOrdinal);
        _stream = new CudaStream();

        // Load kernels from embedded .cu resources (JIT compiled as PTX by the driver)
        _elementwiseModule = CudaModule.FromEmbeddedResource("elementwise.cu");
        _matmulModule = CudaModule.FromEmbeddedResource("dequant_matmul.cu");
        _compositeModule = CudaModule.FromEmbeddedResource("composite_ops.cu");

        // Set the active stream so D2H transfers sync properly
        CudaTensor.ActiveStream = _stream;
    }

    /// <inheritdoc />
    public string Name => $"CUDA ({_context.DeviceName})";

    /// <inheritdoc />
    public ITensor CreateTensor(string name, GgmlType type, ReadOnlySpan<long> dimensions)
        => new CudaTensor(name, type, dimensions);

    /// <inheritdoc />
    public ITensor LoadTensor(string name, GgmlType type, ReadOnlySpan<long> dimensions, ReadOnlySpan<byte> data)
        => new CudaTensor(name, type, dimensions, data);

    /// <inheritdoc />
    public unsafe void MatMul(ITensor output, ITensor a, ITensor b, int M, int K, int N)
    {
        var outT = (CudaTensor)output;
        var aT = (CudaTensor)a;
        var bT = (CudaTensor)b;

        ulong outPtr = outT.DevicePtr;
        ulong aPtr = aT.DevicePtr;
        ulong bPtr = bT.DevicePtr;

        // One block per output neuron, one warp (32 threads) cooperates on dot product
        uint gridX = (uint)N;
        int matmulBlockSize = 32;
        uint sharedMem = sizeof(float);

        if (b.Type == GgmlType.F32)
        {
            var func = _matmulModule.GetFunction("matmul_f32");
            int nVal = N;
            nint* kArgs = stackalloc nint[6];
            kArgs[0] = (nint)(&outPtr);
            kArgs[1] = (nint)(&aPtr);
            kArgs[2] = (nint)(&bPtr);
            kArgs[3] = (nint)(&M);
            kArgs[4] = (nint)(&K);
            kArgs[5] = (nint)(&nVal);
            _stream.Launch(func, gridX, 1, 1, (uint)matmulBlockSize, 1, 1, sharedMem, kArgs);
        }
        else if (b.Type == GgmlType.Q8_0)
        {
            var func = _matmulModule.GetFunction("dequant_matmul_q8_0");
            int nVal = N;
            nint* kArgs = stackalloc nint[6];
            kArgs[0] = (nint)(&outPtr);
            kArgs[1] = (nint)(&aPtr);
            kArgs[2] = (nint)(&bPtr);
            kArgs[3] = (nint)(&M);
            kArgs[4] = (nint)(&K);
            kArgs[5] = (nint)(&nVal);
            _stream.Launch(func, gridX, 1, 1, (uint)matmulBlockSize, 1, 1, sharedMem, kArgs);
        }
        else if (b.Type == GgmlType.I2_S)
        {
            var func = _matmulModule.GetFunction("dequant_matmul_i2s");
            int nVal = N;
            nint* kArgs = stackalloc nint[6];
            kArgs[0] = (nint)(&outPtr);
            kArgs[1] = (nint)(&aPtr);
            kArgs[2] = (nint)(&bPtr);
            kArgs[3] = (nint)(&M);
            kArgs[4] = (nint)(&K);
            kArgs[5] = (nint)(&nVal);
            _stream.Launch(func, gridX, 1, 1, (uint)matmulBlockSize, 1, 1, sharedMem, kArgs);
        }
        else if (b.Type == GgmlType.TQ1_0)
        {
            var func = _matmulModule.GetFunction("dequant_matmul_tq1_0");
            int nVal = N;
            nint* kArgs = stackalloc nint[6];
            kArgs[0] = (nint)(&outPtr);
            kArgs[1] = (nint)(&aPtr);
            kArgs[2] = (nint)(&bPtr);
            kArgs[3] = (nint)(&M);
            kArgs[4] = (nint)(&K);
            kArgs[5] = (nint)(&nVal);
            _stream.Launch(func, gridX, 1, 1, (uint)matmulBlockSize, 1, 1, sharedMem, kArgs);
        }
        else if (b.Type == GgmlType.F16)
        {
            var func = _matmulModule.GetFunction("dequant_matmul_f16");
            int nVal = N;
            nint* kArgs = stackalloc nint[6];
            kArgs[0] = (nint)(&outPtr);
            kArgs[1] = (nint)(&aPtr);
            kArgs[2] = (nint)(&bPtr);
            kArgs[3] = (nint)(&M);
            kArgs[4] = (nint)(&K);
            kArgs[5] = (nint)(&nVal);
            _stream.Launch(func, gridX, 1, 1, (uint)matmulBlockSize, 1, 1, sharedMem, kArgs);
        }
        else if (b.Type == GgmlType.Q4_K)
        {
            var func = _matmulModule.GetFunction("dequant_matmul_q4_k");
            int nVal = N;
            nint* kArgs = stackalloc nint[6];
            kArgs[0] = (nint)(&outPtr);
            kArgs[1] = (nint)(&aPtr);
            kArgs[2] = (nint)(&bPtr);
            kArgs[3] = (nint)(&M);
            kArgs[4] = (nint)(&K);
            kArgs[5] = (nint)(&nVal);
            _stream.Launch(func, gridX, 1, 1, (uint)matmulBlockSize, 1, 1, sharedMem, kArgs);
        }
        else if (b.Type == GgmlType.Q5_K)
        {
            var func = _matmulModule.GetFunction("dequant_matmul_q5_k");
            int nVal = N;
            nint* kArgs = stackalloc nint[6];
            kArgs[0] = (nint)(&outPtr);
            kArgs[1] = (nint)(&aPtr);
            kArgs[2] = (nint)(&bPtr);
            kArgs[3] = (nint)(&M);
            kArgs[4] = (nint)(&K);
            kArgs[5] = (nint)(&nVal);
            _stream.Launch(func, gridX, 1, 1, (uint)matmulBlockSize, 1, 1, sharedMem, kArgs);
        }
        else if (b.Type == GgmlType.Q6_K)
        {
            var func = _matmulModule.GetFunction("dequant_matmul_q6_k");
            int nVal = N;
            nint* kArgs = stackalloc nint[6];
            kArgs[0] = (nint)(&outPtr);
            kArgs[1] = (nint)(&aPtr);
            kArgs[2] = (nint)(&bPtr);
            kArgs[3] = (nint)(&M);
            kArgs[4] = (nint)(&K);
            kArgs[5] = (nint)(&nVal);
            _stream.Launch(func, gridX, 1, 1, (uint)matmulBlockSize, 1, 1, sharedMem, kArgs);
        }
        else
        {
            // Generic fallback: download data, dequantize on CPU, compute on CPU, upload result
            GenericCpuFallbackMatMul(outT, aT, bT, M, K, N);
        }


    }

    /// <inheritdoc />
    public unsafe void RmsNorm(ITensor output, ITensor input, ITensor weight, float eps)
    {
        var outT = (CudaTensor)output;
        var inT = (CudaTensor)input;
        var wT = (CudaTensor)weight;

        ulong outPtr = outT.DevicePtr;
        ulong inPtr = inT.DevicePtr;
        ulong wPtr = wT.DevicePtr;
        int n = (int)input.ElementCount;

        var func = _elementwiseModule.GetFunction("rms_norm");
        nint* kArgs = stackalloc nint[5];
        kArgs[0] = (nint)(&outPtr);
        kArgs[1] = (nint)(&inPtr);
        kArgs[2] = (nint)(&wPtr);
        kArgs[3] = (nint)(&n);
        kArgs[4] = (nint)(&eps);

        uint sharedMem = (uint)(BlockSize * sizeof(float));
        _stream.Launch(func, 1, 1, 1, (uint)BlockSize, 1, 1, sharedMem, kArgs);

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
            var func = _elementwiseModule.GetFunction("embedding_lookup_q8_0");
            nint* kArgs = stackalloc nint[4];
            kArgs[0] = (nint)(&outPtr);
            kArgs[1] = (nint)(&tablePtr);
            kArgs[2] = (nint)(&hiddenDim);
            kArgs[3] = (nint)(&tokenId);
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
        else
        {
            GenericCpuFallbackEmbeddingLookup(outT, tableT, hiddenDim, tokenId);
        }

    }

    // ── Composite Operations ───────────────────────────────────────────────────

    /// <inheritdoc />
    public void CopyTensor(ITensor dst, ITensor src)
    {
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
        var t = (CudaTensor)tensor;
        CudaApi.Check(CudaApi.MemsetD8(t.DevicePtr, 0, (ulong)t.ByteSize), "cuMemsetD8");
    }

    /// <inheritdoc />
    public void CopyTensorBytes(ITensor dst, ITensor src, long byteCount)
    {
        var dstT = (CudaTensor)dst;
        var srcT = (CudaTensor)src;
        CudaApi.Check(CudaApi.MemcpyDtoD(dstT.DevicePtr, srcT.DevicePtr, (ulong)byteCount), "cuMemcpyDtoD");
    }

    /// <inheritdoc />
    public ITensor CreateHostTensor(string name, GgmlType type, ReadOnlySpan<long> dimensions) =>
        new CudaTensor(name, type, dimensions, pinned: true);

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

    /// <inheritdoc />
    public void Dispose()
    {
        if (!_disposed)
        {
            _stream.Dispose();
            _matmulModule.Dispose();
            _elementwiseModule.Dispose();
            _compositeModule.Dispose();
            _context.Dispose();
            _disposed = true;
        }
    }
}
