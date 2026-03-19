using Daisi.Llama.Gguf;

namespace Daisi.Llama.Cuda;

/// <summary>
/// CUDA compute backend. Implements IComputeBackend using GPU kernels via P/Invoke
/// to the CUDA Driver API. Kernels are JIT-compiled from embedded PTX at initialization.
/// </summary>
public sealed class CudaBackend : IComputeBackend
{
    private readonly CudaContext _context;
    private readonly CudaModule _elementwiseModule;
    private readonly CudaModule _matmulModule;
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

        uint gridX = (uint)((N + BlockSize - 1) / BlockSize);

        if (b.Type == GgmlType.F32)
        {
            var func = _matmulModule.GetFunction("matmul_f32");
            nint* args = stackalloc nint[5];
            args[0] = (nint)(&outPtr);
            args[1] = (nint)(&aPtr);
            args[2] = (nint)(&bPtr);
            args[3] = (nint)(&M);
            args[4] = (nint)(&K);
            // N is captured via closure - need to pass it
            int nVal = N;
            args[4] = (nint)(&K);
            // Re-layout args properly
            nint* kArgs = stackalloc nint[6];
            kArgs[0] = (nint)(&outPtr);
            kArgs[1] = (nint)(&aPtr);
            kArgs[2] = (nint)(&bPtr);
            kArgs[3] = (nint)(&M);
            kArgs[4] = (nint)(&K);
            kArgs[5] = (nint)(&nVal);
            _stream.Launch(func, gridX, 1, 1, (uint)BlockSize, 1, 1, 0, kArgs);
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
            _stream.Launch(func, gridX, 1, 1, (uint)BlockSize, 1, 1, 0, kArgs);
        }
        else
        {
            throw new NotSupportedException($"MatMul not implemented for weight type {b.Type}.");
        }

        _stream.Synchronize();
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
        _stream.Synchronize();
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
        _stream.Synchronize();
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
        _stream.Synchronize();
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
        _stream.Synchronize();
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
        _stream.Synchronize();
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
        _stream.Synchronize();
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
        else
        {
            throw new NotSupportedException($"EmbeddingLookup not implemented for type {table.Type}.");
        }

        _stream.Synchronize();
    }

    /// <inheritdoc />
    public void Dispose()
    {
        if (!_disposed)
        {
            _stream.Dispose();
            _matmulModule.Dispose();
            _elementwiseModule.Dispose();
            _context.Dispose();
            _disposed = true;
        }
    }
}
