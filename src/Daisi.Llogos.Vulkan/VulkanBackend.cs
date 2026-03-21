using Daisi.Llogos.Gguf;

namespace Daisi.Llogos.Vulkan;

/// <summary>
/// Vulkan compute backend. Implements IComputeBackend using SPIR-V compute shaders
/// dispatched via the Vulkan API. Cross-platform GPU support for NVIDIA, AMD, Intel.
/// </summary>
public sealed class VulkanBackend : IComputeBackend
{
    private readonly VulkanDevice _device;
    private readonly VulkanPipeline _rmsNormPipeline;
    private readonly VulkanPipeline _softmaxPipeline;
    private readonly VulkanPipeline _siluPipeline;
    private readonly VulkanPipeline _ropePipeline;
    private readonly VulkanPipeline _elementOpsPipeline;
    private readonly VulkanPipeline _matmulPipeline;
    private readonly VulkanPipeline _matmulBdaPipeline;
    private readonly VulkanPipeline _matmulQ8AlignedPipeline;
    private readonly VulkanPipeline _embeddingPipeline;
    private readonly VulkanPipeline _compositePipeline;
    private readonly VulkanPipeline _attentionPipeline;
    private readonly VulkanPipeline _deltanetPipeline;
    private bool _disposed;

    private const int WorkGroupSize = 256;

    public VulkanBackend(int deviceOrdinal = 0)
    {
        _device = new VulkanDevice(deviceOrdinal);

        // Load pipelines from embedded SPIR-V resources
        _rmsNormPipeline = VulkanPipeline.FromEmbeddedSpirV(_device, "elementwise.spv", 3, pushConstantSize: 32);
        _softmaxPipeline = VulkanPipeline.FromEmbeddedSpirV(_device, "softmax.spv", 2, pushConstantSize: 4);
        _siluPipeline = VulkanPipeline.FromEmbeddedSpirV(_device, "silu.spv", 2, pushConstantSize: 4);
        _ropePipeline = VulkanPipeline.FromEmbeddedSpirV(_device, "rope.spv", 2, pushConstantSize: 24);
        _elementOpsPipeline = VulkanPipeline.FromEmbeddedSpirV(_device, "element_ops.spv", 3, pushConstantSize: 8);
        _matmulPipeline = VulkanPipeline.FromEmbeddedSpirV(_device, "dequant_matmul.spv", 4, pushConstantSize: 16);
        _matmulBdaPipeline = VulkanPipeline.FromEmbeddedSpirVBda(_device, "dequant_matmul_bda.spv", pushConstantSize: 40);
        // Dedicated pipeline for aligned Q8_0: 3 bindings (output, input, weight_u32), 8 bytes push constants
        _matmulQ8AlignedPipeline = VulkanPipeline.FromEmbeddedSpirV(_device, "matmul_q8_0_aligned.spv", 3, pushConstantSize: 8);
        _embeddingPipeline = VulkanPipeline.FromEmbeddedSpirV(_device, "embedding.spv", 2, pushConstantSize: 12);
        _compositePipeline = VulkanPipeline.FromEmbeddedSpirV(_device, "composite_ops.spv", 8, pushConstantSize: 32);
        _attentionPipeline = VulkanPipeline.FromEmbeddedSpirV(_device, "gated_attention.spv", 5, pushConstantSize: 32);
        _deltanetPipeline = VulkanPipeline.FromEmbeddedSpirV(_device, "deltanet_step.spv", 8, pushConstantSize: 16);
    }

    public string Name => $"Vulkan ({_device.DeviceName})";

    public void BeginCommands()
    {
        _lastBoundPipeline = null;
        _device.BeginBatch();
    }

    public void FlushCommands()
    {
        _lastBoundPipeline = null;
        _device.EndBatch();
        // Reset pools AFTER batch completes (sets must stay valid during execution)
        _rmsNormPipeline.ResetDescriptorPool();
        _softmaxPipeline.ResetDescriptorPool();
        _siluPipeline.ResetDescriptorPool();
        _ropePipeline.ResetDescriptorPool();
        _elementOpsPipeline.ResetDescriptorPool();
        _matmulPipeline.ResetDescriptorPool();
        _matmulQ8AlignedPipeline.ResetDescriptorPool();
        _embeddingPipeline.ResetDescriptorPool();
        _compositePipeline.ResetDescriptorPool();
        _attentionPipeline.ResetDescriptorPool();
        _deltanetPipeline.ResetDescriptorPool();
    }

    public ITensor CreateTensor(string name, GgmlType type, ReadOnlySpan<long> dimensions) =>
        new VulkanTensor(_device, name, type, dimensions);

    public ITensor LoadTensor(string name, GgmlType type, ReadOnlySpan<long> dimensions, ReadOnlySpan<byte> data)
    {
        // Aligned Q8_0: repack 34→36 bytes for native uint32 reads
        if (type == GgmlType.Q8_0 && dimensions.Length >= 2)
        {
            int blockCount = data.Length / 34;
            var aligned = new byte[blockCount * 36];
            for (int i = 0; i < blockCount; i++)
            {
                int srcOff = i * 34;
                int dstOff = i * 36;
                aligned[dstOff] = data[srcOff];
                aligned[dstOff + 1] = data[srcOff + 1];
                data.Slice(srcOff + 2, 32).CopyTo(aligned.AsSpan(dstOff + 4, 32));
            }
            return new VulkanTensor(_device, name, type, dimensions, aligned, isAlignedQ8_0: true);
        }

        return new VulkanTensor(_device, name, type, dimensions, data);
    }

    // ── Math Operations ──────────────────────────────────────────────────────

    public unsafe void MatMul(ITensor output, ITensor a, ITensor b, int M, int K, int N)
    {
        var outT = (VulkanTensor)output;
        var aT = (VulkanTensor)a;
        var bT = (VulkanTensor)b;

        // Check if this type has a GPU shader implementation
        // Check for aligned Q8_0 (repacked at load time)
        bool isAlignedQ8 = bT is VulkanTensor vt && vt.IsAlignedQ8_0;

        uint? weightTypeOpt = b.Type switch
        {
            GgmlType.F32 => 0u,
            GgmlType.Q8_0 when isAlignedQ8 => 7u, // aligned 36-byte blocks
            GgmlType.Q8_0 => 1u,
            GgmlType.TQ1_0 => 2u,
            GgmlType.I2_S => 3u,
            GgmlType.F16 => 4u,
            GgmlType.Q4_K => 5u,
            GgmlType.Q6_K => 6u,
            _ => null,
        };

        if (weightTypeOpt == null)
        {
            GenericCpuFallbackMatMul(outT, aT, bT, M, K, N);
            return;
        }

        uint weightType = weightTypeOpt.Value;

        // Use dedicated pipeline for aligned Q8_0 (no branching, optimal codegen)
        if (weightType == 7u)
        {
            var buffers = new VulkanBuffer[] { outT.DeviceBuffer, aT.DeviceBuffer, bT.DeviceBuffer };
            var ds = _matmulQ8AlignedPipeline.AllocateDescriptorSet(buffers);
            uint* pc = stackalloc uint[2];
            pc[0] = (uint)K;
            pc[1] = (uint)N;
            uint grid = ((uint)N + 7) / 8; // 8 rows per workgroup
            Dispatch(_matmulQ8AlignedPipeline, ds, pc, 8, grid, 1, 1);
            return;
        }

        // Generic matmul pipeline for other types
        var buffers2 = new VulkanBuffer[] { outT.DeviceBuffer, aT.DeviceBuffer, bT.DeviceBuffer, bT.DeviceBuffer };
        var ds2 = _matmulPipeline.AllocateDescriptorSet(buffers2);

        uint* pc2 = stackalloc uint[4];
        pc2[0] = (uint)M;
        pc2[1] = (uint)K;
        pc2[2] = (uint)N;
        pc2[3] = weightType;

        uint rowsPerWg = weightType switch { 5 or 6 => 8u, 1 => 4u, _ => 1u };
        uint grid2 = ((uint)N + rowsPerWg - 1) / rowsPerWg;
        Dispatch(_matmulPipeline, ds2, pc2, 16, grid2, 1, 1);
    }

    public unsafe void RmsNorm(ITensor output, ITensor input, ITensor weight, float eps)
    {
        var outT = (VulkanTensor)output;
        var inT = (VulkanTensor)input;
        var wT = (VulkanTensor)weight;
        int n = (int)input.ElementCount;

        var buffers = new VulkanBuffer[] { outT.DeviceBuffer, inT.DeviceBuffer, wT.DeviceBuffer };
        var ds = _rmsNormPipeline.AllocateDescriptorSet(buffers);

        uint* pc = stackalloc uint[8];
        pc[0] = (uint)n;
        *(float*)&pc[1] = eps;

        Dispatch(_rmsNormPipeline, ds, pc, 32, 1, 1, 1);
    }

    public unsafe void Softmax(ITensor output, ITensor input)
    {
        var outT = (VulkanTensor)output;
        var inT = (VulkanTensor)input;
        int n = (int)input.ElementCount;

        var buffers = new VulkanBuffer[] { outT.DeviceBuffer, inT.DeviceBuffer };
        var ds = _softmaxPipeline.AllocateDescriptorSet(buffers);

        uint nVal = (uint)n;
        Dispatch(_softmaxPipeline, ds, &nVal, 4, 1, 1, 1);
    }

    public unsafe void SiLU(ITensor output, ITensor input)
    {
        var outT = (VulkanTensor)output;
        var inT = (VulkanTensor)input;
        int n = (int)input.ElementCount;

        var buffers = new VulkanBuffer[] { outT.DeviceBuffer, inT.DeviceBuffer };
        var ds = _siluPipeline.AllocateDescriptorSet(buffers);

        uint nVal = (uint)n;
        uint grid = ((uint)n + WorkGroupSize - 1) / WorkGroupSize;
        Dispatch(_siluPipeline, ds, &nVal, 4, grid, 1, 1);
    }

    public unsafe void RoPE(ITensor q, ITensor k, int headDim, int ropeDim, int positionOffset, float ropeTheta)
    {
        var qT = (VulkanTensor)q;
        var kT = (VulkanTensor)k;
        int qTotal = (int)q.ElementCount;
        int kTotal = (int)k.ElementCount;
        int effectiveRopeDim = ropeDim > 0 ? ropeDim : headDim;

        var buffers = new VulkanBuffer[] { qT.DeviceBuffer, kT.DeviceBuffer };
        var ds = _ropePipeline.AllocateDescriptorSet(buffers);

        // Pack push constants: qTotal, kTotal, headDim, ropeDim, positionOffset, ropeTheta
        uint* pc = stackalloc uint[6];
        pc[0] = (uint)qTotal;
        pc[1] = (uint)kTotal;
        pc[2] = (uint)headDim;
        pc[3] = (uint)effectiveRopeDim;
        *(int*)&pc[4] = positionOffset;
        *(float*)&pc[5] = ropeTheta;

        int maxPairs = Math.Max(qTotal, kTotal) / 2;
        uint grid = ((uint)maxPairs + WorkGroupSize - 1) / WorkGroupSize;
        Dispatch(_ropePipeline, ds, pc, 24, grid, 1, 1);
    }

    public unsafe void ElementMul(ITensor output, ITensor a, ITensor b)
    {
        var outT = (VulkanTensor)output;
        var aT = (VulkanTensor)a;
        var bT = (VulkanTensor)b;
        int n = (int)a.ElementCount;

        var buffers = new VulkanBuffer[] { outT.DeviceBuffer, aT.DeviceBuffer, bT.DeviceBuffer };
        var ds = _elementOpsPipeline.AllocateDescriptorSet(buffers);

        uint* pc = stackalloc uint[2];
        pc[0] = (uint)n;
        pc[1] = 0; // op = mul

        uint grid = ((uint)n + WorkGroupSize - 1) / WorkGroupSize;
        Dispatch(_elementOpsPipeline, ds, pc, 8, grid, 1, 1);
    }

    public unsafe void ElementAdd(ITensor output, ITensor a, ITensor b)
    {
        var outT = (VulkanTensor)output;
        var aT = (VulkanTensor)a;
        var bT = (VulkanTensor)b;
        int n = (int)a.ElementCount;

        var buffers = new VulkanBuffer[] { outT.DeviceBuffer, aT.DeviceBuffer, bT.DeviceBuffer };
        var ds = _elementOpsPipeline.AllocateDescriptorSet(buffers);

        uint* pc = stackalloc uint[2];
        pc[0] = (uint)n;
        pc[1] = 1; // op = add

        uint grid = ((uint)n + WorkGroupSize - 1) / WorkGroupSize;
        Dispatch(_elementOpsPipeline, ds, pc, 8, grid, 1, 1);
    }

    public unsafe void EmbeddingLookup(ITensor output, ITensor table, int tokenId)
    {
        var outT = (VulkanTensor)output;
        var tableT = (VulkanTensor)table;
        int hiddenDim = (int)table.Dimensions[0];

        var buffers = new VulkanBuffer[] { outT.DeviceBuffer, tableT.DeviceBuffer };
        var ds = _embeddingPipeline.AllocateDescriptorSet(buffers);

        bool isAligned = tableT is VulkanTensor vt2 && vt2.IsAlignedQ8_0;
        uint? tableTypeOpt = table.Type switch
        {
            GgmlType.F32 => 0u,
            GgmlType.Q8_0 when isAligned => 4u, // aligned 36-byte blocks
            GgmlType.Q8_0 => 1u,
            GgmlType.F16 => 2u,
            GgmlType.Q4_K => 3u,
            _ => null,
        };

        if (tableTypeOpt == null)
        {
            GenericCpuFallbackEmbeddingLookup(outT, tableT, hiddenDim, tokenId);
            return;
        }

        uint tableType = tableTypeOpt.Value;
        uint* pc = stackalloc uint[3];
        pc[0] = (uint)hiddenDim;
        *(int*)&pc[1] = tokenId;
        pc[2] = tableType;

        uint grid = ((uint)hiddenDim + WorkGroupSize - 1) / WorkGroupSize;
        Dispatch(_embeddingPipeline, ds, pc, 12, grid, 1, 1);
    }

    // ── Composite Operations ─────────────────────────────────────────────────

    public void CopyTensor(ITensor dst, ITensor src)
    {
        var dstT = (VulkanTensor)dst;
        var srcT = (VulkanTensor)src;
        CopyBuffer(srcT.DeviceBuffer, dstT.DeviceBuffer, (ulong)src.ByteSize);
    }

    public unsafe void SiLUInPlace(ITensor data)
    {
        DispatchComposite(0, (uint)data.ElementCount, 0, 0, 0, 0, 0, 0,
            ((VulkanTensor)data).DeviceBuffer);
    }

    public unsafe void L2NormGroups(ITensor data, int numGroups, int groupDim)
    {
        DispatchComposite(4, (uint)numGroups, (uint)groupDim, 0, 0, 0, 0, 0,
            ((VulkanTensor)data).DeviceBuffer,
            gridOverride: (uint)numGroups);
    }

    public unsafe void PerHeadRmsNorm(ITensor data, ITensor weight, int numHeads, int headDim, float eps)
    {
        uint epsBits;
        *(float*)&epsBits = eps;
        DispatchComposite(5, (uint)numHeads, (uint)headDim, epsBits, 0, 0, 0, 0,
            ((VulkanTensor)data).DeviceBuffer,
            ((VulkanTensor)weight).DeviceBuffer,
            gridOverride: (uint)numHeads);
    }

    public unsafe void DeInterleaveQ(ITensor qAttn, ITensor qGate, ITensor qFull, int numHeads, int headDim)
    {
        uint total = (uint)(numHeads * headDim);
        uint grid = (total + WorkGroupSize - 1) / WorkGroupSize;
        DispatchComposite(3, (uint)numHeads, (uint)headDim, 0, 0, 0, 0, 0,
            ((VulkanTensor)qAttn).DeviceBuffer,
            ((VulkanTensor)qGate).DeviceBuffer,
            ((VulkanTensor)qFull).DeviceBuffer,
            gridOverride: grid);
    }

    public unsafe void KvCacheWrite(ITensor kCache, ITensor vCache, ITensor k, ITensor v,
        int nKvHeads, int keyLength, int valueLength, int maxSeqLen, int position)
    {
        uint cacheIsFp16 = kCache.Type == GgmlType.F16 ? 1u : 0u;
        int maxElems;
        if (cacheIsFp16 != 0)
            maxElems = Math.Max(nKvHeads * keyLength, nKvHeads * valueLength) / 2; // pairs
        else
            maxElems = Math.Max(nKvHeads * keyLength, nKvHeads * valueLength);
        uint grid = ((uint)maxElems + WorkGroupSize - 1) / WorkGroupSize;
        DispatchComposite(6, (uint)nKvHeads, (uint)keyLength, (uint)valueLength, (uint)maxSeqLen, (uint)position, cacheIsFp16, 0,
            ((VulkanTensor)kCache).DeviceBuffer,
            ((VulkanTensor)vCache).DeviceBuffer,
            ((VulkanTensor)k).DeviceBuffer,
            ((VulkanTensor)v).DeviceBuffer,
            gridOverride: grid);
    }

    public unsafe void GatedAttention(ITensor output, ITensor qAttn, ITensor qGate,
        ITensor kCache, ITensor vCache,
        int numHeads, int numKvHeads, int keyLength, int valueLength,
        int maxSeqLen, int seqLen, float scale)
    {
        var outT = (VulkanTensor)output;
        var qaT = (VulkanTensor)qAttn;
        var qgT = (VulkanTensor)qGate;
        var kcT = (VulkanTensor)kCache;
        var vcT = (VulkanTensor)vCache;

        var buffers = new VulkanBuffer[] { outT.DeviceBuffer, qaT.DeviceBuffer, qgT.DeviceBuffer, kcT.DeviceBuffer, vcT.DeviceBuffer };
        var ds = _attentionPipeline.AllocateDescriptorSet(buffers);

        uint cacheIsFp16 = kCache.Type == GgmlType.F16 ? 1u : 0u;
        uint* pc = stackalloc uint[8];
        pc[0] = (uint)numHeads;
        pc[1] = (uint)numKvHeads;
        pc[2] = (uint)keyLength;
        pc[3] = (uint)valueLength;
        pc[4] = (uint)maxSeqLen;
        pc[5] = (uint)seqLen;
        *(float*)&pc[6] = scale;
        pc[7] = cacheIsFp16;

        Dispatch(_attentionPipeline, ds, pc, 32, (uint)numHeads, 1, 1);
    }

    // Persistent temp buffer for CausalConv1d (must outlive batched command buffers)
    private VulkanBuffer? _conv1dTempBuf;
    private int _conv1dTempSize;

    public unsafe void CausalConv1d(ITensor qkv, ITensor convBuffer, ITensor convWeight, int channels, int kernelSize)
    {
        // Use persistent temp buffer (not `using` — must survive until batch completes)
        if (_conv1dTempBuf == null || _conv1dTempSize < channels)
        {
            _conv1dTempBuf?.Dispose();
            _conv1dTempBuf = new VulkanBuffer(_device, (ulong)(channels * 4), hostVisible: false, transferSrc: true, transferDst: true);
            _conv1dTempSize = channels;
        }
        uint grid = ((uint)channels + WorkGroupSize - 1) / WorkGroupSize;

        DispatchComposite(8, (uint)channels, (uint)kernelSize, 0, 0, 0, 0, 0,
            ((VulkanTensor)qkv).DeviceBuffer,
            ((VulkanTensor)convBuffer).DeviceBuffer,
            ((VulkanTensor)convWeight).DeviceBuffer,
            _conv1dTempBuf,
            gridOverride: grid);
    }

    public unsafe void ComputeDecayBeta(ITensor decay, ITensor beta, ITensor alphaProj, ITensor betaProj,
        ITensor ssmA, ITensor dtBias, int groupCount)
    {
        uint grid = ((uint)groupCount + WorkGroupSize - 1) / WorkGroupSize;
        DispatchComposite(7, (uint)groupCount, 0, 0, 0, 0, 0, 0,
            ((VulkanTensor)decay).DeviceBuffer,
            ((VulkanTensor)beta).DeviceBuffer,
            ((VulkanTensor)alphaProj).DeviceBuffer,
            ((VulkanTensor)betaProj).DeviceBuffer,
            ((VulkanTensor)ssmA).DeviceBuffer,
            ((VulkanTensor)dtBias).DeviceBuffer,
            gridOverride: grid);
    }

    public unsafe void DeltaNetStep(ITensor output, ITensor q, ITensor k, ITensor v,
        ITensor state, ITensor decay, ITensor beta,
        ITensor normWeight, int groupCount, int headDim, float scale, float normEps)
    {
        var outT = (VulkanTensor)output;
        var qT = (VulkanTensor)q;
        var kT = (VulkanTensor)k;
        var vT = (VulkanTensor)v;
        var sT = (VulkanTensor)state;
        var decT = (VulkanTensor)decay;
        var betT = (VulkanTensor)beta;
        var nwT = (VulkanTensor)normWeight;

        var buffers = new VulkanBuffer[] { outT.DeviceBuffer, qT.DeviceBuffer, kT.DeviceBuffer, vT.DeviceBuffer,
            sT.DeviceBuffer, decT.DeviceBuffer, betT.DeviceBuffer, nwT.DeviceBuffer };
        var ds = _deltanetPipeline.AllocateDescriptorSet(buffers);

        uint* pc = stackalloc uint[4];
        pc[0] = (uint)groupCount;
        pc[1] = (uint)headDim;
        *(float*)&pc[2] = scale;
        *(float*)&pc[3] = normEps;

        Dispatch(_deltanetPipeline, ds, pc, 16, (uint)groupCount, 1, 1);
    }

    public unsafe void SplitSwiGLU(ITensor output, ITensor fusedInput, int N)
    {
        uint n = (uint)N;
        uint grid = (n + WorkGroupSize - 1) / WorkGroupSize;
        DispatchComposite(11, n, 0, 0, 0, 0, 0, 0,
            ((VulkanTensor)output).DeviceBuffer,
            ((VulkanTensor)fusedInput).DeviceBuffer,
            gridOverride: grid);
    }

    public unsafe void RmsNormResidual(ITensor output, ITensor residual, ITensor input, ITensor weight, float eps)
    {
        uint n = (uint)input.ElementCount;
        uint epsBits;
        *(float*)&epsBits = eps;
        DispatchComposite(12, n, epsBits, 0, 0, 0, 0, 0,
            ((VulkanTensor)output).DeviceBuffer,
            ((VulkanTensor)residual).DeviceBuffer,
            ((VulkanTensor)input).DeviceBuffer,
            ((VulkanTensor)weight).DeviceBuffer,
            gridOverride: 1);
    }

    public unsafe void AddRmsNormResidual(ITensor output, ITensor hidden, ITensor residual, ITensor b, ITensor weight, float eps)
    {
        uint n = (uint)hidden.ElementCount;
        uint epsBits;
        *(float*)&epsBits = eps;
        DispatchComposite(16, n, epsBits, 0, 0, 0, 0, 0,
            ((VulkanTensor)output).DeviceBuffer,
            ((VulkanTensor)hidden).DeviceBuffer,
            ((VulkanTensor)residual).DeviceBuffer,
            ((VulkanTensor)b).DeviceBuffer,
            ((VulkanTensor)weight).DeviceBuffer,
            gridOverride: 1);
    }

    public unsafe void AddRmsNorm(ITensor output, ITensor hidden, ITensor a, ITensor b, ITensor weight, float eps)
    {
        uint n = (uint)a.ElementCount;
        uint epsBits;
        *(float*)&epsBits = eps;
        DispatchComposite(13, n, epsBits, 0, 0, 0, 0, 0,
            ((VulkanTensor)output).DeviceBuffer,
            ((VulkanTensor)hidden).DeviceBuffer,
            ((VulkanTensor)a).DeviceBuffer,
            ((VulkanTensor)b).DeviceBuffer,
            ((VulkanTensor)weight).DeviceBuffer,
            gridOverride: 1);
    }

    public void SplitUnequalQKV(ITensor q, ITensor k, ITensor v, ITensor qkv, int keyDim, int valueDim)
    {
        // GPU buffer copies with proper offsets — no CPU round-trip
        CopyTensorRegion(q, qkv, 0, keyDim);
        CopyTensorRegion(k, qkv, keyDim, keyDim);
        CopyTensorRegion(v, qkv, keyDim * 2, valueDim);
    }

    public unsafe void RepeatTile(ITensor tensor, int numHeads, int headDim, int factor)
    {
        // Use compute shader to tile data (avoids same-buffer copy issues)
        uint srcElems = (uint)(numHeads * headDim);
        uint grid = (srcElems * (uint)factor + WorkGroupSize - 1) / WorkGroupSize;
        DispatchComposite(15, srcElems, (uint)factor, 0, 0, 0, 0, 0,
            ((VulkanTensor)tensor).DeviceBuffer,
            gridOverride: grid);
    }

    public unsafe void SiLUGate(ITensor output, ITensor data, ITensor gate)
    {
        uint n = (uint)data.ElementCount;
        uint grid = (n + WorkGroupSize - 1) / WorkGroupSize;
        DispatchComposite(1, n, 0, 0, 0, 0, 0, 0,
            ((VulkanTensor)output).DeviceBuffer,
            ((VulkanTensor)data).DeviceBuffer,
            ((VulkanTensor)gate).DeviceBuffer,
            gridOverride: grid);
    }

    public unsafe void SplitQKV(ITensor q, ITensor k, ITensor v, ITensor qkv, int innerSize)
    {
        uint grid = ((uint)innerSize + WorkGroupSize - 1) / WorkGroupSize;
        DispatchComposite(2, (uint)innerSize, 0, 0, 0, 0, 0, 0,
            ((VulkanTensor)q).DeviceBuffer,
            ((VulkanTensor)k).DeviceBuffer,
            ((VulkanTensor)v).DeviceBuffer,
            ((VulkanTensor)qkv).DeviceBuffer,
            gridOverride: grid);
    }

    public void ZeroTensor(ITensor tensor)
    {
        var t = (VulkanTensor)tensor;
        _device.SubmitAndWait(cmd =>
            VulkanApi.CmdFillBuffer(cmd, t.DeviceBuffer.Buffer, 0, VkConst.WholeSize, 0));
    }

    public void CopyTensorRegion(ITensor dst, ITensor src, int srcOffset, int count)
    {
        var dstT = (VulkanTensor)dst;
        var srcT = (VulkanTensor)src;
        ulong srcByteOffset = (ulong)srcOffset * sizeof(float);
        ulong byteCount = (ulong)count * sizeof(float);

        // Record copy without barrier — caller or next dispatch handles dependency
        _device.RecordOrSubmit(cmd =>
        {
            unsafe
            {
                var region = new VkBufferCopy { srcOffset = srcByteOffset, dstOffset = 0, size = byteCount };
                VulkanApi.CmdCopyBuffer(cmd, srcT.DeviceBuffer.Buffer, dstT.DeviceBuffer.Buffer, 1, &region);
            }
        });
        _pendingTransferBarrier = true;
    }

    // Track whether a transfer barrier is needed before the next compute dispatch
    private bool _pendingTransferBarrier;

    public void CopyTensorBytes(ITensor dst, ITensor src, long byteCount)
    {
        var dstT = (VulkanTensor)dst;
        var srcT = (VulkanTensor)src;
        CopyBuffer(srcT.DeviceBuffer, dstT.DeviceBuffer, (ulong)byteCount);
    }

    public ITensor CreateHostTensor(string name, GgmlType type, ReadOnlySpan<long> dimensions) =>
        CreateTensor(name, type, dimensions); // Vulkan handles staging internally

    public unsafe void FillTensor(ITensor tensor, float value)
    {
        uint n = (uint)tensor.ElementCount;
        uint valueBits;
        *(float*)&valueBits = value;
        uint grid = (n + WorkGroupSize - 1) / WorkGroupSize;
        DispatchComposite(9, n, valueBits, 0, 0, 0, 0, 0,
            ((VulkanTensor)tensor).DeviceBuffer,
            gridOverride: grid);
    }

    public unsafe void SquaredReLU(ITensor data)
    {
        uint n = (uint)data.ElementCount;
        uint grid = (n + WorkGroupSize - 1) / WorkGroupSize;
        DispatchComposite(10, n, 0, 0, 0, 0, 0, 0,
            ((VulkanTensor)data).DeviceBuffer,
            gridOverride: grid);
    }

    // Persistent small buffer for ArgMax result (avoids allocation per token)
    private VulkanBuffer? _argmaxResultBuf;

    public unsafe int ArgMax(ITensor tensor)
    {
        var t = (VulkanTensor)tensor;
        uint n = (uint)tensor.ElementCount;

        if (_argmaxResultBuf == null)
            _argmaxResultBuf = new VulkanBuffer(_device, 4, hostVisible: false, transferSrc: true, transferDst: true);

        DispatchComposite(14, n, 0, 0, 0, 0, 0, 0,
            t.DeviceBuffer, _argmaxResultBuf, gridOverride: 1);

        // Read result: single uint32 = argmax index
        // Need to flush batch and download
        _device.EndBatch();
        var staging = new VulkanBuffer(_device, 4, hostVisible: true, transferDst: true);
        _device.SubmitAndWait(cmd =>
        {
            var region = new VkBufferCopy { size = 4 };
            VulkanApi.CmdCopyBuffer(cmd, _argmaxResultBuf.Buffer, staging.Buffer, 1, &region);
        });
        var result = new byte[4];
        staging.Download(result);
        staging.Dispose();
        return (int)BitConverter.ToUInt32(result);
    }

    // ── Private Helpers ──────────────────────────────────────────────────────
    private VulkanPipeline? _lastBoundPipeline;

    private unsafe void Dispatch(VulkanPipeline pipeline, ulong descriptorSet, void* pushConstants,
        uint pushConstantSize, uint groupX, uint groupY, uint groupZ)
    {
        _device.RecordOrSubmit(cmd =>
        {
            // Insert transfer→compute barrier if transfers are pending
            if (_pendingTransferBarrier)
            {
                var xferBarrier = new VkMemoryBarrier
                {
                    sType = 46,
                    srcAccessMask = VkConst.AccessTransferWriteBit,
                    dstAccessMask = VkConst.AccessShaderReadBit,
                };
                VulkanApi.CmdPipelineBarrier(cmd,
                    VkConst.PipelineStageFlagTransferBit, VkConst.PipelineStageFlagComputeShaderBit,
                    0, 1, &xferBarrier, 0, 0, 0, 0);
                _pendingTransferBarrier = false;
            }

            if (!_device.IsBatchRecording || pipeline != _lastBoundPipeline)
            {
                VulkanApi.CmdBindPipeline(cmd, VkConst.PipelineBindPointCompute, pipeline.Pipeline);
                _lastBoundPipeline = pipeline;
            }

            ulong dsLocal = descriptorSet;
            VulkanApi.CmdBindDescriptorSets(cmd, VkConst.PipelineBindPointCompute, pipeline.PipelineLayout,
                0, 1, &dsLocal, 0, null);

            if (pushConstantSize > 0)
                VulkanApi.CmdPushConstants(cmd, pipeline.PipelineLayout, VkConst.ShaderStageComputeBit,
                    0, pushConstantSize, pushConstants);

            VulkanApi.CmdDispatch(cmd, groupX, groupY, groupZ);

            // Memory barrier for compute → compute|transfer
            var barrier = new VkMemoryBarrier
            {
                sType = 46,
                srcAccessMask = VkConst.AccessShaderWriteBit,
                dstAccessMask = VkConst.AccessShaderReadBit | VkConst.AccessTransferReadBit,
            };
            VulkanApi.CmdPipelineBarrier(cmd,
                VkConst.PipelineStageFlagComputeShaderBit,
                VkConst.PipelineStageFlagComputeShaderBit | VkConst.PipelineStageFlagTransferBit,
                0, 1, &barrier, 0, 0, 0, 0);
        });
    }

    /// <summary>
    /// Dispatch using buffer device addresses (no descriptor sets).
    /// </summary>
    private unsafe void DispatchBda(VulkanPipeline pipeline, void* pushConstants,
        uint pushConstantSize, uint groupX, uint groupY, uint groupZ)
    {
        _device.RecordOrSubmit(cmd =>
        {
            if (!_device.IsBatchRecording || pipeline != _lastBoundPipeline)
            {
                VulkanApi.CmdBindPipeline(cmd, VkConst.PipelineBindPointCompute, pipeline.Pipeline);
                _lastBoundPipeline = pipeline;
            }

            VulkanApi.CmdPushConstants(cmd, pipeline.PipelineLayout, VkConst.ShaderStageComputeBit,
                0, pushConstantSize, pushConstants);

            VulkanApi.CmdDispatch(cmd, groupX, groupY, groupZ);

            var barrier = new VkMemoryBarrier
            {
                sType = 46,
                srcAccessMask = VkConst.AccessShaderWriteBit,
                dstAccessMask = VkConst.AccessShaderReadBit,
            };
            VulkanApi.CmdPipelineBarrier(cmd,
                VkConst.PipelineStageFlagComputeShaderBit, VkConst.PipelineStageFlagComputeShaderBit,
                0, 1, &barrier, 0, 0, 0, 0);
        });
    }

    private unsafe void DispatchComposite(uint opcode, uint p0, uint p1, uint p2, uint p3, uint p4, uint p5, uint p6,
        VulkanBuffer buf0, VulkanBuffer? buf1 = null, VulkanBuffer? buf2 = null, VulkanBuffer? buf3 = null,
        VulkanBuffer? buf4 = null, VulkanBuffer? buf5 = null, VulkanBuffer? buf6 = null, VulkanBuffer? buf7 = null,
        uint gridOverride = 0)
    {
        // Need 8 buffers for the composite pipeline — use dummy for unused
        var bufs = new VulkanBuffer[8];
        bufs[0] = buf0;
        bufs[1] = buf1 ?? buf0;
        bufs[2] = buf2 ?? buf0;
        bufs[3] = buf3 ?? buf0;
        bufs[4] = buf4 ?? buf0;
        bufs[5] = buf5 ?? buf0;
        bufs[6] = buf6 ?? buf0;
        bufs[7] = buf7 ?? buf0;

        var ds = _compositePipeline.AllocateDescriptorSet(bufs);

        uint* pc = stackalloc uint[8];
        pc[0] = opcode;
        pc[1] = p0;
        pc[2] = p1;
        pc[3] = p2;
        pc[4] = p3;
        pc[5] = p4;
        pc[6] = p5;
        pc[7] = p6;

        uint grid = gridOverride > 0 ? gridOverride : (p0 + WorkGroupSize - 1) / WorkGroupSize;
        if (grid == 0) grid = 1;

        Dispatch(_compositePipeline, ds, pc, 32, grid, 1, 1);
    }

    private void CopyBuffer(VulkanBuffer src, VulkanBuffer dst, ulong byteCount)
    {
        _device.RecordOrSubmit(cmd =>
        {
            unsafe
            {
                var region = new VkBufferCopy { size = byteCount };
                VulkanApi.CmdCopyBuffer(cmd, src.Buffer, dst.Buffer, 1, &region);

                // Memory barrier after copy
                var barrier = new VkMemoryBarrier
                {
                    sType = 46,
                    srcAccessMask = VkConst.AccessTransferWriteBit,
                    dstAccessMask = VkConst.AccessShaderReadBit,
                };
                VulkanApi.CmdPipelineBarrier(cmd,
                    VkConst.PipelineStageFlagTransferBit, VkConst.PipelineStageFlagComputeShaderBit,
                    0, 1, &barrier, 0, (nint)0, 0, (nint)0);
            }
        });
    }

    // ── Generic CPU Fallback ──────────────────────────────────────────────────

    private void GenericCpuFallbackMatMul(VulkanTensor outT, VulkanTensor aT, VulkanTensor bT, int M, int K, int N)
    {
        var aBytes = new byte[aT.ByteSize];
        aT.DownloadToHost(aBytes);
        var bBytes = new byte[bT.ByteSize];
        bT.DownloadToHost(bBytes);

        using var cpu = new Cpu.CpuBackend();
        using var cpuA = cpu.LoadTensor("a", GgmlType.F32, [M * K], aBytes);
        using var cpuB = cpu.LoadTensor("b", bT.Type, bT.Dimensions, bBytes);
        using var cpuOut = cpu.CreateTensor("out", GgmlType.F32, [M * N]);
        cpu.MatMul(cpuOut, cpuA, cpuB, M, K, N);

        var resultBytes = new byte[M * N * 4];
        Buffer.BlockCopy(cpuOut.AsFloatSpan().ToArray(), 0, resultBytes, 0, resultBytes.Length);
        outT.UploadFromHost(resultBytes);
    }

    private void GenericCpuFallbackEmbeddingLookup(VulkanTensor outT, VulkanTensor tableT, int hiddenDim, int tokenId)
    {
        var tableBytes = new byte[tableT.ByteSize];
        tableT.DownloadToHost(tableBytes);

        using var cpu = new Cpu.CpuBackend();
        using var cpuTable = cpu.LoadTensor("t", tableT.Type, tableT.Dimensions, tableBytes);
        using var cpuOut = cpu.CreateTensor("out", GgmlType.F32, [hiddenDim]);
        cpu.EmbeddingLookup(cpuOut, cpuTable, tokenId);

        var resultBytes = new byte[hiddenDim * 4];
        Buffer.BlockCopy(cpuOut.AsFloatSpan().ToArray(), 0, resultBytes, 0, resultBytes.Length);
        outT.UploadFromHost(resultBytes);
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _argmaxResultBuf?.Dispose();
            _conv1dTempBuf?.Dispose();
            _deltanetPipeline.Dispose();
            _attentionPipeline.Dispose();
            _compositePipeline.Dispose();
            _embeddingPipeline.Dispose();
            _matmulQ8AlignedPipeline.Dispose();
            _matmulBdaPipeline.Dispose();
            _matmulPipeline.Dispose();
            _elementOpsPipeline.Dispose();
            _ropePipeline.Dispose();
            _siluPipeline.Dispose();
            _softmaxPipeline.Dispose();
            _rmsNormPipeline.Dispose();
            _device.Dispose();
            _disposed = true;
        }
    }
}
