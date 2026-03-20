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
        _matmulPipeline = VulkanPipeline.FromEmbeddedSpirV(_device, "dequant_matmul.spv", 3, pushConstantSize: 16);
        _embeddingPipeline = VulkanPipeline.FromEmbeddedSpirV(_device, "embedding.spv", 2, pushConstantSize: 12);
        _compositePipeline = VulkanPipeline.FromEmbeddedSpirV(_device, "composite_ops.spv", 8, pushConstantSize: 32);
        _attentionPipeline = VulkanPipeline.FromEmbeddedSpirV(_device, "gated_attention.spv", 5, pushConstantSize: 32);
        _deltanetPipeline = VulkanPipeline.FromEmbeddedSpirV(_device, "deltanet_step.spv", 8, pushConstantSize: 16);
    }

    public string Name => $"Vulkan ({_device.DeviceName})";

    public ITensor CreateTensor(string name, GgmlType type, ReadOnlySpan<long> dimensions) =>
        new VulkanTensor(_device, name, type, dimensions);

    public ITensor LoadTensor(string name, GgmlType type, ReadOnlySpan<long> dimensions, ReadOnlySpan<byte> data) =>
        new VulkanTensor(_device, name, type, dimensions, data);

    // ── Math Operations ──────────────────────────────────────────────────────

    public unsafe void MatMul(ITensor output, ITensor a, ITensor b, int M, int K, int N)
    {
        var outT = (VulkanTensor)output;
        var aT = (VulkanTensor)a;
        var bT = (VulkanTensor)b;

        uint weightType = b.Type == GgmlType.Q8_0 ? 1u : 0u;

        var buffers = new VulkanBuffer[] { outT.DeviceBuffer, aT.DeviceBuffer, bT.DeviceBuffer };
        var ds = _matmulPipeline.AllocateDescriptorSet(buffers);

        uint* pc = stackalloc uint[4];
        pc[0] = (uint)M;
        pc[1] = (uint)K;
        pc[2] = (uint)N;
        pc[3] = weightType;

        Dispatch(_matmulPipeline, ds, pc, 16, (uint)N, 1, 1);
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

        uint tableType = table.Type switch
        {
            GgmlType.Q8_0 => 1u,
            GgmlType.F16 => 2u,
            _ => 0u, // F32
        };
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

    public unsafe void CausalConv1d(ITensor qkv, ITensor convBuffer, ITensor convWeight, int channels, int kernelSize)
    {
        // Need a temp buffer for the conv1d
        using var tempBuf = new VulkanBuffer(_device, (ulong)(channels * 4), hostVisible: false, transferSrc: true, transferDst: true);
        uint grid = ((uint)channels + WorkGroupSize - 1) / WorkGroupSize;

        DispatchComposite(8, (uint)channels, (uint)kernelSize, 0, 0, 0, 0, 0,
            ((VulkanTensor)qkv).DeviceBuffer,
            ((VulkanTensor)convBuffer).DeviceBuffer,
            ((VulkanTensor)convWeight).DeviceBuffer,
            tempBuf,
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

    // ── Private Helpers ──────────────────────────────────────────────────────

    private unsafe void Dispatch(VulkanPipeline pipeline, ulong descriptorSet, void* pushConstants,
        uint pushConstantSize, uint groupX, uint groupY, uint groupZ)
    {
        _device.SubmitAndWait(cmd =>
        {
            VulkanApi.CmdBindPipeline(cmd, VkConst.PipelineBindPointCompute, pipeline.Pipeline);

            ulong dsLocal = descriptorSet;
            VulkanApi.CmdBindDescriptorSets(cmd, VkConst.PipelineBindPointCompute, pipeline.PipelineLayout,
                0, 1, &dsLocal, 0, null);

            if (pushConstantSize > 0)
                VulkanApi.CmdPushConstants(cmd, pipeline.PipelineLayout, VkConst.ShaderStageComputeBit,
                    0, pushConstantSize, pushConstants);

            VulkanApi.CmdDispatch(cmd, groupX, groupY, groupZ);

            // Memory barrier for compute → compute
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
        _device.SubmitAndWait(cmd =>
        {
            unsafe
            {
                var region = new VkBufferCopy { size = byteCount };
                VulkanApi.CmdCopyBuffer(cmd, src.Buffer, dst.Buffer, 1, &region);
            }
        });
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _deltanetPipeline.Dispose();
            _attentionPipeline.Dispose();
            _compositePipeline.Dispose();
            _embeddingPipeline.Dispose();
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
