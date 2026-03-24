using Daisi.Llogos.Gguf;
using Daisi.Llogos.Inference;
using Daisi.Llogos.Model;
using Daisi.Llogos.Tokenizer;

namespace Daisi.Llogos.Tests.Vulkan;

/// <summary>
/// Diagnostic tests to find which shader crashes when batched.
/// Tests each operation type in isolation within a batched command buffer.
/// </summary>
public class VulkanBatchingTests
{
    private static readonly string ModelPath = @"C:\GGUFS\Qwen3.5-0.8B-Q8_0.gguf";

    [Fact]
    public void BatchTest_EmbeddingOnly()
    {
        if (!File.Exists(ModelPath)) return;
        using var ctx = LoadVulkan();
        if (ctx == null) return;

        // Batch: 5× embedding (same pipeline)
        ctx.Backend.BeginCommands();
        for (int i = 0; i < 5; i++)
            ctx.Backend.EmbeddingLookup(ctx.Hidden, ctx.Weights.TokenEmbedding, 9419);
        ctx.Backend.FlushCommands();
        Assert.True(true, "5× embedding batch works");
    }

    [Fact]
    public void BatchTest_MatMulOnly()
    {
        if (!File.Exists(ModelPath)) return;
        using var ctx = LoadVulkan();
        if (ctx == null) return;

        var saw = (StandardAttentionWeights)ctx.Weights.Layers[0];
        ctx.Backend.EmbeddingLookup(ctx.Hidden, ctx.Weights.TokenEmbedding, 9419);

        // Batch: 3× matmul (Q, K, V projections)
        ctx.Backend.BeginCommands();
        ctx.Backend.MatMul(ctx.QAttn, ctx.Hidden, saw.AttnQ, 1, ctx.Config.HiddenDim, ctx.Config.NumHeads * ctx.Config.KeyLength);
        ctx.Backend.MatMul(ctx.KProj, ctx.Hidden, saw.AttnK, 1, ctx.Config.HiddenDim, ctx.Config.NumKvHeads * ctx.Config.KeyLength);
        ctx.Backend.MatMul(ctx.VProj, ctx.Hidden, saw.AttnV, 1, ctx.Config.HiddenDim, ctx.Config.NumKvHeads * ctx.Config.ValueLength);
        ctx.Backend.FlushCommands();
        Assert.True(true, "3× matmul batch works");
    }

    [Fact]
    public void BatchTest_RmsNormOnly()
    {
        if (!File.Exists(ModelPath)) return;
        using var ctx = LoadVulkan();
        if (ctx == null) return;

        ctx.Backend.EmbeddingLookup(ctx.Hidden, ctx.Weights.TokenEmbedding, 9419);

        // Batch: 3× RmsNorm
        ctx.Backend.BeginCommands();
        ctx.Backend.RmsNorm(ctx.NormOut, ctx.Hidden, ctx.Weights.Layers[0].AttnNorm, ctx.Config.NormEps);
        ctx.Backend.RmsNorm(ctx.NormOut, ctx.Hidden, ctx.Weights.Layers[0].AttnNorm, ctx.Config.NormEps);
        ctx.Backend.RmsNorm(ctx.NormOut, ctx.Hidden, ctx.Weights.Layers[0].AttnNorm, ctx.Config.NormEps);
        ctx.Backend.FlushCommands();
        Assert.True(true, "3× RmsNorm batch works");
    }

    [Fact]
    public void BatchTest_ElementOpsOnly()
    {
        if (!File.Exists(ModelPath)) return;
        using var ctx = LoadVulkan();
        if (ctx == null) return;

        ctx.Backend.EmbeddingLookup(ctx.Hidden, ctx.Weights.TokenEmbedding, 9419);

        // Batch: 5× element ops (add, mul, silu)
        ctx.Backend.BeginCommands();
        ctx.Backend.ElementAdd(ctx.Residual, ctx.Hidden, ctx.Hidden);
        ctx.Backend.ElementMul(ctx.Hidden, ctx.Hidden, ctx.Residual);
        ctx.Backend.SiLU(ctx.Hidden, ctx.Hidden);
        ctx.Backend.ElementAdd(ctx.Hidden, ctx.Hidden, ctx.Residual);
        ctx.Backend.ElementAdd(ctx.Hidden, ctx.Hidden, ctx.Residual);
        ctx.Backend.FlushCommands();
        Assert.True(true, "5× element ops batch works");
    }

    [Fact]
    public void BatchTest_CopyTensorInBatch()
    {
        if (!File.Exists(ModelPath)) return;
        using var ctx = LoadVulkan();
        if (ctx == null) return;

        ctx.Backend.EmbeddingLookup(ctx.Hidden, ctx.Weights.TokenEmbedding, 9419);

        // Batch: embedding + CopyTensor (uses CmdCopyBuffer = transfer)
        ctx.Backend.BeginCommands();
        ctx.Backend.EmbeddingLookup(ctx.Hidden, ctx.Weights.TokenEmbedding, 9419);
        ctx.Backend.CopyTensor(ctx.Residual, ctx.Hidden);
        ctx.Backend.FlushCommands();
        Assert.True(true, "CopyTensor in batch works");
    }

    [Fact]
    public void BatchTest_AttentionComposite()
    {
        if (!File.Exists(ModelPath)) return;
        using var ctx = LoadVulkan();
        if (ctx == null) return;

        ctx.Backend.EmbeddingLookup(ctx.Hidden, ctx.Weights.TokenEmbedding, 9419);
        ctx.Backend.RmsNorm(ctx.NormOut, ctx.Hidden, ctx.Weights.Layers[0].AttnNorm, ctx.Config.NormEps);

        var saw = (StandardAttentionWeights)ctx.Weights.Layers[0];
        ctx.Backend.MatMul(ctx.QAttn, ctx.NormOut, saw.AttnQ, 1, ctx.Config.HiddenDim, ctx.Config.NumHeads * ctx.Config.KeyLength);
        ctx.Backend.MatMul(ctx.KProj, ctx.NormOut, saw.AttnK, 1, ctx.Config.HiddenDim, ctx.Config.NumKvHeads * ctx.Config.KeyLength);
        ctx.Backend.MatMul(ctx.VProj, ctx.NormOut, saw.AttnV, 1, ctx.Config.HiddenDim, ctx.Config.NumKvHeads * ctx.Config.ValueLength);

        if (saw.AttnQNorm != null)
        {
            ctx.Backend.PerHeadRmsNorm(ctx.QAttn, saw.AttnQNorm, ctx.Config.NumHeads, ctx.Config.KeyLength, ctx.Config.NormEps);
            ctx.Backend.PerHeadRmsNorm(ctx.KProj, saw.AttnKNorm!, ctx.Config.NumKvHeads, ctx.Config.KeyLength, ctx.Config.NormEps);
        }

        ctx.Backend.RoPE(ctx.QAttn, ctx.KProj, ctx.Config.KeyLength, ctx.Config.RopeDimCount, 0, ctx.Config.RopeTheta);

        // Batch: KV cache write + attention (composite operations)
        ctx.Backend.BeginCommands();
        ctx.Backend.KvCacheWrite(ctx.KCache, ctx.VCache, ctx.KProj, ctx.VProj,
            ctx.Config.NumKvHeads, ctx.Config.KeyLength, ctx.Config.ValueLength, 16, 0);
        float scale = 1.0f / MathF.Sqrt(ctx.Config.KeyLength);
        ctx.Backend.GatedAttention(ctx.AttnOut, ctx.QAttn, ctx.QGate, ctx.KCache, ctx.VCache,
            ctx.Config.NumHeads, ctx.Config.NumKvHeads, ctx.Config.KeyLength, ctx.Config.ValueLength,
            16, 1, scale);
        ctx.Backend.FlushCommands();
        Assert.True(true, "KvCacheWrite + GatedAttention batch works");
    }

    [Fact]
    public void BatchTest_MixedPipelines()
    {
        if (!File.Exists(ModelPath)) return;
        using var ctx = LoadVulkan();
        if (ctx == null) return;

        ctx.Backend.EmbeddingLookup(ctx.Hidden, ctx.Weights.TokenEmbedding, 9419);

        // Batch: RmsNorm + MatMul + ElementAdd (3 different pipelines)
        ctx.Backend.BeginCommands();
        ctx.Backend.RmsNorm(ctx.NormOut, ctx.Hidden, ctx.Weights.Layers[0].AttnNorm, ctx.Config.NormEps);
        var saw = (StandardAttentionWeights)ctx.Weights.Layers[0];
        ctx.Backend.MatMul(ctx.QAttn, ctx.NormOut, saw.AttnQ, 1, ctx.Config.HiddenDim, ctx.Config.NumHeads * ctx.Config.KeyLength);
        ctx.Backend.ElementAdd(ctx.Hidden, ctx.Hidden, ctx.Hidden);
        ctx.Backend.FlushCommands();
        Assert.True(true, "RmsNorm + MatMul + ElementAdd batch works");
    }

    private VulkanTestContext? LoadVulkan()
    {
        try
        {
            var backend = new Llogos.Vulkan.VulkanBackend();
            using var stream = File.OpenRead(ModelPath);
            var gguf = GgufFile.Read(stream);
            var config = ModelConfig.FromGguf(gguf);
            var weights = ModelLoader.Load(gguf, stream, backend, config);
            var kvCache = new KvCache(backend, config, maxSeqLen: 16);

            return new VulkanTestContext(backend, config, weights, kvCache);
        }
        catch
        {
            return null;
        }
    }

    private sealed class VulkanTestContext : IDisposable
    {
        public Llogos.Vulkan.VulkanBackend Backend { get; }
        public ModelConfig Config { get; }
        public ModelWeights Weights { get; }
        public KvCache KvCache { get; }

        public ITensor Hidden { get; }
        public ITensor Residual { get; }
        public ITensor NormOut { get; }
        public ITensor QAttn { get; }
        public ITensor KProj { get; }
        public ITensor VProj { get; }
        public ITensor AttnOut { get; }
        public ITensor QGate { get; }
        public ITensor KCache { get; }
        public ITensor VCache { get; }

        public VulkanTestContext(Llogos.Vulkan.VulkanBackend backend, ModelConfig config, ModelWeights weights, KvCache kvCache)
        {
            Backend = backend;
            Config = config;
            Weights = weights;
            KvCache = kvCache;

            Hidden = backend.CreateTensor("h", GgmlType.F32, [config.HiddenDim]);
            Residual = backend.CreateTensor("r", GgmlType.F32, [config.HiddenDim]);
            NormOut = backend.CreateTensor("n", GgmlType.F32, [config.HiddenDim]);
            QAttn = backend.CreateTensor("q", GgmlType.F32, [config.NumHeads * config.KeyLength]);
            KProj = backend.CreateTensor("k", GgmlType.F32, [config.NumKvHeads * config.KeyLength]);
            VProj = backend.CreateTensor("v", GgmlType.F32, [config.NumKvHeads * config.ValueLength]);
            AttnOut = backend.CreateTensor("ao", GgmlType.F32, [config.NumHeads * config.ValueLength]);
            QGate = backend.CreateTensor("qg", GgmlType.F32, [config.NumHeads * config.KeyLength]);
            backend.FillTensor(QGate, 88.0f);
            KCache = kvCache.GetKCacheTensor(0);
            VCache = kvCache.GetVCacheTensor(0);
        }

        public void Dispose()
        {
            Hidden.Dispose(); Residual.Dispose(); NormOut.Dispose();
            QAttn.Dispose(); KProj.Dispose(); VProj.Dispose();
            AttnOut.Dispose(); QGate.Dispose();
            KvCache.Dispose(); Weights.Dispose(); Backend.Dispose();
        }
    }
}

// Additional test: DeltaNet operations
// File appended by diagnostic - will be cleaned up
