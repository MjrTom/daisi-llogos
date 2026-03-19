using Daisi.Llama.Cpu;
using Daisi.Llama.Gguf;
using Daisi.Llama.Inference;
using Daisi.Llama.Model;

namespace Daisi.Llama.Tests.Model;

public class ModelConfigTests
{
    [Fact]
    public void FromQwen_CorrectArchitecture()
    {
        if (!TestConstants.ModelExists) return;

        using var stream = File.OpenRead(TestConstants.Qwen35_08B_Q8_0);
        var gguf = GgufFile.Read(stream);
        var config = ModelConfig.FromGguf(gguf);

        Assert.Contains("qwen", config.Architecture, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void FromQwen_CorrectDimensions()
    {
        if (!TestConstants.ModelExists) return;

        using var stream = File.OpenRead(TestConstants.Qwen35_08B_Q8_0);
        var gguf = GgufFile.Read(stream);
        var config = ModelConfig.FromGguf(gguf);

        Assert.Equal(1024, config.HiddenDim);
        Assert.Equal(24, config.NumLayers);
        Assert.True(config.NumHeads > 0);
        Assert.True(config.NumKvHeads > 0);
        Assert.True(config.NumHeads >= config.NumKvHeads);
        Assert.True(config.IntermediateDim > config.HiddenDim);
        Assert.True(config.VocabSize > 100_000);
        Assert.True(config.RopeTheta > 0);
        Assert.True(config.NormEps > 0);
    }

    [Fact]
    public void FromQwen_HybridLayerSchedule()
    {
        if (!TestConstants.ModelExists) return;

        using var stream = File.OpenRead(TestConstants.Qwen35_08B_Q8_0);
        var gguf = GgufFile.Read(stream);
        var config = ModelConfig.FromGguf(gguf);

        Assert.Equal(4, config.FullAttentionInterval);
        Assert.True(config.SsmGroupCount > 0);
        Assert.True(config.SsmInnerSize > 0);

        // Standard attention at layers 3, 7, 11, 15, 19, 23
        Assert.True(config.IsStandardAttention(3));
        Assert.True(config.IsStandardAttention(7));
        Assert.False(config.IsStandardAttention(0));
        Assert.False(config.IsStandardAttention(1));
    }
}

public class ModelLoaderTests
{
    [Fact]
    public void Load_AllTensorsPresent()
    {
        if (!TestConstants.ModelExists) return;

        using var stream = File.OpenRead(TestConstants.Qwen35_08B_Q8_0);
        var gguf = GgufFile.Read(stream);
        var config = ModelConfig.FromGguf(gguf);

        using var backend = new CpuBackend();
        using var weights = ModelLoader.Load(gguf, stream, backend, config);

        Assert.NotNull(weights.TokenEmbedding);
        Assert.NotNull(weights.OutputNorm);
        Assert.Equal(config.NumLayers, weights.Layers.Length);

        for (int i = 0; i < config.NumLayers; i++)
        {
            var layer = weights.Layers[i];
            Assert.NotNull(layer.AttnNorm);
            Assert.NotNull(layer.PostAttnNorm);
            Assert.NotNull(layer.FfnGate);
            Assert.NotNull(layer.FfnUp);
            Assert.NotNull(layer.FfnDown);

            if (config.IsStandardAttention(i))
                Assert.IsType<StandardAttentionWeights>(layer);
            else
                Assert.IsType<DeltaNetWeights>(layer);
        }
    }

    [Fact]
    public void Load_TensorShapesCorrect()
    {
        if (!TestConstants.ModelExists) return;

        using var stream = File.OpenRead(TestConstants.Qwen35_08B_Q8_0);
        var gguf = GgufFile.Read(stream);
        var config = ModelConfig.FromGguf(gguf);

        using var backend = new CpuBackend();
        using var weights = ModelLoader.Load(gguf, stream, backend, config);

        // Token embedding: [hiddenDim, vocabSize]
        Assert.Equal(config.HiddenDim, (int)weights.TokenEmbedding.Dimensions[0]);
        Assert.Equal(config.VocabSize, (int)weights.TokenEmbedding.Dimensions[1]);

        // Standard attention layer shapes
        var attnLayer = (StandardAttentionWeights)weights.Layers[3]; // first std attn layer
        Assert.Equal(config.HiddenDim, (int)attnLayer.AttnQ.Dimensions[0]);

        // DeltaNet layer shapes
        var dnLayer = (DeltaNetWeights)weights.Layers[0]; // first DeltaNet layer
        Assert.Equal(config.HiddenDim, (int)dnLayer.AttnQkv.Dimensions[0]);
        Assert.Equal(config.SsmInnerSize * 3, (int)dnLayer.AttnQkv.Dimensions[1]);
    }
}

public class MmapModelLoaderTests
{
    [Fact]
    public void MmapLoad_AllTensorsPresent()
    {
        if (!TestConstants.ModelExists) return;

        using var stream = File.OpenRead(TestConstants.Qwen35_08B_Q8_0);
        var gguf = GgufFile.Read(stream);
        var config = ModelConfig.FromGguf(gguf);

        using var backend = new CpuBackend();
        using var weights = MmapModelLoader.Load(gguf, TestConstants.Qwen35_08B_Q8_0, backend, config);

        Assert.NotNull(weights.TokenEmbedding);
        Assert.NotNull(weights.OutputNorm);
        Assert.Equal(config.NumLayers, weights.Layers.Length);
    }

    [Fact]
    public void MmapLoad_MatchesStreamLoad()
    {
        if (!TestConstants.ModelExists) return;

        using var stream = File.OpenRead(TestConstants.Qwen35_08B_Q8_0);
        var gguf = GgufFile.Read(stream);
        var config = ModelConfig.FromGguf(gguf);

        using var backend1 = new CpuBackend();
        using var streamWeights = ModelLoader.Load(gguf, stream, backend1, config);

        using var backend2 = new CpuBackend();
        using var mmapWeights = MmapModelLoader.Load(gguf, TestConstants.Qwen35_08B_Q8_0, backend2, config);

        // Compare token embedding values
        var streamEmb = new float[streamWeights.TokenEmbedding.ElementCount];
        var mmapEmb = new float[mmapWeights.TokenEmbedding.ElementCount];
        streamWeights.TokenEmbedding.DequantizeTo(streamEmb);
        mmapWeights.TokenEmbedding.DequantizeTo(mmapEmb);

        Assert.Equal(streamEmb.Length, mmapEmb.Length);
        for (int i = 0; i < Math.Min(1000, streamEmb.Length); i++)
            Assert.Equal(streamEmb[i], mmapEmb[i]);
    }

    [Fact]
    public void MmapLoad_GeneratesCoherentOutput()
    {
        if (!TestConstants.ModelExists) return;

        using var stream = File.OpenRead(TestConstants.Qwen35_08B_Q8_0);
        var gguf = GgufFile.Read(stream);
        var config = ModelConfig.FromGguf(gguf);

        using var backend = new CpuBackend();
        using var weights = MmapModelLoader.Load(gguf, TestConstants.Qwen35_08B_Q8_0, backend, config);
        using var kvCache = new KvCache(backend, config, maxSeqLen: 256);
        using var deltaState = new DeltaNetState(backend, config);
        using var forward = new ForwardPass(backend, config, weights, kvCache, deltaState);

        // Forward pass should produce valid logits
        var logits = forward.Forward(1, 0); // token 1 at position 0
        Assert.Equal(config.VocabSize, logits.Length);
        Assert.True(float.IsFinite(logits[0]));
    }
}
