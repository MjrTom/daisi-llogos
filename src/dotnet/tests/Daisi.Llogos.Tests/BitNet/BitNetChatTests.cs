using Daisi.Llogos.Chat;
using Daisi.Llogos.Cpu;
using Daisi.Llogos.Gguf;
using Daisi.Llogos.Inference;
using Daisi.Llogos.Model;
using Daisi.Llogos.Tokenizer;

namespace Daisi.Llogos.Tests.BitNet;

/// <summary>
/// Tests that BitNet models work correctly through the model loading and chat paths.
/// Verifies architecture detection, correct loader selection, and correct forward pass.
/// </summary>
public class BitNetChatTests
{
    private const string BitNetPath = @"C:\GGUFS\ggml-model-i2_s.gguf";

    [Fact]
    public void ModelConfig_IsBitNet_True()
    {
        if (!File.Exists(BitNetPath)) return;

        using var stream = File.OpenRead(BitNetPath);
        var gguf = GgufFile.Read(stream);
        var config = ModelConfig.FromGguf(gguf);

        Assert.True(config.IsBitNet);
    }

    [Fact]
    public void ModelConfig_IsBitNet_False_ForQwen()
    {
        if (!TestConstants.ModelExists) return;

        using var stream = File.OpenRead(TestConstants.Qwen35_08B_Q8_0);
        var gguf = GgufFile.Read(stream);
        var config = ModelConfig.FromGguf(gguf);

        Assert.False(config.IsBitNet);
    }

    [Fact]
    public void MmapModelLoader_BitNet_LoadsBitNetLayerWeights()
    {
        if (!File.Exists(BitNetPath)) return;

        using var stream = File.OpenRead(BitNetPath);
        var gguf = GgufFile.Read(stream);
        var config = ModelConfig.FromGguf(gguf);

        using var backend = new CpuBackend();
        using var weights = MmapModelLoader.Load(gguf, BitNetPath, backend, config);

        // Layers must be BitNetLayerWeights, not StandardAttentionWeights
        Assert.IsType<BitNetLayerWeights>(weights.Layers[0]);

        var layer0 = (BitNetLayerWeights)weights.Layers[0];
        Assert.Equal(GgmlType.I2_S, layer0.AttnQ.Type);
        Assert.NotNull(layer0.AttnSubNorm);
        Assert.NotNull(layer0.FfnSubNorm);
    }

    [Fact]
    public void MmapModelLoader_BitNet_OutputMatchesBitNetLoader()
    {
        if (!File.Exists(BitNetPath)) return;

        using var stream = File.OpenRead(BitNetPath);
        var gguf = GgufFile.Read(stream);
        var config = ModelConfig.FromGguf(gguf);

        using var backend = new CpuBackend();

        // Load with BitNetModelLoader (known good)
        using var refWeights = BitNetModelLoader.Load(gguf, stream, backend, config);
        using var refKv = new BitNetKvCache(backend, config, maxSeqLen: 64);
        using var refForward = new BitNetForwardPass(backend, config, refWeights, refKv);
        var refLogits = refForward.Forward(128000, 0).ToArray();

        // Load with MmapModelLoader (should now support BitNet)
        using var mmapWeights = MmapModelLoader.Load(gguf, BitNetPath, backend, config);
        using var mmapKv = new BitNetKvCache(backend, config, maxSeqLen: 64);
        using var mmapForward = new BitNetForwardPass(backend, config, mmapWeights, mmapKv);
        var mmapLogits = mmapForward.Forward(128000, 0).ToArray();

        // Output must match
        Assert.Equal(refLogits.Length, mmapLogits.Length);
        int refArgmax = ArgMax(refLogits);
        int mmapArgmax = ArgMax(mmapLogits);
        Assert.Equal(refArgmax, mmapArgmax);
    }

    [Fact]
    public async Task TextBackend_BitNet_LoadsAndChats()
    {
        if (!File.Exists(BitNetPath)) return;

        await using var textBackend = new DaisiLlogosTextBackend();
        await textBackend.ConfigureAsync(new Daisi.Inference.Models.BackendConfiguration { Runtime = "cpu" });

        var handle = await textBackend.LoadModelAsync(new Daisi.Inference.Models.ModelLoadRequest
        {
            ModelId = "bitnet-test",
            FilePath = BitNetPath,
            ContextSize = 64,
        });

        Assert.True(handle.IsLoaded);

        using var session = await textBackend.CreateChatSessionAsync(handle);
        var msg = new Daisi.Inference.Models.ChatMessage(
            Daisi.Inference.Models.ChatRole.User, "Hi");
        var genParams = new Daisi.Inference.Models.TextGenerationParams
        {
            MaxTokens = 4,
            Temperature = 0,
        };

        var tokens = new List<string>();
        await foreach (var token in session.ChatAsync(msg, genParams))
            tokens.Add(token);

        // Should produce at least one token without crashing
        Assert.NotEmpty(tokens);

        textBackend.UnloadModel(handle);
    }

    private static int ArgMax(float[] values)
    {
        int best = 0;
        for (int i = 1; i < values.Length; i++)
            if (values[i] > values[best]) best = i;
        return best;
    }
}
