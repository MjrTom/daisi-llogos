using Daisi.Llama.Cpu;
using Daisi.Llama.Gguf;
using Daisi.Llama.Inference;
using Daisi.Llama.Model;
using Daisi.Llama.Tokenizer;

namespace Daisi.Llama.Tests.BitNet;

public class BitNetTests
{
    private const string BitNetPath = @"C:\GGUFS\ggml-model-i2_s.gguf";

    [Fact]
    public void I2S_ByteSize_Correct()
    {
        // 2560 elements → 2560/4 + 32 = 672 bytes
        Assert.Equal(672UL, GgmlTypeInfo.ByteSize(GgmlType.I2_S, 2560));

        // 2560*2560 = 6553600 elements → 6553600/4 + 32 = 1638432 bytes
        Assert.Equal(1638432UL, GgmlTypeInfo.ByteSize(GgmlType.I2_S, 6553600));
    }

    [Fact]
    public void I2S_Dequantize_KnownPattern()
    {
        // Test a simple 128-element block
        // Pack known ternary values: -1=0b00, 0=0b01, +1=0b10
        int elements = 128;
        long packedBytes = elements / 4; // 32 bytes
        var source = new byte[packedBytes + 32]; // 32 packed + 32 trailer

        // Set scale = 1.0f in the trailer
        BitConverter.TryWriteBytes(source.AsSpan((int)packedBytes), 1.0f);

        // Byte 0: groups [7:6]=0b10(+1), [5:4]=0b00(-1), [3:2]=0b01(0), [1:0]=0b10(+1)
        source[0] = 0b10_00_01_10;

        var dest = new float[elements];
        I2SDequant.Dequantize(source, dest, elements);

        // Group 0 (offset 0): elem[0] = +1 (bits 7:6 = 0b10)
        Assert.Equal(1.0f, dest[0]);
        // Group 1 (offset 32): elem[32] = -1 (bits 5:4 = 0b00)
        Assert.Equal(-1.0f, dest[32]);
        // Group 2 (offset 64): elem[64] = 0 (bits 3:2 = 0b01)
        Assert.Equal(0.0f, dest[64]);
        // Group 3 (offset 96): elem[96] = +1 (bits 1:0 = 0b10)
        Assert.Equal(1.0f, dest[96]);
    }

    [Fact]
    public void I2S_Dequantize_ScaleApplied()
    {
        int elements = 128;
        long packedBytes = elements / 4;
        var source = new byte[packedBytes + 32];

        // Set scale = 0.5f
        BitConverter.TryWriteBytes(source.AsSpan((int)packedBytes), 0.5f);

        // Byte 0: all +1 (0b10) in group 0
        source[0] = 0b10_01_01_01; // group0=+1, rest=0

        var dest = new float[elements];
        I2SDequant.Dequantize(source, dest, elements);

        Assert.Equal(0.5f, dest[0]); // +1 * 0.5
    }

    [Fact]
    public void I2S_MatMul_MatchesDequantized()
    {
        // Create a small I2_S weight matrix and verify matmul matches dequant+FP32 matmul
        int K = 128;
        int N = 4;

        long packedBytesPerRow = K / 4; // 32
        long totalPacked = (long)K * N / 4; // 128
        var bPacked = new byte[totalPacked + 32]; // +32 trailer

        // Set scale = 1.0
        BitConverter.TryWriteBytes(bPacked.AsSpan((int)totalPacked), 1.0f);

        // Fill with alternating +1/-1 pattern
        for (int row = 0; row < N; row++)
        {
            for (int gp = 0; gp < 32; gp++)
            {
                int idx = (int)(row * packedBytesPerRow + gp);
                // Group 0: +1 (0b10), Group 1: -1 (0b00), Group 2: +1 (0b10), Group 3: -1 (0b00)
                bPacked[idx] = 0b10_00_10_00;
            }
        }

        // Input vector: all 1.0
        var a = new float[K];
        Array.Fill(a, 1.0f);

        // Run matmul
        var output = new float[N];
        I2SDequant.Multiply(output, a, bPacked, 1, K, N);

        // Expected: 32*(+1) + 32*(-1) + 32*(+1) + 32*(-1) = 0
        for (int j = 0; j < N; j++)
            Assert.Equal(0.0f, output[j], 1e-5);
    }

    [Fact]
    public void ParseBitNetConfig()
    {
        if (!File.Exists(BitNetPath)) return;

        using var stream = File.OpenRead(BitNetPath);
        var gguf = GgufFile.Read(stream);
        var config = ModelConfig.FromGguf(gguf);

        Assert.Equal("bitnet-b1.58", config.Architecture);
        Assert.Equal(30, config.NumLayers);
        Assert.Equal(2560, config.HiddenDim);
        Assert.Equal(6912, config.IntermediateDim);
        Assert.Equal(20, config.NumHeads);
        Assert.Equal(5, config.NumKvHeads);
        Assert.Equal(128, config.KeyLength);  // 2560/20 derived
        Assert.Equal(128, config.ValueLength);
        Assert.Equal(4096, config.MaxContext);
        Assert.Equal(500000f, config.RopeTheta);
        Assert.Equal(128, config.RopeDimCount);
    }

    [Fact]
    public void LoadBitNetTensors()
    {
        if (!File.Exists(BitNetPath)) return;

        using var stream = File.OpenRead(BitNetPath);
        var gguf = GgufFile.Read(stream);
        var config = ModelConfig.FromGguf(gguf);

        using var backend = new CpuBackend();
        using var weights = BitNetModelLoader.Load(gguf, stream, backend, config);

        Assert.NotNull(weights.TokenEmbedding);
        Assert.Equal(GgmlType.F16, weights.TokenEmbedding.Type);
        Assert.Null(weights.Output); // Tied embeddings
        Assert.Equal(config.NumLayers, weights.Layers.Length);

        var layer0 = (BitNetLayerWeights)weights.Layers[0];
        Assert.Equal(GgmlType.I2_S, layer0.AttnQ.Type);
        Assert.Equal(GgmlType.I2_S, layer0.AttnK.Type);
        Assert.Equal(GgmlType.I2_S, layer0.AttnV.Type);
        Assert.Equal(GgmlType.I2_S, layer0.AttnO.Type);
        Assert.Equal(GgmlType.F32, layer0.AttnSubNorm.Type);
        Assert.Equal(GgmlType.F32, layer0.FfnSubNorm.Type);
        Assert.Equal(GgmlType.I2_S, layer0.FfnGate.Type);
        Assert.Equal(GgmlType.I2_S, layer0.FfnUp.Type);
        Assert.Equal(GgmlType.I2_S, layer0.FfnDown.Type);
    }

    [Fact]
    public void BitNetForwardPass_FirstToken()
    {
        if (!File.Exists(BitNetPath)) return;

        using var stream = File.OpenRead(BitNetPath);
        var gguf = GgufFile.Read(stream);
        var config = ModelConfig.FromGguf(gguf);

        using var backend = new CpuBackend();
        using var weights = BitNetModelLoader.Load(gguf, stream, backend, config);
        using var kvCache = new BitNetKvCache(backend, config, maxSeqLen: 64);
        using var forward = new BitNetForwardPass(backend, config, weights, kvCache);

        // Run a single forward pass with token 128000 (BOS)
        var logits = forward.Forward(128000, 0);

        Assert.Equal(config.VocabSize, logits.Length);

        // Logits should not be all zeros
        float sum = 0;
        for (int i = 0; i < logits.Length; i++)
            sum += MathF.Abs(logits[i]);
        Assert.True(sum > 0, "Logits are all zeros");

        // Argmax should be a reasonable token
        int argmax = 0;
        float maxVal = float.NegativeInfinity;
        for (int i = 0; i < logits.Length; i++)
        {
            if (logits[i] > maxVal)
            {
                maxVal = logits[i];
                argmax = i;
            }
        }
        Assert.True(argmax >= 0 && argmax < config.VocabSize);
    }

    [Fact]
    public void BitNetGenerate_ProducesText()
    {
        if (!File.Exists(BitNetPath)) return;

        using var stream = File.OpenRead(BitNetPath);
        var gguf = GgufFile.Read(stream);
        var config = ModelConfig.FromGguf(gguf);

        using var backend = new CpuBackend();
        using var weights = BitNetModelLoader.Load(gguf, stream, backend, config);
        using var kvCache = new BitNetKvCache(backend, config, maxSeqLen: 128);
        using var forward = new BitNetForwardPass(backend, config, weights, kvCache);
        var tokenizer = TokenizerFactory.FromGguf(gguf);
        var gen = new BitNetTextGenerator(forward, tokenizer, seed: 42);

        var parameters = new GenerationParams { MaxTokens = 16, Temperature = 0.7f, TopK = 40, TopP = 0.9f };
        var tokens = gen.Generate("Hello", parameters).ToList();

        // Should produce at least some tokens before the done signal
        Assert.True(tokens.Count >= 2, $"Only got {tokens.Count} tokens");
        Assert.True(tokens.Last().IsDone);
    }
}
