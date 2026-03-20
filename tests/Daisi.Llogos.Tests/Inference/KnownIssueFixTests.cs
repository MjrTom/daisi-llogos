using Daisi.Llogos.Cpu;
using Daisi.Llogos.Gguf;
using Daisi.Llogos.Inference;
using Daisi.Llogos.Model;
using Daisi.Llogos.Tokenizer;

namespace Daisi.Llogos.Tests.Inference;

/// <summary>
/// Validation tests for known issue fixes.
/// These tests load actual models and verify that output is coherent (not garbage).
/// </summary>
public class KnownIssueFixTests
{
    // ── Issue 1: K-Quant Models ─────────────────────────────────────────────

    private static readonly string Qwen3_8B_Q4_K_M = @"C:\GGUFS\Qwen3-8B-Q4_K_M.gguf";
    private static readonly string Qwen3_8B_Q8_0 = @"C:\GGUFS\Qwen3-8B-Q8_0.gguf";
    private static readonly string Qwen35_9B_Q4_K_M = @"C:\GGUFS\Qwen3.5-9B-Q4_K_M.gguf";
    private static readonly string Qwen35_9B_Q8_0 = @"C:\GGUFS\Qwen3.5-9B-Q8_0.gguf";

    [Fact]
    public void Qwen3_Q4_K_M_ProducesCoherentOutput()
    {
        if (!File.Exists(Qwen3_8B_Q4_K_M)) return;

        var (text, _) = GenerateText(Qwen3_8B_Q4_K_M, "The capital of France is", maxTokens: 20);

        Assert.True(text.Length > 0, "Generated text is empty");
        // Q4_K_M should produce coherent English text about Paris, not garbage
        Assert.DoesNotContain("鳁", text); // no CJK garbage
        Assert.True(IsCoherentEnglish(text),
            $"Q4_K_M output appears to be garbage: '{text}'");
    }

    [Fact]
    public void Qwen3_Q4_K_M_MatchesQ8_0_TopToken()
    {
        if (!File.Exists(Qwen3_8B_Q4_K_M) || !File.Exists(Qwen3_8B_Q8_0)) return;

        var (_, tokensQ4K) = GenerateText(Qwen3_8B_Q4_K_M, "The capital of France is", maxTokens: 5);
        var (_, tokensQ8) = GenerateText(Qwen3_8B_Q8_0, "The capital of France is", maxTokens: 5);

        // First token should match — both should predict "Paris" or similar
        Assert.Equal(tokensQ8[0], tokensQ4K[0]);
    }

    [Fact]
    public void Qwen3_Q8_0_StillWorks()
    {
        if (!File.Exists(Qwen3_8B_Q8_0)) return;

        var (text, _) = GenerateText(Qwen3_8B_Q8_0, "The capital of France is", maxTokens: 20);

        Assert.True(text.Length > 0, "Generated text is empty");
        Assert.True(IsCoherentEnglish(text),
            $"Q8_0 output appears to be garbage: '{text}'");
    }

    // ── Issue 2: Qwen3.5-9B DeltaNet ────────────────────────────────────────

    [Fact]
    public void Qwen35_9B_Q8_0_ProducesCoherentOutput()
    {
        if (!File.Exists(Qwen35_9B_Q8_0)) return;

        var (text, _) = GenerateText(Qwen35_9B_Q8_0, "The capital of France is", maxTokens: 20);

        Assert.True(text.Length > 0, "Generated text is empty");
        Assert.True(IsCoherentEnglish(text),
            $"9B Q8_0 output appears to be garbage: '{text}'");
    }

    [Fact]
    public void Qwen35_9B_Q4_K_M_ProducesCoherentOutput()
    {
        if (!File.Exists(Qwen35_9B_Q4_K_M)) return;

        var (text, _) = GenerateText(Qwen35_9B_Q4_K_M, "The capital of France is", maxTokens: 20);

        Assert.True(text.Length > 0, "Generated text is empty");
        Assert.True(IsCoherentEnglish(text),
            $"9B Q4_K_M output appears to be garbage: '{text}'");
    }

    [Fact]
    public void Qwen35_9B_DeltaNet_StateBufferSizing()
    {
        // Verify that DeltaNetState allocates correct buffer sizes for the 9B model
        if (!File.Exists(Qwen35_9B_Q8_0)) return;

        using var stream = File.OpenRead(Qwen35_9B_Q8_0);
        var gguf = GgufFile.Read(stream);
        var config = ModelConfig.FromGguf(gguf);
        using var backend = new CpuBackend();
        var weights = ModelLoader.Load(gguf, stream, backend, config);

        // With weights, DeltaNetState should derive correct QKV dim
        using var deltaState = new DeltaNetState(backend, config, weights);

        // Find actual QKV output dim from weights
        DeltaNetWeights? deltaLayer = null;
        for (int i = 0; i < config.NumLayers; i++)
            if (!config.IsStandardAttention(i) && weights.Layers[i] is DeltaNetWeights dw)
                { deltaLayer = dw; break; }

        Assert.NotNull(deltaLayer);
        int actualQkvDim = (int)deltaLayer.AttnQkv.Dimensions[1];
        int numVHeads = (int)deltaLayer.SsmAlpha.Dimensions[1];

        // For 9B: QKV=8192 (not 3*2048=6144), numVHeads=32 (not SsmGroupCount=16)
        Assert.True(actualQkvDim > config.SsmInnerSize,
            $"9B model should have QKV dim ({actualQkvDim}) > SsmInnerSize ({config.SsmInnerSize})");
        Assert.True(numVHeads > config.SsmGroupCount,
            $"9B model should have numVHeads ({numVHeads}) > SsmGroupCount ({config.SsmGroupCount})");

        // Verify conv buffer is accessible without overflow
        // (if sizing was wrong, this would crash or corrupt memory)
        for (int i = 0; i < config.NumLayers; i++)
        {
            if (!config.IsStandardAttention(i))
            {
                var convBuf = deltaState.GetConvBufferTensor(i);
                Assert.True(convBuf.ElementCount >= (config.SsmConvKernel - 1) * actualQkvDim,
                    $"Conv buffer too small: {convBuf.ElementCount} < {(config.SsmConvKernel - 1) * actualQkvDim}");
                break;
            }
        }

        weights.Dispose();
    }

    // ── Dequantization Unit Tests ────────────────────────────────────────────

    [Fact]
    public void Q6_K_ScaleSubIndexing_Correct()
    {
        // Build a Q6_K super-block with known scales to verify l/16 indexing.
        // Q6_K layout: ql[128] + qh[64] + sc[16] + d[2] = 210 bytes
        var block = new byte[210];

        // Set d = 1.0 at offset 208
        BitConverter.TryWriteBytes(block.AsSpan(208), (Half)1.0f);

        // Set scales: sc[0] = 1, sc[1] = 2, sc[2] = 3, ..., sc[7] = 8
        // (signed bytes, used as-is in the dequant)
        for (int i = 0; i < 16; i++)
            block[192 + i] = (byte)(sbyte)(i + 1);

        // Set all ql to encode q=1 (low nibble = 1, high nibble = 0)
        // and all qh to 0 (no high bits), so q_value = 1 - 32 = -31
        for (int i = 0; i < 128; i++)
            block[i] = 0x01; // low nibble = 1, high nibble = 0

        var output = new float[256];
        Dequantize.DequantizeQ6_K(block, output);

        // First half (128 elements), first 32-element group:
        // For l=0..15: isc = 0, scale = sc[0+0] = 1, q = (1 | 0) - 32 = -31
        //   output = 1.0 * 1 * (-31) = -31
        // For l=16..31: isc = 1, scale = sc[0+1] = 2, q = (1 | 0) - 32 = -31
        //   output = 1.0 * 2 * (-31) = -62
        for (int l = 0; l < 16; l++)
            Assert.Equal(-31.0f, output[l], 0.01f);
        for (int l = 16; l < 32; l++)
            Assert.Equal(-62.0f, output[l], 0.01f);
    }

    [Fact]
    public void Q5_K_ChunkedLayout_Correct()
    {
        // Build a Q5_K super-block with known values to verify chunked nibble layout.
        // Q5_K layout: d[2] + dmin[2] + scales[12] + qh[32] + qs[128] = 176 bytes
        var block = new byte[176];

        // Set d = 1.0, dmin = 0.0
        BitConverter.TryWriteBytes(block.AsSpan(0), (Half)1.0f);
        BitConverter.TryWriteBytes(block.AsSpan(2), (Half)0.0f);

        // Set all scales to 1, all mins to 0
        // Unpack6BitScalesMins: scales[0..3] = packed[0..3] & 63
        for (int i = 0; i < 4; i++)
            block[4 + i] = 1; // scales[0..3] = 1

        // Set qs: first 32 bytes encode chunk 0 (elements 0..63)
        // Low nibble = 3, high nibble = 7 → byte = 0x73
        int qsOff = 48;
        for (int i = 0; i < 32; i++)
            block[qsOff + i] = 0x73; // lo=3, hi=7

        // No high bits set (qh all zero)

        var output = new float[256];
        Dequantize.DequantizeQ5_K(block, output);

        // Chunk 0: elements 0..31 use low nibble (3), elements 32..63 use high nibble (7)
        // With scale=1 and min=0: output = 1.0 * 1 * value - 0 = value
        for (int l = 0; l < 32; l++)
            Assert.Equal(3.0f, output[l], 0.01f);
        for (int l = 32; l < 64; l++)
            Assert.Equal(7.0f, output[l], 0.01f);
    }

    // ── Helpers ──────────────────────────────────────────────────────────────

    private static (string text, int[] tokenIds) GenerateText(string modelPath, string prompt, int maxTokens)
    {
        using var stream = File.OpenRead(modelPath);
        var gguf = GgufFile.Read(stream);
        var config = ModelConfig.FromGguf(gguf);
        using var backend = new CpuBackend();
        var weights = ModelLoader.Load(gguf, stream, backend, config);
        using var kvCache = new KvCache(backend, config, maxSeqLen: 256);
        using var deltaState = new DeltaNetState(backend, config, weights);
        using var forward = new ForwardPass(backend, config, weights, kvCache, deltaState);
        var tokenizer = TokenizerFactory.FromGguf(gguf);
        var generator = new TextGenerator(forward, tokenizer, seed: 42);

        var p = new GenerationParams { MaxTokens = maxTokens, Temperature = 0 };
        var results = generator.Generate(prompt, p).Where(t => !t.IsDone).ToList();

        var text = string.Concat(results.Select(t => t.Text));
        var tokenIds = results.Select(t => t.TokenId).ToArray();

        weights.Dispose();
        return (text, tokenIds);
    }

    /// <summary>
    /// Simple heuristic to check if text is coherent English vs garbage.
    /// Garbage output typically has: CJK chars, random symbols, very short words,
    /// or an extremely high ratio of non-ASCII characters.
    /// </summary>
    private static bool IsCoherentEnglish(string text)
    {
        if (string.IsNullOrWhiteSpace(text)) return false;

        int asciiCount = text.Count(c => c >= 32 && c < 127);
        int totalCount = text.Length;

        // At least 70% should be printable ASCII for English text
        return (double)asciiCount / totalCount >= 0.7;
    }
}
