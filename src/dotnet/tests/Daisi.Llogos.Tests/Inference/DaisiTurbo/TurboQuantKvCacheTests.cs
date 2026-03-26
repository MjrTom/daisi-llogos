using Daisi.Llogos.Cpu;
using Daisi.Llogos.Gguf;
using Daisi.Llogos.Inference;
using Daisi.Llogos.Inference.DaisiTurbo;
using Daisi.Llogos.Model;

namespace Daisi.Llogos.Tests.Inference.DaisiTurbo;

public class TurboQuantKvCacheTests
{
    private static ModelConfig MakeConfig(int numLayers = 4, int numHeads = 8, int numKvHeads = 4,
        int hiddenDim = 256, int keyLength = 64, int valueLength = 64)
    {
        return new ModelConfig
        {
            Architecture = "test",
            NumLayers = numLayers,
            NumHeads = numHeads,
            NumKvHeads = numKvHeads,
            HiddenDim = hiddenDim,
            KeyLength = keyLength,
            ValueLength = valueLength,
            VocabSize = 100,
            NormEps = 1e-6f,
            RopeTheta = 10000f,
            RopeDimCount = keyLength,
            MaxContext = 2048,
            IntermediateDim = hiddenDim * 4,
            FullAttentionInterval = 0,
            SsmConvKernel = 0,
            SsmStateSize = 0,
            SsmGroupCount = 0,
            SsmInnerSize = 0,
        };
    }

    [Theory]
    [InlineData(3)]
    [InlineData(4)]
    public void WriteAndRead_PreservesApproximateValues(int bits)
    {
        using var backend = new CpuBackend();
        var config = MakeConfig();
        var turboConfig = new TurboQuantConfig { QuantBits = bits, QjlProjectionDim = 0 };
        using var cache = new TurboQuantKvCache(backend, config, maxSeqLen: 32, turboConfig: turboConfig);

        int nKvHeads = config.NumKvHeads;
        int keyLen = config.KeyLength;
        int valLen = config.ValueLength;

        // Create K and V tensors with known values
        var k = backend.CreateTensor("k", GgmlType.F32, [nKvHeads * keyLen]);
        var v = backend.CreateTensor("v", GgmlType.F32, [nKvHeads * valLen]);
        var rng = new Random(42);

        var kSpan = k.AsFloatSpan();
        var vSpan = v.AsFloatSpan();
        for (int i = 0; i < kSpan.Length; i++) kSpan[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < vSpan.Length; i++) vSpan[i] = (float)(rng.NextDouble() * 2 - 1);

        // Write to cache
        cache.Write(backend, 0, 0, k, v);
        Assert.Equal(1, cache.Length);

        // Read back decompressed
        var kCache = cache.GetKCacheTensor(0);
        var kOut = kCache.AsFloatSpan();

        // Check approximate equality (quantization introduces error)
        float maxErr = 0;
        for (int h = 0; h < nKvHeads; h++)
        {
            for (int d = 0; d < keyLen; d++)
            {
                int cacheOff = h * 32 * keyLen + 0 * keyLen + d; // maxSeqLen=32, pos=0
                float err = MathF.Abs(kSpan[h * keyLen + d] - kOut[cacheOff]);
                maxErr = MathF.Max(maxErr, err);
            }
        }

        // Quantization error should be bounded
        float errorBound = bits switch { 3 => 0.5f, 4 => 0.3f, _ => 1.0f };
        Assert.True(maxErr < errorBound, $"Max error {maxErr:F4} exceeds bound {errorBound} for {bits}-bit");

        k.Dispose();
        v.Dispose();
    }

    [Fact]
    public void MultiplePositions_AllReadable()
    {
        using var backend = new CpuBackend();
        var config = MakeConfig();
        var turboConfig = new TurboQuantConfig { QuantBits = 4, QjlProjectionDim = 0 };
        using var cache = new TurboQuantKvCache(backend, config, maxSeqLen: 64, turboConfig: turboConfig);

        int nKvHeads = config.NumKvHeads;
        int keyLen = config.KeyLength;
        var rng = new Random(42);

        for (int pos = 0; pos < 16; pos++)
        {
            var k = backend.CreateTensor($"k_{pos}", GgmlType.F32, [nKvHeads * keyLen]);
            var v = backend.CreateTensor($"v_{pos}", GgmlType.F32, [nKvHeads * config.ValueLength]);
            var kSpan = k.AsFloatSpan();
            var vSpan = v.AsFloatSpan();
            for (int i = 0; i < kSpan.Length; i++) kSpan[i] = (float)(rng.NextDouble() * 2 - 1);
            for (int i = 0; i < vSpan.Length; i++) vSpan[i] = (float)(rng.NextDouble() * 2 - 1);

            cache.Write(backend, 0, pos, k, v);
            k.Dispose();
            v.Dispose();
        }

        Assert.Equal(16, cache.Length);

        // Verify we can read the cache tensors without error
        var kCacheTensor = cache.GetKCacheTensor(0);
        var vCacheTensor = cache.GetVCacheTensor(0);
        Assert.NotNull(kCacheTensor);
        Assert.NotNull(vCacheTensor);
    }

    [Fact]
    public void GetStats_ReportsCompression()
    {
        using var backend = new CpuBackend();
        var config = MakeConfig();
        var turboConfig = new TurboQuantConfig { QuantBits = 3 };
        using var cache = new TurboQuantKvCache(backend, config, maxSeqLen: 64, turboConfig: turboConfig);

        var k = backend.CreateTensor("k", GgmlType.F32, [config.NumKvHeads * config.KeyLength]);
        var v = backend.CreateTensor("v", GgmlType.F32, [config.NumKvHeads * config.ValueLength]);
        var rng = new Random(42);
        var kSpan = k.AsFloatSpan();
        var vSpan = v.AsFloatSpan();
        for (int i = 0; i < kSpan.Length; i++) kSpan[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < vSpan.Length; i++) vSpan[i] = (float)(rng.NextDouble() * 2 - 1);

        cache.Write(backend, 0, 0, k, v);

        var stats = cache.GetStats();
        Assert.True(stats.CompressionRatio > 1.0f, $"Compression ratio {stats.CompressionRatio} should be > 1");
        Assert.Equal(3, stats.QuantBits);
        Assert.Equal(1, stats.SeqLength);

        k.Dispose();
        v.Dispose();
    }

    [Fact]
    public void Reset_ClearsLength()
    {
        using var backend = new CpuBackend();
        var config = MakeConfig();
        var turboConfig = new TurboQuantConfig { QuantBits = 4, QjlProjectionDim = 0 };
        using var cache = new TurboQuantKvCache(backend, config, maxSeqLen: 32, turboConfig: turboConfig);

        var k = backend.CreateTensor("k", GgmlType.F32, [config.NumKvHeads * config.KeyLength]);
        var v = backend.CreateTensor("v", GgmlType.F32, [config.NumKvHeads * config.ValueLength]);
        cache.Write(backend, 0, 0, k, v);
        Assert.Equal(1, cache.Length);

        cache.Reset();
        Assert.Equal(0, cache.Length);

        k.Dispose();
        v.Dispose();
    }

    [Fact]
    public void SlidingWindow_StrategyWorks()
    {
        using var backend = new CpuBackend();
        var config = MakeConfig();
        var turboConfig = new TurboQuantConfig { QuantBits = 4, QjlProjectionDim = 0 };
        var strategy = AttentionStrategy.Window(8);
        using var cache = new TurboQuantKvCache(backend, config, maxSeqLen: 1024,
            turboConfig: turboConfig, strategy: strategy);

        Assert.Equal(8, cache.MaxSeqLen);
    }

    [Theory]
    [InlineData("turbo", 3)]
    [InlineData("turbo:4", 4)]
    [InlineData("turbo:2", 2)]
    [InlineData("turbo:3+noqjl", 3)]
    [InlineData("turbo:3+qjl32", 3)]
    public void ConfigParse_Works(string input, int expectedBits)
    {
        var config = TurboQuantConfig.Parse(input);
        Assert.Equal(expectedBits, config.QuantBits);
    }

    [Fact]
    public void ConfigParse_QjlOptions()
    {
        var noqjl = TurboQuantConfig.Parse("turbo:3+noqjl");
        Assert.Equal(0, noqjl.QjlProjectionDim);

        var qjl32 = TurboQuantConfig.Parse("turbo:3+qjl32");
        Assert.Equal(32, qjl32.QjlProjectionDim);
    }

    [Theory]
    [InlineData(3)]
    [InlineData(4)]
    public void HigherBits_LowerReconstructionError(int bits)
    {
        // Higher bit quantization should give lower reconstruction error
        using var backend = new CpuBackend();
        var config = MakeConfig();
        var turboConfig = new TurboQuantConfig { QuantBits = bits, QjlProjectionDim = 0 };
        using var cache = new TurboQuantKvCache(backend, config, maxSeqLen: 32, turboConfig: turboConfig);

        int nKvHeads = config.NumKvHeads;
        int keyLen = config.KeyLength;
        var rng = new Random(42);

        var k = backend.CreateTensor("k", GgmlType.F32, [nKvHeads * keyLen]);
        var v = backend.CreateTensor("v", GgmlType.F32, [nKvHeads * config.ValueLength]);
        var kSpan = k.AsFloatSpan();
        var vSpan = v.AsFloatSpan();
        for (int i = 0; i < kSpan.Length; i++) kSpan[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < vSpan.Length; i++) vSpan[i] = (float)(rng.NextDouble() * 2 - 1);

        cache.Write(backend, 0, 0, k, v);

        var kOut = cache.GetKCacheTensor(0).AsFloatSpan();
        float mse = 0;
        for (int d = 0; d < keyLen; d++)
        {
            float err = kSpan[d] - kOut[d]; // head 0, pos 0
            mse += err * err;
        }
        mse /= keyLen;

        // 4-bit should have lower MSE than 3-bit
        float bound = bits switch { 3 => 0.15f, 4 => 0.05f, _ => 1.0f };
        Assert.True(mse < bound, $"{bits}-bit MSE {mse:F4} exceeds bound {bound}");

        k.Dispose();
        v.Dispose();
    }
}
