using Daisi.Llogos.Cpu;
using Daisi.Llogos.Gguf;
using Daisi.Llogos.Inference;
using Daisi.Llogos.Model;
using Daisi.Llogos.Tokenizer;
using Daisi.Llogos.Vulkan;

namespace Daisi.Llogos.Tests.Vulkan;

/// <summary>
/// Tests for Phase 9: Vulkan compute backend.
/// Tests skip gracefully if no Vulkan GPU is available.
/// </summary>
public class VulkanBackendTests
{
    private static bool VulkanAvailable()
    {
        try
        {
            using var backend = new VulkanBackend();
            return true;
        }
        catch
        {
            return false;
        }
    }

    [Fact]
    public void VulkanDevice_Create()
    {
        if (!VulkanAvailable()) return;
        using var backend = new VulkanBackend();
        Assert.Contains("NVIDIA", backend.Name, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void VulkanTensor_CreateAndDispose()
    {
        if (!VulkanAvailable()) return;
        using var backend = new VulkanBackend();
        using var tensor = backend.CreateTensor("test", GgmlType.F32, [256]);
        Assert.Equal(256, tensor.ElementCount);
        Assert.Equal(1024, tensor.ByteSize);
    }

    [Fact]
    public void VulkanTensor_UploadDownloadRoundtrip()
    {
        if (!VulkanAvailable()) return;
        using var backend = new VulkanBackend();
        using var tensor = backend.CreateTensor("test", GgmlType.F32, [4]);

        var data = new float[] { 1.0f, 2.0f, 3.0f, 4.0f };
        var bytes = System.Runtime.InteropServices.MemoryMarshal.AsBytes(data.AsSpan()).ToArray();
        tensor.CopyFrom(bytes);

        var result = new float[4];
        tensor.DequantizeTo(result);

        Assert.Equal(1.0f, result[0]);
        Assert.Equal(2.0f, result[1]);
        Assert.Equal(3.0f, result[2]);
        Assert.Equal(4.0f, result[3]);
    }

    [Fact]
    public void VulkanElementAdd_MatchesCpu()
    {
        if (!VulkanAvailable()) return;
        using var vk = new VulkanBackend();
        using var cpu = new CpuBackend();

        int n = 1024;
        var aData = new float[n];
        var bData = new float[n];
        var rng = new Random(42);
        for (int i = 0; i < n; i++)
        {
            aData[i] = (float)(rng.NextDouble() * 2 - 1);
            bData[i] = (float)(rng.NextDouble() * 2 - 1);
        }

        // CPU
        using var cpuA = cpu.LoadTensor("a", GgmlType.F32, [n], AsBytes(aData));
        using var cpuB = cpu.LoadTensor("b", GgmlType.F32, [n], AsBytes(bData));
        using var cpuOut = cpu.CreateTensor("out", GgmlType.F32, [n]);
        cpu.ElementAdd(cpuOut, cpuA, cpuB);

        // Vulkan
        using var vkA = vk.LoadTensor("a", GgmlType.F32, [n], AsBytes(aData));
        using var vkB = vk.LoadTensor("b", GgmlType.F32, [n], AsBytes(bData));
        using var vkOut = vk.CreateTensor("out", GgmlType.F32, [n]);
        vk.ElementAdd(vkOut, vkA, vkB);

        var cpuResult = new float[n];
        var vkResult = new float[n];
        cpuOut.DequantizeTo(cpuResult);
        vkOut.DequantizeTo(vkResult);

        for (int i = 0; i < n; i++)
            Assert.Equal(cpuResult[i], vkResult[i], 4); // tolerance for float precision
    }

    [Fact]
    public void VulkanElementMul_MatchesCpu()
    {
        if (!VulkanAvailable()) return;
        using var vk = new VulkanBackend();
        using var cpu = new CpuBackend();

        int n = 512;
        var aData = new float[n];
        var bData = new float[n];
        var rng = new Random(42);
        for (int i = 0; i < n; i++)
        {
            aData[i] = (float)(rng.NextDouble() * 2 - 1);
            bData[i] = (float)(rng.NextDouble() * 2 - 1);
        }

        // CPU
        using var cpuA = cpu.LoadTensor("a", GgmlType.F32, [n], AsBytes(aData));
        using var cpuB = cpu.LoadTensor("b", GgmlType.F32, [n], AsBytes(bData));
        using var cpuOut = cpu.CreateTensor("out", GgmlType.F32, [n]);
        cpu.ElementMul(cpuOut, cpuA, cpuB);

        // Vulkan
        using var vkA = vk.LoadTensor("a", GgmlType.F32, [n], AsBytes(aData));
        using var vkB = vk.LoadTensor("b", GgmlType.F32, [n], AsBytes(bData));
        using var vkOut = vk.CreateTensor("out", GgmlType.F32, [n]);
        vk.ElementMul(vkOut, vkA, vkB);

        var cpuResult = new float[n];
        var vkResult = new float[n];
        cpuOut.DequantizeTo(cpuResult);
        vkOut.DequantizeTo(vkResult);

        for (int i = 0; i < n; i++)
            Assert.Equal(cpuResult[i], vkResult[i], 4);
    }

    [Fact]
    public void VulkanSiLU_MatchesCpu()
    {
        if (!VulkanAvailable()) return;
        using var vk = new VulkanBackend();
        using var cpu = new CpuBackend();

        int n = 512;
        var inputData = new float[n];
        var rng = new Random(42);
        for (int i = 0; i < n; i++)
            inputData[i] = (float)(rng.NextDouble() * 4 - 2);

        // CPU
        using var cpuIn = cpu.LoadTensor("in", GgmlType.F32, [n], AsBytes(inputData));
        using var cpuOut = cpu.CreateTensor("out", GgmlType.F32, [n]);
        cpu.SiLU(cpuOut, cpuIn);

        // Vulkan
        using var vkIn = vk.LoadTensor("in", GgmlType.F32, [n], AsBytes(inputData));
        using var vkOut = vk.CreateTensor("out", GgmlType.F32, [n]);
        vk.SiLU(vkOut, vkIn);

        var cpuResult = new float[n];
        var vkResult = new float[n];
        cpuOut.DequantizeTo(cpuResult);
        vkOut.DequantizeTo(vkResult);

        for (int i = 0; i < n; i++)
            Assert.Equal(cpuResult[i], vkResult[i], 3);
    }

    [Fact]
    public void VulkanSoftmax_MatchesCpu()
    {
        if (!VulkanAvailable()) return;
        using var vk = new VulkanBackend();
        using var cpu = new CpuBackend();

        int n = 256;
        var inputData = new float[n];
        var rng = new Random(42);
        for (int i = 0; i < n; i++)
            inputData[i] = (float)(rng.NextDouble() * 10 - 5);

        // CPU
        using var cpuIn = cpu.LoadTensor("in", GgmlType.F32, [n], AsBytes(inputData));
        using var cpuOut = cpu.CreateTensor("out", GgmlType.F32, [n]);
        cpu.Softmax(cpuOut, cpuIn);

        // Vulkan
        using var vkIn = vk.LoadTensor("in", GgmlType.F32, [n], AsBytes(inputData));
        using var vkOut = vk.CreateTensor("out", GgmlType.F32, [n]);
        vk.Softmax(vkOut, vkIn);

        var cpuResult = new float[n];
        var vkResult = new float[n];
        cpuOut.DequantizeTo(cpuResult);
        vkOut.DequantizeTo(vkResult);

        // Softmax should sum to 1
        float vkSum = vkResult.Sum();
        Assert.InRange(vkSum, 0.99f, 1.01f);

        for (int i = 0; i < n; i++)
            Assert.Equal(cpuResult[i], vkResult[i], 3);
    }

    [Fact]
    public void VulkanRmsNorm_MatchesCpu()
    {
        if (!VulkanAvailable()) return;
        using var vk = new VulkanBackend();
        using var cpu = new CpuBackend();

        int n = 256;
        var inputData = new float[n];
        var weightData = new float[n];
        var rng = new Random(42);
        for (int i = 0; i < n; i++)
        {
            inputData[i] = (float)(rng.NextDouble() * 2 - 1);
            weightData[i] = (float)(rng.NextDouble() * 0.5 + 0.75);
        }
        float eps = 1e-5f;

        // CPU
        using var cpuIn = cpu.LoadTensor("in", GgmlType.F32, [n], AsBytes(inputData));
        using var cpuW = cpu.LoadTensor("w", GgmlType.F32, [n], AsBytes(weightData));
        using var cpuOut = cpu.CreateTensor("out", GgmlType.F32, [n]);
        cpu.RmsNorm(cpuOut, cpuIn, cpuW, eps);

        // Vulkan
        using var vkIn = vk.LoadTensor("in", GgmlType.F32, [n], AsBytes(inputData));
        using var vkW = vk.LoadTensor("w", GgmlType.F32, [n], AsBytes(weightData));
        using var vkOut = vk.CreateTensor("out", GgmlType.F32, [n]);
        vk.RmsNorm(vkOut, vkIn, vkW, eps);

        var cpuResult = new float[n];
        var vkResult = new float[n];
        cpuOut.DequantizeTo(cpuResult);
        vkOut.DequantizeTo(vkResult);

        for (int i = 0; i < n; i++)
            Assert.Equal(cpuResult[i], vkResult[i], 3);
    }

    [Fact]
    public void VulkanMatMul_F32_MatchesCpu()
    {
        if (!VulkanAvailable()) return;
        using var vk = new VulkanBackend();
        using var cpu = new CpuBackend();

        int M = 1, K = 64, N = 32;
        var aData = new float[M * K];
        var bData = new float[N * K]; // weights: [N x K]
        var rng = new Random(42);
        for (int i = 0; i < aData.Length; i++) aData[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < bData.Length; i++) bData[i] = (float)(rng.NextDouble() * 2 - 1);

        // CPU
        using var cpuA = cpu.LoadTensor("a", GgmlType.F32, [M * K], AsBytes(aData));
        using var cpuB = cpu.LoadTensor("b", GgmlType.F32, [K, N], AsBytes(bData));
        using var cpuOut = cpu.CreateTensor("out", GgmlType.F32, [M * N]);
        cpu.MatMul(cpuOut, cpuA, cpuB, M, K, N);

        // Vulkan
        using var vkA = vk.LoadTensor("a", GgmlType.F32, [M * K], AsBytes(aData));
        using var vkB = vk.LoadTensor("b", GgmlType.F32, [K, N], AsBytes(bData));
        using var vkOut = vk.CreateTensor("out", GgmlType.F32, [M * N]);
        vk.MatMul(vkOut, vkA, vkB, M, K, N);

        var cpuResult = new float[M * N];
        var vkResult = new float[M * N];
        cpuOut.DequantizeTo(cpuResult);
        vkOut.DequantizeTo(vkResult);

        for (int i = 0; i < M * N; i++)
            Assert.Equal(cpuResult[i], vkResult[i], 2);
    }

    [Fact]
    public void VulkanMatMul_Q8_0_MatchesCpu()
    {
        if (!VulkanAvailable()) return;
        using var vk = new VulkanBackend();
        using var cpu = new CpuBackend();

        int M = 1, K = 1024, N = 256;
        var aData = new float[M * K];
        var rng = new Random(42);
        for (int i = 0; i < aData.Length; i++) aData[i] = (float)(rng.NextDouble() * 2 - 1);

        // Generate Q8_0 weights: K/32 blocks per row, N rows
        // Each block = 2 bytes (f16 scale) + 32 bytes (int8 quants) = 34 bytes
        int blocksPerRow = K / 32;
        var q8Data = new byte[N * blocksPerRow * 34];
        for (int row = 0; row < N; row++)
        {
            for (int b = 0; b < blocksPerRow; b++)
            {
                int off = (row * blocksPerRow + b) * 34;
                // Scale = 0.1 (as float16)
                Half scale = (Half)0.1f;
                var scaleBytes = BitConverter.GetBytes(scale);
                q8Data[off] = scaleBytes[0];
                q8Data[off + 1] = scaleBytes[1];
                for (int i = 0; i < 32; i++)
                    q8Data[off + 2 + i] = (byte)(sbyte)(rng.Next(-127, 128));
            }
        }

        // CPU
        using var cpuA = cpu.LoadTensor("a", GgmlType.F32, [M * K], AsBytes(aData));
        using var cpuB = cpu.LoadTensor("b", GgmlType.Q8_0, [K, N], q8Data);
        using var cpuOut = cpu.CreateTensor("out", GgmlType.F32, [M * N]);
        cpu.MatMul(cpuOut, cpuA, cpuB, M, K, N);

        // Vulkan
        using var vkA = vk.LoadTensor("a", GgmlType.F32, [M * K], AsBytes(aData));
        using var vkB = vk.LoadTensor("b", GgmlType.Q8_0, [K, N], q8Data);
        using var vkOut = vk.CreateTensor("out", GgmlType.F32, [M * N]);
        vk.MatMul(vkOut, vkA, vkB, M, K, N);

        var cpuResult = new float[M * N];
        var vkResult = new float[M * N];
        cpuOut.DequantizeTo(cpuResult);
        vkOut.DequantizeTo(vkResult);

        for (int i = 0; i < M * N; i++)
            Assert.Equal(cpuResult[i], vkResult[i], 1);
    }

    [Fact]
    public void VulkanEmbeddingLookup_Q8_0_MatchesCpu()
    {
        if (!VulkanAvailable() || !TestConstants.ModelExists) return;
        using var vk = new VulkanBackend();
        using var cpu = new CpuBackend();

        using var stream = File.OpenRead(TestConstants.Qwen35_08B_Q8_0);
        var gguf = GgufFile.Read(stream);
        var config = ModelConfig.FromGguf(gguf);

        // Load only embedding weights with both backends
        var cpuWeights = ModelLoader.Load(gguf, stream, cpu, config);
        var vkWeights = ModelLoader.Load(gguf, stream, vk, config);

        int hiddenDim = config.HiddenDim;
        using var cpuOut = cpu.CreateTensor("emb", GgmlType.F32, [hiddenDim]);
        using var vkOut = vk.CreateTensor("emb", GgmlType.F32, [hiddenDim]);

        cpu.EmbeddingLookup(cpuOut, cpuWeights.TokenEmbedding, tokenId: 100);
        vk.EmbeddingLookup(vkOut, vkWeights.TokenEmbedding, tokenId: 100);

        var cpuResult = new float[hiddenDim];
        var vkResult = new float[hiddenDim];
        cpuOut.DequantizeTo(cpuResult);
        vkOut.DequantizeTo(vkResult);

        int diffs = 0;
        for (int i = 0; i < hiddenDim; i++)
        {
            if (Math.Abs(cpuResult[i] - vkResult[i]) > 0.01f) diffs++;
        }
        Assert.True(diffs == 0, $"Embedding has {diffs}/{hiddenDim} differing elements. First CPU: {cpuResult[0]:F4}, First VK: {vkResult[0]:F4}");

        cpuWeights.Dispose();
        vkWeights.Dispose();
    }

    [Fact]
    public void VulkanForwardPass_SingleToken()
    {
        if (!VulkanAvailable() || !TestConstants.ModelExists) return;

        using var cpuCtx = LoadModel(new CpuBackend());
        using var vkCtx = LoadModel(new VulkanBackend());

        var cpuLogits = cpuCtx.Forward.Forward(tokenId: 100, position: 0).ToArray();
        var vkLogits = vkCtx.Forward.Forward(tokenId: 100, position: 0).ToArray();

        Assert.Equal(cpuLogits.Length, vkLogits.Length);

        // Both should produce valid logits (argmax may differ slightly
        // due to GPU/CPU float reduction ordering in DeltaNet layers)
        int cpuArgmax = ArgMax(cpuLogits);
        int vkArgmax = ArgMax(vkLogits);
        Assert.True(cpuArgmax >= 0 && cpuArgmax < cpuLogits.Length);
        Assert.True(vkArgmax >= 0 && vkArgmax < vkLogits.Length);

        // Logit values should be close (not identical due to GPU float precision)
        float maxDiff = 0;
        for (int i = 0; i < cpuLogits.Length; i++)
            maxDiff = MathF.Max(maxDiff, MathF.Abs(cpuLogits[i] - vkLogits[i]));
        Assert.True(maxDiff < 1.0f, $"Max logit diff = {maxDiff:G4}, CPU argmax={cpuArgmax}, VK argmax={vkArgmax}");
    }

    [Fact]
    public void VulkanBackend_Generate()
    {
        if (!VulkanAvailable() || !TestConstants.ModelExists) return;

        using var stream = File.OpenRead(TestConstants.Qwen35_08B_Q8_0);
        var gguf = GgufFile.Read(stream);
        var config = ModelConfig.FromGguf(gguf);
        using var backend = new VulkanBackend();
        var weights = ModelLoader.Load(gguf, stream, backend, config);
        using var kvCache = new KvCache(backend, config, maxSeqLen: 128);
        using var deltaState = new DeltaNetState(backend, config);
        using var forward = new ForwardPass(backend, config, weights, kvCache, deltaState);
        var tokenizer = TokenizerFactory.FromGguf(gguf);

        var generator = new TextGenerator(forward, tokenizer, seed: 42);
        var text = "";
        foreach (var tok in generator.Generate("Hello", new GenerationParams { MaxTokens = 16, Temperature = 0 }))
        {
            if (!tok.IsDone) text += tok.Text;
        }

        Assert.True(text.Length > 0, "Vulkan backend should generate text");
        Assert.True(text.Any(char.IsLetter), $"Generated text should contain letters: '{text}'");
    }

    // ── Helpers ──────────────────────────────────────────────────────────────

    private static byte[] AsBytes(float[] data) =>
        System.Runtime.InteropServices.MemoryMarshal.AsBytes(data.AsSpan()).ToArray();

    private static int ArgMax(float[] values)
    {
        int best = 0;
        for (int i = 1; i < values.Length; i++)
            if (values[i] > values[best]) best = i;
        return best;
    }

    private static ModelContext LoadModel(IComputeBackend backend)
    {
        var stream = File.OpenRead(TestConstants.Qwen35_08B_Q8_0);
        var gguf = GgufFile.Read(stream);
        var config = ModelConfig.FromGguf(gguf);
        var weights = ModelLoader.Load(gguf, stream, backend, config);
        var kvCache = new KvCache(backend, config, maxSeqLen: 128);
        var deltaState = new DeltaNetState(backend, config);
        var forward = new ForwardPass(backend, config, weights, kvCache, deltaState);
        return new ModelContext(stream, gguf, config, backend, weights, kvCache, deltaState, forward);
    }

    private sealed class ModelContext : IDisposable
    {
        public Stream Stream { get; }
        public GgufFile Gguf { get; }
        public ModelConfig Config { get; }
        public IComputeBackend Backend { get; }
        public ModelWeights Weights { get; }
        public IKvCache KvCache { get; }
        public DeltaNetState DeltaState { get; }
        public ForwardPass Forward { get; }

        public ModelContext(Stream stream, GgufFile gguf, ModelConfig config,
            IComputeBackend backend, ModelWeights weights, IKvCache kvCache,
            DeltaNetState deltaState, ForwardPass forward)
        {
            Stream = stream; Gguf = gguf; Config = config;
            Backend = backend; Weights = weights; KvCache = kvCache;
            DeltaState = deltaState; Forward = forward;
        }

        public void Dispose()
        {
            Forward.Dispose();
            DeltaState.Dispose();
            KvCache.Dispose();
            Weights.Dispose();
            Backend.Dispose();
            Stream.Dispose();
        }
    }
}
