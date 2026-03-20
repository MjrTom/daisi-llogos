using System.Runtime.InteropServices;
using Daisi.Llogos.Cpu;
using Daisi.Llogos.Cuda;
using Daisi.Llogos.Gguf;
using Daisi.Llogos.Vulkan;

namespace Daisi.Llogos.Tests.BitNet;

/// <summary>
/// Tests for BitNet GPU compatibility: FillTensor, SquaredReLU, F16 EmbeddingLookup.
/// Each operation is tested on CPU, CUDA, and Vulkan backends.
/// </summary>
public class BitNetGpuTests
{
    // ── FillTensor ──────────────────────────────────────────────────────────

    [Fact]
    public void FillTensor_Cpu_FillsAllElements()
    {
        using var backend = new CpuBackend();
        using var t = backend.CreateTensor("test", GgmlType.F32, [128]);
        backend.FillTensor(t, 88.0f);

        var result = new float[128];
        t.DequantizeTo(result);
        Assert.All(result, v => Assert.Equal(88.0f, v));
    }

    [Fact]
    public void FillTensor_Cpu_ZeroValue()
    {
        using var backend = new CpuBackend();
        using var t = backend.CreateTensor("test", GgmlType.F32, [64]);
        backend.FillTensor(t, 42.0f);
        backend.FillTensor(t, 0.0f);

        var result = new float[64];
        t.DequantizeTo(result);
        Assert.All(result, v => Assert.Equal(0.0f, v));
    }

    [Fact]
    public void FillTensor_Cuda_FillsAllElements()
    {
        using var backend = TryCreateCuda();
        if (backend == null) return;

        using var t = backend.CreateTensor("test", GgmlType.F32, [512]);
        backend.FillTensor(t, 88.0f);

        var result = new float[512];
        t.DequantizeTo(result);
        Assert.All(result, v => Assert.Equal(88.0f, v));
    }

    [Fact]
    public void FillTensor_Cuda_NegativeValue()
    {
        using var backend = TryCreateCuda();
        if (backend == null) return;

        using var t = backend.CreateTensor("test", GgmlType.F32, [256]);
        backend.FillTensor(t, -3.14f);

        var result = new float[256];
        t.DequantizeTo(result);
        Assert.All(result, v => Assert.Equal(-3.14f, v, 1e-5f));
    }

    [Fact]
    public void FillTensor_Vulkan_FillsAllElements()
    {
        using var backend = TryCreateVulkan();
        if (backend == null) return;

        using var t = backend.CreateTensor("test", GgmlType.F32, [512]);
        backend.FillTensor(t, 88.0f);

        var result = new float[512];
        t.DequantizeTo(result);
        Assert.All(result, v => Assert.Equal(88.0f, v));
    }

    [Fact]
    public void FillTensor_Vulkan_NegativeValue()
    {
        using var backend = TryCreateVulkan();
        if (backend == null) return;

        using var t = backend.CreateTensor("test", GgmlType.F32, [256]);
        backend.FillTensor(t, -3.14f);

        var result = new float[256];
        t.DequantizeTo(result);
        Assert.All(result, v => Assert.Equal(-3.14f, v, 1e-5f));
    }

    // ── SquaredReLU ─────────────────────────────────────────────────────────

    [Fact]
    public void SquaredReLU_Cpu_MatchesExpected()
    {
        using var backend = new CpuBackend();
        float[] input = [-2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f, -0.5f, 0.5f];
        float[] expected = [0.0f, 0.0f, 0.0f, 1.0f, 4.0f, 9.0f, 0.0f, 0.25f];

        using var t = LoadF32(backend, "test", input);
        backend.SquaredReLU(t);

        var result = DownloadF32(t, input.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.Equal(expected[i], result[i], 1e-5f);
    }

    [Fact]
    public void SquaredReLU_Cuda_MatchesCpu()
    {
        using var cuda = TryCreateCuda();
        if (cuda == null) return;
        using var cpu = new CpuBackend();

        float[] input = [-2.0f, -1.0f, 0.0f, 0.5f, 1.0f, 2.0f, 3.0f, -0.5f];

        using var cpuT = LoadF32(cpu, "test", input);
        cpu.SquaredReLU(cpuT);
        var expected = DownloadF32(cpuT, input.Length);

        using var cudaT = LoadF32(cuda, "test", input);
        cuda.SquaredReLU(cudaT);
        var result = DownloadF32(cudaT, input.Length);

        for (int i = 0; i < expected.Length; i++)
            Assert.Equal(expected[i], result[i], 1e-5f);
    }

    [Fact]
    public void SquaredReLU_Cuda_LargerBuffer()
    {
        using var cuda = TryCreateCuda();
        if (cuda == null) return;

        int n = 2048;
        var input = new float[n];
        var rng = new Random(42);
        for (int i = 0; i < n; i++)
            input[i] = (float)(rng.NextDouble() * 6 - 3); // range [-3, 3]

        using var t = LoadF32(cuda, "test", input);
        cuda.SquaredReLU(t);
        var result = DownloadF32(t, n);

        for (int i = 0; i < n; i++)
        {
            float x = MathF.Max(0, input[i]);
            float expected = x * x;
            Assert.True(MathF.Abs(result[i] - expected) < 1e-4f,
                $"SquaredReLU[{i}]: expected {expected}, got {result[i]}");
        }
    }

    [Fact]
    public void SquaredReLU_Vulkan_MatchesCpu()
    {
        using var vk = TryCreateVulkan();
        if (vk == null) return;
        using var cpu = new CpuBackend();

        float[] input = [-2.0f, -1.0f, 0.0f, 0.5f, 1.0f, 2.0f, 3.0f, -0.5f];

        using var cpuT = LoadF32(cpu, "test", input);
        cpu.SquaredReLU(cpuT);
        var expected = DownloadF32(cpuT, input.Length);

        using var vkT = LoadF32(vk, "test", input);
        vk.SquaredReLU(vkT);
        var result = DownloadF32(vkT, input.Length);

        for (int i = 0; i < expected.Length; i++)
            Assert.Equal(expected[i], result[i], 1e-5f);
    }

    [Fact]
    public void SquaredReLU_Vulkan_LargerBuffer()
    {
        using var vk = TryCreateVulkan();
        if (vk == null) return;

        int n = 2048;
        var input = new float[n];
        var rng = new Random(42);
        for (int i = 0; i < n; i++)
            input[i] = (float)(rng.NextDouble() * 6 - 3);

        using var t = LoadF32(vk, "test", input);
        vk.SquaredReLU(t);
        var result = DownloadF32(t, n);

        for (int i = 0; i < n; i++)
        {
            float x = MathF.Max(0, input[i]);
            float expected = x * x;
            Assert.True(MathF.Abs(result[i] - expected) < 1e-4f,
                $"SquaredReLU[{i}]: expected {expected}, got {result[i]}");
        }
    }

    // ── F16 EmbeddingLookup ─────────────────────────────────────────────────

    [Fact]
    public void EmbeddingLookup_F16_Cpu_MatchesExpected()
    {
        using var backend = new CpuBackend();
        var (table, hiddenDim, vocabSize) = CreateF16EmbeddingTable();

        using var tableT = backend.LoadTensor("emb", GgmlType.F16, [hiddenDim, vocabSize], table);
        using var outT = backend.CreateTensor("out", GgmlType.F32, [hiddenDim]);

        backend.EmbeddingLookup(outT, tableT, 1);
        var result = DownloadF32(outT, hiddenDim);

        // Token 1 row starts at hiddenDim*2 bytes, each element is (Half)(i + 10)
        for (int i = 0; i < hiddenDim; i++)
            Assert.Equal((float)(Half)(i + 10), result[i], 0.01f);
    }

    [Fact]
    public void EmbeddingLookup_F16_Cuda_MatchesCpu()
    {
        using var cuda = TryCreateCuda();
        if (cuda == null) return;
        using var cpu = new CpuBackend();

        var (table, hiddenDim, vocabSize) = CreateF16EmbeddingTable();

        // CPU reference
        using var cpuTable = cpu.LoadTensor("emb", GgmlType.F16, [hiddenDim, vocabSize], table);
        using var cpuOut = cpu.CreateTensor("out", GgmlType.F32, [hiddenDim]);
        cpu.EmbeddingLookup(cpuOut, cpuTable, 2);
        var expected = DownloadF32(cpuOut, hiddenDim);

        // CUDA
        using var cudaTable = cuda.LoadTensor("emb", GgmlType.F16, [hiddenDim, vocabSize], table);
        using var cudaOut = cuda.CreateTensor("out", GgmlType.F32, [hiddenDim]);
        cuda.EmbeddingLookup(cudaOut, cudaTable, 2);
        var result = DownloadF32(cudaOut, hiddenDim);

        for (int i = 0; i < hiddenDim; i++)
            Assert.Equal(expected[i], result[i], 0.01f);
    }

    [Fact]
    public void EmbeddingLookup_F16_Cuda_AllRows()
    {
        using var cuda = TryCreateCuda();
        if (cuda == null) return;

        var (table, hiddenDim, vocabSize) = CreateF16EmbeddingTable();
        using var tableT = cuda.LoadTensor("emb", GgmlType.F16, [hiddenDim, vocabSize], table);
        using var outT = cuda.CreateTensor("out", GgmlType.F32, [hiddenDim]);

        for (int row = 0; row < vocabSize; row++)
        {
            cuda.EmbeddingLookup(outT, tableT, row);
            var result = DownloadF32(outT, hiddenDim);

            float baseVal = row * 10;
            for (int i = 0; i < hiddenDim; i++)
                Assert.Equal((float)(Half)(i + baseVal), result[i], 0.01f);
        }
    }

    [Fact]
    public void EmbeddingLookup_F16_Vulkan_MatchesCpu()
    {
        using var vk = TryCreateVulkan();
        if (vk == null) return;
        using var cpu = new CpuBackend();

        var (table, hiddenDim, vocabSize) = CreateF16EmbeddingTable();

        // CPU reference
        using var cpuTable = cpu.LoadTensor("emb", GgmlType.F16, [hiddenDim, vocabSize], table);
        using var cpuOut = cpu.CreateTensor("out", GgmlType.F32, [hiddenDim]);
        cpu.EmbeddingLookup(cpuOut, cpuTable, 2);
        var expected = DownloadF32(cpuOut, hiddenDim);

        // Vulkan
        using var vkTable = vk.LoadTensor("emb", GgmlType.F16, [hiddenDim, vocabSize], table);
        using var vkOut = vk.CreateTensor("out", GgmlType.F32, [hiddenDim]);
        vk.EmbeddingLookup(vkOut, vkTable, 2);
        var result = DownloadF32(vkOut, hiddenDim);

        for (int i = 0; i < hiddenDim; i++)
            Assert.Equal(expected[i], result[i], 0.01f);
    }

    // ── Helpers ─────────────────────────────────────────────────────────────

    private static (byte[] table, int hiddenDim, int vocabSize) CreateF16EmbeddingTable()
    {
        int hiddenDim = 64;
        int vocabSize = 4;
        var table = new byte[vocabSize * hiddenDim * 2]; // 2 bytes per Half

        for (int row = 0; row < vocabSize; row++)
        {
            float baseVal = row * 10;
            for (int col = 0; col < hiddenDim; col++)
            {
                int byteOff = (row * hiddenDim + col) * 2;
                BitConverter.TryWriteBytes(table.AsSpan(byteOff), (Half)(col + baseVal));
            }
        }
        return (table, hiddenDim, vocabSize);
    }

    private static ITensor LoadF32(IComputeBackend backend, string name, float[] data)
    {
        var t = backend.CreateTensor(name, GgmlType.F32, [data.Length]);
        var bytes = new byte[data.Length * 4];
        Buffer.BlockCopy(data, 0, bytes, 0, bytes.Length);
        t.CopyFrom(bytes);
        return t;
    }

    private static float[] DownloadF32(ITensor t, int count)
    {
        var result = new float[count];
        t.DequantizeTo(result);
        return result;
    }

    private static CudaBackend? TryCreateCuda()
    {
        try { return new CudaBackend(); }
        catch { return null; }
    }

    private static VulkanBackend? TryCreateVulkan()
    {
        try { return new VulkanBackend(); }
        catch { return null; }
    }
}
