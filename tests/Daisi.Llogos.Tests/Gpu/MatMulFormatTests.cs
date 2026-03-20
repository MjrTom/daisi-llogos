using Daisi.Llogos.Cpu;
using Daisi.Llogos.Cuda;
using Daisi.Llogos.Gguf;
using Daisi.Llogos.Vulkan;

namespace Daisi.Llogos.Tests.Gpu;

/// <summary>
/// Tests that MatMul produces consistent results across CPU, CUDA, and Vulkan
/// for all supported weight formats: I2_S, TQ1_0, and F16.
/// GPU tests skip gracefully when no GPU is available.
/// </summary>
public class MatMulFormatTests
{
    // ── I2_S (BitNet ternary) ───────────────────────────────────────────────

    [Fact]
    public void MatMul_I2S_Cuda_MatchesCpu()
    {
        using var cuda = TryCreateCuda();
        if (cuda == null) return;
        using var cpu = new CpuBackend();

        int K = 256, N = 32;
        var aData = RandomFloats(K, seed: 42);
        var bData = CreateI2SWeights(K, N, seed: 123);

        var cpuResult = RunMatMul(cpu, aData, bData, GgmlType.I2_S, K, N);
        var cudaResult = RunMatMul(cuda, aData, bData, GgmlType.I2_S, K, N);

        AssertClose(cpuResult, cudaResult, 1e-3f, "I2_S CUDA vs CPU");
    }

    [Fact]
    public void MatMul_I2S_Vulkan_MatchesCpu()
    {
        using var vk = TryCreateVulkan();
        if (vk == null) return;
        using var cpu = new CpuBackend();

        int K = 256, N = 32;
        var aData = RandomFloats(K, seed: 42);
        var bData = CreateI2SWeights(K, N, seed: 123);

        var cpuResult = RunMatMul(cpu, aData, bData, GgmlType.I2_S, K, N);
        var vkResult = RunMatMul(vk, aData, bData, GgmlType.I2_S, K, N);

        AssertClose(cpuResult, vkResult, 1e-3f, "I2_S Vulkan vs CPU");
    }

    [Fact]
    public void MatMul_I2S_Cuda_LargerMatrix()
    {
        using var cuda = TryCreateCuda();
        if (cuda == null) return;
        using var cpu = new CpuBackend();

        int K = 1280, N = 128;
        var aData = RandomFloats(K, seed: 77);
        var bData = CreateI2SWeights(K, N, seed: 88);

        var cpuResult = RunMatMul(cpu, aData, bData, GgmlType.I2_S, K, N);
        var cudaResult = RunMatMul(cuda, aData, bData, GgmlType.I2_S, K, N);

        AssertClose(cpuResult, cudaResult, 1e-2f, "I2_S CUDA large");
    }

    [Fact]
    public void MatMul_I2S_Vulkan_LargerMatrix()
    {
        using var vk = TryCreateVulkan();
        if (vk == null) return;
        using var cpu = new CpuBackend();

        int K = 1280, N = 128;
        var aData = RandomFloats(K, seed: 77);
        var bData = CreateI2SWeights(K, N, seed: 88);

        var cpuResult = RunMatMul(cpu, aData, bData, GgmlType.I2_S, K, N);
        var vkResult = RunMatMul(vk, aData, bData, GgmlType.I2_S, K, N);

        AssertClose(cpuResult, vkResult, 1e-2f, "I2_S Vulkan large");
    }

    [Fact]
    public void MatMul_I2S_ScaleApplied()
    {
        using var cuda = TryCreateCuda();
        if (cuda == null) return;
        using var cpu = new CpuBackend();

        int K = 128, N = 4;
        var aData = new float[K];
        Array.Fill(aData, 1.0f);

        // All +1 weights with scale=2.0 — each output should be K * 2.0
        var bData = CreateI2SWeightsAllOnes(K, N, scale: 2.0f);

        var cpuResult = RunMatMul(cpu, aData, bData, GgmlType.I2_S, K, N);
        var cudaResult = RunMatMul(cuda, aData, bData, GgmlType.I2_S, K, N);

        for (int i = 0; i < N; i++)
        {
            Assert.Equal(K * 2.0f, cpuResult[i], 1e-3f);
            Assert.Equal(K * 2.0f, cudaResult[i], 1e-3f);
        }
    }

    // ── TQ1_0 (ternary base-3) ─────────────────────────────────────────────

    [Fact]
    public void MatMul_TQ1_0_Cuda_MatchesCpu()
    {
        using var cuda = TryCreateCuda();
        if (cuda == null) return;
        using var cpu = new CpuBackend();

        int K = 256, N = 16;
        var aData = RandomFloats(K, seed: 42);
        var bData = CreateTQ1_0Weights(K, N, seed: 99);

        var cpuResult = RunMatMul(cpu, aData, bData, GgmlType.TQ1_0, K, N);
        var cudaResult = RunMatMul(cuda, aData, bData, GgmlType.TQ1_0, K, N);

        AssertClose(cpuResult, cudaResult, 1e-3f, "TQ1_0 CUDA vs CPU");
    }

    [Fact]
    public void MatMul_TQ1_0_Vulkan_MatchesCpu()
    {
        using var vk = TryCreateVulkan();
        if (vk == null) return;
        using var cpu = new CpuBackend();

        int K = 256, N = 16;
        var aData = RandomFloats(K, seed: 42);
        var bData = CreateTQ1_0Weights(K, N, seed: 99);

        var cpuResult = RunMatMul(cpu, aData, bData, GgmlType.TQ1_0, K, N);
        var vkResult = RunMatMul(vk, aData, bData, GgmlType.TQ1_0, K, N);

        AssertClose(cpuResult, vkResult, 1e-3f, "TQ1_0 Vulkan vs CPU");
    }

    [Fact]
    public void MatMul_TQ1_0_Cuda_LargerMatrix()
    {
        using var cuda = TryCreateCuda();
        if (cuda == null) return;
        using var cpu = new CpuBackend();

        int K = 1024, N = 64;
        var aData = RandomFloats(K, seed: 55);
        var bData = CreateTQ1_0Weights(K, N, seed: 66);

        var cpuResult = RunMatMul(cpu, aData, bData, GgmlType.TQ1_0, K, N);
        var cudaResult = RunMatMul(cuda, aData, bData, GgmlType.TQ1_0, K, N);

        AssertClose(cpuResult, cudaResult, 1e-2f, "TQ1_0 CUDA large");
    }

    [Fact]
    public void MatMul_TQ1_0_Vulkan_LargerMatrix()
    {
        using var vk = TryCreateVulkan();
        if (vk == null) return;
        using var cpu = new CpuBackend();

        int K = 1024, N = 64;
        var aData = RandomFloats(K, seed: 55);
        var bData = CreateTQ1_0Weights(K, N, seed: 66);

        var cpuResult = RunMatMul(cpu, aData, bData, GgmlType.TQ1_0, K, N);
        var vkResult = RunMatMul(vk, aData, bData, GgmlType.TQ1_0, K, N);

        AssertClose(cpuResult, vkResult, 1e-2f, "TQ1_0 Vulkan large");
    }

    [Fact]
    public void MatMul_TQ1_0_AllOnes_Sums()
    {
        using var cuda = TryCreateCuda();
        if (cuda == null) return;
        using var cpu = new CpuBackend();

        int K = 256, N = 4;
        var aData = new float[K];
        Array.Fill(aData, 1.0f);

        // All +1 weights: each output should equal K
        var bData = CreateTQ1_0WeightsAllOnes(K, N);

        var cpuResult = RunMatMul(cpu, aData, bData, GgmlType.TQ1_0, K, N);
        var cudaResult = RunMatMul(cuda, aData, bData, GgmlType.TQ1_0, K, N);

        for (int i = 0; i < N; i++)
        {
            Assert.Equal((float)K, cpuResult[i], 1e-3f);
            Assert.Equal((float)K, cudaResult[i], 1e-3f);
        }
    }

    // ── F16 ────────────────────────────────────────────────────────────────

    [Fact]
    public void MatMul_F16_Cuda_MatchesCpu()
    {
        using var cuda = TryCreateCuda();
        if (cuda == null) return;
        using var cpu = new CpuBackend();

        int K = 128, N = 32;
        var aData = RandomFloats(K, seed: 42);
        var bData = CreateF16Weights(K, N, seed: 101);

        var cpuResult = RunMatMul(cpu, aData, bData, GgmlType.F16, K, N);
        var cudaResult = RunMatMul(cuda, aData, bData, GgmlType.F16, K, N);

        AssertClose(cpuResult, cudaResult, 1e-2f, "F16 CUDA vs CPU");
    }

    [Fact]
    public void MatMul_F16_Vulkan_MatchesCpu()
    {
        using var vk = TryCreateVulkan();
        if (vk == null) return;
        using var cpu = new CpuBackend();

        int K = 128, N = 32;
        var aData = RandomFloats(K, seed: 42);
        var bData = CreateF16Weights(K, N, seed: 101);

        var cpuResult = RunMatMul(cpu, aData, bData, GgmlType.F16, K, N);
        var vkResult = RunMatMul(vk, aData, bData, GgmlType.F16, K, N);

        AssertClose(cpuResult, vkResult, 1e-2f, "F16 Vulkan vs CPU");
    }

    [Fact]
    public void MatMul_F16_Cuda_LargerMatrix()
    {
        using var cuda = TryCreateCuda();
        if (cuda == null) return;
        using var cpu = new CpuBackend();

        int K = 1024, N = 256;
        var aData = RandomFloats(K, seed: 33);
        var bData = CreateF16Weights(K, N, seed: 44);

        var cpuResult = RunMatMul(cpu, aData, bData, GgmlType.F16, K, N);
        var cudaResult = RunMatMul(cuda, aData, bData, GgmlType.F16, K, N);

        AssertClose(cpuResult, cudaResult, 0.05f, "F16 CUDA large");
    }

    [Fact]
    public void MatMul_F16_Vulkan_LargerMatrix()
    {
        using var vk = TryCreateVulkan();
        if (vk == null) return;
        using var cpu = new CpuBackend();

        int K = 1024, N = 256;
        var aData = RandomFloats(K, seed: 33);
        var bData = CreateF16Weights(K, N, seed: 44);

        var cpuResult = RunMatMul(cpu, aData, bData, GgmlType.F16, K, N);
        var vkResult = RunMatMul(vk, aData, bData, GgmlType.F16, K, N);

        AssertClose(cpuResult, vkResult, 0.05f, "F16 Vulkan large");
    }

    [Fact]
    public void MatMul_F16_Identity()
    {
        using var cuda = TryCreateCuda();
        if (cuda == null) return;

        // a = [1, 0, 0, 0], weights are identity-like rows
        int K = 4, N = 4;
        var aData = new float[] { 1, 0, 0, 0 };
        // Row j has 1.0 at position j, 0 elsewhere
        var bBytes = new byte[N * K * 2];
        for (int j = 0; j < N; j++)
        {
            int off = (j * K + j) * 2;
            BitConverter.TryWriteBytes(bBytes.AsSpan(off), (Half)1.0f);
        }

        var result = RunMatMul(cuda, aData, bBytes, GgmlType.F16, K, N);

        // output[0] = dot(a, row0) = 1.0, output[1..3] = 0
        Assert.Equal(1.0f, result[0], 1e-3f);
        Assert.Equal(0.0f, result[1], 1e-3f);
        Assert.Equal(0.0f, result[2], 1e-3f);
        Assert.Equal(0.0f, result[3], 1e-3f);
    }

    // ── Helpers ─────────────────────────────────────────────────────────────

    private static float[] RunMatMul(IComputeBackend backend, float[] aData, byte[] bData,
        GgmlType bType, int K, int N)
    {
        var aBytes = new byte[aData.Length * 4];
        Buffer.BlockCopy(aData, 0, aBytes, 0, aBytes.Length);

        using var aT = backend.LoadTensor("a", GgmlType.F32, [K], aBytes);
        using var bT = backend.LoadTensor("b", bType, [K, N], bData);
        using var outT = backend.CreateTensor("out", GgmlType.F32, [N]);

        backend.MatMul(outT, aT, bT, 1, K, N);

        var result = new float[N];
        outT.DequantizeTo(result);
        return result;
    }

    private static float[] RandomFloats(int count, int seed)
    {
        var rng = new Random(seed);
        var data = new float[count];
        for (int i = 0; i < count; i++)
            data[i] = (float)(rng.NextDouble() * 2 - 1);
        return data;
    }

    /// <summary>
    /// Create I2_S weight data: [N x K] packed as 2-bit ternary with per-tensor scale.
    /// </summary>
    private static byte[] CreateI2SWeights(int K, int N, int seed)
    {
        var rng = new Random(seed);
        long totalPacked = (long)K * N / 4;
        var data = new byte[totalPacked + 32]; // +32 byte trailer

        int packedPerRow = K / 4;
        for (int row = 0; row < N; row++)
        {
            for (int gp = 0; gp < packedPerRow; gp++)
            {
                // Random 2-bit codes: 0=-1, 1=0, 2=+1 (avoid 3)
                int c0 = rng.Next(3); // 0,1,2
                int c1 = rng.Next(3);
                int c2 = rng.Next(3);
                int c3 = rng.Next(3);
                data[row * packedPerRow + gp] = (byte)((c0 << 6) | (c1 << 4) | (c2 << 2) | c3);
            }
        }

        // Per-tensor scale
        BitConverter.TryWriteBytes(data.AsSpan((int)totalPacked), 0.5f);
        return data;
    }

    /// <summary>
    /// Create I2_S weights where all values are +1 (code=0b10) with the given scale.
    /// </summary>
    private static byte[] CreateI2SWeightsAllOnes(int K, int N, float scale)
    {
        long totalPacked = (long)K * N / 4;
        var data = new byte[totalPacked + 32];

        // 0b10_10_10_10 = all +1 in all four groups
        for (int i = 0; i < (int)totalPacked; i++)
            data[i] = 0b10_10_10_10;

        BitConverter.TryWriteBytes(data.AsSpan((int)totalPacked), scale);
        return data;
    }

    /// <summary>
    /// Create TQ1_0 weight data: [N x K] packed as base-3 ternary blocks.
    /// K must be a multiple of 256. Each block = 54 bytes (52 base-3 + 2 padding).
    /// </summary>
    private static byte[] CreateTQ1_0Weights(int K, int N, int seed)
    {
        var rng = new Random(seed);
        int blocksPerRow = K / 256;
        int bytesPerRow = blocksPerRow * 54;
        var data = new byte[N * bytesPerRow];

        for (int row = 0; row < N; row++)
        {
            for (int blk = 0; blk < blocksPerRow; blk++)
            {
                int blockOff = row * bytesPerRow + blk * 54;
                int elemIdx = 0;
                for (int byteIdx = 0; byteIdx < 52 && elemIdx < 256; byteIdx++)
                {
                    // Pack 5 trits into one byte: trit values 0(-1), 1(0), 2(+1)
                    int packed = 0;
                    int mul = 1;
                    for (int t = 0; t < 5 && elemIdx < 256; t++)
                    {
                        int trit = rng.Next(3); // 0,1,2
                        packed += trit * mul;
                        mul *= 3;
                        elemIdx++;
                    }
                    data[blockOff + byteIdx] = (byte)packed;
                }
                // 2 padding bytes stay zero
            }
        }
        return data;
    }

    /// <summary>
    /// Create TQ1_0 weights where all values are +1 (trit=2).
    /// </summary>
    private static byte[] CreateTQ1_0WeightsAllOnes(int K, int N)
    {
        int blocksPerRow = K / 256;
        int bytesPerRow = blocksPerRow * 54;
        var data = new byte[N * bytesPerRow];

        for (int row = 0; row < N; row++)
        {
            for (int blk = 0; blk < blocksPerRow; blk++)
            {
                int blockOff = row * bytesPerRow + blk * 54;
                int elemIdx = 0;
                for (int byteIdx = 0; byteIdx < 52 && elemIdx < 256; byteIdx++)
                {
                    // All trits = 2 (+1): 2 + 2*3 + 2*9 + 2*27 + 2*81 = 2+6+18+54+162 = 242
                    int tritsThisByte = Math.Min(5, 256 - elemIdx);
                    int packed = 0;
                    int mul = 1;
                    for (int t = 0; t < tritsThisByte; t++)
                    {
                        packed += 2 * mul; // trit=2 means +1
                        mul *= 3;
                    }
                    data[blockOff + byteIdx] = (byte)packed;
                    elemIdx += tritsThisByte;
                }
            }
        }
        return data;
    }

    /// <summary>
    /// Create F16 weight data: [N x K] stored as Half values.
    /// </summary>
    private static byte[] CreateF16Weights(int K, int N, int seed)
    {
        var rng = new Random(seed);
        var data = new byte[N * K * 2];
        for (int i = 0; i < N * K; i++)
        {
            float val = (float)(rng.NextDouble() * 2 - 1);
            BitConverter.TryWriteBytes(data.AsSpan(i * 2), (Half)val);
        }
        return data;
    }

    private static void AssertClose(float[] expected, float[] actual, float tolerance, string label)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(MathF.Abs(expected[i] - actual[i]) < tolerance,
                $"{label}[{i}]: expected {expected[i]:G6}, got {actual[i]:G6}, diff {MathF.Abs(expected[i] - actual[i]):G6}");
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
