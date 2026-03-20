using Daisi.Llogos.Cpu;
using Daisi.Llogos.Cuda;
using Daisi.Llogos.Gguf;
using Daisi.Llogos.Vulkan;

namespace Daisi.Llogos.Tests.Gpu;

/// <summary>
/// Tests that Q4_K EmbeddingLookup works correctly across CPU, CUDA, and Vulkan backends.
/// </summary>
public class EmbeddingLookupQ4KTests
{
    // Q4_K super block: 256 elements, 144 bytes
    // Layout: 2b d + 2b dmin + 12b packed scales/mins + 128b packed nibbles

    [Fact]
    public void EmbeddingLookup_Q4K_Cpu()
    {
        using var cpu = new CpuBackend();
        var (table, hiddenDim, vocabSize) = CreateQ4KEmbeddingTable();

        using var tableT = cpu.LoadTensor("emb", GgmlType.Q4_K, [hiddenDim, vocabSize], table);
        using var outT = cpu.CreateTensor("out", GgmlType.F32, [hiddenDim]);

        // Should not throw
        cpu.EmbeddingLookup(outT, tableT, 0);

        var result = new float[hiddenDim];
        outT.DequantizeTo(result);

        // Output should have non-zero values
        Assert.True(result.Any(v => v != 0), "Q4_K embedding should produce non-zero values");
        Assert.True(result.All(v => !float.IsNaN(v) && !float.IsInfinity(v)), "No NaN/Inf");
    }

    [Fact]
    public void EmbeddingLookup_Q4K_Cpu_MultipleRows()
    {
        using var cpu = new CpuBackend();
        var (table, hiddenDim, vocabSize) = CreateQ4KEmbeddingTable();

        using var tableT = cpu.LoadTensor("emb", GgmlType.Q4_K, [hiddenDim, vocabSize], table);
        using var outT = cpu.CreateTensor("out", GgmlType.F32, [hiddenDim]);

        // Lookup different rows and verify they differ
        cpu.EmbeddingLookup(outT, tableT, 0);
        var row0 = new float[hiddenDim];
        outT.DequantizeTo(row0);

        cpu.EmbeddingLookup(outT, tableT, 1);
        var row1 = new float[hiddenDim];
        outT.DequantizeTo(row1);

        // Rows should be different (different random data)
        int diffs = 0;
        for (int i = 0; i < hiddenDim; i++)
            if (MathF.Abs(row0[i] - row1[i]) > 1e-6f) diffs++;
        Assert.True(diffs > 0, "Different rows should have different values");
    }

    [Fact]
    public void EmbeddingLookup_Q4K_Cuda_MatchesCpu()
    {
        using var cuda = TryCreateCuda();
        if (cuda == null) return;
        using var cpu = new CpuBackend();

        var (table, hiddenDim, vocabSize) = CreateQ4KEmbeddingTable();

        using var cpuTable = cpu.LoadTensor("emb", GgmlType.Q4_K, [hiddenDim, vocabSize], table);
        using var cpuOut = cpu.CreateTensor("out", GgmlType.F32, [hiddenDim]);
        cpu.EmbeddingLookup(cpuOut, cpuTable, 1);
        var cpuResult = new float[hiddenDim];
        cpuOut.DequantizeTo(cpuResult);

        using var cudaTable = cuda.LoadTensor("emb", GgmlType.Q4_K, [hiddenDim, vocabSize], table);
        using var cudaOut = cuda.CreateTensor("out", GgmlType.F32, [hiddenDim]);
        cuda.EmbeddingLookup(cudaOut, cudaTable, 1);
        var cudaResult = new float[hiddenDim];
        cudaOut.DequantizeTo(cudaResult);

        for (int i = 0; i < hiddenDim; i++)
            Assert.True(MathF.Abs(cpuResult[i] - cudaResult[i]) < 0.01f,
                $"Q4_K EmbLookup CUDA[{i}]: cpu={cpuResult[i]:G6}, cuda={cudaResult[i]:G6}");
    }

    [Fact]
    public void EmbeddingLookup_Q4K_Vulkan_MatchesCpu()
    {
        using var vk = TryCreateVulkan();
        if (vk == null) return;
        using var cpu = new CpuBackend();

        var (table, hiddenDim, vocabSize) = CreateQ4KEmbeddingTable();

        using var cpuTable = cpu.LoadTensor("emb", GgmlType.Q4_K, [hiddenDim, vocabSize], table);
        using var cpuOut = cpu.CreateTensor("out", GgmlType.F32, [hiddenDim]);
        cpu.EmbeddingLookup(cpuOut, cpuTable, 1);
        var cpuResult = new float[hiddenDim];
        cpuOut.DequantizeTo(cpuResult);

        using var vkTable = vk.LoadTensor("emb", GgmlType.Q4_K, [hiddenDim, vocabSize], table);
        using var vkOut = vk.CreateTensor("out", GgmlType.F32, [hiddenDim]);
        vk.EmbeddingLookup(vkOut, vkTable, 1);
        var vkResult = new float[hiddenDim];
        vkOut.DequantizeTo(vkResult);

        for (int i = 0; i < hiddenDim; i++)
            Assert.True(MathF.Abs(cpuResult[i] - vkResult[i]) < 0.01f,
                $"Q4_K EmbLookup Vulkan[{i}]: cpu={cpuResult[i]:G6}, vk={vkResult[i]:G6}");
    }

    [Fact]
    public void EmbeddingLookup_Q4K_Cpu_MatchesDequantize()
    {
        using var cpu = new CpuBackend();
        var (table, hiddenDim, vocabSize) = CreateQ4KEmbeddingTable();

        using var tableT = cpu.LoadTensor("emb", GgmlType.Q4_K, [hiddenDim, vocabSize], table);
        using var outT = cpu.CreateTensor("out", GgmlType.F32, [hiddenDim]);

        // Test that EmbeddingLookup matches a full tensor DequantizeTo
        cpu.EmbeddingLookup(outT, tableT, 0);
        var lookupResult = new float[hiddenDim];
        outT.DequantizeTo(lookupResult);

        // Full dequant for comparison
        var fullDequant = new float[hiddenDim * vocabSize];
        tableT.DequantizeTo(fullDequant);

        for (int i = 0; i < hiddenDim; i++)
            Assert.Equal(fullDequant[i], lookupResult[i], 1e-6f);
    }

    // ── Helpers ─────────────────────────────────────────────────────────────

    /// <summary>
    /// Create a Q4_K embedding table with random data.
    /// hiddenDim must be a multiple of 256 (Q4_K super block size).
    /// </summary>
    private static (byte[] table, int hiddenDim, int vocabSize) CreateQ4KEmbeddingTable()
    {
        int hiddenDim = 256; // 1 super block per row
        int vocabSize = 4;
        int blocksPerRow = hiddenDim / 256;
        int bytesPerRow = blocksPerRow * 144;
        var table = new byte[vocabSize * bytesPerRow];
        var rng = new Random(42);

        for (int row = 0; row < vocabSize; row++)
        {
            for (int blk = 0; blk < blocksPerRow; blk++)
            {
                int off = row * bytesPerRow + blk * 144;
                // d = 0.1 (fp16)
                BitConverter.TryWriteBytes(table.AsSpan(off), (Half)(0.1f * (row + 1)));
                // dmin = 0.05 (fp16)
                BitConverter.TryWriteBytes(table.AsSpan(off + 2), (Half)0.05f);
                // scales/mins (12 bytes) — set all sub-block scales to 1, mins to 0
                for (int i = 0; i < 12; i++)
                    table[off + 4 + i] = (byte)(i < 4 ? 1 : 0); // scale[0..3] = 1 (low 6 bits)
                // nibbles (128 bytes) — random 4-bit values
                for (int i = 0; i < 128; i++)
                    table[off + 16 + i] = (byte)rng.Next(256);
            }
        }
        return (table, hiddenDim, vocabSize);
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
