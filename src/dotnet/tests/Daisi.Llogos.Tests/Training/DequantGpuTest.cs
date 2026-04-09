using Daisi.Llogos.Cpu;
using Daisi.Llogos.Cuda;
using Daisi.Llogos.Gguf;
using Daisi.Llogos.Model;
using Daisi.Llogos.Training;

namespace Daisi.Llogos.Tests.Training;

/// <summary>
/// Compares GPU dequant_to_f32 kernel output against CPU F32Ops.Dequantize
/// for aligned Q8_0 weights. Tests the BatchMatMul dequant path used by training.
/// </summary>
public class DequantGpuTest
{
    [Fact]
    public void GpuDequant_MatchesCpuDequant_Q8_0()
    {
        if (!TestConstants.ModelExists) return;
        try { using var t = new CudaBackend(); } catch { return; }

        // Load model
        using var stream = File.OpenRead(TestConstants.Qwen35_08B_Q8_0);
        var gguf = GgufFile.Read(stream);
        var config = ModelConfig.FromGguf(gguf);

        using var cpu = new CpuBackend();
        var cpuWeights = MmapModelLoader.Load(gguf, TestConstants.Qwen35_08B_Q8_0, cpu, config);

        // Pick a Q8_0 weight tensor (FFN gate, layer 0)
        var cpuTensor = cpuWeights.Layers[0].FfnGate;
        Console.Error.WriteLine($"Tensor: {cpuTensor.Name}, type={cpuTensor.Type}, " +
            $"dims=[{string.Join("×", cpuTensor.Dimensions.ToArray())}], bytes={cpuTensor.ByteSize}");

        // CPU dequant (reference)
        var cpuF32 = F32Ops.Dequantize(cpuTensor);
        Console.Error.WriteLine($"CPU dequant: {cpuF32.Length} floats, first 5: [{string.Join(", ", cpuF32.Take(5).Select(f => f.ToString("F6")))}]");

        // GPU path 1: inference CudaBackend.LoadTensor (known working for inference)
        using var gpu = new CudaBackend();

        var rawBytes = new byte[cpuTensor.ByteSize];
        cpuTensor.CopyRawTo(rawBytes);

        int K = (int)cpuTensor.Dimensions[0]; // input dim
        int N = cpuTensor.Dimensions.Length > 1 ? (int)cpuTensor.Dimensions[1] : 1;
        Console.Error.WriteLine($"K={K}, N={N}, rawBytes={rawBytes.Length}");

        // Use LoadTensor directly — this is the inference path
        var gpuWeight = (CudaTensor)gpu.LoadTensor(cpuTensor.Name + "_gpu",
            cpuTensor.Type, cpuTensor.Dimensions, rawBytes);
        Console.Error.WriteLine($"GPU tensor: aligned8={gpuWeight.IsAlignedQ8_0}, aligned4={gpuWeight.IsAlignedQ4_0}, bytes={gpuWeight.ByteSize}");

        // Direct dequant test: call MatMul with M=1, using first row as activation
        // This tests the M=1 dp4a path which uses the aligned kernel
        int totalElements = K * N;

        // Create activation: first K floats = 1.0 (ones vector)
        var onesData = new float[K];
        Array.Fill(onesData, 1.0f);
        var ones = gpu.CreateTensor("ones", GgmlType.F32, [(long)K]);
        unsafe
        {
            fixed (float* ptr = onesData)
                CudaApi.Check(CudaApi.MemcpyHtoD(((CudaTensor)ones).DevicePtr, ptr,
                    (ulong)(K * sizeof(float))), "cuMemcpyHtoD");
        }

        // MatMul M=1: [1×K] × [K×N] → [1×N] — dot product of ones with each column
        // For Q8_0: this sums all elements in each column (tests dequant correctness)
        var gpuRow = gpu.CreateTensor("row", GgmlType.F32, [(long)N]);
        gpu.MatMul(gpuRow, ones, gpuWeight, 1, K, N);
        gpu.Synchronize();

        var gpuRowResult = new float[N];
        ((CudaTensor)gpuRow).DownloadTo(gpuRowResult);

        // CPU reference: matmul ones[1×K] × weight[N rows × K cols]
        // Weight is [N×K] in storage, but matmul treats it as [K×N] (transposed).
        // output[j] = Σ_i ones[i] * weight_transposed[i][j] = Σ_i weight[j][i]
        // = sum of row j in the stored matrix (each row has K elements)
        var cpuRowSums = new float[N];
        for (int row = 0; row < N; row++)
            for (int col = 0; col < K; col++)
                cpuRowSums[row] += cpuF32[row * K + col];

        Console.Error.WriteLine($"GPU column sums: first 5: [{string.Join(", ", gpuRowResult.Take(5).Select(f => f.ToString("F4")))}]");
        Console.Error.WriteLine($"CPU row sums:    first 5: [{string.Join(", ", cpuRowSums.Take(5).Select(f => f.ToString("F4")))}]");

        // Compare column sums
        int mismatches = 0;
        float maxErr = 0;
        for (int i = 0; i < N; i++)
        {
            float err = Math.Abs(cpuRowSums[i] - gpuRowResult[i]);
            if (err > 0.5f) // loose tolerance for accumulated sums
            {
                if (mismatches < 5)
                    Console.Error.WriteLine($"  Mismatch at row[{i}]: cpu={cpuRowSums[i]:F4} gpu={gpuRowResult[i]:F4} err={err:F4}");
                mismatches++;
            }
            maxErr = Math.Max(maxErr, err);
        }

        Console.Error.WriteLine($"Row sum comparison: {mismatches}/{N} mismatches, maxErr={maxErr:F4}");

        cpuWeights.Dispose();
        gpuWeight.Dispose();
        ones.Dispose();
        gpuRow.Dispose();

        Assert.Equal(0, mismatches);
    }

    /// <summary>
    /// Compare BatchMatMul (Q8_0 GPU dequant + SGEMM) against F32 SGEMM
    /// for M=128 (training batch size). This isolates the numerical difference.
    /// </summary>
    [Fact]
    public void BatchMatMul_Q8_0_vs_F32_M128()
    {
        if (!TestConstants.ModelExists) return;
        try { using var t = new CudaBackend(); } catch { return; }

        using var stream = File.OpenRead(TestConstants.Qwen35_08B_Q8_0);
        var gguf = GgufFile.Read(stream);
        var config = ModelConfig.FromGguf(gguf);
        using var cpuBe = new CpuBackend();
        var cpuWeights = MmapModelLoader.Load(gguf, TestConstants.Qwen35_08B_Q8_0, cpuBe, config);
        var cpuTensor = cpuWeights.Layers[0].FfnGate;

        int K = (int)cpuTensor.Dimensions[0];
        int N = cpuTensor.Dimensions.Length > 1 ? (int)cpuTensor.Dimensions[1] : 1;
        int M = 128;

        using var gpu = new CudaBackend();

        // Create random activation [M × K]
        var rng = new Random(42);
        var actData = new float[M * K];
        for (int i = 0; i < actData.Length; i++) actData[i] = (float)(rng.NextDouble() * 2 - 1) * 0.1f;

        var gpuAct = gpu.CreateTensor("act", GgmlType.F32, [(long)(M * K)]);
        unsafe { fixed (float* p = actData) CudaApi.Check(CudaApi.MemcpyHtoD(((CudaTensor)gpuAct).DevicePtr, p, (ulong)(actData.Length * 4)), "H2D"); }

        // Path 1: Q8_0 weight via LoadTensor + BatchMatMul (GPU dequant + SGEMM)
        var rawBytes = new byte[cpuTensor.ByteSize];
        cpuTensor.CopyRawTo(rawBytes);
        var gpuQ8Weight = gpu.LoadTensor("q8_weight", cpuTensor.Type, cpuTensor.Dimensions, rawBytes);
        var gpuOut1 = gpu.CreateTensor("out1", GgmlType.F32, [(long)(M * N)]);
        gpu.MatMul(gpuOut1, gpuAct, gpuQ8Weight, M, K, N);
        gpu.Synchronize();
        var result1 = new float[M * N];
        ((CudaTensor)gpuOut1).DownloadTo(result1);

        // Path 2: F32 weight (CPU dequant → upload → SGEMM)
        var f32Data = F32Ops.Dequantize(cpuTensor);
        var gpuF32Weight = gpu.CreateTensor("f32_weight", GgmlType.F32, [(long)(K * N)]);
        unsafe { fixed (float* p = f32Data) CudaApi.Check(CudaApi.MemcpyHtoD(((CudaTensor)gpuF32Weight).DevicePtr, p, (ulong)(f32Data.Length * 4)), "H2D"); }
        var gpuOut2 = gpu.CreateTensor("out2", GgmlType.F32, [(long)(M * N)]);
        gpu.MatMul(gpuOut2, gpuAct, gpuF32Weight, M, K, N);
        gpu.Synchronize();
        var result2 = new float[M * N];
        ((CudaTensor)gpuOut2).DownloadTo(result2);

        // Compare
        int mismatches = 0;
        float maxErr = 0, sumErr = 0;
        for (int i = 0; i < result1.Length; i++)
        {
            float err = Math.Abs(result1[i] - result2[i]);
            if (err > 0.01f) mismatches++;
            maxErr = Math.Max(maxErr, err);
            sumErr += err;
        }
        float avgErr = sumErr / result1.Length;

        Console.Error.WriteLine($"BatchMatMul Q8_0 vs F32: M={M}, K={K}, N={N}");
        Console.Error.WriteLine($"  Mismatches (>0.01): {mismatches}/{result1.Length}");
        Console.Error.WriteLine($"  Max error: {maxErr:F6}");
        Console.Error.WriteLine($"  Avg error: {avgErr:F6}");
        Console.Error.WriteLine($"  Q8 first 5: [{string.Join(", ", result1.Take(5).Select(f => f.ToString("F6")))}]");
        Console.Error.WriteLine($"  F32 first 5: [{string.Join(", ", result2.Take(5).Select(f => f.ToString("F6")))}]");

        cpuWeights.Dispose();
        gpuQ8Weight.Dispose();
        gpuF32Weight.Dispose();
        gpuAct.Dispose();
        gpuOut1.Dispose();
        gpuOut2.Dispose();

        // Allow small numerical differences but flag large ones
        Assert.True(maxErr < 0.1f, $"Max error {maxErr} exceeds tolerance 0.1");
    }
}
