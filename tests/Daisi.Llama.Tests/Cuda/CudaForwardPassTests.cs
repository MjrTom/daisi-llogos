using Daisi.Llama.Cuda;
using Daisi.Llama.Gguf;

namespace Daisi.Llama.Tests.Cuda;

/// <summary>
/// Tests that validate CUDA kernel correctness on realistic tensor sizes
/// matching the Qwen 3.5 0.8B model dimensions.
/// Full GPU forward pass requires attention/DeltaNet kernels (Phase 8).
/// </summary>
public class CudaRealisticSizeTests : IDisposable
{
    private readonly CudaBackend? _backend;

    public CudaRealisticSizeTests()
    {
        try { _backend = new CudaBackend(); }
        catch { _backend = null; }
    }

    private bool HasGpu => _backend != null;

    [Fact]
    public void MatMul_ModelScale_HiddenToIntermediate()
    {
        if (!HasGpu) return;

        // Simulates hidden→FFN projection: [1×1024] × [3584×1024] → [1×3584]
        int K = 1024, N = 3584;
        var a = RandomFloats(K, 42);
        var b = RandomFloats(N * K, 43);

        // CPU reference
        var expected = CpuMatMul(a, b, 1, K, N);

        using var aT = LoadF32("a", a);
        using var bT = LoadF32("b", b);
        using var outT = _backend!.CreateTensor("out", GgmlType.F32, [N]);

        _backend.MatMul(outT, aT, bT, 1, K, N);

        var result = DownloadF32(outT, N);
        AssertClose(expected, result, 0.05f, "MatMul_HiddenToIntermediate");
    }

    [Fact]
    public void RmsNorm_HiddenDim()
    {
        if (!HasGpu) return;

        int n = 1024;
        var input = RandomFloats(n, 44);
        var weight = RandomFloats(n, 45);
        float eps = 1e-6f;

        // CPU reference
        using var cpuB = new Daisi.Llama.Cpu.CpuBackend();
        using var cpuIn = cpuB.CreateTensor("in", GgmlType.F32, [n]);
        using var cpuW = cpuB.CreateTensor("w", GgmlType.F32, [n]);
        using var cpuOut = cpuB.CreateTensor("out", GgmlType.F32, [n]);
        CopyF32(input, cpuIn);
        CopyF32(weight, cpuW);
        cpuB.RmsNorm(cpuOut, cpuIn, cpuW, eps);
        var expected = cpuOut.AsFloatSpan().ToArray();

        using var inT = LoadF32("in", input);
        using var wT = LoadF32("w", weight);
        using var outT = _backend!.CreateTensor("out", GgmlType.F32, [n]);
        _backend.RmsNorm(outT, inT, wT, eps);
        var result = DownloadF32(outT, n);

        AssertClose(expected, result, 1e-3f, "RmsNorm_HiddenDim");
    }

    [Fact]
    public void Softmax_VocabSize()
    {
        if (!HasGpu) return;

        int n = 151936; // Qwen vocab size
        var input = RandomFloats(n, 46);

        // CPU reference
        using var cpuB = new Daisi.Llama.Cpu.CpuBackend();
        using var cpuIn = cpuB.CreateTensor("in", GgmlType.F32, [n]);
        using var cpuOut = cpuB.CreateTensor("out", GgmlType.F32, [n]);
        CopyF32(input, cpuIn);
        cpuB.Softmax(cpuOut, cpuIn);
        var expected = cpuOut.AsFloatSpan().ToArray();

        using var inT = LoadF32("in", input);
        using var outT = _backend!.CreateTensor("out", GgmlType.F32, [n]);
        _backend.Softmax(outT, inT);
        var result = DownloadF32(outT, n);

        AssertClose(expected, result, 1e-5f, "Softmax_VocabSize");
    }

    [Fact]
    public void SiLU_IntermediateDim()
    {
        if (!HasGpu) return;

        int n = 3584;
        var input = RandomFloats(n, 47);

        using var inT = LoadF32("in", input);
        using var outT = _backend!.CreateTensor("out", GgmlType.F32, [n]);
        _backend.SiLU(outT, inT);
        var result = DownloadF32(outT, n);

        for (int i = 0; i < n; i++)
        {
            float expected = input[i] / (1.0f + MathF.Exp(-input[i]));
            Assert.True(MathF.Abs(result[i] - expected) < 1e-4f,
                $"SiLU[{i}]: expected {expected}, got {result[i]}");
        }
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    private static float[] RandomFloats(int count, int seed)
    {
        var rng = new Random(seed);
        var data = new float[count];
        for (int i = 0; i < count; i++)
            data[i] = (float)(rng.NextDouble() * 2 - 1);
        return data;
    }

    private static float[] CpuMatMul(float[] a, float[] b, int M, int K, int N)
    {
        var output = new float[M * N];
        for (int m = 0; m < M; m++)
            for (int n = 0; n < N; n++)
            {
                float sum = 0;
                for (int k = 0; k < K; k++)
                    sum += a[m * K + k] * b[n * K + k];
                output[m * N + n] = sum;
            }
        return output;
    }

    private ITensor LoadF32(string name, float[] data)
    {
        var t = _backend!.CreateTensor(name, GgmlType.F32, [data.Length]);
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

    private static void CopyF32(float[] data, ITensor t)
    {
        var bytes = new byte[data.Length * 4];
        Buffer.BlockCopy(data, 0, bytes, 0, bytes.Length);
        t.CopyFrom(bytes);
    }

    private static void AssertClose(float[] expected, float[] actual, float tol, string label)
    {
        Assert.Equal(expected.Length, actual.Length);
        int mismatches = 0;
        float maxDiff = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            float diff = MathF.Abs(expected[i] - actual[i]);
            if (diff > tol) mismatches++;
            if (diff > maxDiff) maxDiff = diff;
        }
        Assert.True(mismatches == 0,
            $"{label}: {mismatches}/{expected.Length} elements differ by >{tol} (max diff: {maxDiff})");
    }

    public void Dispose() => _backend?.Dispose();
}
