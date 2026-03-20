using Daisi.Llogos.Cuda;
using Daisi.Llogos.Gguf;

namespace Daisi.Llogos.Tests.Cuda;

/// <summary>
/// Tests for the CUDA backend. Skipped if no NVIDIA GPU is available.
/// </summary>
public class CudaBackendTests : IDisposable
{
    private readonly CudaBackend? _backend;

    public CudaBackendTests()
    {
        try { _backend = new CudaBackend(); }
        catch { _backend = null; }
    }

    private bool HasGpu => _backend != null;

    [Fact]
    public void Context_Creates()
    {
        if (!HasGpu) return;
        Assert.Contains("CUDA", _backend!.Name);
    }

    [Fact]
    public void Tensor_CreateAndDispose()
    {
        if (!HasGpu) return;
        using var t = _backend!.CreateTensor("test", GgmlType.F32, [128]);
        Assert.Equal(128, t.ElementCount);
        Assert.Equal(128 * 4, t.ByteSize);
    }

    [Fact]
    public void DeviceMemory_H2D_D2H_RoundTrip()
    {
        if (!HasGpu) return;
        var data = new float[] { 1.0f, 2.0f, 3.0f, 4.0f };
        using var t = _backend!.CreateTensor("rt", GgmlType.F32, [4]);

        // Upload
        var bytes = new byte[16];
        Buffer.BlockCopy(data, 0, bytes, 0, 16);
        t.CopyFrom(bytes);

        // Download
        var result = new float[4];
        t.DequantizeTo(result);

        Assert.Equal(data, result);
    }

    [Fact]
    public void ElementAdd_MatchesCpu()
    {
        if (!HasGpu) return;

        float[] a = [1.0f, 2.0f, 3.0f, 4.0f];
        float[] b = [5.0f, 6.0f, 7.0f, 8.0f];

        using var aT = LoadF32("a", a);
        using var bT = LoadF32("b", b);
        using var outT = _backend!.CreateTensor("out", GgmlType.F32, [4]);

        _backend.ElementAdd(outT, aT, bT);

        var result = DownloadF32(outT, 4);
        Assert.Equal([6.0f, 8.0f, 10.0f, 12.0f], result);
    }

    [Fact]
    public void ElementMul_MatchesCpu()
    {
        if (!HasGpu) return;

        float[] a = [1.0f, 2.0f, 3.0f, 4.0f];
        float[] b = [5.0f, 6.0f, 7.0f, 8.0f];

        using var aT = LoadF32("a", a);
        using var bT = LoadF32("b", b);
        using var outT = _backend!.CreateTensor("out", GgmlType.F32, [4]);

        _backend.ElementMul(outT, aT, bT);

        var result = DownloadF32(outT, 4);
        Assert.Equal([5.0f, 12.0f, 21.0f, 32.0f], result);
    }

    [Fact]
    public void SiLU_MatchesCpu()
    {
        if (!HasGpu) return;

        float[] input = [0.0f, 1.0f, -1.0f, 2.0f];
        using var inT = LoadF32("in", input);
        using var outT = _backend!.CreateTensor("out", GgmlType.F32, [4]);

        _backend.SiLU(outT, inT);

        var result = DownloadF32(outT, 4);

        // SiLU(x) = x * sigmoid(x)
        for (int i = 0; i < 4; i++)
        {
            float expected = input[i] / (1.0f + MathF.Exp(-input[i]));
            Assert.True(MathF.Abs(result[i] - expected) < 1e-5f,
                $"SiLU[{i}]: expected {expected}, got {result[i]}");
        }
    }

    [Fact]
    public void RmsNorm_MatchesCpu()
    {
        if (!HasGpu) return;

        float[] input = [1.0f, 2.0f, 3.0f, 4.0f];
        float[] weight = [1.0f, 1.0f, 1.0f, 1.0f];
        float eps = 1e-6f;

        // CPU reference
        using var cpuBackend = new Daisi.Llogos.Cpu.CpuBackend();
        using var cpuIn = cpuBackend.CreateTensor("in", GgmlType.F32, [4]);
        using var cpuW = cpuBackend.CreateTensor("w", GgmlType.F32, [4]);
        using var cpuOut = cpuBackend.CreateTensor("out", GgmlType.F32, [4]);
        CopyF32(input, cpuIn);
        CopyF32(weight, cpuW);
        cpuBackend.RmsNorm(cpuOut, cpuIn, cpuW, eps);
        var expected = cpuOut.AsFloatSpan().ToArray();

        // GPU
        using var inT = LoadF32("in", input);
        using var wT = LoadF32("w", weight);
        using var outT = _backend!.CreateTensor("out", GgmlType.F32, [4]);
        _backend.RmsNorm(outT, inT, wT, eps);
        var result = DownloadF32(outT, 4);

        for (int i = 0; i < 4; i++)
            Assert.True(MathF.Abs(result[i] - expected[i]) < 1e-4f,
                $"RmsNorm[{i}]: expected {expected[i]}, got {result[i]}");
    }

    [Fact]
    public void Softmax_MatchesCpu()
    {
        if (!HasGpu) return;

        float[] input = [1.0f, 2.0f, 3.0f, 4.0f];

        // CPU reference
        using var cpuBackend = new Daisi.Llogos.Cpu.CpuBackend();
        using var cpuIn = cpuBackend.CreateTensor("in", GgmlType.F32, [4]);
        using var cpuOut = cpuBackend.CreateTensor("out", GgmlType.F32, [4]);
        CopyF32(input, cpuIn);
        cpuBackend.Softmax(cpuOut, cpuIn);
        var expected = cpuOut.AsFloatSpan().ToArray();

        // GPU
        using var inT = LoadF32("in", input);
        using var outT = _backend!.CreateTensor("out", GgmlType.F32, [4]);
        _backend.Softmax(outT, inT);
        var result = DownloadF32(outT, 4);

        for (int i = 0; i < 4; i++)
            Assert.True(MathF.Abs(result[i] - expected[i]) < 1e-5f,
                $"Softmax[{i}]: expected {expected[i]}, got {result[i]}");
    }

    [Fact]
    public void MatMul_FP32_MatchesCpu()
    {
        if (!HasGpu) return;

        // a = [1,2,3,4] (1×4), b = [[1,2,3,4],[5,6,7,8]] (2×4), out = [1×2]
        float[] a = [1, 2, 3, 4];
        float[] b = [1, 2, 3, 4, 5, 6, 7, 8]; // row 0 = [1,2,3,4], row 1 = [5,6,7,8]
        // out[0] = 1*1+2*2+3*3+4*4 = 30
        // out[1] = 1*5+2*6+3*7+4*8 = 70

        using var aT = LoadF32("a", a);
        using var bT = LoadF32("b", b);
        using var outT = _backend!.CreateTensor("out", GgmlType.F32, [2]);

        _backend.MatMul(outT, aT, bT, 1, 4, 2);

        var result = DownloadF32(outT, 2);
        Assert.True(MathF.Abs(result[0] - 30.0f) < 1e-3f, $"MatMul[0]: expected 30, got {result[0]}");
        Assert.True(MathF.Abs(result[1] - 70.0f) < 1e-3f, $"MatMul[1]: expected 70, got {result[1]}");
    }

    [Fact]
    public void MatMul_FP32_LargerMatrix()
    {
        if (!HasGpu) return;

        int K = 128, N = 64;
        var a = new float[K];
        var b = new float[N * K];
        var rng = new Random(42);
        for (int i = 0; i < K; i++) a[i] = (float)(rng.NextDouble() - 0.5);
        for (int i = 0; i < N * K; i++) b[i] = (float)(rng.NextDouble() - 0.5);

        // CPU reference
        var cpuExpected = new float[N];
        for (int n = 0; n < N; n++)
        {
            float sum = 0;
            for (int k = 0; k < K; k++)
                sum += a[k] * b[n * K + k];
            cpuExpected[n] = sum;
        }

        using var aT = LoadF32("a", a);
        using var bT = LoadF32("b", b);
        using var outT = _backend!.CreateTensor("out", GgmlType.F32, [N]);
        _backend.MatMul(outT, aT, bT, 1, K, N);

        var result = DownloadF32(outT, N);
        for (int i = 0; i < N; i++)
            Assert.True(MathF.Abs(result[i] - cpuExpected[i]) < 1e-2f,
                $"MatMul[{i}]: expected {cpuExpected[i]}, got {result[i]}");
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

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

    public void Dispose() => _backend?.Dispose();
}
