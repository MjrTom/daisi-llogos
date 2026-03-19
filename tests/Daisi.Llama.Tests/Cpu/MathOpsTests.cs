using System.Runtime.InteropServices;
using Daisi.Llama.Cpu;
using Daisi.Llama.Gguf;

namespace Daisi.Llama.Tests.Cpu;

public class MatMulTests
{
    [Fact]
    public void FP32_SmallKnown()
    {
        // A = [[1, 2, 3], [4, 5, 6]]  (M=2, K=3)
        // B stored as [N×K] = [[7, 9, 11], [8, 10, 12]]  (N=2 output rows, K=3 each)
        // This is B^T of the standard math [[7,8],[9,10],[11,12]]
        // C = A × B_math = [[58, 64], [139, 154]]
        using var backend = new CpuBackend();
        using var a = LoadF32(backend, "a", [2, 3], [1, 2, 3, 4, 5, 6]);
        using var b = LoadF32(backend, "b", [3, 2], [7, 9, 11, 8, 10, 12]);
        using var c = backend.CreateTensor("c", GgmlType.F32, [2, 2]);

        backend.MatMul(c, a, b, 2, 3, 2);

        var result = GetFloats(c);
        Assert.Equal(58f, result[0], 0.01f);
        Assert.Equal(64f, result[1], 0.01f);
        Assert.Equal(139f, result[2], 0.01f);
        Assert.Equal(154f, result[3], 0.01f);
    }

    [Fact]
    public void FP32_Identity()
    {
        // Multiplying by identity should return the input
        using var backend = new CpuBackend();
        float[] input = [1, 2, 3, 4, 5, 6, 7, 8, 9];
        float[] identity = [1, 0, 0, 0, 1, 0, 0, 0, 1];
        using var a = LoadF32(backend, "a", [3, 3], input);
        using var b = LoadF32(backend, "b", [3, 3], identity);
        using var c = backend.CreateTensor("c", GgmlType.F32, [3, 3]);

        backend.MatMul(c, a, b, 3, 3, 3);

        var result = GetFloats(c);
        for (int i = 0; i < 9; i++)
            Assert.Equal(input[i], result[i], 0.01f);
    }

    [Fact]
    public void Q8_0_MatchesDequantThenMatMul()
    {
        using var backend = new CpuBackend();
        int M = 2, K = 32, N = 3; // K must be multiple of 32 for Q8_0

        // Create FP32 input [M×K]
        var rng = new Random(42);
        var aData = new float[M * K];
        for (int i = 0; i < aData.Length; i++) aData[i] = (float)(rng.NextDouble() * 2 - 1);
        using var a = LoadF32(backend, "a", [M, K], aData);

        // Create Q8_0 weights [N×K] — N rows of K quantized elements
        // Each row is one Q8_0 block (K=32)
        var bQ8Data = new byte[N * 34]; // N blocks of 34 bytes
        for (int n = 0; n < N; n++)
        {
            BitConverter.TryWriteBytes(bQ8Data.AsSpan(n * 34), (Half)0.5f);
            for (int i = 0; i < 32; i++)
                bQ8Data[n * 34 + 2 + i] = (byte)(sbyte)(rng.Next(-128, 128));
        }
        using var bQ8 = backend.LoadTensor("bq8", GgmlType.Q8_0, [N, K], bQ8Data);

        // Fused matmul
        using var fusedOut = backend.CreateTensor("fused", GgmlType.F32, [M, N]);
        backend.MatMul(fusedOut, a, bQ8, M, K, N);

        // Reference: dequantize then FP32 matmul
        var bDequant = new float[N * K];
        for (int n = 0; n < N; n++)
        {
            Dequantize.DequantizeQ8_0(bQ8Data.AsSpan(n * 34, 34), bDequant.AsSpan(n * K, K));
        }

        // Manual dot products for reference
        var expected = new float[M * N];
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++)
            {
                float dot = 0;
                for (int k = 0; k < K; k++)
                    dot += aData[i * K + k] * bDequant[j * K + k];
                expected[i * N + j] = dot;
            }

        var result = GetFloats(fusedOut);
        for (int i = 0; i < expected.Length; i++)
            Assert.Equal(expected[i], result[i], 0.1f); // Q8_0 has some precision loss
    }

    [Fact]
    public void FP32_VectorDotProduct()
    {
        // 1×4 × 4×1 = scalar
        using var backend = new CpuBackend();
        using var a = LoadF32(backend, "a", [1, 4], [1, 2, 3, 4]);
        using var b = LoadF32(backend, "b", [4, 1], [5, 6, 7, 8]);
        using var c = backend.CreateTensor("c", GgmlType.F32, [1, 1]);

        backend.MatMul(c, a, b, 1, 4, 1);

        var result = GetFloats(c);
        Assert.Equal(70f, result[0], 0.01f); // 5+12+21+32
    }

    private static ITensor LoadF32(CpuBackend backend, string name, long[] dims, float[] data)
    {
        var bytes = MemoryMarshal.AsBytes(data.AsSpan());
        return backend.LoadTensor(name, GgmlType.F32, dims, bytes);
    }

    private static float[] GetFloats(ITensor tensor)
    {
        var result = new float[tensor.ElementCount];
        tensor.DequantizeTo(result);
        return result;
    }
}

public class RmsNormTests
{
    [Fact]
    public void KnownValues()
    {
        using var backend = new CpuBackend();
        float[] inputData = [1, 2, 3, 4];
        float[] weightData = [1, 1, 1, 1]; // unit weights

        using var input = LoadF32(backend, "in", [4], inputData);
        using var weight = LoadF32(backend, "w", [4], weightData);
        using var output = backend.CreateTensor("out", GgmlType.F32, [4]);

        backend.RmsNorm(output, input, weight, 1e-6f);

        // rms = sqrt((1+4+9+16)/4 + 1e-6) = sqrt(7.5) ≈ 2.7386
        float rms = MathF.Sqrt(30f / 4f + 1e-6f);
        var result = GetFloats(output);
        for (int i = 0; i < 4; i++)
            Assert.Equal(inputData[i] / rms, result[i], 0.001f);
    }

    [Fact]
    public void WithWeight()
    {
        using var backend = new CpuBackend();
        float[] inputData = [2, 2, 2, 2];
        float[] weightData = [0.5f, 1.0f, 1.5f, 2.0f];

        using var input = LoadF32(backend, "in", [4], inputData);
        using var weight = LoadF32(backend, "w", [4], weightData);
        using var output = backend.CreateTensor("out", GgmlType.F32, [4]);

        backend.RmsNorm(output, input, weight, 1e-6f);

        // rms = sqrt(16/4 + eps) = 2.0, normalized = [1,1,1,1]
        // output = [0.5, 1.0, 1.5, 2.0]
        var result = GetFloats(output);
        Assert.Equal(0.5f, result[0], 0.001f);
        Assert.Equal(1.0f, result[1], 0.001f);
        Assert.Equal(1.5f, result[2], 0.001f);
        Assert.Equal(2.0f, result[3], 0.001f);
    }

    private static ITensor LoadF32(CpuBackend backend, string name, long[] dims, float[] data)
    {
        return backend.LoadTensor(name, GgmlType.F32, dims, MemoryMarshal.AsBytes(data.AsSpan()));
    }

    private static float[] GetFloats(ITensor tensor)
    {
        var r = new float[tensor.ElementCount];
        tensor.DequantizeTo(r);
        return r;
    }
}

public class SoftmaxTests
{
    [Fact]
    public void UniformInput()
    {
        using var backend = new CpuBackend();
        using var input = LoadF32(backend, "in", [4], [1, 1, 1, 1]);
        using var output = backend.CreateTensor("out", GgmlType.F32, [4]);

        backend.Softmax(output, input);

        var result = GetFloats(output);
        for (int i = 0; i < 4; i++)
            Assert.Equal(0.25f, result[i], 0.001f);
    }

    [Fact]
    public void SumsToOne()
    {
        using var backend = new CpuBackend();
        using var input = LoadF32(backend, "in", [5], [1.0f, 2.0f, 3.0f, 4.0f, 5.0f]);
        using var output = backend.CreateTensor("out", GgmlType.F32, [5]);

        backend.Softmax(output, input);

        var result = GetFloats(output);
        float sum = result.Sum();
        Assert.Equal(1.0f, sum, 0.001f);
    }

    [Fact]
    public void LargeValues_NoOverflow()
    {
        using var backend = new CpuBackend();
        using var input = LoadF32(backend, "in", [3], [100f, 200f, 300f]);
        using var output = backend.CreateTensor("out", GgmlType.F32, [3]);

        backend.Softmax(output, input);

        var result = GetFloats(output);
        // Should not have NaN or Inf
        Assert.All(result, v => Assert.False(float.IsNaN(v) || float.IsInfinity(v)));
        // Largest input should have highest probability
        Assert.Equal(1.0f, result[2], 0.001f); // exp(300-300)=1 dominates
    }

    [Fact]
    public void MonotonicallyIncreasing()
    {
        using var backend = new CpuBackend();
        using var input = LoadF32(backend, "in", [4], [1f, 2f, 3f, 4f]);
        using var output = backend.CreateTensor("out", GgmlType.F32, [4]);

        backend.Softmax(output, input);

        var result = GetFloats(output);
        Assert.True(result[0] < result[1]);
        Assert.True(result[1] < result[2]);
        Assert.True(result[2] < result[3]);
    }

    private static ITensor LoadF32(CpuBackend backend, string name, long[] dims, float[] data)
    {
        return backend.LoadTensor(name, GgmlType.F32, dims, MemoryMarshal.AsBytes(data.AsSpan()));
    }

    private static float[] GetFloats(ITensor tensor)
    {
        var r = new float[tensor.ElementCount];
        tensor.DequantizeTo(r);
        return r;
    }
}

public class SiluTests
{
    [Fact]
    public void Zero()
    {
        using var backend = new CpuBackend();
        using var input = LoadF32(backend, "in", [1], [0f]);
        using var output = backend.CreateTensor("out", GgmlType.F32, [1]);

        backend.SiLU(output, input);

        var result = GetFloats(output);
        Assert.Equal(0f, result[0], 0.001f);
    }

    [Fact]
    public void KnownValues()
    {
        using var backend = new CpuBackend();
        float[] values = [-2f, -1f, 0f, 1f, 2f];
        using var input = LoadF32(backend, "in", [5], values);
        using var output = backend.CreateTensor("out", GgmlType.F32, [5]);

        backend.SiLU(output, input);

        var result = GetFloats(output);
        for (int i = 0; i < values.Length; i++)
        {
            float x = values[i];
            float expected = x / (1f + MathF.Exp(-x));
            Assert.Equal(expected, result[i], 0.001f);
        }
    }

    [Fact]
    public void LargePositive_ApproachesIdentity()
    {
        using var backend = new CpuBackend();
        using var input = LoadF32(backend, "in", [1], [10f]);
        using var output = backend.CreateTensor("out", GgmlType.F32, [1]);

        backend.SiLU(output, input);

        var result = GetFloats(output);
        // sigmoid(10) ≈ 0.99995, so SiLU(10) ≈ 10
        Assert.Equal(10f, result[0], 0.01f);
    }

    private static ITensor LoadF32(CpuBackend backend, string name, long[] dims, float[] data)
    {
        return backend.LoadTensor(name, GgmlType.F32, dims, MemoryMarshal.AsBytes(data.AsSpan()));
    }

    private static float[] GetFloats(ITensor tensor)
    {
        var r = new float[tensor.ElementCount];
        tensor.DequantizeTo(r);
        return r;
    }
}

public class RopeTests
{
    [Fact]
    public void Position0_NoRotation()
    {
        // At position 0, cos(0)=1, sin(0)=0, so no rotation
        using var backend = new CpuBackend();
        float[] qData = [1, 2, 3, 4]; // 1 head, headDim=4
        float[] kData = [5, 6, 7, 8];
        using var q = LoadF32(backend, "q", [4], qData);
        using var k = LoadF32(backend, "k", [4], kData);

        backend.RoPE(q, k, 4, 0, 0, 10000f);

        var qResult = GetFloats(q);
        var kResult = GetFloats(k);
        for (int i = 0; i < 4; i++)
        {
            Assert.Equal(qData[i], qResult[i], 0.001f);
            Assert.Equal(kData[i], kResult[i], 0.001f);
        }
    }

    [Fact]
    public void PreservesNorm()
    {
        // Rotation preserves vector magnitude
        using var backend = new CpuBackend();
        float[] qData = [1, 2, 3, 4];
        float[] kData = [5, 6, 7, 8];
        float qNormBefore = MathF.Sqrt(qData.Select(x => x * x).Sum());
        float kNormBefore = MathF.Sqrt(kData.Select(x => x * x).Sum());

        using var q = LoadF32(backend, "q", [4], qData);
        using var k = LoadF32(backend, "k", [4], kData);

        backend.RoPE(q, k, 4, 0, 42, 10000f);

        var qResult = GetFloats(q);
        var kResult = GetFloats(k);
        float qNormAfter = MathF.Sqrt(qResult.Select(x => x * x).Sum());
        float kNormAfter = MathF.Sqrt(kResult.Select(x => x * x).Sum());

        Assert.Equal(qNormBefore, qNormAfter, 0.001f);
        Assert.Equal(kNormBefore, kNormAfter, 0.001f);
    }

    [Fact]
    public void MultipleHeads()
    {
        // 2 query heads, 1 KV head, headDim=4
        using var backend = new CpuBackend();
        float[] qData = [1, 0, 0, 0, 0, 1, 0, 0]; // 2 heads × 4 dim
        float[] kData = [1, 0, 0, 0]; // 1 head × 4 dim

        using var q = LoadF32(backend, "q", [8], qData);
        using var k = LoadF32(backend, "k", [4], kData);

        backend.RoPE(q, k, 4, 0, 5, 10000f);

        var qResult = GetFloats(q);
        // Both heads should be rotated independently
        // Head 0: [1,0,0,0] → [cos(5*θ₀), sin(5*θ₀), cos(5*θ₁), sin(5*θ₁)]
        // Verify non-trivial rotation happened
        Assert.NotEqual(1f, qResult[0], 0.01f); // Should be rotated
    }

    private static ITensor LoadF32(CpuBackend backend, string name, long[] dims, float[] data)
    {
        return backend.LoadTensor(name, GgmlType.F32, dims, MemoryMarshal.AsBytes(data.AsSpan()));
    }

    private static float[] GetFloats(ITensor tensor)
    {
        var r = new float[tensor.ElementCount];
        tensor.DequantizeTo(r);
        return r;
    }
}

public class ElementOpsTests
{
    [Fact]
    public void Multiply_KnownValues()
    {
        using var backend = new CpuBackend();
        using var a = LoadF32(backend, "a", [4], [1, 2, 3, 4]);
        using var b = LoadF32(backend, "b", [4], [5, 6, 7, 8]);
        using var c = backend.CreateTensor("c", GgmlType.F32, [4]);

        backend.ElementMul(c, a, b);

        var result = GetFloats(c);
        Assert.Equal(5f, result[0], 0.01f);
        Assert.Equal(12f, result[1], 0.01f);
        Assert.Equal(21f, result[2], 0.01f);
        Assert.Equal(32f, result[3], 0.01f);
    }

    [Fact]
    public void Add_KnownValues()
    {
        using var backend = new CpuBackend();
        using var a = LoadF32(backend, "a", [4], [1, 2, 3, 4]);
        using var b = LoadF32(backend, "b", [4], [10, 20, 30, 40]);
        using var c = backend.CreateTensor("c", GgmlType.F32, [4]);

        backend.ElementAdd(c, a, b);

        var result = GetFloats(c);
        Assert.Equal(11f, result[0], 0.01f);
        Assert.Equal(22f, result[1], 0.01f);
        Assert.Equal(33f, result[2], 0.01f);
        Assert.Equal(44f, result[3], 0.01f);
    }

    [Fact]
    public void Multiply_LargeArray_SimdPath()
    {
        // Test array large enough to exercise SIMD path (>= 8 elements)
        using var backend = new CpuBackend();
        int n = 32;
        float[] aData = new float[n], bData = new float[n];
        for (int i = 0; i < n; i++) { aData[i] = i; bData[i] = i + 1; }

        using var a = LoadF32(backend, "a", [n], aData);
        using var b = LoadF32(backend, "b", [n], bData);
        using var c = backend.CreateTensor("c", GgmlType.F32, [n]);

        backend.ElementMul(c, a, b);

        var result = GetFloats(c);
        for (int i = 0; i < n; i++)
            Assert.Equal(aData[i] * bData[i], result[i], 0.01f);
    }

    private static ITensor LoadF32(CpuBackend backend, string name, long[] dims, float[] data)
    {
        return backend.LoadTensor(name, GgmlType.F32, dims, MemoryMarshal.AsBytes(data.AsSpan()));
    }

    private static float[] GetFloats(ITensor tensor)
    {
        var r = new float[tensor.ElementCount];
        tensor.DequantizeTo(r);
        return r;
    }
}
