using Daisi.Llogos.Inference.DaisiTurbo;

namespace Daisi.Llogos.Tests.Inference.DaisiTurbo;

public class ScalarQuantizerTests
{
    [Theory]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)]
    public void QuantizeDequantize_RoundTrips_WithinExpectedError(int bits)
    {
        var q = ScalarQuantizer.Create(bits);
        int dim = 64;
        var rng = new Random(42);

        var input = new float[dim];
        for (int i = 0; i < dim; i++)
            input[i] = (float)(rng.NextDouble() * 4 - 2); // N(0,1)-ish range

        var packed = new byte[q.PackedBytes(dim)];
        var reconstructed = new float[dim];
        q.QuantizeVector(input, packed, reconstructed);

        // Dequantize from packed
        var output = new float[dim];
        q.DequantizeVector(packed, output);

        // Packed round-trip should match reconstructed exactly
        for (int i = 0; i < dim; i++)
            Assert.Equal(reconstructed[i], output[i], tolerance: 1e-6f);
    }

    [Theory]
    [InlineData(2, 1.0f)]  // 2-bit: ~1.0 MSE for unit variance
    [InlineData(3, 0.5f)]  // 3-bit: ~0.5 MSE
    [InlineData(4, 0.15f)] // 4-bit: ~0.15 MSE
    public void Quantization_MSE_WithinBounds(int bits, float maxMse)
    {
        var q = ScalarQuantizer.Create(bits);
        int dim = 1024;
        var rng = new Random(42);

        var input = new float[dim];
        for (int i = 0; i < dim; i++)
            input[i] = (float)(rng.NextDouble() * 4 - 2);

        var packed = new byte[q.PackedBytes(dim)];
        var reconstructed = new float[dim];
        q.QuantizeVector(input, packed, reconstructed);

        float mse = 0;
        for (int i = 0; i < dim; i++)
        {
            float err = input[i] - reconstructed[i];
            mse += err * err;
        }
        mse /= dim;

        Assert.True(mse < maxMse, $"{bits}-bit MSE {mse:F4} exceeds bound {maxMse}");
    }

    [Theory]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)]
    public void PackedBytes_CorrectSize(int bits)
    {
        var q = ScalarQuantizer.Create(bits);
        int dim = 64;
        int expected = bits switch
        {
            2 => 16,   // 64 / 4
            3 => 24,   // ceil(64 * 3 / 8)
            4 => 32,   // 64 / 2
            _ => throw new Exception()
        };
        Assert.Equal(expected, q.PackedBytes(dim));
    }

    [Fact]
    public void Quantize_SymmetricAroundZero()
    {
        var q = ScalarQuantizer.Create(3);
        int levelPos = q.Quantize(0.5f);
        int levelNeg = q.Quantize(-0.5f);
        // Levels should be symmetric: level(x) + level(-x) = Levels - 1
        Assert.Equal(q.Levels - 1, levelPos + levelNeg);
    }

    [Theory]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)]
    public void Quantize_AllLevelsReachable(int bits)
    {
        var q = ScalarQuantizer.Create(bits);
        var reached = new HashSet<int>();
        // Sweep from -4 to +4 to hit all levels
        for (float v = -4.0f; v <= 4.0f; v += 0.01f)
            reached.Add(q.Quantize(v));
        Assert.Equal(q.Levels, reached.Count);
    }
}
