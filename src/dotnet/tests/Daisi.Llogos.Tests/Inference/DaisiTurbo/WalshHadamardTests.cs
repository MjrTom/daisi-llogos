using Daisi.Llogos.Inference.DaisiTurbo;

namespace Daisi.Llogos.Tests.Inference.DaisiTurbo;

public class WalshHadamardTests
{
    [Theory]
    [InlineData(8)]
    [InlineData(16)]
    [InlineData(32)]
    [InlineData(64)]
    [InlineData(128)]
    public void Forward_Inverse_RoundTrips(int dim)
    {
        var signs = WalshHadamard.GenerateSigns(dim, seed: 42);
        var original = new float[dim];
        var rng = new Random(123);
        for (int i = 0; i < dim; i++)
            original[i] = (float)(rng.NextDouble() * 2 - 1);

        var data = (float[])original.Clone();

        WalshHadamard.ForwardInPlace(data, signs);
        WalshHadamard.InverseInPlace(data, signs);

        for (int i = 0; i < dim; i++)
            Assert.Equal(original[i], data[i], tolerance: 1e-4f);
    }

    [Theory]
    [InlineData(8)]
    [InlineData(64)]
    public void Forward_PreservesEnergy(int dim)
    {
        // WHT is orthonormal — ||H·x||² = ||x||²
        var signs = WalshHadamard.GenerateSigns(dim, seed: 99);
        var data = new float[dim];
        var rng = new Random(456);
        for (int i = 0; i < dim; i++)
            data[i] = (float)(rng.NextDouble() * 4 - 2);

        float originalEnergy = data.Sum(x => x * x);

        WalshHadamard.ForwardInPlace(data, signs);

        float transformedEnergy = data.Sum(x => x * x);

        Assert.Equal(originalEnergy, transformedEnergy, tolerance: 1e-3f);
    }

    [Fact]
    public void Forward_SpreadsOutliers()
    {
        // A vector with one large outlier should become more uniform after WHT
        int dim = 64;
        var signs = WalshHadamard.GenerateSigns(dim, seed: 42);
        var data = new float[dim];
        data[0] = 100.0f; // single outlier

        WalshHadamard.ForwardInPlace(data, signs);

        // After WHT, all values should be similar magnitude (no single outlier)
        float maxAbs = data.Max(MathF.Abs);
        float minAbs = data.Min(MathF.Abs);
        // Max/min ratio should be small (all values are ±100/sqrt(64) = ±12.5)
        Assert.True(maxAbs < 20.0f, $"Max value {maxAbs} too large — WHT didn't spread outlier");
        Assert.True(minAbs > 5.0f, $"Min absolute value {minAbs} too small — energy not spread evenly");
    }

    [Fact]
    public void GenerateSigns_Deterministic()
    {
        var signs1 = WalshHadamard.GenerateSigns(64, seed: 42);
        var signs2 = WalshHadamard.GenerateSigns(64, seed: 42);

        for (int i = 0; i < 64; i++)
            Assert.Equal(signs1[i], signs2[i]);
    }

    [Fact]
    public void GenerateSigns_AllPlusMinusOne()
    {
        var signs = WalshHadamard.GenerateSigns(64, seed: 42);
        foreach (var s in signs)
            Assert.True(s == 1.0f || s == -1.0f, $"Sign value {s} is not ±1");
    }
}
