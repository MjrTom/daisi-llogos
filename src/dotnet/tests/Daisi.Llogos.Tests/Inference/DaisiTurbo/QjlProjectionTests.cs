using Daisi.Llogos.Inference.DaisiTurbo;

namespace Daisi.Llogos.Tests.Inference.DaisiTurbo;

public class QjlProjectionTests
{
    [Theory]
    [InlineData(32)]
    [InlineData(64)]
    public void Correction_PreservesSign_OverManyTrials(int dim)
    {
        // The QJL sign-bit correction should preserve the sign of the true dot product
        // more often than not (i.e., if true dot is positive, correction should usually be positive).
        var rng = new Random(123);
        int correctSign = 0;
        int trials = 200;

        for (int t = 0; t < trials; t++)
        {
            var residual = RandomVector(rng, dim, 1.0f);
            var query = RandomVector(rng, dim, 1.0f);

            float trueDot = 0;
            for (int d = 0; d < dim; d++)
                trueDot += query[d] * residual[d];

            if (MathF.Abs(trueDot) < 0.1f) continue; // skip near-zero

            var qjl = new QjlProjection(dim, dim, seed: t);
            var signBits = new byte[qjl.SignBitBytes];
            qjl.ProjectAndSign(residual, signBits);
            float correction = qjl.ComputeCorrection(query, signBits);

            if (MathF.Sign(correction) == MathF.Sign(trueDot))
                correctSign++;
        }

        // Should get the sign right more than chance (50%)
        double accuracy = (double)correctSign / trials;
        Assert.True(accuracy > 0.55, $"QJL sign accuracy {accuracy:P1} should be better than chance");
    }

    [Fact]
    public void BatchCorrection_MatchesSingleCorrection()
    {
        int dim = 64;
        int projDim = 32;
        var qjl = new QjlProjection(dim, projDim, seed: 42);
        var rng = new Random(456);

        int count = 10;
        var query = RandomVector(rng, dim, 1.0f);

        // Generate sign bits for multiple positions
        int stride = qjl.SignBitBytes;
        var allSignBits = new byte[count * stride];
        var singleCorrections = new float[count];

        for (int i = 0; i < count; i++)
        {
            var residual = RandomVector(rng, dim, 0.3f);
            var signBits = allSignBits.AsSpan(i * stride, stride);
            qjl.ProjectAndSign(residual, signBits);

            singleCorrections[i] = qjl.ComputeCorrection(query, signBits);
        }

        // Batch compute
        var batchCorrections = new float[count];
        qjl.BatchCorrection(query, allSignBits, stride, count, batchCorrections);

        for (int i = 0; i < count; i++)
            Assert.Equal(singleCorrections[i], batchCorrections[i], tolerance: 1e-5f);
    }

    [Fact]
    public void SignBitBytes_CorrectSize()
    {
        var qjl = new QjlProjection(64, 32, seed: 42);
        Assert.Equal(4, qjl.SignBitBytes); // 32 bits = 4 bytes

        var qjl2 = new QjlProjection(64, 64, seed: 42);
        Assert.Equal(8, qjl2.SignBitBytes); // 64 bits = 8 bytes

        var qjl3 = new QjlProjection(64, 7, seed: 42);
        Assert.Equal(1, qjl3.SignBitBytes); // ceil(7/8) = 1 byte
    }

    [Fact]
    public void MoreProjectionDims_ReducesVariance()
    {
        int dim = 64;
        var rng = new Random(789);

        var residual = RandomVector(rng, dim, 1.0f);
        var query = RandomVector(rng, dim, 1.0f);

        // True dot product
        float trueDot = 0;
        for (int d = 0; d < dim; d++)
            trueDot += query[d] * residual[d];

        // Measure variance with different projection dims
        double var16 = MeasureVariance(dim, 16, residual, query, trueDot, 200);
        double var64 = MeasureVariance(dim, 64, residual, query, trueDot, 200);

        Assert.True(var64 < var16, $"More projections should reduce variance: var16={var16:F4}, var64={var64:F4}");
    }

    private static double MeasureVariance(int dim, int projDim, float[] residual, float[] query, float trueDot, int trials)
    {
        double sumSqErr = 0;
        for (int seed = 0; seed < trials; seed++)
        {
            var qjl = new QjlProjection(dim, projDim, seed);
            var signBits = new byte[qjl.SignBitBytes];
            qjl.ProjectAndSign(residual, signBits);
            float est = qjl.ComputeCorrection(query, signBits);
            double err = est - trueDot;
            sumSqErr += err * err;
        }
        return sumSqErr / trials;
    }

    [Theory]
    [InlineData(32)]
    [InlineData(64)]
    public void NormAwareCorrection_ReducesDotProductError(int dim)
    {
        // The norm-aware QJL correction should reduce the error in Q·K estimation
        // compared to using only the reconstructed (quantized) K.
        var rng = new Random(42);
        int projDim = dim;
        var qjl = new QjlProjection(dim, projDim, seed: 99);

        double totalErrorNoQjl = 0;
        double totalErrorWithQjl = 0;
        int trials = 500;

        for (int t = 0; t < trials; t++)
        {
            // Simulate: original K, quantized K with some error, query Q
            var originalK = RandomVector(rng, dim, 1.0f);
            var quantError = RandomVector(rng, dim, 0.3f); // quantization residual
            var reconstructedK = new float[dim];
            for (int d = 0; d < dim; d++)
                reconstructedK[d] = originalK[d] - quantError[d];
            var query = RandomVector(rng, dim, 1.0f);

            // True dot product
            float trueDot = 0;
            for (int d = 0; d < dim; d++)
                trueDot += query[d] * originalK[d];

            // Reconstructed dot product (no QJL)
            float reconDot = 0;
            for (int d = 0; d < dim; d++)
                reconDot += query[d] * reconstructedK[d];

            // QJL-corrected dot product
            var signBits = new byte[qjl.SignBitBytes];
            qjl.ProjectAndSign(quantError, signBits, out float resNorm);
            float correction = qjl.ComputeCorrection(query, signBits, resNorm);
            float correctedDot = reconDot + correction;

            totalErrorNoQjl += (trueDot - reconDot) * (trueDot - reconDot);
            totalErrorWithQjl += (trueDot - correctedDot) * (trueDot - correctedDot);
        }

        double mseNoQjl = totalErrorNoQjl / trials;
        double mseWithQjl = totalErrorWithQjl / trials;

        // QJL correction should reduce MSE
        Assert.True(mseWithQjl < mseNoQjl,
            $"QJL should reduce dot product error: without={mseNoQjl:F4}, with={mseWithQjl:F4}");
    }

    private static float[] RandomVector(Random rng, int dim, float scale)
    {
        var v = new float[dim];
        for (int i = 0; i < dim; i++)
            v[i] = (float)(rng.NextDouble() * 2 - 1) * scale;
        return v;
    }
}
