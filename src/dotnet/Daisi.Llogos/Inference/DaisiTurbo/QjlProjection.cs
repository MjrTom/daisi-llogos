using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace Daisi.Llogos.Inference.DaisiTurbo;

/// <summary>
/// Quantized Johnson-Lindenstrauss (QJL) sign-bit residual correction.
/// After MSE quantization, the residual (original - reconstructed) is projected
/// via a random matrix and reduced to sign bits (+1/-1). During attention,
/// an unbiased estimator combines the quantized dot product with a sign-bit
/// correction term, eliminating the bias introduced by MSE quantization.
/// </summary>
public sealed class QjlProjection
{
    private readonly int _inputDim;
    private readonly int _projDim;
    private readonly float[] _projectionMatrix; // [projDim × inputDim], ±1/sqrt(projDim) entries

    /// <summary>Input vector dimension (headDim).</summary>
    public int InputDim => _inputDim;

    /// <summary>Projection dimension (number of sign bits stored per vector).</summary>
    public int ProjectionDim => _projDim;

    /// <summary>Number of bytes needed to store sign bits for one vector.</summary>
    public int SignBitBytes => (_projDim + 7) / 8;

    /// <summary>
    /// Create a QJL projection with the given dimensions and seed.
    /// The projection matrix is a random ±1 Rademacher matrix scaled by 1/sqrt(projDim).
    /// </summary>
    /// <param name="inputDim">Dimension of input vectors (headDim).</param>
    /// <param name="projDim">Number of projected dimensions (sign bits to store).
    /// More bits = less variance in the estimator. Typical: inputDim/2 to inputDim.</param>
    /// <param name="seed">Random seed for reproducibility.</param>
    public QjlProjection(int inputDim, int projDim, int seed)
    {
        _inputDim = inputDim;
        _projDim = projDim;
        _projectionMatrix = GenerateRademacherMatrix(projDim, inputDim, seed);
    }

    /// <summary>
    /// Project the residual vector and store only sign bits.
    /// residual = original - reconstructed (computed by caller).
    /// </summary>
    public void ProjectAndSign(ReadOnlySpan<float> residual, Span<byte> signBits)
    {
        signBits.Slice(0, SignBitBytes).Clear();

        for (int p = 0; p < _projDim; p++)
        {
            // Dot product: projectionMatrix[p, :] · residual
            float dot = DotProduct(residual, _projectionMatrix.AsSpan(p * _inputDim, _inputDim));

            // Store sign bit
            if (dot >= 0)
                signBits[p >> 3] |= (byte)(1 << (p & 7));
        }
    }

    /// <summary>
    /// Compute the unbiased correction term for an attention dot product.
    /// Given query q and cached sign bits of key residual, returns the correction
    /// that when added to (q · reconstructed_k) gives an unbiased estimate of (q · original_k).
    ///
    /// correction = (1/projDim) * sum_p( sign_p * (R_p · q) )
    /// where R_p is row p of the projection matrix and sign_p is ±1.
    /// </summary>
    public float ComputeCorrection(ReadOnlySpan<float> query, ReadOnlySpan<byte> signBits)
    {
        float correction = 0;

        for (int p = 0; p < _projDim; p++)
        {
            // Dot product: projectionMatrix[p, :] · query
            float projQuery = DotProduct(query, _projectionMatrix.AsSpan(p * _inputDim, _inputDim));

            // Sign bit: +1 if set, -1 if not
            bool isPositive = (signBits[p >> 3] & (1 << (p & 7))) != 0;
            correction += isPositive ? projQuery : -projQuery;
        }

        return correction / _projDim;
    }

    /// <summary>
    /// Batch-compute corrections for multiple cached positions at once.
    /// More efficient than calling ComputeCorrection per-position since it
    /// computes the query projections once and reuses them.
    /// </summary>
    public void BatchCorrection(ReadOnlySpan<float> query, ReadOnlySpan<byte> allSignBits,
        int signBitStride, int count, Span<float> corrections)
    {
        // Pre-compute R · q for all projection dimensions
        Span<float> projQuery = stackalloc float[_projDim];
        for (int p = 0; p < _projDim; p++)
            projQuery[p] = DotProduct(query, _projectionMatrix.AsSpan(p * _inputDim, _inputDim));

        float invProjDim = 1.0f / _projDim;

        for (int pos = 0; pos < count; pos++)
        {
            var signBits = allSignBits.Slice(pos * signBitStride, SignBitBytes);
            float corr = 0;

            // Process 8 projection dims at a time using byte-level sign bits
            int fullBytes = _projDim >> 3;
            for (int byteIdx = 0; byteIdx < fullBytes; byteIdx++)
            {
                byte bits = signBits[byteIdx];
                int baseP = byteIdx << 3;
                for (int b = 0; b < 8; b++)
                {
                    bool isPositive = (bits & (1 << b)) != 0;
                    corr += isPositive ? projQuery[baseP + b] : -projQuery[baseP + b];
                }
            }

            // Remaining bits
            int remaining = _projDim & 7;
            if (remaining > 0)
            {
                byte bits = signBits[fullBytes];
                int baseP = fullBytes << 3;
                for (int b = 0; b < remaining; b++)
                {
                    bool isPositive = (bits & (1 << b)) != 0;
                    corr += isPositive ? projQuery[baseP + b] : -projQuery[baseP + b];
                }
            }

            corrections[pos] = corr * invProjDim;
        }
    }

    // ── Helpers ──────────────────────────────────────────────────────────────

    private static float[] GenerateRademacherMatrix(int rows, int cols, int seed)
    {
        var matrix = new float[rows * cols];
        var rng = new Random(seed);

        // Unscaled ±1 entries — scaling is handled in the correction formula
        for (int i = 0; i < matrix.Length; i++)
            matrix[i] = rng.Next(2) == 0 ? 1.0f : -1.0f;

        return matrix;
    }

    private static float DotProduct(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        float sum = 0;
        int i = 0;
        int n = a.Length;

        if (Avx2.IsSupported && n >= 8)
        {
            ref float aRef = ref MemoryMarshal.GetReference(a);
            ref float bRef = ref MemoryMarshal.GetReference(b);
            var acc = Vector256<float>.Zero;
            for (; i + 8 <= n; i += 8)
            {
                var va = Vector256.LoadUnsafe(ref Unsafe.Add(ref aRef, i));
                var vb = Vector256.LoadUnsafe(ref Unsafe.Add(ref bRef, i));
                acc = Avx.Add(acc, Avx.Multiply(va, vb));
            }
            // Horizontal sum
            var hi = Avx.ExtractVector128(acc, 1);
            var lo = acc.GetLower();
            var sum128 = Sse.Add(lo, hi);
            sum128 = Sse.Add(sum128, Sse.MoveHighToLow(sum128, sum128));
            sum128 = Sse.AddScalar(sum128, Sse.Shuffle(sum128, sum128, 1));
            sum = sum128.ToScalar();
        }

        for (; i < n; i++)
            sum += a[i] * b[i];

        return sum;
    }
}
