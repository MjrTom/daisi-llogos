using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace Daisi.Llogos.Inference.DaisiTurbo;

/// <summary>
/// Quantized Johnson-Lindenstrauss (QJL) sign-bit residual correction.
/// After MSE quantization, the residual (original - reconstructed) is projected
/// via a random matrix and reduced to sign bits (+1/-1). During attention,
/// a scale-corrected estimator combines the quantized dot product with a sign-bit
/// correction term that reduces the bias introduced by MSE quantization.
///
/// The estimator uses: correction = (residualNorm / m) × Σ_p sign(R_p · r) × sign(R_p · q) × |R_p · q|
/// where residualNorm is stored per-vector during compression.
/// </summary>
public sealed class QjlProjection
{
    private readonly int _inputDim;
    private readonly int _projDim;
    private readonly float[] _projectionMatrix; // [projDim × inputDim], ±1 entries

    /// <summary>Input vector dimension (headDim).</summary>
    public int InputDim => _inputDim;

    /// <summary>Projection dimension (number of sign bits stored per vector).</summary>
    public int ProjectionDim => _projDim;

    /// <summary>Bytes for sign bits + 4 bytes for residual norm.</summary>
    public int SignBitBytes => (_projDim + 7) / 8;

    /// <summary>Total bytes stored per vector: sign bits + residual norm float.</summary>
    public int StorageBytes => SignBitBytes + sizeof(float);

    /// <summary>
    /// Create a QJL projection with the given dimensions and seed.
    /// </summary>
    public QjlProjection(int inputDim, int projDim, int seed)
    {
        _inputDim = inputDim;
        _projDim = projDim;
        _projectionMatrix = GenerateRademacherMatrix(projDim, inputDim, seed);
    }

    /// <summary>
    /// Project the residual vector, store sign bits and the residual L2 norm.
    /// The norm is needed for the correction estimator's magnitude calibration.
    /// signBitsAndNorm must have length >= SignBitBytes (norm stored separately via out param).
    /// </summary>
    public void ProjectAndSign(ReadOnlySpan<float> residual, Span<byte> signBits, out float residualNorm)
    {
        signBits.Slice(0, SignBitBytes).Clear();

        // Compute and store residual L2 norm
        float normSq = 0;
        for (int d = 0; d < residual.Length; d++)
            normSq += residual[d] * residual[d];
        residualNorm = MathF.Sqrt(normSq);

        for (int p = 0; p < _projDim; p++)
        {
            float dot = DotProduct(residual, _projectionMatrix.AsSpan(p * _inputDim, _inputDim));
            if (dot >= 0)
                signBits[p >> 3] |= (byte)(1 << (p & 7));
        }
    }

    /// <summary>
    /// Backward-compatible overload that discards the residual norm.
    /// </summary>
    public void ProjectAndSign(ReadOnlySpan<float> residual, Span<byte> signBits)
    {
        ProjectAndSign(residual, signBits, out _);
    }

    /// <summary>
    /// Compute the correction term for an attention dot product score.
    /// Uses the scale-corrected sign-bit estimator:
    ///   correction = (residualNorm / (m × queryProjNorm)) × Σ_p sign_r_p × sign_q_p × |R_p · q|
    /// where sign_r_p = stored sign of R_p · residual, sign_q_p = sign of R_p · query.
    ///
    /// This estimates q · residual, correcting the quantization bias in attention scores.
    /// </summary>
    public float ComputeCorrection(ReadOnlySpan<float> query, ReadOnlySpan<byte> signBits, float residualNorm)
    {
        if (residualNorm < 1e-10f) return 0;

        float correction = 0;
        float queryProjNormSq = 0;

        for (int p = 0; p < _projDim; p++)
        {
            float projQuery = DotProduct(query, _projectionMatrix.AsSpan(p * _inputDim, _inputDim));
            float absProjQuery = MathF.Abs(projQuery);
            queryProjNormSq += projQuery * projQuery;

            bool residualPositive = (signBits[p >> 3] & (1 << (p & 7))) != 0;
            bool queryPositive = projQuery >= 0;

            // sign agreement → positive contribution, disagreement → negative
            correction += (residualPositive == queryPositive) ? absProjQuery : -absProjQuery;
        }

        // Scale by residualNorm / queryProjNorm to calibrate magnitude
        float queryProjNorm = MathF.Sqrt(queryProjNormSq);
        if (queryProjNorm < 1e-10f) return 0;

        return correction * residualNorm / (queryProjNorm * MathF.Sqrt(_projDim));
    }

    /// <summary>
    /// Backward-compatible overload without residual norm (uses unit norm).
    /// </summary>
    public float ComputeCorrection(ReadOnlySpan<float> query, ReadOnlySpan<byte> signBits)
    {
        return ComputeCorrection(query, signBits, 1.0f);
    }

    /// <summary>
    /// Batch-compute corrections for multiple cached positions at once.
    /// Pre-computes query projections once and reuses them across all positions.
    /// </summary>
    public void BatchCorrection(ReadOnlySpan<float> query, ReadOnlySpan<byte> allSignBits,
        ReadOnlySpan<float> residualNorms, int signBitStride, int count, Span<float> corrections)
    {
        // Pre-compute R · q and |R · q| for all projection dimensions
        Span<float> projQuery = stackalloc float[_projDim];
        Span<float> absProjQuery = stackalloc float[_projDim];
        float queryProjNormSq = 0;

        for (int p = 0; p < _projDim; p++)
        {
            projQuery[p] = DotProduct(query, _projectionMatrix.AsSpan(p * _inputDim, _inputDim));
            absProjQuery[p] = MathF.Abs(projQuery[p]);
            queryProjNormSq += projQuery[p] * projQuery[p];
        }

        float queryProjNorm = MathF.Sqrt(queryProjNormSq);
        float invScale = queryProjNorm > 1e-10f ? 1.0f / (queryProjNorm * MathF.Sqrt(_projDim)) : 0;

        for (int pos = 0; pos < count; pos++)
        {
            float rNorm = residualNorms[pos];
            if (rNorm < 1e-10f) { corrections[pos] = 0; continue; }

            var signBits = allSignBits.Slice(pos * signBitStride, SignBitBytes);
            float corr = 0;

            int fullBytes = _projDim >> 3;
            for (int byteIdx = 0; byteIdx < fullBytes; byteIdx++)
            {
                byte bits = signBits[byteIdx];
                int baseP = byteIdx << 3;
                for (int b = 0; b < 8; b++)
                {
                    bool residualPositive = (bits & (1 << b)) != 0;
                    bool queryPositive = projQuery[baseP + b] >= 0;
                    corr += (residualPositive == queryPositive)
                        ? absProjQuery[baseP + b] : -absProjQuery[baseP + b];
                }
            }

            int remaining = _projDim & 7;
            if (remaining > 0)
            {
                byte bits = signBits[fullBytes];
                int baseP = fullBytes << 3;
                for (int b = 0; b < remaining; b++)
                {
                    bool residualPositive = (bits & (1 << b)) != 0;
                    bool queryPositive = projQuery[baseP + b] >= 0;
                    corr += (residualPositive == queryPositive)
                        ? absProjQuery[baseP + b] : -absProjQuery[baseP + b];
                }
            }

            corrections[pos] = corr * rNorm * invScale;
        }
    }

    /// <summary>Backward-compatible batch without residual norms.</summary>
    public void BatchCorrection(ReadOnlySpan<float> query, ReadOnlySpan<byte> allSignBits,
        int signBitStride, int count, Span<float> corrections)
    {
        Span<float> unitNorms = stackalloc float[count];
        unitNorms.Fill(1.0f);
        BatchCorrection(query, allSignBits, unitNorms, signBitStride, count, corrections);
    }

    // ── Pre-computed query projection for fused attention ──────────────────

    /// <summary>
    /// Pre-compute R · q and |R · q| for all projection dimensions.
    /// Call once per attention head, then use ComputeCorrectionPrecomputed per position.
    /// </summary>
    public void PrecomputeQueryProjections(ReadOnlySpan<float> query,
        Span<float> projQuery, Span<float> absProjQuery, out float invScale)
    {
        float queryProjNormSq = 0;
        for (int p = 0; p < _projDim; p++)
        {
            float pq = DotProduct(query, _projectionMatrix.AsSpan(p * _inputDim, _inputDim));
            projQuery[p] = pq;
            absProjQuery[p] = MathF.Abs(pq);
            queryProjNormSq += pq * pq;
        }

        float queryProjNorm = MathF.Sqrt(queryProjNormSq);
        invScale = queryProjNorm > 1e-10f ? 1.0f / (queryProjNorm * MathF.Sqrt(_projDim)) : 0;
    }

    /// <summary>
    /// Compute correction using pre-computed query projections. O(projDim) per position
    /// instead of O(projDim × inputDim) — avoids redundant query projection dot products.
    /// </summary>
    public float ComputeCorrectionPrecomputed(ReadOnlySpan<byte> signBits, float residualNorm,
        ReadOnlySpan<float> projQuery, ReadOnlySpan<float> absProjQuery, float invScale)
    {
        float corr = 0;

        int fullBytes = _projDim >> 3;
        for (int byteIdx = 0; byteIdx < fullBytes; byteIdx++)
        {
            byte bits = signBits[byteIdx];
            int baseP = byteIdx << 3;
            for (int b = 0; b < 8; b++)
            {
                bool residualPositive = (bits & (1 << b)) != 0;
                bool queryPositive = projQuery[baseP + b] >= 0;
                corr += (residualPositive == queryPositive)
                    ? absProjQuery[baseP + b] : -absProjQuery[baseP + b];
            }
        }

        int remaining = _projDim & 7;
        if (remaining > 0)
        {
            byte bits = signBits[fullBytes];
            int baseP = fullBytes << 3;
            for (int b = 0; b < remaining; b++)
            {
                bool residualPositive = (bits & (1 << b)) != 0;
                bool queryPositive = projQuery[baseP + b] >= 0;
                corr += (residualPositive == queryPositive)
                    ? absProjQuery[baseP + b] : -absProjQuery[baseP + b];
            }
        }

        return corr * residualNorm * invScale;
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
