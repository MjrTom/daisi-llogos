namespace Daisi.Llogos.Cpu;

/// <summary>
/// Rotary Position Embedding. Rotates pairs of dimensions (2i, 2i+1) by
/// position-dependent angles using sinusoidal frequencies.
/// Supports partial rotation where only the first ropeDim dimensions are rotated.
/// </summary>
internal static class Rope
{
    // Precomputed inverse frequencies per (ropeDim, theta) pair.
    // llama.cpp precomputes these in double precision — critical for matching its output.
    private static double[]? _cachedFreqs;
    private static int _cachedRopeDim;
    private static float _cachedTheta;

    /// <summary>
    /// Apply RoPE in-place to q and k tensors.
    /// q is [nHeads × headDim], k is [nKvHeads × headDim], both stored flat.
    /// Only the first ropeDim dimensions of each head are rotated.
    /// </summary>
    public static void Apply(Span<float> q, Span<float> k, int headDim, int ropeDim, int positionOffset, float ropeTheta)
    {
        int nQHeads = q.Length / headDim;
        int nKvHeads = k.Length / headDim;

        // Precompute frequencies in double precision (matches llama.cpp)
        var freqs = GetFreqs(ropeDim, ropeTheta);

        for (int h = 0; h < nQHeads; h++)
            RotateHead(q.Slice(h * headDim, headDim), ropeDim, positionOffset, freqs);

        for (int h = 0; h < nKvHeads; h++)
            RotateHead(k.Slice(h * headDim, headDim), ropeDim, positionOffset, freqs);
    }

    private static double[] GetFreqs(int ropeDim, float theta)
    {
        if (_cachedFreqs != null && _cachedRopeDim == ropeDim && _cachedTheta == theta)
            return _cachedFreqs;

        int halfDim = ropeDim / 2;
        var freqs = new double[halfDim];
        for (int i = 0; i < halfDim; i++)
            freqs[i] = 1.0 / Math.Pow((double)theta, 2.0 * i / ropeDim);

        _cachedFreqs = freqs;
        _cachedRopeDim = ropeDim;
        _cachedTheta = theta;
        return freqs;
    }

    private static void RotateHead(Span<float> head, int ropeDim, int position, double[] freqs)
    {
        int halfDim = ropeDim / 2;
        for (int i = 0; i < halfDim; i++)
        {
            // Compute angle in double precision (matches llama.cpp)
            double angle = position * freqs[i];
            float cos = (float)Math.Cos(angle);
            float sin = (float)Math.Sin(angle);

            float x0 = head[2 * i];
            float x1 = head[2 * i + 1];
            head[2 * i] = x0 * cos - x1 * sin;
            head[2 * i + 1] = x0 * sin + x1 * cos;
        }
    }
}
