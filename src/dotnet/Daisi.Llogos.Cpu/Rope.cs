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

    /// <summary>
    /// Apply RoPE with explicit per-pair frequency multipliers.
    /// Used by Gemma 4 full-attention layers (proportional / precomputed RoPE).
    /// freqFactors is [ropeDim/2] — each pair index i has its angle scaled by 1/freqFactors[i].
    /// (llama.cpp's ggml_rope_ext divides by freq_factors when applying.)
    /// </summary>
    public static void ApplyWithFreqFactors(Span<float> q, Span<float> k, int headDim, int ropeDim,
        int positionOffset, float ropeTheta, ReadOnlySpan<float> freqFactors)
    {
        int nQHeads = q.Length / headDim;
        int nKvHeads = k.Length / headDim;
        var baseFreqs = GetFreqs(ropeDim, ropeTheta);

        // Apply freq factor: effective_freq[i] = base_freq[i] / freq_factors[i]
        int halfDim = ropeDim / 2;
        var adjusted = new double[halfDim];
        int factorLen = freqFactors.Length;
        for (int i = 0; i < halfDim; i++)
        {
            float factor = i < factorLen ? freqFactors[i] : 1.0f;
            adjusted[i] = factor != 0.0f ? baseFreqs[i] / factor : baseFreqs[i];
        }

        for (int h = 0; h < nQHeads; h++)
            RotateHead(q.Slice(h * headDim, headDim), ropeDim, positionOffset, adjusted);
        for (int h = 0; h < nKvHeads; h++)
            RotateHead(k.Slice(h * headDim, headDim), ropeDim, positionOffset, adjusted);
    }

    // ── NEOX-style RoPE (split-half pairs) ─────────────────────────────────────
    //
    // The "GPT-NeoX" / Falcon / Gemma / Qwen RoPE convention pairs element i with
    // element i + ropeDim/2 (rotating two halves of the head), instead of the
    // "interleaved" convention which pairs (2i, 2i+1).
    //
    // Math is the same; only the pairing differs:
    //   interleaved: (head[2i], head[2i+1])    rotated by angle = pos * theta^(-2i/d)
    //   neox:        (head[i],  head[i+d/2])   rotated by the same angle
    //
    // This is required for Gemma 4 (LLAMA_ROPE_TYPE_NEOX) and any other arch where
    // the converter does NOT pre-permute Q/K weights into interleaved layout.

    /// <summary>NEOX-style RoPE: pairs are (head[i], head[i+d/2]).</summary>
    public static void ApplyNeox(Span<float> q, Span<float> k, int headDim, int ropeDim,
        int positionOffset, float ropeTheta)
    {
        int nQHeads = q.Length / headDim;
        int nKvHeads = k.Length / headDim;
        var freqs = GetFreqs(ropeDim, ropeTheta);

        for (int h = 0; h < nQHeads; h++)
            RotateHeadNeox(q.Slice(h * headDim, headDim), ropeDim, positionOffset, freqs);
        for (int h = 0; h < nKvHeads; h++)
            RotateHeadNeox(k.Slice(h * headDim, headDim), ropeDim, positionOffset, freqs);
    }

    /// <summary>NEOX-style RoPE with per-pair frequency multipliers (Gemma 4 full-attention).</summary>
    public static void ApplyNeoxWithFreqFactors(Span<float> q, Span<float> k, int headDim, int ropeDim,
        int positionOffset, float ropeTheta, ReadOnlySpan<float> freqFactors)
    {
        int nQHeads = q.Length / headDim;
        int nKvHeads = k.Length / headDim;
        var baseFreqs = GetFreqs(ropeDim, ropeTheta);

        int halfDim = ropeDim / 2;
        var adjusted = new double[halfDim];
        int factorLen = freqFactors.Length;
        for (int i = 0; i < halfDim; i++)
        {
            float factor = i < factorLen ? freqFactors[i] : 1.0f;
            adjusted[i] = factor != 0.0f ? baseFreqs[i] / factor : baseFreqs[i];
        }

        for (int h = 0; h < nQHeads; h++)
            RotateHeadNeox(q.Slice(h * headDim, headDim), ropeDim, positionOffset, adjusted);
        for (int h = 0; h < nKvHeads; h++)
            RotateHeadNeox(k.Slice(h * headDim, headDim), ropeDim, positionOffset, adjusted);
    }

    private static void RotateHeadNeox(Span<float> head, int ropeDim, int position, double[] freqs)
    {
        int halfDim = ropeDim / 2;
        for (int i = 0; i < halfDim; i++)
        {
            double angle = position * freqs[i];
            float cos = (float)Math.Cos(angle);
            float sin = (float)Math.Sin(angle);

            float x0 = head[i];
            float x1 = head[i + halfDim];
            head[i]           = x0 * cos - x1 * sin;
            head[i + halfDim] = x0 * sin + x1 * cos;
        }
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
