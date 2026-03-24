namespace Daisi.Llogos.Cpu;

/// <summary>
/// Rotary Position Embedding. Rotates pairs of dimensions (2i, 2i+1) by
/// position-dependent angles using sinusoidal frequencies.
/// Supports partial rotation where only the first ropeDim dimensions are rotated.
/// </summary>
internal static class Rope
{
    /// <summary>
    /// Apply RoPE in-place to q and k tensors.
    /// q is [nHeads × headDim], k is [nKvHeads × headDim], both stored flat.
    /// Only the first ropeDim dimensions of each head are rotated.
    /// </summary>
    public static void Apply(Span<float> q, Span<float> k, int headDim, int ropeDim, int positionOffset, float ropeTheta)
    {
        int nQHeads = q.Length / headDim;
        int nKvHeads = k.Length / headDim;

        for (int h = 0; h < nQHeads; h++)
            RotateHead(q.Slice(h * headDim, headDim), ropeDim, positionOffset, ropeTheta);

        for (int h = 0; h < nKvHeads; h++)
            RotateHead(k.Slice(h * headDim, headDim), ropeDim, positionOffset, ropeTheta);
    }

    private static void RotateHead(Span<float> head, int ropeDim, int position, float theta)
    {
        int halfDim = ropeDim / 2;
        for (int i = 0; i < halfDim; i++)
        {
            float freq = 1.0f / MathF.Pow(theta, 2.0f * i / ropeDim);
            float angle = position * freq;
            float cos = MathF.Cos(angle);
            float sin = MathF.Sin(angle);

            float x0 = head[2 * i];
            float x1 = head[2 * i + 1];
            head[2 * i] = x0 * cos - x1 * sin;
            head[2 * i + 1] = x0 * sin + x1 * cos;
        }
        // Dimensions beyond ropeDim are left unchanged (NoPE)
    }
}
