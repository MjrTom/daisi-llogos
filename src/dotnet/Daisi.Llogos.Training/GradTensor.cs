namespace Daisi.Llogos.Training;

/// <summary>
/// F32 tensor with optional gradient storage for training.
/// All training operations work on these tensors rather than ITensor.
/// </summary>
public sealed class GradTensor : IDisposable
{
    public float[] Data;
    public float[]? Grad;
    public readonly int[] Shape;
    public readonly int Size;
    public readonly bool RequiresGrad;

    public GradTensor(int[] shape, bool requiresGrad = false)
    {
        Shape = shape;
        Size = ComputeSize(shape);
        Data = new float[Size];
        RequiresGrad = requiresGrad;
        if (requiresGrad) Grad = new float[Size];
    }

    public GradTensor(float[] data, int[] shape, bool requiresGrad = false)
    {
        Shape = shape;
        Size = ComputeSize(shape);
        if (data.Length < Size)
            throw new ArgumentException($"Data length {data.Length} < required {Size}");
        Data = data;
        RequiresGrad = requiresGrad;
        if (requiresGrad) Grad = new float[Size];
    }

    public void ZeroGrad()
    {
        if (Grad != null) Array.Clear(Grad);
    }

    public void EnsureGrad()
    {
        Grad ??= new float[Size];
    }

    /// <summary>
    /// Accumulate gradient (used during backward pass).
    /// </summary>
    public void AccumulateGrad(ReadOnlySpan<float> gradient)
    {
        EnsureGrad();
        var g = Grad!;
        for (int i = 0; i < gradient.Length && i < g.Length; i++)
            g[i] += gradient[i];
    }

    /// <summary>
    /// Fill data with a constant value.
    /// </summary>
    public void Fill(float value)
    {
        Array.Fill(Data, value);
    }

    /// <summary>
    /// Fill with Kaiming uniform initialization: U(-bound, bound) where bound = sqrt(6/fan_in).
    /// </summary>
    public void KaimingUniform(int fanIn, Random rng)
    {
        float bound = MathF.Sqrt(6.0f / fanIn);
        for (int i = 0; i < Size; i++)
            Data[i] = (float)(rng.NextDouble() * 2.0 * bound - bound);
    }

    /// <summary>
    /// Fill with Gaussian noise: N(0, std).
    /// </summary>
    public void Gaussian(float std, Random rng)
    {
        for (int i = 0; i < Size; i++)
        {
            // Box-Muller transform
            double u1 = 1.0 - rng.NextDouble();
            double u2 = rng.NextDouble();
            Data[i] = (float)(std * Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2));
        }
    }

    public void Dispose() { }

    private static int ComputeSize(int[] shape)
    {
        int size = 1;
        for (int i = 0; i < shape.Length; i++)
            size *= shape[i];
        return size;
    }
}
