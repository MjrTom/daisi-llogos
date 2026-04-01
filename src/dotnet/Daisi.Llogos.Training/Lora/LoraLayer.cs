namespace Daisi.Llogos.Training.Lora;

/// <summary>
/// A single LoRA decomposition: output += scaling * B @ A @ input.
/// A is [rank × inFeatures], B is [outFeatures × rank].
/// A is initialized with Kaiming uniform, B is initialized to zero (so LoRA starts as identity).
/// </summary>
public sealed class LoraLayer
{
    public readonly string Name;
    public readonly int InFeatures;
    public readonly int OutFeatures;
    public readonly int Rank;
    public readonly float Scaling;

    /// <summary>A matrix: [rank × inFeatures]</summary>
    public readonly GradTensor A;

    /// <summary>B matrix: [outFeatures × rank]</summary>
    public readonly GradTensor B;

    public LoraLayer(string name, int inFeatures, int outFeatures, int rank, float scaling, Random rng)
    {
        Name = name;
        InFeatures = inFeatures;
        OutFeatures = outFeatures;
        Rank = rank;
        Scaling = scaling;

        A = new GradTensor([rank, inFeatures], requiresGrad: true);
        B = new GradTensor([outFeatures, rank], requiresGrad: true);

        // Standard LoRA init: A ~ Kaiming, B = 0
        A.KaimingUniform(inFeatures, rng);
        B.Fill(0);
    }

    /// <summary>
    /// Create from existing weights (for loading from file).
    /// </summary>
    public LoraLayer(string name, int inFeatures, int outFeatures, int rank, float scaling,
        float[] aData, float[] bData)
    {
        Name = name;
        InFeatures = inFeatures;
        OutFeatures = outFeatures;
        Rank = rank;
        Scaling = scaling;

        A = new GradTensor(aData, [rank, inFeatures], requiresGrad: true);
        B = new GradTensor(bData, [outFeatures, rank], requiresGrad: true);
    }

    /// <summary>
    /// Forward: output += scaling * B @ (A @ input).
    /// input: [M × inFeatures], output: [M × outFeatures].
    /// Saves A@input into loraIntermediate for backward.
    /// </summary>
    public void Forward(Span<float> output, ReadOnlySpan<float> input,
        Span<float> loraIntermediate, int M)
    {
        // intermediate = A @ input^T → intermediate is [rank × M] but we compute row by row
        // Actually: for each row m of input, intermediate[m] = A @ input[m] (vector, rank elements)
        // Then: output[m] += scaling * B @ intermediate[m]

        // Step 1: intermediate = input × A^T  → [M × rank]
        // (input is [M × inFeatures], A is [rank × inFeatures], A^T is [inFeatures × rank])
        F32Ops.MatMulTransB(loraIntermediate, input, A.Data, M, InFeatures, Rank);

        // Step 2: loraOut = intermediate × B^T  → [M × outFeatures]
        // (intermediate is [M × rank], B is [outFeatures × rank], B^T is [rank × outFeatures])
        var loraOut = new float[M * OutFeatures];
        F32Ops.MatMulTransB(loraOut, loraIntermediate, B.Data, M, Rank, OutFeatures);

        // Add scaled result to output
        for (int i = 0; i < M * OutFeatures; i++)
            output[i] += Scaling * loraOut[i];
    }

    /// <summary>
    /// Backward: compute gradients for A and B, and return gradient w.r.t. input.
    /// dOutput: [M × outFeatures] gradient flowing back through the LoRA path.
    /// input: [M × inFeatures] saved from forward.
    /// loraIntermediate: [M × rank] saved from forward (A @ input).
    /// Returns dInput contribution: [M × inFeatures].
    /// </summary>
    public void Backward(Span<float> dInput, ReadOnlySpan<float> dOutput,
        ReadOnlySpan<float> input, ReadOnlySpan<float> loraIntermediate, int M)
    {
        // Scale the gradient
        var scaledDOutput = new float[M * OutFeatures];
        for (int i = 0; i < scaledDOutput.Length; i++)
            scaledDOutput[i] = dOutput[i] * Scaling;

        // dB += scaledDOutput^T @ loraIntermediate → [outFeatures × rank]
        // scaledDOutput is [M × outFeatures], loraIntermediate is [M × rank]
        F32Ops.MatMulTransA(B.Grad!, scaledDOutput, loraIntermediate, OutFeatures, M, Rank);

        // dIntermediate = scaledDOutput @ B → [M × rank]
        // scaledDOutput is [M × outFeatures], B is [outFeatures × rank]
        var dIntermediate = new float[M * Rank];
        F32Ops.MatMul(dIntermediate, scaledDOutput, B.Data, M, OutFeatures, Rank);

        // dA += dIntermediate^T @ input → [rank × inFeatures]
        // dIntermediate is [M × rank], input is [M × inFeatures]
        F32Ops.MatMulTransA(A.Grad!, dIntermediate, input, Rank, M, InFeatures);

        // dInput += dIntermediate @ A → [M × inFeatures]
        // dIntermediate is [M × rank], A is [rank × inFeatures]
        var dInputContrib = new float[M * InFeatures];
        F32Ops.MatMul(dInputContrib, dIntermediate, A.Data, M, Rank, InFeatures);
        F32Ops.AddInPlace(dInput, dInputContrib, M * InFeatures);
    }

    public void ZeroGrad()
    {
        A.ZeroGrad();
        B.ZeroGrad();
    }
}
