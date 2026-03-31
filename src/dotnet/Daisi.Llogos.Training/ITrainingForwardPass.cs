namespace Daisi.Llogos.Training;

/// <summary>
/// Common interface for training forward+backward pass implementations.
/// CPU and GPU implementations both implement this interface.
/// </summary>
public interface ITrainingForwardPass : IDisposable
{
    /// <summary>
    /// Run forward + backward pass on a training sequence.
    /// Computes logits, loss, and accumulates gradients into LoRA parameters.
    /// </summary>
    /// <param name="tokenIds">Input token IDs [T].</param>
    /// <param name="totalLoss">Average cross-entropy loss over the sequence.</param>
    /// <param name="targets">Target token IDs [T] (shifted by 1 from input).</param>
    /// <returns>Logits [T × vocabSize] (may be null for GPU implementations that don't download).</returns>
    float[]? Forward(int[] tokenIds, out float totalLoss, int[] targets);
}
