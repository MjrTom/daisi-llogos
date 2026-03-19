namespace Daisi.Llama.Inference;

/// <summary>
/// Transforms raw logits into a selected token ID through a configurable pipeline:
/// repetition penalty → temperature → top-k → top-p → softmax → sample/argmax.
/// </summary>
public sealed class Sampler
{
    private readonly Random _rng;

    public Sampler(int? seed = null)
    {
        _rng = seed.HasValue ? new Random(seed.Value) : new Random();
    }

    /// <summary>
    /// Sample a token ID from logits using the given parameters.
    /// </summary>
    public int Sample(ReadOnlySpan<float> logits, GenerationParams p, ReadOnlySpan<int> previousTokens)
    {
        int vocabSize = logits.Length;

        // Copy logits so we can mutate
        Span<float> work = vocabSize <= 8192
            ? stackalloc float[vocabSize]
            : new float[vocabSize];
        logits.CopyTo(work);

        // 1. Repetition penalty
        if (p.RepetitionPenalty != 1.0f)
            ApplyRepetitionPenalty(work, previousTokens, p.RepetitionPenalty);

        // 2. Greedy shortcut
        if (p.Temperature == 0)
            return ArgMax(work);

        // 3. Temperature
        float invTemp = 1.0f / p.Temperature;
        for (int i = 0; i < vocabSize; i++)
            work[i] *= invTemp;

        // 4. Top-k
        if (p.TopK > 0 && p.TopK < vocabSize)
            ApplyTopK(work, p.TopK);

        // 5. Softmax
        Softmax(work);

        // 6. Top-p (nucleus) — applied after softmax on probabilities
        if (p.TopP > 0 && p.TopP < 1.0f)
            ApplyTopP(work, p.TopP);

        // 7. Weighted random sample
        return SampleFromDistribution(work);
    }

    private static void ApplyRepetitionPenalty(Span<float> logits, ReadOnlySpan<int> previousTokens, float penalty)
    {
        for (int i = 0; i < previousTokens.Length; i++)
        {
            int tokenId = previousTokens[i];
            if ((uint)tokenId >= (uint)logits.Length) continue;

            if (logits[tokenId] > 0)
                logits[tokenId] /= penalty;
            else
                logits[tokenId] *= penalty;
        }
    }

    private static int ArgMax(ReadOnlySpan<float> values)
    {
        int best = 0;
        float bestVal = values[0];
        for (int i = 1; i < values.Length; i++)
        {
            if (values[i] > bestVal)
            {
                bestVal = values[i];
                best = i;
            }
        }
        return best;
    }

    private static void ApplyTopK(Span<float> logits, int k)
    {
        // Find the k-th largest value using partial sort
        int n = logits.Length;

        // Find top-k threshold
        float kthValue = FindKthLargest(logits, k);

        // Zero out everything below threshold
        for (int i = 0; i < n; i++)
        {
            if (logits[i] < kthValue)
                logits[i] = float.NegativeInfinity;
        }
    }

    private static float FindKthLargest(ReadOnlySpan<float> values, int k)
    {
        // Simple approach: collect top-k values
        Span<float> topK = k <= 1024
            ? stackalloc float[k]
            : new float[k];
        topK.Fill(float.NegativeInfinity);

        for (int i = 0; i < values.Length; i++)
        {
            float v = values[i];
            if (v > topK[k - 1])
            {
                topK[k - 1] = v;
                // Bubble up
                for (int j = k - 2; j >= 0; j--)
                {
                    if (topK[j + 1] > topK[j])
                        (topK[j], topK[j + 1]) = (topK[j + 1], topK[j]);
                    else
                        break;
                }
            }
        }

        return topK[k - 1];
    }

    private static void Softmax(Span<float> values)
    {
        float max = float.NegativeInfinity;
        for (int i = 0; i < values.Length; i++)
            if (values[i] > max) max = values[i];

        float sum = 0;
        for (int i = 0; i < values.Length; i++)
        {
            values[i] = MathF.Exp(values[i] - max);
            sum += values[i];
        }

        float invSum = 1.0f / sum;
        for (int i = 0; i < values.Length; i++)
            values[i] *= invSum;
    }

    private static void ApplyTopP(Span<float> probs, float p)
    {
        // Build (index, prob) pairs and sort descending
        int n = probs.Length;
        Span<(int idx, float prob)> sorted = n <= 8192
            ? stackalloc (int, float)[n]
            : new (int, float)[n];

        for (int i = 0; i < n; i++)
            sorted[i] = (i, probs[i]);

        // Sort descending by probability
        sorted.Sort((a, b) => b.prob.CompareTo(a.prob));

        // Find cumulative threshold
        float cumulative = 0;
        int cutoff = n;
        for (int i = 0; i < n; i++)
        {
            cumulative += sorted[i].prob;
            if (cumulative > p)
            {
                cutoff = i + 1;
                break;
            }
        }

        // Zero out tokens beyond cutoff
        for (int i = cutoff; i < n; i++)
            probs[sorted[i].idx] = 0;

        // Renormalize
        float sum = 0;
        for (int i = 0; i < n; i++)
            sum += probs[i];
        if (sum > 0)
        {
            float invSum = 1.0f / sum;
            for (int i = 0; i < n; i++)
                probs[i] *= invSum;
        }
    }

    private int SampleFromDistribution(ReadOnlySpan<float> probs)
    {
        float r = (float)_rng.NextDouble();
        float cumulative = 0;
        for (int i = 0; i < probs.Length; i++)
        {
            cumulative += probs[i];
            if (r < cumulative)
                return i;
        }
        // Fallback: return last non-zero
        for (int i = probs.Length - 1; i >= 0; i--)
            if (probs[i] > 0) return i;
        return 0;
    }
}
