namespace Daisi.Llogos.Inference;

/// <summary>
/// Transforms raw logits into a selected token ID through a configurable pipeline:
/// repetition penalty → temperature → top-k → top-p → softmax → sample/argmax.
/// </summary>
public sealed class Sampler
{
    private readonly Random _rng;
    private float[] _workBuffer = [];

    public Sampler(int? seed = null)
    {
        _rng = seed.HasValue ? new Random(seed.Value) : new Random();
    }

    private void EnsureWorkBuffer(int size)
    {
        if (_workBuffer.Length < size)
            _workBuffer = new float[size];
    }

    /// <summary>
    /// Sample a token ID from logits using the given parameters.
    /// </summary>
    public int Sample(ReadOnlySpan<float> logits, GenerationParams p, ReadOnlySpan<int> previousTokens)
    {
        int vocabSize = logits.Length;

        // Greedy shortcut: no copy needed, just find argmax
        if (p.Temperature == 0 && p.RepetitionPenalty == 1.0f)
            return ArgMax(logits);

        // Copy logits so we can mutate (reuse buffer to avoid GC pressure)
        EnsureWorkBuffer(vocabSize);
        logits.CopyTo(_workBuffer);
        var work = _workBuffer.AsSpan(0, vocabSize);

        // 1. Repetition penalty
        if (p.RepetitionPenalty != 1.0f)
            ApplyRepetitionPenalty(work, previousTokens, p.RepetitionPenalty);

        // 2. Greedy shortcut (after penalty)
        if (p.Temperature == 0)
            return ArgMax(work);

        // 3. Temperature
        float invTemp = 1.0f / p.Temperature;
        for (int i = 0; i < vocabSize; i++)
            work[i] *= invTemp;

        // 4. Top-k: collect indices of top-k candidates
        int effectiveK = (p.TopK > 0 && p.TopK < vocabSize) ? p.TopK : vocabSize;
        var candidates = CollectTopK(work, effectiveK);

        // 5. Softmax over candidates only (not full vocab)
        SoftmaxCandidates(work, candidates);

        // 6. Top-p (nucleus) over candidates
        if (p.TopP > 0 && p.TopP < 1.0f)
            ApplyTopPCandidates(work, candidates, p.TopP);

        // 7. Weighted random sample from candidates
        return SampleFromCandidates(work, candidates);
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

    /// <summary>
    /// Collect indices of top-k logit values. O(N*k) but k is small (typically 40).
    /// Returns the actual number of candidates (may be less than k if vocab is small).
    /// </summary>
    private static int[] CollectTopK(ReadOnlySpan<float> logits, int k)
    {
        int n = logits.Length;
        k = Math.Min(k, n);
        var topIdx = new int[k];
        var topVal = new float[k];
        Array.Fill(topVal, float.NegativeInfinity);

        for (int i = 0; i < n; i++)
        {
            float v = logits[i];
            if (v > topVal[k - 1])
            {
                topVal[k - 1] = v;
                topIdx[k - 1] = i;
                // Bubble up
                for (int j = k - 2; j >= 0; j--)
                {
                    if (topVal[j + 1] > topVal[j])
                    {
                        (topVal[j], topVal[j + 1]) = (topVal[j + 1], topVal[j]);
                        (topIdx[j], topIdx[j + 1]) = (topIdx[j + 1], topIdx[j]);
                    }
                    else break;
                }
            }
        }
        return topIdx;
    }

    /// <summary>
    /// Softmax over only the candidate indices. Much faster than full-vocab softmax.
    /// </summary>
    private static void SoftmaxCandidates(Span<float> logits, int[] candidates)
    {
        float max = float.NegativeInfinity;
        for (int i = 0; i < candidates.Length; i++)
            if (logits[candidates[i]] > max) max = logits[candidates[i]];

        float sum = 0;
        for (int i = 0; i < candidates.Length; i++)
        {
            float e = MathF.Exp(logits[candidates[i]] - max);
            logits[candidates[i]] = e;
            sum += e;
        }

        float invSum = 1.0f / sum;
        for (int i = 0; i < candidates.Length; i++)
            logits[candidates[i]] *= invSum;
    }

    /// <summary>
    /// Apply top-p filtering over candidates only. O(k log k) sort instead of O(N log N).
    /// </summary>
    private static void ApplyTopPCandidates(Span<float> probs, int[] candidates, float p)
    {
        // Sort candidates by probability descending
        // Extract probs into a small array to avoid Span capture issue
        var cp = new float[candidates.Length];
        for (int i = 0; i < candidates.Length; i++) cp[i] = probs[candidates[i]];
        Array.Sort(cp, candidates);
        Array.Reverse(candidates); // descending
        Array.Reverse(cp);

        float cumulative = 0;
        int cutoff = candidates.Length;
        for (int i = 0; i < candidates.Length; i++)
        {
            cumulative += probs[candidates[i]];
            if (cumulative > p)
            {
                cutoff = i + 1;
                break;
            }
        }

        // Zero out tokens beyond cutoff
        for (int i = cutoff; i < candidates.Length; i++)
            probs[candidates[i]] = 0;

        // Renormalize
        float sum = 0;
        for (int i = 0; i < cutoff; i++)
            sum += probs[candidates[i]];
        if (sum > 0)
        {
            float invSum = 1.0f / sum;
            for (int i = 0; i < cutoff; i++)
                probs[candidates[i]] *= invSum;
        }
    }

    private int SampleFromCandidates(ReadOnlySpan<float> probs, int[] candidates)
    {
        float r = (float)_rng.NextDouble();
        float cumulative = 0;
        for (int i = 0; i < candidates.Length; i++)
        {
            cumulative += probs[candidates[i]];
            if (r < cumulative)
                return candidates[i];
        }
        return candidates[0];
    }
}
