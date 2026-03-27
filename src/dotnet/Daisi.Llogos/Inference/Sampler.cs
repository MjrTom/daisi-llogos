namespace Daisi.Llogos.Inference;

/// <summary>
/// Transforms raw logits into a selected token ID through a configurable pipeline:
/// repetition/frequency/presence penalty → temperature → top-k → min-p → softmax → typical-p → top-p → sample.
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
        if (p.Temperature == 0 && p.RepetitionPenalty == 1.0f
            && p.FrequencyPenalty == 0 && p.PresencePenalty == 0)
            return ArgMax(logits);

        // Copy logits so we can mutate (reuse buffer to avoid GC pressure)
        EnsureWorkBuffer(vocabSize);
        logits.CopyTo(_workBuffer);
        var work = _workBuffer.AsSpan(0, vocabSize);

        // 1. Repetition / frequency / presence penalties
        ApplyPenalties(work, previousTokens, p);

        // 2. PreventEOS: set EOS logit to -inf
        if (p.PreventEOS && p.StopTokens != null)
        {
            foreach (int eos in p.StopTokens)
                if ((uint)eos < (uint)vocabSize) work[eos] = float.NegativeInfinity;
        }

        // 3. Greedy shortcut (after penalties)
        if (p.Temperature == 0)
            return ArgMax(work);

        // 4. Temperature
        float invTemp = 1.0f / p.Temperature;
        for (int i = 0; i < vocabSize; i++)
            work[i] *= invTemp;

        // 5. Top-k: collect indices of top-k candidates
        int effectiveK = (p.TopK > 0 && p.TopK < vocabSize) ? p.TopK : vocabSize;
        var candidates = CollectTopK(work, effectiveK);

        // 6. Min-p: remove tokens below min_p × max_prob (before softmax, on logits)
        if (p.MinP > 0 && p.MinP < 1.0f)
            candidates = ApplyMinP(work, candidates, p.MinP, Math.Max(p.MinKeep, 1));

        // 7. Softmax over candidates only
        SoftmaxCandidates(work, candidates);

        // 8. Typical-p: filter by typical probability
        if (p.TypicalP > 0 && p.TypicalP < 1.0f)
            candidates = ApplyTypicalP(work, candidates, p.TypicalP, Math.Max(p.MinKeep, 1));

        // 9. Top-p (nucleus) over candidates
        if (p.TopP > 0 && p.TopP < 1.0f)
            candidates = ApplyTopP(work, candidates, p.TopP, Math.Max(p.MinKeep, 1));

        // 10. Weighted random sample from candidates
        return SampleFromCandidates(work, candidates);
    }

    // ── Penalties ────────────────────────────────────────────────────────────

    private static void ApplyPenalties(Span<float> logits, ReadOnlySpan<int> previousTokens, GenerationParams p)
    {
        if (p.RepetitionPenalty == 1.0f && p.FrequencyPenalty == 0 && p.PresencePenalty == 0)
            return;

        // Determine penalty window
        int windowStart = 0;
        if (p.PenaltyCount > 0 && p.PenaltyCount < previousTokens.Length)
            windowStart = previousTokens.Length - p.PenaltyCount;

        var window = previousTokens[windowStart..];

        // Count token frequencies in window (for frequency penalty)
        Dictionary<int, int>? freqMap = null;
        if (p.FrequencyPenalty != 0 || p.PresencePenalty != 0)
        {
            freqMap = new Dictionary<int, int>();
            for (int i = 0; i < window.Length; i++)
            {
                int t = window[i];
                freqMap[t] = freqMap.GetValueOrDefault(t) + 1;
            }
        }

        // Apply repetition penalty (multiplicative)
        if (p.RepetitionPenalty != 1.0f)
        {
            for (int i = 0; i < window.Length; i++)
            {
                int tokenId = window[i];
                if ((uint)tokenId >= (uint)logits.Length) continue;
                if (!p.PenalizeNewline && IsNewlineToken(tokenId)) continue;

                if (logits[tokenId] > 0)
                    logits[tokenId] /= p.RepetitionPenalty;
                else
                    logits[tokenId] *= p.RepetitionPenalty;
            }
        }

        // Apply frequency + presence penalties (additive)
        if (freqMap != null)
        {
            foreach (var (tokenId, count) in freqMap)
            {
                if ((uint)tokenId >= (uint)logits.Length) continue;
                if (!p.PenalizeNewline && IsNewlineToken(tokenId)) continue;

                // Frequency: proportional to count
                logits[tokenId] -= p.FrequencyPenalty * count;
                // Presence: flat penalty if token appeared at all
                logits[tokenId] -= p.PresencePenalty;
            }
        }
    }

    private static bool IsNewlineToken(int tokenId)
    {
        // Common newline token IDs across models (10 = '\n' in most BPE vocabularies)
        return tokenId == 10 || tokenId == 13;
    }

    // ── Min-P ────────────────────────────────────────────────────────────────

    /// <summary>
    /// Remove candidates whose logit is below min_p × max_logit.
    /// Applied before softmax, so works on raw (temperature-scaled) logits.
    /// </summary>
    private static int[] ApplyMinP(Span<float> logits, int[] candidates, float minP, int minKeep)
    {
        if (candidates.Length <= minKeep) return candidates;

        float maxLogit = float.NegativeInfinity;
        foreach (int c in candidates)
            if (logits[c] > maxLogit) maxLogit = logits[c];

        float threshold = maxLogit + MathF.Log(minP); // log-domain: log(p) > log(min_p) + log(max_p)

        var kept = new List<int>(candidates.Length);
        foreach (int c in candidates)
        {
            if (logits[c] >= threshold || kept.Count < minKeep)
                kept.Add(c);
        }
        return kept.ToArray();
    }

    // ── Typical-P ────────────────────────────────────────────────────────────

    /// <summary>
    /// Typical sampling: keep tokens whose probability is close to the expected information content.
    /// Filters by |log(p) - entropy| sorted ascending, keeping cumulative up to typicalP.
    /// </summary>
    private static int[] ApplyTypicalP(Span<float> probs, int[] candidates, float typicalP, int minKeep)
    {
        if (candidates.Length <= minKeep) return candidates;

        // Compute entropy over candidates
        float entropy = 0;
        foreach (int c in candidates)
        {
            float prob = probs[c];
            if (prob > 0) entropy -= prob * MathF.Log(prob);
        }

        // Score each candidate by |log(p) - entropy|
        var scored = new (int idx, float deviation)[candidates.Length];
        for (int i = 0; i < candidates.Length; i++)
        {
            float prob = probs[candidates[i]];
            float logP = prob > 0 ? -MathF.Log(prob) : float.MaxValue;
            scored[i] = (candidates[i], MathF.Abs(logP - entropy));
        }

        // Sort by deviation ascending (most typical first)
        Array.Sort(scored, (a, b) => a.deviation.CompareTo(b.deviation));

        // Keep cumulative probability up to typicalP
        float cumulative = 0;
        var kept = new List<int>(candidates.Length);
        for (int i = 0; i < scored.Length; i++)
        {
            kept.Add(scored[i].idx);
            cumulative += probs[scored[i].idx];
            if (cumulative >= typicalP && kept.Count >= minKeep)
                break;
        }

        // Renormalize
        float sum = 0;
        foreach (int c in kept) sum += probs[c];
        if (sum > 0)
        {
            // Zero out non-kept
            foreach (int c in candidates)
                if (!kept.Contains(c)) probs[c] = 0;
            float invSum = 1.0f / sum;
            foreach (int c in kept) probs[c] *= invSum;
        }

        return kept.ToArray();
    }

    // ── Core sampling methods (unchanged) ────────────────────────────────────

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

    private static int[] ApplyTopP(Span<float> probs, int[] candidates, float p, int minKeep)
    {
        // Sort candidates by probability descending
        var cp = new float[candidates.Length];
        for (int i = 0; i < candidates.Length; i++) cp[i] = probs[candidates[i]];
        Array.Sort(cp, candidates);
        Array.Reverse(candidates);
        Array.Reverse(cp);

        float cumulative = 0;
        int cutoff = candidates.Length;
        for (int i = 0; i < candidates.Length; i++)
        {
            cumulative += probs[candidates[i]];
            if (cumulative > p && i + 1 >= minKeep)
            {
                cutoff = i + 1;
                break;
            }
        }

        for (int i = cutoff; i < candidates.Length; i++)
            probs[candidates[i]] = 0;

        float sum = 0;
        for (int i = 0; i < cutoff; i++)
            sum += probs[candidates[i]];
        if (sum > 0)
        {
            float invSum = 1.0f / sum;
            for (int i = 0; i < cutoff; i++)
                probs[candidates[i]] *= invSum;
        }

        return candidates[..cutoff];
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
