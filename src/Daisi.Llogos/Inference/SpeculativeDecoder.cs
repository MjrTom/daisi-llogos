using Daisi.Llogos.Tokenizer;

namespace Daisi.Llogos.Inference;

/// <summary>
/// Speculative decoding: a small draft model generates N candidate tokens,
/// then the target model verifies them all in one batched forward pass.
/// Accepted tokens are emitted immediately; on mismatch, the target's
/// prediction replaces the draft's and generation continues.
/// </summary>
public sealed class SpeculativeDecoder
{
    private readonly ForwardPass _target;
    private readonly ForwardPass _draft;
    private readonly BpeTokenizer _tokenizer;
    private readonly int _specDepth;

    public int TotalDraftTokens { get; private set; }
    public int TotalAcceptedTokens { get; private set; }
    public double AcceptanceRate => TotalDraftTokens > 0 ? (double)TotalAcceptedTokens / TotalDraftTokens : 0;

    public SpeculativeDecoder(ForwardPass target, ForwardPass draft, BpeTokenizer tokenizer, int specDepth = 5)
    {
        _target = target;
        _draft = draft;
        _tokenizer = tokenizer;
        _specDepth = specDepth;
    }

    /// <summary>
    /// Generate tokens with speculative decoding.
    /// </summary>
    public IEnumerable<GenerationToken> Generate(string prompt, GenerationParams parameters)
    {
        var stopTokens = parameters.StopTokens
            ?? (_tokenizer.Vocabulary.EosTokenId >= 0
                ? [_tokenizer.Vocabulary.EosTokenId]
                : []);

        var promptIds = _tokenizer.Encode(prompt);
        if (promptIds.Length == 0) yield break;

        // Disable CUDA graph capture — two models sharing one backend corrupt the graph exec
        _target.DisableGraphCapture();
        _draft.DisableGraphCapture();

        // Prefill target model first, then draft (no interleaving — shared GPU state)
        var prefillSw = System.Diagnostics.Stopwatch.StartNew();
        for (int i = 0; i < promptIds.Length - 1; i++)
            _target.ForwardHidden(promptIds[i], i);
        int nextToken = _target.ForwardArgMax(promptIds[^1], promptIds.Length - 1);

        for (int i = 0; i < promptIds.Length - 1; i++)
            _draft.ForwardHidden(promptIds[i], i);
        _draft.ForwardArgMax(promptIds[^1], promptIds.Length - 1);
        prefillSw.Stop();

        // pos = next KV cache write position (both models are synced here)
        int pos = promptIds.Length;
        var decodeSw = System.Diagnostics.Stopwatch.StartNew();
        int generated = 0;
        int t = 0;

        while (t < parameters.MaxTokens)
        {
            if (Array.IndexOf(stopTokens, nextToken) >= 0) break;

            // Emit the confirmed token (predicted by target, not yet in either model's KV cache)
            yield return new GenerationToken(nextToken, _tokenizer.Decode([nextToken]));
            generated++;
            t++;
            if (t >= parameters.MaxTokens) break;

            // ── Draft phase: generate N candidates ────────────────────────
            int N = Math.Min(_specDepth, parameters.MaxTokens - t);
            var draftTokens = new int[N];

            // Feed nextToken to draft at pos, get prediction for pos+1
            draftTokens[0] = _draft.ForwardArgMax(nextToken, pos);
            for (int d = 1; d < N; d++)
                draftTokens[d] = _draft.ForwardArgMax(draftTokens[d - 1], pos + d);
            // Draft KV cache now has entries [0..pos+N-1]
            // draftTokens[d] is the predicted token for position pos+d+1

            TotalDraftTokens += N;

            // ── Verify phase: target processes [nextToken, draftTokens[0..N-2]] ──
            // Sequential verify: uses same M=1 compute path as native decode
            // (batched verify uses FP16 GemmEx which produces different argmax)
            var targetPreds = new int[N];
            targetPreds[0] = _target.ForwardArgMax(nextToken, pos);
            for (int d = 1; d < N; d++)
                targetPreds[d] = _target.ForwardArgMax(draftTokens[d - 1], pos + d);
            // targetPreds[i] = target's prediction for position pos+i+1
            // draftTokens[i] = draft's prediction for position pos+i+1

            // ── Accept/reject ─────────────────────────────────────────────
            int accepted = 0;
            for (int i = 0; i < N; i++)
            {
                if (targetPreds[i] == draftTokens[i])
                    accepted++;
                else
                    break;
            }
            TotalAcceptedTokens += accepted;

            // Emit accepted tokens
            for (int i = 0; i < accepted && t < parameters.MaxTokens; i++)
            {
                if (Array.IndexOf(stopTokens, draftTokens[i]) >= 0) goto done;
                yield return new GenerationToken(draftTokens[i], _tokenizer.Decode([draftTokens[i]]));
                generated++;
                t++;
            }

            if (accepted == N)
            {
                // All N accepted. Sequential verify already processed up to pos+N-1.
                // Feed draftTokens[N-1] at pos+N to both models.
                nextToken = _target.ForwardArgMax(draftTokens[N - 1], pos + N);
                _draft.ForwardArgMax(draftTokens[N - 1], pos + N);
                pos += N + 1;
            }
            else
            {
                // Rejected at position `accepted`. Target's prediction replaces draft's.
                nextToken = targetPreds[accepted];

                // Target KV cache has entries [0..pos+N-1] but only [0..pos+accepted] are valid.
                // Truncate to pos+accepted+1 (accepted tokens + the verify input that produced the correction).
                _target.KvCache.SetLength(pos + accepted + 1);

                // Draft KV cache has entries [0..pos+N-1]. Truncate to match target.
                _draft.KvCache.SetLength(pos + accepted + 1);

                pos += accepted + 1;
            }
        }
        done:

        decodeSw.Stop();
        double prefillTokPerSec = promptIds.Length / prefillSw.Elapsed.TotalSeconds;
        double decodeTokPerSec = generated > 0 ? generated / decodeSw.Elapsed.TotalSeconds : 0;
        yield return GenerationToken.Done(generated, decodeTokPerSec, promptIds.Length, prefillTokPerSec);
    }
}
