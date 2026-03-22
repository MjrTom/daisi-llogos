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
    private readonly VocabRemapper? _remapper; // target uses remapped IDs, draft uses original

    public int TotalDraftTokens { get; private set; }
    public int TotalAcceptedTokens { get; private set; }
    public double AcceptanceRate => TotalDraftTokens > 0 ? (double)TotalAcceptedTokens / TotalDraftTokens : 0;

    public SpeculativeDecoder(ForwardPass target, ForwardPass draft, BpeTokenizer tokenizer,
        int specDepth = 5, VocabRemapper? remapper = null)
    {
        _target = target;
        _draft = draft;
        _tokenizer = tokenizer;
        _specDepth = specDepth;
        _remapper = remapper;
    }

    // Target uses remapped IDs, draft uses original IDs.
    // These translate between the two spaces.
    private int TargetToDraft(int targetId) => _remapper?.RemapDecode(targetId) ?? targetId;
    private int DraftToTarget(int draftId) => _remapper?.RemapEncode(draftId) ?? draftId;

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

        // Note: graph capture with two models re-instantiates frequently (topology changes)
        // but produces correct results since each model's forward pass is self-contained.

        // Prefill target model first (remapped IDs), then draft (original IDs)
        var prefillSw = System.Diagnostics.Stopwatch.StartNew();
        for (int i = 0; i < promptIds.Length - 1; i++)
            _target.ForwardHidden(promptIds[i], i);
        // nextToken is in target (remapped) ID space
        int nextToken = _target.ForwardArgMax(promptIds[^1], promptIds.Length - 1);

        // Draft uses original IDs — translate prompt tokens
        for (int i = 0; i < promptIds.Length - 1; i++)
            _draft.ForwardHidden(TargetToDraft(promptIds[i]), i);
        _draft.ForwardArgMax(TargetToDraft(promptIds[^1]), promptIds.Length - 1);
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

            // ── Draft phase: generate N candidates (in original ID space) ──
            int N = Math.Min(_specDepth, parameters.MaxTokens - t);
            var draftTokensOrig = new int[N]; // original ID space
            var draftTokensRemap = new int[N]; // remapped ID space (for target comparison)

            // Feed nextToken (remapped) to draft (needs original)
            int draftInput = TargetToDraft(nextToken);
            draftTokensOrig[0] = _draft.ForwardArgMax(draftInput, pos);
            draftTokensRemap[0] = DraftToTarget(draftTokensOrig[0]);
            for (int d = 1; d < N; d++)
            {
                draftTokensOrig[d] = _draft.ForwardArgMax(draftTokensOrig[d - 1], pos + d);
                draftTokensRemap[d] = DraftToTarget(draftTokensOrig[d]);
            }

            TotalDraftTokens += N;

            // ── Verify phase: target processes tokens (in remapped ID space) ──
            var targetPreds = new int[N];
            targetPreds[0] = _target.ForwardArgMax(nextToken, pos);
            for (int d = 1; d < N; d++)
                targetPreds[d] = _target.ForwardArgMax(draftTokensRemap[d - 1], pos + d);
            // targetPreds[i] = target's prediction for position pos+i+1
            // draftTokens[i] = draft's prediction for position pos+i+1

            // ── Accept/reject (compare in remapped ID space) ──────────────
            int accepted = 0;
            for (int i = 0; i < N; i++)
            {
                if (targetPreds[i] == draftTokensRemap[i])
                    accepted++;
                else
                    break;
            }
            TotalAcceptedTokens += accepted;

            // Emit accepted tokens (remapped IDs — tokenizer uses remapped space)
            for (int i = 0; i < accepted && t < parameters.MaxTokens; i++)
            {
                int tok = draftTokensRemap[i];
                if (Array.IndexOf(stopTokens, tok) >= 0) goto done;
                yield return new GenerationToken(tok, _tokenizer.Decode([tok]));
                generated++;
                t++;
            }

            if (accepted == N)
            {
                // All N accepted. Feed last accepted token to both models.
                nextToken = _target.ForwardArgMax(draftTokensRemap[N - 1], pos + N);
                _draft.ForwardArgMax(draftTokensOrig[N - 1], pos + N);
                pos += N + 1;
            }
            else
            {
                // Rejected at position `accepted`. Target's prediction replaces draft's.
                nextToken = targetPreds[accepted];

                _target.KvCache.SetLength(pos + accepted + 1);
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
