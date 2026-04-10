using System.Diagnostics;
using Daisi.Llogos.Tokenizer;

namespace Daisi.Llogos.Inference;

/// <summary>
/// Text generation loop for Gemma 4 models. Mirrors <see cref="BitNetTextGenerator"/>
/// — uses <see cref="Gemma4ForwardPass"/> directly without going through the
/// <see cref="IForwardPass"/> abstraction.
/// </summary>
public sealed class Gemma4TextGenerator
{
    private readonly Gemma4ForwardPass _forward;
    private readonly BpeTokenizer _tokenizer;
    private readonly Sampler _sampler;

    public Gemma4TextGenerator(Gemma4ForwardPass forward, BpeTokenizer tokenizer, int? seed = null)
    {
        _forward = forward;
        _tokenizer = tokenizer;
        _sampler = new Sampler(seed);
    }

    public IEnumerable<GenerationToken> Generate(string prompt, GenerationParams parameters)
    {
        var stopTokens = parameters.StopTokens
            ?? (_tokenizer.Vocabulary.EosTokenId >= 0
                ? [_tokenizer.Vocabulary.EosTokenId]
                : []);

        var promptIds = _tokenizer.Encode(prompt);
        if (promptIds.Length == 0)
            yield break;

        var history = new List<int>(promptIds);

        var prefillSw = Stopwatch.StartNew();
        ReadOnlySpan<float> logits = default;

        // Prefill in batches — reuses weight loads across M tokens in each dense matmul.
        // MaxBatch=64 matches the Gemma4ForwardPass default maxBatchSize.
        const int MaxBatch = 64;
        int pos = 0;
        while (pos < promptIds.Length)
        {
            int take = Math.Min(MaxBatch, promptIds.Length - pos);
            logits = _forward.ForwardBatch(promptIds.AsSpan(pos, take), pos);
            pos += take;
        }
        prefillSw.Stop();

        int position = promptIds.Length;
        var decodeSw = Stopwatch.StartNew();
        int generated = 0;

        for (int t = 0; t < parameters.MaxTokens; t++)
        {
            int tokenId = _sampler.Sample(logits, parameters, history.ToArray());
            if (Array.IndexOf(stopTokens, tokenId) >= 0)
                break;

            history.Add(tokenId);
            generated++;

            string text = _tokenizer.Decode([tokenId]);
            yield return new GenerationToken(tokenId, text);

            logits = _forward.Forward(tokenId, position);
            position++;
        }

        decodeSw.Stop();
        double prefillTokPerSec = promptIds.Length > 0 ? promptIds.Length / prefillSw.Elapsed.TotalSeconds : 0;
        double decodeTokPerSec = generated > 0 ? generated / decodeSw.Elapsed.TotalSeconds : 0;
        yield return GenerationToken.Done(generated, decodeTokPerSec, promptIds.Length, prefillTokPerSec);
    }
}
