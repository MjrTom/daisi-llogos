using System.Diagnostics;
using Daisi.Llama.Tokenizer;

namespace Daisi.Llama.Inference;

/// <summary>
/// Runs the full text generation loop: tokenize → prefill → decode → detokenize.
/// Streams tokens as they are generated.
/// </summary>
public sealed class TextGenerator
{
    private readonly ForwardPass _forward;
    private readonly BpeTokenizer _tokenizer;
    private readonly Sampler _sampler;

    public TextGenerator(ForwardPass forward, BpeTokenizer tokenizer, int? seed = null)
    {
        _forward = forward;
        _tokenizer = tokenizer;
        _sampler = new Sampler(seed);
    }

    /// <summary>
    /// Generate text from a prompt, yielding decoded token strings as they are produced.
    /// </summary>
    public IEnumerable<GenerationToken> Generate(string prompt, GenerationParams parameters)
    {
        var stopTokens = parameters.StopTokens
            ?? (_tokenizer.Vocabulary.EosTokenId >= 0
                ? [_tokenizer.Vocabulary.EosTokenId]
                : []);

        // Tokenize prompt
        var promptIds = _tokenizer.Encode(prompt);
        if (promptIds.Length == 0)
            yield break;

        var history = new List<int>(promptIds);

        // Prefill: process all prompt tokens
        ReadOnlySpan<float> logits = default;
        for (int i = 0; i < promptIds.Length; i++)
            logits = _forward.Forward(promptIds[i], i);

        int position = promptIds.Length;
        var sw = Stopwatch.StartNew();
        int generated = 0;

        // Decode loop
        for (int t = 0; t < parameters.MaxTokens; t++)
        {
            int tokenId = _sampler.Sample(logits, parameters, history.ToArray());

            // Check stop condition
            if (Array.IndexOf(stopTokens, tokenId) >= 0)
                break;

            history.Add(tokenId);
            generated++;

            // Decode and yield
            string text = _tokenizer.Decode([tokenId]);
            yield return new GenerationToken(tokenId, text);

            // Next forward pass
            logits = _forward.Forward(tokenId, position);
            position++;
        }

        sw.Stop();
        double tokensPerSec = generated > 0 ? generated / sw.Elapsed.TotalSeconds : 0;
        yield return GenerationToken.Done(generated, tokensPerSec);
    }
}

/// <summary>
/// A single generated token or the final done signal with stats.
/// </summary>
public readonly record struct GenerationToken
{
    /// <summary>The token ID (-1 for done signal).</summary>
    public int TokenId { get; init; }

    /// <summary>The decoded text for this token (empty for done signal).</summary>
    public string Text { get; init; }

    /// <summary>True if this is the final "done" signal with stats.</summary>
    public bool IsDone { get; init; }

    /// <summary>Total tokens generated (only set on done signal).</summary>
    public int TotalTokens { get; init; }

    /// <summary>Tokens per second (only set on done signal).</summary>
    public double TokensPerSecond { get; init; }

    public GenerationToken(int tokenId, string text)
    {
        TokenId = tokenId;
        Text = text;
        IsDone = false;
    }

    internal static GenerationToken Done(int totalTokens, double tokensPerSec) => new()
    {
        TokenId = -1,
        Text = "",
        IsDone = true,
        TotalTokens = totalTokens,
        TokensPerSecond = tokensPerSec
    };
}
