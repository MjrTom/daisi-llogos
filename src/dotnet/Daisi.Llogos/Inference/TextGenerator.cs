using System.Diagnostics;
using Daisi.Llogos.Tokenizer;

namespace Daisi.Llogos.Inference;

/// <summary>
/// Runs the full text generation loop: tokenize → prefill → decode → detokenize.
/// Streams tokens as they are generated.
/// </summary>
public sealed class TextGenerator
{
    private readonly IForwardPass _forward;
    private readonly BpeTokenizer _tokenizer;
    private readonly Sampler _sampler;

    public TextGenerator(IForwardPass forward, BpeTokenizer tokenizer, int? seed = null)
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
        var prefillSw = Stopwatch.StartNew();
        ReadOnlySpan<float> logits = default;
        if (_forward.SupportsBatchedPrefill && promptIds.Length > 2)
        {
            // Batched prefill: process all tokens except the last in one parallel pass
            _forward.ForwardBatchedPrefill(promptIds[..^1], 0);
            // Last token needs full logit output
            logits = _forward.Forward(promptIds[^1], promptIds.Length - 1);
        }
        else
        {
            // Sequential prefill: one token at a time
            for (int i = 0; i < promptIds.Length - 1; i++)
                _forward.ForwardHidden(promptIds[i], i);
            logits = _forward.Forward(promptIds[^1], promptIds.Length - 1);
        }
        prefillSw.Stop();

        int position = promptIds.Length;
        var decodeSw = Stopwatch.StartNew();
        int generated = 0;

        // Anti-prompt buffer for string-based stop detection
        string outputBuffer = "";
        bool hasAntiPrompts = parameters.AntiPrompts is { Length: > 0 };

        // Decode loop
        for (int t = 0; t < parameters.MaxTokens; t++)
        {
            int tokenId = _sampler.Sample(logits, parameters, history.ToArray());

            // Check token-based stop condition
            if (Array.IndexOf(stopTokens, tokenId) >= 0)
                break;

            history.Add(tokenId);
            generated++;

            // Decode and yield
            string text = _tokenizer.Decode([tokenId]);
            yield return new GenerationToken(tokenId, text);

            // Check anti-prompt (string-based stop sequences)
            if (hasAntiPrompts)
            {
                outputBuffer += text;
                // Keep buffer to max anti-prompt length + margin
                int maxLen = 0;
                foreach (var ap in parameters.AntiPrompts!)
                    if (ap.Length > maxLen) maxLen = ap.Length;
                if (outputBuffer.Length > maxLen * 2)
                    outputBuffer = outputBuffer[(outputBuffer.Length - maxLen * 2)..];

                bool stopped = false;
                foreach (var ap in parameters.AntiPrompts)
                {
                    if (outputBuffer.Contains(ap, StringComparison.Ordinal))
                    {
                        stopped = true;
                        break;
                    }
                }
                if (stopped) break;
            }

            // Next forward pass
            logits = _forward.Forward(tokenId, position);
            position++;
        }

        decodeSw.Stop();
        double prefillTokPerSec = promptIds.Length > 0 ? promptIds.Length / prefillSw.Elapsed.TotalSeconds : 0;
        double decodeTokPerSec = generated > 0 ? generated / decodeSw.Elapsed.TotalSeconds : 0;
        yield return GenerationToken.Done(generated, decodeTokPerSec, promptIds.Length, prefillTokPerSec);
    }

    /// <summary>
    /// Run a benchmark: prefill the prompt, generate tokens, and return detailed timing.
    /// </summary>
    public BenchmarkResult Benchmark(string prompt, int maxTokens = 128)
    {
        var promptIds = _tokenizer.Encode(prompt);
        if (promptIds.Length == 0)
            return default;

        // Prefill
        var prefillSw = Stopwatch.StartNew();
        ReadOnlySpan<float> logits = default;
        if (_forward.SupportsBatchedPrefill && promptIds.Length > 2)
        {
            _forward.ForwardBatchedPrefill(promptIds[..^1], 0);
            logits = _forward.Forward(promptIds[^1], promptIds.Length - 1);
        }
        else
        {
            for (int i = 0; i < promptIds.Length - 1; i++)
                _forward.ForwardHidden(promptIds[i], i);
            logits = _forward.Forward(promptIds[^1], promptIds.Length - 1);
        }
        prefillSw.Stop();

        // Decode — use GPU-side argmax for maximum throughput (greedy benchmark)
        var stopTokens = _tokenizer.Vocabulary.EosTokenId >= 0
            ? new[] { _tokenizer.Vocabulary.EosTokenId }
            : Array.Empty<int>();

        int generated = 0;
        var decodeSw = Stopwatch.StartNew();

        // First token uses the prefill logits
        int nextTokenId = _sampler.Sample(logits, new GenerationParams { Temperature = 0 }, []);

        for (int t = 0; t < maxTokens; t++)
        {
            if (Array.IndexOf(stopTokens, nextTokenId) >= 0) break;
            generated++;

            // Use ForwardArgMax to skip full logit download
            nextTokenId = _forward.ForwardArgMax(nextTokenId, promptIds.Length + t);
        }
        decodeSw.Stop();

        return new BenchmarkResult
        {
            PromptTokens = promptIds.Length,
            GeneratedTokens = generated,
            PrefillTime = prefillSw.Elapsed,
            DecodeTime = decodeSw.Elapsed,
            PrefillTokPerSec = promptIds.Length / prefillSw.Elapsed.TotalSeconds,
            DecodeTokPerSec = generated > 0 ? generated / decodeSw.Elapsed.TotalSeconds : 0,
        };
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

    /// <summary>Decode tokens per second (only set on done signal).</summary>
    public double TokensPerSecond { get; init; }

    /// <summary>Number of prompt tokens prefilled (only set on done signal).</summary>
    public int PrefillTokens { get; init; }

    /// <summary>Prefill tokens per second (only set on done signal).</summary>
    public double PrefillTokensPerSecond { get; init; }

    public GenerationToken(int tokenId, string text)
    {
        TokenId = tokenId;
        Text = text;
        IsDone = false;
    }

    internal static GenerationToken Done(int totalTokens, double tokensPerSec,
        int prefillTokens = 0, double prefillTokPerSec = 0) => new()
    {
        TokenId = -1,
        Text = "",
        IsDone = true,
        TotalTokens = totalTokens,
        TokensPerSecond = tokensPerSec,
        PrefillTokens = prefillTokens,
        PrefillTokensPerSecond = prefillTokPerSec
    };
}

/// <summary>
/// Detailed benchmark results for prefill and decode phases.
/// </summary>
public readonly record struct BenchmarkResult
{
    public int PromptTokens { get; init; }
    public int GeneratedTokens { get; init; }
    public TimeSpan PrefillTime { get; init; }
    public TimeSpan DecodeTime { get; init; }
    public double PrefillTokPerSec { get; init; }
    public double DecodeTokPerSec { get; init; }
    public TimeSpan TotalTime => PrefillTime + DecodeTime;
}
