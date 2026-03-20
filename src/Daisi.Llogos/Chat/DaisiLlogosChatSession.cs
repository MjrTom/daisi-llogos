using System.Runtime.CompilerServices;
using Daisi.Llogos.Inference;
using Daisi.Llogos.Tokenizer;

namespace Daisi.Llogos.Chat;

/// <summary>
/// Multi-turn chat session backed by daisi-llogos's ForwardPass and KV cache.
/// Maintains conversation history and reuses KV cache across turns by only
/// prefilling the delta (new tokens since last turn).
/// </summary>
public sealed class DaisiLlogosChatSession : IDisposable
{
    private readonly ForwardPass _forward;
    private readonly BpeTokenizer _tokenizer;
    private readonly Sampler _sampler;
    private readonly ChatTemplateRenderer _renderer;
    private readonly string[] _stopSequences;
    private readonly List<ChatMessage> _history = [];

    /// <summary>Number of tokens already in KV cache from previous turns.</summary>
    private int _cachedTokenCount;

    /// <summary>Token IDs from the last rendered prompt (for delta computation).</summary>
    private int[] _lastTokenIds = [];

    /// <summary>Buffer for logits to avoid holding ReadOnlySpan across yield.</summary>
    private float[]? _logitsBuffer;

    public IReadOnlyList<ChatMessage> History => _history;

    public DaisiLlogosChatSession(
        ForwardPass forward,
        BpeTokenizer tokenizer,
        ChatTemplateRenderer renderer,
        string[] stopSequences,
        int? seed = null)
    {
        _forward = forward;
        _tokenizer = tokenizer;
        _renderer = renderer;
        _stopSequences = stopSequences;
        _sampler = new Sampler(seed);
    }

    /// <summary>
    /// Add a message to the conversation history without generating a response.
    /// </summary>
    public void AddMessage(ChatMessage message)
    {
        _history.Add(message);
    }

    /// <summary>
    /// Send a message and stream back the assistant's response tokens.
    /// Uses KV cache reuse: only prefills tokens that are new since the last turn.
    /// </summary>
    public async IAsyncEnumerable<string> ChatAsync(
        ChatMessage message,
        GenerationParams parameters,
        [EnumeratorCancellation] CancellationToken ct = default)
    {
        _history.Add(message);

        // Render full conversation with generation prompt
        var prompt = _renderer.Render(_history, addGenerationPrompt: true);
        var allTokenIds = _tokenizer.Encode(prompt);

        // Compute delta: find how many tokens from previous render are reusable
        int commonPrefix = ComputeCommonPrefix(_lastTokenIds, allTokenIds);

        // If the model's KV cache was reset or we diverged, we need to re-prefill from the divergence point
        if (commonPrefix < _cachedTokenCount)
        {
            // KV cache is stale beyond commonPrefix — reset and re-prefill
            _forward.KvCache.Reset();
            _forward.DeltaState.Reset();
            _cachedTokenCount = 0;
            commonPrefix = 0;
        }

        // Prefill new tokens (delta from cached position)
        {
            ReadOnlySpan<float> logits = default;
            for (int i = commonPrefix; i < allTokenIds.Length; i++)
            {
                ct.ThrowIfCancellationRequested();
                logits = _forward.Forward(allTokenIds[i], i);
            }
            // Copy logits to heap buffer so they survive across yield
            if (_logitsBuffer == null || _logitsBuffer.Length != logits.Length)
                _logitsBuffer = new float[logits.Length];
            logits.CopyTo(_logitsBuffer);
        }
        _cachedTokenCount = allTokenIds.Length;

        // Set up stop sequence detection
        var stopDetector = new StopSequenceDetector(_stopSequences);
        var stopTokens = parameters.StopTokens
            ?? (_tokenizer.Vocabulary.EosTokenId >= 0
                ? [_tokenizer.Vocabulary.EosTokenId]
                : []);

        // Decode loop
        var generatedTokens = new List<int>();
        var responseText = new System.Text.StringBuilder();
        int position = _cachedTokenCount;

        for (int t = 0; t < parameters.MaxTokens; t++)
        {
            ct.ThrowIfCancellationRequested();

            var allHistory = new int[allTokenIds.Length + generatedTokens.Count];
            allTokenIds.CopyTo(allHistory, 0);
            generatedTokens.CopyTo(allHistory, allTokenIds.Length);

            int tokenId = _sampler.Sample(_logitsBuffer, parameters, allHistory);

            // Check token-level stop
            if (Array.IndexOf(stopTokens, tokenId) >= 0)
            {
                var flushed = stopDetector.Flush();
                if (flushed.Length > 0)
                {
                    responseText.Append(flushed);
                    yield return flushed;
                }
                break;
            }

            generatedTokens.Add(tokenId);
            string tokenText = _tokenizer.Decode([tokenId]);

            // Check string-level stop sequences
            if (stopDetector.Process(tokenText, out var emittable))
            {
                if (emittable.Length > 0)
                {
                    responseText.Append(emittable);
                    yield return emittable;
                }
                break;
            }

            if (emittable.Length > 0)
            {
                responseText.Append(emittable);
                yield return emittable;
            }

            _forward.Forward(tokenId, position).CopyTo(_logitsBuffer);
            position++;

            // Yield to allow async processing
            await Task.CompletedTask;
        }

        // If loop ended without stop, flush remaining buffer
        if (generatedTokens.Count >= parameters.MaxTokens)
        {
            var flushed = stopDetector.Flush();
            if (flushed.Length > 0)
            {
                responseText.Append(flushed);
                yield return flushed;
            }
        }

        _cachedTokenCount = position;

        // Save the assistant response to history
        _history.Add(new ChatMessage("assistant", responseText.ToString()));

        // Update last token IDs for next turn's delta computation
        var newPrompt = _renderer.Render(_history, addGenerationPrompt: false);
        _lastTokenIds = _tokenizer.Encode(newPrompt);
    }

    private static int ComputeCommonPrefix(int[] a, int[] b)
    {
        int minLen = Math.Min(a.Length, b.Length);
        int common = 0;
        for (int i = 0; i < minLen; i++)
        {
            if (a[i] != b[i]) break;
            common++;
        }
        return common;
    }

    public void Dispose()
    {
        // ForwardPass and its resources are owned by the model handle, not the session
    }
}
