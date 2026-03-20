using Daisi.Llama.Chat;
using Daisi.Llama.Cpu;
using Daisi.Llama.Gguf;
using Daisi.Llama.Inference;
using Daisi.Llama.Model;
using Daisi.Llama.Tokenizer;

namespace Daisi.Llama.Tests.Chat;

public class DaisiLlamaChatSessionTests : IDisposable
{
    private readonly CpuBackend? _backend;
    private readonly ModelConfig? _config;
    private readonly ModelWeights? _weights;
    private readonly BpeTokenizer? _tokenizer;
    private readonly ChatTemplate? _chatTemplate;

    public DaisiLlamaChatSessionTests()
    {
        if (!TestConstants.ModelExists) return;

        using var stream = File.OpenRead(TestConstants.Qwen35_08B_Q8_0);
        var gguf = GgufFile.Read(stream);
        _config = ModelConfig.FromGguf(gguf);
        _backend = new CpuBackend();
        _weights = MmapModelLoader.Load(gguf, TestConstants.Qwen35_08B_Q8_0, _backend, _config);
        _tokenizer = TokenizerFactory.FromGguf(gguf);
        _chatTemplate = ChatTemplate.FromGguf(gguf);
    }

    private bool Ready => _backend != null;

    private DaisiLlamaChatSession CreateSession(int? seed = 42)
    {
        var kvCache = new KvCache(_backend!, _config!, maxSeqLen: 512);
        var deltaState = new DeltaNetState(_backend!, _config!);
        var forward = new ForwardPass(_backend!, _config!, _weights!, kvCache, deltaState);
        var renderer = new ChatTemplateRenderer(_chatTemplate!);
        var stopSequences = _chatTemplate!.GetStopSequences();
        return new DaisiLlamaChatSession(forward, _tokenizer!, renderer, stopSequences, seed);
    }

    [Fact]
    public void AddMessage_AppearsInHistory()
    {
        if (!Ready) return;
        using var session = CreateSession();

        session.AddMessage(new ChatMessage("system", "You are helpful."));
        session.AddMessage(new ChatMessage("user", "Hello"));

        Assert.Equal(2, session.History.Count);
        Assert.Equal("system", session.History[0].Role);
        Assert.Equal("You are helpful.", session.History[0].Content);
        Assert.Equal("user", session.History[1].Role);
        Assert.Equal("Hello", session.History[1].Content);
    }

    [Fact]
    public async Task ChatAsync_ProducesResponse()
    {
        if (!Ready) return;
        using var session = CreateSession();

        var parameters = new GenerationParams
        {
            MaxTokens = 16,
            Temperature = 0,
        };

        var tokens = new List<string>();
        await foreach (var token in session.ChatAsync(new ChatMessage("user", "Say hello"), parameters))
            tokens.Add(token);

        Assert.NotEmpty(tokens);
        var response = string.Join("", tokens);
        Assert.True(response.Length > 0, "Response should not be empty");
    }

    [Fact]
    public async Task ChatAsync_AddsAssistantToHistory()
    {
        if (!Ready) return;
        using var session = CreateSession();

        var parameters = new GenerationParams { MaxTokens = 8, Temperature = 0 };

        await foreach (var _ in session.ChatAsync(new ChatMessage("user", "Hi"), parameters))
        { }

        // History should have: user message + assistant response
        Assert.Equal(2, session.History.Count);
        Assert.Equal("user", session.History[0].Role);
        Assert.Equal("assistant", session.History[1].Role);
        Assert.True(session.History[1].Content.Length > 0);
    }

    [Fact]
    public async Task ChatAsync_MultiTurn_ReuseKvCache()
    {
        if (!Ready) return;
        using var session = CreateSession();

        var parameters = new GenerationParams { MaxTokens = 8, Temperature = 0 };

        // First turn
        await foreach (var _ in session.ChatAsync(new ChatMessage("user", "What is 2+2?"), parameters))
        { }

        // Second turn — should reuse KV cache (delta prefill)
        var tokens2 = new List<string>();
        await foreach (var token in session.ChatAsync(new ChatMessage("user", "And 3+3?"), parameters))
            tokens2.Add(token);

        Assert.NotEmpty(tokens2);
        // History should have: user1, assistant1, user2, assistant2
        Assert.Equal(4, session.History.Count);
        Assert.Equal("user", session.History[0].Role);
        Assert.Equal("assistant", session.History[1].Role);
        Assert.Equal("user", session.History[2].Role);
        Assert.Equal("assistant", session.History[3].Role);
    }

    [Fact]
    public async Task ChatAsync_SystemPrompt_IncludedInContext()
    {
        if (!Ready) return;
        using var session = CreateSession();

        session.AddMessage(new ChatMessage("system", "You are a calculator. Only respond with numbers."));

        var parameters = new GenerationParams { MaxTokens = 16, Temperature = 0 };

        var tokens = new List<string>();
        await foreach (var token in session.ChatAsync(new ChatMessage("user", "What is 5+5?"), parameters))
            tokens.Add(token);

        Assert.NotEmpty(tokens);
        // System prompt should be in history
        Assert.Equal("system", session.History[0].Role);
    }

    [Fact]
    public async Task ChatAsync_Deterministic_SameSeed()
    {
        if (!Ready) return;

        var parameters = new GenerationParams { MaxTokens = 8, Temperature = 0 };

        // Run twice with same seed
        string response1, response2;

        using (var session1 = CreateSession(seed: 42))
        {
            var tokens = new List<string>();
            await foreach (var t in session1.ChatAsync(new ChatMessage("user", "Hello"), parameters))
                tokens.Add(t);
            response1 = string.Join("", tokens);
        }

        using (var session2 = CreateSession(seed: 42))
        {
            var tokens = new List<string>();
            await foreach (var t in session2.ChatAsync(new ChatMessage("user", "Hello"), parameters))
                tokens.Add(t);
            response2 = string.Join("", tokens);
        }

        Assert.Equal(response1, response2);
    }

    [Fact]
    public async Task ChatAsync_Cancellation_Stops()
    {
        if (!Ready) return;
        using var session = CreateSession();
        using var cts = new CancellationTokenSource();

        var parameters = new GenerationParams { MaxTokens = 100, Temperature = 0.7f };

        int count = 0;
        await Assert.ThrowsAnyAsync<OperationCanceledException>(async () =>
        {
            await foreach (var _ in session.ChatAsync(new ChatMessage("user", "Tell me a long story"), parameters, cts.Token))
            {
                count++;
                if (count >= 2)
                    cts.Cancel();
            }
        });

        Assert.True(count >= 2);
    }

    [Fact]
    public void Dispose_IsIdempotent()
    {
        if (!Ready) return;
        var session = CreateSession();
        session.Dispose();
        session.Dispose(); // should not throw
    }

    public void Dispose()
    {
        _weights?.Dispose();
        _backend?.Dispose();
    }
}
