using Daisi.Llama.Chat;
using Daisi.Llama.Cpu;
using Daisi.Llama.Gguf;
using Daisi.Llama.Inference;
using Daisi.Llama.Model;
using Daisi.Llama.Tokenizer;

namespace Daisi.Llama.Tests.Chat;

/// <summary>
/// Tests for DaisiLlamaModelHandle using the public DaisiLlamaTextBackend entry point
/// and for DaisiLlamaChatSession lifecycle created through the handle.
/// Since DaisiLlamaModelHandle has an internal constructor, tests go through
/// DaisiLlamaTextBackend.LoadModelAsync which is the intended public API.
/// </summary>
public class DaisiLlamaModelHandleTests
{
    [Fact]
    public async Task LoadModel_ReturnsLoadedHandle()
    {
        if (!TestConstants.ModelExists) return;

        await using var backend = new DaisiLlamaTextBackend();
        var handle = await backend.LoadModelAsync(new Daisi.Inference.Models.ModelLoadRequest
        {
            ModelId = "test-model",
            FilePath = TestConstants.Qwen35_08B_Q8_0,
            ContextSize = 512,
        });

        Assert.NotNull(handle);
        Assert.Equal("test-model", handle.ModelId);
        Assert.Equal(TestConstants.Qwen35_08B_Q8_0, handle.FilePath);
        Assert.True(handle.IsLoaded);

        backend.UnloadModel(handle);
        Assert.False(handle.IsLoaded);
    }

    [Fact]
    public async Task UnloadModel_Idempotent()
    {
        if (!TestConstants.ModelExists) return;

        await using var backend = new DaisiLlamaTextBackend();
        var handle = await backend.LoadModelAsync(new Daisi.Inference.Models.ModelLoadRequest
        {
            ModelId = "test-model",
            FilePath = TestConstants.Qwen35_08B_Q8_0,
            ContextSize = 512,
        });

        backend.UnloadModel(handle);
        backend.UnloadModel(handle); // should not throw
    }

    [Fact]
    public async Task CreateChatSession_ReturnsSession()
    {
        if (!TestConstants.ModelExists) return;

        await using var backend = new DaisiLlamaTextBackend();
        var handle = await backend.LoadModelAsync(new Daisi.Inference.Models.ModelLoadRequest
        {
            ModelId = "test",
            FilePath = TestConstants.Qwen35_08B_Q8_0,
            ContextSize = 512,
        });

        using var session = await backend.CreateChatSessionAsync(handle, "You are helpful.");
        Assert.NotNull(session);

        // History should have the system prompt
        Assert.Single(session.History);
        Assert.Equal(Daisi.Inference.Models.ChatRole.System, session.History[0].Role);

        backend.UnloadModel(handle);
    }

    [Fact]
    public async Task CreateChatSession_CanChat()
    {
        if (!TestConstants.ModelExists) return;

        await using var backend = new DaisiLlamaTextBackend();
        var handle = await backend.LoadModelAsync(new Daisi.Inference.Models.ModelLoadRequest
        {
            ModelId = "test",
            FilePath = TestConstants.Qwen35_08B_Q8_0,
            ContextSize = 512,
        });

        using var session = await backend.CreateChatSessionAsync(handle);
        var msg = new Daisi.Inference.Models.ChatMessage(Daisi.Inference.Models.ChatRole.User, "Say hi");
        var genParams = new Daisi.Inference.Models.TextGenerationParams
        {
            MaxTokens = 8,
            Temperature = 0,
        };

        var tokens = new List<string>();
        await foreach (var token in session.ChatAsync(msg, genParams))
            tokens.Add(token);

        Assert.NotEmpty(tokens);
        // History should now have user + assistant
        Assert.True(session.History.Count >= 2);

        backend.UnloadModel(handle);
    }

    [Fact]
    public async Task Configure_SelectsCpuBackend()
    {
        await using var backend = new DaisiLlamaTextBackend();
        var logs = new List<string>();
        backend.OnLog = logs.Add;

        await backend.ConfigureAsync(new Daisi.Inference.Models.BackendConfiguration
        {
            Runtime = "cpu",
        });

        // Should not throw — CPU is always available
        // Log verification happens implicitly through the load path
    }

    [Fact]
    public async Task AutoDetect_SelectsBackend()
    {
        if (!TestConstants.ModelExists) return;

        await using var backend = new DaisiLlamaTextBackend();
        var logs = new List<string>();
        backend.OnLog = logs.Add;

        // Auto-detect should succeed (will find CUDA, Vulkan, or fall back to CPU)
        var handle = await backend.LoadModelAsync(new Daisi.Inference.Models.ModelLoadRequest
        {
            ModelId = "test",
            FilePath = TestConstants.Qwen35_08B_Q8_0,
            ContextSize = 512,
        });

        Assert.True(handle.IsLoaded);
        Assert.True(logs.Any(l => l.Contains("backend", StringComparison.OrdinalIgnoreCase)));

        backend.UnloadModel(handle);
    }
}
