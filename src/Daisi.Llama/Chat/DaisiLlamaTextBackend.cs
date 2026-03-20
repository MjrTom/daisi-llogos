using System.Runtime.CompilerServices;
using Daisi.Inference;
using Daisi.Inference.Interfaces;
using Daisi.Inference.Models;
using Daisi.Llama.Gguf;
using Daisi.Llama.Model;
using Daisi.Llama.Tokenizer;

namespace Daisi.Llama.Chat;

/// <summary>
/// ITextInferenceBackend implementation backed by daisi-llama.
/// Entry point for Host.Core integration — loads GGUF models and creates chat sessions.
/// </summary>
public sealed class DaisiLlamaTextBackend : ITextInferenceBackend
{
    public string BackendName => "DaisiLlama";

    private Func<IComputeBackend>? _backendFactory;

    /// <summary>
    /// Callback for diagnostic messages during backend detection and model loading.
    /// </summary>
    public Action<string>? OnLog { get; set; }

    /// <summary>
    /// Configure the backend. The runtime string selects the compute backend (cpu/cuda/vulkan).
    /// </summary>
    public Task ConfigureAsync(BackendConfiguration config)
    {
        _backendFactory = ResolveBackendFactory(config.Runtime);
        return Task.CompletedTask;
    }

    /// <summary>
    /// Load a GGUF model from disk and return a handle.
    /// </summary>
    public Task<IModelHandle> LoadModelAsync(ModelLoadRequest request)
    {
        var factory = _backendFactory ?? ResolveBackendFactory("Auto");
        var backend = factory();
        OnLog?.Invoke($"Using backend: {backend.Name}");

        using var stream = File.OpenRead(request.FilePath);
        var gguf = GgufFile.Read(stream);
        var config = ModelConfig.FromGguf(gguf);
        var tokenizer = TokenizerFactory.FromGguf(gguf);
        var chatTemplate = ChatTemplate.FromGguf(gguf);
        var weights = MmapModelLoader.Load(gguf, request.FilePath, backend, config);

        var contextSize = request.ContextSize > 0
            ? (int)Math.Min(request.ContextSize, config.MaxContext)
            : config.MaxContext;

        var handle = new DaisiLlamaModelHandle(
            request.ModelId,
            request.FilePath,
            backend,
            config,
            weights,
            tokenizer,
            chatTemplate,
            gguf,
            contextSize,
            seed: null);

        return Task.FromResult<IModelHandle>(new DaisiLlamaModelHandleAdapter(handle));
    }

    /// <summary>
    /// Unload a previously loaded model, freeing all resources.
    /// </summary>
    public void UnloadModel(IModelHandle handle)
    {
        if (handle is DaisiLlamaModelHandleAdapter adapter)
            adapter.Inner.Dispose();
    }

    /// <summary>
    /// Create a chat session with optional system prompt.
    /// </summary>
    public Task<IChatSession> CreateChatSessionAsync(IModelHandle handle, string? systemPrompt = null)
    {
        if (handle is not DaisiLlamaModelHandleAdapter adapter)
            throw new ArgumentException("Handle was not created by this backend.");

        var session = adapter.Inner.CreateChatSession(systemPrompt);
        return Task.FromResult<IChatSession>(new DaisiLlamaChatSessionAdapter(session));
    }

    public ValueTask DisposeAsync()
    {
        return ValueTask.CompletedTask;
    }

    private Func<IComputeBackend> ResolveBackendFactory(string runtime)
    {
        var r = runtime.ToLowerInvariant();
        return r switch
        {
            "cuda" => CreateCudaBackend,
            "vulkan" => CreateVulkanBackend,
            "cpu" or "avx" or "avx2" or "avx512" => CreateCpuBackend,
            _ => AutoDetectBackend(),
        };
    }

    private Func<IComputeBackend> AutoDetectBackend()
    {
        // Try CUDA first, then Vulkan, then CPU
        try
        {
            OnLog?.Invoke("Probing CUDA...");
            var cuda = CreateCudaBackend();
            OnLog?.Invoke($"CUDA available: {cuda.Name}");
            cuda.Dispose();
            return CreateCudaBackend;
        }
        catch (Exception ex)
        {
            OnLog?.Invoke($"CUDA not available: {ex.Message}");
        }

        try
        {
            OnLog?.Invoke("Probing Vulkan...");
            var vulkan = CreateVulkanBackend();
            OnLog?.Invoke($"Vulkan available: {vulkan.Name}");
            vulkan.Dispose();
            return CreateVulkanBackend;
        }
        catch (Exception ex)
        {
            OnLog?.Invoke($"Vulkan not available: {ex.Message}");
        }

        OnLog?.Invoke("Falling back to CPU backend.");
        return CreateCpuBackend;
    }

    private static IComputeBackend CreateCpuBackend()
    {
        var type = Type.GetType("Daisi.Llama.Cpu.CpuBackend, Daisi.Llama.Cpu")
            ?? throw new InvalidOperationException("CPU backend not available. Reference Daisi.Llama.Cpu.");
        return (IComputeBackend)Activator.CreateInstance(type)!;
    }

    private static IComputeBackend CreateCudaBackend()
    {
        var type = Type.GetType("Daisi.Llama.Cuda.CudaBackend, Daisi.Llama.Cuda")
            ?? throw new InvalidOperationException("CUDA backend not available. Reference Daisi.Llama.Cuda.");
        return (IComputeBackend)Activator.CreateInstance(type, 0)!;
    }

    private static IComputeBackend CreateVulkanBackend()
    {
        var type = Type.GetType("Daisi.Llama.Vulkan.VulkanBackend, Daisi.Llama.Vulkan")
            ?? throw new InvalidOperationException("Vulkan backend not available. Reference Daisi.Llama.Vulkan.");
        return (IComputeBackend)Activator.CreateInstance(type, 0)!;
    }
}

/// <summary>
/// Adapts DaisiLlamaModelHandle to the IModelHandle interface.
/// </summary>
public sealed class DaisiLlamaModelHandleAdapter : IModelHandle
{
    public DaisiLlamaModelHandle Inner { get; }

    public string ModelId => Inner.ModelId;
    public string FilePath => Inner.FilePath;
    public bool IsLoaded => Inner.IsLoaded;

    public DaisiLlamaModelHandleAdapter(DaisiLlamaModelHandle inner) => Inner = inner;

    public void Dispose() => Inner.Dispose();
}

/// <summary>
/// Adapts DaisiLlamaChatSession to the IChatSession interface from Daisi.Inference.
/// </summary>
internal sealed class DaisiLlamaChatSessionAdapter : IChatSession
{
    private readonly DaisiLlamaChatSession _inner;

    public DaisiLlamaChatSessionAdapter(DaisiLlamaChatSession inner) => _inner = inner;

    public IReadOnlyList<Daisi.Inference.Models.ChatMessage> History =>
        _inner.History.Select(m => new Daisi.Inference.Models.ChatMessage(
            MapRole(m.Role), m.Content)).ToList();

    public async IAsyncEnumerable<string> ChatAsync(
        Daisi.Inference.Models.ChatMessage message,
        TextGenerationParams parameters,
        [EnumeratorCancellation] CancellationToken ct = default)
    {
        var innerMsg = new ChatMessage(MapRoleToString(message.Role), message.Content);
        var genParams = MapParams(parameters);

        await foreach (var token in _inner.ChatAsync(innerMsg, genParams, ct))
            yield return token;
    }

    public ChatSessionState GetState() => new()
    {
        PastTokensCount = _inner.History.Count,
    };

    public void AddMessage(Daisi.Inference.Models.ChatMessage message)
    {
        _inner.AddMessage(new ChatMessage(MapRoleToString(message.Role), message.Content));
    }

    public void Dispose() => _inner.Dispose();

    private static Inference.GenerationParams MapParams(TextGenerationParams p) => new()
    {
        MaxTokens = p.MaxTokens,
        Temperature = p.Temperature,
        TopK = p.TopK,
        TopP = p.TopP,
        RepetitionPenalty = p.RepeatPenalty,
        Seed = p.Seed > 0 ? (int)p.Seed : null,
    };

    private static ChatRole MapRole(string role) => role switch
    {
        "system" => ChatRole.System,
        "assistant" => ChatRole.Assistant,
        _ => ChatRole.User,
    };

    private static string MapRoleToString(ChatRole role) => role switch
    {
        ChatRole.System => "system",
        ChatRole.Assistant => "assistant",
        ChatRole.User => "user",
        _ => "user",
    };
}
