using System.Runtime.CompilerServices;
using Daisi.Inference;
using Daisi.Inference.Interfaces;
using Daisi.Inference.Models;
using Daisi.Llogos.Gguf;
using Daisi.Llogos.Inference;
using Daisi.Llogos.Model;
using Daisi.Llogos.Tokenizer;

namespace Daisi.Llogos.Chat;

/// <summary>
/// ITextInferenceBackend implementation backed by daisi-llogos.
/// Entry point for Host.Core integration — loads GGUF models and creates chat sessions.
/// </summary>
public sealed class DaisiLlogosTextBackend : ITextInferenceBackend
{
    public string BackendName => "DaisiLlogos";

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

        var contextSize = request.ContextSize > 0
            ? (int)Math.Min(request.ContextSize, config.MaxContext)
            : config.MaxContext;

        // Pipeline mode: use shard directory (explicit or auto-detected)
        var shardDir = request.ShardDirectory ?? (request.Pipeline ? request.FilePath + ".shards" : null);
        if (shardDir == null && Directory.Exists(request.FilePath + ".shards"))
            shardDir = request.FilePath + ".shards"; // auto-detect
        if (shardDir != null && Directory.Exists(shardDir))
        {
            OnLog?.Invoke($"Pipeline shards detected: {shardDir}");
            // Create lightweight placeholder weights — pipeline loads its own from shards
            var placeholder = backend.CreateTensor("placeholder", Gguf.GgmlType.F32, [1]);
            var placeholderWeights = new ModelWeights
            {
                TokenEmbedding = placeholder,
                OutputNorm = backend.CreateTensor("onorm.placeholder", Gguf.GgmlType.F32, [1]),
                Output = null,
                Layers = Enumerable.Range(0, config.NumLayers)
                    .Select(_ => CreatePlaceholderLayer(backend)).ToArray(),
            };

            var handle = new DaisiLlogosModelHandle(
                request.ModelId, request.FilePath, backend, config,
                placeholderWeights, tokenizer, chatTemplate, gguf, contextSize, seed: null);
            handle.PipelineShardDir = shardDir;
            return Task.FromResult<IModelHandle>(new DaisiLlogosModelHandleAdapter(handle));
        }

        ModelWeights weights;
        IForwardPass? offloadForward = null;

        if (!config.IsBitNet && request.GpuLayerCount > 0 && request.GpuLayerCount < config.NumLayers)
        {
            (weights, offloadForward) = TryLoadWithOffload(gguf, request, backend, config);
        }
        else
        {
            weights = config.IsBitNet
                ? BitNetModelLoader.Load(gguf, stream, backend, config)
                : MmapModelLoader.Load(gguf, request.FilePath, backend, config);
        }

        var handle2 = new DaisiLlogosModelHandle(
            request.ModelId, request.FilePath, backend, config,
            weights, tokenizer, chatTemplate, gguf, contextSize, seed: null);

        if (offloadForward != null)
            handle2.OffloadForwardFactory = offloadForward;

        if (request.GpuLayerCount > 0)
            OnLog?.Invoke($"GPU layers: {Math.Min(request.GpuLayerCount, config.NumLayers)}/{config.NumLayers}");

        return Task.FromResult<IModelHandle>(new DaisiLlogosModelHandleAdapter(handle2));
    }

    /// <summary>
    /// Try to load model with partial GPU offloading via CudaLayerOffload (reflection).
    /// Falls back to standard loading if CUDA offload classes aren't available.
    /// </summary>
    private (ModelWeights weights, IForwardPass? offloadForward) TryLoadWithOffload(
        GgufFile gguf, ModelLoadRequest request, IComputeBackend backend, ModelConfig config)
    {
        try
        {
            // CudaLayerOffload.LoadWithOffload(gguf, filePath, cudaBackend, config, gpuLayers, remapper)
            var offloadType = Type.GetType("Daisi.Llogos.Cuda.CudaLayerOffload, Daisi.Llogos.Cuda");
            if (offloadType == null)
                throw new InvalidOperationException("CudaLayerOffload not available");

            var loadMethod = offloadType.GetMethod("LoadWithOffload",
                System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Static);
            if (loadMethod == null)
                throw new InvalidOperationException("LoadWithOffload method not found");

            OnLog?.Invoke($"Loading with GPU offload: {request.GpuLayerCount}/{config.NumLayers} layers in VRAM...");

            var weights = (ModelWeights)loadMethod.Invoke(null, [
                gguf, request.FilePath, backend, config, request.GpuLayerCount, null
            ])!;

            // OffloadSwapper is stored statically by CudaLayerOffload.LoadWithOffload
            // OffloadForwardPass wraps the standard forward pass
            var swapperProp = offloadType.GetProperty("Swapper",
                System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Static);
            var swapper = swapperProp?.GetValue(null);

            IForwardPass? offloadForward = null;
            if (swapper != null)
            {
                var offloadFpType = Type.GetType("Daisi.Llogos.Cuda.OffloadForwardPass, Daisi.Llogos.Cuda");
                if (offloadFpType != null)
                {
                    // Store info needed to create OffloadForwardPass later (when creating sessions)
                    // For now, store the gpu layer count on the handle
                }
            }

            return (weights, null);
        }
        catch (Exception ex)
        {
            OnLog?.Invoke($"GPU offload failed: {ex.Message}. Falling back to standard loading.");
            using var stream = File.OpenRead(request.FilePath);
            return (MmapModelLoader.Load(gguf, request.FilePath, backend, config), null);
        }
    }

    /// <summary>
    /// Unload a previously loaded model, freeing all resources.
    /// </summary>
    public void UnloadModel(IModelHandle handle)
    {
        if (handle is DaisiLlogosModelHandleAdapter adapter)
            adapter.Inner.Dispose();
    }

    /// <summary>
    /// Create a chat session with optional system prompt.
    /// </summary>
    public Task<IChatSession> CreateChatSessionAsync(IModelHandle handle, string? systemPrompt = null)
    {
        if (handle is not DaisiLlogosModelHandleAdapter adapter)
            throw new ArgumentException("Handle was not created by this backend.");

        var session = adapter.Inner.CreateChatSession(systemPrompt);
        return Task.FromResult<IChatSession>(new DaisiLlogosChatSessionAdapter(session));
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

    private static Model.StandardAttentionWeights CreatePlaceholderLayer(IComputeBackend backend)
    {
        ITensor P(string n) => backend.CreateTensor(n, Gguf.GgmlType.F32, [1]);
        return new Model.StandardAttentionWeights
        {
            AttnNorm = P("an"), PostAttnNorm = P("pan"),
            AttnQ = P("aq"), AttnK = P("ak"), AttnV = P("av"), AttnO = P("ao"),
            FfnGate = P("fg"), FfnUp = P("fu"), FfnDown = P("fd"),
        };
    }

    private static IComputeBackend CreateCpuBackend()
    {
        var type = Type.GetType("Daisi.Llogos.Cpu.CpuBackend, Daisi.Llogos.Cpu")
            ?? throw new InvalidOperationException("CPU backend not available. Reference Daisi.Llogos.Cpu.");
        return (IComputeBackend)Activator.CreateInstance(type)!;
    }

    private static IComputeBackend CreateCudaBackend()
    {
        var type = Type.GetType("Daisi.Llogos.Cuda.CudaBackend, Daisi.Llogos.Cuda")
            ?? throw new InvalidOperationException("CUDA backend not available. Reference Daisi.Llogos.Cuda.");
        return (IComputeBackend)Activator.CreateInstance(type, 0)!;
    }

    private static IComputeBackend CreateVulkanBackend()
    {
        var type = Type.GetType("Daisi.Llogos.Vulkan.VulkanBackend, Daisi.Llogos.Vulkan")
            ?? throw new InvalidOperationException("Vulkan backend not available. Reference Daisi.Llogos.Vulkan.");
        return (IComputeBackend)Activator.CreateInstance(type, 0)!;
    }
}

/// <summary>
/// Adapts DaisiLlogosModelHandle to the IModelHandle interface.
/// </summary>
public sealed class DaisiLlogosModelHandleAdapter : IModelHandle
{
    public DaisiLlogosModelHandle Inner { get; }

    public string ModelId => Inner.ModelId;
    public string FilePath => Inner.FilePath;
    public bool IsLoaded => Inner.IsLoaded;

    public DaisiLlogosModelHandleAdapter(DaisiLlogosModelHandle inner) => Inner = inner;

    public void Dispose() => Inner.Dispose();
}

/// <summary>
/// Adapts DaisiLlogosChatSession to the IChatSession interface from Daisi.Inference.
/// </summary>
internal sealed class DaisiLlogosChatSessionAdapter : IChatSession
{
    private readonly DaisiLlogosChatSession _inner;

    public DaisiLlogosChatSessionAdapter(DaisiLlogosChatSession inner) => _inner = inner;

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
        FrequencyPenalty = p.FrequencyPenalty,
        PresencePenalty = p.PresencePenalty,
        MinP = p.MinP,
        TypicalP = p.TypicalP,
        PenalizeNewline = p.PenalizeNewline,
        PenaltyCount = p.PenaltyCount,
        MinKeep = p.MinKeep,
        PreventEOS = p.PreventEOS,
        AntiPrompts = p.AntiPrompts?.Count > 0 ? p.AntiPrompts.ToArray() : null,
        GrammarText = p.GrammarText,
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
