using Daisi.Llogos.Gguf;
using Daisi.Llogos.Inference;
using Daisi.Llogos.Inference.DaisiTurbo;
using Daisi.Llogos.Model;
using Daisi.Llogos.Tokenizer;

namespace Daisi.Llogos.Chat;

/// <summary>
/// Handle to a loaded GGUF model with all resources needed for inference.
/// Owns the compute backend, weights, KV cache, forward pass, and tokenizer.
/// Multiple chat sessions can share a model handle (each gets its own ForwardPass + KV cache).
/// </summary>
public sealed class DaisiLlogosModelHandle : IDisposable
{
    public string ModelId { get; }
    public string FilePath { get; }
    public bool IsLoaded { get; private set; }

    public IComputeBackend Backend { get; }
    public ModelConfig Config { get; }
    public ModelWeights Weights { get; }
    public BpeTokenizer Tokenizer { get; }
    public ChatTemplate ChatTemplate { get; }
    public GgufFile Gguf { get; }

    /// <summary>
    /// Create a tool formatter that uses the model-native preamble for tool definitions.
    /// </summary>
    public IToolFormatter CreateToolFormatter() => new DefaultToolFormatter(ChatTemplate.Format);

    private readonly int _contextSize;
    private readonly int? _seed;

    /// <summary>The context size this model was loaded with.</summary>
    public int ContextSize => _contextSize;

    /// <summary>
    /// Optional TurboQuant KV compression config. Set after loading to enable
    /// compressed KV cache for all new sessions created from this handle.
    /// </summary>
    public TurboQuantConfig? TurboConfig { get; set; }

    /// <summary>
    /// Optional offload forward pass (wraps standard forward with layer swapping).
    /// Set by the backend when --gpu-layers is used.
    /// </summary>
    public IForwardPass? OffloadForwardFactory { get; set; }

    /// <summary>Number of layers kept in VRAM (0 = all). Set by backend.</summary>
    public int GpuLayerCount { get; set; }

    internal DaisiLlogosModelHandle(
        string modelId,
        string filePath,
        IComputeBackend backend,
        ModelConfig config,
        ModelWeights weights,
        BpeTokenizer tokenizer,
        ChatTemplate chatTemplate,
        GgufFile gguf,
        int contextSize,
        int? seed)
    {
        ModelId = modelId;
        FilePath = filePath;
        Backend = backend;
        Config = config;
        Weights = weights;
        Tokenizer = tokenizer;
        ChatTemplate = chatTemplate;
        Gguf = gguf;
        IsLoaded = true;
        _contextSize = contextSize;
        _seed = seed;
    }

    /// <summary>
    /// Create a new forward pass + KV cache for a new session using the default context size.
    /// Detects BitNet architecture and creates the appropriate types.
    /// Each session gets its own inference state while sharing the model weights.
    /// </summary>
    public (IForwardPass forward, IKvCache kvCache) CreateInferenceResources()
        => CreateInferenceResources(_contextSize);

    /// <summary>
    /// Create a new forward pass + KV cache with a custom context size.
    /// Use this for minion sessions that need smaller context windows to fit more sessions in VRAM.
    /// </summary>
    public (IForwardPass forward, IKvCache kvCache) CreateInferenceResources(int contextSize)
    {
        if (Config.IsBitNet)
        {
            var kvCache = new BitNetKvCache(Backend, Config, contextSize);
            var forward = new BitNetForwardPass(Backend, Config, Weights, kvCache);
            return (forward, kvCache);
        }
        else
        {
            IKvCache kvCache;
            if (TurboConfig != null)
            {
                // Use CPU TurboQuant — CUDA version has dispose threading issues
                // TODO: switch to CudaTurboQuantKvCache when CUDA dispose is fixed
                kvCache = new TurboQuantKvCache(Backend, Config, contextSize, TurboConfig);
            }
            else
            {
                kvCache = new KvCache(Backend, Config, contextSize);
            }
            var deltaState = new DeltaNetState(Backend, Config, Weights);
            IForwardPass forward = new ForwardPass(Backend, Config, Weights, kvCache, deltaState);

            // Wrap with offload forward pass if GPU layer offloading is active
            if (GpuLayerCount > 0)
            {
                forward = TryWrapWithOffload(forward) ?? forward;
            }

            return (forward, kvCache);
        }
    }

    /// <summary>
    /// Try to wrap a forward pass with OffloadForwardPass via reflection.
    /// Returns null if offload classes aren't available or swapper isn't initialized.
    /// </summary>
    private IForwardPass? TryWrapWithOffload(IForwardPass inner)
    {
        try
        {
            // Get the static Swapper from CudaLayerOffload
            var offloadType = Type.GetType("Daisi.Llogos.Cuda.CudaLayerOffload, Daisi.Llogos.Cuda");
            var swapperProp = offloadType?.GetProperty("Swapper",
                System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Static);
            var swapper = swapperProp?.GetValue(null);
            if (swapper == null) return null;

            // new OffloadForwardPass(inner, swapper, gpuLayers)
            var offloadFpType = Type.GetType("Daisi.Llogos.Cuda.OffloadForwardPass, Daisi.Llogos.Cuda");
            if (offloadFpType == null) return null;

            return (IForwardPass?)Activator.CreateInstance(offloadFpType, inner, swapper, GpuLayerCount);
        }
        catch { return null; }
    }

    /// <summary>
    /// Create a chat session using this model handle with the default context size.
    /// Optionally provide a custom chat renderer to override the default template-based rendering.
    /// </summary>
    public DaisiLlogosChatSession CreateChatSession(string? systemPrompt = null, IChatRenderer? customRenderer = null)
        => CreateChatSession(_contextSize, systemPrompt, customRenderer);

    /// <summary>
    /// Create a chat session with a custom context size.
    /// Use this for minion sessions that need smaller context windows.
    /// </summary>
    public DaisiLlogosChatSession CreateChatSession(int contextSize, string? systemPrompt = null, IChatRenderer? customRenderer = null)
    {
        var (forward, _) = CreateInferenceResources(contextSize);
        var renderer = customRenderer ?? new ChatTemplateRenderer(ChatTemplate);
        var stopSequences = renderer.GetStopSequences();

        var session = new DaisiLlogosChatSession(forward, Tokenizer, renderer, stopSequences, _seed);

        if (!string.IsNullOrEmpty(systemPrompt))
            session.AddMessage(new ChatMessage("system", systemPrompt));

        return session;
    }

    public void Dispose()
    {
        if (!IsLoaded) return;
        IsLoaded = false;
        Weights.Dispose();
        Backend.Dispose();
    }
}
