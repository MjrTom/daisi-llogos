using Daisi.Llogos.Gguf;
using Daisi.Llogos.Inference;
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

    private readonly int _contextSize;
    private readonly int? _seed;

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
    /// Create a new ForwardPass + KV cache + DeltaNetState for a new session.
    /// Each session gets its own inference state while sharing the model weights.
    /// </summary>
    internal (ForwardPass forward, KvCache kvCache, DeltaNetState deltaState) CreateInferenceResources()
    {
        var kvCache = new KvCache(Backend, Config, _contextSize);
        var deltaState = new DeltaNetState(Backend, Config);
        var forward = new ForwardPass(Backend, Config, Weights, kvCache, deltaState);
        return (forward, kvCache, deltaState);
    }

    /// <summary>
    /// Create a chat session using this model handle.
    /// </summary>
    public DaisiLlogosChatSession CreateChatSession(string? systemPrompt = null)
    {
        var (forward, _, _) = CreateInferenceResources();
        var renderer = new ChatTemplateRenderer(ChatTemplate);
        var stopSequences = ChatTemplate.GetStopSequences();

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
