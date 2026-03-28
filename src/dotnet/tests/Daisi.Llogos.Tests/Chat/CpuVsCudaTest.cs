using Daisi.Llogos.Chat;
using Daisi.Llogos.Cpu;
using Daisi.Llogos.Gguf;
using Daisi.Llogos.Inference;
using Daisi.Llogos.Model;
using Daisi.Llogos.Tokenizer;
using System.Text;

namespace Daisi.Llogos.Tests.Chat;

/// <summary>
/// Compare CPU vs CUDA output to isolate whether the DeltaNet CUDA kernel is the issue.
/// Uses the 0.8B model (fast on CPU) with a short prompt.
/// </summary>
public class CpuVsCudaTest : IDisposable
{
    private readonly GgufFile? _gguf;
    private readonly ModelConfig? _config;
    private readonly BpeTokenizer? _tokenizer;
    private readonly ChatTemplate? _chatTemplate;
    private ModelWeights? _cpuWeights;
    private ModelWeights? _cudaWeights;
    private IComputeBackend? _cpuBackend;
    private IComputeBackend? _cudaBackend;

    public CpuVsCudaTest()
    {
        if (!TestConstants.ModelExists) return; // 0.8B model
        using var stream = File.OpenRead(TestConstants.Qwen35_08B_Q8_0);
        _gguf = GgufFile.Read(stream);
        _config = ModelConfig.FromGguf(_gguf);
        _tokenizer = TokenizerFactory.FromGguf(_gguf);
        _chatTemplate = ChatTemplate.FromGguf(_gguf);

        _cpuBackend = new CpuBackend();
        _cpuWeights = MmapModelLoader.Load(_gguf, TestConstants.Qwen35_08B_Q8_0, _cpuBackend, _config);

        try
        {
            var cudaType = Type.GetType("Daisi.Llogos.Cuda.CudaBackend, Daisi.Llogos.Cuda");
            _cudaBackend = cudaType != null ? (IComputeBackend)Activator.CreateInstance(cudaType, 0)! : null;
            if (_cudaBackend != null)
                _cudaWeights = MmapModelLoader.Load(_gguf, TestConstants.Qwen35_08B_Q8_0, _cudaBackend, _config);
        }
        catch { _cudaBackend = null; }
    }

    private bool Ready => _cpuBackend != null && _cudaBackend != null;

    private DaisiLlogosChatSession CreateSession(IComputeBackend backend, ModelWeights weights, string systemPrompt)
    {
        var kvCache = new KvCache(backend, _config!, maxSeqLen: 512);
        var deltaState = new DeltaNetState(backend, _config!, weights);
        var forward = new ForwardPass(backend, _config!, weights, kvCache, deltaState);
        var renderer = new ChatTemplateRenderer(_chatTemplate!);
        var stops = _chatTemplate!.GetStopSequences();
        var session = new DaisiLlogosChatSession(forward, _tokenizer!, renderer, stops, seed: 42);
        session.AddMessage(new ChatMessage("system", systemPrompt));
        return session;
    }

    [Fact]
    public async Task CpuAndCuda_ShouldProduceSameOutput()
    {
        if (!Ready) return;

        var prompt = "Return ONLY valid JSON: {\"greeting\": \"string\"}";
        var userMsg = "Say hello";

        var cpuResult = await Generate(_cpuBackend!, _cpuWeights!, prompt, userMsg);
        var cudaResult = await Generate(_cudaBackend!, _cudaWeights!, prompt, userMsg);

        File.WriteAllText(@"C:\repos\daisinet-qwen-integration-crm\test-cpu-output.txt", cpuResult);
        File.WriteAllText(@"C:\repos\daisinet-qwen-integration-crm\test-cuda-output.txt", cudaResult);

        // They should be identical (same seed, same model, same prompt)
        Assert.Equal(cpuResult, cudaResult);
    }

    private async Task<string> Generate(IComputeBackend backend, ModelWeights weights, string system, string user)
    {
        using var session = CreateSession(backend, weights, system);
        var sb = new StringBuilder();
        await foreach (var token in session.ChatAsync(new ChatMessage("user", user), new GenerationParams
        {
            MaxTokens = 64,
            Temperature = 0,  // Greedy for determinism
        }))
        {
            sb.Append(token);
        }
        return sb.ToString();
    }

    public void Dispose()
    {
        _cpuWeights?.Dispose();
        _cudaWeights?.Dispose();
        _cpuBackend?.Dispose();
        _cudaBackend?.Dispose();
    }
}
