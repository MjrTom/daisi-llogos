using Daisi.Llogos.Chat;
using Daisi.Llogos.Gguf;
using Daisi.Llogos.Inference;
using Daisi.Llogos.Model;
using Daisi.Llogos.Tokenizer;
using System.Text;

namespace Daisi.Llogos.Tests.Chat;

/// <summary>
/// Compare TextGenerator (old host path) vs DaisiLlogosChatSession (new host path)
/// using the exact same prompt to identify where output degradation occurs.
/// </summary>
public class TextGeneratorComparisonTest : IDisposable
{
    private readonly IComputeBackend? _backend;
    private readonly ModelConfig? _config;
    private readonly ModelWeights? _weights;
    private readonly BpeTokenizer? _tokenizer;
    private readonly ChatTemplate? _chatTemplate;

    private const string SystemPrompt =
        "You are a business analytics expert. Return ONLY valid JSON. Do NOT abbreviate.\n\n" +
        "Schema: {\"executiveSummary\": \"string\", \"keyMetrics\": [{\"label\": \"string\", \"value\": \"string\", \"trend\": \"up|down|neutral\"}]}";

    private const string UserText = "Report: Revenue: $32,980. Orders: 78. Top Product: Virtual Desk (79 sold, $30,929).";

    public TextGeneratorComparisonTest()
    {
        if (!TestConstants.Model9BExists) return;
        using var stream = File.OpenRead(TestConstants.Qwen35_9B_Q8_0);
        var gguf = GgufFile.Read(stream);
        _config = ModelConfig.FromGguf(gguf);
        _backend = Type.GetType("Daisi.Llogos.Cuda.CudaBackend, Daisi.Llogos.Cuda") is Type t
            ? (IComputeBackend)Activator.CreateInstance(t, 0)! : null;
        if (_backend == null) return;
        _weights = MmapModelLoader.Load(gguf, TestConstants.Qwen35_9B_Q8_0, _backend, _config);
        _tokenizer = TokenizerFactory.FromGguf(gguf);
        _chatTemplate = ChatTemplate.FromGguf(gguf);
    }

    private bool Ready => _backend != null;

    /// <summary>
    /// TextGenerator path (old host path): format prompt manually, generate stateless.
    /// </summary>
    [Fact]
    public void TextGenerator_Output()
    {
        if (!Ready) return;

        var kvCache = new KvCache(_backend!, _config!, maxSeqLen: 4096);
        var deltaState = new DeltaNetState(_backend!, _config!, _weights!);
        var forward = new ForwardPass(_backend!, _config!, _weights!, kvCache, deltaState);
        var generator = new TextGenerator(forward, _tokenizer!);

        // Format prompt like the old adapter did (ChatML)
        var prompt = $"<|im_start|>system\n{SystemPrompt}<|im_end|>\n<|im_start|>user\n{UserText}<|im_end|>\n<|im_start|>assistant\n";

        var sb = new StringBuilder();
        foreach (var token in generator.Generate(prompt, new GenerationParams
        {
            MaxTokens = 512,
            Temperature = 0.4f,
            TopP = 0.9f,
            TopK = 40,
            RepetitionPenalty = 1.1f,
            MinP = 0.1f,
            AntiPrompts = ["<|im_end|>", "</response>"],
        }))
        {
            if (token.IsDone) break;
            sb.Append(token.Text);
        }

        var result = sb.ToString();
        File.WriteAllText(@"C:\repos\daisinet-qwen-integration-crm\test-textgenerator.txt", result);

        Assert.True(result.Length > 10, $"TextGenerator produced too little: {result}");
    }

    /// <summary>
    /// Core ChatSession path (new host path): ChatML template + KV cache + stop sequences.
    /// </summary>
    [Fact]
    public async Task CoreChatSession_Output()
    {
        if (!Ready) return;

        var kvCache = new KvCache(_backend!, _config!, maxSeqLen: 4096);
        var deltaState = new DeltaNetState(_backend!, _config!, _weights!);
        var forward = new ForwardPass(_backend!, _config!, _weights!, kvCache, deltaState);
        var renderer = new ChatTemplateRenderer(_chatTemplate!);
        var stopSequences = _chatTemplate!.GetStopSequences();
        var session = new DaisiLlogosChatSession(forward, _tokenizer!, renderer, stopSequences);
        session.AddMessage(new ChatMessage("system", SystemPrompt));

        var sb = new StringBuilder();
        await foreach (var token in session.ChatAsync(new ChatMessage("user", UserText), new GenerationParams
        {
            MaxTokens = 512,
            Temperature = 0.4f,
            TopP = 0.9f,
            TopK = 40,
            RepetitionPenalty = 1.1f,
            MinP = 0.1f,
            AntiPrompts = ["</response>"],
        }))
        {
            sb.Append(token);
        }

        var result = sb.ToString();
        File.WriteAllText(@"C:\repos\daisinet-qwen-integration-crm\test-coresession.txt", result);

        Assert.True(result.Length > 10, $"CoreSession produced too little: {result}");
    }

    public void Dispose()
    {
        _weights?.Dispose();
        _backend?.Dispose();
    }
}
