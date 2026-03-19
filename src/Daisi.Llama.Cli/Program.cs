using System.Diagnostics;
using Daisi.Llama.Cpu;
using Daisi.Llama.Cuda;
using Daisi.Llama.Gguf;
using Daisi.Llama.Inference;
using Daisi.Llama.Model;
using Daisi.Llama;
using Daisi.Llama.Tokenizer;

// Parse arguments
var options = ParseArgs(args);

if (options.ShowHelp || options.ModelPath == null || (options.Prompt == null && !options.Bench))
{
    PrintUsage();
    return options.ShowHelp ? 0 : 1;
}

if (!File.Exists(options.ModelPath))
{
    Console.Error.WriteLine($"Error: model file not found: {options.ModelPath}");
    return 1;
}

// Load model
Console.Error.WriteLine($"Loading model: {options.ModelPath}");
var loadSw = Stopwatch.StartNew();

using var stream = File.OpenRead(options.ModelPath);
var gguf = GgufFile.Read(stream);
var config = ModelConfig.FromGguf(gguf);
IComputeBackend backend = options.Backend == "cuda"
    ? new CudaBackend()
    : new CpuBackend();

// Use memory-mapped loading if --mmap is set (default: true)
ModelWeights weights;
if (options.UseMmap)
    weights = MmapModelLoader.Load(gguf, options.ModelPath, backend, config);
else
    weights = ModelLoader.Load(gguf, stream, backend, config);

var strategy = AttentionStrategy.Parse(options.Attention);
int maxContext = strategy.Mode != AttentionMode.Full && strategy.CacheCapacity > 0
    ? strategy.CacheCapacity
    : options.MaxContext;
var kvCache = new KvCache(backend, config, maxSeqLen: maxContext, strategy: strategy);
var deltaState = new DeltaNetState(backend, config);
var forward = new ForwardPass(backend, config, weights, kvCache, deltaState);
var tokenizer = TokenizerFactory.FromGguf(gguf);

loadSw.Stop();
var attnInfo = strategy.Mode switch
{
    AttentionMode.Window => $", window:{strategy.WindowSize}",
    AttentionMode.Sinks => $", sinks:{strategy.SinkTokens},{strategy.WindowSize}",
    _ => ""
};
Console.Error.WriteLine($"Model loaded in {loadSw.Elapsed.TotalSeconds:F1}s " +
    $"({config.Architecture}, {config.NumLayers} layers, {config.HiddenDim}d" +
    $"{(options.UseMmap ? ", mmap" : "")}{attnInfo})");

var generator = new TextGenerator(forward, tokenizer, options.Seed);

if (options.Bench)
{
    // Benchmark mode
    string benchPrompt = options.Prompt ?? "The meaning of life is";
    Console.Error.WriteLine($"Benchmarking with prompt: \"{benchPrompt}\"");
    Console.Error.WriteLine($"Backend: {backend.Name}, Max tokens: {options.MaxTokens}");
    Console.Error.WriteLine();

    var result = generator.Benchmark(benchPrompt, options.MaxTokens);

    Console.Error.WriteLine("=== Benchmark Results ===");
    Console.Error.WriteLine($"  Prefill:  {result.PromptTokens,6} tokens in {result.PrefillTime.TotalMilliseconds,8:F1} ms  ({result.PrefillTokPerSec,8:F1} tok/s)");
    Console.Error.WriteLine($"  Decode:   {result.GeneratedTokens,6} tokens in {result.DecodeTime.TotalMilliseconds,8:F1} ms  ({result.DecodeTokPerSec,8:F1} tok/s)");
    Console.Error.WriteLine($"  Total:    {result.PromptTokens + result.GeneratedTokens,6} tokens in {result.TotalTime.TotalMilliseconds,8:F1} ms");
    Console.Error.WriteLine($"  Load:     {loadSw.Elapsed.TotalMilliseconds,8:F1} ms");
}
else
{
    // Generate mode
    var parameters = new GenerationParams
    {
        MaxTokens = options.MaxTokens,
        Temperature = options.Temperature,
        TopK = options.TopK,
        TopP = options.TopP,
        RepetitionPenalty = options.RepeatPenalty,
        Seed = options.Seed,
    };

    foreach (var token in generator.Generate(options.Prompt!, parameters))
    {
        if (token.IsDone)
        {
            Console.Error.WriteLine();
            Console.Error.WriteLine($"\n[prefill: {token.PrefillTokens} tokens, {token.PrefillTokensPerSecond:F1} tok/s | " +
                $"decode: {token.TotalTokens} tokens, {token.TokensPerSecond:F1} tok/s]");
        }
        else
        {
            Console.Write(token.Text);
        }
    }
}

// Cleanup
forward.Dispose();
deltaState.Dispose();
kvCache.Dispose();
weights.Dispose();
backend.Dispose();

return 0;

// ── Argument parsing ─────────────────────────────────────────────────────────

static CliArgs ParseArgs(string[] args)
{
    var result = new CliArgs();
    for (int i = 0; i < args.Length; i++)
    {
        switch (args[i])
        {
            case "--model" or "-m":
                result.ModelPath = NextArg(args, ref i);
                break;
            case "--prompt" or "-p":
                result.Prompt = NextArg(args, ref i);
                break;
            case "--max-tokens" or "-n":
                result.MaxTokens = int.Parse(NextArg(args, ref i));
                break;
            case "--max-context":
                result.MaxContext = int.Parse(NextArg(args, ref i));
                break;
            case "--temperature" or "-t":
                result.Temperature = float.Parse(NextArg(args, ref i));
                break;
            case "--top-k":
                result.TopK = int.Parse(NextArg(args, ref i));
                break;
            case "--top-p":
                result.TopP = float.Parse(NextArg(args, ref i));
                break;
            case "--repeat-penalty":
                result.RepeatPenalty = float.Parse(NextArg(args, ref i));
                break;
            case "--seed":
                result.Seed = int.Parse(NextArg(args, ref i));
                break;
            case "--backend" or "-b":
                result.Backend = NextArg(args, ref i);
                break;
            case "--bench":
                result.Bench = true;
                break;
            case "--no-mmap":
                result.UseMmap = false;
                break;
            case "--attention":
                result.Attention = NextArg(args, ref i);
                break;
            case "--help" or "-h":
                result.ShowHelp = true;
                break;
        }
    }
    return result;
}

static string NextArg(string[] args, ref int i) =>
    ++i < args.Length ? args[i] : throw new ArgumentException($"Missing value for {args[i - 1]}");

static void PrintUsage()
{
    Console.Error.WriteLine("""
        daisi-llama - C# LLM inference engine

        Usage: daisi-llama --model <path> --prompt <text> [options]

        Options:
          --model, -m <path>       Path to GGUF model file (required)
          --prompt, -p <text>      Input prompt (required for generate, optional for bench)
          --max-tokens, -n <n>     Maximum tokens to generate (default: 256)
          --max-context <n>        Maximum context length (default: 2048)
          --temperature, -t <f>    Sampling temperature, 0=greedy (default: 0.7)
          --top-k <n>              Top-k sampling, 0=disabled (default: 40)
          --top-p <f>              Top-p nucleus sampling (default: 0.9)
          --repeat-penalty <f>     Repetition penalty (default: 1.1)
          --seed <n>               Random seed for reproducibility
          --backend, -b <name>     Compute backend: cpu or cuda (default: cpu)
          --bench                  Run benchmark (prefill + decode timing)
          --no-mmap                Disable memory-mapped loading (use stream loading)
          --attention <mode>       Attention strategy: full, window:<N>, sinks:<S>,<W> (default: full)
          --help, -h               Show this help
        """);
}

class CliArgs
{
    public string? ModelPath;
    public string? Prompt;
    public int MaxTokens = 256;
    public int MaxContext = 2048;
    public float Temperature = 0.7f;
    public int TopK = 40;
    public float TopP = 0.9f;
    public float RepeatPenalty = 1.1f;
    public int? Seed;
    public string Backend = "cpu";
    public bool ShowHelp;
    public bool Bench;
    public bool UseMmap = true;
    public string Attention = "full";
}
