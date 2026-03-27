using System.Diagnostics;
using Daisi.Llogos.Cpu;
using Daisi.Llogos.Cuda;
using Daisi.Llogos.Vulkan;
using Daisi.Llogos.Gguf;
using Daisi.Llogos.Inference;
using Daisi.Llogos.Model;
using Daisi.Llogos;
using Daisi.Llogos.Cuda;
using Daisi.Llogos.Inference.DaisiTurbo;
using Daisi.Llogos.Tokenizer;

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
IComputeBackend backend = options.Backend switch
{
    "cuda" => new CudaBackend(),
    "vulkan" => new VulkanBackend(),
    _ => new CpuBackend(),
};

var tokenizer = TokenizerFactory.FromGguf(gguf);
bool isBitNet = config.Architecture.StartsWith("bitnet", StringComparison.OrdinalIgnoreCase);

if (isBitNet)
{
    // ── BitNet path ─────────────────────────────────────────────────────────
    var weights = BitNetModelLoader.Load(gguf, stream, backend, config);
    var kvCache = new BitNetKvCache(backend, config, maxSeqLen: options.MaxContext);
    var forward = new BitNetForwardPass(backend, config, weights, kvCache);

    loadSw.Stop();
    Console.Error.WriteLine($"Model loaded in {loadSw.Elapsed.TotalSeconds:F1}s " +
        $"({config.Architecture}, {config.NumLayers} layers, {config.HiddenDim}d, BitNet)");

    var generator = new BitNetTextGenerator(forward, tokenizer, options.Seed);
    RunGeneration(generator.Generate, options);

    forward.Dispose();
    kvCache.Dispose();
    weights.Dispose();
}
else
{
    // ── Standard path (Qwen / hybrid) ───────────────────────────────────────
    // Build vocab remapper if partial vocab is active (vocab-limit > 1)
    // Build vocab remapper if partial vocab is active (vocab-limit > 1)
    // Same-family models (Qwen3.5) have identical vocabularies, so same remapper works for both
    VocabRemapper? remapper = null;
    // Disable remapper for speculative decoding (shared ID space) and layer offloading
    // (offload uses ArgMaxVocabLimit for partial logits without weight remapping)
    int vocabDivisor = (options.DraftModelPath != null || options.GpuLayers > 0) ? 1 : (options.VocabLimit ?? 32);
    if (vocabDivisor > 1)
    {
        var tokens = gguf.GetMetadata<string[]>("tokenizer.ggml.tokens")!;
        remapper = new VocabRemapper(tokens);
        tokenizer.Vocabulary.ApplyRemapper(remapper);
    }

    ModelWeights weights;
    if (options.GpuLayers > 0 && backend is CudaBackend cudaBackendForOffload)
    {
        // Layer offload doesn't support vocab remapping — load without remapper.
        // Vocab divisor forced to 1 (full vocab logits).
        weights = CudaLayerOffload.LoadWithOffload(gguf, options.ModelPath,
            cudaBackendForOffload, config, options.GpuLayers);
        vocabDivisor = 1;
    }
    else if (options.UseMmap)
        weights = MmapModelLoader.Load(gguf, options.ModelPath, backend, config, remapper);
    else
        weights = ModelLoader.Load(gguf, stream, backend, config);

    var strategy = AttentionStrategy.Parse(options.Attention);
    int maxContext = strategy.Mode != AttentionMode.Full && strategy.CacheCapacity > 0
        ? strategy.CacheCapacity
        : options.MaxContext;
    IKvCache kvCache;
    if (options.KvQuant != null)
    {
        var turboConfig = TurboQuantConfig.Parse(options.KvQuant);
        if (backend is CudaBackend cudaBackend)
            kvCache = new CudaTurboQuantKvCache(cudaBackend, config, maxSeqLen: maxContext,
                turboConfig: turboConfig, strategy: strategy);
        else
            kvCache = new TurboQuantKvCache(backend, config, maxSeqLen: maxContext,
                turboConfig: turboConfig, strategy: strategy);
        Console.Error.WriteLine($"  LLogos Turbo: {turboConfig.EffectiveBitsPerDim(config.KeyLength):F1} bits/dim " +
            $"(q{turboConfig.QuantBits}" +
            $"{(turboConfig.QjlProjectionDim is > 0 ? $"+qjl{turboConfig.QjlProjectionDim}" : turboConfig.QjlProjectionDim == 0 ? "+noqjl" : "+qjl")}" +
            $", {(backend is CudaBackend ? "CUDA" : "CPU")})");
    }
    else if (options.Paged)
        kvCache = new PagedKvCache(backend, config, maxSeqLen: maxContext, strategy: strategy,
            vramPageBudget: options.OffloadPages);
    else
        kvCache = new KvCache(backend, config, maxSeqLen: maxContext, strategy: strategy);
    var deltaState = new DeltaNetState(backend, config, weights);
    var forward = new ForwardPass(backend, config, weights, kvCache, deltaState);
    // For offloading: use partial vocab logits without remapping (ArgMaxVocabLimit
    // just computes fewer columns of LM head — doesn't need remapped weights)
    int argMaxDivisor = options.GpuLayers > 0 ? (options.VocabLimit ?? 32) : vocabDivisor;
    forward.ArgMaxVocabLimit = config.VocabSize / argMaxDivisor;

    // Early exit profiling: measure at which layer the token prediction stabilizes
    if (options.ProfileEarlyExit)
    {
        forward.EarlyExitProfile = true;
        forward.EarlyExitTokens = new int[config.NumLayers];
        Array.Fill(forward.EarlyExitTokens, -1);
    }

    loadSw.Stop();
    var attnInfo = strategy.Mode switch
    {
        AttentionMode.Window => $", window:{strategy.WindowSize}",
        AttentionMode.Sinks => $", sinks:{strategy.SinkTokens},{strategy.WindowSize}",
        _ => ""
    };
    var pagedInfo = options.Paged ? $", paged{(options.OffloadPages > 0 ? $" offload>{options.OffloadPages}" : "")}" : "";
    Console.Error.WriteLine($"Model loaded in {loadSw.Elapsed.TotalSeconds:F1}s " +
        $"({config.Architecture}, {config.NumLayers} layers, {config.HiddenDim}d" +
        $"{(options.UseMmap ? ", mmap" : "")}{attnInfo}{pagedInfo})");

    // ── Speculative decoding (optional draft model) ─────────────────────────
    ForwardPass? draftForward = null;
    SpeculativeDecoder? specDecoder = null;
    if (options.DraftModelPath != null)
    {
        Console.Error.Write($"Loading draft model: {options.DraftModelPath}... ");
        using var draftStream = File.OpenRead(options.DraftModelPath);
        var draftGguf = GgufFile.Read(draftStream);
        var draftConfig = ModelConfig.FromGguf(draftGguf);

        // Draft uses NO remapper — its token IDs are in the original space.
        // The SpeculativeDecoder translates between remapped (target) and original (draft) IDs.
        ModelWeights draftWeights;
        if (options.UseMmap)
            draftWeights = MmapModelLoader.Load(draftGguf, options.DraftModelPath, backend, draftConfig, null);
        else
            draftWeights = ModelLoader.Load(draftGguf, draftStream, backend, draftConfig);

        var draftKvCache = new KvCache(backend, draftConfig, maxSeqLen: options.MaxContext);
        var draftDeltaState = new DeltaNetState(backend, draftConfig, draftWeights);
        draftForward = new ForwardPass(backend, draftConfig, draftWeights, draftKvCache, draftDeltaState);
        // Draft uses partial vocab (same raw token order as target since both un-remapped)
        // /4 gives ~62K tokens — generous enough for common tokens without remapper
        draftForward.ArgMaxVocabLimit = draftConfig.VocabSize / 4;

        specDecoder = new SpeculativeDecoder(forward, draftForward, tokenizer, options.SpecDepth, remapper)
        {
            BatchedVerify = options.BatchedVerify
        };
        Console.Error.WriteLine($"done ({draftConfig.Architecture}, {draftConfig.NumLayers}L, {draftConfig.HiddenDim}d)");
    }

    // Use OffloadForwardPass when layer offloading is active.
    // REQUIRED: pinned memory DevicePtr doesn't work with fused matmul kernels.
    // OffloadForwardPass DMA-copies each offloaded layer to VRAM staging before execution.
    IForwardPass activeForward = forward;
    if (options.GpuLayers > 0 && CudaLayerOffload.Swapper != null)
    {
        activeForward = new OffloadForwardPass(forward, CudaLayerOffload.Swapper, options.GpuLayers);
    }
    var generator = new TextGenerator(activeForward, tokenizer, options.Seed);

    if (options.Bench)
    {
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
        var parameters = new GenerationParams
        {
            MaxTokens = options.MaxTokens,
            Temperature = options.Temperature,
            TopK = options.TopK,
            TopP = options.TopP,
            RepetitionPenalty = options.RepeatPenalty,
            Seed = options.Seed,
        };

        var generateFn = specDecoder != null
            ? specDecoder.Generate(options.Prompt!, parameters)
            : generator.Generate(options.Prompt!, parameters);

        foreach (var token in generateFn)
        {
            if (token.IsDone)
            {
                Console.Error.WriteLine();
                var specInfo = specDecoder != null
                    ? $" | accept: {specDecoder.AcceptanceRate:P0} ({specDecoder.TotalAcceptedTokens}/{specDecoder.TotalDraftTokens})"
                    : "";
                Console.Error.WriteLine($"\n[prefill: {token.PrefillTokens} tokens, {token.PrefillTokensPerSecond:F1} tok/s | " +
                    $"decode: {token.TotalTokens} tokens, {token.TokensPerSecond:F1} tok/s{specInfo}]");
            }
            else
            {
                Console.Write(token.Text);
            }
        }
    }

    // Print early exit profiling results
    if (options.ProfileEarlyExit && forward.EarlyExitTokens != null)
    {
        Console.Error.WriteLine("\n[Early Exit Profile — token predicted at each layer]");
        int finalToken = forward.EarlyExitTokens[config.NumLayers - 1];
        int firstStableLayer = -1;
        for (int i = config.NumLayers / 4; i < config.NumLayers; i++)
        {
            int tok = forward.EarlyExitTokens[i];
            if (tok < 0) continue;
            string tokStr = tok < tokenizer.Vocabulary.Count ? tokenizer.Vocabulary.IdToToken(tok) : $"<{tok}>";
            bool isFinal = tok == finalToken;
            Console.Error.Write($"  L{i}: {tok}({tokStr}){(isFinal ? " ✓" : " ✗")}");
            if (isFinal && firstStableLayer < 0) firstStableLayer = i;
            if ((i + 1) % 4 == 0) Console.Error.WriteLine();
        }
        if (firstStableLayer >= 0)
            Console.Error.WriteLine($"\n  → Token stabilizes at layer {firstStableLayer}/{config.NumLayers} ({100 * firstStableLayer / config.NumLayers}% through)");
        Console.Error.WriteLine();
    }

    // Print LLogos Turbo compression stats
    TurboQuantStats? turboStats = kvCache switch
    {
        TurboQuantKvCache tq when tq.Length > 0 => tq.GetStats(),
        CudaTurboQuantKvCache ctq when ctq.Length > 0 => ctq.GetStats(),
        _ => null
    };
    if (turboStats is { } stats)
    {
        Console.Error.WriteLine($"\n[LLogos Turbo KV Cache]");
        Console.Error.WriteLine($"  Compressed:   {stats.CompressedBytes / 1024.0:F1} KB");
        Console.Error.WriteLine($"  Uncompressed: {stats.UncompressedBytes / 1024.0:F1} KB");
        Console.Error.WriteLine($"  Ratio:        {stats.CompressionRatio:F1}x ({stats.EffectiveBitsPerDim:F1} bits/dim)");
        Console.Error.WriteLine($"  Layers:       {stats.NumLayers}, Seq length: {stats.SeqLength}");
    }

    draftForward?.Dispose();
    forward.Dispose();
    deltaState.Dispose();
    kvCache.Dispose();
    weights.Dispose();
}

backend.Dispose();

return 0;

// ── Shared generation / bench logic ──────────────────────────────────────────

static void RunGeneration(
    Func<string, GenerationParams, IEnumerable<GenerationToken>> generateFn,
    CliArgs options)
{
    if (options.Bench)
    {
        string benchPrompt = options.Prompt ?? "The meaning of life is";
        Console.Error.WriteLine($"Benchmarking with prompt: \"{benchPrompt}\"");
        Console.Error.WriteLine($"Max tokens: {options.MaxTokens}");
        Console.Error.WriteLine();

        var parameters = new GenerationParams
        {
            MaxTokens = options.MaxTokens,
            Temperature = 0.7f,
            TopK = 40,
            TopP = 0.9f,
        };
        var sw = Stopwatch.StartNew();
        int totalTokens = 0;
        foreach (var token in generateFn(benchPrompt, parameters))
        {
            if (token.IsDone)
            {
                Console.Error.WriteLine("=== Benchmark Results ===");
                Console.Error.WriteLine($"  Prefill:  {token.PrefillTokens,6} tokens ({token.PrefillTokensPerSecond,8:F1} tok/s)");
                Console.Error.WriteLine($"  Decode:   {token.TotalTokens,6} tokens ({token.TokensPerSecond,8:F1} tok/s)");
                Console.Error.WriteLine($"  Total:    {sw.Elapsed.TotalMilliseconds,8:F1} ms");
            }
            totalTokens++;
        }
    }
    else
    {
        var parameters = new GenerationParams
        {
            MaxTokens = options.MaxTokens,
            Temperature = options.Temperature,
            TopK = options.TopK,
            TopP = options.TopP,
            RepetitionPenalty = options.RepeatPenalty,
            Seed = options.Seed,
        };

        foreach (var token in generateFn(options.Prompt!, parameters))
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
}

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
            case "--vocab-limit":
                result.VocabLimit = int.Parse(NextArg(args, ref i));
                break;
            case "--profile-early-exit":
                result.ProfileEarlyExit = true;
                break;
            case "--paged":
                result.Paged = true;
                break;
            case "--offload-pages":
                result.Paged = true;
                result.OffloadPages = int.Parse(NextArg(args, ref i));
                break;
            case "--draft":
                result.DraftModelPath = NextArg(args, ref i);
                break;
            case "--spec-depth":
                result.SpecDepth = int.Parse(NextArg(args, ref i));
                break;
            case "--batched-verify":
                result.BatchedVerify = true;
                break;
            case "--kv-quant":
                result.KvQuant = NextArg(args, ref i);
                break;
            case "--gpu-layers":
                result.GpuLayers = int.Parse(NextArg(args, ref i));
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
        daisi-llogos - C# LLM inference engine

        Usage: daisi-llogos --model <path> --prompt <text> [options]

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
          --backend, -b <name>     Compute backend: cpu, cuda, or vulkan (default: cpu)
          --bench                  Run benchmark (prefill + decode timing)
          --vocab-limit <n>        Vocab divisor for greedy argmax (1=full, 32=3%, default: 32)
          --no-mmap                Disable memory-mapped loading (use stream loading)
          --attention <mode>       Attention strategy: full, window:<N>, sinks:<S>,<W> (default: full)
          --paged                  Use paged KV cache (dynamic allocation, grows with context)
          --offload-pages <n>      Enable RAM offloading: keep first N pages in VRAM, rest in RAM
          --draft <path>           Draft model for speculative decoding (smaller, same family)
          --spec-depth <n>         Speculation depth (default: 5)
          --batched-verify         Use batched verify (faster, higher acceptance, different FP from native)
          --kv-quant <mode>        KV cache compression: turbo, turbo:3, turbo:4, turbo:3+qjl32, turbo:3+noqjl
          --gpu-layers <n>         Keep first N layers in VRAM, offload rest to pinned RAM (PCIe)
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
    public bool Paged;
    public int OffloadPages;
    public int? VocabLimit;
    public bool ProfileEarlyExit;
    public string? DraftModelPath;
    public int SpecDepth = 5;
    public bool BatchedVerify;
    public string? KvQuant;
    public int GpuLayers;
}
