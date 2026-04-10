using Daisi.Llogos;
using Daisi.Llogos.Cpu;
using Daisi.Llogos.Gguf;
using Daisi.Llogos.Inference;
using Daisi.Llogos.Model;
using Daisi.Llogos.Tokenizer;

namespace Daisi.Llogos.Tests.Model;

public class InspectModelTests
{
    [Fact]
    public void InspectTinyLlama()
    {
        var path = @"C:\GGUFS\tinyllama-1.1b-chat-v1.0.Q8_0.gguf";
        if (!File.Exists(path)) return;

        using var stream = File.OpenRead(path);
        var gguf = GgufFile.Read(stream);

        var lines = new List<string>();
        lines.Add("=== METADATA ===");
        foreach (var kv in gguf.Metadata.OrderBy(x => x.Key))
            lines.Add($"  {kv.Key} = {kv.Value}");

        lines.Add($"\n=== TENSORS ({gguf.Tensors.Count}) ===");
        foreach (var t in gguf.Tensors.Take(30))
            lines.Add($"  {t.Name} [{t.Type}] dims={string.Join("x", t.Dimensions)}");

        File.WriteAllLines(@"C:\GGUFS\tinyllama-inspect.txt", lines);

        var config = ModelConfig.FromGguf(gguf);
        Assert.Equal("llama", config.Architecture);
        Assert.True(config.NumLayers > 0);
    }

    [Fact]
    public void InspectGemma4_Q4_0() => DumpGguf(
        @"C:\GGUFS\gemma-4-E4B-it-Q4_0.gguf",
        @"C:\GGUFS\gemma-4-E4B-it-Q4_0-inspect.txt");

    [Fact]
    public void InspectGemma4_Q8_0() => DumpGguf(
        @"C:\GGUFS\gemma-4-E4B-it-Q8_0.gguf",
        @"C:\GGUFS\gemma-4-E4B-it-Q8_0-inspect.txt");

    /// <summary>
    /// Dump sample float values from a handful of F32 tensors in the Gemma 4 GGUF.
    /// Empirically resolves: (a) does the GGUF pre-bake the Gemma "(1+w)" convention into the
    /// norm weights (values near 1.0) or store raw weights (values near 0.0)?
    /// (b) what does layer_output_scale look like? (c) what's the rope_freqs table?
    /// </summary>
    [Fact]
    public void InspectGemma4_FloatSamples()
    {
        var path = @"C:\GGUFS\gemma-4-E4B-it-Q8_0.gguf";
        if (!File.Exists(path)) return;

        using var stream = File.OpenRead(path);
        var gguf = GgufFile.Read(stream);
        var byName = gguf.Tensors.ToDictionary(t => t.Name);

        // Tensors we want to peek at — chosen to resolve specific implementation questions.
        string[] targets =
        {
            "output_norm.weight",                  // final RmsNorm — convention check
            "blk.0.attn_norm.weight",              // pre-attn RmsNorm
            "blk.0.post_attention_norm.weight",    // post-attn RmsNorm
            "blk.0.ffn_norm.weight",               // pre-FFN RmsNorm
            "blk.0.post_ffw_norm.weight",          // post-FFN RmsNorm
            "blk.0.post_norm.weight",              // per-layer-embed post-norm
            "blk.0.attn_q_norm.weight",            // per-head Q RmsNorm (sliding layer, dim 256)
            "blk.0.attn_k_norm.weight",
            "blk.5.attn_q_norm.weight",            // per-head Q RmsNorm (full attn layer, dim 512)
            "blk.0.layer_output_scale.weight",     // [1] scalar — what value?
            "blk.5.layer_output_scale.weight",
            "blk.41.layer_output_scale.weight",
            "rope_freqs.weight",                   // proportional RoPE freq table for full attn layers
            "per_layer_proj_norm.weight",          // per-layer projection norm
        };

        var lines = new List<string>
        {
            $"=== FILE: {path} ===",
            "",
            "Sample float values from key tensors. Used to verify Gemma's (1+w) RmsNorm",
            "convention is pre-baked into the GGUF (values near 1.0) vs stored raw (values near 0).",
            "",
        };

        foreach (var name in targets)
        {
            if (!byName.TryGetValue(name, out var info))
            {
                lines.Add($"  {name,-45}  <missing>");
                continue;
            }
            if (info.Type != GgmlType.F32)
            {
                lines.Add($"  {name,-45}  [type={info.Type} — skipped, only F32 supported here]");
                continue;
            }

            int elem = (int)info.ElementCount;
            int n = Math.Min(elem, 16);
            var bytes = gguf.ReadTensorData(stream, info);
            var floats = new float[n];
            Buffer.BlockCopy(bytes, 0, floats, 0, n * sizeof(float));

            // Stats over the full tensor (not just the head sample) — useful when n > 16.
            var allFloats = new float[elem];
            Buffer.BlockCopy(bytes, 0, allFloats, 0, elem * sizeof(float));
            float min = float.MaxValue, max = float.MinValue, sum = 0;
            foreach (var v in allFloats)
            {
                if (v < min) min = v;
                if (v > max) max = v;
                sum += v;
            }
            float mean = sum / elem;

            var head = string.Join(", ", floats.Select(v => v.ToString("F5", System.Globalization.CultureInfo.InvariantCulture)));
            lines.Add($"  {name,-45}  dims=[{string.Join("x", info.Dimensions)}]");
            lines.Add($"      mean={mean:F5}  min={min:F5}  max={max:F5}");
            lines.Add($"      head={head}");
            lines.Add("");
        }

        File.WriteAllLines(@"C:\GGUFS\gemma-4-E4B-it-floatsample.txt", lines);
    }

    /// <summary>
    /// Dump full GGUF metadata + every tensor (name, type, dims, byte size) to a text file.
    /// Used to evaluate new architectures (Gemma 4) before adding loader support.
    /// </summary>
    private static void DumpGguf(string ggufPath, string outPath)
    {
        if (!File.Exists(ggufPath)) return;

        using var stream = File.OpenRead(ggufPath);
        var gguf = GgufFile.Read(stream);

        var lines = new List<string>
        {
            $"=== FILE: {ggufPath} ===",
            $"GGUF v{gguf.Header.Version}  metadata_count={gguf.Header.MetadataKvCount}  tensor_count={gguf.Header.TensorCount}",
            $"tensor_data_offset={gguf.TensorDataOffset}",
            "",
            "=== METADATA ===",
        };

        foreach (var kv in gguf.Metadata.OrderBy(x => x.Key))
            lines.Add($"  [{kv.Type,-7}] {kv.Key} = {FormatMetadataValue(kv)}");

        // Group tensors by prefix (token_embd, output, blk.0, blk.1, ...) for readability.
        // Also dump unique top-level prefixes so we can spot per-layer-embedding / multimodal
        // tensors (per_layer_token_embd, vision.*, audio.*, etc.).
        var prefixes = gguf.Tensors
            .Select(t => t.Name.Split('.')[0])
            .Distinct()
            .OrderBy(p => p)
            .ToList();

        lines.Add("");
        lines.Add($"=== TENSOR PREFIXES ({prefixes.Count}) ===");
        foreach (var p in prefixes)
        {
            int n = gguf.Tensors.Count(t => t.Name.StartsWith(p + ".") || t.Name == p);
            lines.Add($"  {p}  ({n})");
        }

        // Dump every tensor in declaration order so we can see the layer/sub-layer schedule.
        lines.Add("");
        lines.Add($"=== ALL TENSORS ({gguf.Tensors.Count}) ===");
        long totalBytes = 0;
        foreach (var t in gguf.Tensors)
        {
            var dims = string.Join("x", t.Dimensions);
            lines.Add($"  {t.Name,-60}  [{t.Type,-8}]  dims={dims,-20}  bytes={t.ByteSize}");
            totalBytes += (long)t.ByteSize;
        }
        lines.Add("");
        lines.Add($"Total tensor bytes: {totalBytes:N0}");

        Directory.CreateDirectory(Path.GetDirectoryName(outPath)!);
        File.WriteAllLines(outPath, lines);
    }

    /// <summary>
    /// Smoke test: load Gemma 4 Q4_0, run a single forward pass on CPU,
    /// dump the top-K logits and the predicted token. Doesn't assert
    /// correctness — just verifies that the load + forward pass don't crash
    /// and that the argmax token is in a sensible range.
    ///
    /// Uses Q4_0 (not Q8_0) because the per_layer_token_embd in Q8_0 is 2.99 GB
    /// and the current MmapModelLoader uses int-sized spans (max 2.1 GB per tensor).
    /// Supporting >2GB tensors is a separate workstream.
    /// </summary>
    [Fact]
    public void Gemma4_Q4_0_SingleForwardPass_DoesNotCrash()
    {
        var path = @"C:\GGUFS\gemma-4-E4B-it-Q4_0.gguf";
        if (!File.Exists(path)) return;

        using var stream = File.OpenRead(path);
        var gguf = GgufFile.Read(stream);
        var config = ModelConfig.FromGguf(gguf);

        Assert.True(config.IsGemma4, $"Expected gemma4 architecture, got {config.Architecture}");
        Assert.Equal(42, config.NumLayers);
        Assert.Equal(2560, config.HiddenDim);
        Assert.Equal(262144, config.VocabSize);
        Assert.Equal(8, config.NumHeads);
        Assert.Equal(2, config.NumKvHeads);
        Assert.Equal(512, config.KeyLength);          // full attention head dim
        Assert.Equal(256, config.KeyLengthSwa);       // sliding head dim
        Assert.Equal(512, config.SlidingWindow);
        Assert.Equal(256, config.PerLayerInputDim);
        Assert.Equal(30.0f, config.FinalLogitSoftcap);
        Assert.Equal(42, config.LayerSlidingMask.Length);
        Assert.True(config.LayerSlidingMask[0],  "Layer 0 should be sliding");
        Assert.False(config.LayerSlidingMask[5], "Layer 5 should be full attention");
        // KV-share: 42 - 18 = 24 layers compute their own K/V
        Assert.Equal(24, config.NumLayerKvFromStart);

        using var backend = new CpuBackend();
        using var weights = MmapModelLoader.Load(gguf, path, backend, config);

        Assert.NotNull(weights.PerLayerTokenEmbd);
        Assert.NotNull(weights.PerLayerModelProj);
        Assert.NotNull(weights.PerLayerProjNorm);
        Assert.NotNull(weights.RopeFreqs);

        // Run forward on a small max context to keep memory tiny.
        using var kvCache = new Gemma4KvCache(backend, config, maxSeqLen: 64);
        using var forward = new Gemma4ForwardPass(backend, config, weights, kvCache);

        // Pick a real token from the vocab. Token id 100 should be safe (well past special tokens).
        int testTokenId = 100;
        var sw = System.Diagnostics.Stopwatch.StartNew();
        var logits = forward.Forward(testTokenId, position: 0);
        sw.Stop();

        Assert.Equal(config.VocabSize, logits.Length);

        // Find argmax (CPU since IForwardPass.ArgMax isn't on this path)
        int argmax = 0;
        float argmaxVal = logits[0];
        for (int i = 1; i < logits.Length; i++)
            if (logits[i] > argmaxVal) { argmaxVal = logits[i]; argmax = i; }

        // Check there are no NaNs/Infs in the logits
        int nanCount = 0, infCount = 0;
        for (int i = 0; i < logits.Length; i++)
        {
            if (float.IsNaN(logits[i])) nanCount++;
            else if (float.IsInfinity(logits[i])) infCount++;
        }

        // Compute mean and stddev of logits for sanity
        double sum = 0, sumSq = 0;
        for (int i = 0; i < logits.Length; i++)
        {
            sum += logits[i];
            sumSq += (double)logits[i] * logits[i];
        }
        double mean = sum / logits.Length;
        double variance = sumSq / logits.Length - mean * mean;
        double stddev = Math.Sqrt(Math.Max(0, variance));

        // Copy logits to a regular array so we can use LINQ (can't capture ReadOnlySpan in lambdas).
        var logitArr = logits.ToArray();
        var top = Enumerable.Range(0, logitArr.Length)
            .OrderByDescending(i => logitArr[i])
            .Take(10)
            .ToList();

        var report = new List<string>
        {
            $"Gemma 4 single forward pass complete in {sw.ElapsedMilliseconds} ms",
            $"  Token: {testTokenId}",
            $"  Vocab size: {logitArr.Length}",
            $"  Argmax token: {argmax}  (logit={argmaxVal:F4})",
            $"  Logit stats: mean={mean:F4}  stddev={stddev:F4}",
            $"  NaN count: {nanCount}",
            $"  Inf count: {infCount}",
            $"  Top-10 tokens:",
        };
        foreach (var idx in top) report.Add($"    {idx,7} {logitArr[idx],10:F4}");
        File.WriteAllLines(@"C:\GGUFS\gemma-4-E4B-it-smoketest.txt", report);

        // Hard assertions (should be true even with imperfect implementation)
        Assert.Equal(0, nanCount);
        Assert.Equal(0, infCount);
        Assert.True(argmax >= 0 && argmax < config.VocabSize);
        // With logit softcap at 30, no logit should exceed 30 in magnitude
        Assert.True(Math.Abs(argmaxVal) <= 30.5f, $"argmax logit {argmaxVal} exceeds softcap=30");
    }

    /// <summary>
    /// Dequantize the embedding row for one token via two paths:
    /// (a) the EmbeddingLookup op (which uses CpuBackend's table-row code path),
    /// (b) the full DequantizeTo on the entire token_embd tensor.
    /// They should produce identical row contents. Mismatch => embedding lookup is buggy.
    /// </summary>
    [Fact]
    public void Gemma4_Q4_0_EmbeddingLookup_RowMatchesFullDequant()
    {
        var path = @"C:\GGUFS\gemma-4-E4B-it-Q4_0.gguf";
        if (!File.Exists(path)) return;

        using var stream = File.OpenRead(path);
        var gguf = GgufFile.Read(stream);
        var config = ModelConfig.FromGguf(gguf);

        using var backend = new CpuBackend();
        using var weights = MmapModelLoader.Load(gguf, path, backend, config);

        int testToken = 818; // "The"
        int hidden = config.HiddenDim;

        // Path A: EmbeddingLookup
        using var outA = backend.CreateTensor("test_emb_a", GgmlType.F32, [(long)hidden]);
        backend.EmbeddingLookup(outA, weights.TokenEmbedding, testToken);
        var rowA = outA.AsFloatSpan().ToArray();

        // Path B: dequantize the entire token_embd tensor and slice the row
        var fullEmbed = new float[weights.TokenEmbedding.ElementCount];
        weights.TokenEmbedding.DequantizeTo(fullEmbed);
        var rowB = new float[hidden];
        Array.Copy(fullEmbed, (long)testToken * hidden, rowB, 0, hidden);

        var report = new List<string>
        {
            $"Token {testToken} ('The') embedding row comparison:",
            $"  Token embed type: {weights.TokenEmbedding.Type}",
            $"  Hidden dim: {hidden}",
            "",
            $"  rowA (EmbeddingLookup) head: [{string.Join(", ", rowA.Take(8).Select(v => v.ToString("F5")))}]",
            $"  rowB (full dequant)    head: [{string.Join(", ", rowB.Take(8).Select(v => v.ToString("F5")))}]",
            "",
            $"  rowA stats: mean={rowA.Average():F5} min={rowA.Min():F5} max={rowA.Max():F5}",
            $"  rowB stats: mean={rowB.Average():F5} min={rowB.Min():F5} max={rowB.Max():F5}",
        };

        // Element-wise compare
        int maxAbsDiffIdx = 0;
        float maxAbsDiff = 0;
        for (int i = 0; i < hidden; i++)
        {
            float diff = Math.Abs(rowA[i] - rowB[i]);
            if (diff > maxAbsDiff) { maxAbsDiff = diff; maxAbsDiffIdx = i; }
        }
        report.Add($"  Max abs diff: {maxAbsDiff} at index {maxAbsDiffIdx}");
        report.Add($"    rowA[{maxAbsDiffIdx}] = {rowA[maxAbsDiffIdx]}");
        report.Add($"    rowB[{maxAbsDiffIdx}] = {rowB[maxAbsDiffIdx]}");

        File.WriteAllLines(@"C:\GGUFS\gemma-4-E4B-it-embedlookup.txt", report);
    }

    /// <summary>
    /// Feed a single BOS token at position 0. Sanity-check that the predicted argmax
    /// after BOS is a "sensible starting word" — for an instruction-tuned model, the
    /// most likely first tokens are typically chat-template markers, "I", or "The".
    /// </summary>
    [Fact]
    public void Gemma4_Q4_0_BosOnly_Top10()
    {
        var path = @"C:\GGUFS\gemma-4-E4B-it-Q4_0.gguf";
        if (!File.Exists(path)) return;

        using var stream = File.OpenRead(path);
        var gguf = GgufFile.Read(stream);
        var config = ModelConfig.FromGguf(gguf);
        var tokenizer = TokenizerFactory.FromGguf(gguf);

        using var backend = new CpuBackend();
        using var weights = MmapModelLoader.Load(gguf, path, backend, config);
        using var kv = new Gemma4KvCache(backend, config, maxSeqLen: 8);
        using var fwd = new Gemma4ForwardPass(backend, config, weights, kv);

        var logits = fwd.Forward(tokenizer.Vocabulary.BosTokenId, position: 0);
        var logitArr = logits.ToArray();

        var top = Enumerable.Range(0, logitArr.Length)
            .OrderByDescending(i => logitArr[i])
            .Take(20)
            .ToList();

        var report = new List<string>
        {
            $"After BOS (token id={tokenizer.Vocabulary.BosTokenId}) at position 0:",
            $"  Top 20 predictions:",
        };
        foreach (var idx in top)
        {
            string text = tokenizer.Decode(new[] { idx });
            report.Add($"    {idx,6}  logit={logitArr[idx],8:F4}  text='{text}'");
        }
        File.WriteAllLines(@"C:\GGUFS\gemma-4-E4B-it-bostop.txt", report);
    }

    /// <summary>
    /// Try the same prompt with various combinations of debug flags disabled.
    /// Helps isolate which Gemma 4 special feature is buggy.
    /// </summary>
    [Fact]
    public void Gemma4_Q4_0_DebugMatrix()
    {
        var path = @"C:\GGUFS\gemma-4-E4B-it-Q4_0.gguf";
        if (!File.Exists(path)) return;

        using var stream = File.OpenRead(path);
        var gguf = GgufFile.Read(stream);
        var config = ModelConfig.FromGguf(gguf);
        var tokenizer = TokenizerFactory.FromGguf(gguf);

        using var backend = new CpuBackend();
        using var weights = MmapModelLoader.Load(gguf, path, backend, config);

        string prompt = "The capital of France is";
        var encoded = tokenizer.Encode(prompt);
        var promptIds = new int[encoded.Length + 1];
        promptIds[0] = tokenizer.Vocabulary.BosTokenId;
        Array.Copy(encoded, 0, promptIds, 1, encoded.Length);

        var report = new List<string>
        {
            $"Prompt: \"{prompt}\"",
            $"Tokens: [{string.Join(", ", promptIds)}]",
            "",
        };

        // Test combinations
        var configs = new (string label, bool noPle, bool noScale, bool noEmbScale, bool noQkNorm, bool noVNorm)[]
        {
            ("ALL ON  (default)",        false, false, false, false, false),
            ("noQkNorm",                 false, false, false, true,  false),
            ("noVNorm",                  false, false, false, false, true),
            ("noQkNorm+noVNorm",         false, false, false, true,  true),
            ("noPle+noScale",            true,  true,  false, false, false),
            ("noEmbScale",               false, false, true,  false, false),
        };

        foreach (var c in configs)
        {
            using var kv = new Gemma4KvCache(backend, config, maxSeqLen: 64);
            using var fwd = new Gemma4ForwardPass(backend, config, weights, kv)
            {
                DebugDisablePle = c.noPle,
                DebugDisableLayerOutScale = c.noScale,
                DebugDisableEmbeddingScale = c.noEmbScale,
                DebugDisableQkNorm = c.noQkNorm,
                DebugDisableVNorm = c.noVNorm,
            };

            ReadOnlySpan<float> logits = default;
            for (int i = 0; i < promptIds.Length; i++)
                logits = fwd.Forward(promptIds[i], i);

            // Greedy 4 tokens
            var gen = new List<int>();
            int position = promptIds.Length;
            for (int t = 0; t < 4; t++)
            {
                int argmax = 0;
                float bestVal = logits[0];
                for (int i = 1; i < logits.Length; i++)
                    if (logits[i] > bestVal) { bestVal = logits[i]; argmax = i; }
                gen.Add(argmax);
                logits = fwd.Forward(argmax, position);
                position++;
            }

            string text = tokenizer.Decode(gen.ToArray());
            report.Add($"  [{c.label,-30}] -> '{text}'  (ids: {string.Join(",", gen)})");
        }

        File.WriteAllLines(@"C:\GGUFS\gemma-4-E4B-it-debugmatrix.txt", report);
    }

    /// <summary>
    /// Generate a few tokens with greedy decoding from a real prompt.
    /// Dumps the generated text to a file for visual inspection of coherence.
    /// </summary>
    [Fact]
    public void Gemma4_Q4_0_GreedyGenerate_DumpsText()
    {
        var path = @"C:\GGUFS\gemma-4-E4B-it-Q4_0.gguf";
        if (!File.Exists(path)) return;

        using var stream = File.OpenRead(path);
        var gguf = GgufFile.Read(stream);
        var config = ModelConfig.FromGguf(gguf);
        var tokenizer = TokenizerFactory.FromGguf(gguf);

        using var backend = new CpuBackend();
        using var weights = MmapModelLoader.Load(gguf, path, backend, config);
        using var kvCache = new Gemma4KvCache(backend, config, maxSeqLen: 256);
        using var forward = new Gemma4ForwardPass(backend, config, weights, kvCache);

        // Use the Gemma chat template — system+user wrapped in <start_of_turn>...<end_of_turn>
        // For this raw smoke test we'll feed plain text without the template.
        // Gemma requires BOS at the start of every sequence.
        string prompt = "The capital of France is";
        var encoded = tokenizer.Encode(prompt);
        var promptIds = new int[encoded.Length + 1];
        promptIds[0] = tokenizer.Vocabulary.BosTokenId; // <bos>
        Array.Copy(encoded, 0, promptIds, 1, encoded.Length);

        var report = new List<string>
        {
            $"Prompt: \"{prompt}\"  (BOS-prepended)",
            $"Prompt token ids: [{string.Join(", ", promptIds)}]",
            $"Decoded prompt tokens:",
        };
        for (int i = 0; i < promptIds.Length; i++)
            report.Add($"  [{promptIds[i],6}] = '{tokenizer.Decode(new[] { promptIds[i] })}'");

        var prefillSw = System.Diagnostics.Stopwatch.StartNew();
        ReadOnlySpan<float> logits = default;
        for (int i = 0; i < promptIds.Length; i++)
            logits = forward.Forward(promptIds[i], i);
        prefillSw.Stop();

        report.Add("");
        report.Add($"Prefill: {promptIds.Length} tokens in {prefillSw.ElapsedMilliseconds} ms");
        report.Add("");
        report.Add("Greedy decode (16 tokens):");

        var generated = new List<int>();
        int position = promptIds.Length;
        var decodeSw = System.Diagnostics.Stopwatch.StartNew();
        for (int t = 0; t < 16; t++)
        {
            // Greedy argmax from the logits we have on hand
            int argmax = 0;
            float bestVal = logits[0];
            for (int i = 1; i < logits.Length; i++)
                if (logits[i] > bestVal) { bestVal = logits[i]; argmax = i; }

            generated.Add(argmax);
            string text = tokenizer.Decode(new[] { argmax });
            report.Add($"  step={t,2} pos={position,3} token={argmax,6} logit={bestVal,8:F4} text='{text}'");

            logits = forward.Forward(argmax, position);
            position++;
        }
        decodeSw.Stop();

        report.Add("");
        report.Add($"Decode: 16 tokens in {decodeSw.ElapsedMilliseconds} ms ({16000.0 / decodeSw.ElapsedMilliseconds:F2} tok/s)");
        report.Add("");
        report.Add($"Final output: \"{prompt}{tokenizer.Decode(generated.ToArray())}\"");

        File.WriteAllLines(@"C:\GGUFS\gemma-4-E4B-it-greedy.txt", report);
    }

    [Fact]
    public void Gemma4_Q4_0_ForwardBatch_MatchesSequential()
    {
        var path = @"C:\GGUFS\gemma-4-E4B-it-Q4_0.gguf";
        if (!File.Exists(path)) return;

        using var stream = File.OpenRead(path);
        var gguf = GgufFile.Read(stream);
        var config = ModelConfig.FromGguf(gguf);
        var tokenizer = TokenizerFactory.FromGguf(gguf);

        using var backend = new CpuBackend();
        using var weights = MmapModelLoader.Load(gguf, path, backend, config);

        // Sequential: run Forward(i) for each prompt token
        using var kvSeq = new Gemma4KvCache(backend, config, maxSeqLen: 256);
        using var forwardSeq = new Gemma4ForwardPass(backend, config, weights, kvSeq);

        string prompt = "The capital of France is";
        var encoded = tokenizer.Encode(prompt);
        var promptIds = new int[encoded.Length + 1];
        promptIds[0] = tokenizer.Vocabulary.BosTokenId;
        Array.Copy(encoded, 0, promptIds, 1, encoded.Length);

        var seqSw = System.Diagnostics.Stopwatch.StartNew();
        ReadOnlySpan<float> seqLogits = default;
        for (int i = 0; i < promptIds.Length; i++)
            seqLogits = forwardSeq.Forward(promptIds[i], i);
        seqSw.Stop();
        var seqTop = ArgMax(seqLogits);
        var seqLogitsCopy = seqLogits.ToArray();

        // Batched: run ForwardBatch(all, 0)
        using var kvBat = new Gemma4KvCache(backend, config, maxSeqLen: 256);
        using var forwardBat = new Gemma4ForwardPass(backend, config, weights, kvBat);

        var batSw = System.Diagnostics.Stopwatch.StartNew();
        var batLogits = forwardBat.ForwardBatch(promptIds, 0);
        batSw.Stop();
        var batTop = ArgMax(batLogits);

        // Compare: top-1 token must match, top logits should match to 4 decimal places
        var report = new List<string>
        {
            $"Prompt: \"{prompt}\" ({promptIds.Length} tokens after BOS)",
            "",
            $"Sequential ({promptIds.Length} single-token Forward calls):",
            $"  Time: {seqSw.ElapsedMilliseconds} ms",
            $"  Top token: {seqTop} '{tokenizer.Decode(new[] { seqTop })}' logit={seqLogitsCopy[seqTop]:F4}",
            "",
            $"Batched (1 ForwardBatch call):",
            $"  Time: {batSw.ElapsedMilliseconds} ms",
            $"  Top token: {batTop} '{tokenizer.Decode(new[] { batTop })}' logit={batLogits[batTop]:F4}",
            "",
            $"Speedup: {(double)seqSw.ElapsedMilliseconds / batSw.ElapsedMilliseconds:F2}x",
            "",
            "Top-5 logit comparison:",
        };

        // Report top-5 of both
        var seqIdx = TopK(seqLogitsCopy, 5);
        var batIdx = TopK(batLogits.ToArray(), 5);
        report.Add("  Sequential top-5:");
        foreach (var i in seqIdx)
            report.Add($"    {i,7} '{tokenizer.Decode(new[] { i })}' logit={seqLogitsCopy[i]:F4}");
        report.Add("  Batched top-5:");
        foreach (var i in batIdx)
            report.Add($"    {i,7} '{tokenizer.Decode(new[] { i })}' logit={batLogits[i]:F4}");

        File.WriteAllLines(@"C:\GGUFS\gemma-4-E4B-it-batch-compare.txt", report);

        // Correctness: top token must match
        Assert.Equal(seqTop, batTop);
    }

    private static int ArgMax(ReadOnlySpan<float> v)
    {
        int best = 0; float bestVal = v[0];
        for (int i = 1; i < v.Length; i++)
            if (v[i] > bestVal) { bestVal = v[i]; best = i; }
        return best;
    }

    private static int[] TopK(float[] v, int k)
    {
        var idx = new int[v.Length];
        for (int i = 0; i < idx.Length; i++) idx[i] = i;
        Array.Sort(idx, (a, b) => v[b].CompareTo(v[a]));
        return idx.AsSpan(0, k).ToArray();
    }

    [Fact]
    public void Gemma4_Q4_0_ProfileDecode()
    {
        var path = @"C:\GGUFS\gemma-4-E4B-it-Q4_0.gguf";
        if (!File.Exists(path)) return;
        RunProfileDecode(path, Daisi.Llogos.Cpu.CpuThreading.ThreadCount);
    }

    [Fact]
    public void Gemma4_Q4_0_BenchBatchedPrefill()
    {
        var path = @"C:\GGUFS\gemma-4-E4B-it-Q4_0.gguf";
        if (!File.Exists(path)) return;

        using var stream = File.OpenRead(path);
        var gguf = GgufFile.Read(stream);
        var config = ModelConfig.FromGguf(gguf);

        using var backend = new CpuBackend();
        using var weights = MmapModelLoader.Load(gguf, path, backend, config);

        // Different batch sizes to see how speedup scales
        var batchSizes = new[] { 1, 2, 4, 8, 12, 16, 24, 32 };
        var lines = new List<string>
        {
            "Gemma 4 Q4_0 batched prefill sweep + profile:",
            "",
        };

        foreach (int M in batchSizes)
        {
            using var kvCache = new Gemma4KvCache(backend, config, maxSeqLen: 512);
            using var forward = new Gemma4ForwardPass(backend, config, weights, kvCache, maxBatchSize: 32);

            var ids = new int[M];
            for (int i = 0; i < ids.Length; i++) ids[i] = 100 + i;

            // Warmup (uses profiling internally but we reset after)
            forward.ForwardBatch(ids, 0);
            kvCache.Reset();

            forward.EnableProfiling = true;
            forward.ResetProfile();

            var sw = System.Diagnostics.Stopwatch.StartNew();
            forward.ForwardBatch(ids, 0);
            sw.Stop();

            forward.EnableProfiling = false;

            double total = sw.Elapsed.TotalMilliseconds;
            double tickMs = 1000.0 / System.Diagnostics.Stopwatch.Frequency;
            lines.Add($"M={M}: total {total:F1} ms ({total / M:F2} ms/tok, {1000.0 * M / total:F2} tok/s)");
            lines.Add($"  Embedding:    {forward.ProfileEmbTicks * tickMs,8:F2} ms");
            lines.Add($"  PLE setup:    {forward.ProfilePleSetupTicks * tickMs,8:F2} ms");
            lines.Add($"  Attn matmul:  {forward.ProfileAttnMatmulTicks * tickMs,8:F2} ms");
            lines.Add($"  Attn other:   {forward.ProfileAttnOtherTicks * tickMs,8:F2} ms");
            lines.Add($"  FFN matmul:   {forward.ProfileFfnMatmulTicks * tickMs,8:F2} ms");
            lines.Add($"  FFN other:    {forward.ProfileFfnOtherTicks * tickMs,8:F2} ms");
            lines.Add($"  Norm:         {forward.ProfileNormTicks * tickMs,8:F2} ms");
            lines.Add($"  PLE block:    {forward.ProfilePleBlockTicks * tickMs,8:F2} ms");
            lines.Add($"  lm_head:      {forward.ProfileLmHeadTicks * tickMs,8:F2} ms");
            lines.Add("");
        }

        File.WriteAllLines(@"C:\GGUFS\gemma-4-E4B-it-batch-sweep.txt", lines);
    }

    private static void RunProfileDecode(string path, int threadTag)
    {
        using var stream = File.OpenRead(path);
        var gguf = GgufFile.Read(stream);
        var config = ModelConfig.FromGguf(gguf);
        var tokenizer = TokenizerFactory.FromGguf(gguf);

        using var backend = new CpuBackend();
        using var weights = MmapModelLoader.Load(gguf, path, backend, config);
        using var kvCache = new Gemma4KvCache(backend, config, maxSeqLen: 256);
        using var forward = new Gemma4ForwardPass(backend, config, weights, kvCache);

        // Prefill a short prompt first (no profiling — warm caches)
        string prompt = "The capital of France is";
        var encoded = tokenizer.Encode(prompt);
        var promptIds = new int[encoded.Length + 1];
        promptIds[0] = tokenizer.Vocabulary.BosTokenId;
        Array.Copy(encoded, 0, promptIds, 1, encoded.Length);

        ReadOnlySpan<float> logits = default;
        for (int i = 0; i < promptIds.Length; i++)
            logits = forward.Forward(promptIds[i], i);

        // Profile the next 10 decode steps
        forward.EnableProfiling = true;
        forward.ResetProfile();

        int position = promptIds.Length;
        var totalSw = System.Diagnostics.Stopwatch.StartNew();
        for (int t = 0; t < 10; t++)
        {
            int argmax = 0;
            float bestVal = logits[0];
            for (int i = 1; i < logits.Length; i++)
                if (logits[i] > bestVal) { bestVal = logits[i]; argmax = i; }

            logits = forward.Forward(argmax, position);
            position++;
        }
        totalSw.Stop();
        forward.EnableProfiling = false;

        double tickToMs = 1000.0 / System.Diagnostics.Stopwatch.Frequency;
        double total = totalSw.Elapsed.TotalMilliseconds;
        double embMs = forward.ProfileEmbTicks * tickToMs;
        double pleSetupMs = forward.ProfilePleSetupTicks * tickToMs;
        double attnMmMs = forward.ProfileAttnMatmulTicks * tickToMs;
        double attnOtMs = forward.ProfileAttnOtherTicks * tickToMs;
        double ffnMmMs = forward.ProfileFfnMatmulTicks * tickToMs;
        double ffnOtMs = forward.ProfileFfnOtherTicks * tickToMs;
        double normMs = forward.ProfileNormTicks * tickToMs;
        double pleBlMs = forward.ProfilePleBlockTicks * tickToMs;
        double lmHeadMs = forward.ProfileLmHeadTicks * tickToMs;
        double accounted = embMs + pleSetupMs + attnMmMs + attnOtMs + ffnMmMs + ffnOtMs + normMs + pleBlMs + lmHeadMs;
        double unaccounted = total - accounted;

        var lines = new List<string>
        {
            $"Gemma 4 Q4_0 — decode profile (threads={threadTag}, 10 steps, total {total:F1} ms, {10000.0 / total:F2} tok/s):",
            "",
            $"  Embedding lookup+scale   : {embMs,8:F2} ms  ({embMs / total * 100,5:F1}%)",
            $"  PLE setup (per token)    : {pleSetupMs,8:F2} ms  ({pleSetupMs / total * 100,5:F1}%)",
            $"  Attention matmul (Q/K/V/O): {attnMmMs,8:F2} ms  ({attnMmMs / total * 100,5:F1}%)",
            $"  Attention other (RoPE/sm): {attnOtMs,8:F2} ms  ({attnOtMs / total * 100,5:F1}%)",
            $"  FFN matmul (gate/up/down): {ffnMmMs,8:F2} ms  ({ffnMmMs / total * 100,5:F1}%)",
            $"  FFN other (GeGLU)        : {ffnOtMs,8:F2} ms  ({ffnOtMs / total * 100,5:F1}%)",
            $"  RmsNorm (5 per layer)    : {normMs,8:F2} ms  ({normMs / total * 100,5:F1}%)",
            $"  PLE block (per layer)    : {pleBlMs,8:F2} ms  ({pleBlMs / total * 100,5:F1}%)",
            $"  lm_head + final norm     : {lmHeadMs,8:F2} ms  ({lmHeadMs / total * 100,5:F1}%)",
            $"  (unaccounted)            : {unaccounted,8:F2} ms  ({unaccounted / total * 100,5:F1}%)",
            "",
            $"Per-token breakdown (ms): {total / 10:F2} ms/tok",
        };
        File.WriteAllLines(@"C:\GGUFS\gemma-4-E4B-it-profile.txt", lines);
    }

    private static string FormatMetadataValue(GgufMetadataKv kv)
    {
        // Strings, numbers, bools — direct ToString.
        if (kv.Type != GgufMetadataValueType.Array)
            return kv.Value?.ToString() ?? "<null>";

        // Arrays — show length + a small head sample. Don't dump the full 262K vocab.
        var arr = (Array)kv.Value;
        const int sample = 8;
        var head = new List<string>();
        for (int i = 0; i < Math.Min(sample, arr.Length); i++)
        {
            var v = arr.GetValue(i);
            var s = v?.ToString() ?? "<null>";
            if (s.Length > 40) s = s.Substring(0, 40) + "…";
            head.Add(s);
        }
        var suffix = arr.Length > sample ? $", … (+{arr.Length - sample} more)" : "";
        return $"array[{arr.Length}] = [{string.Join(", ", head)}{suffix}]";
    }
}
