using Daisi.Llogos.Cpu;
using Daisi.Llogos.Gguf;
using Daisi.Llogos.Inference;
using Daisi.Llogos.Model;
using Daisi.Llogos.Tokenizer;

namespace Daisi.Llogos.Tests.Inference;

/// <summary>
/// Diagnostic tests for DeltaNet forward pass — dumps intermediate values
/// to identify where the 9B model diverges from expected behavior.
/// </summary>
public class DeltaNetDiagnosticTests
{
    private static readonly string Qwen35_08B = @"C:\GGUFS\Qwen3.5-0.8B-Q8_0.gguf";
    private static readonly string Qwen35_9B = @"C:\GGUFS\Qwen3.5-9B-Q8_0.gguf";

    /// <summary>
    /// Compare embedding + first norm between 0.8B and 9B to verify basic setup.
    /// </summary>
    [Fact]
    public void Diagnostic_CompareModels()
    {
        var lines = new List<string>();

        foreach (var (path, label) in new[] { (Qwen35_08B, "0.8B"), (Qwen35_9B, "9B") })
        {
            if (!File.Exists(path)) continue;

            using var stream = File.OpenRead(path);
            var gguf = GgufFile.Read(stream);
            var config = ModelConfig.FromGguf(gguf);
            using var backend = new CpuBackend();
            var weights = ModelLoader.Load(gguf, stream, backend, config);
            var tokenizer = TokenizerFactory.FromGguf(gguf);

            // Encode "Hello"
            var tokens = tokenizer.Encode("Hello");
            int tokenId = tokens[0];

            lines.Add($"=== {label} (token={tokenId}) ===");
            lines.Add($"Config: layers={config.NumLayers}, hidden={config.HiddenDim}, " +
                $"ssmInner={config.SsmInnerSize}, ssmGroup={config.SsmGroupCount}, " +
                $"ssmState={config.SsmStateSize}, ssmHeadDim={config.SsmHeadDim}");

            // Create forward pass resources
            using var kvCache = new KvCache(backend, config, maxSeqLen: 16);
            using var deltaState = new DeltaNetState(backend, config, weights);
            using var forward = new ForwardPass(backend, config, weights, kvCache, deltaState);

            // Run one forward pass
            var logits = forward.Forward(tokenId, 0);

            // Dump logits stats
            float logitSum = 0, logitMax = float.MinValue;
            int argmax = 0;
            for (int i = 0; i < logits.Length; i++)
            {
                logitSum += logits[i];
                if (logits[i] > logitMax) { logitMax = logits[i]; argmax = i; }
            }
            lines.Add($"Logits: sum={logitSum:F2}, max={logitMax:F4}, argmax={argmax}");
            lines.Add($"Top logits: [{string.Join(", ", GetTopK(logits, 10).Select(x => $"{x.idx}:{x.val:F3}"))}]");

            // Decode argmax
            string topToken = tokenizer.Decode([argmax]);
            lines.Add($"Top token: '{topToken}'");

            // Run a few more tokens to see if output is coherent
            var generated = new List<int>();
            for (int pos = 1; pos < 6; pos++)
            {
                int nextToken = pos == 1 ? argmax : generated[^1];
                var nextLogits = forward.Forward(nextToken, pos);
                float nMax = float.MinValue;
                int nArgmax = 0;
                for (int i = 0; i < nextLogits.Length; i++)
                    if (nextLogits[i] > nMax) { nMax = nextLogits[i]; nArgmax = i; }
                generated.Add(nArgmax);
            }
            string genText = tokenizer.Decode(generated.ToArray());
            lines.Add($"Generated (5 tokens): '{genText}'");
            lines.Add("");

            weights.Dispose();
        }

        File.WriteAllLines(@"C:\GGUFS\deltanet-diagnostic.txt", lines);
        foreach (var line in lines)
            Console.WriteLine(line);
    }

    /// <summary>
    /// Run the 9B model and dump DeltaNet layer intermediate values.
    /// This helps identify which step in the DeltaNet computation goes wrong.
    /// </summary>
    [Fact]
    public void Diagnostic_9B_DeltaNetLayerDump()
    {
        if (!File.Exists(Qwen35_9B)) return;

        using var stream = File.OpenRead(Qwen35_9B);
        var gguf = GgufFile.Read(stream);
        var config = ModelConfig.FromGguf(gguf);
        using var backend = new CpuBackend();
        var weights = ModelLoader.Load(gguf, stream, backend, config);
        var tokenizer = TokenizerFactory.FromGguf(gguf);

        var tokens = tokenizer.Encode("Hello");
        int tokenId = tokens[0];

        var lines = new List<string>();
        lines.Add($"=== 9B DeltaNet Layer Dump (token={tokenId}) ===");

        // Manually run the forward pass with instrumentation
        var hidden = backend.CreateTensor("hidden", GgmlType.F32, [config.HiddenDim]);
        var residual = backend.CreateTensor("residual", GgmlType.F32, [config.HiddenDim]);
        var normOut = backend.CreateTensor("normOut", GgmlType.F32, [config.HiddenDim]);

        // Embedding
        backend.EmbeddingLookup(hidden, weights.TokenEmbedding, tokenId);
        lines.Add($"Embedding: {DumpStats(hidden)}");

        // Process first few layers
        for (int layer = 0; layer < Math.Min(config.NumLayers, 4); layer++)
        {
            var lw = weights.Layers[layer];
            bool isDeltaNet = !config.IsStandardAttention(layer);

            backend.CopyTensor(residual, hidden);
            backend.RmsNorm(normOut, hidden, lw.AttnNorm, config.NormEps);
            lines.Add($"Layer {layer} ({(isDeltaNet ? "DeltaNet" : "Attention")}) norm: {DumpStats(normOut)}");

            if (isDeltaNet && lw is DeltaNetWeights dnw)
            {
                int headDim = config.SsmStateSize > 0 ? config.SsmStateSize : config.SsmHeadDim;
                int numVHeads = (int)dnw.SsmAlpha.Dimensions[1];
                int qkvOutDim = (int)dnw.AttnQkv.Dimensions[1];
                int valueDim = numVHeads * headDim;
                int keyDim = (qkvOutDim - valueDim) / 2;
                int numKHeads = keyDim / headDim;

                lines.Add($"  Dims: qkvOut={qkvOutDim}, keyDim={keyDim}, valueDim={valueDim}, numKHeads={numKHeads}, numVHeads={numVHeads}");

                // QKV projection
                var qkvBuf = backend.CreateTensor("qkv", GgmlType.F32, [qkvOutDim]);
                backend.MatMul(qkvBuf, normOut, dnw.AttnQkv, 1, config.HiddenDim, qkvOutDim);
                lines.Add($"  QKV proj: {DumpStats(qkvBuf)}");

                // Conv1d
                int convChannels = (int)(dnw.SsmConv1d.ElementCount / config.SsmConvKernel);
                // Skip conv for diagnostic — just check QKV values
                lines.Add($"  Conv channels: {convChannels}");

                // Split
                var qkvData = new float[qkvOutDim];
                qkvBuf.DequantizeTo(qkvData);

                float qSum = 0, kSum = 0, vSum = 0;
                for (int i = 0; i < keyDim; i++) qSum += qkvData[i];
                for (int i = keyDim; i < keyDim * 2; i++) kSum += qkvData[i];
                for (int i = keyDim * 2; i < qkvOutDim; i++) vSum += qkvData[i];
                lines.Add($"  Q region sum={qSum:F4}, K region sum={kSum:F4}, V region sum={vSum:F4}");

                // Alpha/beta projections
                var alpha = new float[numVHeads];
                var beta = new float[numVHeads];
                var alphaTensor = backend.CreateTensor("alpha", GgmlType.F32, [numVHeads]);
                var betaTensor = backend.CreateTensor("beta", GgmlType.F32, [numVHeads]);
                backend.MatMul(alphaTensor, normOut, dnw.SsmAlpha, 1, config.HiddenDim, numVHeads);
                backend.MatMul(betaTensor, normOut, dnw.SsmBeta, 1, config.HiddenDim, numVHeads);
                alphaTensor.DequantizeTo(alpha);
                betaTensor.DequantizeTo(beta);
                lines.Add($"  Alpha[0..3]: [{alpha[0]:F4}, {alpha[1]:F4}, {alpha[2]:F4}, {alpha[3]:F4}]");
                lines.Add($"  Beta[0..3]: [{beta[0]:F4}, {beta[1]:F4}, {beta[2]:F4}, {beta[3]:F4}]");

                // Decay computation
                var ssmA = new float[numVHeads];
                var dtBias = new float[numVHeads];
                dnw.SsmA.DequantizeTo(ssmA);
                dnw.SsmDtBias.DequantizeTo(dtBias);
                for (int g = 0; g < Math.Min(4, numVHeads); g++)
                {
                    float softplus = MathF.Log(1.0f + MathF.Exp(alpha[g] + dtBias[g]));
                    float decay = MathF.Exp(ssmA[g] * softplus);
                    float betaVal = 1.0f / (1.0f + MathF.Exp(-beta[g]));
                    lines.Add($"  Head {g}: softplus={softplus:F4}, decay={decay:F6}, beta={betaVal:F4}");
                }

                qkvBuf.Dispose();
                alphaTensor.Dispose();
                betaTensor.Dispose();
            }
            else if (lw is StandardAttentionWeights)
            {
                lines.Add($"  (Standard attention layer — skipping detailed dump)");
            }

            // Simplified: skip the actual attention/DeltaNet computation for dump
            // Just check that layer norms produce reasonable values
        }

        lines.Add("");
        File.WriteAllLines(@"C:\GGUFS\deltanet-layer-dump.txt", lines);
        foreach (var line in lines)
            Console.WriteLine(line);

        hidden.Dispose();
        residual.Dispose();
        normOut.Dispose();
        weights.Dispose();
    }

    /// <summary>
    /// Run both models through a FULL forward pass and dump hidden state after each layer.
    /// This shows exactly where the 9B diverges.
    /// </summary>
    [Fact]
    public void Diagnostic_9B_HiddenStatePerLayer()
    {
        var lines = new List<string>();

        foreach (var (path, label) in new[] { (Qwen35_08B, "0.8B"), (Qwen35_9B, "9B") })
        {
            if (!File.Exists(path)) continue;

            using var stream = File.OpenRead(path);
            var gguf = GgufFile.Read(stream);
            var config = ModelConfig.FromGguf(gguf);
            using var backend = new CpuBackend();
            var weights = ModelLoader.Load(gguf, stream, backend, config);
            var tokenizer = TokenizerFactory.FromGguf(gguf);

            var tokens = tokenizer.Encode("Hello");
            int tokenId = tokens[0];

            lines.Add($"=== {label} hidden state per layer ===");

            // Manual forward pass with per-layer hidden state dump
            var hidden = backend.CreateTensor("h", GgmlType.F32, [config.HiddenDim]);
            var residual = backend.CreateTensor("r", GgmlType.F32, [config.HiddenDim]);
            var normOut = backend.CreateTensor("n", GgmlType.F32, [config.HiddenDim]);
            using var kvCache = new KvCache(backend, config, maxSeqLen: 16);
            using var deltaState = new DeltaNetState(backend, config, weights);
            using var forward = new ForwardPass(backend, config, weights, kvCache, deltaState);

            // Use the actual forward pass but intercept by running it and
            // comparing the logits - the issue is we can't easily intercept.
            // Instead, let's just run the full forward and check logits quality.

            // First: run with ALL layers
            var logitsAll = forward.Forward(tokenId, 0);
            float maxAll = float.MinValue;
            int argmaxAll = 0;
            for (int i = 0; i < logitsAll.Length; i++)
                if (logitsAll[i] > maxAll) { maxAll = logitsAll[i]; argmaxAll = i; }
            lines.Add($"All layers: argmax={argmaxAll} ('{tokenizer.Decode([argmaxAll])}'), max={maxAll:F4}");

            // Now: test with a modified approach - run layer 0 (DeltaNet) independently
            // by doing embedding + norm + DeltaNet manually
            backend.EmbeddingLookup(hidden, weights.TokenEmbedding, tokenId);
            var embData = hidden.AsFloatSpan().ToArray();
            float embNorm = 0;
            for (int i = 0; i < embData.Length; i++) embNorm += embData[i] * embData[i];
            embNorm = MathF.Sqrt(embNorm);
            lines.Add($"Embedding L2 norm: {embNorm:F4}");

            // Check if any standard attention layer (layer 3) produces reasonable output
            // by running the full model and checking
            forward.ResetState();
            var logitsReset = forward.Forward(tokenId, 0);
            float maxReset = float.MinValue;
            int argmaxReset = 0;
            for (int i = 0; i < logitsReset.Length; i++)
                if (logitsReset[i] > maxReset) { maxReset = logitsReset[i]; argmaxReset = i; }
            lines.Add($"Reset + rerun: argmax={argmaxReset} ('{tokenizer.Decode([argmaxReset])}'), max={maxReset:F4}");
            lines.Add($"Same as first run: {argmaxAll == argmaxReset}");
            lines.Add("");

            hidden.Dispose();
            residual.Dispose();
            normOut.Dispose();
            weights.Dispose();
        }

        File.WriteAllLines(@"C:\GGUFS\deltanet-hidden-state.txt", lines);
        foreach (var line in lines)
            Console.WriteLine(line);
    }

    /// <summary>
    /// Test if the 9B model's standard attention layers work correctly
    /// by comparing output when DeltaNet layers are skipped.
    /// </summary>
    [Fact]
    public void Diagnostic_9B_SkipDeltaNet()
    {
        if (!File.Exists(Qwen35_9B)) return;

        using var stream = File.OpenRead(Qwen35_9B);
        var gguf = GgufFile.Read(stream);
        var config = ModelConfig.FromGguf(gguf);
        using var backend = new CpuBackend();
        var weights = ModelLoader.Load(gguf, stream, backend, config);
        var tokenizer = TokenizerFactory.FromGguf(gguf);

        var tokens = tokenizer.Encode("The capital of France is");

        var lines = new List<string>();
        lines.Add("=== 9B with DeltaNet layers as identity (skip) ===");

        // Manual forward pass: run all layers but make DeltaNet layers
        // act as identity (just residual passthrough)
        var hidden = backend.CreateTensor("h", GgmlType.F32, [config.HiddenDim]);
        var residual = backend.CreateTensor("r", GgmlType.F32, [config.HiddenDim]);
        var normOut = backend.CreateTensor("n", GgmlType.F32, [config.HiddenDim]);
        var logits = backend.CreateTensor("logits", GgmlType.F32, [config.VocabSize]);
        var gate = backend.CreateTensor("gate", GgmlType.F32, [config.IntermediateDim]);
        var up = backend.CreateTensor("up", GgmlType.F32, [config.IntermediateDim]);
        using var kvCache = new KvCache(backend, config, maxSeqLen: 64);

        // Process first token
        int tokenId = tokens[0];
        backend.EmbeddingLookup(hidden, weights.TokenEmbedding, tokenId);

        for (int layer = 0; layer < config.NumLayers; layer++)
        {
            var lw = weights.Layers[layer];
            bool isDeltaNet = !config.IsStandardAttention(layer);

            backend.CopyTensor(residual, hidden);
            backend.RmsNorm(normOut, hidden, lw.AttnNorm, config.NormEps);

            if (isDeltaNet)
            {
                // Skip DeltaNet: set hidden to zero (only residual passes through)
                backend.ZeroTensor(hidden);
            }
            else if (lw is StandardAttentionWeights saw)
            {
                // Run standard attention normally
                int numHeads = config.NumHeads;
                int numKvHeads = config.NumKvHeads;
                int keyLen = config.KeyLength;
                int valLen = config.ValueLength;

                var qAttn = backend.CreateTensor("q", GgmlType.F32, [numHeads * keyLen]);
                var kProj = backend.CreateTensor("k", GgmlType.F32, [numKvHeads * keyLen]);
                var vProj = backend.CreateTensor("v", GgmlType.F32, [numKvHeads * valLen]);
                var attnOut = backend.CreateTensor("ao", GgmlType.F32, [numHeads * valLen]);

                backend.MatMul(qAttn, normOut, saw.AttnQ, 1, config.HiddenDim, numHeads * keyLen);
                backend.MatMul(kProj, normOut, saw.AttnK, 1, config.HiddenDim, numKvHeads * keyLen);
                backend.MatMul(vProj, normOut, saw.AttnV, 1, config.HiddenDim, numKvHeads * valLen);

                if (saw.AttnQNorm != null)
                {
                    backend.PerHeadRmsNorm(qAttn, saw.AttnQNorm, numHeads, keyLen, config.NormEps);
                    backend.PerHeadRmsNorm(kProj, saw.AttnKNorm!, numKvHeads, keyLen, config.NormEps);
                }

                backend.RoPE(qAttn, kProj, keyLen, config.RopeDimCount, 0, config.RopeTheta);
                kvCache.Write(backend, layer, 0, kProj, vProj);

                var kCache = kvCache.GetKCacheTensor(layer);
                var vCache = kvCache.GetVCacheTensor(layer);
                float scale = 1.0f / MathF.Sqrt(keyLen);

                // For gated attention, create a gate tensor filled with 88 (sigmoid≈1)
                var qGate = backend.CreateTensor("qg", GgmlType.F32, [numHeads * keyLen]);
                backend.FillTensor(qGate, 88.0f);

                backend.GatedAttention(attnOut, qAttn, qGate, kCache, vCache,
                    numHeads, numKvHeads, keyLen, valLen, kvCache.MaxSeqLen, kvCache.Length, scale);

                backend.MatMul(hidden, attnOut, saw.AttnO, 1,
                    numHeads * valLen, config.HiddenDim);

                qAttn.Dispose(); kProj.Dispose(); vProj.Dispose(); attnOut.Dispose(); qGate.Dispose();
            }

            // Residual add
            backend.ElementAdd(hidden, hidden, residual);

            // FFN
            backend.CopyTensor(residual, hidden);
            backend.RmsNorm(normOut, hidden, lw.PostAttnNorm, config.NormEps);
            backend.MatMul(gate, normOut, lw.FfnGate, 1, config.HiddenDim, config.IntermediateDim);
            backend.MatMul(up, normOut, lw.FfnUp, 1, config.HiddenDim, config.IntermediateDim);
            backend.SiLU(gate, gate);
            backend.ElementMul(gate, gate, up);
            backend.MatMul(hidden, gate, lw.FfnDown, 1, config.IntermediateDim, config.HiddenDim);
            backend.ElementAdd(hidden, hidden, residual);
        }

        // Final norm + logits
        backend.RmsNorm(normOut, hidden, weights.OutputNorm, config.NormEps);
        backend.MatMul(logits, normOut, weights.OutputWeight, 1, config.HiddenDim, config.VocabSize);

        var logitData = new float[config.VocabSize];
        logits.DequantizeTo(logitData);

        float maxVal = float.MinValue;
        int argmax = 0;
        for (int i = 0; i < logitData.Length; i++)
            if (logitData[i] > maxVal) { maxVal = logitData[i]; argmax = i; }

        string topToken = tokenizer.Decode([argmax]);
        lines.Add($"Top token (DeltaNet skipped): '{topToken}' (id={argmax}, logit={maxVal:F4})");
        lines.Add($"Top 5: [{string.Join(", ", GetTopK(logitData, 5).Select(x => $"'{tokenizer.Decode([x.idx])}':{x.val:F2}"))}]");

        File.WriteAllLines(@"C:\GGUFS\deltanet-skip.txt", lines);
        foreach (var line in lines)
            Console.WriteLine(line);

        hidden.Dispose(); residual.Dispose(); normOut.Dispose();
        logits.Dispose(); gate.Dispose(); up.Dispose();
        weights.Dispose();
    }

    /// <summary>
    /// Test the DeltaNet computation in isolation: check L2 norm, split, repeat-interleave,
    /// and the actual DeltaNet step output for the 9B model.
    /// </summary>
    [Fact]
    public void Diagnostic_9B_DeltaNetStepIsolated()
    {
        if (!File.Exists(Qwen35_9B)) return;

        using var stream = File.OpenRead(Qwen35_9B);
        var gguf = GgufFile.Read(stream);
        var config = ModelConfig.FromGguf(gguf);
        using var backend = new CpuBackend();
        var weights = ModelLoader.Load(gguf, stream, backend, config);
        var tokenizer = TokenizerFactory.FromGguf(gguf);

        var tokens = tokenizer.Encode("Hello");
        int tokenId = tokens[0];

        var lines = new List<string>();
        lines.Add("=== 9B DeltaNet Step Isolated ===");

        // Get first DeltaNet layer
        DeltaNetWeights? dnw = null;
        int dnLayer = -1;
        for (int i = 0; i < config.NumLayers; i++)
        {
            if (!config.IsStandardAttention(i) && weights.Layers[i] is DeltaNetWeights dw)
            {
                dnw = dw;
                dnLayer = i;
                break;
            }
        }
        Assert.NotNull(dnw);

        int headDim = config.SsmStateSize > 0 ? config.SsmStateSize : config.SsmHeadDim;
        int numVHeads = (int)dnw.SsmAlpha.Dimensions[1];
        int qkvOutDim = (int)dnw.AttnQkv.Dimensions[1];
        int valueDim = numVHeads * headDim;
        int keyDim = (qkvOutDim - valueDim) / 2;
        int numKHeads = keyDim / headDim;
        int repeatFactor = numVHeads / numKHeads;

        lines.Add($"Layer {dnLayer}: numKHeads={numKHeads}, numVHeads={numVHeads}, headDim={headDim}, keyDim={keyDim}, valueDim={valueDim}");

        // 1. Embedding + norm
        var hidden = backend.CreateTensor("h", GgmlType.F32, [config.HiddenDim]);
        var normOut = backend.CreateTensor("n", GgmlType.F32, [config.HiddenDim]);
        backend.EmbeddingLookup(hidden, weights.TokenEmbedding, tokenId);
        backend.RmsNorm(normOut, hidden, dnw.AttnNorm, config.NormEps);

        // 2. QKV projection
        var qkvBuf = backend.CreateTensor("qkv", GgmlType.F32, [qkvOutDim]);
        backend.MatMul(qkvBuf, normOut, dnw.AttnQkv, 1, config.HiddenDim, qkvOutDim);

        // 3. Conv1d (skip for first token - buffer is zero, result = input * weight[last])
        int convKernel = config.SsmConvKernel;
        int convChannels = (int)(dnw.SsmConv1d.ElementCount / convKernel);
        var convBuf = backend.CreateTensor("cb", GgmlType.F32, [(convKernel - 1) * convChannels]);
        backend.CausalConv1d(qkvBuf, convBuf, dnw.SsmConv1d, convChannels, convKernel);

        // 4. SiLU
        backend.SiLUInPlace(qkvBuf);

        // 5. Split
        var ssmQ = backend.CreateTensor("q", GgmlType.F32, [valueDim]);
        var ssmK = backend.CreateTensor("k", GgmlType.F32, [valueDim]);
        var ssmV = backend.CreateTensor("v", GgmlType.F32, [valueDim]);

        // Manual split
        var qkvData = new float[qkvOutDim];
        qkvBuf.DequantizeTo(qkvData);

        var qBuf = new byte[valueDim * sizeof(float)];
        var kBuf = new byte[valueDim * sizeof(float)];
        var vBuf = new byte[valueDim * sizeof(float)];
        Buffer.BlockCopy(qkvData, 0, qBuf, 0, keyDim * sizeof(float));
        Buffer.BlockCopy(qkvData, keyDim * sizeof(float), kBuf, 0, keyDim * sizeof(float));
        Buffer.BlockCopy(qkvData, keyDim * 2 * sizeof(float), vBuf, 0, valueDim * sizeof(float));
        ssmQ.CopyFrom(qBuf);
        ssmK.CopyFrom(kBuf);
        ssmV.CopyFrom(vBuf);

        // Check Q, K, V norms before L2 norm
        lines.Add($"Pre-L2norm Q head 0 L2: {L2Norm(ssmQ.AsFloatSpan().Slice(0, headDim)):F4}");
        lines.Add($"Pre-L2norm K head 0 L2: {L2Norm(ssmK.AsFloatSpan().Slice(0, headDim)):F4}");
        lines.Add($"V head 0 L2: {L2Norm(ssmV.AsFloatSpan().Slice(0, headDim)):F4}");

        // 6. L2 normalize
        backend.L2NormGroups(ssmQ, numKHeads, headDim);
        backend.L2NormGroups(ssmK, numKHeads, headDim);
        lines.Add($"Post-L2norm Q head 0 L2: {L2Norm(ssmQ.AsFloatSpan().Slice(0, headDim)):F4}");
        lines.Add($"Post-L2norm K head 0 L2: {L2Norm(ssmK.AsFloatSpan().Slice(0, headDim)):F4}");

        // 7. Repeat interleave
        if (repeatFactor > 1)
        {
            RepeatInterleaveInPlace(ssmQ, numKHeads, headDim, repeatFactor);
            RepeatInterleaveInPlace(ssmK, numKHeads, headDim, repeatFactor);
        }
        lines.Add($"Post-repeat Q head 0 L2: {L2Norm(ssmQ.AsFloatSpan().Slice(0, headDim)):F4}");
        lines.Add($"Post-repeat Q head 1 L2: {L2Norm(ssmQ.AsFloatSpan().Slice(headDim, headDim)):F4}");
        float dot01 = DotProduct(ssmQ.AsFloatSpan().Slice(0, headDim), ssmQ.AsFloatSpan().Slice(headDim, headDim));
        lines.Add($"Q head 0 · Q head 1 (should be 1.0): {dot01:F6}");

        // 8. Compute decay/beta
        var alphaTensor = backend.CreateTensor("a", GgmlType.F32, [numVHeads]);
        var betaTensor = backend.CreateTensor("b", GgmlType.F32, [numVHeads]);
        var decayTensor = backend.CreateTensor("d", GgmlType.F32, [numVHeads]);
        var betaValTensor = backend.CreateTensor("bv", GgmlType.F32, [numVHeads]);
        backend.MatMul(alphaTensor, normOut, dnw.SsmAlpha, 1, config.HiddenDim, numVHeads);
        backend.MatMul(betaTensor, normOut, dnw.SsmBeta, 1, config.HiddenDim, numVHeads);
        backend.ComputeDecayBeta(decayTensor, betaValTensor, alphaTensor, betaTensor,
            dnw.SsmA, dnw.SsmDtBias, numVHeads);

        var decay = decayTensor.AsFloatSpan();
        var betaVal = betaValTensor.AsFloatSpan();
        lines.Add($"Decay[0..3]: [{decay[0]:F6}, {decay[1]:F6}, {decay[2]:F6}, {decay[3]:F6}]");
        lines.Add($"BetaVal[0..3]: [{betaVal[0]:F4}, {betaVal[1]:F4}, {betaVal[2]:F4}, {betaVal[3]:F4}]");

        // 9. DeltaNet step (with zero initial state)
        var stateTensor = backend.CreateTensor("state", GgmlType.F32, [numVHeads * headDim * headDim]);
        var output = backend.CreateTensor("out", GgmlType.F32, [valueDim]);

        // Test with scale = 1.0 (no scaling) first
        backend.DeltaNetStep(output, ssmQ, ssmK, ssmV,
            stateTensor, decayTensor, betaValTensor,
            dnw.SsmNorm, numVHeads, headDim, 1.0f, config.NormEps);
        lines.Add($"DeltaNet output (scale=1.0): {DumpStatsSpan(output.AsFloatSpan())}");

        // Reset state and test with scale = 1/sqrt(headDim)
        backend.ZeroTensor(stateTensor);
        float scale = 1.0f / MathF.Sqrt(headDim);
        backend.DeltaNetStep(output, ssmQ, ssmK, ssmV,
            stateTensor, decayTensor, betaValTensor,
            dnw.SsmNorm, numVHeads, headDim, scale, config.NormEps);
        lines.Add($"DeltaNet output (scale=1/√d): {DumpStatsSpan(output.AsFloatSpan())}");

        // 10. Gate
        var gateTensor = backend.CreateTensor("gate", GgmlType.F32, [valueDim]);
        backend.MatMul(gateTensor, normOut, dnw.AttnGate, 1, config.HiddenDim, valueDim);
        lines.Add($"Gate: {DumpStatsSpan(gateTensor.AsFloatSpan())}");

        // Apply gate
        backend.SiLUGate(output, output, gateTensor);
        lines.Add($"After SiLUGate: {DumpStatsSpan(output.AsFloatSpan())}");

        // 11. Output projection
        var finalOut = backend.CreateTensor("final", GgmlType.F32, [config.HiddenDim]);
        backend.MatMul(finalOut, output, dnw.SsmOut, 1, valueDim, config.HiddenDim);
        lines.Add($"Output proj: {DumpStatsSpan(finalOut.AsFloatSpan())}");

        // === Test: swap alpha/beta and compare ===
        // The GGUF might have alpha/beta swapped (in_proj_ba outputs [beta, alpha] in HuggingFace)
        backend.ZeroTensor(stateTensor);
        var decaySwap = backend.CreateTensor("ds", GgmlType.F32, [numVHeads]);
        var betaSwap = backend.CreateTensor("bs", GgmlType.F32, [numVHeads]);
        // Swap: use betaTensor as alpha, alphaTensor as beta
        backend.ComputeDecayBeta(decaySwap, betaSwap, betaTensor, alphaTensor,
            dnw.SsmA, dnw.SsmDtBias, numVHeads);
        lines.Add($"SWAPPED Decay[0..3]: [{decaySwap.AsFloatSpan()[0]:F6}, {decaySwap.AsFloatSpan()[1]:F6}, {decaySwap.AsFloatSpan()[2]:F6}, {decaySwap.AsFloatSpan()[3]:F6}]");
        lines.Add($"SWAPPED BetaVal[0..3]: [{betaSwap.AsFloatSpan()[0]:F4}, {betaSwap.AsFloatSpan()[1]:F4}, {betaSwap.AsFloatSpan()[2]:F4}, {betaSwap.AsFloatSpan()[3]:F4}]");

        backend.DeltaNetStep(output, ssmQ, ssmK, ssmV,
            stateTensor, decaySwap, betaSwap,
            dnw.SsmNorm, numVHeads, headDim, scale, config.NormEps);
        backend.SiLUGate(output, output, gateTensor);
        backend.MatMul(finalOut, output, dnw.SsmOut, 1, valueDim, config.HiddenDim);
        lines.Add($"SWAPPED output proj: {DumpStatsSpan(finalOut.AsFloatSpan())}");

        decaySwap.Dispose();
        betaSwap.Dispose();

        File.WriteAllLines(@"C:\GGUFS\deltanet-step-isolated.txt", lines);
        foreach (var line in lines)
            Console.WriteLine(line);

        // Cleanup
        foreach (var t in new ITensor[] { hidden, normOut, qkvBuf, convBuf, ssmQ, ssmK, ssmV,
            alphaTensor, betaTensor, decayTensor, betaValTensor, stateTensor, output, gateTensor, finalOut })
            t.Dispose();
        weights.Dispose();
    }

    private static float L2Norm(Span<float> data)
    {
        float sum = 0;
        for (int i = 0; i < data.Length; i++) sum += data[i] * data[i];
        return MathF.Sqrt(sum);
    }

    private static float DotProduct(Span<float> a, Span<float> b)
    {
        float sum = 0;
        for (int i = 0; i < a.Length; i++) sum += a[i] * b[i];
        return sum;
    }

    private static void RepeatInterleaveInPlace(ITensor tensor, int numHeads, int headDim, int factor)
    {
        var src = tensor.AsFloatSpan();
        for (int h = numHeads - 1; h >= 0; h--)
        {
            int srcOff = h * headDim;
            for (int r = factor - 1; r >= 0; r--)
            {
                int dstOff = (h * factor + r) * headDim;
                src.Slice(srcOff, headDim).CopyTo(src.Slice(dstOff, headDim));
            }
        }
    }

    private static string DumpStatsSpan(Span<float> data)
    {
        float sum = 0, min = float.MaxValue, max = float.MinValue;
        for (int i = 0; i < data.Length; i++)
        {
            sum += data[i];
            if (data[i] < min) min = data[i];
            if (data[i] > max) max = data[i];
        }
        return $"sum={sum:F4}, min={min:F4}, max={max:F4}, len={data.Length}";
    }

    private static string DumpStats(ITensor tensor)
    {
        var data = tensor.AsFloatSpan();
        float sum = 0, min = float.MaxValue, max = float.MinValue;
        int nanCount = 0;
        for (int i = 0; i < data.Length; i++)
        {
            float v = data[i];
            if (float.IsNaN(v)) { nanCount++; continue; }
            sum += v;
            if (v < min) min = v;
            if (v > max) max = v;
        }
        return $"sum={sum:F4}, min={min:F4}, max={max:F4}, len={data.Length}{(nanCount > 0 ? $", NaN={nanCount}" : "")}";
    }

    private static (int idx, float val)[] GetTopK(ReadOnlySpan<float> data, int k)
    {
        var top = new (int idx, float val)[k];
        for (int i = 0; i < k; i++) top[i] = (-1, float.MinValue);
        for (int i = 0; i < data.Length; i++)
        {
            if (data[i] > top[k - 1].val)
            {
                top[k - 1] = (i, data[i]);
                Array.Sort(top, (a, b) => b.val.CompareTo(a.val));
            }
        }
        return top;
    }
}
