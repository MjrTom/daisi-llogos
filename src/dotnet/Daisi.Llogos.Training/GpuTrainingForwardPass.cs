using Daisi.Llogos.Cuda;
using Daisi.Llogos.Gguf;
using Daisi.Llogos.Model;
using Daisi.Llogos.Training.Lora;

namespace Daisi.Llogos.Training;

/// <summary>
/// Full GPU training forward + backward pass. All activations, gradients, and intermediates
/// stay on device. Only scalar loss and LoRA gradients are downloaded to CPU.
/// Requires CudaBackend.
/// </summary>
public sealed class GpuTrainingForwardPass : ITrainingForwardPass
{
    private readonly ModelConfig _config;
    private readonly ModelWeights _weights;
    private readonly LoraAdapter _adapter;
    private readonly CudaBackend _gpu;

    // Cached model-level tensors (dequantized once, stay on GPU)
    private ITensor? _embeddingTable;   // F32 [V × H] on GPU
    private ITensor? _outputNormWeight; // F32 [H] on GPU
    // Note: output weight stays quantized — use _gpu.MatMul for forward

    // Per-layer GPU saved activations for backward
    private GpuLayerActivations[]? _saved;

    // GPU tensor pool — pre-allocates on first call, reuses on subsequent calls.
    // Eliminates thousands of cuMemAlloc/cuMemFree per step.
    private readonly Dictionary<string, ITensor> _pool = new();
    private int _lastT;

    // LoRA tensors on GPU (uploaded from CPU, re-uploaded after optimizer step)
    private readonly Dictionary<string, (ITensor a, ITensor b)> _gpuLora = new();
    private readonly Dictionary<string, (ITensor dA, ITensor dB)> _gpuLoraGrad = new();
    // LoRA intermediates per layer (reused)
    private readonly Dictionary<string, ITensor> _gpuLoraIntermediate = new();
    // GPU optimizer state (AdamW m and v for each LoRA param)
    private readonly Dictionary<string, (ITensor mA, ITensor vA, ITensor mB, ITensor vB)> _gpuOptState = new();
    private int _optimStep;

    public GpuTrainingForwardPass(ModelConfig config, ModelWeights weights,
        LoraAdapter adapter, CudaBackend gpu)
    {
        _config = config;
        _weights = weights;
        _adapter = adapter;
        _gpu = gpu;

        // Upload LoRA weights to GPU
        foreach (var (name, layer) in adapter.Layers)
        {
            var gpuA = CreateF32Tensor($"lora.{name}.A", layer.A.Data);
            var gpuB = CreateF32Tensor($"lora.{name}.B", layer.B.Data);
            var gpuDA = Pool($"lora.{name}.dA", layer.A.Size);
            var gpuDB = Pool($"lora.{name}.dB", layer.B.Size);
            _gpuLora[name] = (gpuA, gpuB);
            _gpuLoraGrad[name] = (gpuDA, gpuDB);

            // Intermediate buffer for LoRA forward
            var inter = Pool($"lora.{name}.inter", 1); // resized later
            _gpuLoraIntermediate[name] = inter;

            // Optimizer state (m and v for AdamW, initialized to zero)
            var mA = Pool($"opt.{name}.mA", layer.A.Size);
            var vA = Pool($"opt.{name}.vA", layer.A.Size);
            var mB = Pool($"opt.{name}.mB", layer.B.Size);
            var vB = Pool($"opt.{name}.vB", layer.B.Size);
            _gpu.ZeroTensor(mA); _gpu.ZeroTensor(vA);
            _gpu.ZeroTensor(mB); _gpu.ZeroTensor(vB);
            _gpuOptState[name] = (mA, vA, mB, vB);
        }
    }

    public float[]? Forward(int[] tokenIds, out float totalLoss, int[] targets)
    {
        int T = tokenIds.Length;
        int H = _config.HiddenDim;
        int V = _config.VocabSize;


        // Dequantize embedding table (cached on GPU)
        if (_embeddingTable == null)
        {
            var embF32 = F32Ops.Dequantize(_weights.TokenEmbedding);
            _embeddingTable = CreateF32Tensor("emb_table", embF32);
        }
        if (_outputNormWeight == null)
        {
            var normF32 = F32Ops.Dequantize(_weights.OutputNorm);
            _outputNormWeight = CreateF32Tensor("out_norm", normF32);
        }

        // Allocate activation storage
        _saved = new GpuLayerActivations[_config.NumLayers];

        // 1. Embedding lookup → hidden [T × H] (GPU kernel)
        var gpuTokenIds = CreateIntTensor("token_ids", tokenIds);
        _gpu.BatchedEmbeddingLookup(Pool("hidden", T * H), _embeddingTable, gpuTokenIds, T, H);

        // 2. Transformer layers
        for (int layer = 0; layer < _config.NumLayers; layer++)
        {
            var lw = _weights.Layers[layer];

            // Pre-attention norm
            var normWeight = DequantToGpu($"l{layer}.attn_norm", lw.AttnNorm);
            _gpu.CopyTensor(Pool("residual", T * H), Pool("hidden", T * H));
            _gpu.RmsNorm(Pool("normout", T * H), Pool("hidden", T * H), normWeight, _config.NormEps);

            var saved = new GpuLayerActivations
            {
                InputToLayer = CloneGpuTensor($"l{layer}.input", Pool("residual", T * H)),
                NormOut = CloneGpuTensor($"l{layer}.normout", Pool("normout", T * H)),
                AttnNormWeight = normWeight,
            };

            if (lw is StandardAttentionWeights saw && _config.IsStandardAttention(layer))
                ForwardAttentionLayer(saw, saved, layer, T);
            else if (lw is DeltaNetWeights dnw)
                ForwardDeltaNetLayer(dnw, saved, layer, T);

            // Add residual
            _gpu.ElementAdd(Pool("hidden", T * H), Pool("hidden", T * H), Pool("residual", T * H));

            // Post-attention norm + FFN
            var postNormWeight = DequantToGpu($"l{layer}.post_norm", lw.PostAttnNorm);
            saved.PostAttnNormWeight = postNormWeight;
            saved.PostAttnHidden = CloneGpuTensor($"l{layer}.post_hidden", Pool("hidden", T * H));

            var ffnNormOut = Pool($"l{layer}.ffn_norm", T * H);
            _gpu.RmsNorm(ffnNormOut, Pool("hidden", T * H), postNormWeight, _config.NormEps);
            saved.FfnNormOut = ffnNormOut;
            saved.FfnResidual = CloneGpuTensor($"l{layer}.ffn_res", Pool("hidden", T * H));

            int I = _config.IntermediateDim;
            var gateOut = Pool($"l{layer}.gate", T * I);
            var upOut = Pool($"l{layer}.up", T * I);
            var ffnGated = Pool($"l{layer}.ffn_gated", T * I);

            _gpu.MatMul(gateOut, ffnNormOut, GpuWeight(lw.FfnGate), T, H, I);
            _gpu.MatMul(upOut, ffnNormOut, GpuWeight(lw.FfnUp), T, H, I);
            _gpu.SwiGLU(ffnGated, gateOut, upOut);

            var ffnOut = Pool($"l{layer}.ffn_out", T * H);
            _gpu.MatMul(ffnOut, ffnGated, GpuWeight(lw.FfnDown), T, I, H);

            saved.GateInput = gateOut;
            saved.UpInput = upOut;
            saved.FfnGated = ffnGated;

            // hidden = ffnOut + residual
            _gpu.ElementAdd(Pool("hidden", T * H), ffnOut, saved.FfnResidual!);

            _saved[layer] = saved;
        }

        // 3. Final norm + logit projection
        var finalNormOut = Pool("final_norm", T * H);
        _gpu.RmsNorm(finalNormOut, Pool("hidden", T * H), _outputNormWeight!, _config.NormEps);

        var logits = Pool("logits", T * V);
        _gpu.MatMul(logits, finalNormOut, GpuWeight(_weights.OutputWeight), T, H, V);

        // 4. Cross-entropy loss + gradient (on GPU, only downloads scalar loss)
        var dLogits = Pool("dlogits", T * V);
        var gpuTargets = CreateIntTensor("targets", targets);
        totalLoss = _gpu.CrossEntropyLoss(dLogits, logits, gpuTargets, T, V);

        // 5. Backward pass
        Backward(dLogits, finalNormOut, T);

        // Note: LoRA gradients stay on GPU — GpuOptimizerStep uses them directly

        return null; // logits not downloaded
    }

    private void ForwardAttentionLayer(StandardAttentionWeights saw, GpuLayerActivations saved,
        int layer, int T)
    {
        int H = _config.HiddenDim;
        int numHeads = _config.NumHeads;
        int numKvHeads = _config.NumKvHeads;
        int keyDim = _config.KeyLength;
        int valDim = _config.ValueLength;
        int ropeDim = _config.RopeDimCount;
        float scale = 1.0f / MathF.Sqrt(keyDim);
        bool hasGatedQ = saw.HasGatedQ;

        int qOutDim = hasGatedQ ? numHeads * keyDim * 2 : numHeads * keyDim;
        int kOutDim = numKvHeads * keyDim;
        int vOutDim = numKvHeads * valDim;

        var qRaw = Pool($"l{layer}.q", T * qOutDim);
        var kRaw = Pool($"l{layer}.k", T * kOutDim);
        var vRaw = Pool($"l{layer}.v", T * vOutDim);

        _gpu.MatMul(qRaw, saved.NormOut!, GpuWeight(saw.AttnQ), T, H, qOutDim);
        _gpu.MatMul(kRaw, saved.NormOut!, GpuWeight(saw.AttnK), T, H, kOutDim);
        _gpu.MatMul(vRaw, saved.NormOut!, GpuWeight(saw.AttnV), T, H, vOutDim);

        // Apply LoRA
        string prefix = $"blk.{layer}";
        ApplyLoraForward($"{prefix}.attn_q", saved.NormOut!, qRaw, T);
        ApplyLoraForward($"{prefix}.attn_k", saved.NormOut!, kRaw, T);
        ApplyLoraForward($"{prefix}.attn_v", saved.NormOut!, vRaw, T);

        // De-interleave gated Q if needed
        int qAttnDim = numHeads * keyDim;
        ITensor qAttn, qGate;
        if (hasGatedQ)
        {
            qAttn = Pool($"l{layer}.qattn", T * qAttnDim);
            qGate = Pool($"l{layer}.qgate", T * qAttnDim);
            // DeInterleaveQ exists on GPU backend — call per token row
            for (int t = 0; t < T; t++)
            {
                // Use backend's existing DeInterleaveQ (works on GPU tensors per-row)
                // For batched: offset into the tensor. Use CopyTensorSlice to extract rows.
                // Simpler: batch the whole thing on CPU (small T, fast)
                // TODO: add batched GPU DeInterleaveQ kernel
            }
            // CPU fallback for gated Q (only 6 layers, T is small)
            var qRawCpu = DownloadF32(qRaw, T * qOutDim);
            var qAttnCpu = new float[T * qAttnDim];
            var qGateCpu = new float[T * qAttnDim];
            for (int t = 0; t < T; t++)
                F32Ops.DeInterleaveQ(qAttnCpu.AsSpan(t * qAttnDim, qAttnDim),
                    qGateCpu.AsSpan(t * qAttnDim, qAttnDim),
                    qRawCpu.AsSpan(t * qOutDim, qOutDim), numHeads, keyDim);
            UploadF32(qAttn, qAttnCpu);
            UploadF32(qGate, qGateCpu);
        }
        else
        {
            qAttn = qRaw; // same tensor, no gate
            qGate = Pool($"l{layer}.qgate", T * qAttnDim);
            _gpu.FillTensor(qGate, 88.0f); // sigmoid(88) ≈ 1
        }

        // RoPE (batched)
        _gpu.BatchedRoPE(qAttn, kRaw, keyDim, ropeDim, 0, _config.RopeTheta, numHeads, numKvHeads);

        // Causal gated attention — GPU kernel (saves probs for backward)
        int attnOutDim = numHeads * valDim;
        var attnOut = Pool($"l{layer}.attn_out", T * attnOutDim);
        var savedProbs = Pool($"l{layer}.probs", numHeads * T * T);
        _gpu.TrainingCausalGatedAttention(attnOut, savedProbs, qAttn, qGate, kRaw, vRaw,
            T, numHeads, numKvHeads, keyDim, valDim, scale);

        // O projection
        int oDim = (int)saw.AttnO.Dimensions[0];
        var oOut = Pool($"l{layer}.oout", T * H);
        _gpu.MatMul(oOut, attnOut, GpuWeight(saw.AttnO), T, oDim, H);
        ApplyLoraForward($"{prefix}.attn_o", attnOut, oOut, T);

        // Write to hidden
        _gpu.CopyTensor(Pool("hidden", T * H), oOut);

        saved.IsAttention = true;
        saved.HasGatedQ = hasGatedQ;
        saved.QRaw = qRaw; saved.KRaw = kRaw; saved.VRaw = vRaw;
        saved.QAttn = qAttn; saved.QGate = qGate;
        saved.AttnOut = attnOut;
        saved.SavedProbs = savedProbs;
    }

    private void ForwardDeltaNetLayer(DeltaNetWeights dnw, GpuLayerActivations saved,
        int layer, int T)
    {
        // Real DeltaNet forward — uses the same kernels as inference.
        // Processes tokens sequentially (DeltaNet is recurrent).
        // No simplified approximation — correct hidden states for FFN LoRA.
        int H = _config.HiddenDim;
        int convKernel = _config.SsmConvKernel;
        // Derive dimensions from weight tensors (matching inference ForwardDeltaNet exactly)
        int headDim = _config.SsmStateSize > 0 ? _config.SsmStateSize : _config.SsmHeadDim;
        int numVHeads = (int)dnw.SsmAlpha.Dimensions[1];
        int valueDim = numVHeads * headDim;
        int qkvDim = (int)dnw.AttnQkv.Dimensions[1];
        int keyDim = (qkvDim - valueDim) / 2;
        int numKHeads = keyDim / headDim;
        int repeatFactor = numVHeads / Math.Max(numKHeads, 1);
        float scale = 1.0f / MathF.Sqrt(headDim);
        int convChannels = (int)(dnw.SsmConv1d.ElementCount / convKernel);

        // State and conv buffer — reset per sequence
        // Debug: print dimensions on first call
        if (layer <= 1 && !_pool.ContainsKey($"delta.state.{layer}"))
        {
            Console.Error.WriteLine($"  DeltaNet L{layer}: H={H} headDim={headDim} numVHeads={numVHeads} " +
                $"valueDim={valueDim} qkvDim={qkvDim} keyDim={keyDim} numKHeads={numKHeads} " +
                $"repeat={repeatFactor} convK={convKernel} convCh={convChannels} " +
                $"SsmAlpha.Dims=[{string.Join(",", dnw.SsmAlpha.Dimensions.ToArray())}] " +
                $"AttnQkv.Dims=[{string.Join(",", dnw.AttnQkv.Dimensions.ToArray())}]");
        }

        var state = PoolZ($"delta.state.{layer}", numVHeads * headDim * headDim);
        var convBuf = PoolZ($"delta.conv.{layer}", (convKernel - 1) * convChannels);

        // Single-token scratch tensors (reused across tokens)
        var qkvBuf = Pool("delta.qkv_tok", qkvDim);
        var ssmQ = Pool("delta.q", valueDim);
        var ssmK = Pool("delta.k", valueDim);
        var ssmV = Pool("delta.v", valueDim);
        var ssmAlpha = Pool("delta.alpha", numVHeads);
        var ssmBeta = Pool("delta.beta", numVHeads);
        var ssmDecay = Pool("delta.decay", numVHeads);
        var ssmBetaVal = Pool("delta.betaval", numVHeads);
        var ssmOutput = Pool("delta.output", valueDim);
        var ssmGate = Pool("delta.gate", valueDim);
        var normOut1 = Pool("delta.normout1", H);
        var hidden1 = Pool("delta.hidden1", H);

        var hiddenBuf = Pool("hidden", T * H);

        // Process each token sequentially
        for (int t = 0; t < T; t++)
        {
            // Extract single token's normOut
            _gpu.CopyTensorSlice(normOut1, 0, saved.NormOut!, t * H, H);

            // Step 1: QKV projection
            _gpu.MatMul(qkvBuf, normOut1, GpuWeight(dnw.AttnQkv), 1, H, qkvDim);

            // Step 2: CausalConv1d (in-place on qkvBuf)
            _gpu.CausalConv1d(qkvBuf, convBuf, GpuWeight(dnw.SsmConv1d), convChannels, convKernel);

            // Step 3: SiLU in-place
            _gpu.SiLUInPlace(qkvBuf);

            // Step 4: Split QKV
            if (keyDim == valueDim)
                _gpu.SplitQKV(ssmQ, ssmK, ssmV, qkvBuf, keyDim);
            else
                _gpu.SplitUnequalQKV(ssmQ, ssmK, ssmV, qkvBuf, keyDim, valueDim);

            // Step 5: L2 normalize Q and K by group
            _gpu.L2NormGroups(ssmQ, numKHeads, headDim);
            _gpu.L2NormGroups(ssmK, numKHeads, headDim);

            // Step 6: Repeat-tile if needed (GQA expansion)
            if (repeatFactor > 1)
            {
                _gpu.RepeatTile(ssmQ, numKHeads, headDim, repeatFactor);
                _gpu.RepeatTile(ssmK, numKHeads, headDim, repeatFactor);
            }

            // Step 7: Alpha/Beta projections
            _gpu.MatMul(ssmAlpha, normOut1, GpuWeight(dnw.SsmAlpha), 1, H, numVHeads);
            _gpu.MatMul(ssmBeta, normOut1, GpuWeight(dnw.SsmBeta), 1, H, numVHeads);

            // Step 8: Compute decay and beta values
            _gpu.ComputeDecayBeta(ssmDecay, ssmBetaVal, ssmAlpha, ssmBeta,
                GpuWeight(dnw.SsmA), GpuWeight(dnw.SsmDtBias), numVHeads);

            // Step 9: DeltaNet state update + output (the core SSM operation)
            _gpu.DeltaNetStep(ssmOutput, ssmQ, ssmK, ssmV,
                state, ssmDecay, ssmBetaVal,
                GpuWeight(dnw.SsmNorm), numVHeads, headDim, scale, _config.NormEps);

            // Step 10: Gate projection + SiLU gate
            _gpu.MatMul(ssmGate, normOut1, GpuWeight(dnw.AttnGate), 1, H, valueDim);
            _gpu.SiLUGate(ssmOutput, ssmOutput, ssmGate);

            // Step 11: Output projection → single token hidden
            _gpu.MatMul(hidden1, ssmOutput, GpuWeight(dnw.SsmOut), 1, valueDim, H);

            // Store output at position t
            _gpu.CopyTensorSlice(hiddenBuf, t * H, hidden1, 0, H);
        }

        saved.IsDeltaNet = true;
        // No DeltaNet-specific activations saved — backward passes gradient through residual.
        // The correct forward hidden states ensure FFN LoRA learns properly.
    }

    private void Backward(ITensor dLogits, ITensor finalNormOut, int T)
    {
        int H = _config.HiddenDim;
        int V = _config.VocabSize;

        // dFinalNormOut = dLogits × outputWeight
        var dFinalNormOut = Pool("d_fnorm", T * H);
        // Need F32 output weight for backward matmul
        var outWeightF32 = DequantToGpu("out_weight_f32", _weights.OutputWeight);
        _gpu.SgemmF32(dFinalNormOut, dLogits, outWeightF32, T, V, H);

        // Backward through final RMS norm
        _gpu.ZeroTensor(Pool("dhidden", T * H));
        _gpu.BatchedRmsNormBackward(Pool("dhidden", T * H), dFinalNormOut, Pool("hidden", T * H), _outputNormWeight!,
            _config.NormEps, H, T);

        // Backward through layers in reverse
        for (int layer = _config.NumLayers - 1; layer >= 0; layer--)
        {
            var saved = _saved![layer];
            var lw = _weights.Layers[layer];
            int I = _config.IntermediateDim;

            // FFN backward
            var dFfnOut = CloneGpuTensor($"d_l{layer}.ffn_out", Pool("dhidden", T * H));

            // Down projection backward
            var downF32 = DequantToGpu($"d_l{layer}.down", lw.FfnDown);
            var dFfnGated = Pool($"d_l{layer}.ffn_gated", T * I);
            _gpu.SgemmF32(dFfnGated, dFfnOut, downF32, T, H, I);

            // SwiGLU backward
            var dGate = Pool($"d_l{layer}.gate", T * I);
            var dUp = Pool($"d_l{layer}.up", T * I);
            _gpu.ZeroTensor(dGate);
            _gpu.ZeroTensor(dUp);
            _gpu.SwiGLUBackward(dGate, dUp, dFfnGated, saved.GateInput!, saved.UpInput!);

            // Gate/up projection backward
            var dFfnNormOut = Pool($"d_l{layer}.ffn_norm", T * H);
            _gpu.ZeroTensor(dFfnNormOut);
            var gateF32 = DequantToGpu($"d_l{layer}.gate_w", lw.FfnGate);
            var upF32 = DequantToGpu($"d_l{layer}.up_w", lw.FfnUp);
            var dFromGate = Pool($"d_l{layer}.from_gate", T * H);
            var dFromUp = Pool($"d_l{layer}.from_up", T * H);
            _gpu.SgemmF32(dFromGate, dGate, gateF32, T, I, H);
            _gpu.AddInPlace(dFfnNormOut, dFromGate);
            _gpu.SgemmF32(dFromUp, dUp, upF32, T, I, H);
            _gpu.AddInPlace(dFfnNormOut, dFromUp);

            // Post-attention RmsNorm backward
            var dPostAttnHidden = Pool($"d_l{layer}.post_attn", T * H);
            _gpu.ZeroTensor(dPostAttnHidden);
            _gpu.BatchedRmsNormBackward(dPostAttnHidden, dFfnNormOut,
                saved.PostAttnHidden!, saved.PostAttnNormWeight!, _config.NormEps, H, T);

            // Add residual gradient
            _gpu.AddInPlace(dPostAttnHidden, dFfnOut);

            var dAttnOut = CloneGpuTensor($"d_l{layer}.attn", dPostAttnHidden);

            // Layer-specific backward
            if (saved.IsAttention)
                BackwardAttentionLayer(saved, dAttnOut, layer, T);
            // DeltaNet: gradient passes through residual only.
            // Real DeltaNet forward produces correct hidden states for FFN,
            // but we don't backprop through the SSM (no DeltaNet LoRA needed).

            // Pre-attention RmsNorm backward
            var dInputToLayer = Pool($"d_l{layer}.input", T * H);
            _gpu.ZeroTensor(dInputToLayer);
            if (saved.DNormOut != null)
            {
                _gpu.BatchedRmsNormBackward(dInputToLayer, saved.DNormOut,
                    saved.InputToLayer!, saved.AttnNormWeight!, _config.NormEps, H, T);
            }

            // Combine: dHidden = dResidual + dInputToLayer
            _gpu.ElementAdd(Pool("dhidden", T * H), dPostAttnHidden, dInputToLayer);

            // Release saved activations
            saved.Dispose();
            _saved[layer] = null!;
        }
    }

    private void BackwardAttentionLayer(GpuLayerActivations saved, ITensor dOOut, int layer, int T)
    {
        int H = _config.HiddenDim;
        int numHeads = _config.NumHeads;
        int numKvHeads = _config.NumKvHeads;
        int keyDim = _config.KeyLength;
        int valDim = _config.ValueLength;
        int ropeDim = _config.RopeDimCount;
        float scale = 1.0f / MathF.Sqrt(keyDim);
        string prefix = $"blk.{layer}";
        var saw = (StandardAttentionWeights)_weights.Layers[layer];

        int oDim = (int)saved.AttnOut!.ElementCount / T;
        int qAttnDim = numHeads * keyDim;
        int kOutDim = numKvHeads * keyDim;
        int vOutDim = numKvHeads * valDim;

        // O LoRA backward
        ApplyLoraBackward($"{prefix}.attn_o", saved.AttnOut!, dOOut, T);

        // O projection backward: dAttnOut = dOOut × wO
        var oF32 = DequantToGpu($"d_l{layer}.oW", saw.AttnO);
        var dAttnOutFromO = Pool($"d_l{layer}.attn_out", T * oDim);
        _gpu.SgemmF32(dAttnOutFromO, dOOut, oF32, T, H, oDim);

        // Attention backward — GPU kernel
        var dQAttn = Pool($"d_l{layer}.dqattn", T * qAttnDim);
        var dQGate = Pool($"d_l{layer}.dqgate", T * qAttnDim);
        var dK = Pool($"d_l{layer}.dk", T * kOutDim);
        var dV = Pool($"d_l{layer}.dv", T * vOutDim);
        _gpu.ZeroTensor(dQAttn); _gpu.ZeroTensor(dQGate);
        _gpu.ZeroTensor(dK); _gpu.ZeroTensor(dV);
        _gpu.CausalGatedAttentionBackward(dQAttn, dQGate, dK, dV,
            dAttnOutFromO, saved.QAttn!, saved.QGate!,
            saved.KRaw!, saved.VRaw!, saved.SavedProbs!, saved.AttnOut!,
            T, numHeads, numKvHeads, keyDim, valDim, scale);

        // RoPE backward — GPU kernel
        _gpu.BatchedRoPEBackward(dQAttn, T, numHeads, keyDim, ropeDim, 0, _config.RopeTheta);
        _gpu.BatchedRoPEBackward(dK, T, numKvHeads, keyDim, ropeDim, 0, _config.RopeTheta);

        // De-interleave Q backward (CPU fallback for gated Q, GPU for non-gated)
        int qRawDim = saved.HasGatedQ ? numHeads * keyDim * 2 : qAttnDim;
        ITensor dQRaw;
        if (saved.HasGatedQ)
        {
            var dQAttnCpu = DownloadF32(dQAttn, T * qAttnDim);
            var dQGateCpu = DownloadF32(dQGate, T * qAttnDim);
            var dQRawCpu = new float[T * qRawDim];
            for (int t = 0; t < T; t++)
                F32Ops.DeInterleaveQBackward(dQRawCpu.AsSpan(t * qRawDim, qRawDim),
                    dQAttnCpu.AsSpan(t * qAttnDim, qAttnDim),
                    dQGateCpu.AsSpan(t * qAttnDim, qAttnDim), numHeads, keyDim);
            dQRaw = CreateF32Tensor($"d_l{layer}.dq", dQRawCpu);
        }
        else
        {
            dQRaw = dQAttn; // same tensor, no gate
        }

        var dNormOut = Pool($"d_l{layer}.dnorm", T * H);
        _gpu.ZeroTensor(dNormOut);

        var qF32 = DequantToGpu($"d_l{layer}.qW", saw.AttnQ);
        var kF32 = DequantToGpu($"d_l{layer}.kW", saw.AttnK);
        var vF32 = DequantToGpu($"d_l{layer}.vW", saw.AttnV);

        var dFromQ = Pool($"d_l{layer}.fq", T * H);
        var dFromK = Pool($"d_l{layer}.fk", T * H);
        var dFromV = Pool($"d_l{layer}.fv", T * H);
        _gpu.SgemmF32(dFromQ, dQRaw, qF32, T, qRawDim, H);
        _gpu.AddInPlace(dNormOut, dFromQ);
        _gpu.SgemmF32(dFromK, dK, kF32, T, kOutDim, H);
        _gpu.AddInPlace(dNormOut, dFromK);
        _gpu.SgemmF32(dFromV, dV, vF32, T, vOutDim, H);
        _gpu.AddInPlace(dNormOut, dFromV);

        // LoRA backward for Q, K, V
        ApplyLoraBackward($"{prefix}.attn_q", saved.NormOut!, dQRaw, T);
        ApplyLoraBackward($"{prefix}.attn_k", saved.NormOut!, dK, T);
        ApplyLoraBackward($"{prefix}.attn_v", saved.NormOut!, dV, T);


        saved.DNormOut = dNormOut;
    }

    private void BackwardDeltaNetLayer(GpuLayerActivations saved, ITensor dLayerOut, int layer, int T)
    {
        int H = _config.HiddenDim;
        string prefix = $"blk.{layer}";
        var dnw = (DeltaNetWeights)_weights.Layers[layer];
        int outInDim = (int)dnw.SsmOut.Dimensions[0];
        int qkvDim = (int)dnw.AttnQkv.Dimensions[1];
        int gateDim = (int)dnw.AttnGate.Dimensions[1];

        // SsmOut LoRA backward
        ApplyLoraBackward($"{prefix}.delta_out", saved.DeltaSsmApprox!, dLayerOut, T);

        // SsmOut backward
        var outF32 = DequantToGpu($"d_l{layer}.ssmW", dnw.SsmOut);
        var dSsmApprox = Pool($"d_l{layer}.dssm", T * outInDim);
        _gpu.SgemmF32(dSsmApprox, dLayerOut, outF32, T, H, outInDim);

        // Element-wise product backward (GPU kernel handles dimension mismatch)
        var dQkvPost = Pool($"d_l{layer}.dqkv_post", T * qkvDim);
        var dGatePost = Pool($"d_l{layer}.dgate_post", T * gateDim);
        _gpu.ZeroTensor(dQkvPost);
        _gpu.ZeroTensor(dGatePost);
        _gpu.TruncatedElementMulBackward(dQkvPost, dGatePost, dSsmApprox,
            saved.DeltaQkvPostSilu!, saved.DeltaGatePostSilu!, T, outInDim, qkvDim, gateDim);

        // SiLU backward (GPU)
        var dQkvPre = Pool($"d_l{layer}.dqkv", T * qkvDim);
        _gpu.ZeroTensor(dQkvPre);
        _gpu.SiLUBackward(dQkvPre, dQkvPost, saved.DeltaQkvPreSilu!);

        var dGatePre = Pool($"d_l{layer}.dgate", T * gateDim);
        _gpu.ZeroTensor(dGatePre);
        _gpu.SiLUBackward(dGatePre, dGatePost, saved.DeltaGatePreSilu!);

        // LoRA backward for AttnQkv
        ApplyLoraBackward($"{prefix}.delta_qkv", saved.NormOut!, dQkvPre, T);

        // Projection backward → dNormOut
        var dNormOut = Pool($"d_l{layer}.dnorm", T * H);
        _gpu.ZeroTensor(dNormOut);

        var qkvF32 = DequantToGpu($"d_l{layer}.qkvW", dnw.AttnQkv);
        var gateF32 = DequantToGpu($"d_l{layer}.gateW", dnw.AttnGate);
        var dFromQkv = Pool($"d_l{layer}.fqkv", T * H);
        var dFromGate = Pool($"d_l{layer}.fgate", T * H);
        _gpu.SgemmF32(dFromQkv, dQkvPre, qkvF32, T, qkvDim, H);
        _gpu.AddInPlace(dNormOut, dFromQkv);
        _gpu.SgemmF32(dFromGate, dGatePre, gateF32, T, gateDim, H);
        _gpu.AddInPlace(dNormOut, dFromGate);

        saved.DNormOut = dNormOut;
    }

    // ── LoRA helpers ────────────────────────────────────────────────────────

    private void ApplyLoraForward(string name, ITensor input, ITensor output, int T)
    {
        if (!_gpuLora.TryGetValue(name, out var lora)) return;
        var layer = _adapter.GetLayer(name)!;
        int inDim = layer.InFeatures;
        int outDim = layer.OutFeatures;
        int rank = layer.Rank;

        // Ensure intermediate buffer is correct size
        int interSize = T * rank;
        var inter = EnsureLoraIntermediate(name, interSize);

        // intermediate = input × A^T → [T × rank]
        _gpu.SgemmTransB(inter, input, lora.a, T, inDim, rank);
        // loraOut = intermediate × B^T → [T × outDim]
        var loraOut = Pool($"lora.{name}.out", T * outDim);
        _gpu.SgemmTransB(loraOut, inter, lora.b, T, rank, outDim);
        // output += scaling * loraOut
        _gpu.ScaleInPlace(loraOut, layer.Scaling);
        _gpu.AddInPlace(output, loraOut);
    }

    private void ApplyLoraBackward(string name, ITensor input, ITensor dOutput, int T)
    {
        if (!_gpuLora.TryGetValue(name, out var lora)) return;
        if (!_gpuLoraGrad.TryGetValue(name, out var grad)) return;
        var layer = _adapter.GetLayer(name)!;
        int inDim = layer.InFeatures;
        int outDim = layer.OutFeatures;
        int rank = layer.Rank;

        // Scale dOutput for LoRA path
        var scaledDOut = CloneGpuTensor($"lora.{name}.sdout", dOutput);
        _gpu.ScaleInPlace(scaledDOut, layer.Scaling);

        // dB += scaledDOut^T × intermediate → [outDim × rank]
        var inter = _gpuLoraIntermediate[name];
        _gpu.SgemmTransAAccumulate(grad.dB, scaledDOut, inter, outDim, T, rank);

        // dIntermediate = scaledDOut × B → [T × rank]
        var dInter = Pool($"lora.{name}.dinter", T * rank);
        _gpu.SgemmF32(dInter, scaledDOut, lora.b, T, outDim, rank);

        // dA += dIntermediate^T × input → [rank × inDim]
        _gpu.SgemmTransAAccumulate(grad.dA, dInter, input, rank, T, inDim);

    }

    private void DownloadLoraGradients()
    {
        foreach (var (name, layer) in _adapter.Layers)
        {
            var grad = _gpuLoraGrad[name];
            DownloadF32To(grad.dA, layer.A.Grad!);
            DownloadF32To(grad.dB, layer.B.Grad!);
        }
    }

    /// <summary>Re-upload LoRA weights after CPU optimizer step.</summary>
    public void SyncLoraWeights()
    {
        foreach (var (name, layer) in _adapter.Layers)
        {
            var lora = _gpuLora[name];
            UploadF32(lora.a, layer.A.Data);
            UploadF32(lora.b, layer.B.Data);
            var grad = _gpuLoraGrad[name];
            _gpu.ZeroTensor(grad.dA);
            _gpu.ZeroTensor(grad.dB);
        }
    }

    /// <summary>
    /// Run AdamW optimizer step entirely on GPU. No CPU round-trip.
    /// Clips gradients, updates LoRA params, zeros gradients — all on device.
    /// </summary>
    public void GpuOptimizerStep(float lr, float beta1, float beta2, float eps,
        float weightDecay, float maxGradNorm, float lrMultiplier)
    {
        _optimStep++;
        float effectiveLr = lr * lrMultiplier;
        float bc1 = 1.0f - MathF.Pow(beta1, _optimStep);
        float bc2 = 1.0f - MathF.Pow(beta2, _optimStep);

        // Gradient clipping: compute total norm across all LoRA params
        float totalNormSq = 0;
        foreach (var (name, _) in _adapter.Layers)
        {
            var grad = _gpuLoraGrad[name];
            totalNormSq += _gpu.GradNormSq(grad.dA);
            totalNormSq += _gpu.GradNormSq(grad.dB);
        }
        float totalNorm = MathF.Sqrt(totalNormSq);
        if (totalNorm > maxGradNorm)
        {
            float scale = maxGradNorm / totalNorm;
            foreach (var (name, _) in _adapter.Layers)
            {
                var grad = _gpuLoraGrad[name];
                _gpu.ScaleInPlace(grad.dA, scale);
                _gpu.ScaleInPlace(grad.dB, scale);
            }
        }

        // AdamW step on GPU for each LoRA parameter
        foreach (var (name, _) in _adapter.Layers)
        {
            var lora = _gpuLora[name];
            var grad = _gpuLoraGrad[name];
            var opt = _gpuOptState[name];

            _gpu.AdamWStep(lora.a, grad.dA, opt.mA, opt.vA,
                effectiveLr, beta1, beta2, eps, weightDecay, bc1, bc2);
            _gpu.AdamWStep(lora.b, grad.dB, opt.mB, opt.vB,
                effectiveLr, beta1, beta2, eps, weightDecay, bc1, bc2);

            // Zero gradients for next step
            _gpu.ZeroTensor(grad.dA);
            _gpu.ZeroTensor(grad.dB);
        }
    }

    /// <summary>Download final LoRA weights from GPU to CPU adapter (for saving).</summary>
    public void DownloadLoraWeights()
    {
        foreach (var (name, layer) in _adapter.Layers)
        {
            var lora = _gpuLora[name];
            DownloadF32To(lora.a, layer.A.Data);
            DownloadF32To(lora.b, layer.B.Data);
        }
    }

    // ── GPU tensor pool ──────────────────────────────────────────────────

    /// <summary>
    /// Get or create a named GPU tensor. Pre-allocates on first call,
    /// returns the same tensor on subsequent calls. Eliminates per-step
    /// cuMemAlloc/cuMemFree overhead.
    /// </summary>
    private ITensor Pool(string name, int size)
    {
        if (_pool.TryGetValue(name, out var existing))
        {
            if (existing.ElementCount == size)
                return existing;
            existing.Dispose();
        }
        var t = _gpu.CreateTensor(name, GgmlType.F32, [size]);
        _pool[name] = t;
        return t;
    }

    /// <summary>Get a pooled tensor and zero it.</summary>
    private ITensor PoolZ(string name, int size)
    {
        var t = Pool(name, size);
        _gpu.ZeroTensor(t);
        return t;
    }

    private ITensor CreateF32Tensor(string name, float[] data)
    {
        var t = Pool(name, data.Length);
        UploadF32(t, data);
        return t;
    }

    private ITensor CreateIntTensor(string name, int[] data)
    {
        var t = Pool(name, data.Length);
        var bytes = new byte[data.Length * sizeof(int)];
        Buffer.BlockCopy(data, 0, bytes, 0, bytes.Length);
        t.CopyFrom(bytes);
        return t;
    }

    private unsafe void UploadF32(ITensor tensor, float[] data)
    {
        ((CudaTensor)tensor).UploadFrom(data);
    }

    private float[] DownloadF32(ITensor tensor, int count)
    {
        var result = new float[count];
        _gpu.Synchronize();
        ((CudaTensor)tensor).DownloadTo(result);
        return result;
    }

    private void DownloadF32To(ITensor tensor, float[] dest)
    {
        _gpu.Synchronize();
        ((CudaTensor)tensor).DownloadTo(dest);
    }

    private ITensor CloneGpuTensor(string name, ITensor src)
    {
        var dst = Pool(name, (int)src.ElementCount);
        _gpu.CopyTensor(dst, src);
        return dst;
    }

    // Cache of model weights uploaded to GPU (keyed by CPU tensor reference)
    private readonly Dictionary<ITensor, ITensor> _gpuWeightCache = new();
    private readonly Dictionary<string, ITensor> _dequantCache = new();

    /// <summary>Upload a CPU model weight to GPU for forward MatMul (cached).</summary>
    private ITensor GpuWeight(ITensor cpuWeight)
    {
        if (_gpuWeightCache.TryGetValue(cpuWeight, out var cached))
            return cached;
        var rawBytes = new byte[cpuWeight.ByteSize];
        cpuWeight.CopyRawTo(rawBytes);
        // Ensure Q8_1 scratch is allocated for Q8_0 matmul (small models may not trigger it in LoadTensor)
        if (cpuWeight.Type == Gguf.GgmlType.Q8_0 && cpuWeight.Dimensions.Length >= 2)
            _gpu.EnsureQ8_1Scratch((int)cpuWeight.Dimensions[0]);
        var gpu = _gpu.LoadTensor(cpuWeight.Name, cpuWeight.Type, cpuWeight.Dimensions, rawBytes);
        _gpuWeightCache[cpuWeight] = gpu;
        return gpu;
    }

    private ITensor DequantToGpu(string name, ITensor quantized)
    {
        // Cache permanently — base model weights never change during training
        if (_dequantCache.TryGetValue(name, out var cached))
            return cached;
        var data = F32Ops.Dequantize(quantized);
        var t = CreateF32Tensor(name, data);
        _dequantCache[name] = t;
        return t;
    }

    private ITensor EnsureLoraIntermediate(string name, int size)
    {
        if (_gpuLoraIntermediate.TryGetValue(name, out var existing) && existing.ElementCount >= size)
            return existing;
        existing?.Dispose();
        var t = Pool($"lora.{name}.inter", size);
        _gpuLoraIntermediate[name] = t;
        return t;
    }

    public void Dispose()
    {
        foreach (var t in _pool.Values) t.Dispose();
        _pool.Clear();
        _embeddingTable?.Dispose();
        _outputNormWeight?.Dispose();
        foreach (var (a, b) in _gpuLora.Values) { a.Dispose(); b.Dispose(); }
        foreach (var (da, db) in _gpuLoraGrad.Values) { da.Dispose(); db.Dispose(); }
        foreach (var t in _gpuLoraIntermediate.Values) t.Dispose();
        foreach (var t in _dequantCache.Values) t.Dispose();
        foreach (var t in _gpuWeightCache.Values) t.Dispose();
        foreach (var (mA, vA, mB, vB) in _gpuOptState.Values)
            { mA.Dispose(); vA.Dispose(); mB.Dispose(); vB.Dispose(); }
    }
}

/// <summary>GPU layer activations — stores ITensor references on device.</summary>
internal sealed class GpuLayerActivations : IDisposable
{
    public bool IsAttention;
    public bool IsDeltaNet;
    public bool HasGatedQ;

    public ITensor? InputToLayer;
    public ITensor? NormOut;
    public ITensor? PostAttnHidden;
    public ITensor? FfnNormOut;
    public ITensor? FfnResidual;
    public ITensor? AttnNormWeight;
    public ITensor? PostAttnNormWeight;

    // Attention
    public ITensor? QRaw, KRaw, VRaw;
    public ITensor? QAttn, QGate;
    public ITensor? AttnOut;
    public ITensor? SavedProbs;

    // FFN
    public ITensor? GateInput, UpInput, FfnGated;

    // DeltaNet
    public ITensor? DeltaQkvPreSilu, DeltaQkvPostSilu;
    public ITensor? DeltaGatePreSilu, DeltaGatePostSilu;
    public ITensor? DeltaSsmApprox;

    // Backward output
    public ITensor? DNormOut;

    public void Dispose()
    {
        // Tensors are owned by the pool — nothing to dispose here.
    }
}
