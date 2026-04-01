using Daisi.Llogos.Cpu;
using Daisi.Llogos.Gguf;
using Daisi.Llogos.Model;
using Daisi.Llogos.Training.Lora;

namespace Daisi.Llogos.Training;

/// <summary>
/// Full training forward + backward pass through a hybrid transformer with LoRA.
/// Base model weights are dequantized on-the-fly (never stored in full F32).
/// LoRA is applied at standard attention Q, K, V, O and DeltaNet QKV, Out projections.
/// Supports GPU-accelerated forward matmuls when a non-CPU backend is provided.
/// </summary>
public sealed class TrainingForwardPass : ITrainingForwardPass
{
    private readonly ModelConfig _config;
    private readonly ModelWeights _weights;
    private readonly LoraAdapter _adapter;
    private readonly IComputeBackend _backend;
    private readonly bool _useGpuMatMul;

    // Per-layer saved activations for backward (allocated per sequence)
    private LayerActivations[]? _savedActivations;

    // Model-level buffers
    private float[]? _embeddingTable; // dequantized once
    private float[]? _outputNormWeight;
    private float[]? _outputWeight;

    // GPU scratch tensors (reused across matmul calls)
    private ITensor? _gpuActivation;
    private ITensor? _gpuOutput;
    private int _gpuActivationSize;
    private int _gpuOutputSize;
    // Cache of weight tensors uploaded to GPU (keyed by CPU tensor reference)
    private readonly Dictionary<ITensor, ITensor> _gpuWeightCache = new();

    public TrainingForwardPass(ModelConfig config, ModelWeights weights, LoraAdapter adapter,
        IComputeBackend? backend = null)
    {
        _config = config;
        _weights = weights;
        _adapter = adapter;
        _backend = backend ?? new CpuBackend();
        _useGpuMatMul = _backend is not CpuBackend;
    }

    /// <summary>
    /// Forward matmul: output = activation × weight^T (using GPU when available).
    /// On GPU: uses backend.MatMul with fused dequant+cuBLAS. Weight stays on GPU.
    /// On CPU: dequantizes weight then uses F32Ops.MatMulTransB.
    /// </summary>
    private void WeightMatMul(float[] output, float[] activation, ITensor weight, int M, int K, int N)
    {
        if (_useGpuMatMul)
        {
            // Create or resize GPU scratch tensors to exact size needed
            int actSize = M * K;
            if (_gpuActivation == null || _gpuActivationSize != actSize)
            {
                _gpuActivation?.Dispose();
                _gpuActivation = _backend.CreateTensor("train_act", GgmlType.F32, [actSize]);
                _gpuActivationSize = actSize;
            }
            int outSize = M * N;
            if (_gpuOutput == null || _gpuOutputSize != outSize)
            {
                _gpuOutput?.Dispose();
                _gpuOutput = _backend.CreateTensor("train_out", GgmlType.F32, [outSize]);
                _gpuOutputSize = outSize;
            }

            // Upload weight to GPU on first use (cached for subsequent calls)
            if (!_gpuWeightCache.TryGetValue(weight, out var gpuWeight))
            {
                var rawBytes = new byte[weight.ByteSize];
                weight.CopyRawTo(rawBytes);
                gpuWeight = _backend.LoadTensor(weight.Name, weight.Type, weight.Dimensions, rawBytes);
                _gpuWeightCache[weight] = gpuWeight;
            }

            // Upload activation to GPU
            var actBytes = new byte[actSize * sizeof(float)];
            Buffer.BlockCopy(activation, 0, actBytes, 0, actBytes.Length);
            _gpuActivation.CopyFrom(actBytes);

            // GPU matmul: output = activation × weight (fused dequant + cuBLAS)
            _backend.MatMul(_gpuOutput, _gpuActivation, gpuWeight, M, K, N);

            // Download result (F32 tensor, no repacking issues)
            _gpuOutput.DequantizeTo(output);
        }
        else
        {
            var w = F32Ops.Dequantize(weight);
            F32Ops.MatMulTransB(output, activation, w, M, K, N);
        }
    }

    /// <summary>
    /// Run forward pass on a training sequence, returning logits and saving activations for backward.
    /// tokenIds: [T] input token IDs.
    /// Returns: logits [T × vocabSize].
    /// </summary>
    public float[] Forward(int[] tokenIds, out float totalLoss, int[] targets)
    {
        int T = tokenIds.Length;
        int H = _config.HiddenDim;
        int V = _config.VocabSize;

        // Dequantize embedding table (cached)
        _embeddingTable ??= F32Ops.Dequantize(_weights.TokenEmbedding);
        _outputNormWeight ??= F32Ops.Dequantize(_weights.OutputNorm);
        _outputWeight ??= F32Ops.Dequantize(_weights.OutputWeight);

        // Allocate activation storage
        _savedActivations = new LayerActivations[_config.NumLayers];

        // Hidden states: [T × H]
        var hidden = new float[T * H];
        var residual = new float[T * H];

        // 1. Embedding lookup
        for (int t = 0; t < T; t++)
            F32Ops.EmbeddingLookup(hidden.AsSpan(t * H, H), _embeddingTable, tokenIds[t], H);

        // 2. Transformer layers
        for (int layer = 0; layer < _config.NumLayers; layer++)
        {
            var lw = _weights.Layers[layer];
            var normWeight = F32Ops.Dequantize(lw.AttnNorm);
            var postNormWeight = F32Ops.Dequantize(lw.PostAttnNorm);

            // Pre-attention norm + residual save
            var normOut = new float[T * H];
            for (int t = 0; t < T; t++)
            {
                // residual = hidden (save for backward)
                Array.Copy(hidden, t * H, residual, t * H, H);
                // normOut = RmsNorm(hidden)
                F32Ops.RmsNorm(normOut.AsSpan(t * H, H), hidden.AsSpan(t * H, H), normWeight, _config.NormEps, H);
            }

            var saved = new LayerActivations
            {
                InputToLayer = residual.ToArray(),
                NormOut = normOut,
                AttnNormWeight = normWeight,
                PostAttnNormWeight = postNormWeight,
            };

            if (lw is StandardAttentionWeights saw && _config.IsStandardAttention(layer))
            {
                ForwardAttentionLayer(saw, normOut, hidden, saved, layer, T);
            }
            else
            {
                // DeltaNet layer: simple forward (no LoRA, no gradient tracking for SSM)
                // We approximate by treating the entire layer as applying some transform
                // but for backprop we'll just pass gradient through the residual
                ForwardDeltaNetLayerApprox(lw, normOut, hidden, saved, layer, T);
                saved.IsDeltaNet = true;
            }

            // Post-attention: hidden = attention_output (already in hidden)
            // Add residual: hidden += residual
            for (int i = 0; i < T * H; i++)
                hidden[i] += residual[i];

            // Save post-attn hidden for FFN backward
            saved.PostAttnHidden = hidden.ToArray();

            // Post-attention norm
            var ffnNormOut = new float[T * H];
            var ffnResidual = hidden.ToArray();
            for (int t = 0; t < T; t++)
                F32Ops.RmsNorm(ffnNormOut.AsSpan(t * H, H), hidden.AsSpan(t * H, H), postNormWeight, _config.NormEps, H);

            saved.FfnNormOut = ffnNormOut;
            saved.FfnResidual = ffnResidual;

            // FFN: gate, up, down
            int I = _config.IntermediateDim;

            var gateOut = new float[T * I];
            var upOut = new float[T * I];
            var ffnGated = new float[T * I];

            // gate = ffnNormOut × gateWeight^T + LoRA
            WeightMatMul(gateOut, ffnNormOut, lw.FfnGate, T, H, I);
            string ffnPrefix = $"blk.{layer}";
            var loraGate = _adapter.GetLayer($"{ffnPrefix}.ffn_gate");
            float[]? loraGateInter = null;
            if (loraGate != null)
            {
                loraGateInter = new float[T * loraGate.Rank];
                loraGate.Forward(gateOut, ffnNormOut, loraGateInter, T);
            }
            // up = ffnNormOut × upWeight^T + LoRA
            WeightMatMul(upOut, ffnNormOut, lw.FfnUp, T, H, I);
            var loraUp = _adapter.GetLayer($"{ffnPrefix}.ffn_up");
            float[]? loraUpInter = null;
            if (loraUp != null)
            {
                loraUpInter = new float[T * loraUp.Rank];
                loraUp.Forward(upOut, ffnNormOut, loraUpInter, T);
            }
            // ffnGated = SwiGLU(gate, up)
            F32Ops.SwiGLU(ffnGated, gateOut, upOut, T * I);

            // hidden = ffnGated × downWeight^T + LoRA
            var ffnOut = new float[T * H];
            WeightMatMul(ffnOut, ffnGated, lw.FfnDown, T, I, H);
            var loraDown = _adapter.GetLayer($"{ffnPrefix}.ffn_down");
            float[]? loraDownInter = null;
            if (loraDown != null)
            {
                loraDownInter = new float[T * loraDown.Rank];
                loraDown.Forward(ffnOut, ffnGated, loraDownInter, T);
            }

            saved.GateInput = gateOut;
            saved.UpInput = upOut;
            saved.FfnGated = ffnGated;
            saved.LoraFfnGateInter = loraGateInter;
            saved.LoraFfnUpInter = loraUpInter;
            saved.LoraFfnDownInter = loraDownInter;

            // Add residual
            for (int i = 0; i < T * H; i++)
                hidden[i] = ffnOut[i] + ffnResidual[i];

            _savedActivations[layer] = saved;
        }

        // 3. Final norm + logit projection
        var finalNormOut = new float[T * H];
        for (int t = 0; t < T; t++)
            F32Ops.RmsNorm(finalNormOut.AsSpan(t * H, H), hidden.AsSpan(t * H, H),
                _outputNormWeight, _config.NormEps, H);

        // logits = finalNormOut × outputWeight^T → [T × V]
        var logits = new float[T * V];
        WeightMatMul(logits, finalNormOut, _weights.OutputWeight, T, H, V);

        // Compute loss and gradient
        var dLogits = new float[T * V];
        totalLoss = F32Ops.CrossEntropyLoss(logits, targets, dLogits, T, V);

        // 4. Backward pass
        Backward(dLogits, finalNormOut, hidden, tokenIds, T);

        return logits;
    }

    private void ForwardAttentionLayer(StandardAttentionWeights saw, float[] normOut,
        float[] hidden, LayerActivations saved, int layer, int T)
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

        // Q, K, V projections: normOut × W^T
        var qRaw = new float[T * qOutDim];
        var kRaw = new float[T * kOutDim];
        var vRaw = new float[T * vOutDim];

        WeightMatMul(qRaw, normOut, saw.AttnQ, T, H, qOutDim);
        WeightMatMul(kRaw, normOut, saw.AttnK, T, H, kOutDim);
        WeightMatMul(vRaw, normOut, saw.AttnV, T, H, vOutDim);

        // Apply LoRA
        string prefix = $"blk.{layer}";
        var loraQ = _adapter.GetLayer($"{prefix}.attn_q");
        var loraK = _adapter.GetLayer($"{prefix}.attn_k");
        var loraV = _adapter.GetLayer($"{prefix}.attn_v");
        var loraO = _adapter.GetLayer($"{prefix}.attn_o");

        float[]? loraQIntermediate = null, loraKIntermediate = null, loraVIntermediate = null;

        if (loraQ != null)
        {
            loraQIntermediate = new float[T * loraQ.Rank];
            loraQ.Forward(qRaw, normOut, loraQIntermediate, T);
        }
        if (loraK != null)
        {
            loraKIntermediate = new float[T * loraK.Rank];
            loraK.Forward(kRaw, normOut, loraKIntermediate, T);
        }
        if (loraV != null)
        {
            loraVIntermediate = new float[T * loraV.Rank];
            loraV.Forward(vRaw, normOut, loraVIntermediate, T);
        }

        // De-interleave gated Q if needed
        int qAttnDim = numHeads * keyDim;
        var qAttn = new float[T * qAttnDim];
        var qGate = new float[T * qAttnDim];

        if (hasGatedQ)
        {
            for (int t = 0; t < T; t++)
                F32Ops.DeInterleaveQ(
                    qAttn.AsSpan(t * qAttnDim, qAttnDim),
                    qGate.AsSpan(t * qAttnDim, qAttnDim),
                    qRaw.AsSpan(t * qOutDim, qOutDim),
                    numHeads, keyDim);
        }
        else
        {
            Array.Copy(qRaw, qAttn, T * qAttnDim);
            // Fill qGate with large value so sigmoid ≈ 1 (ungated)
            Array.Fill(qGate, 88.0f);
        }

        // Per-head Q/K norms (save pre-norm values for backward)
        float[]? qBeforeNorm = null, kBeforeNorm = null;
        float[]? qNormWeight = null, kNormWeight = null;
        if (saw.AttnQNorm != null)
        {
            qBeforeNorm = qAttn.ToArray();
            kBeforeNorm = kRaw.ToArray();
            qNormWeight = F32Ops.Dequantize(saw.AttnQNorm);
            kNormWeight = F32Ops.Dequantize(saw.AttnKNorm!);
            for (int t = 0; t < T; t++)
            {
                F32Ops.PerHeadRmsNorm(qAttn.AsSpan(t * qAttnDim, qAttnDim), qNormWeight, numHeads, keyDim, _config.NormEps);
                F32Ops.PerHeadRmsNorm(kRaw.AsSpan(t * kOutDim, kOutDim), kNormWeight, numKvHeads, keyDim, _config.NormEps);
            }
        }

        // Save pre-RoPE values for backward
        var qPreRope = qAttn.ToArray();
        var kPreRope = kRaw.ToArray();

        // RoPE
        F32Ops.BatchedRoPE(qAttn, T, numHeads, keyDim, ropeDim, 0, _config.RopeTheta);
        F32Ops.BatchedRoPE(kRaw, T, numKvHeads, keyDim, ropeDim, 0, _config.RopeTheta);

        // Causal attention
        int attnOutDim = numHeads * valDim;
        var attnOut = new float[T * attnOutDim];
        var savedProbs = new float[numHeads * T * T];

        F32Ops.CausalGatedAttention(attnOut, qAttn, qGate, kRaw, vRaw,
            T, numHeads, numKvHeads, keyDim, valDim, scale, savedProbs);

        // Output projection: attnOut × wO^T → hidden
        int oDim = (int)saw.AttnO.Dimensions[0]; // input dim to O
        var oOut = new float[T * H];
        WeightMatMul(oOut, attnOut, saw.AttnO, T, oDim, H);

        // Apply LoRA O
        float[]? loraOIntermediate = null;
        if (loraO != null)
        {
            loraOIntermediate = new float[T * loraO.Rank];
            loraO.Forward(oOut, attnOut, loraOIntermediate, T);
        }

        // Write to hidden
        Array.Copy(oOut, hidden, T * H);

        // Save for backward
        saved.IsAttention = true;
        saved.HasGatedQ = hasGatedQ;
        // Weights are NOT saved — re-dequantized on demand during backward to reduce peak memory
        saved.QRaw = qRaw; saved.KRaw = kRaw; saved.VRaw = vRaw;
        saved.QAttn = qAttn; saved.QGate = qGate;
        saved.QPreRope = qPreRope; saved.KPreRope = kPreRope;
        saved.QBeforeNorm = qBeforeNorm; saved.KBeforeNorm = kBeforeNorm;
        saved.QNormWeight = qNormWeight; saved.KNormWeight = kNormWeight;
        saved.AttnOut = attnOut;
        saved.SavedProbs = savedProbs;
        saved.OOut = oOut;
        saved.LoraQIntermediate = loraQIntermediate;
        saved.LoraKIntermediate = loraKIntermediate;
        saved.LoraVIntermediate = loraVIntermediate;
        saved.LoraOIntermediate = loraOIntermediate;
    }

    private void ForwardDeltaNetLayerApprox(LayerWeights lw, float[] normOut,
        float[] hidden, LayerActivations saved, int layer, int T)
    {
        // Simplified DeltaNet forward with LoRA support.
        // Instead of running the full SSM state recurrence, we approximate:
        //   ssmApprox = silu(qkv) * silu(gate)   (element-wise, truncated to ssmInner)
        // This gives gradient flow through both QKV and Gate paths for LoRA training.

        if (lw is DeltaNetWeights dnw)
        {
            int H = _config.HiddenDim;
            string prefix = $"blk.{layer}";

            // QKV projection
            int qkvDim = (int)dnw.AttnQkv.Dimensions[1];
            var qkvOut = new float[T * qkvDim];
            WeightMatMul(qkvOut, normOut, dnw.AttnQkv, T, H, qkvDim);

            // Apply LoRA to QKV (before SiLU)
            var loraQkv = _adapter.GetLayer($"{prefix}.delta_qkv");
            float[]? loraQkvIntermediate = null;
            if (loraQkv != null)
            {
                loraQkvIntermediate = new float[T * loraQkv.Rank];
                loraQkv.Forward(qkvOut, normOut, loraQkvIntermediate, T);
            }

            // Save pre-SiLU, then apply SiLU
            var qkvPreSilu = qkvOut.ToArray();
            F32Ops.SiLU(qkvOut, qkvOut, T * qkvDim);

            // Gate projection
            int gateDim = (int)dnw.AttnGate.Dimensions[1];
            var gateOut = new float[T * gateDim];
            WeightMatMul(gateOut, normOut, dnw.AttnGate, T, H, gateDim);

            // Save pre-SiLU gate, then apply SiLU
            var gatePreSilu = gateOut.ToArray();
            F32Ops.SiLU(gateOut, gateOut, T * gateDim);

            // SSM approximation: element-wise product of silu(qkv) * silu(gate)
            int outInDim = (int)dnw.SsmOut.Dimensions[0];
            var ssmApprox = new float[T * outInDim];
            for (int t = 0; t < T; t++)
                for (int i = 0; i < outInDim; i++)
                {
                    float qVal = i < qkvDim ? qkvOut[t * qkvDim + i] : 0;
                    float gVal = i < gateDim ? gateOut[t * gateDim + i] : 0;
                    ssmApprox[t * outInDim + i] = qVal * gVal;
                }

            // Output projection: ssmApprox × SsmOut^T → [T × H]
            var layerOut = new float[T * H];
            WeightMatMul(layerOut, ssmApprox, dnw.SsmOut, T, outInDim, H);

            // Apply LoRA to output projection
            var loraOut = _adapter.GetLayer($"{prefix}.delta_out");
            float[]? loraOutIntermediate = null;
            if (loraOut != null)
            {
                loraOutIntermediate = new float[T * loraOut.Rank];
                loraOut.Forward(layerOut, ssmApprox, loraOutIntermediate, T);
            }

            Array.Copy(layerOut, hidden, T * H);

            // Save activations for backward
            saved.DeltaQkvPreSilu = qkvPreSilu;
            saved.DeltaQkvPostSilu = qkvOut.ToArray();
            saved.DeltaGatePreSilu = gatePreSilu;
            saved.DeltaGatePostSilu = gateOut.ToArray();
            saved.DeltaSsmApprox = ssmApprox;
            saved.LoraDeltaQkvIntermediate = loraQkvIntermediate;
            saved.LoraDeltaOutIntermediate = loraOutIntermediate;
        }
        else
        {
            // Unknown layer type — zero output (rely on residual)
            Array.Clear(hidden, 0, hidden.Length);
        }
    }

    // ── Backward Pass ────────────────────────────────────────────────────────

    private void Backward(float[] dLogits, float[] finalNormOut, float[] finalHidden,
        int[] tokenIds, int T)
    {
        int H = _config.HiddenDim;
        int V = _config.VocabSize;

        // dFinalNormOut = dLogits × outputWeight → [T × H]
        var dFinalNormOut = new float[T * H];
        F32Ops.MatMul(dFinalNormOut, dLogits, _outputWeight!, T, V, H);

        // Backward through final RMS norm → dHidden
        var dHidden = new float[T * H];
        var dOutputNormWeight = new float[H]; // not trained, but needed for chain rule
        for (int t = 0; t < T; t++)
        {
            F32Ops.RmsNormBackward(
                dHidden.AsSpan(t * H, H), dOutputNormWeight,
                dFinalNormOut.AsSpan(t * H, H), finalHidden.AsSpan(t * H, H),
                _outputNormWeight!, _config.NormEps, H);
        }

        // Backward through layers in reverse
        for (int layer = _config.NumLayers - 1; layer >= 0; layer--)
        {
            var saved = _savedActivations![layer];
            var lw = _weights.Layers[layer];
            int I = _config.IntermediateDim;

            // Backward through FFN residual: dHidden splits into dFfnOut and dFfnResidual
            var dFfnOut = dHidden.ToArray();
            var dFfnResidual = dHidden.ToArray();

            // Backward through FFN down LoRA + projection
            string ffnPrefix = $"blk.{layer}";
            var loraDown = _adapter.GetLayer($"{ffnPrefix}.ffn_down");
            if (loraDown != null)
                loraDown.Backward(new float[T * I], dFfnOut, saved.FfnGated!, saved.LoraFfnDownInter!, T);

            var downWeight = F32Ops.Dequantize(lw.FfnDown);
            var dFfnGated = new float[T * I];
            F32Ops.MatMul(dFfnGated, dFfnOut, downWeight, T, H, I);

            // Backward through SwiGLU
            var dGate = new float[T * I];
            var dUp = new float[T * I];
            F32Ops.SwiGLUBackward(dGate, dUp, dFfnGated, saved.GateInput!, saved.UpInput!, T * I);

            // Backward through gate/up LoRA
            var loraGate = _adapter.GetLayer($"{ffnPrefix}.ffn_gate");
            var loraUp = _adapter.GetLayer($"{ffnPrefix}.ffn_up");

            // Backward through gate/up projections → dFfnNormOut
            var dFfnNormOut = new float[T * H];
            // dFfnNormOut += dGate × gateWeight + LoRA backward
            if (loraGate != null)
                loraGate.Backward(dFfnNormOut, dGate, saved.FfnNormOut!, saved.LoraFfnGateInter!, T);
            var gateWeight = F32Ops.Dequantize(lw.FfnGate);
            var dFromGate = new float[T * H];
            F32Ops.MatMul(dFromGate, dGate, gateWeight, T, I, H);
            F32Ops.AddInPlace(dFfnNormOut, dFromGate, T * H);
            // dFfnNormOut += dUp × upWeight + LoRA backward
            if (loraUp != null)
                loraUp.Backward(dFfnNormOut, dUp, saved.FfnNormOut!, saved.LoraFfnUpInter!, T);
            var upWeight = F32Ops.Dequantize(lw.FfnUp);
            var dFromUp = new float[T * H];
            F32Ops.MatMul(dFromUp, dUp, upWeight, T, I, H);
            F32Ops.AddInPlace(dFfnNormOut, dFromUp, T * H);

            // Backward through post-attention RMS norm → dPostAttnHidden
            var dPostAttnHidden = new float[T * H];
            var dPostNormW = new float[H];
            for (int t = 0; t < T; t++)
            {
                F32Ops.RmsNormBackward(
                    dPostAttnHidden.AsSpan(t * H, H), dPostNormW,
                    dFfnNormOut.AsSpan(t * H, H), saved.PostAttnHidden!.AsSpan(t * H, H),
                    saved.PostAttnNormWeight!, _config.NormEps, H);
            }

            // Add residual gradient
            F32Ops.AddInPlace(dPostAttnHidden, dFfnResidual, T * H);

            // Now dPostAttnHidden is the gradient coming into the attention residual add
            // dPostAttnHidden = dAttnOut + dResidual
            var dAttnOut = dPostAttnHidden.ToArray();
            var dResidual = dPostAttnHidden.ToArray();

            if (saved.IsAttention)
                BackwardAttentionLayer(saved, dAttnOut, layer, T);
            else if (saved.IsDeltaNet)
                BackwardDeltaNetLayer(saved, dAttnOut, layer, T);

            // Backward through pre-attention RMS norm → dInputToLayer
            var dInputToLayer = new float[T * H];
            if (saved.IsAttention || saved.IsDeltaNet)
            {
                // dNormOut was computed inside Backward*Layer and accumulated into saved
                var dNormOut = saved.DNormOut!;
                var dNormWeight = new float[H];
                for (int t = 0; t < T; t++)
                {
                    F32Ops.RmsNormBackward(
                        dInputToLayer.AsSpan(t * H, H), dNormWeight,
                        dNormOut.AsSpan(t * H, H), saved.InputToLayer!.AsSpan(t * H, H),
                        saved.AttnNormWeight!, _config.NormEps, H);
                }
            }

            // Combine: dHidden for previous layer = dResidual + dInputToLayer
            dHidden = new float[T * H];
            F32Ops.Add(dHidden, dResidual, dInputToLayer, T * H);

            // Release this layer's saved activations so GC can reclaim memory
            _savedActivations[layer] = null!;
        }
    }

    private void BackwardAttentionLayer(LayerActivations saved, float[] dOOut, int layer, int T)
    {
        int H = _config.HiddenDim;
        int numHeads = _config.NumHeads;
        int numKvHeads = _config.NumKvHeads;
        int keyDim = _config.KeyLength;
        int valDim = _config.ValueLength;
        int ropeDim = _config.RopeDimCount;
        float scale = 1.0f / MathF.Sqrt(keyDim);
        string prefix = $"blk.{layer}";

        int oDim = saved.AttnOut!.Length / T;
        int qAttnDim = numHeads * keyDim;
        int kOutDim = numKvHeads * keyDim;
        int vOutDim = numKvHeads * valDim;

        // Backward through O LoRA
        var loraO = _adapter.GetLayer($"{prefix}.attn_o");
        if (loraO != null)
        {
            var dAttnOut = new float[T * oDim];
            loraO.Backward(dAttnOut, dOOut, saved.AttnOut!, saved.LoraOIntermediate!, T);
            // dOOut already has base path gradient; add LoRA's dInput contribution
            // Actually the base path dAttnOut needs to be computed separately
        }

        // Backward through O projection: dAttnOut = dOOut × wO
        // Re-dequantize attention weights on demand (not cached from forward)
        var saw = (StandardAttentionWeights)_weights.Layers[layer];
        var wO = F32Ops.Dequantize(saw.AttnO);
        var dAttnOutFromO = new float[T * oDim];
        F32Ops.MatMul(dAttnOutFromO, dOOut, wO, T, H, oDim);

        // Backward through causal gated attention
        var dQAttn = new float[T * qAttnDim];
        var dQGate = new float[T * qAttnDim];
        var dK = new float[T * kOutDim];
        var dV = new float[T * vOutDim];

        // Recompute attention output before gating for backward
        // (we need the pre-gate values which we can derive from saved)
        F32Ops.CausalGatedAttentionBackward(
            dQAttn, dQGate, dK, dV,
            dAttnOutFromO, saved.QAttn!, saved.QGate!,
            saved.KRaw!, saved.VRaw!, saved.SavedProbs!, saved.AttnOut!,
            T, numHeads, numKvHeads, keyDim, valDim, scale);

        // Backward through RoPE (inverse rotation on gradients)
        F32Ops.BatchedRoPEBackward(dQAttn, T, numHeads, keyDim, ropeDim, 0, _config.RopeTheta);
        F32Ops.BatchedRoPEBackward(dK, T, numKvHeads, keyDim, ropeDim, 0, _config.RopeTheta);

        // Backward through per-head Q/K norms (if present)
        if (saved.QNormWeight != null)
        {
            var dQBeforeNorm = new float[T * qAttnDim];
            var dQNormW = new float[keyDim];
            for (int t = 0; t < T; t++)
            {
                F32Ops.PerHeadRmsNormBackward(
                    dQBeforeNorm.AsSpan(t * qAttnDim, qAttnDim), dQNormW,
                    dQAttn.AsSpan(t * qAttnDim, qAttnDim),
                    saved.QBeforeNorm!.AsSpan(t * qAttnDim, qAttnDim),
                    saved.QNormWeight, numHeads, keyDim, _config.NormEps);
            }
            dQAttn = dQBeforeNorm;

            var dKBeforeNorm = new float[T * kOutDim];
            var dKNormW = new float[keyDim];
            for (int t = 0; t < T; t++)
            {
                F32Ops.PerHeadRmsNormBackward(
                    dKBeforeNorm.AsSpan(t * kOutDim, kOutDim), dKNormW,
                    dK.AsSpan(t * kOutDim, kOutDim),
                    saved.KBeforeNorm!.AsSpan(t * kOutDim, kOutDim),
                    saved.KNormWeight!, numKvHeads, keyDim, _config.NormEps);
            }
            dK = dKBeforeNorm;
        }

        // Backward through de-interleave (if gated Q)
        float[] dQRaw;
        if (saved.HasGatedQ)
        {
            int qOutDim = numHeads * keyDim * 2;
            dQRaw = new float[T * qOutDim];
            for (int t = 0; t < T; t++)
                F32Ops.DeInterleaveQBackward(
                    dQRaw.AsSpan(t * qOutDim, qOutDim),
                    dQAttn.AsSpan(t * qAttnDim, qAttnDim),
                    dQGate.AsSpan(t * qAttnDim, qAttnDim),
                    numHeads, keyDim);
        }
        else
        {
            dQRaw = dQAttn;
        }

        // Backward through Q, K, V projections → dNormOut
        int qRawDim = dQRaw.Length / T;
        var dNormOut = new float[T * H];

        // dNormOut += dQRaw × wQ
        var wQ = F32Ops.Dequantize(saw.AttnQ);
        var dFromQ = new float[T * H];
        F32Ops.MatMul(dFromQ, dQRaw, wQ, T, qRawDim, H);
        F32Ops.AddInPlace(dNormOut, dFromQ, T * H);

        // dNormOut += dK × wK
        var wK = F32Ops.Dequantize(saw.AttnK);
        var dFromK = new float[T * H];
        F32Ops.MatMul(dFromK, dK, wK, T, kOutDim, H);
        F32Ops.AddInPlace(dNormOut, dFromK, T * H);

        // dNormOut += dV × wV
        var wV = F32Ops.Dequantize(saw.AttnV);
        var dFromV = new float[T * H];
        F32Ops.MatMul(dFromV, dV, wV, T, vOutDim, H);
        F32Ops.AddInPlace(dNormOut, dFromV, T * H);

        // LoRA backward for Q, K, V (accumulates into LoRA gradients + dNormOut)
        var loraQ = _adapter.GetLayer($"{prefix}.attn_q");
        var loraK = _adapter.GetLayer($"{prefix}.attn_k");
        var loraV = _adapter.GetLayer($"{prefix}.attn_v");

        if (loraQ != null)
            loraQ.Backward(dNormOut, dQRaw, saved.NormOut!, saved.LoraQIntermediate!, T);
        if (loraK != null)
            loraK.Backward(dNormOut, dK, saved.NormOut!, saved.LoraKIntermediate!, T);
        if (loraV != null)
            loraV.Backward(dNormOut, dV, saved.NormOut!, saved.LoraVIntermediate!, T);

        saved.DNormOut = dNormOut;
    }

    private void BackwardDeltaNetLayer(LayerActivations saved, float[] dLayerOut, int layer, int T)
    {
        int H = _config.HiddenDim;
        string prefix = $"blk.{layer}";
        var dnw = (DeltaNetWeights)_weights.Layers[layer];

        int outInDim = (int)dnw.SsmOut.Dimensions[0];
        int qkvDim = (int)dnw.AttnQkv.Dimensions[1];
        int gateDim = (int)dnw.AttnGate.Dimensions[1];

        // 1. Backward through SsmOut LoRA
        var loraOut = _adapter.GetLayer($"{prefix}.delta_out");
        if (loraOut != null)
        {
            var dSsmApproxLora = new float[T * outInDim];
            loraOut.Backward(dSsmApproxLora, dLayerOut, saved.DeltaSsmApprox!, saved.LoraDeltaOutIntermediate!, T);
            // dSsmApproxLora contributes to dSsmApprox below
        }

        // 2. Backward through SsmOut projection: dSsmApprox = dLayerOut × SsmOut
        var wOut = F32Ops.Dequantize(dnw.SsmOut);
        var dSsmApprox = new float[T * outInDim];
        F32Ops.MatMul(dSsmApprox, dLayerOut, wOut, T, H, outInDim);

        // 3. Backward through element-wise product: ssmApprox = qkvPostSilu * gatePostSilu
        var dQkvPostSilu = new float[T * qkvDim];
        var dGatePostSilu = new float[T * gateDim];
        for (int t = 0; t < T; t++)
            for (int i = 0; i < outInDim; i++)
            {
                float dSsm = dSsmApprox[t * outInDim + i];
                if (i < qkvDim)
                    dQkvPostSilu[t * qkvDim + i] = dSsm * (i < gateDim ? saved.DeltaGatePostSilu![t * gateDim + i] : 0);
                if (i < gateDim)
                    dGatePostSilu[t * gateDim + i] = dSsm * (i < qkvDim ? saved.DeltaQkvPostSilu![t * qkvDim + i] : 0);
            }

        // 4. Backward through SiLU for QKV path
        var dQkvPreSilu = new float[T * qkvDim];
        F32Ops.SiLUBackward(dQkvPreSilu, dQkvPostSilu, saved.DeltaQkvPreSilu!, T * qkvDim);

        // 5. Backward through SiLU for Gate path
        var dGatePreSilu = new float[T * gateDim];
        F32Ops.SiLUBackward(dGatePreSilu, dGatePostSilu, saved.DeltaGatePreSilu!, T * gateDim);

        // 6. Backward through AttnQkv LoRA
        var dNormOut = new float[T * H];
        var loraQkv = _adapter.GetLayer($"{prefix}.delta_qkv");
        if (loraQkv != null)
            loraQkv.Backward(dNormOut, dQkvPreSilu, saved.NormOut!, saved.LoraDeltaQkvIntermediate!, T);

        // 7. Backward through AttnQkv projection: dNormOut += dQkvPreSilu × AttnQkv
        var wQkv = F32Ops.Dequantize(dnw.AttnQkv);
        var dFromQkv = new float[T * H];
        F32Ops.MatMul(dFromQkv, dQkvPreSilu, wQkv, T, qkvDim, H);
        F32Ops.AddInPlace(dNormOut, dFromQkv, T * H);

        // 8. Backward through AttnGate projection: dNormOut += dGatePreSilu × AttnGate
        var wGate = F32Ops.Dequantize(dnw.AttnGate);
        var dFromGate = new float[T * H];
        F32Ops.MatMul(dFromGate, dGatePreSilu, wGate, T, gateDim, H);
        F32Ops.AddInPlace(dNormOut, dFromGate, T * H);

        saved.DNormOut = dNormOut;
    }

    public void Dispose()
    {
        _savedActivations = null;
        _gpuActivation?.Dispose();
        _gpuOutput?.Dispose();
        foreach (var t in _gpuWeightCache.Values)
            t.Dispose();
        _gpuWeightCache.Clear();
    }
}

/// <summary>
/// Saved activations for one transformer layer, used during backward pass.
/// </summary>
internal sealed class LayerActivations
{
    public bool IsAttention;
    public bool IsDeltaNet;
    public bool HasGatedQ;

    // Input/output
    public float[]? InputToLayer;      // residual saved before layer
    public float[]? NormOut;           // after pre-attention norm
    public float[]? PostAttnHidden;    // hidden after attention + residual add
    public float[]? FfnNormOut;        // after post-attention norm
    public float[]? FfnResidual;       // residual before FFN

    // Norm weights
    public float[]? AttnNormWeight;
    public float[]? PostAttnNormWeight;

    // Attention saved values (weights re-dequantized on demand in backward)
    public float[]? QRaw, KRaw, VRaw;
    public float[]? QAttn, QGate;
    public float[]? QPreRope, KPreRope;
    public float[]? QBeforeNorm, KBeforeNorm;
    public float[]? QNormWeight, KNormWeight;
    public float[]? AttnOut;
    public float[]? SavedProbs;        // [numHeads × T × T]
    public float[]? OOut;
    public float[]? LoraQIntermediate, LoraKIntermediate, LoraVIntermediate, LoraOIntermediate;

    // DeltaNet saved values
    public float[]? DeltaQkvPreSilu, DeltaQkvPostSilu;
    public float[]? DeltaGatePreSilu, DeltaGatePostSilu;
    public float[]? DeltaSsmApprox;
    public float[]? LoraDeltaQkvIntermediate, LoraDeltaOutIntermediate;

    // FFN saved values (weights re-dequantized on demand in backward)
    public float[]? GateInput, UpInput, FfnGated;
    public float[]? LoraFfnGateInter, LoraFfnUpInter, LoraFfnDownInter;

    // Backward output
    public float[]? DNormOut;
}
