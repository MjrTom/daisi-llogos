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

    // GPU scratch tensors (reused across layers)
    private ITensor? _hidden;     // [T × H]
    private ITensor? _residual;   // [T × H]
    private ITensor? _normOut;    // [T × H]
    private ITensor? _dHidden;    // [T × H]
    private int _lastT;

    // LoRA tensors on GPU (uploaded from CPU, re-uploaded after optimizer step)
    private readonly Dictionary<string, (ITensor a, ITensor b)> _gpuLora = new();
    private readonly Dictionary<string, (ITensor dA, ITensor dB)> _gpuLoraGrad = new();
    // LoRA intermediates per layer (reused)
    private readonly Dictionary<string, ITensor> _gpuLoraIntermediate = new();

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
            var gpuDA = _gpu.CreateTensor($"lora.{name}.dA", GgmlType.F32, [layer.A.Size]);
            var gpuDB = _gpu.CreateTensor($"lora.{name}.dB", GgmlType.F32, [layer.B.Size]);
            _gpuLora[name] = (gpuA, gpuB);
            _gpuLoraGrad[name] = (gpuDA, gpuDB);

            // Intermediate buffer for LoRA forward
            var inter = _gpu.CreateTensor($"lora.{name}.inter", GgmlType.F32, [1]); // resized later
            _gpuLoraIntermediate[name] = inter;
        }
    }

    public float[]? Forward(int[] tokenIds, out float totalLoss, int[] targets)
    {
        int T = tokenIds.Length;
        int H = _config.HiddenDim;
        int V = _config.VocabSize;

        EnsureScratch(T, H);

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

        // 1. Embedding lookup → hidden [T × H]
        // Upload token IDs, do embedding on CPU, upload result
        var hiddenCpu = new float[T * H];
        var embTable = new float[_embeddingTable.ElementCount];
        _embeddingTable.DequantizeTo(embTable);
        for (int t = 0; t < T; t++)
            F32Ops.EmbeddingLookup(hiddenCpu.AsSpan(t * H, H), embTable, tokenIds[t], H);
        UploadF32(_hidden!, hiddenCpu);

        // 2. Transformer layers
        for (int layer = 0; layer < _config.NumLayers; layer++)
        {
            var lw = _weights.Layers[layer];

            // Pre-attention norm
            var normWeight = DequantToGpu($"l{layer}.attn_norm", lw.AttnNorm);
            _gpu.CopyTensor(_residual!, _hidden!);
            _gpu.RmsNorm(_normOut!, _hidden!, normWeight, _config.NormEps);

            var saved = new GpuLayerActivations
            {
                InputToLayer = CloneGpuTensor($"l{layer}.input", _residual!),
                NormOut = CloneGpuTensor($"l{layer}.normout", _normOut!),
                AttnNormWeight = normWeight,
            };

            if (lw is StandardAttentionWeights saw && _config.IsStandardAttention(layer))
                ForwardAttentionLayer(saw, saved, layer, T);
            else if (lw is DeltaNetWeights dnw)
                ForwardDeltaNetLayer(dnw, saved, layer, T);

            // Add residual
            _gpu.ElementAdd(_hidden!, _hidden!, _residual!);

            // Post-attention norm + FFN
            var postNormWeight = DequantToGpu($"l{layer}.post_norm", lw.PostAttnNorm);
            saved.PostAttnNormWeight = postNormWeight;
            saved.PostAttnHidden = CloneGpuTensor($"l{layer}.post_hidden", _hidden!);

            var ffnNormOut = _gpu.CreateTensor($"l{layer}.ffn_norm", GgmlType.F32, [T * H]);
            _gpu.RmsNorm(ffnNormOut, _hidden!, postNormWeight, _config.NormEps);
            saved.FfnNormOut = ffnNormOut;
            saved.FfnResidual = CloneGpuTensor($"l{layer}.ffn_res", _hidden!);

            int I = _config.IntermediateDim;
            var gateOut = _gpu.CreateTensor($"l{layer}.gate", GgmlType.F32, [T * I]);
            var upOut = _gpu.CreateTensor($"l{layer}.up", GgmlType.F32, [T * I]);
            var ffnGated = _gpu.CreateTensor($"l{layer}.ffn_gated", GgmlType.F32, [T * I]);

            _gpu.MatMul(gateOut, ffnNormOut, GpuWeight(lw.FfnGate), T, H, I);
            _gpu.MatMul(upOut, ffnNormOut, GpuWeight(lw.FfnUp), T, H, I);
            _gpu.SwiGLU(ffnGated, gateOut, upOut);

            var ffnOut = _gpu.CreateTensor($"l{layer}.ffn_out", GgmlType.F32, [T * H]);
            _gpu.MatMul(ffnOut, ffnGated, GpuWeight(lw.FfnDown), T, I, H);

            saved.GateInput = gateOut;
            saved.UpInput = upOut;
            saved.FfnGated = ffnGated;

            // hidden = ffnOut + residual
            _gpu.ElementAdd(_hidden!, ffnOut, saved.FfnResidual!);
            ffnOut.Dispose();

            _saved[layer] = saved;
        }

        // 3. Final norm + logit projection
        var finalNormOut = _gpu.CreateTensor("final_norm", GgmlType.F32, [T * H]);
        _gpu.RmsNorm(finalNormOut, _hidden!, _outputNormWeight!, _config.NormEps);

        var logits = _gpu.CreateTensor("logits", GgmlType.F32, [T * V]);
        _gpu.MatMul(logits, finalNormOut, GpuWeight(_weights.OutputWeight), T, H, V);

        // 4. Cross-entropy loss + gradient (on GPU, only downloads scalar loss)
        var dLogits = _gpu.CreateTensor("dlogits", GgmlType.F32, [T * V]);
        var gpuTargets = CreateIntTensor("targets", targets);
        totalLoss = _gpu.CrossEntropyLoss(dLogits, logits, gpuTargets, T, V);
        gpuTargets.Dispose();
        logits.Dispose();

        // 5. Backward pass
        Backward(dLogits, finalNormOut, T);
        dLogits.Dispose();
        finalNormOut.Dispose();

        // 6. Download LoRA gradients to CPU
        DownloadLoraGradients();

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

        var qRaw = _gpu.CreateTensor($"l{layer}.q", GgmlType.F32, [T * qOutDim]);
        var kRaw = _gpu.CreateTensor($"l{layer}.k", GgmlType.F32, [T * kOutDim]);
        var vRaw = _gpu.CreateTensor($"l{layer}.v", GgmlType.F32, [T * vOutDim]);

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
            qAttn = _gpu.CreateTensor($"l{layer}.qattn", GgmlType.F32, [T * qAttnDim]);
            qGate = _gpu.CreateTensor($"l{layer}.qgate", GgmlType.F32, [T * qAttnDim]);
            // Download, de-interleave on CPU, upload (TODO: GPU kernel for this)
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
            qGate = _gpu.CreateTensor($"l{layer}.qgate", GgmlType.F32, [T * qAttnDim]);
            _gpu.FillTensor(qGate, 88.0f); // sigmoid(88) ≈ 1
        }

        // RoPE (batched)
        _gpu.BatchedRoPE(qAttn, kRaw, keyDim, ropeDim, 0, _config.RopeTheta, numHeads, numKvHeads);

        // Causal attention — download to CPU, compute, upload
        // (The existing GPU GatedAttention uses KV cache which doesn't fit training.
        //  For training we need the full T×T attention matrix. Do on CPU for now.)
        int attnOutDim = numHeads * valDim;
        var attnOutCpu = new float[T * attnOutDim];
        var savedProbsCpu = new float[numHeads * T * T];
        var qAttnCpu2 = DownloadF32(qAttn, T * qAttnDim);
        var qGateCpu2 = DownloadF32(qGate, T * qAttnDim);
        var kRawCpu = DownloadF32(kRaw, T * kOutDim);
        var vRawCpu = DownloadF32(vRaw, T * vOutDim);
        F32Ops.CausalGatedAttention(attnOutCpu, qAttnCpu2, qGateCpu2,
            kRawCpu, vRawCpu, T, numHeads, numKvHeads, keyDim, valDim, scale, savedProbsCpu);

        var attnOut = CreateF32Tensor($"l{layer}.attn_out", attnOutCpu);
        var savedProbs = CreateF32Tensor($"l{layer}.probs", savedProbsCpu);

        // O projection
        int oDim = (int)saw.AttnO.Dimensions[0];
        var oOut = _gpu.CreateTensor($"l{layer}.oout", GgmlType.F32, [T * H]);
        _gpu.MatMul(oOut, attnOut, GpuWeight(saw.AttnO), T, oDim, H);
        ApplyLoraForward($"{prefix}.attn_o", attnOut, oOut, T);

        // Write to hidden
        _gpu.CopyTensor(_hidden!, oOut);
        oOut.Dispose();

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
        int H = _config.HiddenDim;
        string prefix = $"blk.{layer}";

        int qkvDim = (int)dnw.AttnQkv.Dimensions[1];
        var qkvOut = _gpu.CreateTensor($"l{layer}.qkv", GgmlType.F32, [T * qkvDim]);
        _gpu.MatMul(qkvOut, saved.NormOut!, GpuWeight(dnw.AttnQkv), T, H, qkvDim);
        ApplyLoraForward($"{prefix}.delta_qkv", saved.NormOut!, qkvOut, T);

        // Save pre-SiLU
        saved.DeltaQkvPreSilu = CloneGpuTensor($"l{layer}.qkv_pre", qkvOut);
        _gpu.SiLU(qkvOut, qkvOut);
        saved.DeltaQkvPostSilu = CloneGpuTensor($"l{layer}.qkv_post", qkvOut);

        int gateDim = (int)dnw.AttnGate.Dimensions[1];
        var gateOut = _gpu.CreateTensor($"l{layer}.gate", GgmlType.F32, [T * gateDim]);
        _gpu.MatMul(gateOut, saved.NormOut!, GpuWeight(dnw.AttnGate), T, H, gateDim);
        saved.DeltaGatePreSilu = CloneGpuTensor($"l{layer}.gate_pre", gateOut);
        _gpu.SiLU(gateOut, gateOut);
        saved.DeltaGatePostSilu = CloneGpuTensor($"l{layer}.gate_post", gateOut);

        // SSM approximation: ssmApprox = qkvPostSilu * gatePostSilu (truncated to outInDim)
        int outInDim = (int)dnw.SsmOut.Dimensions[0];
        // Element-wise on min(qkvDim, gateDim, outInDim) — do on CPU for dimension mismatch
        var qkvCpu = DownloadF32(qkvOut, T * qkvDim);
        var gateCpu = DownloadF32(gateOut, T * gateDim);
        var ssmCpu = new float[T * outInDim];
        for (int t = 0; t < T; t++)
            for (int i = 0; i < outInDim; i++)
            {
                float q = i < qkvDim ? qkvCpu[t * qkvDim + i] : 0;
                float g = i < gateDim ? gateCpu[t * gateDim + i] : 0;
                ssmCpu[t * outInDim + i] = q * g;
            }
        var ssmApprox = CreateF32Tensor($"l{layer}.ssm", ssmCpu);
        saved.DeltaSsmApprox = ssmApprox;

        var layerOut = _gpu.CreateTensor($"l{layer}.delta_out", GgmlType.F32, [T * H]);
        _gpu.MatMul(layerOut, ssmApprox, GpuWeight(dnw.SsmOut), T, outInDim, H);
        ApplyLoraForward($"{prefix}.delta_out", ssmApprox, layerOut, T);

        _gpu.CopyTensor(_hidden!, layerOut);
        layerOut.Dispose();
        qkvOut.Dispose();
        gateOut.Dispose();

        saved.IsDeltaNet = true;
    }

    private void Backward(ITensor dLogits, ITensor finalNormOut, int T)
    {
        int H = _config.HiddenDim;
        int V = _config.VocabSize;

        // dFinalNormOut = dLogits × outputWeight
        var dFinalNormOut = _gpu.CreateTensor("d_fnorm", GgmlType.F32, [T * H]);
        // Need F32 output weight for backward matmul
        var outWeightF32 = DequantToGpu("out_weight_f32", _weights.OutputWeight);
        _gpu.SgemmF32(dFinalNormOut, dLogits, outWeightF32, T, V, H);

        // Backward through final RMS norm
        EnsureDHidden(T, H);
        _gpu.ZeroTensor(_dHidden!);
        _gpu.BatchedRmsNormBackward(_dHidden!, dFinalNormOut, _hidden!, _outputNormWeight!,
            _config.NormEps, H, T);
        dFinalNormOut.Dispose();

        // Backward through layers in reverse
        for (int layer = _config.NumLayers - 1; layer >= 0; layer--)
        {
            var saved = _saved![layer];
            var lw = _weights.Layers[layer];
            int I = _config.IntermediateDim;

            // FFN backward
            var dFfnOut = CloneGpuTensor($"d_l{layer}.ffn_out", _dHidden!);

            // Down projection backward
            var downF32 = DequantToGpu($"d_l{layer}.down", lw.FfnDown);
            var dFfnGated = _gpu.CreateTensor($"d_l{layer}.ffn_gated", GgmlType.F32, [T * I]);
            _gpu.SgemmF32(dFfnGated, dFfnOut, downF32, T, H, I);

            // SwiGLU backward
            var dGate = _gpu.CreateTensor($"d_l{layer}.gate", GgmlType.F32, [T * I]);
            var dUp = _gpu.CreateTensor($"d_l{layer}.up", GgmlType.F32, [T * I]);
            _gpu.ZeroTensor(dGate);
            _gpu.ZeroTensor(dUp);
            _gpu.SwiGLUBackward(dGate, dUp, dFfnGated, saved.GateInput!, saved.UpInput!);
            dFfnGated.Dispose();

            // Gate/up projection backward
            var dFfnNormOut = _gpu.CreateTensor($"d_l{layer}.ffn_norm", GgmlType.F32, [T * H]);
            _gpu.ZeroTensor(dFfnNormOut);
            var gateF32 = DequantToGpu($"d_l{layer}.gate_w", lw.FfnGate);
            var upF32 = DequantToGpu($"d_l{layer}.up_w", lw.FfnUp);
            var dFromGate = _gpu.CreateTensor($"d_l{layer}.from_gate", GgmlType.F32, [T * H]);
            var dFromUp = _gpu.CreateTensor($"d_l{layer}.from_up", GgmlType.F32, [T * H]);
            _gpu.SgemmF32(dFromGate, dGate, gateF32, T, I, H);
            _gpu.AddInPlace(dFfnNormOut, dFromGate);
            _gpu.SgemmF32(dFromUp, dUp, upF32, T, I, H);
            _gpu.AddInPlace(dFfnNormOut, dFromUp);
            dGate.Dispose(); dUp.Dispose(); dFromGate.Dispose(); dFromUp.Dispose();

            // Post-attention RmsNorm backward
            var dPostAttnHidden = _gpu.CreateTensor($"d_l{layer}.post_attn", GgmlType.F32, [T * H]);
            _gpu.ZeroTensor(dPostAttnHidden);
            _gpu.BatchedRmsNormBackward(dPostAttnHidden, dFfnNormOut,
                saved.PostAttnHidden!, saved.PostAttnNormWeight!, _config.NormEps, H, T);
            dFfnNormOut.Dispose();

            // Add residual gradient
            _gpu.AddInPlace(dPostAttnHidden, dFfnOut);
            dFfnOut.Dispose();

            var dAttnOut = CloneGpuTensor($"d_l{layer}.attn", dPostAttnHidden);

            // Layer-specific backward
            if (saved.IsAttention)
                BackwardAttentionLayer(saved, dAttnOut, layer, T);
            else if (saved.IsDeltaNet)
                BackwardDeltaNetLayer(saved, dAttnOut, layer, T);
            dAttnOut.Dispose();

            // Pre-attention RmsNorm backward
            var dInputToLayer = _gpu.CreateTensor($"d_l{layer}.input", GgmlType.F32, [T * H]);
            _gpu.ZeroTensor(dInputToLayer);
            if (saved.DNormOut != null)
            {
                _gpu.BatchedRmsNormBackward(dInputToLayer, saved.DNormOut,
                    saved.InputToLayer!, saved.AttnNormWeight!, _config.NormEps, H, T);
            }

            // Combine: dHidden = dResidual + dInputToLayer
            _gpu.ElementAdd(_dHidden!, dPostAttnHidden, dInputToLayer);
            dPostAttnHidden.Dispose();
            dInputToLayer.Dispose();

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
        var dAttnOutFromO = _gpu.CreateTensor($"d_l{layer}.attn_out", GgmlType.F32, [T * oDim]);
        _gpu.SgemmF32(dAttnOutFromO, dOOut, oF32, T, H, oDim);

        // Attention backward — GPU kernel
        var dQAttn = _gpu.CreateTensor($"d_l{layer}.dqattn", GgmlType.F32, [T * qAttnDim]);
        var dQGate = _gpu.CreateTensor($"d_l{layer}.dqgate", GgmlType.F32, [T * qAttnDim]);
        var dK = _gpu.CreateTensor($"d_l{layer}.dk", GgmlType.F32, [T * kOutDim]);
        var dV = _gpu.CreateTensor($"d_l{layer}.dv", GgmlType.F32, [T * vOutDim]);
        _gpu.ZeroTensor(dQAttn); _gpu.ZeroTensor(dQGate);
        _gpu.ZeroTensor(dK); _gpu.ZeroTensor(dV);
        _gpu.CausalGatedAttentionBackward(dQAttn, dQGate, dK, dV,
            dAttnOutFromO, saved.QAttn!, saved.QGate!,
            saved.KRaw!, saved.VRaw!, saved.SavedProbs!, saved.AttnOut!,
            T, numHeads, numKvHeads, keyDim, valDim, scale);
        dAttnOutFromO.Dispose();

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
        dQGate.Dispose();

        var dNormOut = _gpu.CreateTensor($"d_l{layer}.dnorm", GgmlType.F32, [T * H]);
        _gpu.ZeroTensor(dNormOut);

        var qF32 = DequantToGpu($"d_l{layer}.qW", saw.AttnQ);
        var kF32 = DequantToGpu($"d_l{layer}.kW", saw.AttnK);
        var vF32 = DequantToGpu($"d_l{layer}.vW", saw.AttnV);

        var dFromQ = _gpu.CreateTensor($"d_l{layer}.fq", GgmlType.F32, [T * H]);
        var dFromK = _gpu.CreateTensor($"d_l{layer}.fk", GgmlType.F32, [T * H]);
        var dFromV = _gpu.CreateTensor($"d_l{layer}.fv", GgmlType.F32, [T * H]);
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

        dQRaw.Dispose(); dK.Dispose(); dV.Dispose();
        dFromQ.Dispose(); dFromK.Dispose(); dFromV.Dispose();

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
        var dSsmApprox = _gpu.CreateTensor($"d_l{layer}.dssm", GgmlType.F32, [T * outInDim]);
        _gpu.SgemmF32(dSsmApprox, dLayerOut, outF32, T, H, outInDim);

        // Element-wise product backward (CPU for dimension mismatch)
        var dSsmCpu = DownloadF32(dSsmApprox, T * outInDim);
        var qkvPostCpu = DownloadF32(saved.DeltaQkvPostSilu!, T * qkvDim);
        var gatePostCpu = DownloadF32(saved.DeltaGatePostSilu!, T * gateDim);
        var dQkvPostCpu = new float[T * qkvDim];
        var dGatePostCpu = new float[T * gateDim];
        for (int t = 0; t < T; t++)
            for (int i = 0; i < outInDim; i++)
            {
                float dS = dSsmCpu[t * outInDim + i];
                if (i < qkvDim)
                    dQkvPostCpu[t * qkvDim + i] = dS * (i < gateDim ? gatePostCpu[t * gateDim + i] : 0);
                if (i < gateDim)
                    dGatePostCpu[t * gateDim + i] = dS * (i < qkvDim ? qkvPostCpu[t * qkvDim + i] : 0);
            }
        dSsmApprox.Dispose();

        // SiLU backward (CPU)
        var qkvPreCpu = DownloadF32(saved.DeltaQkvPreSilu!, T * qkvDim);
        var dQkvPreCpu = new float[T * qkvDim];
        F32Ops.SiLUBackward(dQkvPreCpu, dQkvPostCpu, qkvPreCpu, T * qkvDim);

        var gatePreCpu = DownloadF32(saved.DeltaGatePreSilu!, T * gateDim);
        var dGatePreCpu = new float[T * gateDim];
        F32Ops.SiLUBackward(dGatePreCpu, dGatePostCpu, gatePreCpu, T * gateDim);

        // Upload gradients for projection backward
        var dQkvPre = CreateF32Tensor($"d_l{layer}.dqkv", dQkvPreCpu);
        var dGatePre = CreateF32Tensor($"d_l{layer}.dgate", dGatePreCpu);

        // LoRA backward for AttnQkv
        ApplyLoraBackward($"{prefix}.delta_qkv", saved.NormOut!, dQkvPre, T);

        // Projection backward → dNormOut
        var dNormOut = _gpu.CreateTensor($"d_l{layer}.dnorm", GgmlType.F32, [T * H]);
        _gpu.ZeroTensor(dNormOut);

        var qkvF32 = DequantToGpu($"d_l{layer}.qkvW", dnw.AttnQkv);
        var gateF32 = DequantToGpu($"d_l{layer}.gateW", dnw.AttnGate);
        var dFromQkv = _gpu.CreateTensor($"d_l{layer}.fqkv", GgmlType.F32, [T * H]);
        var dFromGate = _gpu.CreateTensor($"d_l{layer}.fgate", GgmlType.F32, [T * H]);
        _gpu.SgemmF32(dFromQkv, dQkvPre, qkvF32, T, qkvDim, H);
        _gpu.AddInPlace(dNormOut, dFromQkv);
        _gpu.SgemmF32(dFromGate, dGatePre, gateF32, T, gateDim, H);
        _gpu.AddInPlace(dNormOut, dFromGate);

        dQkvPre.Dispose(); dGatePre.Dispose(); dFromQkv.Dispose(); dFromGate.Dispose();
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
        var loraOut = _gpu.CreateTensor($"lora.{name}.out", GgmlType.F32, [T * outDim]);
        _gpu.SgemmTransB(loraOut, inter, lora.b, T, rank, outDim);
        // output += scaling * loraOut
        _gpu.ScaleInPlace(loraOut, layer.Scaling);
        _gpu.AddInPlace(output, loraOut);
        loraOut.Dispose();
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
        var dInter = _gpu.CreateTensor($"lora.{name}.dinter", GgmlType.F32, [T * rank]);
        _gpu.SgemmF32(dInter, scaledDOut, lora.b, T, outDim, rank);
        scaledDOut.Dispose();

        // dA += dIntermediate^T × input → [rank × inDim]
        _gpu.SgemmTransAAccumulate(grad.dA, dInter, input, rank, T, inDim);

        dInter.Dispose();
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

    /// <summary>Re-upload LoRA weights after optimizer step.</summary>
    public void SyncLoraWeights()
    {
        foreach (var (name, layer) in _adapter.Layers)
        {
            var lora = _gpuLora[name];
            UploadF32(lora.a, layer.A.Data);
            UploadF32(lora.b, layer.B.Data);
            // Zero gradients on GPU for next step
            var grad = _gpuLoraGrad[name];
            _gpu.ZeroTensor(grad.dA);
            _gpu.ZeroTensor(grad.dB);
        }
    }

    // ── GPU tensor helpers ──────────────────────────────────────────────────

    private void EnsureScratch(int T, int H)
    {
        if (_lastT == T) return;
        _hidden?.Dispose();
        _residual?.Dispose();
        _normOut?.Dispose();
        _hidden = _gpu.CreateTensor("hidden", GgmlType.F32, [T * H]);
        _residual = _gpu.CreateTensor("residual", GgmlType.F32, [T * H]);
        _normOut = _gpu.CreateTensor("normout", GgmlType.F32, [T * H]);
        _lastT = T;
    }

    private void EnsureDHidden(int T, int H)
    {
        if (_dHidden != null && _dHidden.ElementCount == T * H) return;
        _dHidden?.Dispose();
        _dHidden = _gpu.CreateTensor("dhidden", GgmlType.F32, [T * H]);
    }

    private ITensor CreateF32Tensor(string name, float[] data)
    {
        var t = _gpu.CreateTensor(name, GgmlType.F32, [data.Length]);
        UploadF32(t, data);
        return t;
    }

    private ITensor CreateIntTensor(string name, int[] data)
    {
        // Store as raw bytes (4 bytes per int)
        var t = _gpu.CreateTensor(name, GgmlType.F32, [data.Length]); // abuse F32 for int storage
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
        var dst = _gpu.CreateTensor(name, GgmlType.F32, [(int)src.ElementCount]);
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
        var gpu = _gpu.LoadTensor(cpuWeight.Name, cpuWeight.Type, cpuWeight.Dimensions, rawBytes);
        _gpuWeightCache[cpuWeight] = gpu;
        return gpu;
    }

    private ITensor DequantToGpu(string name, ITensor quantized)
    {
        if (_dequantCache.TryGetValue(name, out var cached))
        {
            // Check if same size
            if (cached.ElementCount == quantized.ElementCount)
            {
                var f32 = F32Ops.Dequantize(quantized);
                UploadF32(cached, f32);
                return cached;
            }
            cached.Dispose();
        }
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
        var t = _gpu.CreateTensor($"lora.{name}.inter", GgmlType.F32, [size]);
        _gpuLoraIntermediate[name] = t;
        return t;
    }

    public void Dispose()
    {
        _hidden?.Dispose();
        _residual?.Dispose();
        _normOut?.Dispose();
        _dHidden?.Dispose();
        _embeddingTable?.Dispose();
        _outputNormWeight?.Dispose();
        foreach (var (a, b) in _gpuLora.Values) { a.Dispose(); b.Dispose(); }
        foreach (var (da, db) in _gpuLoraGrad.Values) { da.Dispose(); db.Dispose(); }
        foreach (var t in _gpuLoraIntermediate.Values) t.Dispose();
        foreach (var t in _dequantCache.Values) t.Dispose();
        foreach (var t in _gpuWeightCache.Values) t.Dispose();
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
        InputToLayer?.Dispose(); NormOut?.Dispose();
        PostAttnHidden?.Dispose(); FfnNormOut?.Dispose(); FfnResidual?.Dispose();
        QRaw?.Dispose(); KRaw?.Dispose(); VRaw?.Dispose();
        QAttn?.Dispose(); QGate?.Dispose();
        AttnOut?.Dispose(); SavedProbs?.Dispose();
        GateInput?.Dispose(); UpInput?.Dispose(); FfnGated?.Dispose();
        DeltaQkvPreSilu?.Dispose(); DeltaQkvPostSilu?.Dispose();
        DeltaGatePreSilu?.Dispose(); DeltaGatePostSilu?.Dispose();
        DeltaSsmApprox?.Dispose();
        DNormOut?.Dispose();
        // Don't dispose AttnNormWeight/PostAttnNormWeight — owned by dequant cache
    }
}
