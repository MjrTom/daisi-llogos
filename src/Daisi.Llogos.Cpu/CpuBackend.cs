using System.Runtime.InteropServices;
using Daisi.Llogos.Gguf;

namespace Daisi.Llogos.Cpu;

/// <summary>
/// CPU compute backend with SIMD-optimized tensor operations.
/// </summary>
public sealed class CpuBackend : IComputeBackend
{
    /// <inheritdoc />
    public string Name => "CPU";

    /// <inheritdoc />
    public ITensor CreateTensor(string name, GgmlType type, ReadOnlySpan<long> dimensions)
    {
        return new CpuTensor(name, type, dimensions);
    }

    /// <inheritdoc />
    public ITensor LoadTensor(string name, GgmlType type, ReadOnlySpan<long> dimensions, ReadOnlySpan<byte> data)
    {
        return new CpuTensor(name, type, dimensions, data);
    }

    /// <inheritdoc />
    public void MatMul(ITensor output, ITensor a, ITensor b, int M, int K, int N)
    {
        var o = ((CpuTensor)output).AsFloatSpan();
        var aSpan = ((CpuTensor)a).AsFloatSpan();

        if (b.Type == GgmlType.F32)
        {
            var bSpan = ((CpuTensor)b).AsFloatSpan();
            Cpu.MatMul.Multiply(o, aSpan, bSpan, M, K, N);
        }
        else if (b.Type == GgmlType.Q8_0)
        {
            var bRaw = ((CpuTensor)b).RawData;
            Cpu.MatMul.MultiplyQ8_0(o, aSpan, bRaw, M, K, N);
        }
        else if (b.Type == GgmlType.TQ1_0)
        {
            var bRaw = ((CpuTensor)b).RawData;
            Cpu.TernaryMatMul.MultiplyTQ1_0(o, aSpan, bRaw, M, K, N);
        }
        else if (b.Type == GgmlType.I2_S)
        {
            var bRaw = ((CpuTensor)b).RawData;
            Cpu.I2SDequant.Multiply(o, aSpan, bRaw, M, K, N);
        }
        else if (b.Type == GgmlType.F16)
        {
            var bRaw = ((CpuTensor)b).RawData;
            var bHalf = MemoryMarshal.Cast<byte, Half>(bRaw);
            Cpu.MatMul.MultiplyF16(o, aSpan, bHalf, M, K, N);
        }
        else
        {
            throw new NotSupportedException($"MatMul not implemented for weight type {b.Type}.");
        }
    }

    /// <inheritdoc />
    public void RmsNorm(ITensor output, ITensor input, ITensor weight, float eps)
    {
        var o = ((CpuTensor)output).AsFloatSpan();
        var i = ((CpuTensor)input).AsFloatSpan();
        var w = ((CpuTensor)weight).AsFloatSpan();
        Cpu.RmsNorm.Apply(o, i, w, eps);
    }

    /// <inheritdoc />
    public void Softmax(ITensor output, ITensor input)
    {
        var o = ((CpuTensor)output).AsFloatSpan();
        var i = ((CpuTensor)input).AsFloatSpan();
        Cpu.Softmax.Apply(o, i);
    }

    /// <inheritdoc />
    public void SiLU(ITensor output, ITensor input)
    {
        var o = ((CpuTensor)output).AsFloatSpan();
        var i = ((CpuTensor)input).AsFloatSpan();
        Cpu.Silu.Apply(o, i);
    }

    /// <inheritdoc />
    public void RoPE(ITensor q, ITensor k, int headDim, int ropeDim, int positionOffset, float ropeTheta)
    {
        var qSpan = ((CpuTensor)q).AsFloatSpan();
        var kSpan = ((CpuTensor)k).AsFloatSpan();
        int effectiveRopeDim = ropeDim > 0 ? ropeDim : headDim;
        Cpu.Rope.Apply(qSpan, kSpan, headDim, effectiveRopeDim, positionOffset, ropeTheta);
    }

    /// <inheritdoc />
    public void ElementMul(ITensor output, ITensor a, ITensor b)
    {
        var o = ((CpuTensor)output).AsFloatSpan();
        var aSpan = ((CpuTensor)a).AsFloatSpan();
        var bSpan = ((CpuTensor)b).AsFloatSpan();
        Cpu.ElementOps.Multiply(o, aSpan, bSpan);
    }

    /// <inheritdoc />
    public void ElementAdd(ITensor output, ITensor a, ITensor b)
    {
        var o = ((CpuTensor)output).AsFloatSpan();
        var aSpan = ((CpuTensor)a).AsFloatSpan();
        var bSpan = ((CpuTensor)b).AsFloatSpan();
        Cpu.ElementOps.Add(o, aSpan, bSpan);
    }

    /// <inheritdoc />
    public void EmbeddingLookup(ITensor output, ITensor table, int tokenId)
    {
        var outSpan = ((CpuTensor)output).AsFloatSpan();
        var raw = ((CpuTensor)table).RawData;
        int hiddenDim = (int)table.Dimensions[0]; // GGUF dim[0] = row length

        switch (table.Type)
        {
            case GgmlType.F32:
            {
                var floats = MemoryMarshal.Cast<byte, float>(raw);
                floats.Slice(tokenId * hiddenDim, hiddenDim).CopyTo(outSpan);
                break;
            }
            case GgmlType.Q8_0:
            {
                int blocksPerRow = hiddenDim / 32;
                int bytesPerRow = blocksPerRow * 34;
                var rowData = raw.Slice(tokenId * bytesPerRow, bytesPerRow);
                Dequantize.DequantizeQ8_0(rowData, outSpan.Slice(0, hiddenDim));
                break;
            }
            case GgmlType.F16:
            {
                int bytesPerRow = hiddenDim * 2;
                var rowBytes = raw.Slice(tokenId * bytesPerRow, bytesPerRow);
                var halfs = MemoryMarshal.Cast<byte, Half>(rowBytes);
                for (int i = 0; i < hiddenDim; i++)
                    outSpan[i] = (float)halfs[i];
                break;
            }
            default:
                throw new NotSupportedException($"EmbeddingLookup not implemented for type {table.Type}.");
        }
    }

    // ── Composite Operations ────────────────────────────────────────────────

    /// <inheritdoc />
    public void CopyTensor(ITensor dst, ITensor src) =>
        src.AsFloatSpan().CopyTo(dst.AsFloatSpan());

    /// <inheritdoc />
    public void SiLUInPlace(ITensor data)
    {
        var span = data.AsFloatSpan();
        for (int i = 0; i < span.Length; i++)
        {
            float x = span[i];
            span[i] = x / (1.0f + MathF.Exp(-x));
        }
    }

    /// <inheritdoc />
    public void L2NormGroups(ITensor data, int numGroups, int groupDim)
    {
        var span = data.AsFloatSpan();
        for (int g = 0; g < numGroups; g++)
        {
            int off = g * groupDim;
            float sumSq = 0;
            for (int i = 0; i < groupDim; i++)
                sumSq += span[off + i] * span[off + i];
            float invNorm = 1.0f / MathF.Sqrt(sumSq + 1e-12f);
            for (int i = 0; i < groupDim; i++)
                span[off + i] *= invNorm;
        }
    }

    /// <inheritdoc />
    public void PerHeadRmsNorm(ITensor data, ITensor weight, int numHeads, int headDim, float eps)
    {
        var span = data.AsFloatSpan();
        var w = weight.AsFloatSpan();
        for (int h = 0; h < numHeads; h++)
        {
            int off = h * headDim;
            float sumSq = 0;
            for (int i = 0; i < headDim; i++)
                sumSq += span[off + i] * span[off + i];
            float rms = MathF.Sqrt(sumSq / headDim + eps);
            float invRms = 1.0f / rms;
            for (int i = 0; i < headDim; i++)
                span[off + i] = span[off + i] * invRms * w[i];
        }
    }

    /// <inheritdoc />
    public void DeInterleaveQ(ITensor qAttn, ITensor qGate, ITensor qFull, int numHeads, int headDim)
    {
        var full = qFull.AsFloatSpan();
        var attn = qAttn.AsFloatSpan();
        var gate = qGate.AsFloatSpan();
        for (int h = 0; h < numHeads; h++)
        {
            int srcOff = h * 2 * headDim;
            int dstOff = h * headDim;
            full.Slice(srcOff, headDim).CopyTo(attn.Slice(dstOff, headDim));
            full.Slice(srcOff + headDim, headDim).CopyTo(gate.Slice(dstOff, headDim));
        }
    }

    /// <inheritdoc />
    public void KvCacheWrite(ITensor kCache, ITensor vCache, ITensor k, ITensor v,
        int nKvHeads, int keyLength, int valueLength, int maxSeqLen, int position)
    {
        var kSpan = k.AsFloatSpan();
        var vSpan = v.AsFloatSpan();

        if (kCache.Type == GgmlType.F16)
        {
            var kCacheHalf = MemoryMarshal.Cast<byte, Half>(((CpuTensor)kCache).RawDataMut);
            var vCacheHalf = MemoryMarshal.Cast<byte, Half>(((CpuTensor)vCache).RawDataMut);
            for (int h = 0; h < nKvHeads; h++)
            {
                int kCacheOff = h * maxSeqLen * keyLength + position * keyLength;
                for (int d = 0; d < keyLength; d++)
                    kCacheHalf[kCacheOff + d] = (Half)kSpan[h * keyLength + d];

                int vCacheOff = h * maxSeqLen * valueLength + position * valueLength;
                for (int d = 0; d < valueLength; d++)
                    vCacheHalf[vCacheOff + d] = (Half)vSpan[h * valueLength + d];
            }
        }
        else
        {
            var kCacheSpan = kCache.AsFloatSpan();
            var vCacheSpan = vCache.AsFloatSpan();
            for (int h = 0; h < nKvHeads; h++)
            {
                int kCacheOff = h * maxSeqLen * keyLength + position * keyLength;
                kSpan.Slice(h * keyLength, keyLength).CopyTo(kCacheSpan.Slice(kCacheOff, keyLength));

                int vCacheOff = h * maxSeqLen * valueLength + position * valueLength;
                vSpan.Slice(h * valueLength, valueLength).CopyTo(vCacheSpan.Slice(vCacheOff, valueLength));
            }
        }
    }

    /// <inheritdoc />
    public void GatedAttention(ITensor output, ITensor qAttn, ITensor qGate,
        ITensor kCache, ITensor vCache,
        int numHeads, int numKvHeads, int keyLength, int valueLength,
        int maxSeqLen, int seqLen, float scale)
    {
        var outSpan = output.AsFloatSpan();
        var qAttnSpan = qAttn.AsFloatSpan();
        var qGateSpan = qGate.AsFloatSpan();
        bool isFp16 = kCache.Type == GgmlType.F16;

        int headsPerGroup = numHeads / numKvHeads;
        int kHeadStride = maxSeqLen * keyLength;
        int vHeadStride = maxSeqLen * valueLength;

        // Tiled online softmax — constant memory regardless of seqLen
        const int tileSize = 256;
        Span<float> tileScores = stackalloc float[tileSize];

        // Get cache spans based on type
        Span<Half> kCacheHalf = default, vCacheHalf = default;
        Span<float> kCacheF32 = default, vCacheF32 = default;
        if (isFp16)
        {
            kCacheHalf = MemoryMarshal.Cast<byte, Half>(((CpuTensor)kCache).RawDataMut);
            vCacheHalf = MemoryMarshal.Cast<byte, Half>(((CpuTensor)vCache).RawDataMut);
        }
        else
        {
            kCacheF32 = kCache.AsFloatSpan();
            vCacheF32 = vCache.AsFloatSpan();
        }

        for (int h = 0; h < numHeads; h++)
        {
            int kvHead = h / headsPerGroup;
            int qOff = h * keyLength;
            int kvKBase = kvHead * kHeadStride;
            int kvVBase = kvHead * vHeadStride;
            int outOff = h * valueLength;

            // Initialize output to zero
            for (int d = 0; d < valueLength; d++)
                outSpan[outOff + d] = 0;

            float runningMax = float.NegativeInfinity;
            float runningSum = 0;

            // Process tiles
            for (int tileStart = 0; tileStart < seqLen; tileStart += tileSize)
            {
                int tileEnd = Math.Min(tileStart + tileSize, seqLen);
                int tileLen = tileEnd - tileStart;

                // Compute attention scores for this tile
                for (int t = 0; t < tileLen; t++)
                {
                    int p = tileStart + t;
                    float dot = 0;
                    int kOff = kvKBase + p * keyLength;
                    if (isFp16)
                        for (int d = 0; d < keyLength; d++)
                            dot += qAttnSpan[qOff + d] * (float)kCacheHalf[kOff + d];
                    else
                        for (int d = 0; d < keyLength; d++)
                            dot += qAttnSpan[qOff + d] * kCacheF32[kOff + d];
                    tileScores[t] = dot * scale;
                }

                // Tile max
                float tileMax = float.NegativeInfinity;
                for (int t = 0; t < tileLen; t++)
                    if (tileScores[t] > tileMax) tileMax = tileScores[t];

                // Exp and tile sum
                float tileSum = 0;
                for (int t = 0; t < tileLen; t++)
                {
                    tileScores[t] = MathF.Exp(tileScores[t] - tileMax);
                    tileSum += tileScores[t];
                }

                // Online softmax merge
                float newMax = MathF.Max(runningMax, tileMax);
                float correctionOld = MathF.Exp(runningMax - newMax);
                float correctionNew = MathF.Exp(tileMax - newMax);

                // Update output: rescale old + add new tile
                for (int d = 0; d < valueLength; d++)
                {
                    float tileVal = 0;
                    if (isFp16)
                        for (int t = 0; t < tileLen; t++)
                            tileVal += tileScores[t] * (float)vCacheHalf[kvVBase + (tileStart + t) * valueLength + d];
                    else
                        for (int t = 0; t < tileLen; t++)
                            tileVal += tileScores[t] * vCacheF32[kvVBase + (tileStart + t) * valueLength + d];

                    outSpan[outOff + d] = outSpan[outOff + d] * correctionOld + tileVal * correctionNew;
                }

                runningSum = runningSum * correctionOld + tileSum * correctionNew;
                runningMax = newMax;
            }

            // Normalize and apply sigmoid gating
            float invSum = runningSum > 0 ? 1.0f / runningSum : 0;
            for (int d = 0; d < valueLength; d++)
            {
                float gateVal = d < keyLength
                    ? 1.0f / (1.0f + MathF.Exp(-qGateSpan[h * keyLength + d]))
                    : 1.0f;
                outSpan[outOff + d] = outSpan[outOff + d] * invSum * gateVal;
            }
        }
    }

    /// <inheritdoc />
    public void CausalConv1d(ITensor qkv, ITensor convBuffer, ITensor convWeight, int channels, int kernelSize)
    {
        var qkvSpan = qkv.AsFloatSpan();
        var convBuf = convBuffer.AsFloatSpan();
        var convW = convWeight.AsFloatSpan();
        int bufSlots = kernelSize - 1;

        var result = new float[channels];

        for (int c = 0; c < channels; c++)
        {
            float s = 0;
            for (int k = 0; k < bufSlots; k++)
                s += convBuf[k * channels + c] * convW[c * kernelSize + k];
            s += qkvSpan[c] * convW[c * kernelSize + bufSlots];
            result[c] = s;
        }

        // Shift buffer: discard oldest, add current (pre-conv)
        for (int k = 0; k < bufSlots - 1; k++)
            convBuf.Slice((k + 1) * channels, channels).CopyTo(convBuf.Slice(k * channels, channels));
        if (bufSlots > 0)
            qkvSpan.Slice(0, channels).CopyTo(convBuf.Slice((bufSlots - 1) * channels, channels));

        result.AsSpan().CopyTo(qkvSpan);
    }

    /// <inheritdoc />
    public void ComputeDecayBeta(ITensor decay, ITensor beta, ITensor alphaProj, ITensor betaProj,
        ITensor ssmA, ITensor dtBias, int groupCount)
    {
        var dSpan = decay.AsFloatSpan();
        var bSpan = beta.AsFloatSpan();
        var aProj = alphaProj.AsFloatSpan();
        var bProj = betaProj.AsFloatSpan();
        var sA = ssmA.AsFloatSpan();
        var dt = dtBias.AsFloatSpan();

        for (int g = 0; g < groupCount; g++)
        {
            float softplus = MathF.Log(1.0f + MathF.Exp(aProj[g] + dt[g]));
            dSpan[g] = MathF.Exp(sA[g] * softplus);
            bSpan[g] = 1.0f / (1.0f + MathF.Exp(-bProj[g]));
        }
    }

    /// <inheritdoc />
    public void DeltaNetStep(ITensor output, ITensor q, ITensor k, ITensor v,
        ITensor state, ITensor decay, ITensor beta,
        ITensor normWeight, int groupCount, int headDim, float scale, float normEps)
    {
        var outSpan = output.AsFloatSpan();
        var qSpan = q.AsFloatSpan();
        var kSpan = k.AsFloatSpan();
        var vSpan = v.AsFloatSpan();
        var stateSpan = state.AsFloatSpan();
        var dSpan = decay.AsFloatSpan();
        var bSpan = beta.AsFloatSpan();
        var normW = normWeight.AsFloatSpan();

        var error = new float[headDim];

        for (int g = 0; g < groupCount; g++)
        {
            int baseOff = g * headDim;
            int stateOff = g * headDim * headDim;
            float d = dSpan[g];
            float b = bSpan[g];

            // 1. sk = S^T * k (from OLD state)
            // 2. error = (v - d*sk) * beta
            for (int j = 0; j < headDim; j++)
            {
                float sk = 0;
                for (int i = 0; i < headDim; i++)
                    sk += stateSpan[stateOff + i * headDim + j] * kSpan[baseOff + i];
                error[j] = (vSpan[baseOff + j] - d * sk) * b;
            }

            // 3. S_new = d * S_old + k * error^T
            for (int i = 0; i < headDim; i++)
            {
                float ki = kSpan[baseOff + i];
                int rowOff = stateOff + i * headDim;
                for (int j = 0; j < headDim; j++)
                    stateSpan[rowOff + j] = d * stateSpan[rowOff + j] + ki * error[j];
            }

            // 4. o = S_new^T * q * scale
            for (int j = 0; j < headDim; j++)
            {
                float sum = 0;
                for (int i = 0; i < headDim; i++)
                    sum += stateSpan[stateOff + i * headDim + j] * qSpan[baseOff + i];
                outSpan[baseOff + j] = sum * scale;
            }

            // Per-head RMSNorm
            float sumSq = 0;
            for (int i = 0; i < headDim; i++)
                sumSq += outSpan[baseOff + i] * outSpan[baseOff + i];
            float rms = MathF.Sqrt(sumSq / headDim + normEps);
            float invRms = 1.0f / rms;
            for (int i = 0; i < headDim; i++)
                outSpan[baseOff + i] = outSpan[baseOff + i] * invRms * normW[i];
        }
    }

    /// <inheritdoc />
    public void SiLUGate(ITensor output, ITensor data, ITensor gate)
    {
        var outSpan = output.AsFloatSpan();
        var dataSpan = data.AsFloatSpan();
        var gateSpan = gate.AsFloatSpan();
        for (int i = 0; i < dataSpan.Length; i++)
        {
            float g = gateSpan[i];
            outSpan[i] = dataSpan[i] * g / (1.0f + MathF.Exp(-g));
        }
    }

    /// <inheritdoc />
    public void SplitQKV(ITensor q, ITensor k, ITensor v, ITensor qkv, int innerSize)
    {
        var src = qkv.AsFloatSpan();
        src.Slice(0, innerSize).CopyTo(q.AsFloatSpan());
        src.Slice(innerSize, innerSize).CopyTo(k.AsFloatSpan());
        src.Slice(innerSize * 2, innerSize).CopyTo(v.AsFloatSpan());
    }

    /// <inheritdoc />
    public void ZeroTensor(ITensor tensor)
    {
        tensor.AsFloatSpan().Clear();
    }

    /// <inheritdoc />
    public void CopyTensorBytes(ITensor dst, ITensor src, long byteCount)
    {
        var srcData = ((CpuTensor)src).RawData;
        var dstData = ((CpuTensor)dst).RawDataMut;
        srcData.Slice(0, (int)byteCount).CopyTo(dstData.Slice(0, (int)byteCount));
    }

    /// <inheritdoc />
    public ITensor CreateHostTensor(string name, GgmlType type, ReadOnlySpan<long> dimensions) =>
        CreateTensor(name, type, dimensions); // CPU tensors are already in host memory

    /// <inheritdoc />
    public void FillTensor(ITensor tensor, float value) =>
        tensor.AsFloatSpan().Fill(value);

    /// <inheritdoc />
    public void SquaredReLU(ITensor data)
    {
        var span = data.AsFloatSpan();
        Cpu.Relu2.ApplyInPlace(span);
    }

    /// <inheritdoc />
    public void Dispose()
    {
        // No unmanaged resources to clean up.
    }
}
