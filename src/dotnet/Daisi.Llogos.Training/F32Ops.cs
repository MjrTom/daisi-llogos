using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace Daisi.Llogos.Training;

/// <summary>
/// Pure F32 math operations for training with backward passes.
/// SIMD-optimized for AVX2/AVX-512 where it matters.
/// All operations work on raw float spans — no tensor abstraction overhead.
/// </summary>
public static class F32Ops
{
    // ── Matrix Multiply ──────────────────────────────────────────────────────

    /// <summary>
    /// C = A × B. A is [M×K], B is [K×N], C is [M×N]. Row-major.
    /// Parallelized across output rows when large enough.
    /// </summary>
    public static void MatMul(Span<float> c, ReadOnlySpan<float> a, ReadOnlySpan<float> b,
        int M, int K, int N)
    {
        c.Clear();

        // Small matrices: single-threaded tiled
        if ((long)M * N < 4096)
        {
            MatMulTiledCore(c, a, b, M, K, N);
            return;
        }

        // Large matrices: parallelize across rows using pinned arrays
        var aArr = a.ToArray();
        var bArr = b.ToArray();
        var cArr = new float[M * N];
        Parallel.For(0, M, i =>
        {
            int aOff = i * K;
            int cOff = i * N;
            for (int j = 0; j < N; j++)
            {
                float sum = 0;
                for (int k = 0; k < K; k++)
                    sum += aArr[aOff + k] * bArr[k * N + j];
                cArr[cOff + j] = sum;
            }
        });
        cArr.AsSpan().CopyTo(c);
    }

    private static void MatMulTiledCore(Span<float> c, ReadOnlySpan<float> a, ReadOnlySpan<float> b,
        int M, int K, int N)
    {
        const int TILE = 64;
        for (int i0 = 0; i0 < M; i0 += TILE)
        {
            int iEnd = Math.Min(i0 + TILE, M);
            for (int j0 = 0; j0 < N; j0 += TILE)
            {
                int jEnd = Math.Min(j0 + TILE, N);
                for (int k0 = 0; k0 < K; k0 += TILE)
                {
                    int kEnd = Math.Min(k0 + TILE, K);
                    for (int i = i0; i < iEnd; i++)
                    {
                        int aRowOff = i * K;
                        int cRowOff = i * N;
                        for (int k = k0; k < kEnd; k++)
                        {
                            float aik = a[aRowOff + k];
                            int bRowOff = k * N;
                            int j = j0;

                            if (Avx2.IsSupported)
                            {
                                var va = Vector256.Create(aik);
                                for (; j + 7 < jEnd; j += 8)
                                {
                                    ref float cRef = ref Unsafe.Add(ref MemoryMarshal.GetReference(c), cRowOff + j);
                                    ref readonly float bRef = ref Unsafe.Add(ref MemoryMarshal.GetReference(b), bRowOff + j);
                                    var vc = Vector256.LoadUnsafe(ref cRef);
                                    var vb = Vector256.LoadUnsafe(ref Unsafe.AsRef(in bRef));
                                    vc = Avx2.Add(vc.AsSingle(), Avx2.Multiply(va.AsSingle(), vb.AsSingle()).AsSingle()).AsSingle();
                                    vc.StoreUnsafe(ref cRef);
                                }
                            }

                            for (; j < jEnd; j++)
                                c[cRowOff + j] += aik * b[bRowOff + j];
                        }
                    }
                }
            }
        }
    }

    /// <summary>
    /// C = A × B^T. A is [M×K], B is [N×K], C is [M×N].
    /// Parallelized across output rows when M*N is large enough to justify threading.
    /// </summary>
    public static void MatMulTransB(Span<float> c, ReadOnlySpan<float> a, ReadOnlySpan<float> b,
        int M, int K, int N)
    {
        // Small matrices: single-threaded to avoid thread pool overhead
        if ((long)M * N < 4096)
        {
            for (int i = 0; i < M; i++)
            {
                int aOff = i * K;
                int cOff = i * N;
                for (int j = 0; j < N; j++)
                {
                    int bOff = j * K;
                    float sum = DotProduct(a.Slice(aOff, K), b.Slice(bOff, K));
                    c[cOff + j] = sum;
                }
            }
            return;
        }

        // Large matrices: parallelize across rows
        var aArr = a.ToArray();
        var bArr = b.ToArray();
        var cArr = new float[M * N];
        Parallel.For(0, M, i =>
        {
            int aOff = i * K;
            int cOff = i * N;
            for (int j = 0; j < N; j++)
            {
                int bOff = j * K;
                float sum = DotProductArr(aArr, aOff, bArr, bOff, K);
                cArr[cOff + j] = sum;
            }
        });
        cArr.AsSpan().CopyTo(c);
    }

    /// <summary>
    /// C = A^T × B. A is [K×M] (transposed to [M×K]), B is [K×N], C is [M×N].
    /// </summary>
    public static void MatMulTransA(Span<float> c, ReadOnlySpan<float> a, ReadOnlySpan<float> b,
        int M, int K, int N)
    {
        c.Clear();
        for (int k = 0; k < K; k++)
        {
            int bOff = k * N;
            for (int i = 0; i < M; i++)
            {
                float aVal = a[k * M + i]; // A[k,i] = A^T[i,k]
                int cOff = i * N;
                for (int j = 0; j < N; j++)
                    c[cOff + j] += aVal * b[bOff + j];
            }
        }
    }

    // ── Backward: MatMul ────────────────────────────────────────────────────

    /// <summary>
    /// Backward for C = A × B.
    /// dA += dC × B^T, dB += A^T × dC.
    /// </summary>
    public static void MatMulBackward(
        Span<float> dA, Span<float> dB,
        ReadOnlySpan<float> dC, ReadOnlySpan<float> a, ReadOnlySpan<float> b,
        int M, int K, int N)
    {
        // dA = dC × B^T  (dC is [M×N], B is [K×N], dA is [M×K])
        if (dA.Length > 0)
            MatMulTransB(dA, dC, b, M, N, K);

        // dB = A^T × dC  (A is [M×K], dC is [M×N], dB is [K×N])
        if (dB.Length > 0)
            MatMulTransA(dB, a, dC, K, M, N);
    }

    // ── RMS Normalization ───────────────────────────────────────────────────

    /// <summary>
    /// output[i] = (input[i] / rms) * weight[i], where rms = sqrt(mean(input^2) + eps).
    /// </summary>
    public static void RmsNorm(Span<float> output, ReadOnlySpan<float> input,
        ReadOnlySpan<float> weight, float eps, int dim)
    {
        float sumSq = 0;
        for (int i = 0; i < dim; i++)
            sumSq += input[i] * input[i];
        float invRms = 1.0f / MathF.Sqrt(sumSq / dim + eps);
        for (int i = 0; i < dim; i++)
            output[i] = input[i] * invRms * weight[i];
    }

    /// <summary>
    /// Backward for RMS normalization.
    /// Accumulates into dInput and dWeight.
    /// </summary>
    public static void RmsNormBackward(
        Span<float> dInput, Span<float> dWeight,
        ReadOnlySpan<float> dOutput, ReadOnlySpan<float> input,
        ReadOnlySpan<float> weight, float eps, int dim)
    {
        // Forward: y_i = x_i * w_i / rms, where rms = sqrt(mean(x^2) + eps)
        float sumSq = 0;
        for (int i = 0; i < dim; i++)
            sumSq += input[i] * input[i];
        float variance = sumSq / dim;
        float rms = MathF.Sqrt(variance + eps);
        float invRms = 1.0f / rms;

        // dWeight: dL/dw_i = dL/dy_i * x_i / rms
        for (int i = 0; i < dim; i++)
            dWeight[i] += dOutput[i] * input[i] * invRms;

        // dInput: dL/dx_i = (dL/dy_i * w_i / rms) - x_i * (sum(dL/dy * w * x) / (dim * rms^3))
        float dotDywx = 0;
        for (int i = 0; i < dim; i++)
            dotDywx += dOutput[i] * weight[i] * input[i];
        float coeff = dotDywx / (dim * rms * rms * rms);

        for (int i = 0; i < dim; i++)
            dInput[i] += dOutput[i] * weight[i] * invRms - input[i] * coeff;
    }

    // ── Softmax ─────────────────────────────────────────────────────────────

    /// <summary>
    /// Stable softmax over a 1D span: output[i] = exp(input[i] - max) / sum.
    /// </summary>
    public static void Softmax(Span<float> output, ReadOnlySpan<float> input, int len)
    {
        float max = float.MinValue;
        for (int i = 0; i < len; i++)
            if (input[i] > max) max = input[i];

        float sum = 0;
        for (int i = 0; i < len; i++)
        {
            output[i] = MathF.Exp(input[i] - max);
            sum += output[i];
        }

        float invSum = 1.0f / sum;
        for (int i = 0; i < len; i++)
            output[i] *= invSum;
    }

    /// <summary>
    /// Backward for softmax. dInput[i] = output[i] * (dOutput[i] - sum(dOutput * output)).
    /// </summary>
    public static void SoftmaxBackward(Span<float> dInput, ReadOnlySpan<float> dOutput,
        ReadOnlySpan<float> output, int len)
    {
        float dot = 0;
        for (int i = 0; i < len; i++)
            dot += dOutput[i] * output[i];
        for (int i = 0; i < len; i++)
            dInput[i] += output[i] * (dOutput[i] - dot);
    }

    // ── SiLU ────────────────────────────────────────────────────────────────

    /// <summary>
    /// SiLU: output[i] = input[i] * sigmoid(input[i]).
    /// </summary>
    public static void SiLU(Span<float> output, ReadOnlySpan<float> input, int len)
    {
        for (int i = 0; i < len; i++)
        {
            float x = input[i];
            float sig = 1.0f / (1.0f + MathF.Exp(-x));
            output[i] = x * sig;
        }
    }

    /// <summary>
    /// Backward for SiLU. dInput[i] += dOutput[i] * (sig(x) + x * sig(x) * (1 - sig(x))).
    /// </summary>
    public static void SiLUBackward(Span<float> dInput, ReadOnlySpan<float> dOutput,
        ReadOnlySpan<float> input, int len)
    {
        for (int i = 0; i < len; i++)
        {
            float x = input[i];
            float sig = 1.0f / (1.0f + MathF.Exp(-x));
            dInput[i] += dOutput[i] * (sig + x * sig * (1.0f - sig));
        }
    }

    // ── SwiGLU ──────────────────────────────────────────────────────────────

    /// <summary>
    /// SwiGLU: output[i] = silu(gate[i]) * up[i].
    /// </summary>
    public static void SwiGLU(Span<float> output, ReadOnlySpan<float> gate,
        ReadOnlySpan<float> up, int len)
    {
        for (int i = 0; i < len; i++)
        {
            float g = gate[i];
            float sig = 1.0f / (1.0f + MathF.Exp(-g));
            output[i] = g * sig * up[i];
        }
    }

    /// <summary>
    /// Backward for SwiGLU.
    /// dGate[i] += dOut[i] * up[i] * (sig + gate * sig * (1-sig))
    /// dUp[i] += dOut[i] * silu(gate[i])
    /// </summary>
    public static void SwiGLUBackward(Span<float> dGate, Span<float> dUp,
        ReadOnlySpan<float> dOutput, ReadOnlySpan<float> gate,
        ReadOnlySpan<float> up, int len)
    {
        for (int i = 0; i < len; i++)
        {
            float g = gate[i];
            float sig = 1.0f / (1.0f + MathF.Exp(-g));
            float siluG = g * sig;
            dGate[i] += dOutput[i] * up[i] * (sig + g * sig * (1.0f - sig));
            dUp[i] += dOutput[i] * siluG;
        }
    }

    // ── RoPE ────────────────────────────────────────────────────────────────

    /// <summary>
    /// Apply rotary position embeddings in-place.
    /// data is [numHeads × headDim], rotates pairs (2i, 2i+1) for i &lt; ropeDim/2.
    /// </summary>
    public static void RoPE(Span<float> data, int numHeads, int headDim,
        int ropeDim, int position, float theta)
    {
        if (ropeDim == 0) ropeDim = headDim;
        for (int h = 0; h < numHeads; h++)
        {
            int off = h * headDim;
            for (int i = 0; i < ropeDim / 2; i++)
            {
                float freq = 1.0f / MathF.Pow(theta, (float)(2 * i) / ropeDim);
                float angle = position * freq;
                float cos = MathF.Cos(angle);
                float sin = MathF.Sin(angle);

                float x0 = data[off + 2 * i];
                float x1 = data[off + 2 * i + 1];
                data[off + 2 * i] = x0 * cos - x1 * sin;
                data[off + 2 * i + 1] = x0 * sin + x1 * cos;
            }
        }
    }

    /// <summary>
    /// Backward for RoPE: inverse rotation (negate sin).
    /// </summary>
    public static void RoPEBackward(Span<float> dData, int numHeads, int headDim,
        int ropeDim, int position, float theta)
    {
        if (ropeDim == 0) ropeDim = headDim;
        for (int h = 0; h < numHeads; h++)
        {
            int off = h * headDim;
            for (int i = 0; i < ropeDim / 2; i++)
            {
                float freq = 1.0f / MathF.Pow(theta, (float)(2 * i) / ropeDim);
                float angle = position * freq;
                float cos = MathF.Cos(angle);
                float sin = MathF.Sin(angle);

                float d0 = dData[off + 2 * i];
                float d1 = dData[off + 2 * i + 1];
                // Inverse rotation: negate sin
                dData[off + 2 * i] = d0 * cos + d1 * sin;
                dData[off + 2 * i + 1] = -d0 * sin + d1 * cos;
            }
        }
    }

    // ── Batched RoPE (for training sequences) ───────────────────────────────

    /// <summary>
    /// Apply RoPE to a batch of tokens: data is [T × numHeads × headDim].
    /// </summary>
    public static void BatchedRoPE(Span<float> data, int T, int numHeads, int headDim,
        int ropeDim, int startPos, float theta)
    {
        int stride = numHeads * headDim;
        for (int t = 0; t < T; t++)
            RoPE(data.Slice(t * stride, stride), numHeads, headDim, ropeDim, startPos + t, theta);
    }

    /// <summary>
    /// Backward for batched RoPE.
    /// </summary>
    public static void BatchedRoPEBackward(Span<float> dData, int T, int numHeads, int headDim,
        int ropeDim, int startPos, float theta)
    {
        int stride = numHeads * headDim;
        for (int t = 0; t < T; t++)
            RoPEBackward(dData.Slice(t * stride, stride), numHeads, headDim, ropeDim, startPos + t, theta);
    }

    // ── Cross-Entropy Loss ──────────────────────────────────────────────────

    /// <summary>
    /// Cross-entropy loss for language modeling.
    /// logits: [T × vocabSize], targets: T target token IDs.
    /// Returns average loss. Writes gradient into dLogits: softmax(logits) - one_hot(target).
    /// </summary>
    public static float CrossEntropyLoss(ReadOnlySpan<float> logits, ReadOnlySpan<int> targets,
        Span<float> dLogits, int T, int V)
    {
        float totalLoss = 0;
        var softmaxBuf = new float[V];

        for (int t = 0; t < T; t++)
        {
            var logitSlice = logits.Slice(t * V, V);
            var dSlice = dLogits.Slice(t * V, V);

            // Softmax
            Softmax(softmaxBuf, logitSlice, V);

            // NLL loss: -log(softmax[target])
            int target = targets[t];
            float prob = Math.Max(softmaxBuf[target], 1e-10f);
            totalLoss -= MathF.Log(prob);

            // Gradient: softmax - one_hot
            for (int v = 0; v < V; v++)
                dSlice[v] = softmaxBuf[v];
            dSlice[target] -= 1.0f;

            // Average over T
            float invT = 1.0f / T;
            for (int v = 0; v < V; v++)
                dSlice[v] *= invT;
        }

        return totalLoss / T;
    }

    // ── Element-wise Operations ─────────────────────────────────────────────

    /// <summary>output[i] = a[i] + b[i]</summary>
    public static void Add(Span<float> output, ReadOnlySpan<float> a, ReadOnlySpan<float> b, int len)
    {
        int i = 0;
        if (Avx2.IsSupported)
        {
            for (; i + 7 < len; i += 8)
            {
                var va = Vector256.LoadUnsafe(ref Unsafe.AsRef(in a[i]));
                var vb = Vector256.LoadUnsafe(ref Unsafe.AsRef(in b[i]));
                Avx2.Add(va.AsSingle(), vb.AsSingle()).AsSingle().StoreUnsafe(ref output[i]);
            }
        }
        for (; i < len; i++)
            output[i] = a[i] + b[i];
    }

    /// <summary>Accumulate: dst[i] += src[i]</summary>
    public static void AddInPlace(Span<float> dst, ReadOnlySpan<float> src, int len)
    {
        for (int i = 0; i < len; i++)
            dst[i] += src[i];
    }

    /// <summary>Element-wise multiply: output[i] = a[i] * b[i]</summary>
    public static void Mul(Span<float> output, ReadOnlySpan<float> a, ReadOnlySpan<float> b, int len)
    {
        for (int i = 0; i < len; i++)
            output[i] = a[i] * b[i];
    }

    /// <summary>Scale: output[i] = input[i] * scale</summary>
    public static void Scale(Span<float> data, float scale, int len)
    {
        for (int i = 0; i < len; i++)
            data[i] *= scale;
    }

    // ── Per-Head RMS Normalization ──────────────────────────────────────────

    /// <summary>
    /// Apply RMS normalization independently to each head-sized slice.
    /// data: [numHeads × headDim], weight: [headDim] shared across heads.
    /// </summary>
    public static void PerHeadRmsNorm(Span<float> data, ReadOnlySpan<float> weight,
        int numHeads, int headDim, float eps)
    {
        for (int h = 0; h < numHeads; h++)
        {
            var slice = data.Slice(h * headDim, headDim);
            float sumSq = 0;
            for (int i = 0; i < headDim; i++)
                sumSq += slice[i] * slice[i];
            float invRms = 1.0f / MathF.Sqrt(sumSq / headDim + eps);
            for (int i = 0; i < headDim; i++)
                slice[i] = slice[i] * invRms * weight[i];
        }
    }

    /// <summary>
    /// Backward for per-head RMS norm.
    /// </summary>
    public static void PerHeadRmsNormBackward(Span<float> dData, Span<float> dWeight,
        ReadOnlySpan<float> dOutput, ReadOnlySpan<float> inputBeforeNorm,
        ReadOnlySpan<float> weight, int numHeads, int headDim, float eps)
    {
        for (int h = 0; h < numHeads; h++)
        {
            int off = h * headDim;
            float sumSq = 0;
            for (int i = 0; i < headDim; i++)
                sumSq += inputBeforeNorm[off + i] * inputBeforeNorm[off + i];
            float rms = MathF.Sqrt(sumSq / headDim + eps);
            float invRms = 1.0f / rms;

            float dotDywx = 0;
            for (int i = 0; i < headDim; i++)
                dotDywx += dOutput[off + i] * weight[i] * inputBeforeNorm[off + i];
            float coeff = dotDywx / (headDim * rms * rms * rms);

            for (int i = 0; i < headDim; i++)
            {
                dWeight[i] += dOutput[off + i] * inputBeforeNorm[off + i] * invRms;
                dData[off + i] += dOutput[off + i] * weight[i] * invRms - inputBeforeNorm[off + i] * coeff;
            }
        }
    }

    // ── De-interleave Gated Q ───────────────────────────────────────────────

    /// <summary>
    /// De-interleave: qFull [h0_attn, h0_gate, h1_attn, h1_gate, ...] → qAttn, qGate.
    /// </summary>
    public static void DeInterleaveQ(Span<float> qAttn, Span<float> qGate,
        ReadOnlySpan<float> qFull, int numHeads, int headDim)
    {
        for (int h = 0; h < numHeads; h++)
        {
            int srcOff = h * headDim * 2;
            int dstOff = h * headDim;
            qFull.Slice(srcOff, headDim).CopyTo(qAttn.Slice(dstOff, headDim));
            qFull.Slice(srcOff + headDim, headDim).CopyTo(qGate.Slice(dstOff, headDim));
        }
    }

    /// <summary>
    /// Backward: re-interleave gradients from qAttn and qGate back to qFull layout.
    /// </summary>
    public static void DeInterleaveQBackward(Span<float> dQFull,
        ReadOnlySpan<float> dQAttn, ReadOnlySpan<float> dQGate,
        int numHeads, int headDim)
    {
        for (int h = 0; h < numHeads; h++)
        {
            int dstOff = h * headDim * 2;
            int srcOff = h * headDim;
            dQAttn.Slice(srcOff, headDim).CopyTo(dQFull.Slice(dstOff, headDim));
            dQGate.Slice(srcOff, headDim).CopyTo(dQFull.Slice(dstOff + headDim, headDim));
        }
    }

    // ── Causal Multi-Head Attention (training, full sequence) ───────────────

    /// <summary>
    /// Full causal multi-head attention for training.
    /// Q: [T × numHeads × headDim], K: [T × numKvHeads × headDim], V: [T × numKvHeads × valDim]
    /// Output: [T × numHeads × valDim].
    /// Applies GQA repeat and causal mask. qGate: [T × numHeads × headDim] for sigmoid gating.
    /// </summary>
    public static void CausalGatedAttention(
        Span<float> output, ReadOnlySpan<float> qAttn, ReadOnlySpan<float> qGate,
        ReadOnlySpan<float> k, ReadOnlySpan<float> v,
        int T, int numHeads, int numKvHeads, int keyDim, int valDim, float scale,
        Span<float> savedProbs) // [numHeads × T × T] for backward
    {
        int headsPerKvGroup = numHeads / numKvHeads;

        for (int h = 0; h < numHeads; h++)
        {
            int kvH = h / headsPerKvGroup;

            for (int t = 0; t < T; t++)
            {
                // Compute attention scores for query at position t
                int qOff = t * numHeads * keyDim + h * keyDim;
                int seqLen = t + 1; // causal: attend to 0..t

                // Scores: Q[t,h] @ K[0..t,kvH]^T
                float maxScore = float.MinValue;
                for (int s = 0; s < seqLen; s++)
                {
                    int kOff = s * numKvHeads * keyDim + kvH * keyDim;
                    float score = 0;
                    for (int d = 0; d < keyDim; d++)
                        score += qAttn[qOff + d] * k[kOff + d];
                    score *= scale;
                    savedProbs[h * T * T + t * T + s] = score; // temporarily store scores
                    if (score > maxScore) maxScore = score;
                }

                // Softmax
                float sum = 0;
                for (int s = 0; s < seqLen; s++)
                {
                    float p = MathF.Exp(savedProbs[h * T * T + t * T + s] - maxScore);
                    savedProbs[h * T * T + t * T + s] = p;
                    sum += p;
                }
                float invSum = 1.0f / sum;
                for (int s = 0; s < seqLen; s++)
                    savedProbs[h * T * T + t * T + s] *= invSum;
                // Zero out non-causal entries
                for (int s = seqLen; s < T; s++)
                    savedProbs[h * T * T + t * T + s] = 0;

                // Weighted sum of V
                int outOff = t * numHeads * valDim + h * valDim;
                for (int d = 0; d < valDim; d++)
                    output[outOff + d] = 0;
                for (int s = 0; s < seqLen; s++)
                {
                    float p = savedProbs[h * T * T + t * T + s];
                    int vOff = s * numKvHeads * valDim + kvH * valDim;
                    for (int d = 0; d < valDim; d++)
                        output[outOff + d] += p * v[vOff + d];
                }

                // Sigmoid gating
                int gateOff = t * numHeads * keyDim + h * keyDim;
                for (int d = 0; d < valDim; d++)
                {
                    // Use the gate values (one per head dim, but applied to valDim output)
                    // For models where keyDim == valDim, this is straightforward
                    float gateVal = d < keyDim ? qGate[gateOff + d] : 0.0f;
                    float sig = 1.0f / (1.0f + MathF.Exp(-gateVal));
                    output[outOff + d] *= sig;
                }
            }
        }
    }

    /// <summary>
    /// Backward for causal gated attention.
    /// Computes dQ, dK, dV, dQGate from dOutput and saved probabilities.
    /// </summary>
    public static void CausalGatedAttentionBackward(
        Span<float> dQAttn, Span<float> dQGate, Span<float> dK, Span<float> dV,
        ReadOnlySpan<float> dOutput, ReadOnlySpan<float> qAttn, ReadOnlySpan<float> qGate,
        ReadOnlySpan<float> kData, ReadOnlySpan<float> vData,
        ReadOnlySpan<float> savedProbs, ReadOnlySpan<float> attnOutput,
        int T, int numHeads, int numKvHeads, int keyDim, int valDim, float scale)
    {
        int headsPerKvGroup = numHeads / numKvHeads;

        for (int h = 0; h < numHeads; h++)
        {
            int kvH = h / headsPerKvGroup;

            for (int t = 0; t < T; t++)
            {
                int outOff = t * numHeads * valDim + h * valDim;
                int gateOff = t * numHeads * keyDim + h * keyDim;

                // Backward through sigmoid gate
                var dAfterGate = new float[valDim];
                for (int d = 0; d < valDim; d++)
                {
                    float gateVal = d < keyDim ? qGate[gateOff + d] : 0.0f;
                    float sig = 1.0f / (1.0f + MathF.Exp(-gateVal));

                    // output = beforeGate * sig
                    // dBeforeGate = dOut * sig
                    // dGate = dOut * beforeGate * sig * (1-sig)
                    dAfterGate[d] = dOutput[outOff + d] * sig;

                    // Compute beforeGate = attnOutput / sig (but use saved probs to recompute)
                    float beforeGate = sig > 1e-10f ? attnOutput[outOff + d] / sig : 0;
                    if (d < keyDim)
                        dQGate[gateOff + d] += dOutput[outOff + d] * beforeGate * sig * (1.0f - sig);
                }

                int seqLen = t + 1;

                // Backward through weighted V sum: dV and dProbs
                var dProbs = new float[T];
                for (int s = 0; s < seqLen; s++)
                {
                    int vOff = s * numKvHeads * valDim + kvH * valDim;
                    float p = savedProbs[h * T * T + t * T + s];
                    for (int d = 0; d < valDim; d++)
                    {
                        dV[vOff + d] += p * dAfterGate[d];
                        dProbs[s] += vData[vOff + d] * dAfterGate[d];
                    }
                }

                // Backward through softmax
                float dot = 0;
                for (int s = 0; s < seqLen; s++)
                    dot += dProbs[s] * savedProbs[h * T * T + t * T + s];
                var dScores = new float[T];
                for (int s = 0; s < seqLen; s++)
                    dScores[s] = savedProbs[h * T * T + t * T + s] * (dProbs[s] - dot) * scale;

                // Backward through Q @ K^T
                int qOff = t * numHeads * keyDim + h * keyDim;
                for (int s = 0; s < seqLen; s++)
                {
                    int kOff = s * numKvHeads * keyDim + kvH * keyDim;
                    for (int d = 0; d < keyDim; d++)
                    {
                        dQAttn[qOff + d] += dScores[s] * kData[kOff + d];
                        dK[kOff + d] += dScores[s] * qAttn[qOff + d];
                    }
                }
            }
        }
    }

    // ── Embedding Lookup ────────────────────────────────────────────────────

    /// <summary>
    /// Extract row from embedding table: output = table[tokenId * dim .. (tokenId+1)*dim].
    /// </summary>
    public static void EmbeddingLookup(Span<float> output, ReadOnlySpan<float> table,
        int tokenId, int dim)
    {
        table.Slice(tokenId * dim, dim).CopyTo(output);
    }

    /// <summary>
    /// Backward: accumulate gradient into embedding table row.
    /// </summary>
    public static void EmbeddingBackward(Span<float> dTable, ReadOnlySpan<float> dOutput,
        int tokenId, int dim)
    {
        int off = tokenId * dim;
        for (int i = 0; i < dim; i++)
            dTable[off + i] += dOutput[i];
    }

    // ── Utilities ───────────────────────────────────────────────────────────

    /// <summary>Array-based dot product for use in Parallel.For lambdas.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static float DotProductArr(float[] a, int aOff, float[] b, int bOff, int len)
    {
        return DotProduct(a.AsSpan(aOff, len), b.AsSpan(bOff, len));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float DotProduct(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        float sum = 0;
        int i = 0;
        int len = Math.Min(a.Length, b.Length);

        if (Avx2.IsSupported)
        {
            var vsum = Vector256<float>.Zero;
            for (; i + 7 < len; i += 8)
            {
                var va = Vector256.LoadUnsafe(ref Unsafe.AsRef(in a[i]));
                var vb = Vector256.LoadUnsafe(ref Unsafe.AsRef(in b[i]));
                vsum = Avx2.Add(vsum.AsSingle(), Avx2.Multiply(va.AsSingle(), vb.AsSingle()).AsSingle()).AsSingle();
            }
            // Horizontal sum
            var upper = Avx2.ExtractVector128(vsum.AsSingle(), 1);
            var lower = vsum.GetLower().AsSingle();
            var sum128 = Sse.Add(upper, lower);
            sum128 = Sse3.IsSupported
                ? Sse3.HorizontalAdd(sum128, sum128)
                : Sse.Add(sum128, Sse.MoveHighToLow(sum128, sum128));
            sum128 = Sse.AddScalar(sum128, Sse.Shuffle(sum128, sum128, 1));
            sum = sum128.ToScalar();
        }

        for (; i < len; i++)
            sum += a[i] * b[i];
        return sum;
    }

    /// <summary>
    /// Dequantize an ITensor to a float array.
    /// </summary>
    public static float[] Dequantize(ITensor tensor)
    {
        var buf = new float[tensor.ElementCount];
        tensor.DequantizeTo(buf);
        return buf;
    }
}
