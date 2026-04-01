// daisi-llogos CUDA kernels: backward operations for training
// Compiled to .cubin and embedded as assembly resources.

extern "C" {

// ── SiLU Backward ───────────────────────────────────────────────────────────
// dInput[i] += dOutput[i] * (sig(x) + x * sig(x) * (1 - sig(x)))
__global__ void silu_backward(float* dInput, const float* dOutput, const float* input, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        float x = input[idx];
        float sig = 1.0f / (1.0f + expf(-x));
        dInput[idx] += dOutput[idx] * (sig + x * sig * (1.0f - sig));
    }
}

// ── SwiGLU Backward ────────────────────────────────────────────────────────
// Forward: output = silu(gate) * up
// dGate[i] += dOutput[i] * up[i] * (sig + gate * sig * (1-sig))
// dUp[i] += dOutput[i] * silu(gate[i])
__global__ void swiglu_backward(float* dGate, float* dUp, const float* dOutput,
                                 const float* gate, const float* up, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        float g = gate[idx];
        float sig = 1.0f / (1.0f + expf(-g));
        float siluG = g * sig;
        dGate[idx] += dOutput[idx] * up[idx] * (sig + g * sig * (1.0f - sig));
        dUp[idx] += dOutput[idx] * siluG;
    }
}

// ── RMSNorm Backward (single row) ──────────────────────────────────────────
// Forward: y_i = x_i * w_i / rms, rms = sqrt(mean(x^2) + eps)
// Backward: dInput, given dOutput, input, weight, eps
__global__ void rms_norm_backward(float* dInput, const float* dOutput, const float* input,
                                   const float* weight, float eps, int dim)
{
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int stride = blockDim.x;

    // Pass 1: sum of squares
    float sumSq = 0.0f;
    for (int i = tid; i < dim; i += stride)
        sumSq += input[i] * input[i];
    sdata[tid] = sumSq;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float variance = sdata[0] / (float)dim;
    float rms = sqrtf(variance + eps);
    float invRms = 1.0f / rms;
    __syncthreads();

    // Pass 2: dot(dOutput * weight * input)
    float dotDywx = 0.0f;
    for (int i = tid; i < dim; i += stride)
        dotDywx += dOutput[i] * weight[i] * input[i];
    sdata[tid] = dotDywx;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float coeff = sdata[0] / ((float)dim * rms * rms * rms);
    __syncthreads();

    // Pass 3: compute dInput
    for (int i = tid; i < dim; i += stride)
        dInput[i] += dOutput[i] * weight[i] * invRms - input[i] * coeff;
}

// ── Batched RMSNorm Backward ───────────────────────────────────────────────
// One block per row (M rows)
__global__ void batched_rms_norm_backward(float* dInput, const float* dOutput,
                                           const float* input, const float* weight,
                                           float eps, int dim, int M)
{
    extern __shared__ float sdata[];

    int row = blockIdx.x;
    if (row >= M) return;

    int tid = threadIdx.x;
    int stride = blockDim.x;
    int off = row * dim;

    // Pass 1: sum of squares
    float sumSq = 0.0f;
    for (int i = tid; i < dim; i += stride)
        sumSq += input[off + i] * input[off + i];
    sdata[tid] = sumSq;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float rms = sqrtf(sdata[0] / (float)dim + eps);
    float invRms = 1.0f / rms;
    __syncthreads();

    // Pass 2: dot(dOutput * weight * input)
    float dotDywx = 0.0f;
    for (int i = tid; i < dim; i += stride)
        dotDywx += dOutput[off + i] * weight[i] * input[off + i];
    sdata[tid] = dotDywx;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float coeff = sdata[0] / ((float)dim * rms * rms * rms);
    __syncthreads();

    // Pass 3: compute dInput
    for (int i = tid; i < dim; i += stride)
        dInput[off + i] += dOutput[off + i] * weight[i] * invRms - input[off + i] * coeff;
}

// ── RoPE Backward ──────────────────────────────────────────────────────────
// Inverse rotation: negate sin. Applied per-token for batched data.
// data layout: [T × numHeads × headDim]
__global__ void batched_rope_backward(float* dData, int T, int numHeads, int headDim,
                                       int ropeDim, int startPos, float theta)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalPairs = T * numHeads * (ropeDim / 2);
    if (idx >= totalPairs) return;

    int pairsPerToken = numHeads * (ropeDim / 2);
    int t = idx / pairsPerToken;
    int rem = idx % pairsPerToken;
    int h = rem / (ropeDim / 2);
    int i = rem % (ropeDim / 2);

    int position = startPos + t;
    float freq = 1.0f / powf(theta, (float)(2 * i) / (float)ropeDim);
    float angle = (float)position * freq;
    float cosA = cosf(angle);
    float sinA = sinf(angle);

    int off = t * numHeads * headDim + h * headDim;
    float d0 = dData[off + 2 * i];
    float d1 = dData[off + 2 * i + 1];
    // Inverse rotation: negate sin
    dData[off + 2 * i]     = d0 * cosA + d1 * sinA;
    dData[off + 2 * i + 1] = -d0 * sinA + d1 * cosA;
}

// ── Cross-Entropy Loss + Gradient ──────────────────────────────────────────
// Per-token: softmax, NLL loss, gradient = softmax - one_hot
// One block per token. Downloads only scalar loss.
// target=-1 means ignore this token (completion-only training).
__global__ void cross_entropy_loss(float* dLogits, float* lossOut,
                                    const float* logits, const int* targets,
                                    int T, int V)
{
    extern __shared__ float sdata[];

    int t = blockIdx.x;
    if (t >= T) return;

    int tid = threadIdx.x;
    int stride = blockDim.x;
    const float* logitRow = logits + t * V;
    float* dRow = dLogits + t * V;
    int target = targets[t];

    // Skip masked tokens (target=-1): zero gradient, no loss contribution
    if (target < 0)
    {
        for (int v = tid; v < V; v += stride)
            dRow[v] = 0.0f;
        return;
    }

    // Count non-masked tokens for proper loss averaging
    int validTokens = 0;
    for (int tt = 0; tt < T; tt++)
        if (targets[tt] >= 0) validTokens++;
    float invT = (validTokens > 0) ? 1.0f / (float)validTokens : 0.0f;

    // Pass 1: find max
    float localMax = -1e30f;
    for (int v = tid; v < V; v += stride)
        localMax = fmaxf(localMax, logitRow[v]);
    sdata[tid] = localMax;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    float maxVal = sdata[0];
    __syncthreads();

    // Pass 2: exp + sum
    float localSum = 0.0f;
    for (int v = tid; v < V; v += stride)
    {
        float e = expf(logitRow[v] - maxVal);
        dRow[v] = e;
        localSum += e;
    }
    sdata[tid] = localSum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float sumExp = sdata[0];
    __syncthreads();

    // Pass 3: normalize to softmax, compute gradient, accumulate loss
    float invSum = 1.0f / sumExp;
    float tokenLoss = 0.0f;
    for (int v = tid; v < V; v += stride)
    {
        float prob = dRow[v] * invSum;
        dRow[v] = (prob - (v == target ? 1.0f : 0.0f)) * invT;
        if (v == target)
            tokenLoss = -logf(fmaxf(prob, 1e-10f));
    }

    // Reduce loss within block
    sdata[tid] = tokenLoss;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0)
        atomicAdd(lossOut, sdata[0] / (float)T);
}

// ── Causal Gated Attention Backward ────────────────────────────────────────
// One block per (head, query_position) pair.
// Grid: (numHeads * T). Block: min(T, 256) threads.
__global__ void causal_gated_attention_backward(
    float* dQAttn, float* dQGate, float* dK, float* dV,
    const float* dOutput, const float* qAttn, const float* qGate,
    const float* kData, const float* vData,
    const float* savedProbs, const float* attnOutput,
    int T, int numHeads, int numKvHeads, int keyDim, int valDim, float scale)
{
    int blockId = blockIdx.x;
    int h = blockId / T;
    int t = blockId % T;
    if (h >= numHeads) return;

    int tid = threadIdx.x;
    int kvH = h / (numHeads / numKvHeads);
    int seqLen = t + 1;

    int outOff = t * numHeads * valDim + h * valDim;
    int gateOff = t * numHeads * keyDim + h * keyDim;
    int qOff = t * numHeads * keyDim + h * keyDim;

    // Backward through sigmoid gate
    // For each output dim d: dAfterGate[d] = dOutput[d] * sig(gate[d])
    //                        dGate[d] += dOutput[d] * beforeGate * sig * (1-sig)
    // beforeGate = attnOutput / sig (recover from saved)
    for (int d = tid; d < valDim; d += blockDim.x)
    {
        float gateVal = (d < keyDim) ? qGate[gateOff + d] : 0.0f;
        float sig = 1.0f / (1.0f + expf(-gateVal));
        float beforeGate = (sig > 1e-10f) ? attnOutput[outOff + d] / sig : 0.0f;

        float dAfterGate = dOutput[outOff + d] * sig;

        if (d < keyDim)
            atomicAdd(&dQGate[gateOff + d], dOutput[outOff + d] * beforeGate * sig * (1.0f - sig));

        // Backward through weighted V sum: dV and dProbs
        for (int s = 0; s < seqLen; s++)
        {
            int vOff = s * numKvHeads * valDim + kvH * valDim;
            float p = savedProbs[h * T * T + t * T + s];
            atomicAdd(&dV[vOff + d], p * dAfterGate);
        }
    }

    __syncthreads();

    // Backward through softmax: need dProbs first
    // dProbs[s] = sum_d(v[s,d] * dAfterGate[d])
    for (int s = tid; s < seqLen; s += blockDim.x)
    {
        float dProb = 0.0f;
        for (int d = 0; d < valDim; d++)
        {
            float gateVal = (d < keyDim) ? qGate[gateOff + d] : 0.0f;
            float sig = 1.0f / (1.0f + expf(-gateVal));
            float dAfterGate = dOutput[outOff + d] * sig;

            int vOff = s * numKvHeads * valDim + kvH * valDim;
            dProb += vData[vOff + d] * dAfterGate;
        }

        // Softmax backward: dScore = prob * (dProb - dot) * scale
        // Need dot = sum_s'(dProbs[s'] * probs[s'])
        // Compute dot across all s for this (h,t) — use atomics or shared mem
        // For simplicity: compute locally per-s, then fix up
        float prob = savedProbs[h * T * T + t * T + s];

        // Store dProb temporarily for second pass
        // We'll compute the dot product in a separate loop
        // For now use a simpler approach: compute dScores directly
        // dScore[s] = prob * dProb (will subtract dot * prob later)

        // Backward through Q @ K^T: dQ += dScore * K, dK += dScore * Q
        // First compute full dot for softmax backward
        float dot = 0.0f;
        for (int s2 = 0; s2 < seqLen; s2++)
        {
            float p2 = savedProbs[h * T * T + t * T + s2];
            float dp2 = 0.0f;
            for (int d = 0; d < valDim; d++)
            {
                float gVal = (d < keyDim) ? qGate[gateOff + d] : 0.0f;
                float sig2 = 1.0f / (1.0f + expf(-gVal));
                float dag = dOutput[outOff + d] * sig2;
                int vOff2 = s2 * numKvHeads * valDim + kvH * valDim;
                dp2 += vData[vOff2 + d] * dag;
            }
            dot += dp2 * p2;
        }

        float dScore = prob * (dProb - dot) * scale;

        // Accumulate dQ and dK
        int kOff = s * numKvHeads * keyDim + kvH * keyDim;
        for (int d = 0; d < keyDim; d++)
        {
            atomicAdd(&dQAttn[qOff + d], dScore * kData[kOff + d]);
            atomicAdd(&dK[kOff + d], dScore * qAttn[qOff + d]);
        }
    }
}

// ── Element-wise product backward ──────────────────────────────────────────
// Forward: c[i] = a[i] * b[i]
// Backward: dA[i] += dC[i] * b[i], dB[i] += dC[i] * a[i]
__global__ void element_mul_backward(float* dA, float* dB, const float* dC,
                                      const float* a, const float* b, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        dA[idx] += dC[idx] * b[idx];
        dB[idx] += dC[idx] * a[idx];
    }
}

// ── Add in-place ────────────────────────────────────────────────────────────
// dst[i] += src[i]
__global__ void add_inplace(float* dst, const float* src, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        dst[idx] += src[idx];
}

// ── Scale ───────────────────────────────────────────────────────────────────
// data[i] *= scale
__global__ void scale_inplace(float* data, float s, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        data[idx] *= s;
}

// ── AdamW Step ──────────────────────────────────────────────────────────────
// In-place parameter update with decoupled weight decay
__global__ void adamw_step(float* param, const float* grad, float* m, float* v,
                            float lr, float beta1, float beta2, float eps,
                            float weightDecay, float bc1, float bc2, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        // Decoupled weight decay
        param[idx] -= lr * weightDecay * param[idx];

        // Adam update
        float g = grad[idx];
        m[idx] = beta1 * m[idx] + (1.0f - beta1) * g;
        v[idx] = beta2 * v[idx] + (1.0f - beta2) * g * g;

        float mHat = m[idx] / bc1;
        float vHat = v[idx] / bc2;

        param[idx] -= lr * mHat / (sqrtf(vHat) + eps);
    }
}

// ── Gradient Norm ───────────────────────────────────────────────────────────
// Compute sum of squares for gradient clipping
__global__ void grad_norm_sq(float* output, const float* grad, int n)
{
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int stride = blockDim.x;

    float sum = 0.0f;
    for (int i = tid; i < n; i += stride)
        sum += grad[i] * grad[i];

    sdata[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0)
        atomicAdd(output, sdata[0]);
}

// ── Training Causal Gated Attention (forward) ──────────────────────────────
// Full T×T causal attention for training. Saves probability matrix for backward.
// One block per (head, query_position). Grid: numHeads * T.
__global__ void training_causal_gated_attention(
    float* output, float* savedProbs,
    const float* qAttn, const float* qGate,
    const float* k, const float* v,
    int T, int numHeads, int numKvHeads, int keyDim, int valDim, float scale)
{
    int blockId = blockIdx.x;
    int h = blockId / T;
    int t = blockId % T;
    if (h >= numHeads) return;

    int tid = threadIdx.x;
    int kvH = h / (numHeads / numKvHeads);
    int seqLen = t + 1;

    int qOff = t * numHeads * keyDim + h * keyDim;

    // Compute attention scores
    for (int s = tid; s < seqLen; s += blockDim.x)
    {
        int kOff = s * numKvHeads * keyDim + kvH * keyDim;
        float score = 0;
        for (int d = 0; d < keyDim; d++)
            score += qAttn[qOff + d] * k[kOff + d];
        score *= scale;
        savedProbs[h * T * T + t * T + s] = score;
    }
    // Zero non-causal
    for (int s = seqLen + tid; s < T; s += blockDim.x)
        savedProbs[h * T * T + t * T + s] = 0;
    __syncthreads();

    // Softmax over [0..seqLen-1]
    extern __shared__ float sdata[];

    // Find max
    float localMax = -1e30f;
    for (int s = tid; s < seqLen; s += blockDim.x)
        localMax = fmaxf(localMax, savedProbs[h * T * T + t * T + s]);
    sdata[tid] = localMax;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    float maxVal = sdata[0];
    __syncthreads();

    // Exp + sum
    float localSum = 0;
    for (int s = tid; s < seqLen; s += blockDim.x)
    {
        float p = expf(savedProbs[h * T * T + t * T + s] - maxVal);
        savedProbs[h * T * T + t * T + s] = p;
        localSum += p;
    }
    sdata[tid] = localSum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float invSum = 1.0f / sdata[0];
    __syncthreads();

    for (int s = tid; s < seqLen; s += blockDim.x)
        savedProbs[h * T * T + t * T + s] *= invSum;
    __syncthreads();

    // Weighted sum of V + sigmoid gating
    int outOff = t * numHeads * valDim + h * valDim;
    int gateOff = t * numHeads * keyDim + h * keyDim;
    for (int d = tid; d < valDim; d += blockDim.x)
    {
        float sum = 0;
        for (int s = 0; s < seqLen; s++)
        {
            float p = savedProbs[h * T * T + t * T + s];
            int vOff = s * numKvHeads * valDim + kvH * valDim;
            sum += p * v[vOff + d];
        }
        // Sigmoid gating
        float gateVal = (d < keyDim) ? qGate[gateOff + d] : 0.0f;
        float sig = 1.0f / (1.0f + expf(-gateVal));
        output[outOff + d] = sum * sig;
    }
}

// ── Truncated element-wise product ─────────────────────────────────────────
// c[t*cDim + i] = a[t*aDim + i] * b[t*bDim + i] for i < min(cDim, aDim, bDim)
// Handles dimension mismatch for DeltaNet SSM approximation
__global__ void truncated_element_mul(float* c, const float* a, const float* b,
                                       int T, int cDim, int aDim, int bDim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = T * cDim;
    if (idx >= total) return;
    int t = idx / cDim;
    int i = idx % cDim;
    float aVal = (i < aDim) ? a[t * aDim + i] : 0.0f;
    float bVal = (i < bDim) ? b[t * bDim + i] : 0.0f;
    c[idx] = aVal * bVal;
}

// ── Truncated element-wise product backward ────────────────────────────────
__global__ void truncated_element_mul_backward(float* dA, float* dB, const float* dC,
                                                const float* a, const float* b,
                                                int T, int cDim, int aDim, int bDim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = T * cDim;
    if (idx >= total) return;
    int t = idx / cDim;
    int i = idx % cDim;
    float dCval = dC[idx];
    if (i < aDim)
        atomicAdd(&dA[t * aDim + i], dCval * ((i < bDim) ? b[t * bDim + i] : 0.0f));
    if (i < bDim)
        atomicAdd(&dB[t * bDim + i], dCval * ((i < aDim) ? a[t * aDim + i] : 0.0f));
}

// ── Batched embedding lookup ───────────────────────────────────────────────
// output[t*dim .. (t+1)*dim] = table[tokenIds[t]*dim .. ]
__global__ void batched_embedding_lookup(float* output, const float* table,
                                          const int* tokenIds, int T, int dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = T * dim;
    if (idx >= total) return;
    int t = idx / dim;
    int d = idx % dim;
    output[idx] = table[tokenIds[t] * dim + d];
}

} // extern "C"
