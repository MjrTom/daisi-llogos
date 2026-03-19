// daisi-llama CUDA kernels: composite operations for forward pass
// These support the IComputeBackend composite operations used by ForwardPass.

extern "C" {

// ── SiLU in-place ────────────────────────────────────────────────────────────
// data[i] = data[i] * sigmoid(data[i])

__global__ void silu_inplace(float* data, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        float x = data[idx];
        data[idx] = x / (1.0f + expf(-x));
    }
}

// ── SiLU gate ────────────────────────────────────────────────────────────────
// output[i] = data[i] * silu(gate[i])

__global__ void silu_gate(float* output, const float* data, const float* gate, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        float g = gate[idx];
        output[idx] = data[idx] * g / (1.0f + expf(-g));
    }
}

// ── L2 norm groups ───────────────────────────────────────────────────────────
// Normalize each group of groupDim elements to unit L2 norm.
// Grid: numGroups blocks, each block handles one group.

__global__ void l2_norm_groups(float* data, int numGroups, int groupDim)
{
    extern __shared__ float sdata[];
    int group = blockIdx.x;
    int tid = threadIdx.x;
    int stride = blockDim.x;
    int base_idx = group * groupDim;

    // Sum of squares
    float sum = 0.0f;
    for (int i = tid; i < groupDim; i += stride)
    {
        float val = data[base_idx + i];
        sum += val * val;
    }

    sdata[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    float inv_norm = rsqrtf(sdata[0] + 1e-12f);

    // Normalize
    for (int i = tid; i < groupDim; i += stride)
        data[base_idx + i] *= inv_norm;
}

// ── Per-head RMSNorm in-place ────────────────────────────────────────────────
// Grid: numHeads blocks, each handles one head of headDim elements.
// weight has headDim elements, shared across all heads.

__global__ void per_head_rms_norm(float* data, const float* weight,
                                   int numHeads, int headDim, float eps)
{
    extern __shared__ float sdata[];
    int head = blockIdx.x;
    int tid = threadIdx.x;
    int stride = blockDim.x;
    int base_idx = head * headDim;

    float sum = 0.0f;
    for (int i = tid; i < headDim; i += stride)
    {
        float val = data[base_idx + i];
        sum += val * val;
    }

    sdata[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    float inv_rms = rsqrtf(sdata[0] / (float)headDim + eps);

    for (int i = tid; i < headDim; i += stride)
        data[base_idx + i] = data[base_idx + i] * inv_rms * weight[i];
}

// ── De-interleave Q ──────────────────────────────────────────────────────────
// qFull layout: [q_h0(headDim), gate_h0(headDim), q_h1(headDim), gate_h1(headDim), ...]
// qAttn: [q_h0, q_h1, ...], qGate: [gate_h0, gate_h1, ...]

__global__ void deinterleave_q(float* qAttn, float* qGate, const float* qFull,
                                int numHeads, int headDim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = numHeads * headDim;
    if (idx >= total) return;

    int head = idx / headDim;
    int dim = idx % headDim;

    int src_base = head * headDim * 2;
    qAttn[head * headDim + dim] = qFull[src_base + dim];
    qGate[head * headDim + dim] = qFull[src_base + headDim + dim];
}

// ── KV cache write ───────────────────────────────────────────────────────────
// Write k[nKvHeads * keyLength] and v[nKvHeads * valueLength] into cache at position.
// kCache: [nKvHeads × maxSeqLen × keyLength], vCache: [nKvHeads × maxSeqLen × valueLength]

__global__ void kv_cache_write(float* kCache, float* vCache,
                                const float* k, const float* v,
                                int nKvHeads, int keyLength, int valueLength,
                                int maxSeqLen, int position)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalK = nKvHeads * keyLength;
    int totalV = nKvHeads * valueLength;

    // Write K
    if (idx < totalK)
    {
        int head = idx / keyLength;
        int dim = idx % keyLength;
        int cache_idx = head * maxSeqLen * keyLength + position * keyLength + dim;
        kCache[cache_idx] = k[idx];
    }

    // Write V
    if (idx < totalV)
    {
        int head = idx / valueLength;
        int dim = idx % valueLength;
        int cache_idx = head * maxSeqLen * valueLength + position * valueLength + dim;
        vCache[cache_idx] = v[idx];
    }
}

// ── Gated Attention ──────────────────────────────────────────────────────────
// One block per attention head.
// output[h] = sigmoid(qGate[h]) * softmax(qAttn[h] @ kCache[kv_h]^T * scale) @ vCache[kv_h]

__global__ void gated_attention(float* output,
                                 const float* qAttn, const float* qGate,
                                 const float* kCache, const float* vCache,
                                 int numHeads, int numKvHeads, int keyLength,
                                 int valueLength, int maxSeqLen, int seqLen,
                                 float scale)
{
    extern __shared__ float shared[];
    int head = blockIdx.x;
    int tid = threadIdx.x;
    int stride = blockDim.x;
    int kvHead = head * numKvHeads / numHeads;

    const float* q = qAttn + head * keyLength;
    const float* qg = qGate + head * keyLength;
    const float* kBase = kCache + kvHead * maxSeqLen * keyLength;
    const float* vBase = vCache + kvHead * maxSeqLen * valueLength;

    // shared layout: [seqLen scores] then [seqLen temps]
    float* scores = shared;

    // Step 1: Compute attention scores = q @ K^T * scale
    for (int pos = tid; pos < seqLen; pos += stride)
    {
        float dot = 0.0f;
        const float* kVec = kBase + pos * keyLength;
        for (int d = 0; d < keyLength; d++)
            dot += q[d] * kVec[d];
        scores[pos] = dot * scale;
    }
    __syncthreads();

    // Step 2: Softmax over scores[0..seqLen-1]
    // Find max
    float local_max = -1e30f;
    for (int pos = tid; pos < seqLen; pos += stride)
        local_max = fmaxf(local_max, scores[pos]);

    // Use second part of shared mem for reduction
    float* temp = shared + seqLen;
    temp[tid] = local_max;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
            temp[tid] = fmaxf(temp[tid], temp[tid + s]);
        __syncthreads();
    }
    float max_val = temp[0];
    __syncthreads();

    // Exp and sum
    float local_sum = 0.0f;
    for (int pos = tid; pos < seqLen; pos += stride)
    {
        float val = expf(scores[pos] - max_val);
        scores[pos] = val;
        local_sum += val;
    }

    temp[tid] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
            temp[tid] += temp[tid + s];
        __syncthreads();
    }
    float inv_sum = 1.0f / temp[0];
    __syncthreads();

    for (int pos = tid; pos < seqLen; pos += stride)
        scores[pos] *= inv_sum;
    __syncthreads();

    // Step 3: Compute sigmoid gate for this head
    float gate_sum = 0.0f;
    for (int d = tid; d < keyLength; d += stride)
        gate_sum += 1.0f; // placeholder for gate norm; actually per-element
    // Actually: sigmoid(qGate) is per-element, applied to output

    // Step 4: Weighted V sum with gating
    float* outHead = output + head * valueLength;
    for (int d = tid; d < valueLength; d += stride)
    {
        float val = 0.0f;
        for (int pos = 0; pos < seqLen; pos++)
            val += scores[pos] * vBase[pos * valueLength + d];

        // Sigmoid gate: average qGate elements for this head's contribution
        // Actually: gate is sigmoid per-element on qGate, then applied as scalar per head
        outHead[d] = val;
    }
    __syncthreads();

    // Apply sigmoid gating: multiply entire head output by mean(sigmoid(qGate[h]))
    // Wait - the actual gating is: output *= sigmoid(qGate) per dimension
    // But qGate is keyLength dimensions and output is valueLength dimensions.
    // Looking at the CPU code: gate = sigmoid(qGate), then output[d] *= gate[d]
    // But keyLength != valueLength in general... Actually for Qwen 3.5 keyLen == valLen == 256.
    // The correct approach: gate is computed per-head as dot or per-element.
    // From the CPU code in CpuBackend.GatedAttention:
    //   sigmoid gate applied per-dimension of output, using qGate[h*keyLen + d]
    //   since keyLen == valLen for this model.

    // Simple approach: multiply output[d] by sigmoid(qGate[h*keyLen + d])
    for (int d = tid; d < valueLength; d += stride)
    {
        if (d < keyLength)
        {
            float g = qg[d];
            float sig = 1.0f / (1.0f + expf(-g));
            outHead[d] *= sig;
        }
    }
}

// ── Causal Conv1d ────────────────────────────────────────────────────────────
// Depthwise causal conv1d with shift buffer.
// qkv: [channels], convBuffer: [(kernelSize-1) × channels], convWeight: [channels × kernelSize]
// Updates convBuffer (shift left, append new) and writes conv output back to qkv.

__global__ void causal_conv1d(float* qkv, float* convBuffer, const float* convWeight,
                               int channels, int kernelSize)
{
    int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= channels) return;

    int historyLen = kernelSize - 1;

    // Compute conv output for this channel
    float sum = 0.0f;

    // From history buffer (older samples)
    for (int t = 0; t < historyLen; t++)
    {
        float sample = convBuffer[t * channels + ch];
        float w = convWeight[ch * kernelSize + t];
        sum += sample * w;
    }

    // Current sample (newest)
    float current = qkv[ch];
    sum += current * convWeight[ch * kernelSize + historyLen];

    // Shift buffer left: drop oldest, append current
    for (int t = 0; t < historyLen - 1; t++)
        convBuffer[t * channels + ch] = convBuffer[(t + 1) * channels + ch];
    if (historyLen > 0)
        convBuffer[(historyLen - 1) * channels + ch] = current;

    // Write conv result back
    qkv[ch] = sum;
}

// ── Compute Decay and Beta ───────────────────────────────────────────────────
// decay[g] = exp(ssmA[g] * softplus(alpha[g] + dtBias[g]))
// beta[g] = sigmoid(betaProj[g])
// One thread per group.

__global__ void compute_decay_beta(float* decay, float* beta,
                                    const float* alphaProj, const float* betaProj,
                                    const float* ssmA, const float* dtBias,
                                    int groupCount)
{
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= groupCount) return;

    // softplus(x) = log(1 + exp(x))
    float x = alphaProj[g] + dtBias[g];
    float sp = (x > 20.0f) ? x : logf(1.0f + expf(x));
    float dt = sp;

    decay[g] = expf(ssmA[g] * dt);

    float b = betaProj[g];
    beta[g] = 1.0f / (1.0f + expf(-b));
}

// ── Split QKV ────────────────────────────────────────────────────────────────
// Split [Q|K|V] concatenated buffer into separate tensors.

__global__ void split_qkv(float* q, float* k, float* v, const float* qkv, int innerSize)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= innerSize) return;

    q[idx] = qkv[idx];
    k[idx] = qkv[innerSize + idx];
    v[idx] = qkv[innerSize * 2 + idx];
}

// ── DeltaNet Step ────────────────────────────────────────────────────────────
// Fused state update + output computation + per-head RMSNorm.
// One block per group (head).
// state: [groupCount × headDim × headDim], q/k/v: [groupCount × headDim]
// For each group:
//   sk = S^T * k, error = (v - decay * sk) * beta
//   S = decay * S + outer(k, error)
//   o = S^T * q * scale
//   Then per-head RMSNorm on o.

__global__ void deltanet_step(float* output, const float* q, const float* k,
                               const float* v, float* state,
                               const float* decay, const float* beta,
                               const float* normWeight,
                               int groupCount, int headDim, float scale, float normEps)
{
    extern __shared__ float sdata[];
    int group = blockIdx.x;
    int tid = threadIdx.x;
    int stride = blockDim.x;

    float d = decay[group];
    float b = beta[group];

    const float* qh = q + group * headDim;
    const float* kh = k + group * headDim;
    const float* vh = v + group * headDim;
    float* S = state + group * headDim * headDim;
    float* oh = output + group * headDim;

    // For each row i of the state matrix (parallelized across threads):
    // sk[i] = sum_j(S[i][j] * k[j])
    // error[i] = (v[i] - d * sk[i]) * b
    // S[i][j] = d * S[i][j] + k[j] * error[i]
    // o[i] = sum_j(S[i][j] * q[j]) * scale

    for (int i = tid; i < headDim; i += stride)
    {
        float* Si = S + i * headDim;

        // sk = S[i,:] . k
        float sk = 0.0f;
        for (int j = 0; j < headDim; j++)
            sk += Si[j] * kh[j];

        // error
        float error = (vh[i] - d * sk) * b;

        // Update state row and compute output
        float oi = 0.0f;
        for (int j = 0; j < headDim; j++)
        {
            Si[j] = d * Si[j] + kh[j] * error;
            oi += Si[j] * qh[j];
        }

        oh[i] = oi * scale;
    }
    __syncthreads();

    // Per-head RMSNorm on output
    float sum_sq = 0.0f;
    for (int i = tid; i < headDim; i += stride)
        sum_sq += oh[i] * oh[i];

    sdata[tid] = sum_sq;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    float inv_rms = rsqrtf(sdata[0] / (float)headDim + normEps);

    for (int i = tid; i < headDim; i += stride)
        oh[i] = oh[i] * inv_rms * normWeight[i];
}

} // extern "C"
