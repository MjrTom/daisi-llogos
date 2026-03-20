// daisi-llogos CUDA kernels: composite operations for forward pass
// These support the IComputeBackend composite operations used by ForwardPass.

// FP16 ↔ FP32 conversion using bit manipulation (no cuda_fp16.h needed for NVRTC)
__device__ float fp16_to_fp32(unsigned short h)
{
    unsigned int sign = (h >> 15) & 1;
    unsigned int exp_val = (h >> 10) & 0x1f;
    unsigned int mant = h & 0x3ff;
    unsigned int f;
    if (exp_val == 0) {
        if (mant == 0) f = sign << 31;
        else { exp_val = 1; while (!(mant & 0x400)) { mant <<= 1; exp_val--; } mant &= 0x3ff; f = (sign << 31) | ((exp_val + 127 - 15) << 23) | (mant << 13); }
    } else if (exp_val == 31) {
        f = (sign << 31) | 0x7f800000 | (mant << 13);
    } else {
        f = (sign << 31) | ((exp_val + 127 - 15) << 23) | (mant << 13);
    }
    return *reinterpret_cast<float*>(&f);
}

__device__ unsigned short fp32_to_fp16(float val)
{
    unsigned int f = *reinterpret_cast<unsigned int*>(&val);
    unsigned int sign = (f >> 16) & 0x8000;
    int exp_val = ((f >> 23) & 0xff) - 127 + 15;
    unsigned int mant = (f >> 13) & 0x3ff;
    if (exp_val <= 0) return (unsigned short)sign;  // flush to zero
    if (exp_val >= 31) return (unsigned short)(sign | 0x7c00);  // infinity
    return (unsigned short)(sign | (exp_val << 10) | mant);
}

// Helper to load float from either FP32 or FP16 cache
__device__ float load_cache_val(const void* cache, int idx, int isFp16)
{
    if (isFp16)
        return fp16_to_fp32(((const unsigned short*)cache)[idx]);
    else
        return ((const float*)cache)[idx];
}

// Helper to store float to either FP32 or FP16 cache
__device__ void store_cache_val(void* cache, int idx, float val, int isFp16)
{
    if (isFp16)
        ((unsigned short*)cache)[idx] = fp32_to_fp16(val);
    else
        ((float*)cache)[idx] = val;
}

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
// Supports FP16 cache: if cacheIsFp16, converts FP32 inputs to FP16 on write.

__global__ void kv_cache_write(void* kCache, void* vCache,
                                const float* k, const float* v,
                                int nKvHeads, int keyLength, int valueLength,
                                int maxSeqLen, int position, int cacheIsFp16)
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
        store_cache_val(kCache, cache_idx, k[idx], cacheIsFp16);
    }

    // Write V
    if (idx < totalV)
    {
        int head = idx / valueLength;
        int dim = idx % valueLength;
        int cache_idx = head * maxSeqLen * valueLength + position * valueLength + dim;
        store_cache_val(vCache, cache_idx, v[idx], cacheIsFp16);
    }
}

// ── Gated Attention (Tiled / Flash) ─────────────────────────────────────────
// One block per attention head. Uses online softmax to tile the computation,
// so shared memory usage is O(tileSize + blockDim) regardless of context length.
// Supports FP16 KV cache via cacheIsFp16 parameter.
//
// output[h] = sigmoid(qGate[h]) * softmax(qAttn[h] @ kCache[kv_h]^T * scale) @ vCache[kv_h]

#define ATTN_TILE_SIZE 256

__global__ void gated_attention(float* output,
                                 const float* qAttn, const float* qGate,
                                 const void* kCache, const void* vCache,
                                 int numHeads, int numKvHeads, int keyLength,
                                 int valueLength, int maxSeqLen, int seqLen,
                                 float scale, int cacheIsFp16)
{
    extern __shared__ float shared[];
    int head = blockIdx.x;
    int tid = threadIdx.x;
    int stride = blockDim.x;
    int kvHead = head * numKvHeads / numHeads;

    const float* q = qAttn + head * keyLength;
    const float* qg = qGate + head * keyLength;

    // Byte offsets into cache depend on element size
    int elemSize = cacheIsFp16 ? 1 : 1; // handled by load_cache_val
    int kBaseOffset = kvHead * maxSeqLen * keyLength;
    int vBaseOffset = kvHead * maxSeqLen * valueLength;

    // Shared memory layout: [ATTN_TILE_SIZE scores] + [blockDim.x temp for reduction]
    float* scores = shared;
    float* temp = shared + ATTN_TILE_SIZE;

    // Initialize output to zero
    float* outHead = output + head * valueLength;
    for (int d = tid; d < valueLength; d += stride)
        outHead[d] = 0.0f;
    __syncthreads();

    // Online softmax state (same across all threads after reductions)
    float running_max = -1e30f;
    float running_sum = 0.0f;

    // Process tiles
    for (int tile_start = 0; tile_start < seqLen; tile_start += ATTN_TILE_SIZE)
    {
        int tile_end = tile_start + ATTN_TILE_SIZE;
        if (tile_end > seqLen) tile_end = seqLen;
        int tile_len = tile_end - tile_start;

        // Compute attention scores for this tile: q @ K_tile^T * scale
        for (int t = tid; t < tile_len; t += stride)
        {
            float dot = 0.0f;
            int kOffset = kBaseOffset + (tile_start + t) * keyLength;
            for (int d = 0; d < keyLength; d++)
                dot += q[d] * load_cache_val(kCache, kOffset + d, cacheIsFp16);
            scores[t] = dot * scale;
        }
        __syncthreads();

        // Find tile max via reduction
        float local_max = -1e30f;
        for (int t = tid; t < tile_len; t += stride)
            local_max = fmaxf(local_max, scores[t]);
        temp[tid] = local_max;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) temp[tid] = fmaxf(temp[tid], temp[tid + s]);
            __syncthreads();
        }
        float tile_max = temp[0];
        __syncthreads();

        // Compute exp(scores - tile_max) and tile sum
        float local_sum = 0.0f;
        for (int t = tid; t < tile_len; t += stride) {
            float val = expf(scores[t] - tile_max);
            scores[t] = val;
            local_sum += val;
        }
        temp[tid] = local_sum;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) temp[tid] += temp[tid + s];
            __syncthreads();
        }
        float tile_sum = temp[0];
        __syncthreads();

        // Online softmax merge: rescale running output
        float new_max = fmaxf(running_max, tile_max);
        float correction_old = expf(running_max - new_max);
        float correction_new = expf(tile_max - new_max);

        // Update output: rescale old accumulation + add new tile's weighted V
        for (int d = tid; d < valueLength; d += stride)
        {
            float tile_val = 0.0f;
            for (int t = 0; t < tile_len; t++)
            {
                int vIdx = vBaseOffset + (tile_start + t) * valueLength + d;
                tile_val += scores[t] * load_cache_val(vCache, vIdx, cacheIsFp16);
            }
            outHead[d] = outHead[d] * correction_old + tile_val * correction_new;
        }
        __syncthreads();

        running_sum = running_sum * correction_old + tile_sum * correction_new;
        running_max = new_max;
    }

    // Normalize by total sum and apply sigmoid gating
    float inv_sum = (running_sum > 0.0f) ? 1.0f / running_sum : 0.0f;
    for (int d = tid; d < valueLength; d += stride)
    {
        float g = (d < keyLength) ? qg[d] : 0.0f;
        float sig = 1.0f / (1.0f + expf(-g));
        outHead[d] = outHead[d] * inv_sum * sig;
    }
}

// ── Causal Conv1d ────────────────────────────────────────────────────────────
// Depthwise causal conv1d with shift buffer.
// qkv: [channels], convBuffer: [(kernelSize-1) × channels], convWeight: [kernelSize × channels]
// GGUF dim layout: [kernelSize, channels] → row-major = channels × kernelSize.
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

// ── Split QKV with unequal sizes ──────────────────────────────────────────
// Layout: [Q:keyDim, K:keyDim, V:valueDim] → separate Q, K, V tensors.
// Q and K are valueDim-sized (zero-padded). V is valueDim-sized.

__global__ void split_unequal_qkv(float* q, float* k, float* v, const float* qkv,
                                   int keyDim, int valueDim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Zero Q and K first (they're valueDim-sized but only keyDim elements filled)
    if (idx < valueDim)
    {
        q[idx] = idx < keyDim ? qkv[idx] : 0.0f;
        k[idx] = idx < keyDim ? qkv[keyDim + idx] : 0.0f;
    }

    // Copy V (valueDim elements)
    if (idx < valueDim)
        v[idx] = qkv[keyDim * 2 + idx];
}

// ── Tile heads: [h0..hN] → [h0..hN, h0..hN] repeated factor times ───────
// In-place: reads from first srcSize elements, tiles into full dstSize.

__global__ void repeat_tile(float* data, int srcSize, int dstSize)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dstSize || idx < srcSize) return;  // skip source region
    data[idx] = data[idx % srcSize];
}

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
