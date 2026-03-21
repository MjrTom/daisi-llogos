// daisi-llogos CUDA kernels: element-wise operations
// Compiled to .cubin and embedded as assembly resources.

// FP32 → FP16 via PTX
__device__ __forceinline__ unsigned short fp32_to_fp16(float val)
{
    unsigned short result;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(result) : "f"(val));
    return result;
}

extern "C" {

// ── RMSNorm ──────────────────────────────────────────────────────────────────
// output[i] = (input[i] / rms) * weight[i]
// Two-pass: first compute sum of squares (block reduction), then normalize.

__global__ void rms_norm(float* output, const float* input, const float* weight,
                         int n, float eps)
{
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int stride = blockDim.x;

    // Pass 1: compute sum of squares
    float sum = 0.0f;
    for (int i = tid; i < n; i += stride)
        sum += input[i] * input[i];

    sdata[tid] = sum;
    __syncthreads();

    // Block reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    float rms = sqrtf(sdata[0] / (float)n + eps);
    float inv_rms = 1.0f / rms;

    // Pass 2: normalize
    for (int i = tid; i < n; i += stride)
        output[i] = input[i] * inv_rms * weight[i];
}

// ── Softmax ──────────────────────────────────────────────────────────────────
// Numerically stable: subtract max, exp, normalize.
// Single block launch for typical vocab sizes.

__global__ void softmax(float* output, const float* input, int n)
{
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int stride = blockDim.x;

    // Pass 1: find max
    float local_max = -1e30f;
    for (int i = tid; i < n; i += stride)
        local_max = fmaxf(local_max, input[i]);

    sdata[tid] = local_max;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    float max_val = sdata[0];
    __syncthreads();

    // Pass 2: exp and sum
    float local_sum = 0.0f;
    for (int i = tid; i < n; i += stride)
    {
        float val = expf(input[i] - max_val);
        output[i] = val;
        local_sum += val;
    }

    sdata[tid] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float sum = sdata[0];
    __syncthreads();

    // Pass 3: normalize
    float inv_sum = 1.0f / sum;
    for (int i = tid; i < n; i += stride)
        output[i] *= inv_sum;
}

// ── SiLU ─────────────────────────────────────────────────────────────────────
// output[i] = input[i] * sigmoid(input[i])

__global__ void silu(float* output, const float* input, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        output[idx] = input[idx] / (1.0f + expf(-input[idx]));
}

// ── RoPE ─────────────────────────────────────────────────────────────────────
// Apply rotary position embedding to q and k tensors.
// Operates on pairs of dimensions (2i, 2i+1) within each head.
// q: [nHeads * headDim], k: [nKvHeads * headDim]

__global__ void rope(float* q, float* k,
                     int q_total, int k_total,
                     int head_dim, int rope_dim,
                     int position, float theta)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half_rope = rope_dim / 2;

    // Process Q
    if (idx < q_total / 2)
    {
        int head = idx / (head_dim / 2);
        int pair = idx % (head_dim / 2);
        if (pair < half_rope)
        {
            float freq = 1.0f / powf(theta, (float)(2 * pair) / (float)rope_dim);
            float angle = (float)position * freq;
            float cos_a = cosf(angle);
            float sin_a = sinf(angle);

            int base_idx = head * head_dim + pair * 2;
            float v0 = q[base_idx];
            float v1 = q[base_idx + 1];
            q[base_idx]     = v0 * cos_a - v1 * sin_a;
            q[base_idx + 1] = v0 * sin_a + v1 * cos_a;
        }
    }

    // Process K
    if (idx < k_total / 2)
    {
        int head = idx / (head_dim / 2);
        int pair = idx % (head_dim / 2);
        if (pair < half_rope)
        {
            float freq = 1.0f / powf(theta, (float)(2 * pair) / (float)rope_dim);
            float angle = (float)position * freq;
            float cos_a = cosf(angle);
            float sin_a = sinf(angle);

            int base_idx = head * head_dim + pair * 2;
            float v0 = k[base_idx];
            float v1 = k[base_idx + 1];
            k[base_idx]     = v0 * cos_a - v1 * sin_a;
            k[base_idx + 1] = v0 * sin_a + v1 * cos_a;
        }
    }
}

// ── Element-wise multiply ────────────────────────────────────────────────────

__global__ void element_mul(float* output, const float* a, const float* b, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        output[idx] = a[idx] * b[idx];
}

// ── Element-wise add ─────────────────────────────────────────────────────────

__global__ void element_add(float* output, const float* a, const float* b, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        output[idx] = a[idx] + b[idx];
}

// ── Embedding lookup ─────────────────────────────────────────────────────────
// Copy a single row from an FP32 embedding table to output.

// Flexible Q8_0 embedding lookup that handles both 34-byte and 36-byte aligned blocks
__global__ void embedding_lookup_q8_0_v2(float* output, const unsigned char* table,
                                          int hidden_dim, int token_id,
                                          int block_stride, int quant_offset)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= hidden_dim) return;

    int blocks_per_row = hidden_dim / 32;
    int bytes_per_row = blocks_per_row * block_stride;
    const unsigned char* row = table + (long long)token_id * bytes_per_row;

    int block_idx = idx / 32;
    int elem_idx = idx % 32;
    const unsigned char* block = row + block_idx * block_stride;

    unsigned short h = ((unsigned short)block[1] << 8) | block[0];
    // Inline FP16→FP32
    unsigned int sign = (h >> 15) & 1;
    unsigned int exp_val = (h >> 10) & 0x1f;
    unsigned int mant = h & 0x3ff;
    unsigned int f;
    if (exp_val == 0) { f = sign << 31; }
    else if (exp_val == 31) { f = (sign << 31) | 0x7f800000 | (mant << 13); }
    else { f = (sign << 31) | ((exp_val - 15 + 127) << 23) | (mant << 13); }
    float scale = *reinterpret_cast<float*>(&f);
    signed char quant = (signed char)block[quant_offset + elem_idx];
    output[idx] = scale * (float)quant;
}

__global__ void embedding_lookup_f32(float* output, const float* table,
                                      int hidden_dim, int token_id)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < hidden_dim)
        output[idx] = table[token_id * hidden_dim + idx];
}

// Q8_0 embedding lookup: dequantize one row.
// Q8_0 block: 2 bytes (fp16 scale) + 32 bytes (int8 quants) = 34 bytes per block.

__global__ void embedding_lookup_q8_0(float* output, const unsigned char* table,
                                       int hidden_dim, int token_id)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= hidden_dim) return;

    int blocks_per_row = hidden_dim / 32;
    int bytes_per_row = blocks_per_row * 34;
    const unsigned char* row = table + token_id * bytes_per_row;

    int block_idx = idx / 32;
    int elem_idx = idx % 32;

    const unsigned char* block = row + block_idx * 34;
    // First 2 bytes are FP16 scale
    unsigned short scale_bits = ((unsigned short)block[1] << 8) | block[0];
    // Convert FP16 to FP32
    int sign = (scale_bits >> 15) & 1;
    int exp = (scale_bits >> 10) & 0x1f;
    int mant = scale_bits & 0x3ff;
    float scale;
    if (exp == 0)
        scale = (sign ? -1.0f : 1.0f) * (mant / 1024.0f) * (1.0f / 16384.0f);
    else if (exp == 31)
        scale = 0.0f;
    else
        scale = (sign ? -1.0f : 1.0f) * (1.0f + mant / 1024.0f) * powf(2.0f, (float)(exp - 15));

    signed char quant = (signed char)block[2 + elem_idx];
    output[idx] = scale * (float)quant;
}

// ── F16 embedding lookup ──────────────────────────────────────────────────────
// Each element is 2 bytes (FP16). Dequantize one row to FP32.

__global__ void embedding_lookup_f16(float* output, const unsigned char* table,
                                      int hidden_dim, int token_id)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= hidden_dim) return;

    int bytes_per_row = hidden_dim * 2;
    const unsigned char* row = table + token_id * bytes_per_row;
    int byte_off = idx * 2;

    unsigned short bits = ((unsigned short)row[byte_off + 1] << 8) | row[byte_off];
    // Convert FP16 to FP32
    int sign = (bits >> 15) & 1;
    int exp = (bits >> 10) & 0x1f;
    int mant = bits & 0x3ff;
    float val;
    if (exp == 0)
        val = (sign ? -1.0f : 1.0f) * (mant / 1024.0f) * (1.0f / 16384.0f);
    else if (exp == 31)
        val = 0.0f;
    else
        val = (sign ? -1.0f : 1.0f) * (1.0f + mant / 1024.0f) * powf(2.0f, (float)(exp - 15));

    output[idx] = val;
}

// FP16 → FP32 via PTX cvt instruction
__device__ float fp16_to_fp32_emb(unsigned short h)
{
    float result;
    asm("cvt.f32.f16 %0, %1;" : "=f"(result) : "h"(h));
    return result;
}

// ── Q4_K embedding lookup ─────────────────────────────────────────────────────
// Q4_K super block: 256 elements, 144 bytes.
// Layout: 2b d (fp16) + 2b dmin (fp16) + 12b packed scales/mins + 128b nibbles.

__global__ void embedding_lookup_q4_k(float* output, const unsigned char* table,
                                       int hidden_dim, int token_id)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= hidden_dim) return;

    int blocks_per_row = hidden_dim / 256;
    int bytes_per_row = blocks_per_row * 144;
    const unsigned char* row = table + (long long)token_id * bytes_per_row;

    int sb_idx = idx / 256;
    int elem_in_sb = idx % 256;
    const unsigned char* sb = row + sb_idx * 144;

    unsigned short d_bits = ((unsigned short)sb[1] << 8) | sb[0];
    unsigned short dmin_bits = ((unsigned short)sb[3] << 8) | sb[2];
    float d = fp16_to_fp32_emb(d_bits);
    float dmin = fp16_to_fp32_emb(dmin_bits);

    // Chunk layout: 4 chunks of 64 elements, 32 qs bytes each.
    // lo nibble → first 32 elems, hi nibble → second 32 elems.
    int chunk = elem_in_sb / 64;
    int pos_in_chunk = elem_in_sb % 64;
    int l = pos_in_chunk % 32;
    int is_hi = pos_in_chunk / 32; // 0 for first sub-block, 1 for second

    int j = chunk * 2 + is_hi; // sub-block index for scales
    int sc, m;
    if (j < 4) {
        sc = sb[4 + j] & 63;
        m  = sb[4 + j + 4] & 63;
    } else {
        sc = (sb[4 + j + 4] & 0xF) | ((sb[4 + j - 4] >> 6) << 4);
        m  = (sb[4 + j + 4] >> 4)  | ((sb[4 + j]     >> 6) << 4);
    }

    unsigned char packed = sb[16 + chunk * 32 + l];
    int nibble = is_hi ? (packed >> 4) : (packed & 0xF);

    output[idx] = d * (float)sc * (float)nibble - dmin * (float)m;
}

// ── Fused: Residual save + RMSNorm ────────────────────────────────────────────
// residual[i] = input[i]; output[i] = (input[i] / rms) * weight[i]
// Saves one kernel launch + one full read/write of the hidden state.

__global__ void rms_norm_residual(float* output, float* residual, const float* input,
                                   const float* weight, int n, float eps)
{
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int stride = blockDim.x;

    // Pass 1: copy to residual + compute sum of squares
    float sum = 0.0f;
    for (int i = tid; i < n; i += stride)
    {
        float v = input[i];
        residual[i] = v;
        sum += v * v;
    }

    sdata[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    float inv_rms = 1.0f / sqrtf(sdata[0] / (float)n + eps);

    // Pass 2: normalize
    for (int i = tid; i < n; i += stride)
        output[i] = input[i] * inv_rms * weight[i];
}

// ── Fused: SwiGLU (SiLU + ElementMul) ────────────────────────────────────────
// output[i] = (gate[i] * sigmoid(gate[i])) * up[i]
// Saves one kernel launch + two memory round-trips.

__global__ void swiglu(float* output, const float* gate, const float* up, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        float g = gate[idx];
        output[idx] = (g / (1.0f + expf(-g))) * up[idx];
    }
}

// ── Fused: Element add + RMSNorm (residual add then norm) ────────────────────
// hidden[i] = a[i] + b[i]; output[i] = (hidden[i] / rms) * weight[i]

__global__ void add_rms_norm(float* output, float* hidden, const float* a, const float* b,
                              const float* weight, int n, float eps)
{
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int stride = blockDim.x;

    // Pass 1: add + sum of squares
    float sum = 0.0f;
    for (int i = tid; i < n; i += stride)
    {
        float v = a[i] + b[i];
        hidden[i] = v;
        sum += v * v;
    }

    sdata[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    float inv_rms = 1.0f / sqrtf(sdata[0] / (float)n + eps);

    // Pass 2: normalize from hidden
    for (int i = tid; i < n; i += stride)
        output[i] = hidden[i] * inv_rms * weight[i];
}

// ── ArgMax reduction ──────────────────────────────────────────────────────────
// Find the index of the maximum value. Result written to output[0] as float (cast to int).
// Single block launch with shared memory reduction.

__global__ void argmax(float* output, const float* input, int n)
{
    extern __shared__ float sdata[];  // [2 * blockDim.x]: values then indices
    int tid = threadIdx.x;
    int stride = blockDim.x;
    float* svals = sdata;
    float* sidxs = sdata + blockDim.x;

    // Phase 1: each thread finds its local max
    float bestVal = -1e30f;
    float bestIdx = 0;
    for (int i = tid; i < n; i += stride)
    {
        float v = input[i];
        if (v > bestVal) { bestVal = v; bestIdx = (float)i; }
    }
    svals[tid] = bestVal;
    sidxs[tid] = bestIdx;
    __syncthreads();

    // Phase 2: tree reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s && svals[tid + s] > svals[tid])
        {
            svals[tid] = svals[tid + s];
            sidxs[tid] = sidxs[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0)
        output[0] = sidxs[0];
}

// ── Fused: Split gate+up + SwiGLU ─────────────────────────────────────────────
// input is [gate(N) | up(N)], output[i] = SiLU(gate[i]) * up[i]
// Replaces 2 CopyTensorRegion + 1 SwiGLU = 3 launches → 1.

__global__ void split_swiglu(float* output, const float* fused_input, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        float g = fused_input[idx];         // gate region
        float u = fused_input[N + idx];     // up region
        output[idx] = (g / (1.0f + expf(-g))) * u;
    }
}

// post_qkv_norm_rope_cache removed — KV cache state management too complex for single kernel

#if 0
// DISABLED: kept for reference
__global__ void post_qkv_norm_rope_cache_DISABLED(
    float* qOut, float* kOut, float* vOut,
    const float* fusedQkv,
    void* kCache, void* vCache,
    int qDim, int kDim, int vDim,
    int numHeads, int numKvHeads, int headDim, int ropeDim,
    int position, float ropeTheta, float normEps,
    int maxSeqLen, int cacheIsFp16,
    const float* qNormWeight, const float* kNormWeight)
{
    int tid = threadIdx.x;
    int stride = blockDim.x;
    int head = blockIdx.x;
    int headsPerGroup = numHeads / numKvHeads;

    if (head < numHeads)
    {
        // --- Q head processing ---
        int qOff = head * headDim;
        // Copy from fused output
        for (int i = tid; i < headDim; i += stride)
            qOut[qOff + i] = fusedQkv[qOff + i];

        __syncthreads();

        // Per-head RmsNorm (if norm weights provided)
        if (qNormWeight != 0)
        {
            extern __shared__ float sdata[];
            float sumSq = 0;
            for (int i = tid; i < headDim; i += stride)
                sumSq += qOut[qOff + i] * qOut[qOff + i];
            sdata[tid] = sumSq;
            __syncthreads();
            for (int s = blockDim.x/2; s > 0; s >>= 1) {
                if (tid < s) sdata[tid] += sdata[tid + s];
                __syncthreads();
            }
            float invRms = 1.0f / sqrtf(sdata[0] / (float)headDim + normEps);
            for (int i = tid; i < headDim; i += stride)
                qOut[qOff + i] = qOut[qOff + i] * invRms * qNormWeight[i];
            __syncthreads();
        }

        // RoPE on Q
        int halfRope = ropeDim / 2;
        for (int pair = tid; pair < headDim/2; pair += stride)
        {
            if (pair < halfRope)
            {
                float freq = 1.0f / powf(ropeTheta, (float)(2*pair) / (float)ropeDim);
                float angle = (float)position * freq;
                float cos_a = cosf(angle), sin_a = sinf(angle);
                int bi = qOff + pair * 2;
                float v0 = qOut[bi], v1 = qOut[bi + 1];
                qOut[bi]     = v0 * cos_a - v1 * sin_a;
                qOut[bi + 1] = v0 * sin_a + v1 * cos_a;
            }
        }
    }

    // --- K/V head processing (only numKvHeads blocks do this) ---
    int kvHead = head / headsPerGroup;
    bool isFirstInGroup = (head % headsPerGroup == 0);

    if (isFirstInGroup && kvHead < numKvHeads)
    {
        int kOff = kvHead * headDim;
        int kSrc = qDim + kOff;  // K starts after Q in fused output

        // Copy K from fused output
        for (int i = tid; i < headDim; i += stride)
            kOut[kOff + i] = fusedQkv[kSrc + i];

        // Copy V from fused output
        int vOff = kvHead * headDim; // assuming valLen == headDim
        int vSrc = qDim + kDim + vOff;
        for (int i = tid; i < headDim; i += stride)
            vOut[vOff + i] = fusedQkv[vSrc + i];

        __syncthreads();

        // Per-head RmsNorm on K
        if (kNormWeight != 0)
        {
            extern __shared__ float sdata[];
            float sumSq = 0;
            for (int i = tid; i < headDim; i += stride)
                sumSq += kOut[kOff + i] * kOut[kOff + i];
            sdata[tid] = sumSq;
            __syncthreads();
            for (int s = blockDim.x/2; s > 0; s >>= 1) {
                if (tid < s) sdata[tid] += sdata[tid + s];
                __syncthreads();
            }
            float invRms = 1.0f / sqrtf(sdata[0] / (float)headDim + normEps);
            for (int i = tid; i < headDim; i += stride)
                kOut[kOff + i] = kOut[kOff + i] * invRms * kNormWeight[i];
            __syncthreads();
        }

        // RoPE on K
        int halfRope = ropeDim / 2;
        for (int pair = tid; pair < headDim/2; pair += stride)
        {
            if (pair < halfRope)
            {
                float freq = 1.0f / powf(ropeTheta, (float)(2*pair) / (float)ropeDim);
                float angle = (float)position * freq;
                float cos_a = cosf(angle), sin_a = sinf(angle);
                int bi = kOff + pair * 2;
                float v0 = kOut[bi], v1 = kOut[bi + 1];
                kOut[bi]     = v0 * cos_a - v1 * sin_a;
                kOut[bi + 1] = v0 * sin_a + v1 * cos_a;
            }
        }

        __syncthreads();

        // KV cache write
        int keyLength = headDim;
        int valueLength = headDim;
        int kCacheOff = kvHead * maxSeqLen * keyLength + position * keyLength;
        int vCacheOff = kvHead * maxSeqLen * valueLength + position * valueLength;

        if (cacheIsFp16)
        {
            for (int i = tid; i < keyLength; i += stride)
                ((unsigned short*)kCache)[kCacheOff + i] = fp32_to_fp16(kOut[kOff + i]);
            for (int i = tid; i < valueLength; i += stride)
                ((unsigned short*)vCache)[vCacheOff + i] = fp32_to_fp16(vOut[vOff + i]);
        }
        else
        {
            for (int i = tid; i < keyLength; i += stride)
                ((float*)kCache)[kCacheOff + i] = kOut[kOff + i];
            for (int i = tid; i < valueLength; i += stride)
                ((float*)vCache)[vCacheOff + i] = vOut[vOff + i];
        }
    }
}

#endif // post_qkv disabled

// ── Fill tensor ───────────────────────────────────────────────────────────────
// output[i] = value

__global__ void fill_tensor(float* output, int n, float value)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        output[idx] = value;
}

// ── Squared ReLU (in-place) ──────────────────────────────────────────────────
// data[i] = max(0, data[i])^2

__global__ void squared_relu(float* data, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        float x = fmaxf(0.0f, data[idx]);
        data[idx] = x * x;
    }
}

} // extern "C"
