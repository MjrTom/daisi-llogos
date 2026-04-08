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

    // Pass 1: compute sum of squares (double precision accumulation for precision)
    double sum = 0.0;
    for (int i = tid; i < n; i += stride)
        sum += (double)input[i] * (double)input[i];

    sdata[tid] = (float)sum;
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

// ── Batched RMSNorm ─────────────────────────────────────────────────────────
// Same as rms_norm but processes M independent rows. blockIdx.x selects the row.
// input/output are [M × n] row-major. weight is [n] (shared across rows).

__global__ void batched_rms_norm(float* output, const float* input, const float* weight,
                                  int n, int M, float eps)
{
    extern __shared__ float sdata[];

    int row = blockIdx.x;
    if (row >= M) return;

    int tid = threadIdx.x;
    int stride = blockDim.x;
    int offset = row * n;

    // Pass 1: sum of squares for this row
    float sum = 0.0f;
    for (int i = tid; i < n; i += stride)
        sum += input[offset + i] * input[offset + i];

    sdata[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    float inv_rms = 1.0f / sqrtf(sdata[0] / (float)n + eps);

    // Pass 2: normalize this row
    for (int i = tid; i < n; i += stride)
        output[offset + i] = input[offset + i] * inv_rms * weight[i];
}

// ── Batched RmsNormResidual ──────────────────────────────────────────────────
// residual[row] = input[row]; output[row] = RmsNorm(input[row], weight)

__global__ void batched_rms_norm_residual(float* output, float* residual, const float* input,
                                           const float* weight, int n, int M, float eps)
{
    extern __shared__ float sdata[];
    int row = blockIdx.x;
    if (row >= M) return;
    int tid = threadIdx.x;
    int stride = blockDim.x;
    int off = row * n;

    float sum = 0.0f;
    for (int i = tid; i < n; i += stride)
    {
        float v = input[off + i];
        residual[off + i] = v;
        sum += v * v;
    }
    sdata[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) { if (tid < s) sdata[tid] += sdata[tid + s]; __syncthreads(); }
    float inv_rms = 1.0f / sqrtf(sdata[0] / (float)n + eps);
    for (int i = tid; i < n; i += stride)
        output[off + i] = input[off + i] * inv_rms * weight[i];
}

// ── Batched AddRmsNormResidual ──────────────────────────────────────────────
// hidden[row] += b[row]; residual[row] = hidden[row]; output[row] = RmsNorm(hidden[row], weight)

__global__ void batched_add_rms_norm_residual(float* output, float* hidden, float* residual,
                                               const float* b, const float* weight,
                                               int n, int M, float eps)
{
    extern __shared__ float sdata[];
    int row = blockIdx.x;
    if (row >= M) return;
    int tid = threadIdx.x;
    int stride = blockDim.x;
    int off = row * n;

    float sum = 0.0f;
    for (int i = tid; i < n; i += stride)
    {
        float v = hidden[off + i] + b[off + i];
        hidden[off + i] = v;
        residual[off + i] = v;
        sum += v * v;
    }
    sdata[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) { if (tid < s) sdata[tid] += sdata[tid + s]; __syncthreads(); }
    float inv_rms = 1.0f / sqrtf(sdata[0] / (float)n + eps);
    for (int i = tid; i < n; i += stride)
        output[off + i] = hidden[off + i] * inv_rms * weight[i];
}

// ── Batched AddRmsNorm ──────────────────────────────────────────────────────
// hidden[row] = a[row] + b[row]; output[row] = RmsNorm(hidden[row], weight)

__global__ void batched_add_rms_norm(float* output, float* hidden, const float* a, const float* b,
                                      const float* weight, int n, int M, float eps)
{
    extern __shared__ float sdata[];
    int row = blockIdx.x;
    if (row >= M) return;
    int tid = threadIdx.x;
    int stride = blockDim.x;
    int off = row * n;

    float sum = 0.0f;
    for (int i = tid; i < n; i += stride)
    {
        float v = a[off + i] + b[off + i];
        hidden[off + i] = v;
        sum += v * v;
    }
    sdata[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) { if (tid < s) sdata[tid] += sdata[tid + s]; __syncthreads(); }
    float inv_rms = 1.0f / sqrtf(sdata[0] / (float)n + eps);
    for (int i = tid; i < n; i += stride)
        output[off + i] = hidden[off + i] * inv_rms * weight[i];
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
            double freq = 1.0 / pow((double)theta, (double)(2 * pair) / (double)rope_dim);
            double angle = (double)position * freq;
            float cos_a = (float)cos(angle);
            float sin_a = (float)sin(angle);

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
            double freq = 1.0 / pow((double)theta, (double)(2 * pair) / (double)rope_dim);
            double angle = (double)position * freq;
            float cos_a = (float)cos(angle);
            float sin_a = (float)sin(angle);

            int base_idx = head * head_dim + pair * 2;
            float v0 = k[base_idx];
            float v1 = k[base_idx + 1];
            k[base_idx]     = v0 * cos_a - v1 * sin_a;
            k[base_idx + 1] = v0 * sin_a + v1 * cos_a;
        }
    }
}

// ── Batched RoPE ─────────────────────────────────────────────────────────────
// Apply rotary position embedding to M tokens with positions [startPos..startPos+M-1].
// q: [M * nHeads * headDim], k: [M * nKvHeads * headDim]
// Each token's heads are contiguous: [token0_head0..token0_headN, token1_head0..]

__global__ void batched_rope(float* q, float* k,
                             int q_total, int k_total,
                             int head_dim, int rope_dim,
                             int start_position, float theta,
                             int q_heads_per_token, int k_heads_per_token)
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
            int token = head / q_heads_per_token;
            int position = start_position + token;
            double freq = 1.0 / pow((double)theta, (double)(2 * pair) / (double)rope_dim);
            double angle = (double)position * freq;
            float cos_a = (float)cos(angle);
            float sin_a = (float)sin(angle);

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
            int token = head / k_heads_per_token;
            int position = start_position + token;
            double freq = 1.0 / pow((double)theta, (double)(2 * pair) / (double)rope_dim);
            double angle = (double)position * freq;
            float cos_a = (float)cos(angle);
            float sin_a = (float)sin(angle);

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

// BF16 embedding lookup: convert one row from BF16 to F32.
// BF16: 2 bytes per element, sign(1) + exp(8) + mantissa(7).

__global__ void embedding_lookup_bf16(float* output, const unsigned short* table,
                                       int hidden_dim, int token_id)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= hidden_dim) return;
    unsigned short bf = table[(long long)token_id * hidden_dim + idx];
    // BF16 → F32: just shift left by 16 (BF16 is truncated F32)
    unsigned int f32_bits = (unsigned int)bf << 16;
    output[idx] = *reinterpret_cast<float*>(&f32_bits);
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

// ── Q4_K embedding lookup (original 144-byte layout) ─────────────────────────
// Q4_K super block: 256 elements, 144 bytes.
// Layout: 2b d(f16) + 2b dmin(f16) + 12b packed scales/mins + 128b nibbles.

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

    float d = fp16_to_fp32_emb(((unsigned short)sb[1] << 8) | sb[0]);
    float dmin = fp16_to_fp32_emb(((unsigned short)sb[3] << 8) | sb[2]);

    // Chunk layout: 4 chunks of 64 elements, 32 qs bytes each.
    // lo nibble -> first 32 elems, hi nibble -> second 32 elems.
    int chunk = elem_in_sb / 64;
    int pos_in_chunk = elem_in_sb % 64;
    int l = pos_in_chunk % 32;
    int is_hi = pos_in_chunk / 32;

    int j = chunk * 2 + is_hi;

    // Unpack scales from packed 12-byte format
    int sc, m;
    if (j < 4) {
        sc = sb[4 + j] & 63;
        m  = sb[4 + j + 4] & 63;
    } else {
        sc = (sb[4 + j + 4] & 0xF) | ((sb[4 + j - 4] >> 6) << 4);
        m  = (sb[4 + j + 4] >> 4)  | ((sb[4 + j]     >> 6) << 4);
    }
    float ss = d * (float)sc;
    float sm = dmin * (float)m;

    // Nibbles at offset 16 within the 144-byte super-block
    unsigned char packed = sb[16 + chunk * 32 + l];
    int nibble = is_hi ? (packed >> 4) : (packed & 0xF);

    output[idx] = ss * (float)nibble - sm;
}

// ── Q4_0 embedding lookup ─────────────────────────────────────────────────────
// Flexible: handles both 18-byte and 20-byte (aligned) blocks via stride/offset params.

__global__ void embedding_lookup_q4_0_v2(float* output, const unsigned char* table,
                                          int hidden_dim, int token_id,
                                          int block_stride, int nibble_offset)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= hidden_dim) return;

    int blocks_per_row = hidden_dim / 32;
    int bytes_per_row = blocks_per_row * block_stride;
    const unsigned char* row = table + (long long)token_id * bytes_per_row;

    int block_idx = idx / 32;
    int elem_in_block = idx % 32;
    const unsigned char* blk = row + block_idx * block_stride;

    unsigned short scale_bits = ((unsigned short)blk[1] << 8) | blk[0];
    float scale = fp16_to_fp32_emb(scale_bits);

    int byte_idx = elem_in_block < 16 ? elem_in_block : (elem_in_block - 16);
    unsigned char packed = blk[nibble_offset + byte_idx];
    int nibble = (elem_in_block < 16) ? (packed & 0xF) : (packed >> 4);
    output[idx] = scale * (float)(nibble - 8);
}

// ── Q4_1 embedding lookup ─────────────────────────────────────────────────────
// Q4_1 block: 32 elements, 20 bytes.
// Layout: 2b FP16 scale (d) + 2b FP16 min (m) + 16b packed nibbles.
// Same nibble layout as Q4_0, but value = d * nibble + m (no subtract 8).

__global__ void embedding_lookup_q4_1(float* output, const unsigned char* table,
                                       int hidden_dim, int token_id)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= hidden_dim) return;

    int blocks_per_row = hidden_dim / 32;
    int bytes_per_row = blocks_per_row * 20;
    const unsigned char* row = table + (long long)token_id * bytes_per_row;

    int block_idx = idx / 32;
    int elem_in_block = idx % 32;
    const unsigned char* blk = row + block_idx * 20;

    unsigned short d_bits = ((unsigned short)blk[1] << 8) | blk[0];
    unsigned short m_bits = ((unsigned short)blk[3] << 8) | blk[2];
    float d = fp16_to_fp32_emb(d_bits);
    float m = fp16_to_fp32_emb(m_bits);

    int byte_idx = elem_in_block < 16 ? elem_in_block : (elem_in_block - 16);
    unsigned char packed = blk[4 + byte_idx];
    int nibble = (elem_in_block < 16) ? (packed & 0xF) : (packed >> 4);
    output[idx] = d * (float)nibble + m;
}

// ── Q1_0 embedding lookup ─────────────────────────────────────────────────────
// Handles both Q1_0 (32 elem/block, 6 bytes) and Q1_0_g128 (128 elem/block, 18 bytes).
// Block layout: [FP16 scale (2b)] [sign bits (blockSize/8 bytes)].
// bit=1 → +scale, bit=0 → -scale.

__global__ void embedding_lookup_q1_0(float* output, const unsigned char* table,
                                       int hidden_dim, int token_id,
                                       int block_size_q, int bytes_per_block)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= hidden_dim) return;

    int blocks_per_row = hidden_dim / block_size_q;
    int bytes_per_row = blocks_per_row * bytes_per_block;
    const unsigned char* row = table + (long long)token_id * bytes_per_row;

    int block_idx = idx / block_size_q;
    int elem_in_block = idx % block_size_q;
    const unsigned char* blk = row + block_idx * bytes_per_block;

    unsigned short scale_bits = ((unsigned short)blk[1] << 8) | blk[0];
    float scale = fp16_to_fp32_emb(scale_bits);

    int bit = (blk[2 + elem_in_block / 8] >> (elem_in_block % 8)) & 1;
    output[idx] = bit ? scale : -scale;
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

// ── Fused: ElementAdd + RmsNormResidual ──────────────────────────────────────
// hidden[i] += b[i]; residual[i] = hidden[i]; output[i] = RmsNorm(hidden, weight)
// Fuses ElementAdd + CopyTensor + RmsNorm across layer boundaries.
__global__ void add_rms_norm_residual(float* output, float* hidden, float* residual,
                                       const float* b, const float* weight,
                                       int n, float eps)
{
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int stride = blockDim.x;

    float sum = 0.0f;
    for (int i = tid; i < n; i += stride)
    {
        float v = hidden[i] + b[i];
        hidden[i] = v;
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

    for (int i = tid; i < n; i += stride)
        output[i] = hidden[i] * inv_rms * weight[i];
}

// ── Fused: AddRmsNormResidual + Q8_1 Quantization ───────────────────────────
// Same as add_rms_norm_residual but also outputs Q8_1 quantized version of the
// normalized result. Eliminates separate quantize_f32_q8_1 kernel call.
// Q8_1 layout: [d(f16,2b) + sum(f16,2b) + quants(32b)] = 36 bytes per block.

__global__ void add_rms_norm_residual_q8_1(
    float* output, float* hidden, float* residual,
    const float* b, const float* weight,
    void* q8_1_out, int n, float eps)
{
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int stride = blockDim.x;

    // Pass 1: hidden += b, residual = hidden, compute sum of squares
    float sq_sum = 0.0f;
    for (int i = tid; i < n; i += stride)
    {
        float v = hidden[i] + b[i];
        hidden[i] = v;
        residual[i] = v;
        sq_sum += v * v;
    }

    sdata[tid] = sq_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float inv_rms = 1.0f / sqrtf(sdata[0] / (float)n + eps);

    // Pass 2: normalize + write output + quantize to Q8_1
    // Each thread processes elements with stride, spanning multiple Q8_1 blocks.
    // We need per-block amax and sum, which requires block-level cooperation.
    // Since blockDim.x >= n/32 (threads >= blocks), each thread handles ≤1 block.

    int num_q8_blocks = n / 32;
    unsigned char* q8_out = (unsigned char*)q8_1_out;

    // First: normalize and write output (all threads)
    for (int i = tid; i < n; i += stride)
        output[i] = hidden[i] * inv_rms * weight[i];

    __syncthreads(); // ensure all output values written

    // Then: quantize output to Q8_1 (1 thread per Q8_1 block)
    for (int blk = tid; blk < num_q8_blocks; blk += stride)
    {
        const float* sp = output + blk * 32;
        unsigned char* dp = q8_out + blk * 36;

        float amax = 0.0f;
        float sum = 0.0f;
        #pragma unroll
        for (int i = 0; i < 32; i++) {
            float v = sp[i];
            sum += v;
            float av = fabsf(v);
            if (av > amax) amax = av;
        }

        float d = amax / 127.0f;
        unsigned short d_fp16 = fp32_to_fp16(d);
        unsigned short s_fp16 = fp32_to_fp16(sum);
        dp[0] = d_fp16 & 0xFF; dp[1] = d_fp16 >> 8;
        dp[2] = s_fp16 & 0xFF; dp[3] = s_fp16 >> 8;

        signed char* qs = (signed char*)(dp + 4);
        #pragma unroll
        for (int i = 0; i < 32; i++)
            qs[i] = (amax == 0.0f) ? 0 : (signed char)__float2int_rn(sp[i] / d);
    }
}

// ── Fused: RmsNormResidual + Q8_1 Quantization ─────────────────────────────
__global__ void rms_norm_residual_q8_1(
    float* output, float* residual, const float* input,
    const float* weight, void* q8_1_out, int n, float eps)
{
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int stride = blockDim.x;

    // Pass 1: copy to residual + compute sum of squares
    float sq_sum = 0.0f;
    for (int i = tid; i < n; i += stride)
    {
        float v = input[i];
        residual[i] = v;
        sq_sum += v * v;
    }

    sdata[tid] = sq_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float inv_rms = 1.0f / sqrtf(sdata[0] / (float)n + eps);

    // Pass 2: normalize + write output
    for (int i = tid; i < n; i += stride)
        output[i] = input[i] * inv_rms * weight[i];

    __syncthreads();

    // Pass 3: quantize output to Q8_1
    int num_q8_blocks = n / 32;
    unsigned char* q8_out = (unsigned char*)q8_1_out;

    for (int blk = tid; blk < num_q8_blocks; blk += stride)
    {
        const float* sp = output + blk * 32;
        unsigned char* dp = q8_out + blk * 36;

        float amax = 0.0f;
        float sum = 0.0f;
        #pragma unroll
        for (int i = 0; i < 32; i++) {
            float v = sp[i];
            sum += v;
            float av = fabsf(v);
            if (av > amax) amax = av;
        }

        float d = amax / 127.0f;
        unsigned short d_fp16 = fp32_to_fp16(d);
        unsigned short s_fp16 = fp32_to_fp16(sum);
        dp[0] = d_fp16 & 0xFF; dp[1] = d_fp16 >> 8;
        dp[2] = s_fp16 & 0xFF; dp[3] = s_fp16 >> 8;

        signed char* qs = (signed char*)(dp + 4);
        #pragma unroll
        for (int i = 0; i < 32; i++)
            qs[i] = (amax == 0.0f) ? 0 : (signed char)__float2int_rn(sp[i] / d);
    }
}

// ── Fused: AddRmsNorm + Q8_1 Quantization ──────────────────────────────────
__global__ void add_rms_norm_q8_1(
    float* output, float* hidden, const float* a, const float* b,
    const float* weight, void* q8_1_out, int n, float eps)
{
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int stride = blockDim.x;

    float sq_sum = 0.0f;
    for (int i = tid; i < n; i += stride)
    {
        float v = a[i] + b[i];
        hidden[i] = v;
        sq_sum += v * v;
    }

    sdata[tid] = sq_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float inv_rms = 1.0f / sqrtf(sdata[0] / (float)n + eps);

    for (int i = tid; i < n; i += stride)
        output[i] = hidden[i] * inv_rms * weight[i];

    __syncthreads();

    int num_q8_blocks = n / 32;
    unsigned char* q8_out = (unsigned char*)q8_1_out;

    for (int blk = tid; blk < num_q8_blocks; blk += stride)
    {
        const float* sp = output + blk * 32;
        unsigned char* dp = q8_out + blk * 36;

        float amax = 0.0f;
        float sum = 0.0f;
        #pragma unroll
        for (int i = 0; i < 32; i++) {
            float v = sp[i];
            sum += v;
            float av = fabsf(v);
            if (av > amax) amax = av;
        }

        float d = amax / 127.0f;
        unsigned short d_fp16 = fp32_to_fp16(d);
        unsigned short s_fp16 = fp32_to_fp16(sum);
        dp[0] = d_fp16 & 0xFF; dp[1] = d_fp16 >> 8;
        dp[2] = s_fp16 & 0xFF; dp[3] = s_fp16 >> 8;

        signed char* qs = (signed char*)(dp + 4);
        #pragma unroll
        for (int i = 0; i < 32; i++)
            qs[i] = (amax == 0.0f) ? 0 : (signed char)__float2int_rn(sp[i] / d);
    }
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

// ── Fused: SwiGLU + Q8_1 quantization ───────────────────────────────────────
// Computes SwiGLU and quantizes output to Q8_1 in one pass.
// Grid: ceil(n/256) blocks, 256 threads per block.
// Each thread computes one SwiGLU element and contributes to Q8_1 block stats.

__global__ void swiglu_q8_1(float* output, void* __restrict__ q8_1_dst,
                             const float* gate, const float* up, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Step 1: Compute SwiGLU and write to output
    float v = 0.0f;
    if (idx < n)
    {
        float g = gate[idx];
        v = (g / (1.0f + expf(-g))) * up[idx];
        output[idx] = v;
    }

    // Step 2: Quantize to Q8_1 blocks.
    // Each Q8_1 block has 32 elements. Thread lane within a warp maps to block element.
    int lane = threadIdx.x & 31;
    int blk_id = idx / 32;
    int num_blocks = n / 32;

    if (blk_id < num_blocks)
    {
        // Warp-level reduction for amax and sum (all 32 lanes have their element)
        float av = fabsf(v);
        float warp_max = av;
        float warp_sum = v;
        for (int offset = 16; offset > 0; offset >>= 1) {
            warp_max = fmaxf(warp_max, __shfl_down_sync(0xffffffff, warp_max, offset));
            warp_sum += __shfl_down_sync(0xffffffff, warp_sum, offset);
        }

        // Broadcast amax back to all lanes
        float amax = __shfl_sync(0xffffffff, warp_max, 0);
        float d = amax / 127.0f;

        // Each lane quantizes its own element
        signed char q = (amax == 0.0f) ? 0 : (signed char)__float2int_rn(v / d);

        // Lane 0 writes the header, all lanes write their quant
        unsigned char* dp = (unsigned char*)q8_1_dst + blk_id * 36;
        if (lane == 0) {
            unsigned short d_fp16 = fp32_to_fp16(d);
            float sum = __shfl_sync(0xffffffff, warp_sum, 0);
            unsigned short s_fp16 = fp32_to_fp16(sum);
            dp[0] = d_fp16 & 0xFF; dp[1] = d_fp16 >> 8;
            dp[2] = s_fp16 & 0xFF; dp[3] = s_fp16 >> 8;
        }
        ((signed char*)(dp + 4))[lane] = q;
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

// ── Dequantize weight tensor to FP32 (for batch matmul) ─────────────────────
// Generic kernel that handles Q8_0, Q4_0, Q4_K, Q6_K, F16 etc.
// Each thread dequantizes one element.

__device__ float fp16_to_fp32_dq(unsigned short h)
{
    float result;
    asm("cvt.f32.f16 %0, %1;" : "=f"(result) : "h"(h));
    return result;
}

__global__ void dequant_to_f32(float* __restrict__ output,
                                const unsigned char* __restrict__ input,
                                int totalElements, int typeTag, int blockSizeQ,
                                int isAligned)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalElements) return;

    int blk = idx / blockSizeQ;
    int elem = idx % blockSizeQ;

    float val = 0.0f;

    if (typeTag == 8 && isAligned) { // Q8_0 aligned: 36 bytes per 32 elements (scale+pad+quants)
        const unsigned char* bp = input + blk * 36;
        float scale = fp16_to_fp32_dq(((unsigned short)bp[1] << 8) | bp[0]);
        val = scale * (float)((signed char)bp[4 + elem]);
    }
    else if (typeTag == 8) { // Q8_0 unaligned: 34 bytes per 32 elements (scale+quants)
        const unsigned char* bp = input + blk * 34;
        float scale = fp16_to_fp32_dq(((unsigned short)bp[1] << 8) | bp[0]);
        val = scale * (float)((signed char)bp[2 + elem]);
    }
    else if (typeTag == 2 && isAligned) { // Q4_0 aligned: 20 bytes per 32 elements (scale+pad+nibbles)
        const unsigned char* bp = input + blk * 20;
        float scale = fp16_to_fp32_dq(((unsigned short)bp[1] << 8) | bp[0]);
        if (elem < 16) {
            val = scale * (float)((int)(bp[4 + elem] & 0xF) - 8);
        } else {
            val = scale * (float)((int)(bp[4 + elem - 16] >> 4) - 8);
        }
    }
    else if (typeTag == 2) { // Q4_0 unaligned: 18 bytes per 32 elements (scale+nibbles)
        const unsigned char* bp = input + blk * 18;
        float scale = fp16_to_fp32_dq(((unsigned short)bp[1] << 8) | bp[0]);
        if (elem < 16) {
            val = scale * (float)((int)(bp[2 + elem] & 0xF) - 8);
        } else {
            val = scale * (float)((int)(bp[2 + elem - 16] >> 4) - 8);
        }
    }
    else if (typeTag == 1) { // F16: 2 bytes per element
        const unsigned char* bp = input + idx * 2;
        val = fp16_to_fp32_dq(((unsigned short)bp[1] << 8) | bp[0]);
    }
    else if (typeTag == 12) { // Q4_K: 144 bytes per 256 elements
        const unsigned char* sbp = input + blk * 144;
        float d = fp16_to_fp32_dq(((unsigned short)sbp[1] << 8) | sbp[0]);
        float dmin = fp16_to_fp32_dq(((unsigned short)sbp[3] << 8) | sbp[2]);
        int chunk = elem / 64;
        int pos = elem % 64;
        int l = pos % 32;
        int is_hi = pos / 32;
        int j = chunk * 2 + is_hi;
        int sc, m;
        if (j < 4) { sc = sbp[4+j] & 63; m = sbp[8+j] & 63; }
        else { sc = (sbp[4+j+4]&0xF)|((sbp[4+j-4]>>6)<<4); m = (sbp[4+j+4]>>4)|((sbp[4+j]>>6)<<4); }
        unsigned char packed = sbp[16 + chunk * 32 + l];
        int nibble = is_hi ? (packed >> 4) : (packed & 0xF);
        val = d * (float)sc * (float)nibble - dmin * (float)m;
    }
    else if (typeTag == 14) { // Q6_K: 210 bytes per 256 elements
        const unsigned char* sbp = input + blk * 210;
        float d = fp16_to_fp32_dq(((unsigned short)sbp[209] << 8) | sbp[208]);
        int half = elem / 128;
        int rem = elem % 128;
        int quadrant = rem / 32;
        int l = rem % 32;
        int qlOff = half * 64 + (quadrant & 1) * 32;
        int qlShift = (quadrant >> 1) * 4;
        int qhShift = quadrant * 2;
        int q = (((sbp[qlOff + l] >> qlShift) & 0xF) | (((sbp[128 + half*32 + l] >> qhShift) & 3) << 4)) - 32;
        float sc = d * (float)((signed char)sbp[192 + half*8 + quadrant*2 + l/16]);
        val = sc * (float)q;
    }
    else if (typeTag == 30) { // BF16: 2 bytes per element, shift left 16 to get F32
        const unsigned char* bp = input + idx * 2;
        unsigned int f32_bits = ((unsigned int)bp[1] << 24) | ((unsigned int)bp[0] << 16);
        val = *reinterpret_cast<float*>(&f32_bits);
    }
    else if (typeTag == 40) { // Q1_0: 6 bytes per 32 elements (FP16 scale + 4 sign bytes)
        const unsigned char* bp = input + blk * 6;
        float scale = fp16_to_fp32_dq(((unsigned short)bp[1] << 8) | bp[0]);
        int bit = (bp[2 + elem/8] >> (elem%8)) & 1;
        val = bit ? scale : -scale;
    }
    else if (typeTag == 41) { // Q1_0_g128: 18 bytes per 128 elements (FP16 scale + 16 sign bytes)
        const unsigned char* bp = input + blk * 18;
        float scale = fp16_to_fp32_dq(((unsigned short)bp[1] << 8) | bp[0]);
        int bit = (bp[2 + elem/8] >> (elem%8)) & 1;
        val = bit ? scale : -scale;
    }

    output[idx] = val;
}

// ── FP32 → FP16 batch conversion ────────────────────────────────────────────
__global__ void convert_f32_to_f16(unsigned short* output, const float* input, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        output[idx] = fp32_to_fp16(input[idx]);
}

// ── Dequantize weight tensor to FP16 (for tensor core batch matmul) ─────────

__global__ void dequant_to_f16(unsigned short* __restrict__ output,
                                const unsigned char* __restrict__ input,
                                int totalElements, int typeTag, int blockSizeQ,
                                int isAligned)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalElements) return;

    int blk = idx / blockSizeQ;
    int elem = idx % blockSizeQ;

    float val = 0.0f;

    if (typeTag == 8 && isAligned) {
        const unsigned char* bp = input + blk * 36;
        float scale = fp16_to_fp32_dq(((unsigned short)bp[1] << 8) | bp[0]);
        val = scale * (float)((signed char)bp[4 + elem]);
    }
    else if (typeTag == 8) {
        const unsigned char* bp = input + blk * 34;
        float scale = fp16_to_fp32_dq(((unsigned short)bp[1] << 8) | bp[0]);
        val = scale * (float)((signed char)bp[2 + elem]);
    }
    else if (typeTag == 2 && isAligned) {
        const unsigned char* bp = input + blk * 20;
        float scale = fp16_to_fp32_dq(((unsigned short)bp[1] << 8) | bp[0]);
        if (elem < 16) val = scale * (float)((int)(bp[4 + elem] & 0xF) - 8);
        else val = scale * (float)((int)(bp[4 + elem - 16] >> 4) - 8);
    }
    else if (typeTag == 2) {
        const unsigned char* bp = input + blk * 18;
        float scale = fp16_to_fp32_dq(((unsigned short)bp[1] << 8) | bp[0]);
        if (elem < 16) val = scale * (float)((int)(bp[2 + elem] & 0xF) - 8);
        else val = scale * (float)((int)(bp[2 + elem - 16] >> 4) - 8);
    }
    else if (typeTag == 1) {
        output[idx] = ((const unsigned short*)input)[idx];
        return;
    }
    else if (typeTag == 12) { // Q4_K
        const unsigned char* sbp = input + blk * 144;
        float d = fp16_to_fp32_dq(((unsigned short)sbp[1] << 8) | sbp[0]);
        float dmin = fp16_to_fp32_dq(((unsigned short)sbp[3] << 8) | sbp[2]);
        int subBlock = elem / 32;
        int subElem = elem % 32;
        unsigned char rawScale = sbp[4 + subBlock];
        float sc, mn;
        if (subBlock < 4) {
            sc = (float)(rawScale & 0x3F);
            mn = (float)(sbp[4 + subBlock + 4] & 0x3F);
        } else {
            unsigned char low = rawScale;
            unsigned char highBits = sbp[4 + subBlock - 4];
            sc = (float)((low & 0x3F) | ((highBits >> 4) & 0x0C));
            mn = (float)(((low >> 4) & 0x0F) | ((sbp[4 + subBlock] >> 2) & 0x30));
        }
        const unsigned char* qs = sbp + 16 + subBlock * 16 + (subElem < 16 ? 0 : -16);
        int nibbleIdx = subElem < 16 ? subElem : subElem - 16;
        unsigned char qByte = qs[nibbleIdx];
        int qval = subElem < 16 ? (qByte & 0xF) : (qByte >> 4);
        val = d * sc * (float)qval - dmin * mn;
    }
    else if (typeTag == 14) { // Q6_K
        const unsigned char* sbp = input + blk * 210;
        int subBlock = elem / 16;
        int subElem = elem % 16;
        float d = fp16_to_fp32_dq(((unsigned short)sbp[209] << 8) | sbp[208]);
        signed char sc = ((const signed char*)sbp)[192 + subBlock];
        const unsigned char* ql = sbp + subBlock * 16;
        const unsigned char* qh = sbp + 128 + subBlock * 8;
        int low, high;
        if (subElem < 8) { low = ql[subElem] & 0xF; high = (qh[subElem] & 3) << 4; }
        else { low = ql[subElem - 8] >> 4; high = ((qh[subElem - 8] >> 2) & 3) << 4; }
        int qval = low | high;
        val = d * (float)sc * ((float)qval - 32.0f);
    }
    else if (typeTag == 30) { // BF16
        const unsigned char* bp = input + idx * 2;
        unsigned int f32_bits = ((unsigned int)bp[1] << 24) | ((unsigned int)bp[0] << 16);
        val = *reinterpret_cast<float*>(&f32_bits);
    }
    else if (typeTag == 40) { // Q1_0
        const unsigned char* bp = input + blk * 6;
        float scale = fp16_to_fp32_dq(((unsigned short)bp[1] << 8) | bp[0]);
        int bit = (bp[2 + elem/8] >> (elem%8)) & 1;
        val = bit ? scale : -scale;
    }
    else if (typeTag == 41) { // Q1_0_g128
        const unsigned char* bp = input + blk * 18;
        float scale = fp16_to_fp32_dq(((unsigned short)bp[1] << 8) | bp[0]);
        int bit = (bp[2 + elem/8] >> (elem%8)) & 1;
        val = bit ? scale : -scale;
    }

    output[idx] = fp32_to_fp16(val);
}

} // extern "C"
