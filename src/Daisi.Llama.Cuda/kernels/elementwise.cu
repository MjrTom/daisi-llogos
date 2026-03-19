// daisi-llama CUDA kernels: element-wise operations
// Compiled to .cubin and embedded as assembly resources.

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

} // extern "C"
