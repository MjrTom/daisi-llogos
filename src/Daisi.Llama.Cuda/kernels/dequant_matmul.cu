// daisi-llama CUDA kernels: fused dequantization + matrix multiplication
// Compiled to .cubin and embedded as assembly resources.
//
// Convention: output[M×N] = a[1×K] × b^T[N×K]
// For inference, M=1 (single token), so this is really a batched dot product.
// b is stored in GGUF layout [N × K] (each row is one output neuron's weights).

extern "C" {

// ── FP32 MatMul (M=1 vector × matrix) ───────────────────────────────────────
// output[n] = dot(a[K], b[n*K .. (n+1)*K])

__global__ void matmul_f32(float* output, const float* a, const float* b,
                           int M, int K, int N)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;

    for (int m = 0; m < M; m++)
    {
        float sum = 0.0f;
        const float* a_row = a + m * K;
        const float* b_row = b + n * K;
        for (int k = 0; k < K; k++)
            sum += a_row[k] * b_row[k];
        output[m * N + n] = sum;
    }
}

// ── Fused Q8_0 Dequant + MatMul ─────────────────────────────────────────────
// b is Q8_0 quantized: each block is 34 bytes (2-byte FP16 scale + 32 int8 quants).
// b layout: [N rows × (K/32 blocks per row)]
// output[n] = dot(a[K], dequant(b_row_n))

__global__ void dequant_matmul_q8_0(float* output, const float* a,
                                     const unsigned char* b,
                                     int M, int K, int N)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;

    int blocks_per_row = K / 32;
    int bytes_per_row = blocks_per_row * 34;

    for (int m = 0; m < M; m++)
    {
        float sum = 0.0f;
        const float* a_row = a + m * K;
        const unsigned char* b_row = b + n * bytes_per_row;

        for (int blk = 0; blk < blocks_per_row; blk++)
        {
            const unsigned char* block = b_row + blk * 34;

            // Decode FP16 scale
            unsigned short scale_bits = ((unsigned short)block[1] << 8) | block[0];
            int sign = (scale_bits >> 15) & 1;
            int exp_val = (scale_bits >> 10) & 0x1f;
            int mant = scale_bits & 0x3ff;
            float scale;
            if (exp_val == 0)
                scale = (sign ? -1.0f : 1.0f) * (mant / 1024.0f) * (1.0f / 16384.0f);
            else if (exp_val == 31)
                scale = 0.0f;
            else
                scale = (sign ? -1.0f : 1.0f) * (1.0f + mant / 1024.0f) * powf(2.0f, (float)(exp_val - 15));

            const signed char* quants = (const signed char*)(block + 2);
            int a_offset = blk * 32;

            // Accumulate dot product for this block
            float block_sum = 0.0f;
            for (int i = 0; i < 32; i++)
                block_sum += a_row[a_offset + i] * (float)quants[i];

            sum += scale * block_sum;
        }

        output[m * N + n] = sum;
    }
}

} // extern "C"
