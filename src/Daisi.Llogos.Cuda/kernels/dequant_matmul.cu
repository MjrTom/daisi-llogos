// daisi-llogos CUDA kernels: fused dequantization + matrix multiplication
// Compiled to PTX via NVRTC at runtime.
//
// Convention: output[M×N] = a[1×K] × b^T[N×K]
// For inference, M=1 (single token), so this is a batched dot product.
// b is stored in GGUF layout [N × K] (each row is one output neuron's weights).
//
// Architecture: One block per output neuron. Threads cooperatively compute
// the dot product with warp-level reduction. This maximizes SM occupancy.

// FP16 → FP32 conversion using bit manipulation
__device__ float fp16_to_fp32(unsigned short h)
{
    unsigned int sign = (h >> 15) & 1;
    unsigned int exp_val = (h >> 10) & 0x1f;
    unsigned int mant = h & 0x3ff;

    unsigned int f;
    if (exp_val == 0)
    {
        if (mant == 0)
            f = sign << 31;
        else
        {
            exp_val = 1;
            while (!(mant & 0x400)) { mant <<= 1; exp_val--; }
            mant &= 0x3ff;
            f = (sign << 31) | ((exp_val + 127 - 15) << 23) | (mant << 13);
        }
    }
    else if (exp_val == 31)
        f = (sign << 31) | 0x7f800000 | (mant << 13);
    else
        f = (sign << 31) | ((exp_val - 15 + 127) << 23) | (mant << 13);

    return *reinterpret_cast<float*>(&f);
}

// Warp-level reduction using shuffle
__device__ float warp_reduce_sum(float val)
{
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

extern "C" {

// ── FP32 MatMul (M=1 vector × matrix) ───────────────────────────────────────
// One block per output neuron. Threads cooperatively compute dot product.
// Grid: N blocks, Block: 256 threads.

__global__ void matmul_f32(float* output, const float* a, const float* b,
                           int M, int K, int N)
{
    extern __shared__ float shared[];
    int n = blockIdx.x;
    int tid = threadIdx.x;
    int blockSize = blockDim.x;

    if (n >= N) return;

    const float* b_row = b + (long)n * K;

    // Each thread computes partial sum
    float sum = 0.0f;
    for (int k = tid; k < K; k += blockSize)
        sum += a[k] * b_row[k];

    // Warp reduction
    sum = warp_reduce_sum(sum);

    // Write warp sums to shared memory
    int lane = tid & 31;
    int warp = tid >> 5;
    if (lane == 0)
        shared[warp] = sum;
    __syncthreads();

    // Final reduction by first warp
    int numWarps = blockSize >> 5;
    if (tid < numWarps)
        sum = shared[tid];
    else
        sum = 0.0f;
    sum = warp_reduce_sum(sum);

    if (tid == 0)
        output[n] = sum;
}

// ── Fused Q8_0 Dequant + MatMul ─────────────────────────────────────────────
// One block per output neuron. Threads cooperatively process Q8_0 blocks.
// Grid: N blocks, Block: 256 threads.
// Each thread handles K/(32*blockSize) Q8_0 blocks minimum, striding over all blocks.

__global__ void dequant_matmul_q8_0(float* output, const float* a,
                                     const unsigned char* b,
                                     int M, int K, int N)
{
    extern __shared__ float shared[];
    int n = blockIdx.x;
    int tid = threadIdx.x;
    int blockSize = blockDim.x;

    if (n >= N) return;

    int blocks_per_row = K / 32;
    long bytes_per_row = (long)blocks_per_row * 34;
    const unsigned char* b_row = b + (long)n * bytes_per_row;

    // Each thread processes elements at stride, with Q8_0 block-aligned groups
    float sum = 0.0f;

    // Process full Q8_0 blocks per thread for coalesced access
    for (int blk = tid; blk < blocks_per_row; blk += blockSize)
    {
        const unsigned char* block_ptr = b_row + blk * 34;

        // FP16 → FP32 scale
        unsigned short scale_bits = ((unsigned short)block_ptr[1] << 8) | block_ptr[0];
        float scale = fp16_to_fp32(scale_bits);

        const signed char* quants = (const signed char*)(block_ptr + 2);
        const float* a_ptr = a + blk * 32;

        // Dot product for 32 quants (unrolled)
        float block_sum = 0.0f;
        block_sum += a_ptr[0] * (float)quants[0];
        block_sum += a_ptr[1] * (float)quants[1];
        block_sum += a_ptr[2] * (float)quants[2];
        block_sum += a_ptr[3] * (float)quants[3];
        block_sum += a_ptr[4] * (float)quants[4];
        block_sum += a_ptr[5] * (float)quants[5];
        block_sum += a_ptr[6] * (float)quants[6];
        block_sum += a_ptr[7] * (float)quants[7];
        block_sum += a_ptr[8] * (float)quants[8];
        block_sum += a_ptr[9] * (float)quants[9];
        block_sum += a_ptr[10] * (float)quants[10];
        block_sum += a_ptr[11] * (float)quants[11];
        block_sum += a_ptr[12] * (float)quants[12];
        block_sum += a_ptr[13] * (float)quants[13];
        block_sum += a_ptr[14] * (float)quants[14];
        block_sum += a_ptr[15] * (float)quants[15];
        block_sum += a_ptr[16] * (float)quants[16];
        block_sum += a_ptr[17] * (float)quants[17];
        block_sum += a_ptr[18] * (float)quants[18];
        block_sum += a_ptr[19] * (float)quants[19];
        block_sum += a_ptr[20] * (float)quants[20];
        block_sum += a_ptr[21] * (float)quants[21];
        block_sum += a_ptr[22] * (float)quants[22];
        block_sum += a_ptr[23] * (float)quants[23];
        block_sum += a_ptr[24] * (float)quants[24];
        block_sum += a_ptr[25] * (float)quants[25];
        block_sum += a_ptr[26] * (float)quants[26];
        block_sum += a_ptr[27] * (float)quants[27];
        block_sum += a_ptr[28] * (float)quants[28];
        block_sum += a_ptr[29] * (float)quants[29];
        block_sum += a_ptr[30] * (float)quants[30];
        block_sum += a_ptr[31] * (float)quants[31];

        sum += scale * block_sum;
    }

    // Warp reduction
    sum = warp_reduce_sum(sum);

    // Write warp sums to shared memory
    int lane = tid & 31;
    int warp = tid >> 5;
    if (lane == 0)
        shared[warp] = sum;
    __syncthreads();

    // Final reduction by first warp
    int numWarps = blockSize >> 5;
    if (tid < numWarps)
        sum = shared[tid];
    else
        sum = 0.0f;
    sum = warp_reduce_sum(sum);

    if (tid == 0)
        output[n] = sum;
}

} // extern "C"
