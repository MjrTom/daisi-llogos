// daisi-llogos CUDA kernels: fused dequantization + matrix multiplication
// Compiled to PTX via NVRTC at runtime.
//
// Convention: output[M×N] = a[1×K] × b^T[N×K]
// For inference, M=1 (single token), so this is a batched dot product.
// b is stored in GGUF layout [N × K] (each row is one output neuron's weights).
//
// Architecture: One block per output neuron. Threads cooperatively compute
// the dot product with warp-level reduction. This maximizes SM occupancy.

// FP16 → FP32 conversion via PTX cvt instruction (single hardware instruction, no cuda_fp16.h needed)
__device__ __forceinline__ float fp16_to_fp32(unsigned short h)
{
    float result;
    asm("cvt.f32.f16 %0, %1;" : "=f"(result) : "h"(h));
    return result;
}

// Warp-level reduction using shuffle
__device__ __forceinline__ float warp_reduce_sum(float val)
{
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// FP32 → FP16 conversion via PTX
__device__ __forceinline__ unsigned short matmul_fp32_to_fp16(float val)
{
    unsigned short result;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(result) : "f"(val));
    return result;
}

#if 0 // Old software fallback (replaced by PTX)
__device__ unsigned short matmul_fp32_to_fp16_sw(float val)
{
    unsigned int f = __float_as_uint(val);
    unsigned int sign = (f >> 16) & 0x8000;
    int exp_val = ((f >> 23) & 0xff) - 127 + 15;
    unsigned int mant = (f >> 13) & 0x3ff;
    if (exp_val <= 0) return (unsigned short)sign;
    if (exp_val >= 31) return (unsigned short)(sign | 0x7c00);
    return (unsigned short)(sign | (exp_val << 10) | mant);
}
#endif

extern "C" {

// ── Quantize FP32 activation → Q8_1 blocks for __dp4a dot products ──────────
// Q8_1 layout: d(fp16, 2b) + s(fp16, 2b) + qs(32 × int8) = 36 bytes per block
// Each thread processes one 32-element block.

__global__ void quantize_f32_q8_1(void* __restrict__ dst,
                                   const float* __restrict__ src, int K)
{
    int blk = blockIdx.x * blockDim.x + threadIdx.x;
    int num_blocks = K / 32;
    if (blk >= num_blocks) return;

    const float* sp = src + blk * 32;
    unsigned char* dp = (unsigned char*)dst + blk * 36;

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
    float id = d > 0.0f ? 127.0f / amax : 0.0f;

    unsigned short d_fp16 = matmul_fp32_to_fp16(d);
    unsigned short s_fp16 = matmul_fp32_to_fp16(sum);
    dp[0] = d_fp16 & 0xFF; dp[1] = d_fp16 >> 8;
    dp[2] = s_fp16 & 0xFF; dp[3] = s_fp16 >> 8;

    signed char* qs = (signed char*)(dp + 4);
    #pragma unroll
    for (int i = 0; i < 32; i++)
        qs[i] = (signed char)__float2int_rn(sp[i] * id);
}

// ── Q8_0 × Q8_1 MatMul using __dp4a (integer dot product hardware) ──────────
// Activation pre-quantized to Q8_1. Weight is Q8_0.
// __dp4a does 4 × int8*int8 multiply-adds in 1 instruction = 4× throughput.
// Grid: N blocks, Block: adaptive threads.

// Aligned Q8_0 variant: weights repacked to 36-byte blocks [scale(2) + pad(2) + quants(32)]
// Quants are 4-byte aligned, enabling direct int* loads for __dp4a.
__global__ void dequant_matmul_q8_0_q8_1_aligned(float* __restrict__ output,
                                                    const void* __restrict__ vq8_1,
                                                    const unsigned char* __restrict__ b,
                                                    int M, int K, int N)
{
    extern __shared__ float shared[];
    int n = blockIdx.x;
    int tid = threadIdx.x;
    int blockSize = blockDim.x;

    if (n >= N) return;

    int blocks_per_row = K / 32;
    long bytes_per_row = (long)blocks_per_row * 36;  // 36-byte aligned blocks
    const unsigned char* b_row = b + (long)n * bytes_per_row;
    const unsigned char* a_q8_1 = (const unsigned char*)vq8_1;

    float sum = 0.0f;

    for (int blk = tid; blk < blocks_per_row; blk += blockSize)
    {
        // Aligned Q8_0: scale at +0, quants at +4 (4-byte aligned!)
        const unsigned char* wblk = b_row + blk * 36;
        float w_scale = fp16_to_fp32(*reinterpret_cast<const unsigned short*>(wblk));
        const int* w_qs = reinterpret_cast<const int*>(wblk + 4); // ALIGNED

        const unsigned char* ablk = a_q8_1 + blk * 36;
        float a_scale = fp16_to_fp32(*reinterpret_cast<const unsigned short*>(ablk));
        const int* a_qs = reinterpret_cast<const int*>(ablk + 4); // ALIGNED

        int sumi = 0;
        sumi = __dp4a(a_qs[0], w_qs[0], sumi);
        sumi = __dp4a(a_qs[1], w_qs[1], sumi);
        sumi = __dp4a(a_qs[2], w_qs[2], sumi);
        sumi = __dp4a(a_qs[3], w_qs[3], sumi);
        sumi = __dp4a(a_qs[4], w_qs[4], sumi);
        sumi = __dp4a(a_qs[5], w_qs[5], sumi);
        sumi = __dp4a(a_qs[6], w_qs[6], sumi);
        sumi = __dp4a(a_qs[7], w_qs[7], sumi);

        sum += a_scale * w_scale * (float)sumi;
    }

    // Warp reduction
    sum = warp_reduce_sum(sum);
    int lane = tid & 31;
    int warp = tid >> 5;
    if (lane == 0) shared[warp] = sum;
    __syncthreads();
    int numWarps = blockSize >> 5;
    if (tid < numWarps) sum = shared[tid]; else sum = 0.0f;
    sum = warp_reduce_sum(sum);
    if (tid == 0) output[n] = sum;
}

// Unaligned Q8_0 variant (fallback for non-repacked data)
__global__ void dequant_matmul_q8_0_q8_1(float* __restrict__ output,
                                           const void* __restrict__ vq8_1,
                                           const unsigned char* __restrict__ b,
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
    const unsigned char* a_q8_1 = (const unsigned char*)vq8_1;

    float sum = 0.0f;

    for (int blk = tid; blk < blocks_per_row; blk += blockSize)
    {
        const unsigned char* wblk = b_row + blk * 34;
        float w_scale = fp16_to_fp32(wblk[0] | ((unsigned short)wblk[1] << 8));

        const unsigned char* ablk = a_q8_1 + blk * 36;
        float a_scale = fp16_to_fp32(ablk[0] | ((unsigned short)ablk[1] << 8));
        const int* a_qs = reinterpret_cast<const int*>(ablk + 4);

        int w_qs[8];
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            const unsigned char* src = wblk + 2 + i * 4;
            w_qs[i] = src[0] | (src[1] << 8) | (src[2] << 16) | (src[3] << 24);
        }

        int sumi = 0;
        sumi = __dp4a(a_qs[0], w_qs[0], sumi);
        sumi = __dp4a(a_qs[1], w_qs[1], sumi);
        sumi = __dp4a(a_qs[2], w_qs[2], sumi);
        sumi = __dp4a(a_qs[3], w_qs[3], sumi);
        sumi = __dp4a(a_qs[4], w_qs[4], sumi);
        sumi = __dp4a(a_qs[5], w_qs[5], sumi);
        sumi = __dp4a(a_qs[6], w_qs[6], sumi);
        sumi = __dp4a(a_qs[7], w_qs[7], sumi);

        sum += a_scale * w_scale * (float)sumi;
    }

    // Warp reduction
    sum = warp_reduce_sum(sum);
    int lane = tid & 31;
    int warp = tid >> 5;
    if (lane == 0) shared[warp] = sum;
    __syncthreads();

    int numWarps = blockSize >> 5;
    if (tid < numWarps) sum = shared[tid]; else sum = 0.0f;
    sum = warp_reduce_sum(sum);

    if (tid == 0) output[n] = sum;
}

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

// ── Fused Q8_0 Dequant + MatMul (aligned 36-byte blocks) ────────────────────
// Multi-row constant (shared by aligned and unaligned Q8_0 kernels)
#ifndef Q8_ROWS_PER_BLOCK
#define Q8_ROWS_PER_BLOCK 4
#endif

// Multi-row aligned Q8_0: 4 output neurons per block with activation reuse
__global__ void dequant_matmul_q8_0_aligned(float* __restrict__ output,
                                              const float* __restrict__ a,
                                              const unsigned char* __restrict__ b,
                                              int M, int K, int N)
{
    extern __shared__ float shared[];
    int n_base = blockIdx.x * Q8_ROWS_PER_BLOCK;
    int tid = threadIdx.x;
    int blockSize = blockDim.x;
    if (n_base >= N) return;

    int blocks_per_row = K / 32;
    long bytes_per_row = (long)blocks_per_row * 36;

    float sums[Q8_ROWS_PER_BLOCK] = {0};

    for (int blk = tid; blk < blocks_per_row; blk += blockSize)
    {
        // Use __ldg for read-only activation loads (texture cache path)
        const float4* ap = reinterpret_cast<const float4*>(a + blk * 32);
        float4 a_cache[8];
        #pragma unroll
        for (int i = 0; i < 8; i++) a_cache[i] = __ldg(&ap[i]);

        #pragma unroll
        for (int r = 0; r < Q8_ROWS_PER_BLOCK; r++)
        {
            int n = n_base + r;
            if (n >= N) break;
            const unsigned char* block_ptr = b + (long)n * bytes_per_row + blk * 36;
            float scale = fp16_to_fp32(__ldg(reinterpret_cast<const unsigned short*>(block_ptr)));
            // Native uint reads via __ldg for read-only weight data
            const unsigned int* q32 = reinterpret_cast<const unsigned int*>(block_ptr + 4);
            float bs = 0.0f;
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                float4 ai = a_cache[i];
                unsigned int packed = __ldg(&q32[i]);
                bs += ai.x*(float)(int)((signed char)(packed))
                    + ai.y*(float)(int)((signed char)(packed >> 8))
                    + ai.z*(float)(int)((signed char)(packed >> 16))
                    + ai.w*(float)(int)((signed char)(packed >> 24));
            }
            sums[r] += scale * bs;
        }
    }

    int numWarps = blockSize >> 5;
    #pragma unroll
    for (int r = 0; r < Q8_ROWS_PER_BLOCK; r++) {
        float sum = warp_reduce_sum(sums[r]);
        int lane = tid & 31; int warp = tid >> 5;
        if (lane == 0) shared[warp] = sum;
        __syncthreads();
        if (tid < numWarps) sum = shared[tid]; else sum = 0.0f;
        sum = warp_reduce_sum(sum);
        if (tid == 0 && n_base + r < N) output[n_base + r] = sum;
        __syncthreads();
    }
}

// ── Fused Q8_0 Dequant + MatMul (original 34-byte blocks) ──────────────────
// Multi-row Q8_0: 4 output neurons per block with activation reuse
#define Q8_ROWS_PER_BLOCK 4

__global__ void dequant_matmul_q8_0(float* __restrict__ output,
                                     const float* __restrict__ a,
                                     const unsigned char* __restrict__ b,
                                     int M, int K, int N)
{
    extern __shared__ float shared[];
    int n_base = blockIdx.x * Q8_ROWS_PER_BLOCK;
    int tid = threadIdx.x;
    int blockSize = blockDim.x;

    if (n_base >= N) return;

    int blocks_per_row = K / 32;
    long bytes_per_row = (long)blocks_per_row * 34;

    // Accumulate all rows in registers (no sync between rows)
    float sums[Q8_ROWS_PER_BLOCK] = {0};

    for (int blk = tid; blk < blocks_per_row; blk += blockSize)
    {
        // Load activation via __ldg (read-only cache)
        const float4* ap = reinterpret_cast<const float4*>(a + blk * 32);
        float4 a_cache[8];
        #pragma unroll
        for (int i = 0; i < 8; i++) a_cache[i] = __ldg(&ap[i]);

        #pragma unroll
        for (int r = 0; r < Q8_ROWS_PER_BLOCK; r++)
        {
            int n = n_base + r;
            if (n >= N) break;

            const unsigned char* block_ptr = b + (long)n * bytes_per_row + blk * 34;
            float scale = fp16_to_fp32(__ldg(reinterpret_cast<const unsigned short*>(block_ptr)));
            const signed char* quants = (const signed char*)(block_ptr + 2);

            float bs = 0.0f;
            #pragma unroll
            for (int i = 0; i < 8; i++)
            {
                float4 ai = a_cache[i];
                int base = i * 4;
                bs += ai.x * (float)quants[base]     + ai.y * (float)quants[base + 1]
                    + ai.z * (float)quants[base + 2] + ai.w * (float)quants[base + 3];
            }
            sums[r] += scale * bs;
        }
    }

    // Reduce and write each row independently (no inter-row sync)
    int numWarps = blockSize >> 5;
    #pragma unroll
    for (int r = 0; r < Q8_ROWS_PER_BLOCK; r++)
    {
        float sum = warp_reduce_sum(sums[r]);
        int lane = tid & 31;
        int warp = tid >> 5;
        if (lane == 0) shared[warp] = sum;
        __syncthreads();
        if (tid < numWarps) sum = shared[tid]; else sum = 0.0f;
        sum = warp_reduce_sum(sum);
        if (tid == 0 && n_base + r < N) output[n_base + r] = sum;
        __syncthreads();
    }
}

// ── Fused I2_S (BitNet ternary) Dequant + MatMul ───────────────────────────
// I2_S: 2-bit packed, 4 values per byte, 128-element interleaved groups.
// Each byte: bits[7:6]=elem+0*32, [5:4]=elem+1*32, [3:2]=elem+2*32, [1:0]=elem+3*32.
// Encoding: 0b00=-1, 0b01=0, 0b10=+1, 0b11=0.
// Per-tensor float32 scale at byte offset (K*N/4).
// Grid: N blocks, Block: 32 threads.

__global__ void dequant_matmul_i2s(float* output, const float* a,
                                   const unsigned char* b,
                                   int M, int K, int N)
{
    extern __shared__ float shared[];
    int n = blockIdx.x;
    int tid = threadIdx.x;
    int blockSize = blockDim.x;

    if (n >= N) return;

    // Per-tensor scale at end of packed data
    long long totalPacked = (long long)K * N / 4;
    float scale = *reinterpret_cast<const float*>(b + totalPacked);

    int packedPerRow = K / 4;
    const unsigned char* b_row = b + (long long)n * packedPerRow;

    int chunks = K / 128; // 128-element groups, 32 bytes each
    float sum = 0.0f;

    for (int chunk = tid; chunk < chunks; chunk += blockSize)
    {
        const unsigned char* bp = b_row + chunk * 32;
        const float* ap = a + chunk * 128;

        for (int gp = 0; gp < 32; gp++)
        {
            unsigned char bv = bp[gp];

            int c0 = (bv >> 6) & 3;
            int c1 = (bv >> 4) & 3;
            int c2 = (bv >> 2) & 3;
            int c3 = bv & 3;

            // Branchless: 0=-1, 1=0, 2=+1, 3=0
            float w0 = (float)(c0 == 2) - (float)(c0 == 0);
            float w1 = (float)(c1 == 2) - (float)(c1 == 0);
            float w2 = (float)(c2 == 2) - (float)(c2 == 0);
            float w3 = (float)(c3 == 2) - (float)(c3 == 0);

            sum += ap[0 * 32 + gp] * w0;
            sum += ap[1 * 32 + gp] * w1;
            sum += ap[2 * 32 + gp] * w2;
            sum += ap[3 * 32 + gp] * w3;
        }
    }

    sum *= scale;

    // Warp reduction
    sum = warp_reduce_sum(sum);

    int lane = tid & 31;
    int warp = tid >> 5;
    if (lane == 0)
        shared[warp] = sum;
    __syncthreads();

    int numWarps = blockSize >> 5;
    if (tid < numWarps)
        sum = shared[tid];
    else
        sum = 0.0f;
    sum = warp_reduce_sum(sum);

    if (tid == 0)
        output[n] = sum;
}

// ── Fused TQ1_0 (ternary) Dequant + MatMul ────────────────────────────────
// TQ1_0: 256 elements per block, 54 bytes (52 base-3 packed + 2 padding).
// 5 trits per byte (base-3): trit 0=-1, 1=0, 2=+1.
// No scale factor — pure ternary.
// Grid: N blocks, Block: 32 threads.

__global__ void dequant_matmul_tq1_0(float* output, const float* a,
                                      const unsigned char* b,
                                      int M, int K, int N)
{
    extern __shared__ float shared[];
    int n = blockIdx.x;
    int tid = threadIdx.x;
    int blockSize = blockDim.x;

    if (n >= N) return;

    int blocksPerRow = K / 256;
    long long bytesPerRow = (long long)blocksPerRow * 54;
    const unsigned char* b_row = b + (long long)n * bytesPerRow;

    float sum = 0.0f;

    for (int blk = tid; blk < blocksPerRow; blk += blockSize)
    {
        const unsigned char* blockPtr = b_row + blk * 54;
        const float* aPtr = a + blk * 256;

        int elemIdx = 0;
        for (int byteIdx = 0; byteIdx < 52 && elemIdx < 256; byteIdx++)
        {
            int packed = blockPtr[byteIdx];
            // Decode 5 trits from one byte using base-3
            for (int t = 0; t < 5 && elemIdx < 256; t++)
            {
                int trit = packed % 3;
                packed /= 3;
                int weight = trit - 1; // 0->-1, 1->0, 2->+1
                if (weight == 1)
                    sum += aPtr[elemIdx];
                else if (weight == -1)
                    sum -= aPtr[elemIdx];
                elemIdx++;
            }
        }
    }

    // Warp reduction
    sum = warp_reduce_sum(sum);

    int lane = tid & 31;
    int warp = tid >> 5;
    if (lane == 0)
        shared[warp] = sum;
    __syncthreads();

    int numWarps = blockSize >> 5;
    if (tid < numWarps)
        sum = shared[tid];
    else
        sum = 0.0f;
    sum = warp_reduce_sum(sum);

    if (tid == 0)
        output[n] = sum;
}

// ── FP16 Dequant + MatMul ─────────────────────────────────────────────────
// Weights stored as FP16 (2 bytes per element), dequantized to FP32 on the fly.
// Grid: N blocks, Block: 32 threads.

__global__ void dequant_matmul_f16(float* output, const float* a,
                                    const unsigned short* b,
                                    int M, int K, int N)
{
    extern __shared__ float shared[];
    int n = blockIdx.x;
    int tid = threadIdx.x;
    int blockSize = blockDim.x;

    if (n >= N) return;

    const unsigned short* b_row = b + (long long)n * K;

    float sum = 0.0f;
    for (int k = tid; k < K; k += blockSize)
        sum += a[k] * fp16_to_fp32(b_row[k]);

    // Warp reduction
    sum = warp_reduce_sum(sum);

    int lane = tid & 31;
    int warp = tid >> 5;
    if (lane == 0)
        shared[warp] = sum;
    __syncthreads();

    int numWarps = blockSize >> 5;
    if (tid < numWarps)
        sum = shared[tid];
    else
        sum = 0.0f;
    sum = warp_reduce_sum(sum);

    if (tid == 0)
        output[n] = sum;
}

// ── Fused Q4_K Dequant + MatMul ────────────────────────────────────────────
// Q4_K super block: 256 elements, 144 bytes.
// Layout: 2b d(f16) + 2b dmin(f16) + 12b packed scales/mins + 128b nibbles.
// Grid: N blocks, Block: 32 threads.

__device__ __forceinline__ void unpack_q4k_scales(const unsigned char* sb,
    int j, float d, float dmin, float &sub_scale, float &sub_min)
{
    int sc, m;
    if (j < 4) {
        sc = sb[4 + j] & 63;
        m  = sb[4 + j + 4] & 63;
    } else {
        sc = (sb[4 + j + 4] & 0xF) | ((sb[4 + j - 4] >> 6) << 4);
        m  = (sb[4 + j + 4] >> 4)  | ((sb[4 + j]     >> 6) << 4);
    }
    sub_scale = d * (float)sc;
    sub_min = dmin * (float)m;
}

// Multi-row Q4_K: 8 output neurons per block with activation reuse + uint reads
#define Q4K_ROWS_PER_BLOCK 8

__global__ void dequant_matmul_q4_k(float* output, const float* a,
                                     const unsigned char* b,
                                     int M, int K, int N)
{
    extern __shared__ float shared[];
    int n_base = blockIdx.x * Q4K_ROWS_PER_BLOCK;
    int tid = threadIdx.x;
    int blockSize = blockDim.x;
    if (n_base >= N) return;

    int sbPerRow = K / 256;
    long long bytesPerRow = (long long)sbPerRow * 144;
    int totalChunks = sbPerRow * 4;

    float sums[Q4K_ROWS_PER_BLOCK];
    #pragma unroll
    for (int r = 0; r < Q4K_ROWS_PER_BLOCK; r++) sums[r] = 0.0f;

    for (int item = tid; item < totalChunks; item += blockSize)
    {
        int sb = item / 4;
        int chunk = item % 4;

        // Load activation once (shared across rows)
        const float* apLo = a + sb * 256 + chunk * 64;
        const float* apHi = apLo + 32;
        const float4* ap4Lo = reinterpret_cast<const float4*>(apLo);
        const float4* ap4Hi = reinterpret_cast<const float4*>(apHi);

        float4 aLoCache[8], aHiCache[8];
        float asumLo = 0.0f, asumHi = 0.0f;
        #pragma unroll
        for (int l = 0; l < 8; l++) {
            aLoCache[l] = __ldg(&ap4Lo[l]);
            aHiCache[l] = __ldg(&ap4Hi[l]);
            asumLo += aLoCache[l].x + aLoCache[l].y + aLoCache[l].z + aLoCache[l].w;
            asumHi += aHiCache[l].x + aHiCache[l].y + aHiCache[l].z + aHiCache[l].w;
        }

        // Process each row
        #pragma unroll
        for (int r = 0; r < Q4K_ROWS_PER_BLOCK; r++)
        {
            int n = n_base + r;
            if (n >= N) break;

            const unsigned char* sbp = b + (long long)n * bytesPerRow + sb * 144;
            // Read d + dmin as single uint via __ldg
            unsigned int d_dmin = __ldg(reinterpret_cast<const unsigned int*>(sbp));
            float d = fp16_to_fp32((unsigned short)(d_dmin & 0xFFFF));
            float dmin = fp16_to_fp32((unsigned short)(d_dmin >> 16));

            float ss0, sm0, ss1, sm1;
            unpack_q4k_scales(sbp, chunk * 2, d, dmin, ss0, sm0);
            unpack_q4k_scales(sbp, chunk * 2 + 1, d, dmin, ss1, sm1);

            // Read quants as uint via __ldg for read-only cache
            const unsigned int* qs32 = reinterpret_cast<const unsigned int*>(sbp + 16 + chunk * 32);
            float dotLo = 0.0f, dotHi = 0.0f;

            #pragma unroll
            for (int l = 0; l < 8; l++) {
                unsigned int packed = __ldg(&qs32[l]);
                dotLo += aLoCache[l].x*(float)(packed      &0xF) + aLoCache[l].y*(float)((packed>>8) &0xF)
                       + aLoCache[l].z*(float)((packed>>16)&0xF) + aLoCache[l].w*(float)((packed>>24)&0xF);
                dotHi += aHiCache[l].x*(float)((packed>>4) &0xF) + aHiCache[l].y*(float)((packed>>12)&0xF)
                       + aHiCache[l].z*(float)((packed>>20)&0xF) + aHiCache[l].w*(float)((packed>>28)&0xF);
            }
            sums[r] += ss0 * dotLo - sm0 * asumLo + ss1 * dotHi - sm1 * asumHi;
        }
    }

    int numWarps = blockSize >> 5;
    #pragma unroll
    for (int r = 0; r < Q4K_ROWS_PER_BLOCK; r++) {
        float sum = warp_reduce_sum(sums[r]);
        int lane = tid & 31; int warp = tid >> 5;
        if (lane == 0) shared[warp] = sum;
        __syncthreads();
        if (tid < numWarps) sum = shared[tid]; else sum = 0.0f;
        sum = warp_reduce_sum(sum);
        if (tid == 0 && n_base + r < N) output[n_base + r] = sum;
        __syncthreads();
    }
}

// ── Fused Q5_K Dequant + MatMul ────────────────────────────────────────────
// Q5_K super block: 256 elements, 176 bytes.
// Layout: 2b d + 2b dmin + 12b scales/mins + 32b high bits + 128b low nibbles.

__global__ void dequant_matmul_q5_k(float* output, const float* a,
                                     const unsigned char* b,
                                     int M, int K, int N)
{
    extern __shared__ float shared[];
    int n = blockIdx.x;
    int tid = threadIdx.x;
    int blockSize = blockDim.x;
    if (n >= N) return;

    int sbPerRow = K / 256;
    long long bytesPerRow = (long long)sbPerRow * 176;
    const unsigned char* b_row = b + (long long)n * bytesPerRow;

    float sum = 0.0f;
    for (int sb = tid; sb < sbPerRow; sb += blockSize)
    {
        const unsigned char* sbp = b_row + sb * 176;
        const float* ap = a + sb * 256;

        float d = fp16_to_fp32(((unsigned short)sbp[1] << 8) | sbp[0]);
        float dmin = fp16_to_fp32(((unsigned short)sbp[3] << 8) | sbp[2]);

        // Chunked layout matching ggml: 4 chunks of 64 elements each.
        // Each 32-byte qs block: low nibble → first 32, high nibble → next 32.
        // qh bits: element n's high bit at qh[n%32] bit (n/32), using rotating bitmask.
        const unsigned char* qh = sbp + 16;
        const unsigned char* qs = sbp + 48;
        int isIdx = 0;
        unsigned char u1 = 1, u2 = 2;
        for (int j = 0; j < 4; j++)
        {
            float ss0, sm0, ss1, sm1;
            unpack_q4k_scales(sbp, isIdx, d, dmin, ss0, sm0);
            unpack_q4k_scales(sbp, isIdx + 1, d, dmin, ss1, sm1);

            for (int l = 0; l < 32; l++)
            {
                int lo = qs[j * 32 + l] & 0xF;
                int hbLo = (qh[l] & u1) ? 16 : 0;
                sum += ap[j * 64 + l] * (ss0 * (lo + hbLo) - sm0);

                int hi = qs[j * 32 + l] >> 4;
                int hbHi = (qh[l] & u2) ? 16 : 0;
                sum += ap[j * 64 + l + 32] * (ss1 * (hi + hbHi) - sm1);
            }

            u1 <<= 2;
            u2 <<= 2;
            isIdx += 2;
        }
    }

    sum = warp_reduce_sum(sum);
    int lane = tid & 31;
    int warp = tid >> 5;
    if (lane == 0) shared[warp] = sum;
    __syncthreads();
    int numWarps = blockSize >> 5;
    if (tid < numWarps) sum = shared[tid]; else sum = 0.0f;
    sum = warp_reduce_sum(sum);
    if (tid == 0) output[n] = sum;
}

// ── Fused Q6_K Dequant + MatMul ────────────────────────────────────────────
// Q6_K super block: 256 elements, 210 bytes.
// Layout: ql[128] + qh[64] + sc[16] + d[2].

// Multi-row Q6_K: 2 output neurons per block with activation reuse
#define Q6K_ROWS_PER_BLOCK 2

__global__ void dequant_matmul_q6_k(float* output, const float* a,
                                     const unsigned char* b,
                                     int M, int K, int N)
{
    extern __shared__ float shared[];
    int n_base = blockIdx.x * Q6K_ROWS_PER_BLOCK;
    int tid = threadIdx.x;
    int blockSize = blockDim.x;
    if (n_base >= N) return;

    int sbPerRow = K / 256;
    long long bytesPerRow = (long long)sbPerRow * 210;
    int totalGroups = sbPerRow * 8;

    float sums[Q6K_ROWS_PER_BLOCK] = {0};

    for (int grp = tid; grp < totalGroups; grp += blockSize)
    {
        int sb = grp / 8;
        int gIdx = grp % 8;
        int halfIdx = gIdx / 4;
        int quadrant = gIdx % 4;
        const float* aBase = a + sb * 256 + halfIdx * 128 + quadrant * 32;

        // Load activation once (shared across rows)
        float a_vals[32];
        #pragma unroll
        for (int l = 0; l < 32; l++) a_vals[l] = __ldg(&aBase[l]);

        int qlOff = (quadrant & 1) * 32;
        int qlShift = (quadrant >> 1) * 4;
        int qhShift = quadrant * 2;

        #pragma unroll
        for (int r = 0; r < Q6K_ROWS_PER_BLOCK; r++)
        {
            int n = n_base + r;
            if (n >= N) break;

            const unsigned char* sbp = b + (long long)n * bytesPerRow + sb * 210;
            float d = fp16_to_fp32(*reinterpret_cast<const unsigned short*>(sbp + 208));
            const unsigned char* ql = sbp + halfIdx * 64;
            const unsigned char* qh = sbp + 128 + halfIdx * 32;
            int scIdx = halfIdx * 8;

            float sc0 = d * (float)((signed char)sbp[192 + scIdx + 0 + quadrant * 2]);
            float sc1 = d * (float)((signed char)sbp[192 + scIdx + 1 + quadrant * 2]);

            float local = 0.0f;
            #pragma unroll
            for (int l = 0; l < 32; l++) {
                int q = (((ql[qlOff + l] >> qlShift) & 0xF) | (((qh[l] >> qhShift) & 3) << 4)) - 32;
                float sc = (l < 16) ? sc0 : sc1;
                local += a_vals[l] * (sc * (float)q);
            }
            sums[r] += local;
        }
    }

    int numWarps = blockSize >> 5;
    #pragma unroll
    for (int r = 0; r < Q6K_ROWS_PER_BLOCK; r++) {
        float sum = warp_reduce_sum(sums[r]);
        int lane = tid & 31;
        int warp = tid >> 5;
        if (lane == 0) shared[warp] = sum;
        __syncthreads();
        if (tid < numWarps) sum = shared[tid]; else sum = 0.0f;
        sum = warp_reduce_sum(sum);
        if (tid == 0 && n_base + r < N) output[n_base + r] = sum;
        __syncthreads();
    }
}

} // extern "C"
