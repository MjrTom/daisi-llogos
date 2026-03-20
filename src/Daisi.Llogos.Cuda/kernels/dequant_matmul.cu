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
// Uses vectorized loads (float4/char4) for 4× fewer memory transactions.
// Grid: N blocks, Block: 256 threads.

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

    float sum = 0.0f;

    for (int blk = tid; blk < blocks_per_row; blk += blockSize)
    {
        const unsigned char* block_ptr = b_row + blk * 34;

        // FP16 → FP32 scale
        float scale = fp16_to_fp32(*reinterpret_cast<const unsigned short*>(block_ptr));

        // Vectorized float4 loads for activation (aligned), scalar for quants (unaligned)
        const signed char* quants = (const signed char*)(block_ptr + 2);
        const float4* ap = reinterpret_cast<const float4*>(a + blk * 32);

        float block_sum = 0.0f;
        #pragma unroll
        for (int i = 0; i < 8; i++)
        {
            float4 ai = ap[i];
            int base = i * 4;
            block_sum += ai.x * (float)quants[base]     + ai.y * (float)quants[base + 1]
                       + ai.z * (float)quants[base + 2] + ai.w * (float)quants[base + 3];
        }

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

__device__ void unpack_q4k_scales(const unsigned char* sb,
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

__global__ void dequant_matmul_q4_k(float* output, const float* a,
                                     const unsigned char* b,
                                     int M, int K, int N)
{
    extern __shared__ float shared[];
    int n = blockIdx.x;
    int tid = threadIdx.x;
    int blockSize = blockDim.x;
    if (n >= N) return;

    int sbPerRow = K / 256;
    long long bytesPerRow = (long long)sbPerRow * 144;
    const unsigned char* b_row = b + (long long)n * bytesPerRow;

    // Each thread processes one chunk-half (32 elements) at a time.
    // 8 chunk-halves per super-block × sbPerRow = total work items.
    int totalItems = sbPerRow * 8;

    float sum = 0.0f;
    for (int item = tid; item < totalItems; item += blockSize)
    {
        int sb = item / 8;
        int halfIdx = item % 8;         // 0..7: which chunk-half
        int chunk = halfIdx / 2;         // 0..3
        int isHi = halfIdx & 1;          // 0=lo nibble, 1=hi nibble

        const unsigned char* sbp = b_row + sb * 144;
        const float* ap = a + sb * 256 + chunk * 64 + isHi * 32;

        float d = fp16_to_fp32(((unsigned short)sbp[1] << 8) | sbp[0]);
        float dmin = fp16_to_fp32(((unsigned short)sbp[3] << 8) | sbp[2]);

        float ss, sm;
        unpack_q4k_scales(sbp, chunk * 2 + isHi, d, dmin, ss, sm);

        const unsigned char* qs = sbp + 16 + chunk * 32;
        const float4* ap4 = reinterpret_cast<const float4*>(ap);

        float dot = 0.0f;
        float asum = 0.0f;
        #pragma unroll
        for (int l = 0; l < 8; l++)
        {
            float4 av = ap4[l];
            asum += av.x + av.y + av.z + av.w;
            int base = l * 4;
            unsigned char p0 = qs[base], p1 = qs[base+1], p2 = qs[base+2], p3 = qs[base+3];
            if (isHi) {
                dot += av.x * (float)(p0 >> 4) + av.y * (float)(p1 >> 4)
                     + av.z * (float)(p2 >> 4) + av.w * (float)(p3 >> 4);
            } else {
                dot += av.x * (float)(p0 & 0xF) + av.y * (float)(p1 & 0xF)
                     + av.z * (float)(p2 & 0xF) + av.w * (float)(p3 & 0xF);
            }
        }
        sum += ss * dot - sm * asum;
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

__global__ void dequant_matmul_q6_k(float* output, const float* a,
                                     const unsigned char* b,
                                     int M, int K, int N)
{
    extern __shared__ float shared[];
    int n = blockIdx.x;
    int tid = threadIdx.x;
    int blockSize = blockDim.x;
    if (n >= N) return;

    int sbPerRow = K / 256;
    long long bytesPerRow = (long long)sbPerRow * 210;
    const unsigned char* b_row = b + (long long)n * bytesPerRow;

    float sum = 0.0f;
    for (int sb = tid; sb < sbPerRow; sb += blockSize)
    {
        const unsigned char* sbp = b_row + sb * 210;
        const float* ap = a + sb * 256;

        // ql at 0, qh at 128, sc at 192, d at 208
        float d = fp16_to_fp32(((unsigned short)sbp[209] << 8) | sbp[208]);

        // Process two 128-element halves with interleaved ql/qh layout
        for (int half = 0; half < 2; half++)
        {
            const unsigned char* ql = sbp + half * 64;
            const unsigned char* qh = sbp + 128 + half * 32;
            int scIdx = half * 8;
            const float* aHalf = ap + half * 128;

            for (int l = 0; l < 32; l++)
            {
                int q1 = ((ql[l] & 0xF)      | (((qh[l] >> 0) & 3) << 4)) - 32;
                int q2 = ((ql[l + 32] & 0xF)  | (((qh[l] >> 2) & 3) << 4)) - 32;
                int q3 = ((ql[l] >> 4)         | (((qh[l] >> 4) & 3) << 4)) - 32;
                int q4 = ((ql[l + 32] >> 4)    | (((qh[l] >> 6) & 3) << 4)) - 32;

                // ggml uses is = l/16 to select sub-group scales
                int isc = l / 16;
                sum += aHalf[l]      * (d * (float)((signed char)sbp[192 + scIdx + isc + 0]) * q1);
                sum += aHalf[l + 32] * (d * (float)((signed char)sbp[192 + scIdx + isc + 2]) * q2);
                sum += aHalf[l + 64] * (d * (float)((signed char)sbp[192 + scIdx + isc + 4]) * q3);
                sum += aHalf[l + 96] * (d * (float)((signed char)sbp[192 + scIdx + isc + 6]) * q4);
            }
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

} // extern "C"
