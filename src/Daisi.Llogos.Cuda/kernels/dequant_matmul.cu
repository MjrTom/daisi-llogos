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
// One block per output neuron. Vectorized float4 loads for activation.
// Grid: N blocks, Block: adaptive threads.

__global__ void dequant_matmul_q8_0(float* __restrict__ output,
                                     const float* __restrict__ a,
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
    const unsigned char* __restrict__ b_row = b + (long)n * bytes_per_row;

    // Two accumulators to reduce dependency chains
    float sum0 = 0.0f, sum1 = 0.0f;

    for (int blk = tid; blk < blocks_per_row; blk += blockSize)
    {
        const unsigned char* block_ptr = b_row + blk * 34;
        float scale = fp16_to_fp32(*reinterpret_cast<const unsigned short*>(block_ptr));
        const signed char* quants = (const signed char*)(block_ptr + 2);
        const float4* ap = reinterpret_cast<const float4*>(a + blk * 32);

        float bs0 = 0.0f, bs1 = 0.0f;
        #pragma unroll
        for (int i = 0; i < 4; i++)
        {
            float4 ai = ap[i];
            int base = i * 4;
            bs0 += ai.x * (float)quants[base]     + ai.y * (float)quants[base + 1]
                 + ai.z * (float)quants[base + 2] + ai.w * (float)quants[base + 3];
        }
        #pragma unroll
        for (int i = 4; i < 8; i++)
        {
            float4 ai = ap[i];
            int base = i * 4;
            bs1 += ai.x * (float)quants[base]     + ai.y * (float)quants[base + 1]
                 + ai.z * (float)quants[base + 2] + ai.w * (float)quants[base + 3];
        }
        sum0 += scale * bs0;
        sum1 += scale * bs1;
    }

    float sum = sum0 + sum1;

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

    // Each thread processes one FULL chunk (64 elements = lo + hi halves) at a time.
    // 4 chunks per super-block × sbPerRow = total work items.
    // Processing both halves together avoids redundant scale unpacking and byte reloads.
    int totalChunks = sbPerRow * 4;

    float sum = 0.0f;
    for (int item = tid; item < totalChunks; item += blockSize)
    {
        int sb = item / 4;
        int chunk = item % 4;

        const unsigned char* sbp = b_row + sb * 144;
        const float* apLo = a + sb * 256 + chunk * 64;
        const float* apHi = apLo + 32;

        float d = fp16_to_fp32(*reinterpret_cast<const unsigned short*>(sbp));
        float dmin = fp16_to_fp32(*reinterpret_cast<const unsigned short*>(sbp + 2));

        float ss0, sm0, ss1, sm1;
        unpack_q4k_scales(sbp, chunk * 2, d, dmin, ss0, sm0);
        unpack_q4k_scales(sbp, chunk * 2 + 1, d, dmin, ss1, sm1);

        const unsigned char* qs = sbp + 16 + chunk * 32;
        const float4* ap4Lo = reinterpret_cast<const float4*>(apLo);
        const float4* ap4Hi = reinterpret_cast<const float4*>(apHi);

        float dotLo = 0.0f, dotHi = 0.0f;
        float asumLo = 0.0f, asumHi = 0.0f;

        #pragma unroll
        for (int l = 0; l < 8; l++)
        {
            float4 aLo = ap4Lo[l];
            float4 aHi = ap4Hi[l];
            asumLo += aLo.x + aLo.y + aLo.z + aLo.w;
            asumHi += aHi.x + aHi.y + aHi.z + aHi.w;

            unsigned char p0 = qs[l*4], p1 = qs[l*4+1], p2 = qs[l*4+2], p3 = qs[l*4+3];

            dotLo += aLo.x * (float)(p0 & 0xF) + aLo.y * (float)(p1 & 0xF)
                   + aLo.z * (float)(p2 & 0xF) + aLo.w * (float)(p3 & 0xF);
            dotHi += aHi.x * (float)(p0 >> 4) + aHi.y * (float)(p1 >> 4)
                   + aHi.z * (float)(p2 >> 4) + aHi.w * (float)(p3 >> 4);
        }
        sum += ss0 * dotLo - sm0 * asumLo + ss1 * dotHi - sm1 * asumHi;
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

    // Each thread processes one 32-element quadrant (8 quadrants per super-block).
    int totalGroups = sbPerRow * 8;

    float sum = 0.0f;
    for (int grp = tid; grp < totalGroups; grp += blockSize)
    {
        int sb = grp / 8;
        int gIdx = grp % 8;
        int half = gIdx / 4;
        int quadrant = gIdx % 4;

        const unsigned char* sbp = b_row + sb * 210;
        float d = fp16_to_fp32(*reinterpret_cast<const unsigned short*>(sbp + 208));

        const unsigned char* ql = sbp + half * 64;
        const unsigned char* qh = sbp + 128 + half * 32;
        int scIdx = half * 8;
        const float* aBase = a + sb * 256 + half * 128 + quadrant * 32;

        // Precompute scale pairs for both sub-groups (l/16 = 0 and 1)
        float sc0 = d * (float)((signed char)sbp[192 + scIdx + 0 + quadrant * 2]);
        float sc1 = d * (float)((signed char)sbp[192 + scIdx + 1 + quadrant * 2]);

        // Bit extraction constants for this quadrant
        int qlOff = (quadrant & 1) * 32;  // 0 for q0,q2; 32 for q1,q3
        int qlShift = (quadrant >> 1) * 4; // 0 for q0,q1; 4 for q2,q3
        int qhShift = quadrant * 2;        // 0,2,4,6

        float local = 0.0f;
        #pragma unroll
        for (int l = 0; l < 32; l++)
        {
            int q = (((ql[qlOff + l] >> qlShift) & 0xF) | (((qh[l] >> qhShift) & 3) << 4)) - 32;
            float sc = (l < 16) ? sc0 : sc1;
            local += aBase[l] * (sc * (float)q);
        }
        sum += local;
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
