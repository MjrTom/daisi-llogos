// Tiled Quantized GEMM: Q8_0 weights × Q8_1 activations → F32 output
// Compiled to cubin via NVRTC at runtime.
//
// output[M×N] = activation[M×K] × weight^T[N×K]
//   activation: Q8_1 format [M rows × (K/32) blocks × 36 bytes per block]
//   weight:     Q8_0 aligned [N rows × (K/32) blocks × 36 bytes per block]
//
// Q8_1 block (36 bytes): [d: fp16 (2B)] [sum: fp16 (2B)] [qs: 32 × int8 (32B)]
// Q8_0 block (36 bytes): [d: fp16 (2B)] [pad: 2B]        [qs: 32 × int8 (32B)]
//
// dp4a: __dp4a(int a, int b, int acc) → acc += a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3]
//   Processes 4 × int8 multiply-adds per instruction.

__device__ __forceinline__ float fp16_to_fp32_mmq(unsigned short h)
{
    float result;
    asm("cvt.f32.f16 %0, %1;" : "=f"(result) : "h"(h));
    return result;
}

extern "C" {

// ── Tile parameters ─────────────────────────────────────────────────────────
// Each block computes a TILE_M × TILE_N output tile.
// K dimension processed in chunks of K_CHUNK Q8 blocks (K_CHUNK × 32 elements).
//
// Thread mapping: 256 threads = 8 warps × 32 lanes
//   threadIdx.y (warp index 0..7) → selects N column within tile
//   threadIdx.x (lane 0..31)     → selects M row within tile
//
// Each thread accumulates one output element: output[m_base + lane, n_base + warp]

#define MMQ_TILE_M  32   // M rows per block (= warp_size, 1 per lane)
#define MMQ_TILE_N  64   // N cols per block (8 per warp × 8 warps)
#define MMQ_K_CHUNK 8    // Q8 blocks per shared memory load (256 elements)
#define MMQ_NWARPS  8
#define MMQ_N_PER_WARP 8 // Each warp handles 8 N columns — 8× activation reuse
#define MMQ_M_PER_LANE 1 // Each lane handles 1 M row
#define MMQ_BLOCK_SIZE (32 * MMQ_NWARPS)  // 256 threads

// Shared memory: separate arrays for scales and quants for coalesced access
// Weight tile: MMQ_TILE_N × MMQ_K_CHUNK scales + quants
// Activation tile: MMQ_TILE_M × MMQ_K_CHUNK scales + quants

__global__ __launch_bounds__(MMQ_BLOCK_SIZE)
void tiled_mmq_q8_0_q8_1(
    float* __restrict__ output,           // [M × N] row-major
    const unsigned char* __restrict__ act, // Q8_1 [M × blocks_per_row × 36]
    const unsigned char* __restrict__ wt,  // Q8_0 aligned [N × blocks_per_row × 36]
    int M, int K, int N)
{
    // Block's output tile position
    int n_base = blockIdx.x * MMQ_TILE_N;
    int m_base = blockIdx.y * MMQ_TILE_M;

    // Thread's output position
    int lane = threadIdx.x;  // 0..31 → M row within tile
    int warp = threadIdx.y;  // 0..7  → N col within tile
    int tid = warp * 32 + lane;

    int m = m_base + lane;
    int n = n_base + warp;

    int blocks_per_row = K / 32;
    long bytes_per_row = (long)blocks_per_row * 36;

    // Shared memory layout:
    //   w_scales[TILE_N][K_CHUNK]     — weight scales (float)
    //   w_quants[TILE_N][K_CHUNK][8]  — weight quants (int32, 8 per block = 32 int8)
    //   a_scales[TILE_M][K_CHUNK]     — activation scales (float)
    //   a_quants[TILE_M][K_CHUNK][8]  — activation quants (int32)
    extern __shared__ char smem_raw[];

    float*  w_scales = (float*)smem_raw;
    int*    w_quants = (int*)(w_scales + MMQ_TILE_N * MMQ_K_CHUNK);
    float*  a_scales = (float*)(w_quants + MMQ_TILE_N * MMQ_K_CHUNK * 8);
    int*    a_quants = (int*)(a_scales + MMQ_TILE_M * MMQ_K_CHUNK);

    // Accumulators: each thread computes M_PER_LANE × N_PER_WARP output elements
    float acc[MMQ_M_PER_LANE][MMQ_N_PER_WARP];
    #pragma unroll
    for (int mi = 0; mi < MMQ_M_PER_LANE; mi++)
        #pragma unroll
        for (int ni = 0; ni < MMQ_N_PER_WARP; ni++)
            acc[mi][ni] = 0.0f;

    // Process K in chunks
    for (int kb = 0; kb < blocks_per_row; kb += MMQ_K_CHUNK)
    {
        int chunk = MMQ_K_CHUNK;
        if (kb + chunk > blocks_per_row) chunk = blocks_per_row - kb;

        // ── Cooperative load: weight tile [TILE_N × chunk] ──
        // 256 threads load TILE_N × chunk = 8 × 8 = 64 block entries
        {
            int total_w = MMQ_TILE_N * chunk;
            for (int i = tid; i < total_w; i += MMQ_BLOCK_SIZE)
            {
                int tn = i / chunk;  // which N in tile
                int tk = i % chunk;  // which K block in chunk
                int gn = n_base + tn;
                if (gn >= N) continue;

                const unsigned char* blk = wt + (long)gn * bytes_per_row + (long)(kb + tk) * 36;
                w_scales[tn * MMQ_K_CHUNK + tk] = fp16_to_fp32_mmq(
                    ((unsigned short)blk[1] << 8) | blk[0]);

                const int* qs = (const int*)(blk + 4);
                int base = (tn * MMQ_K_CHUNK + tk) * 8;
                w_quants[base+0] = __ldg(&qs[0]);
                w_quants[base+1] = __ldg(&qs[1]);
                w_quants[base+2] = __ldg(&qs[2]);
                w_quants[base+3] = __ldg(&qs[3]);
                w_quants[base+4] = __ldg(&qs[4]);
                w_quants[base+5] = __ldg(&qs[5]);
                w_quants[base+6] = __ldg(&qs[6]);
                w_quants[base+7] = __ldg(&qs[7]);
            }
        }

        // ── Cooperative load: activation tile [TILE_M × chunk] ──
        {
            int total_a = MMQ_TILE_M * chunk;
            for (int i = tid; i < total_a; i += MMQ_BLOCK_SIZE)
            {
                int tm = i / chunk;  // which M in tile
                int tk = i % chunk;  // which K block in chunk
                int gm = m_base + tm;
                if (gm >= M) continue;

                const unsigned char* blk = act + (long)gm * bytes_per_row + (long)(kb + tk) * 36;
                a_scales[tm * MMQ_K_CHUNK + tk] = fp16_to_fp32_mmq(
                    ((unsigned short)blk[1] << 8) | blk[0]);

                const int* qs = (const int*)(blk + 4);
                int base = (tm * MMQ_K_CHUNK + tk) * 8;
                a_quants[base+0] = qs[0];
                a_quants[base+1] = qs[1];
                a_quants[base+2] = qs[2];
                a_quants[base+3] = qs[3];
                a_quants[base+4] = qs[4];
                a_quants[base+5] = qs[5];
                a_quants[base+6] = qs[6];
                a_quants[base+7] = qs[7];
            }
        }

        __syncthreads();

        // ── Compute: activation-first, N-inner loop for register reuse ──
        if (m < M)
        {
            for (int tk = 0; tk < chunk; tk++)
            {
                float as_ = a_scales[lane * MMQ_K_CHUNK + tk];
                int a_base = (lane * MMQ_K_CHUNK + tk) * 8;

                // Load activation quants into registers ONCE
                int a0 = a_quants[a_base+0], a1 = a_quants[a_base+1];
                int a2 = a_quants[a_base+2], a3 = a_quants[a_base+3];
                int a4 = a_quants[a_base+4], a5 = a_quants[a_base+5];
                int a6 = a_quants[a_base+6], a7 = a_quants[a_base+7];

                // Reuse across all N columns
                #pragma unroll
                for (int ni = 0; ni < MMQ_N_PER_WARP; ni++)
                {
                    int tn = warp * MMQ_N_PER_WARP + ni;
                    if (n_base + tn >= N) continue;

                    float ws = w_scales[tn * MMQ_K_CHUNK + tk];
                    int w_base = (tn * MMQ_K_CHUNK + tk) * 8;

                    int sumi = 0;
                    sumi = __dp4a(a0, w_quants[w_base+0], sumi);
                    sumi = __dp4a(a1, w_quants[w_base+1], sumi);
                    sumi = __dp4a(a2, w_quants[w_base+2], sumi);
                    sumi = __dp4a(a3, w_quants[w_base+3], sumi);
                    sumi = __dp4a(a4, w_quants[w_base+4], sumi);
                    sumi = __dp4a(a5, w_quants[w_base+5], sumi);
                    sumi = __dp4a(a6, w_quants[w_base+6], sumi);
                    sumi = __dp4a(a7, w_quants[w_base+7], sumi);

                    acc[0][ni] += as_ * ws * (float)sumi;
                }
            }
        }

        __syncthreads();
    }

    // Write results
    if (m < M)
    {
        #pragma unroll
        for (int ni = 0; ni < MMQ_N_PER_WARP; ni++)
        {
            int gn = n_base + warp * MMQ_N_PER_WARP + ni;
            if (gn < N)
                output[m * N + gn] = acc[0][ni];
        }
    }
}

// ── Batched Q8_1 quantization (M rows in one launch) ────────────────────────
// Quantizes M × K float activations to Q8_1 format.
// Each thread processes one 32-element block.

__device__ __forceinline__ unsigned short fp32_to_fp16_mmq(float val)
{
    unsigned short result;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(result) : "f"(val));
    return result;
}

__global__ void batched_quantize_f32_q8_1(
    unsigned char* __restrict__ dst,   // Q8_1 output [M × blocks_per_row × 36]
    const float* __restrict__ src,     // F32 input [M × K]
    int M, int K)
{
    int blocks_per_row = K / 32;
    int total_blocks = M * blocks_per_row;
    int blk = blockIdx.x * blockDim.x + threadIdx.x;
    if (blk >= total_blocks) return;

    int row = blk / blocks_per_row;
    int col_blk = blk % blocks_per_row;

    const float* sp = src + row * K + col_blk * 32;
    unsigned char* dp = dst + (long)row * blocks_per_row * 36 + col_blk * 36;

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
    unsigned short d_fp16 = fp32_to_fp16_mmq(d);
    unsigned short s_fp16 = fp32_to_fp16_mmq(sum);
    dp[0] = d_fp16 & 0xFF; dp[1] = d_fp16 >> 8;
    dp[2] = s_fp16 & 0xFF; dp[3] = s_fp16 >> 8;

    signed char* qs = (signed char*)(dp + 4);
    #pragma unroll
    for (int i = 0; i < 32; i++)
        qs[i] = (amax == 0.0f) ? 0 : (signed char)__float2int_rn(sp[i] / d);
}

} // extern "C"
