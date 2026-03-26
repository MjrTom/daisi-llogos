// daisi-llogos CUDA kernels: LLogos Turbo KV cache compression
// Implements fused Walsh-Hadamard rotation, scalar quantization, and compressed attention.
//
// Two main kernels:
//   turbo_kv_write:     F32 K/V → rotate → quantize → pack bits + store scale
//   turbo_gated_attention: Read compressed KV, dequant inline during attention dot products

// ── Helpers ──────────────────────────────────────────────────────────────────

// Walsh-Hadamard butterfly transform (in-place, in registers/shared mem)
// Operates on dim elements. dim must be power of 2.
__device__ void wht_butterfly(float* data, int dim)
{
    for (int halfSize = 1; halfSize < dim; halfSize <<= 1)
    {
        for (int i = 0; i < dim; i += halfSize << 1)
        {
            for (int j = i; j < i + halfSize; j++)
            {
                float a = data[j];
                float b = data[j + halfSize];
                data[j] = a + b;
                data[j + halfSize] = a - b;
            }
        }
    }
}

// Apply random sign flips: data[i] *= signs[i]
__device__ void apply_signs(float* data, const float* signs, int dim)
{
    for (int i = 0; i < dim; i++)
        data[i] *= signs[i];
}

// 4-bit scalar quantize: find level via binary search into boundaries
__device__ int quantize_4bit(float val, const float* boundaries)
{
    // 15 boundaries for 16 levels, binary search
    int lo = 0, hi = 15;
    while (lo < hi)
    {
        int mid = (lo + hi) >> 1;
        if (val > boundaries[mid]) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

// 3-bit scalar quantize: 7 boundaries for 8 levels
__device__ int quantize_3bit(float val, const float* boundaries)
{
    int lo = 0, hi = 7;
    while (lo < hi)
    {
        int mid = (lo + hi) >> 1;
        if (val > boundaries[mid]) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

// 2-bit scalar quantize: 3 boundaries for 4 levels
__device__ int quantize_2bit(float val, const float* boundaries)
{
    int lo = 0, hi = 3;
    while (lo < hi)
    {
        int mid = (lo + hi) >> 1;
        if (val > boundaries[mid]) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

// ── Quantization grids (Lloyd-Max for N(0,1)) ──
// Stored in constant memory for fast broadcast across threads

// 4-bit: 16 levels
__constant__ float boundaries_4bit[15] = {
    -2.401f, -1.844f, -1.437f, -1.099f, -0.7975f, -0.5141f, -0.2391f,
    0.0f,
    0.2391f, 0.5141f, 0.7975f, 1.099f, 1.437f, 1.844f, 2.401f
};
__constant__ float centroids_4bit[16] = {
    -2.733f, -2.069f, -1.618f, -1.256f, -0.9414f, -0.6522f, -0.3747f, -0.1194f,
    0.1194f, 0.3747f, 0.6522f, 0.9414f, 1.256f, 1.618f, 2.069f, 2.733f
};

// 3-bit: 8 levels
__constant__ float boundaries_3bit[7] = {
    -1.748f, -1.050f, -0.5006f, 0.0f, 0.5006f, 1.050f, 1.748f
};
__constant__ float centroids_3bit[8] = {
    -2.152f, -1.344f, -0.7560f, -0.2451f, 0.2451f, 0.7560f, 1.344f, 2.152f
};

// 2-bit: 4 levels
__constant__ float boundaries_2bit[3] = {
    -0.9816f, 0.0f, 0.9816f
};
__constant__ float centroids_2bit[4] = {
    -1.51f, -0.4528f, 0.4528f, 1.51f
};

// Lookup correct grid based on quantBits
__device__ const float* get_boundaries(int quantBits)
{
    switch (quantBits) {
        case 2: return boundaries_2bit;
        case 3: return boundaries_3bit;
        default: return boundaries_4bit;
    }
}

__device__ const float* get_centroids(int quantBits)
{
    switch (quantBits) {
        case 2: return centroids_2bit;
        case 3: return centroids_3bit;
        default: return centroids_4bit;
    }
}

__device__ int quantize_scalar(float val, int quantBits, const float* boundaries)
{
    switch (quantBits) {
        case 2: return quantize_2bit(val, boundaries);
        case 3: return quantize_3bit(val, boundaries);
        default: return quantize_4bit(val, boundaries);
    }
}

// ── Pack 4-bit values into bytes (2 values per byte) ──
__device__ void pack_4bit(unsigned char* packed, const int* levels, int dim)
{
    for (int i = 0; i < dim; i += 2)
    {
        int l0 = levels[i];
        int l1 = (i + 1 < dim) ? levels[i + 1] : 0;
        packed[i >> 1] = (unsigned char)(l0 | (l1 << 4));
    }
}

// ── Unpack 4-bit values from bytes ──
__device__ void unpack_4bit(const unsigned char* packed, float* output, const float* centroids, int dim)
{
    for (int i = 0; i < dim; i += 2)
    {
        unsigned char b = packed[i >> 1];
        output[i] = centroids[b & 0xF];
        if (i + 1 < dim)
            output[i + 1] = centroids[(b >> 4) & 0xF];
    }
}

// ── Pack 3-bit values into bytes (bit-packed) ──
__device__ void pack_3bit(unsigned char* packed, const int* levels, int dim)
{
    int bitPos = 0;
    int numBytes = (dim * 3 + 7) / 8;
    for (int i = 0; i < numBytes; i++) packed[i] = 0;

    for (int i = 0; i < dim; i++)
    {
        int byteIdx = bitPos >> 3;
        int bitOff = bitPos & 7;
        packed[byteIdx] |= (unsigned char)(levels[i] << bitOff);
        if (bitOff > 5)
            packed[byteIdx + 1] |= (unsigned char)(levels[i] >> (8 - bitOff));
        bitPos += 3;
    }
}

// ── Unpack 3-bit values from bytes ──
__device__ void unpack_3bit(const unsigned char* packed, float* output, const float* centroids, int dim)
{
    int bitPos = 0;
    for (int i = 0; i < dim; i++)
    {
        int byteIdx = bitPos >> 3;
        int bitOff = bitPos & 7;
        int level = (packed[byteIdx] >> bitOff) & 0x7;
        if (bitOff > 5)
            level |= (packed[byteIdx + 1] << (8 - bitOff)) & 0x7;
        output[i] = centroids[level];
        bitPos += 3;
    }
}

// ── Pack/unpack 2-bit values (4 per byte) ──
__device__ void pack_2bit(unsigned char* packed, const int* levels, int dim)
{
    for (int i = 0; i < dim; i += 4)
    {
        unsigned char b = 0;
        for (int j = 0; j < 4 && i + j < dim; j++)
            b |= (unsigned char)(levels[i + j] << (j * 2));
        packed[i >> 2] = b;
    }
}

__device__ void unpack_2bit(const unsigned char* packed, float* output, const float* centroids, int dim)
{
    for (int i = 0; i < dim; i++)
    {
        int byteIdx = i >> 2;
        int shift = (i & 3) * 2;
        int level = (packed[byteIdx] >> shift) & 0x3;
        output[i] = centroids[level];
    }
}

// Generic dequant dispatcher
__device__ void dequant_vector(const unsigned char* packed, float* output, int dim, int quantBits)
{
    const float* cents = get_centroids(quantBits);
    switch (quantBits) {
        case 2: unpack_2bit(packed, output, cents, dim); break;
        case 3: unpack_3bit(packed, output, cents, dim); break;
        default: unpack_4bit(packed, output, cents, dim); break;
    }
}

extern "C" {

// ═══════════════════════════════════════════════════════════════════════════
// turbo_kv_write: Compress K and V for one position into packed storage
//
// Grid: nKvHeads blocks (one per head)
// Block: 1 thread per head (sequential per-dim work — dim is small, 64-128)
//
// Each thread processes one head: normalize → apply signs → WHT → quantize → pack
// ═══════════════════════════════════════════════════════════════════════════

__global__ void turbo_kv_write(
    const float* k,             // [nKvHeads × keyLength] input K
    const float* v,             // [nKvHeads × valueLength] input V
    unsigned char* kPacked,     // compressed K storage (all layers, all heads, all positions)
    unsigned char* vPacked,     // compressed V storage
    float* kScales,             // [nKvHeads × maxSeqLen] per-head K scales
    float* vScales,             // [nKvHeads × maxSeqLen] per-head V scales
    const float* kSigns,        // [keyLength] WHT sign flips for K
    const float* vSigns,        // [valueLength] WHT sign flips for V
    int nKvHeads, int keyLength, int valueLength,
    int maxSeqLen, int position,
    int kPackedPerHead, int vPackedPerHead,
    int quantBits)
{
    int head = blockIdx.x;
    if (head >= nKvHeads) return;

    // Thread-local buffers (dim <= 128 for all current models)
    float buf[128];
    int levels[128];

    const float* bounds = get_boundaries(quantBits);

    // ── Compress Key ──
    {
        const float* kHead = k + head * keyLength;

        // Compute RMS scale
        float normSq = 0;
        for (int d = 0; d < keyLength; d++)
            normSq += kHead[d] * kHead[d];
        float scale = sqrtf(normSq / keyLength);
        float invScale = (scale > 1e-10f) ? 1.0f / scale : 0.0f;

        // Normalize
        for (int d = 0; d < keyLength; d++)
            buf[d] = kHead[d] * invScale;

        // Apply signs + WHT
        apply_signs(buf, kSigns, keyLength);
        wht_butterfly(buf, keyLength);

        // Normalize WHT
        float whtNorm = rsqrtf((float)keyLength);
        for (int d = 0; d < keyLength; d++)
            buf[d] *= whtNorm;

        // Quantize
        for (int d = 0; d < keyLength; d++)
            levels[d] = quantize_scalar(buf[d], quantBits, bounds);

        // Pack into output
        unsigned char* kOut = kPacked + (head * maxSeqLen + position) * kPackedPerHead;
        switch (quantBits) {
            case 2: pack_2bit(kOut, levels, keyLength); break;
            case 3: pack_3bit(kOut, levels, keyLength); break;
            default: pack_4bit(kOut, levels, keyLength); break;
        }

        // Store scale
        kScales[head * maxSeqLen + position] = scale;
    }

    // ── Compress Value ──
    {
        const float* vHead = v + head * valueLength;

        float normSq = 0;
        for (int d = 0; d < valueLength; d++)
            normSq += vHead[d] * vHead[d];
        float scale = sqrtf(normSq / valueLength);
        float invScale = (scale > 1e-10f) ? 1.0f / scale : 0.0f;

        for (int d = 0; d < valueLength; d++)
            buf[d] = vHead[d] * invScale;

        apply_signs(buf, vSigns, valueLength);
        wht_butterfly(buf, valueLength);

        float whtNorm = rsqrtf((float)valueLength);
        for (int d = 0; d < valueLength; d++)
            buf[d] *= whtNorm;

        for (int d = 0; d < valueLength; d++)
            levels[d] = quantize_scalar(buf[d], quantBits, bounds);

        unsigned char* vOut = vPacked + (head * maxSeqLen + position) * vPackedPerHead;
        switch (quantBits) {
            case 2: pack_2bit(vOut, levels, valueLength); break;
            case 3: pack_3bit(vOut, levels, valueLength); break;
            default: pack_4bit(vOut, levels, valueLength); break;
        }

        vScales[head * maxSeqLen + position] = scale;
    }
}


// ═══════════════════════════════════════════════════════════════════════════
// turbo_gated_attention: Compressed KV attention with fused dequant
//
// Grid: numHeads blocks (one per attention head)
// Block: BlockSize threads (256)
//
// Each block handles one attention head. Reads compressed K/V from packed
// byte storage, dequantizes inline (unpack → inverse WHT → rescale),
// computes tiled online softmax attention with sigmoid gating.
//
// The bandwidth savings come from reading packed 3-4 bit data instead of
// F16/F32 from HBM — this is where TurboQuant delivers its speedup on GPU.
// ═══════════════════════════════════════════════════════════════════════════

#define TQ_ATTN_TILE_SIZE 256

__global__ void turbo_gated_attention(
    float* output,                  // [numHeads × valueLength]
    const float* qAttn,             // [numHeads × keyLength]
    const float* qGate,             // [numHeads × keyLength]
    const unsigned char* kPacked,   // compressed K: [nKvHeads × maxSeqLen × kPackedPerHead]
    const unsigned char* vPacked,   // compressed V: [nKvHeads × maxSeqLen × vPackedPerHead]
    const float* kScales,           // [nKvHeads × maxSeqLen]
    const float* vScales,           // [nKvHeads × maxSeqLen]
    const float* kSigns,            // [keyLength]
    const float* vSigns,            // [valueLength]
    int numHeads, int numKvHeads,
    int keyLength, int valueLength,
    int maxSeqLen, int seqLen,
    float scale,
    int kPackedPerHead, int vPackedPerHead,
    int quantBits)
{
    extern __shared__ float shared[];
    int head = blockIdx.x;
    int tid = threadIdx.x;
    int stride = blockDim.x;
    int kvHead = head * numKvHeads / numHeads;

    const float* q = qAttn + head * keyLength;
    const float* qg = qGate + head * keyLength;

    // Shared memory layout:
    // [TQ_ATTN_TILE_SIZE scores] + [blockDim.x reduction temp] + [128 V broadcast buffer]
    float* scores = shared;
    float* temp = shared + TQ_ATTN_TILE_SIZE;
    float* vShared = temp + blockDim.x;  // V broadcast buffer (max 128 dims)

    // ── Pre-rotate Q into the K quantization domain (one WHT per head) ──
    // Math: q · k_original = scale_k × q_rotated · k_centroids
    // where q_rotated = (1/√n) · H · (D_k · q)
    // This eliminates ALL per-position inverse WHTs for K scoring.
    float qRot[128];
    if (tid == 0)
    {
        for (int d = 0; d < keyLength; d++)
            qRot[d] = q[d] * kSigns[d];
        wht_butterfly(qRot, keyLength);
        float whtNorm = rsqrtf((float)keyLength);
        for (int d = 0; d < keyLength; d++)
            qRot[d] *= whtNorm;
    }
    // Broadcast qRot to all threads via shared vShared (reuse temporarily)
    if (tid == 0)
        for (int d = 0; d < keyLength; d++) vShared[d] = qRot[d];
    __syncthreads();
    for (int d = 0; d < keyLength; d++) qRot[d] = vShared[d];
    __syncthreads();

    // Zero output
    float* outHead = output + head * valueLength;
    for (int d = tid; d < valueLength; d += stride)
        outHead[d] = 0.0f;
    __syncthreads();

    float running_max = -1e30f;
    float running_sum = 0.0f;

    // Thread-local dequant buffer
    float kBuf[128];
    float vBuf[128];

    for (int tile_start = 0; tile_start < seqLen; tile_start += TQ_ATTN_TILE_SIZE)
    {
        int tile_end = tile_start + TQ_ATTN_TILE_SIZE;
        if (tile_end > seqLen) tile_end = seqLen;
        int tile_len = tile_end - tile_start;

        // Step 1: K scores in rotated domain — NO per-position WHT!
        for (int t = tid; t < tile_len; t += stride)
        {
            int p = tile_start + t;
            const unsigned char* kData = kPacked + (kvHead * maxSeqLen + p) * kPackedPerHead;
            float kScale = kScales[kvHead * maxSeqLen + p];

            // Just centroid lookup — no WHT, no sign flips
            dequant_vector(kData, kBuf, keyLength, quantBits);

            // Dot product in rotated domain: q_rot · k_centroids × scale_k
            float dot = 0.0f;
            for (int d = 0; d < keyLength; d++)
                dot += qRot[d] * kBuf[d];
            scores[t] = dot * kScale * scale;
        }
        __syncthreads();

        // Step 2: Tile max via tree reduction
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

        // Step 3: Exp and sum
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

        // Step 4: Online softmax merge + weighted V sum
        float new_max = fmaxf(running_max, tile_max);
        float correction_old = expf(running_max - new_max);
        float correction_new = expf(tile_max - new_max);

        // Rescale old output
        for (int d = tid; d < valueLength; d += stride)
            outHead[d] *= correction_old;
        __syncthreads();

        // Accumulate weighted V centroids in rotated domain — NO per-position WHT!
        // outHead accumulates: Σ w_t × scale_vt × v_centroids_t
        // Thread 0 decompresses centroids (just lookup, no WHT), broadcasts via vShared
        for (int t = 0; t < tile_len; t++)
        {
            if (tid == 0)
            {
                int p = tile_start + t;
                const unsigned char* vData = vPacked + (kvHead * maxSeqLen + p) * vPackedPerHead;
                float vScale = vScales[kvHead * maxSeqLen + p];

                // Just centroid lookup — no WHT, no signs
                dequant_vector(vData, vBuf, valueLength, quantBits);

                for (int dd = 0; dd < valueLength; dd++)
                    vShared[dd] = vBuf[dd] * vScale;
            }
            __syncthreads();

            float w = scores[t] * correction_new;
            for (int d = tid; d < valueLength; d += stride)
                outHead[d] += w * vShared[d];
            __syncthreads();
        }

        running_sum = running_sum * correction_old + tile_sum * correction_new;
        running_max = new_max;
    }

    // Step 5: Inverse WHT the accumulated rotated V, normalize, gate
    // output = D_v · (1/√n) · H · (accumulated_centroids / attention_sum)
    float inv_sum = (running_sum > 0.0f) ? 1.0f / running_sum : 0.0f;

    // Thread 0 applies inverse WHT to accumulated output
    if (tid == 0)
    {
        for (int d = 0; d < valueLength; d++)
            vBuf[d] = outHead[d] * inv_sum;

        float whtNorm = rsqrtf((float)valueLength);
        for (int d = 0; d < valueLength; d++)
            vBuf[d] *= whtNorm;
        wht_butterfly(vBuf, valueLength);
        apply_signs(vBuf, vSigns, valueLength);

        for (int d = 0; d < valueLength; d++)
            vShared[d] = vBuf[d];
    }
    __syncthreads();

    // All threads write gated output
    for (int d = tid; d < valueLength; d += stride)
    {
        float g = (d < keyLength) ? qg[d] : 0.0f;
        float sig = 1.0f / (1.0f + expf(-g));
        outHead[d] = vShared[d] * sig;
    }
}

} // extern "C"
