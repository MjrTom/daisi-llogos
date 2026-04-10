using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace Daisi.Llogos.Cpu;

/// <summary>
/// Matrix multiplication with optional fused dequantization.
/// b is always [N×K] (GGUF convention: each of N output rows has K input weights).
/// output[i,j] = dot(a[i,:], b[j,:])
/// </summary>
internal static class MatMul
{
    private const int Q8_0BlockSize = 32;
    private const int Q8_0TypeSize = 34;

    private const int Q4_0BlockSize = 32;
    private const int Q4_0TypeSize = 18;

    private const int Q4_KSuperBlockSize = 256;
    private const int Q4_KTypeSize = 144;

    // Minimum N to justify thread pool dispatch overhead.
    // On a 24-thread CPU, at N=32 each thread does ~1 row — dispatch still wins
    // over serial for the attention/PLE projections. Larger thresholds (512+)
    // hurt medium matmuls without helping the FFN, which is already memory-bound.
    private const int ParallelThreshold = 32;

    /// <summary>
    /// FP32 × FP32 matrix multiply. b is [N×K].
    /// </summary>
    public static void Multiply(Span<float> output, ReadOnlySpan<float> a, ReadOnlySpan<float> b, int M, int K, int N)
    {
        if (Fma.IsSupported && K >= 8)
            MultiplyFma(output, a, b, M, K, N);
        else
            MultiplyScalar(output, a, b, M, K, N);
    }

    /// <summary>
    /// FP32 × Q4_0 fused dequant+matmul. b is [N×K] quantized.
    /// Each Q4_0 block is 2 bytes FP16 scale + 16 packed bytes (32 × 4-bit nibbles), 18 bytes total.
    /// Low nibble of byte i → element i; high nibble of byte i → element i+16.
    /// Quants are [0..15] re-centered by subtracting 8 → signed [-8..7].
    ///
    /// Fast path: quantizes the activation row(s) to Q8_0, then uses an int8×int8 dot product
    /// kernel via the sign_epi8 + maddubs_epi16 + madd_epi16 trick (matches llama.cpp's AVX2
    /// path on Alder Lake without requiring VNNI).
    ///
    /// For M &gt; 1 (batched prefill) a weight-reuse inner loop amortises the weight block
    /// decode across all M activation rows — per-block cost drops from ~10 cycles × M to
    /// ~5 cycles (shared) + ~4 cycles × M.
    /// </summary>
    public static unsafe void MultiplyQ4_0(Span<float> output, ReadOnlySpan<float> a, ReadOnlySpan<byte> b, int M, int K, int N)
    {
        if (M > 1)
        {
            MultiplyQ4_0_Batched(output, a, b, M, K, N);
            return;
        }

        // M == 1 fast path (decode) — inline to preserve the exact instruction
        // sequence the JIT was producing before the batched path was added.
        int blocksPerRow = K / Q4_0BlockSize;
        int bytesPerRow = blocksPerRow * Q4_0TypeSize;
        int q8BytesPerInput = blocksPerRow * Q8_0TypeSize;

        fixed (float* aFixedPtr = a)
        fixed (byte* bFixedPtr = b)
        fixed (float* oFixedPtr = output)
        {
            nint aBase = (nint)aFixedPtr;
            nint bBase = (nint)bFixedPtr;
            nint oBase = (nint)oFixedPtr;

            byte* aQBuf = stackalloc byte[q8BytesPerInput];

            float* aRow = (float*)aBase;
            float* oRow = (float*)oBase;

            QuantizeRowQ8_0(aRow, aQBuf, K);

            nint aQBase = (nint)aQBuf;
            int blkCapture = blocksPerRow;

            if (N >= ParallelThreshold)
            {
                nint bCapture = bBase;
                int bprCapture = bytesPerRow;
                Parallel.For(0, N, CpuThreading.Options, j =>
                {
                    byte* bRow = (byte*)bCapture + j * bprCapture;
                    oRow[j] = DotQ4_0Q8_0Ptr((byte*)aQBase, bRow, blkCapture);
                });
            }
            else
            {
                for (int j = 0; j < N; j++)
                {
                    byte* bRow = (byte*)bBase + j * bytesPerRow;
                    oRow[j] = DotQ4_0Q8_0Ptr((byte*)aQBase, bRow, blkCapture);
                }
            }
        }
    }

    /// <summary>
    /// M &gt; 1 batched path: quantize all M activation rows once, then for each
    /// weight row iterate blocks where each block is decoded ONCE and used for
    /// all M activation rows. Memory-bound workloads get ~M× speedup from reduced
    /// weight memory traffic.
    /// </summary>
    private static unsafe void MultiplyQ4_0_Batched(Span<float> output, ReadOnlySpan<float> a, ReadOnlySpan<byte> b, int M, int K, int N)
    {
        int blocksPerRow = K / Q4_0BlockSize;
        int bytesPerRow = blocksPerRow * Q4_0TypeSize;
        int q8BytesPerInput = blocksPerRow * Q8_0TypeSize;

        // Rent a heap buffer for M quantized activations — M=64 × K=2560 ~ 175 KB,
        // too big for the stack.
        byte[] aQBufArr = System.Buffers.ArrayPool<byte>.Shared.Rent(M * q8BytesPerInput);
        try
        {
            fixed (float* aFixedPtr = a)
            fixed (byte* bFixedPtr = b)
            fixed (float* oFixedPtr = output)
            fixed (byte* aQFixedPtr = aQBufArr)
            {
                // Quantize all M activation rows up-front.
                for (int i = 0; i < M; i++)
                    QuantizeRowQ8_0(aFixedPtr + i * K, aQFixedPtr + i * q8BytesPerInput, K);

                nint aQBase = (nint)aQFixedPtr;
                nint bBase = (nint)bFixedPtr;
                nint oBase = (nint)oFixedPtr;
                int mCap = M;
                int blkCap = blocksPerRow;
                int bprCap = bytesPerRow;
                int q8StrideCap = q8BytesPerInput;
                int nCap = N;

                if (N >= ParallelThreshold)
                {
                    Parallel.For(0, N, CpuThreading.Options, j =>
                    {
                        byte* bRow = (byte*)bBase + j * bprCap;
                        DotQ4_0Q8_0BatchedAvx2((byte*)aQBase, bRow, blkCap, mCap, q8StrideCap,
                            (float*)oBase + j, nCap);
                    });
                }
                else
                {
                    for (int j = 0; j < N; j++)
                    {
                        byte* bRow = (byte*)bFixedPtr + j * bytesPerRow;
                        DotQ4_0Q8_0BatchedAvx2(aQFixedPtr, bRow, blocksPerRow, M, q8BytesPerInput,
                            oFixedPtr + j, N);
                    }
                }
            }
        }
        finally
        {
            System.Buffers.ArrayPool<byte>.Shared.Return(aQBufArr);
        }
    }

    /// <summary>
    /// Batched Q4_0 × Q8_0 inner kernel: decode one weight row (blockCount Q4_0 blocks)
    /// and multiply against M pre-quantized activation rows in a single pass. Writes
    /// M outputs at <paramref name="outPtr"/>[0], <paramref name="outPtr"/>[N],
    /// <paramref name="outPtr"/>[2*N], … (strided by N so adjacent j iterations write
    /// into the right output[m × N + j] slots).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void DotQ4_0Q8_0BatchedAvx2(
        byte* aQ, byte* bQ4, int blockCount, int M, int aQStride,
        float* outPtr, int outStride)
    {
        // Register-tiled inner loop: process M in tiles of 4 activations at a time.
        // 4 is the sweet spot on Alder Lake — 4 accumulators fit alongside weight
        // decode state without spilling, and empirically beat both 2-wide (more
        // weight re-reads) and 8-wide (register spills from insufficient YMM slots).
        int mBase = 0;
        while (mBase + 4 <= M)
        {
            DotQ4_0Q8_0Batched4(aQ, bQ4, blockCount, mBase, aQStride, outPtr, outStride);
            mBase += 4;
        }
        if (mBase < M)
            DotQ4_0Q8_0BatchedTail(aQ, bQ4, blockCount, mBase, M - mBase, aQStride, outPtr, outStride);
    }


    /// <summary>
    /// Inner 4×1 tile: 4 activation rows × 1 weight row. Accumulators stay in registers.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void DotQ4_0Q8_0Batched4(
        byte* aQ, byte* bQ4, int blockCount, int mBase, int aQStride,
        float* outPtr, int outStride)
    {
        var mask0x0F = Vector256.Create((byte)0x0F);
        var offset8 = Vector256.Create((sbyte)8);
        var ones = Vector256.Create((short)1);

        var acc0 = Vector256<float>.Zero;
        var acc1 = Vector256<float>.Zero;
        var acc2 = Vector256<float>.Zero;
        var acc3 = Vector256<float>.Zero;

        byte* aBase0 = aQ + (mBase + 0) * aQStride;
        byte* aBase1 = aQ + (mBase + 1) * aQStride;
        byte* aBase2 = aQ + (mBase + 2) * aQStride;
        byte* aBase3 = aQ + (mBase + 3) * aQStride;

        for (int bi = 0; bi < blockCount; bi++)
        {
            byte* bBlk = bQ4 + bi * Q4_0TypeSize;
            float bScale = (float)Unsafe.ReadUnaligned<Half>(bBlk);

            // Decode weight block once per 4-activation tile.
            var qx_lo128 = Sse2.LoadVector128(bBlk + 2);
            var qx_hi128 = Sse2.ShiftRightLogical(qx_lo128.AsUInt16(), 4).AsByte();
            var qx = Vector256.Create(qx_lo128, qx_hi128);
            var qxMasked = Avx2.And(qx, mask0x0F);
            var qxSigned = Avx2.Subtract(qxMasked.AsSByte(), offset8);
            var ax = Avx2.Sign(qxSigned, qxSigned);
            var axB = ax.AsByte();

            // 4 independent int8 dots — JIT should pipeline these
            byte* a0 = aBase0 + bi * Q8_0TypeSize;
            byte* a1 = aBase1 + bi * Q8_0TypeSize;
            byte* a2 = aBase2 + bi * Q8_0TypeSize;
            byte* a3 = aBase3 + bi * Q8_0TypeSize;

            float s0 = (float)Unsafe.ReadUnaligned<Half>(a0);
            float s1 = (float)Unsafe.ReadUnaligned<Half>(a1);
            float s2 = (float)Unsafe.ReadUnaligned<Half>(a2);
            float s3 = (float)Unsafe.ReadUnaligned<Half>(a3);

            var qy0 = Avx.LoadVector256((sbyte*)(a0 + 2));
            var qy1 = Avx.LoadVector256((sbyte*)(a1 + 2));
            var qy2 = Avx.LoadVector256((sbyte*)(a2 + 2));
            var qy3 = Avx.LoadVector256((sbyte*)(a3 + 2));

            var sy0 = Avx2.Sign(qy0, qxSigned);
            var sy1 = Avx2.Sign(qy1, qxSigned);
            var sy2 = Avx2.Sign(qy2, qxSigned);
            var sy3 = Avx2.Sign(qy3, qxSigned);

            var d160 = Avx2.MultiplyAddAdjacent(axB, sy0);
            var d161 = Avx2.MultiplyAddAdjacent(axB, sy1);
            var d162 = Avx2.MultiplyAddAdjacent(axB, sy2);
            var d163 = Avx2.MultiplyAddAdjacent(axB, sy3);

            var d320 = Avx2.MultiplyAddAdjacent(d160, ones);
            var d321 = Avx2.MultiplyAddAdjacent(d161, ones);
            var d322 = Avx2.MultiplyAddAdjacent(d162, ones);
            var d323 = Avx2.MultiplyAddAdjacent(d163, ones);

            var f0 = Avx.ConvertToVector256Single(d320);
            var f1 = Avx.ConvertToVector256Single(d321);
            var f2 = Avx.ConvertToVector256Single(d322);
            var f3 = Avx.ConvertToVector256Single(d323);

            var vD0 = Vector256.Create(s0 * bScale);
            var vD1 = Vector256.Create(s1 * bScale);
            var vD2 = Vector256.Create(s2 * bScale);
            var vD3 = Vector256.Create(s3 * bScale);

            acc0 = Fma.MultiplyAdd(f0, vD0, acc0);
            acc1 = Fma.MultiplyAdd(f1, vD1, acc1);
            acc2 = Fma.MultiplyAdd(f2, vD2, acc2);
            acc3 = Fma.MultiplyAdd(f3, vD3, acc3);
        }

        outPtr[(mBase + 0) * outStride] = Vector256.Sum(acc0);
        outPtr[(mBase + 1) * outStride] = Vector256.Sum(acc1);
        outPtr[(mBase + 2) * outStride] = Vector256.Sum(acc2);
        outPtr[(mBase + 3) * outStride] = Vector256.Sum(acc3);
    }

    /// <summary>
    /// Tail for M not divisible by 4 — process remaining rows one at a time.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void DotQ4_0Q8_0BatchedTail(
        byte* aQ, byte* bQ4, int blockCount, int mBase, int tileSize, int aQStride,
        float* outPtr, int outStride)
    {
        for (int i = 0; i < tileSize; i++)
        {
            byte* aRow = aQ + (mBase + i) * aQStride;
            outPtr[(mBase + i) * outStride] = DotQ4_0Q8_0Ptr(aRow, bQ4, blockCount);
        }
    }

    /// <summary>
    /// Quantize a row of FP32 values to Q8_0 format in place.
    /// Each 32-float block becomes 2 bytes FP16 scale + 32 signed bytes.
    /// Scale = max|a[i]| / 127, then q[i] = round(a[i] / scale).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void QuantizeRowQ8_0(float* a, byte* aQ, int K)
    {
        int blockCount = K / Q8_0BlockSize;

        for (int b = 0; b < blockCount; b++)
        {
            float* aBlk = a + b * Q8_0BlockSize;
            byte* outBlk = aQ + b * Q8_0TypeSize;

            // Find max |a| over 32 elements (vectorized)
            if (Avx.IsSupported)
            {
                var absMask = Vector256.Create(0x7FFFFFFFu).AsSingle();
                var maxAbs = Avx.And(Avx.LoadVector256(aBlk), absMask);
                maxAbs = Avx.Max(maxAbs, Avx.And(Avx.LoadVector256(aBlk + 8), absMask));
                maxAbs = Avx.Max(maxAbs, Avx.And(Avx.LoadVector256(aBlk + 16), absMask));
                maxAbs = Avx.Max(maxAbs, Avx.And(Avx.LoadVector256(aBlk + 24), absMask));

                // Horizontal max across 8 lanes
                var hi = maxAbs.GetUpper();
                var lo = maxAbs.GetLower();
                var m128 = Sse.Max(lo, hi);
                m128 = Sse.Max(m128, Sse.Shuffle(m128, m128, 0b11_10_11_10));
                m128 = Sse.Max(m128, Sse.Shuffle(m128, m128, 0b01_01_01_01));
                float amax = m128.ToScalar();

                float scale = amax / 127.0f;
                float invScale = scale > 0 ? 1.0f / scale : 0;

                *(Half*)outBlk = (Half)scale;
                sbyte* qPtr = (sbyte*)(outBlk + 2);

                var vInvScale = Vector256.Create(invScale);

                // Quantize and convert to int8. Process 32 floats in 4 chunks of 8.
                // Use 128-bit PACKSSDW/PACKSSWB to avoid AVX2's lane-wise packing pitfall.
                for (int k = 0; k < 4; k++)
                {
                    var v = Avx.LoadVector256(aBlk + k * 8);
                    var scaled = Avx.Multiply(v, vInvScale);
                    var rounded = Avx.RoundToNearestInteger(scaled);
                    var ints = Avx.ConvertToVector256Int32(rounded);

                    // Pack 8 int32 → 8 int16 via 128-bit halves (avoids AVX2 lane trap)
                    var loHalf = ints.GetLower();   // int32[4]
                    var hiHalf = ints.GetUpper();   // int32[4]
                    var shorts128 = Sse2.PackSignedSaturate(loHalf, hiHalf);  // int16[8]
                    // Pack 8 int16 → 8 int8 (low 64 bits of the result)
                    var bytes128 = Sse2.PackSignedSaturate(shorts128, shorts128);  // int8[16], low 8 are what we want
                    long lane = bytes128.AsInt64().ToScalar();
                    *(long*)(qPtr + k * 8) = lane;
                }
            }
            else
            {
                float amax = 0;
                for (int i = 0; i < Q8_0BlockSize; i++)
                {
                    float av = MathF.Abs(aBlk[i]);
                    if (av > amax) amax = av;
                }
                float scale = amax / 127.0f;
                float invScale = scale > 0 ? 1.0f / scale : 0;
                *(Half*)outBlk = (Half)scale;
                sbyte* qPtr = (sbyte*)(outBlk + 2);
                for (int i = 0; i < Q8_0BlockSize; i++)
                {
                    qPtr[i] = (sbyte)Math.Clamp((int)MathF.Round(aBlk[i] * invScale), -128, 127);
                }
            }
        }
    }

    /// <summary>
    /// Quantized int8 × int8 dot product for Q4_0 weights × Q8_0 activations.
    /// Uses the llama.cpp AVX2 sign-trick to avoid needing VNNI.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe float DotQ4_0Q8_0Ptr(byte* aQ8, byte* bQ4, int blockCount)
    {
        // Note: AVX-VNNI is available on Alder Lake but in our tests it produced
        // a 2× decode regression vs the AVX2 sign+maddubs+madd path. Likely because
        // the vpdpbusd port contention with the FMA unit on these SKUs creates
        // pipeline stalls. Keep the AVX2 path as the default.
        if (Avx2.IsSupported && Fma.IsSupported)
            return DotQ4_0Q8_0Avx2(aQ8, bQ4, blockCount);
        return DotQ4_0Q8_0Scalar(aQ8, bQ4, blockCount);
    }

    /// <summary>
    /// AVX-VNNI fused Q4_0 × Q8_0 dot product — available on Alder Lake and newer
    /// Intel/AMD (256-bit VNNI doesn't require AVX-512). Replaces the sign_epi8 +
    /// maddubs_epi16 + madd_epi16 chain with a single vpdpbusd instruction per
    /// block, cutting the critical path from ~5 ops to ~1 + 1 FMA.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe float DotQ4_0Q8_0Vnni(byte* aQ8, byte* bQ4, int blockCount)
    {
        var acc = Vector256<float>.Zero;
        var mask0x0F = Vector256.Create((byte)0x0F);
        var offset8 = Vector256.Create((sbyte)8);

        for (int bi = 0; bi < blockCount; bi++)
        {
            byte* aBlk = aQ8 + bi * Q8_0TypeSize;
            byte* bBlk = bQ4 + bi * Q4_0TypeSize;

            float aScale = (float)Unsafe.ReadUnaligned<Half>(aBlk);
            float bScale = (float)Unsafe.ReadUnaligned<Half>(bBlk);
            var vD = Vector256.Create(aScale * bScale);

            // Decode Q4_0 block: 16 packed bytes → 32 nibbles in [0..15]
            var qx_lo128 = Sse2.LoadVector128(bBlk + 2);
            var qx_hi128 = Sse2.ShiftRightLogical(qx_lo128.AsUInt16(), 4).AsByte();
            var qx = Vector256.Create(qx_lo128, qx_hi128);
            var qxFinal = Avx2.And(qx, mask0x0F);                         // [0..15] bytes
            var qxSigned = Avx2.Subtract(qxFinal.AsSByte(), offset8);     // [-8..7] sbyte

            // Load 32 int8 activation values
            var qy = Avx.LoadVector256((sbyte*)(aBlk + 2));

            // vpdpbusd wants (unsigned_byte, signed_byte). Fold sign of qx into qy:
            //   ax = |qxSigned|                           (unsigned byte [0..8])
            //   sy = sign(qy, qxSigned)                   (signed byte, qy × sign(qx))
            // Then acc32 += vpdpbusd(ax, sy)
            var ax = Avx2.Sign(qxSigned, qxSigned);
            var sy = Avx2.Sign(qy, qxSigned);
            var dot32 = AvxVnni.MultiplyWideningAndAdd(Vector256<int>.Zero, ax.AsByte(), sy);

            var dotF = Avx.ConvertToVector256Single(dot32);
            acc = Fma.MultiplyAdd(dotF, vD, acc);
        }

        return Vector256.Sum(acc);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe float DotQ4_0Q8_0Avx2(byte* aQ8, byte* bQ4, int blockCount)
    {
        var acc = Vector256<float>.Zero;
        var mask0x0F = Vector256.Create((byte)0x0F);
        var offset8 = Vector256.Create((sbyte)8);
        var ones = Vector256.Create((short)1);

        for (int bi = 0; bi < blockCount; bi++)
        {
            byte* aBlk = aQ8 + bi * Q8_0TypeSize;
            byte* bBlk = bQ4 + bi * Q4_0TypeSize;

            // Combined scale = aScale * bScale (both FP16 → FP32)
            float aScale = (float)Unsafe.ReadUnaligned<Half>(aBlk);
            float bScale = (float)Unsafe.ReadUnaligned<Half>(bBlk);
            var vD = Vector256.Create(aScale * bScale);

            // Load 16 packed bytes of Q4_0 into lower 128 bits and the shifted-by-4 copy
            // into upper 128 bits. After the AND, lower has low nibbles, upper has high nibbles.
            var qx_lo128 = Sse2.LoadVector128(bBlk + 2);
            var qx_hi128 = Sse2.ShiftRightLogical(qx_lo128.AsUInt16(), 4).AsByte();
            var qx = Vector256.Create(qx_lo128, qx_hi128);            // [low nibs raw | high nibs raw]
            var qxFinal = Avx2.And(qx, mask0x0F);                     // nibbles [0..15]
            var qxSigned = Avx2.Subtract(qxFinal.AsSByte(), offset8); // [-8..7]

            // Load 32 int8 activation values
            var qy = Avx.LoadVector256((sbyte*)(aBlk + 2));

            // Sign trick for AVX2 int8×int8 without VNNI:
            //   ax = sign_epi8(qx, qx)  → |qx| (unsigned magnitude)
            //   sy = sign_epi8(qy, qx)  → qy with sign flipped where qx is negative
            //   dot = maddubs_epi16(ax, sy)   → 16 int16 pairs (pair-summed)
            //   sum = madd_epi16(dot, ones)   → 8 int32 (horizontal pair-sum)
            var ax = Avx2.Sign(qxSigned, qxSigned);
            var sy = Avx2.Sign(qy, qxSigned);
            var dot16 = Avx2.MultiplyAddAdjacent(ax.AsByte(), sy);
            var dot32 = Avx2.MultiplyAddAdjacent(dot16, ones);

            var dotF = Avx.ConvertToVector256Single(dot32);
            acc = Fma.MultiplyAdd(dotF, vD, acc);
        }

        return Vector256.Sum(acc);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe float DotQ4_0Q8_0Scalar(byte* aQ8, byte* bQ4, int blockCount)
    {
        float total = 0;
        for (int b = 0; b < blockCount; b++)
        {
            byte* aBlk = aQ8 + b * Q8_0TypeSize;
            byte* bBlk = bQ4 + b * Q4_0TypeSize;
            float aScale = (float)Unsafe.ReadUnaligned<Half>(aBlk);
            float bScale = (float)Unsafe.ReadUnaligned<Half>(bBlk);
            sbyte* qy = (sbyte*)(aBlk + 2);
            byte* packed = bBlk + 2;

            int isum = 0;
            for (int i = 0; i < 16; i++)
            {
                byte p = packed[i];
                int qLo = (p & 0x0F) - 8;
                int qHi = (p >> 4) - 8;
                isum += qLo * qy[i] + qHi * qy[i + 16];
            }
            total += aScale * bScale * isum;
        }
        return total;
    }

    /// <summary>
    /// FP32 × Q8_0 fused dequant+matmul. b is [N×K] quantized.
    /// Multi-threaded over N when N is large enough.
    /// </summary>
    public static unsafe void MultiplyQ8_0(Span<float> output, ReadOnlySpan<float> a, ReadOnlySpan<byte> b, int M, int K, int N)
    {
        int blocksPerRow = K / Q8_0BlockSize;
        int bytesPerRow = blocksPerRow * Q8_0TypeSize;

        fixed (float* aFixedPtr = a)
        fixed (byte* bFixedPtr = b)
        fixed (float* oFixedPtr = output)
        {
            // Copy to local to avoid fixed-in-lambda restriction
            nint aBase = (nint)aFixedPtr;
            nint bBase = (nint)bFixedPtr;
            nint oBase = (nint)oFixedPtr;

            for (int i = 0; i < M; i++)
            {
                float* aRow = (float*)aBase + i * K;
                float* oRow = (float*)oBase + i * N;

                if (N >= ParallelThreshold)
                {
                    nint bCapture = bBase;
                    int bprCapture = bytesPerRow;
                    int blkCapture = blocksPerRow;
                    Parallel.For(0, N, CpuThreading.Options, j =>
                    {
                        byte* bRow = (byte*)bCapture + j * bprCapture;
                        oRow[j] = DotQ8_0Ptr(aRow, bRow, blkCapture);
                    });
                }
                else
                {
                    for (int j = 0; j < N; j++)
                    {
                        byte* bRow = (byte*)bBase + j * bytesPerRow;
                        oRow[j] = DotQ8_0Ptr(aRow, bRow, blocksPerRow);
                    }
                }
            }
        }
    }

    /// <summary>
    /// FP32 × BF16 matrix multiply. b is [N×K] stored as BF16 (top 16 bits of FP32).
    /// Used by Gemma4's per_layer_model_proj tensor.
    /// Conversion BF16→FP32 is a single left-shift by 16 (no exponent adjust).
    /// </summary>
    public static unsafe void MultiplyBF16(Span<float> output, ReadOnlySpan<float> a, ReadOnlySpan<byte> b, int M, int K, int N)
    {
        if (M > 1)
        {
            MultiplyBF16_Batched(output, a, b, M, K, N);
            return;
        }

        // M == 1 fast path (decode / PLE on single-token Forward)
        fixed (float* aFixedPtr = a)
        fixed (byte* bFixedPtr = b)
        fixed (float* oFixedPtr = output)
        {
            nint aBase = (nint)aFixedPtr;
            nint bBase = (nint)bFixedPtr;
            nint oBase = (nint)oFixedPtr;

            float* aRow = (float*)aBase;
            float* oRow = (float*)oBase;

            if (N >= ParallelThreshold)
            {
                nint bCapture = bBase;
                int kCapture = K;
                Parallel.For(0, N, CpuThreading.Options, j =>
                {
                    ushort* bRow = (ushort*)((byte*)bCapture + j * kCapture * 2);
                    oRow[j] = DotBF16Ptr(aRow, bRow, kCapture);
                });
            }
            else
            {
                for (int j = 0; j < N; j++)
                {
                    ushort* bRow = (ushort*)((byte*)bBase + j * K * 2);
                    oRow[j] = DotBF16Ptr(aRow, bRow, K);
                }
            }
        }
    }

    /// <summary>
    /// Batched BF16 matmul with weight reuse. For each weight row j, decode 8
    /// BF16 lanes at a time and FMA into 4 activation accumulators. 4-wide
    /// register tile keeps accumulators in YMM registers. Used by the batched
    /// Gemma 4 forward pass's PLE setup (per_layer_model_proj: K=HiddenDim,
    /// N=PerLayerInputDim × NumLayers).
    /// </summary>
    private static unsafe void MultiplyBF16_Batched(Span<float> output, ReadOnlySpan<float> a, ReadOnlySpan<byte> b, int M, int K, int N)
    {
        fixed (float* aFixedPtr = a)
        fixed (byte* bFixedPtr = b)
        fixed (float* oFixedPtr = output)
        {
            nint aBase = (nint)aFixedPtr;
            nint bBase = (nint)bFixedPtr;
            nint oBase = (nint)oFixedPtr;
            int mCap = M;
            int kCap = K;
            int nCap = N;

            if (N >= ParallelThreshold)
            {
                Parallel.For(0, N, CpuThreading.Options, j =>
                {
                    ushort* bRow = (ushort*)((byte*)bBase + j * kCap * 2);
                    DotBF16BatchedAvx2((float*)aBase, bRow, kCap, mCap, (float*)oBase + j, nCap);
                });
            }
            else
            {
                for (int j = 0; j < N; j++)
                {
                    ushort* bRow = (ushort*)((byte*)bFixedPtr + j * K * 2);
                    DotBF16BatchedAvx2(aFixedPtr, bRow, K, M, oFixedPtr + j, N);
                }
            }
        }
    }

    /// <summary>
    /// Batched BF16 inner kernel: one weight row × M activation rows. Weight
    /// values are loaded once per 8-lane chunk and FMA'd into 4-wide register
    /// tiles of activation accumulators.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void DotBF16BatchedAvx2(
        float* aBase, ushort* bRow, int K, int M, float* outPtr, int outStride)
    {
        // Process M in tiles of 4 — 4 accumulators stay in YMM registers.
        int mBase = 0;
        while (mBase + 4 <= M)
        {
            DotBF16BatchedAvx2Tile4(aBase + mBase * K, aBase + (mBase + 1) * K,
                aBase + (mBase + 2) * K, aBase + (mBase + 3) * K,
                bRow, K, outPtr + mBase * outStride, outStride);
            mBase += 4;
        }
        // Tail for remaining rows
        for (; mBase < M; mBase++)
            outPtr[mBase * outStride] = DotBF16Ptr(aBase + mBase * K, bRow, K);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void DotBF16BatchedAvx2Tile4(
        float* a0, float* a1, float* a2, float* a3,
        ushort* bRow, int K, float* outPtr, int outStride)
    {
        var acc0 = Vector256<float>.Zero;
        var acc1 = Vector256<float>.Zero;
        var acc2 = Vector256<float>.Zero;
        var acc3 = Vector256<float>.Zero;
        var mask16 = Vector256.Create(0x0000FFFFu);
        int k = 0;

        for (; k + 8 <= K; k += 8)
        {
            // Load 8 BF16 values → 8 FP32 via shift-left-16
            var bf = Vector128.Load(bRow + k);                                        // 8 × uint16
            var w = Avx2.ConvertToVector256Int32(bf.AsInt16()).AsUInt32();             // sign-extends
            w = Avx2.And(w, mask16);                                                   // keep low 16 bits
            var f = Avx2.ShiftLeftLogical(w, 16).AsSingle();                           // → FP32

            var v0 = Avx.LoadVector256(a0 + k);
            var v1 = Avx.LoadVector256(a1 + k);
            var v2 = Avx.LoadVector256(a2 + k);
            var v3 = Avx.LoadVector256(a3 + k);

            acc0 = Fma.MultiplyAdd(v0, f, acc0);
            acc1 = Fma.MultiplyAdd(v1, f, acc1);
            acc2 = Fma.MultiplyAdd(v2, f, acc2);
            acc3 = Fma.MultiplyAdd(v3, f, acc3);
        }

        float s0 = Vector256.Sum(acc0);
        float s1 = Vector256.Sum(acc1);
        float s2 = Vector256.Sum(acc2);
        float s3 = Vector256.Sum(acc3);

        for (; k < K; k++)
        {
            float fv = BitConverter.UInt32BitsToSingle(((uint)bRow[k]) << 16);
            s0 += a0[k] * fv;
            s1 += a1[k] * fv;
            s2 += a2[k] * fv;
            s3 += a3[k] * fv;
        }

        outPtr[0] = s0;
        outPtr[outStride] = s1;
        outPtr[outStride * 2] = s2;
        outPtr[outStride * 3] = s3;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe float DotBF16Ptr(float* aVec, ushort* bBf16, int K)
    {
        if (Avx2.IsSupported && Fma.IsSupported)
            return DotBF16Avx2(aVec, bBf16, K);
        return DotBF16Scalar(aVec, bBf16, K);
    }

    /// <summary>
    /// AVX2 fused BF16 dot product. BF16 is the top 16 bits of FP32, so
    /// conversion is: load 16 uint16, zero-extend to uint32, shift left by 16.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe float DotBF16Avx2(float* aVec, ushort* bBf16, int K)
    {
        var acc0 = Vector256<float>.Zero;
        var acc1 = Vector256<float>.Zero;
        int k = 0;

        // Process 16 elements per iteration (2 × 8-wide FMA)
        for (; k + 16 <= K; k += 16)
        {
            // Load 16 × uint16 into 256-bit register (two 128-bit halves)
            var bf0 = Vector128.Load(bBf16 + k);      // 8 × uint16
            var bf1 = Vector128.Load(bBf16 + k + 8);  // 8 × uint16

            // Zero-extend uint16 → uint32 (8 lanes each), then shift left by 16
            var w0 = Avx2.ConvertToVector256Int32(bf0.AsInt16()).AsUInt32();  // sign-extend is fine, high bits masked next
            var w1 = Avx2.ConvertToVector256Int32(bf1.AsInt16()).AsUInt32();
            // Strip sign-extended high bits
            var mask16 = Vector256.Create(0x0000FFFFu);
            w0 = Avx2.And(w0, mask16);
            w1 = Avx2.And(w1, mask16);
            // Shift into FP32 position
            var f0 = Avx2.ShiftLeftLogical(w0, 16).AsSingle();
            var f1 = Avx2.ShiftLeftLogical(w1, 16).AsSingle();

            var a0 = Avx.LoadVector256(aVec + k);
            var a1 = Avx.LoadVector256(aVec + k + 8);

            acc0 = Fma.MultiplyAdd(a0, f0, acc0);
            acc1 = Fma.MultiplyAdd(a1, f1, acc1);
        }

        float sum = Vector256.Sum(acc0 + acc1);

        for (; k < K; k++)
        {
            uint bits = ((uint)bBf16[k]) << 16;
            sum += aVec[k] * BitConverter.UInt32BitsToSingle(bits);
        }

        return sum;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe float DotBF16Scalar(float* aVec, ushort* bBf16, int K)
    {
        float sum = 0;
        for (int k = 0; k < K; k++)
        {
            uint bits = ((uint)bBf16[k]) << 16;
            sum += aVec[k] * BitConverter.UInt32BitsToSingle(bits);
        }
        return sum;
    }

    /// <summary>
    /// FP32 × FP16 matrix multiply. b is [N×K] stored as Half.
    /// Used for tied embeddings (F16 token_embd as output weight).
    /// AVX2+F16C: vcvtph2ps converts 8 half→8 float, then FMA accumulate.
    /// </summary>
    public static unsafe void MultiplyF16(Span<float> output, ReadOnlySpan<float> a, ReadOnlySpan<Half> b, int M, int K, int N)
    {
        fixed (float* aPtr = a)
        fixed (Half* bPtr = b)
        fixed (float* oPtr = output)
        {
            nint bBase = (nint)bPtr;

            for (int i = 0; i < M; i++)
            {
                float* aRow = aPtr + i * K;
                float* oRow = oPtr + i * N;

                if (N >= ParallelThreshold)
                {
                    nint bCapture = bBase;
                    int kCapture = K;
                    Parallel.For(0, N, CpuThreading.Options, j =>
                    {
                        Half* bRow = (Half*)bCapture + j * kCapture;
                        oRow[j] = DotF16(aRow, bRow, kCapture);
                    });
                }
                else
                {
                    for (int j = 0; j < N; j++)
                    {
                        Half* bRow = bPtr + j * K;
                        oRow[j] = DotF16(aRow, bRow, K);
                    }
                }
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe float DotF16(float* aVec, Half* bHalf, int K)
    {
        if (Fma.IsSupported)
            return DotF16Fma(aVec, bHalf, K);
        return DotF16Scalar(aVec, bHalf, K);
    }

    /// <summary>
    /// FMA-accelerated FP16 dot product. Uses bit manipulation to convert FP16→FP32
    /// in integer domain (avoiding managed Half conversion overhead), then FMA accumulate.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe float DotF16Fma(float* aVec, Half* bHalf, int K)
    {
        var acc0 = Vector256<float>.Zero;
        var acc1 = Vector256<float>.Zero;
        ushort* bPtr = (ushort*)bHalf;
        int k = 0;

        // Process 16 elements per iteration (2 × 8-wide FMA)
        for (; k + 16 <= K; k += 16)
        {
            var f0 = ConvertHalf8ToFloat(bPtr + k);
            var f1 = ConvertHalf8ToFloat(bPtr + k + 8);
            var a0 = Avx.LoadVector256(aVec + k);
            var a1 = Avx.LoadVector256(aVec + k + 8);

            acc0 = Fma.MultiplyAdd(a0, f0, acc0);
            acc1 = Fma.MultiplyAdd(a1, f1, acc1);
        }

        for (; k + 8 <= K; k += 8)
        {
            var f = ConvertHalf8ToFloat(bPtr + k);
            var a = Avx.LoadVector256(aVec + k);
            acc0 = Fma.MultiplyAdd(a, f, acc0);
        }

        float sum = Vector256.Sum(acc0 + acc1);

        for (; k < K; k++)
            sum += aVec[k] * (float)bHalf[k];

        return sum;
    }

    /// <summary>
    /// Convert 8 FP16 values to 8 FP32 using AVX2 integer bit manipulation.
    /// FP16: 1 sign + 5 exp + 10 mantissa → FP32: 1 sign + 8 exp + 23 mantissa
    /// Handles normal values only (denorms treated as zero, which is fine for model weights).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe Vector256<float> ConvertHalf8ToFloat(ushort* src)
    {
        // Load 8 × uint16, zero-extend to 8 × uint32
        var h = Vector128.Create(
            src[0], src[1], src[2], src[3],
            src[4], src[5], src[6], src[7]);
        var w = Avx2.ConvertToVector256Int32(h.AsInt16()).AsUInt32();

        // Extract sign, exponent, mantissa
        var sign = Avx2.And(w, Vector256.Create(0x8000u));
        var exp = Avx2.And(w, Vector256.Create(0x7C00u));
        var man = Avx2.And(w, Vector256.Create(0x03FFu));

        // Shift to FP32 positions: sign << 16, exp << 13, man << 13
        // But exp needs bias adjustment: FP16 bias=15, FP32 bias=127, diff=112
        // New exp = (old_exp << 13) + (112 << 23)
        var signF = Avx2.ShiftLeftLogical(sign, 16);
        var expF = Avx2.ShiftLeftLogical(exp, 13);
        var manF = Avx2.ShiftLeftLogical(man, 13);

        // Add exp bias (112 << 23 = 0x38000000) only for non-zero exponents
        var expBias = Vector256.Create(0x38000000u);
        var isZeroExp = Avx2.CompareEqual(exp, Vector256<uint>.Zero);
        var biasedExp = Avx2.Add(expF, Avx2.AndNot(isZeroExp, expBias));

        var result = Avx2.Or(Avx2.Or(signF, biasedExp), manF);
        return result.AsSingle();
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe float DotF16Scalar(float* aVec, Half* bHalf, int K)
    {
        float sum = 0;
        for (int k = 0; k < K; k++)
            sum += aVec[k] * (float)bHalf[k];
        return sum;
    }

    // ── FP32 Scalar ──────────────────────────────────────────────────────────

    private static void MultiplyScalar(Span<float> output, ReadOnlySpan<float> a, ReadOnlySpan<float> b, int M, int K, int N)
    {
        for (int i = 0; i < M; i++)
        {
            for (int j = 0; j < N; j++)
            {
                float sum = 0;
                int aOff = i * K;
                int bOff = j * K;
                for (int k = 0; k < K; k++)
                    sum += a[aOff + k] * b[bOff + k];
                output[i * N + j] = sum;
            }
        }
    }

    // ── FP32 FMA ──────────────────────────────────────────────────────────────

    private static void MultiplyFma(Span<float> output, ReadOnlySpan<float> a, ReadOnlySpan<float> b, int M, int K, int N)
    {
        ref float aRef = ref MemoryMarshal.GetReference(a);
        ref float bRef = ref MemoryMarshal.GetReference(b);
        ref float oRef = ref MemoryMarshal.GetReference(output);

        for (int i = 0; i < M; i++)
        {
            int aBase = i * K;
            for (int j = 0; j < N; j++)
            {
                int bBase = j * K;
                var vSum = Vector256<float>.Zero;
                int k = 0;
                for (; k + 8 <= K; k += 8)
                {
                    var vA = Vector256.LoadUnsafe(ref Unsafe.Add(ref aRef, aBase + k));
                    var vB = Vector256.LoadUnsafe(ref Unsafe.Add(ref bRef, bBase + k));
                    vSum = Fma.MultiplyAdd(vA, vB, vSum);
                }
                float sum = Vector256.Sum(vSum);
                for (; k < K; k++)
                    sum += Unsafe.Add(ref aRef, aBase + k) * Unsafe.Add(ref bRef, bBase + k);
                Unsafe.Add(ref oRef, i * N + j) = sum;
            }
        }
    }

    // ── Q8_0 Fused Dot Product ──────────────────────────────────────────────

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe float DotQ8_0Ptr(float* aVec, byte* bQ8, int blockCount)
    {
        if (Avx512BW.IsSupported)
            return DotQ8_0Avx512(aVec, bQ8, blockCount);
        if (Avx2.IsSupported)
            return DotQ8_0Avx2(aVec, bQ8, blockCount);
        return DotQ8_0Scalar(aVec, bQ8, blockCount);
    }

    /// <summary>
    /// AVX-512 fused Q8_0 dot product. Processes 16 int8 → 16 float at once using 512-bit ops.
    /// 2 groups of 16 = 32 elements per block (same as AVX2 but with wider vectors).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe float DotQ8_0Avx512(float* aVec, byte* bQ8, int blockCount)
    {
        var acc = Vector512<float>.Zero;

        for (int b = 0; b < blockCount; b++)
        {
            byte* blockPtr = bQ8 + b * Q8_0TypeSize;
            float* aPtr = aVec + b * Q8_0BlockSize;
            sbyte* quants = (sbyte*)(blockPtr + 2);

            float scale = (float)Unsafe.ReadUnaligned<Half>(blockPtr);
            var vScale = Vector512.Create(scale);

            // Load 32 int8 quants in 2 groups of 16:
            // vpmovsxbd: 16 bytes → 16 int32 → 16 float (in 512-bit register)
            var q0 = Avx512BW.ConvertToVector512Int32(
                Vector128.Load(quants).AsSByte());
            var q1 = Avx512BW.ConvertToVector512Int32(
                Vector128.Load(quants + 16).AsSByte());

            var f0 = Avx512F.ConvertToVector512Single(q0);
            var f1 = Avx512F.ConvertToVector512Single(q1);

            var a0 = Vector512.Load(aPtr);
            var a1 = Vector512.Load(aPtr + 16);

            // FMA: acc += scale * (a0*f0 + a1*f1)
            var blockSum = Avx512F.FusedMultiplyAdd(a0, f0, a1 * f1);
            acc = Avx512F.FusedMultiplyAdd(vScale, blockSum, acc);
        }

        return Vector512.Sum(acc);
    }

    /// <summary>
    /// AVX2 fused Q8_0 dot product. Each Q8_0 block is 2 bytes FP16 scale + 32 int8 quants.
    /// Uses vpmovsxbd (sign-extend byte→dword) + vcvtdq2ps + vfmadd for high throughput.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe float DotQ8_0Avx2(float* aVec, byte* bQ8, int blockCount)
    {
        var acc = Vector256<float>.Zero;
        bool hasFma = Fma.IsSupported;

        for (int b = 0; b < blockCount; b++)
        {
            byte* blockPtr = bQ8 + b * Q8_0TypeSize;
            float* aPtr = aVec + b * Q8_0BlockSize;
            sbyte* quants = (sbyte*)(blockPtr + 2);

            // Read FP16 scale → FP32
            float scale = (float)Unsafe.ReadUnaligned<Half>(blockPtr);
            var vScale = Vector256.Create(scale);

            // Sign-extend 4 groups of 8 int8 → 8 int32 → 8 float, then multiply with a
            // vpmovsxbd: load 8 bytes from memory, sign-extend to 8 int32s in ymm
            var q0 = Avx2.ConvertToVector256Int32(Vector128.CreateScalarUnsafe(
                Unsafe.ReadUnaligned<long>(quants)).AsSByte());
            var q1 = Avx2.ConvertToVector256Int32(Vector128.CreateScalarUnsafe(
                Unsafe.ReadUnaligned<long>(quants + 8)).AsSByte());
            var q2 = Avx2.ConvertToVector256Int32(Vector128.CreateScalarUnsafe(
                Unsafe.ReadUnaligned<long>(quants + 16)).AsSByte());
            var q3 = Avx2.ConvertToVector256Int32(Vector128.CreateScalarUnsafe(
                Unsafe.ReadUnaligned<long>(quants + 24)).AsSByte());

            // Convert int32 → float
            var f0 = Avx.ConvertToVector256Single(q0);
            var f1 = Avx.ConvertToVector256Single(q1);
            var f2 = Avx.ConvertToVector256Single(q2);
            var f3 = Avx.ConvertToVector256Single(q3);

            // Load 32 floats from activation vector
            var a0 = Avx.LoadVector256(aPtr);
            var a1 = Avx.LoadVector256(aPtr + 8);
            var a2 = Avx.LoadVector256(aPtr + 16);
            var a3 = Avx.LoadVector256(aPtr + 24);

            // Accumulate: acc += scale * sum(a[i] * quant[i])
            if (hasFma)
            {
                var blockSum = Fma.MultiplyAdd(a0, f0,
                               Fma.MultiplyAdd(a1, f1,
                               Fma.MultiplyAdd(a2, f2, a3 * f3)));
                acc = Fma.MultiplyAdd(vScale, blockSum, acc);
            }
            else
            {
                var blockSum = a0 * f0 + a1 * f1 + a2 * f2 + a3 * f3;
                acc += vScale * blockSum;
            }
        }

        return Vector256.Sum(acc);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe float DotQ8_0Scalar(float* aVec, byte* bQ8, int blockCount)
    {
        float sum = 0;
        for (int b = 0; b < blockCount; b++)
        {
            byte* blockPtr = bQ8 + b * Q8_0TypeSize;
            float* aPtr = aVec + b * Q8_0BlockSize;
            float scale = (float)Unsafe.ReadUnaligned<Half>(blockPtr);
            sbyte* quants = (sbyte*)(blockPtr + 2);

            float blockSum = 0;
            for (int i = 0; i < Q8_0BlockSize; i++)
                blockSum += aPtr[i] * quants[i];
            sum += scale * blockSum;
        }
        return sum;
    }

    // ── Q4_0 Fused Dot Product ──────────────────────────────────────────────

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe float DotQ4_0Ptr(float* aVec, byte* bQ4, int blockCount)
    {
        if (Avx2.IsSupported && Fma.IsSupported)
            return DotQ4_0Avx2(aVec, bQ4, blockCount);
        return DotQ4_0Scalar(aVec, bQ4, blockCount);
    }

    /// <summary>
    /// AVX2 fused Q4_0 dot product. Each Q4_0 block is 2 bytes FP16 scale +
    /// 16 packed bytes (32 × 4-bit nibbles). Low nibble of byte i → element i;
    /// high nibble of byte i → element i+16. Nibbles are [0..15], re-centered
    /// by subtracting 8.
    ///
    /// Algorithm:
    ///  1. Load 16 packed bytes into a 128-bit register.
    ///  2. Mask low nibbles (bytes 0..15 become q[0..15]).
    ///  3. Shift-right-by-4 in UInt16 domain then mask → high nibbles (q[16..31]).
    ///  4. Sign-correct by subtracting 8 (byte domain, treated as signed after).
    ///  5. Widen sbyte → int16 → int32 → float in two 8-lane chunks each.
    ///  6. FMA with 32 activation floats, scaled by FP16 delta.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe float DotQ4_0Avx2(float* aVec, byte* bQ4, int blockCount)
    {
        var acc = Vector256<float>.Zero;
        var maskLow = Vector128.Create((byte)0x0F);
        var offset8 = Vector128.Create((byte)8);

        for (int b = 0; b < blockCount; b++)
        {
            byte* blockPtr = bQ4 + b * Q4_0TypeSize;
            float* aPtr = aVec + b * Q4_0BlockSize;

            float scale = (float)Unsafe.ReadUnaligned<Half>(blockPtr);
            var vScale = Vector256.Create(scale);

            // Load 16 packed bytes
            var packed = Sse2.LoadVector128(blockPtr + 2);

            // Extract nibbles
            var loNibbles = Sse2.And(packed, maskLow);                                     // q[0..15] (unsigned 0..15)
            var shiftedHigh = Sse2.ShiftRightLogical(packed.AsUInt16(), 4).AsByte();
            var hiNibbles = Sse2.And(shiftedHigh, maskLow);                                // q[16..31]

            // Centre by -8: 0..15 → -8..7 (byte subtract; reinterpret as sbyte below)
            var loCentered = Sse2.Subtract(loNibbles, offset8);
            var hiCentered = Sse2.Subtract(hiNibbles, offset8);

            // Widen 16 sbyte → 16 int16 (sign extending)
            var lo16 = Avx2.ConvertToVector256Int16(loCentered.AsSByte());
            var hi16 = Avx2.ConvertToVector256Int16(hiCentered.AsSByte());

            // 16 int16 → 2 × 8 int32 → 2 × 8 float
            var fLo0 = Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(lo16.GetLower()));  // q[0..7]
            var fLo1 = Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(lo16.GetUpper()));  // q[8..15]
            var fHi0 = Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(hi16.GetLower()));  // q[16..23]
            var fHi1 = Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(hi16.GetUpper()));  // q[24..31]

            // Load 32 activation floats
            var a0 = Avx.LoadVector256(aPtr);        // a[0..7]
            var a1 = Avx.LoadVector256(aPtr + 8);    // a[8..15]
            var a2 = Avx.LoadVector256(aPtr + 16);   // a[16..23]
            var a3 = Avx.LoadVector256(aPtr + 24);   // a[24..31]

            // acc += scale * (a0·fLo0 + a1·fLo1 + a2·fHi0 + a3·fHi1)
            var blockSum = Fma.MultiplyAdd(a0, fLo0,
                           Fma.MultiplyAdd(a1, fLo1,
                           Fma.MultiplyAdd(a2, fHi0, a3 * fHi1)));
            acc = Fma.MultiplyAdd(vScale, blockSum, acc);
        }

        return Vector256.Sum(acc);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe float DotQ4_0Scalar(float* aVec, byte* bQ4, int blockCount)
    {
        float sum = 0;
        for (int b = 0; b < blockCount; b++)
        {
            byte* blockPtr = bQ4 + b * Q4_0TypeSize;
            float* aPtr = aVec + b * Q4_0BlockSize;
            float scale = (float)Unsafe.ReadUnaligned<Half>(blockPtr);
            byte* packed = blockPtr + 2;

            float blockSum = 0;
            for (int i = 0; i < 16; i++)
            {
                byte p = packed[i];
                int lo = (p & 0x0F) - 8;
                int hi = (p >> 4) - 8;
                blockSum += aPtr[i] * lo + aPtr[i + 16] * hi;
            }
            sum += scale * blockSum;
        }
        return sum;
    }

    // ── Q4_K Fused Dot Product ──────────────────────────────────────────────

    /// <summary>
    /// FP32 × Q4_K fused dequant+matmul. Each super-block is 256 values (144 bytes):
    /// 2b d (FP16) + 2b dmin (FP16) + 12b packed 6-bit scales/mins + 128b nibbles.
    /// Used by the gemma4 lm_head tied embedding.
    /// </summary>
    public static unsafe void MultiplyQ4_K(Span<float> output, ReadOnlySpan<float> a, ReadOnlySpan<byte> b, int M, int K, int N)
    {
        int superBlocksPerRow = K / Q4_KSuperBlockSize;
        int bytesPerRow = superBlocksPerRow * Q4_KTypeSize;

        fixed (float* aFixedPtr = a)
        fixed (byte* bFixedPtr = b)
        fixed (float* oFixedPtr = output)
        {
            nint aBase = (nint)aFixedPtr;
            nint bBase = (nint)bFixedPtr;
            nint oBase = (nint)oFixedPtr;

            for (int i = 0; i < M; i++)
            {
                float* aRow = (float*)aBase + i * K;
                float* oRow = (float*)oBase + i * N;

                if (N >= ParallelThreshold)
                {
                    nint bCapture = bBase;
                    int bprCapture = bytesPerRow;
                    int sbCapture = superBlocksPerRow;
                    Parallel.For(0, N, CpuThreading.Options, j =>
                    {
                        byte* bRow = (byte*)bCapture + j * bprCapture;
                        oRow[j] = DotQ4_KPtr(aRow, bRow, sbCapture);
                    });
                }
                else
                {
                    for (int j = 0; j < N; j++)
                    {
                        byte* bRow = (byte*)bBase + j * bytesPerRow;
                        oRow[j] = DotQ4_KPtr(aRow, bRow, superBlocksPerRow);
                    }
                }
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe float DotQ4_KPtr(float* aVec, byte* bQ4K, int superBlockCount)
    {
        if (Avx2.IsSupported && Fma.IsSupported)
            return DotQ4_KAvx2(aVec, bQ4K, superBlockCount);
        return DotQ4_KScalar(aVec, bQ4K, superBlockCount);
    }

    /// <summary>
    /// AVX2 fused Q4_K dot product.
    ///
    /// Per super-block:
    ///   Header: d (FP16, +0), dmin (FP16, +2), scales[12] (+4), qs[128] (+16)
    ///
    /// Scale/min unpacking (8 of each per super-block):
    ///   j &lt; 4: sc = scales[j] &amp; 63;            m = scales[j+4] &amp; 63
    ///   j ≥ 4: sc = (scales[j+4] &amp; 0x0F) | ((scales[j-4] &gt;&gt; 6) &lt;&lt; 4)
    ///          m  = (scales[j+4] &gt;&gt; 4)   | ((scales[j]   &gt;&gt; 6) &lt;&lt; 4)
    ///
    /// Nibble layout: qs is 4 chunks of 32 bytes. For chunk c:
    ///   byte[c*32 + l] low nibble  → element[c*64 + l]      (sub-block 2c)
    ///   byte[c*32 + l] high nibble → element[c*64 + l + 32] (sub-block 2c+1)
    ///
    /// Dequant value: ss_j * q - sm_j where ss_j = d * sc_j, sm_j = dmin * m_j.
    /// Dot contribution per sub-block of 32: ss * sum(a*q) - sm * sum(a).
    ///
    /// Inner loop keeps four vector accumulators (loDot, loSum, hiDot, hiSum)
    /// running across both 16-byte halves of a chunk, so horizontal reductions
    /// are done only once per chunk (16 per super-block total).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe float DotQ4_KAvx2(float* aVec, byte* bQ4K, int superBlockCount)
    {
        float total = 0;
        var maskLow128 = Vector128.Create((byte)0x0F);

        for (int sb = 0; sb < superBlockCount; sb++)
        {
            byte* sbPtr = bQ4K + sb * Q4_KTypeSize;
            float* aSB = aVec + sb * Q4_KSuperBlockSize;

            float d = (float)Unsafe.ReadUnaligned<Half>(sbPtr);
            float dmin = (float)Unsafe.ReadUnaligned<Half>(sbPtr + 2);

            byte* scalesPacked = sbPtr + 4;
            byte* qs = sbPtr + 16;

            // Each chunk produces 2 sub-blocks (sub-block 2*chunk from low nibbles,
            // sub-block 2*chunk+1 from high nibbles).
            for (int chunk = 0; chunk < 4; chunk++)
            {
                int j0 = 2 * chunk;
                int j1 = 2 * chunk + 1;
                int sc0, mn0, sc1, mn1;
                if (j0 < 4)
                {
                    sc0 = scalesPacked[j0] & 63;
                    mn0 = scalesPacked[j0 + 4] & 63;
                }
                else
                {
                    sc0 = (scalesPacked[j0 + 4] & 0x0F) | ((scalesPacked[j0 - 4] >> 6) << 4);
                    mn0 = (scalesPacked[j0 + 4] >> 4)   | ((scalesPacked[j0]     >> 6) << 4);
                }
                if (j1 < 4)
                {
                    sc1 = scalesPacked[j1] & 63;
                    mn1 = scalesPacked[j1 + 4] & 63;
                }
                else
                {
                    sc1 = (scalesPacked[j1 + 4] & 0x0F) | ((scalesPacked[j1 - 4] >> 6) << 4);
                    mn1 = (scalesPacked[j1 + 4] >> 4)   | ((scalesPacked[j1]     >> 6) << 4);
                }
                float ss0 = d * sc0, sm0 = dmin * mn0;
                float ss1 = d * sc1, sm1 = dmin * mn1;

                byte* chunkBytes = qs + chunk * 32;
                float* aLowBase = aSB + chunk * 64;
                float* aHighBase = aSB + chunk * 64 + 32;

                // Vector accumulators running across the 2 halves of 16 bytes
                var loDot = Vector256<float>.Zero;
                var loSum = Vector256<float>.Zero;
                var hiDot = Vector256<float>.Zero;
                var hiSum = Vector256<float>.Zero;

                for (int half = 0; half < 2; half++)
                {
                    byte* packed16 = chunkBytes + half * 16;
                    float* aLow = aLowBase + half * 16;
                    float* aHigh = aHighBase + half * 16;

                    var packed = Sse2.LoadVector128(packed16);
                    var loN = Sse2.And(packed, maskLow128);
                    var hiN = Sse2.And(
                        Sse2.ShiftRightLogical(packed.AsUInt16(), 4).AsByte(),
                        maskLow128);

                    // Zero-extend 16 nibbles to 16 int16 → 2 × 8 int32 → float
                    var lo16 = Avx2.ConvertToVector256Int16(loN);
                    var hi16 = Avx2.ConvertToVector256Int16(hiN);

                    var fLo0 = Avx.ConvertToVector256Single(
                        Avx2.ConvertToVector256Int32(lo16.GetLower()));
                    var fLo1 = Avx.ConvertToVector256Single(
                        Avx2.ConvertToVector256Int32(lo16.GetUpper()));
                    var fHi0 = Avx.ConvertToVector256Single(
                        Avx2.ConvertToVector256Int32(hi16.GetLower()));
                    var fHi1 = Avx.ConvertToVector256Single(
                        Avx2.ConvertToVector256Int32(hi16.GetUpper()));

                    var aL0 = Avx.LoadVector256(aLow);
                    var aL1 = Avx.LoadVector256(aLow + 8);
                    var aH0 = Avx.LoadVector256(aHigh);
                    var aH1 = Avx.LoadVector256(aHigh + 8);

                    // FMA into running accumulators — no horizontal sums here.
                    loDot = Fma.MultiplyAdd(aL0, fLo0, loDot);
                    loDot = Fma.MultiplyAdd(aL1, fLo1, loDot);
                    hiDot = Fma.MultiplyAdd(aH0, fHi0, hiDot);
                    hiDot = Fma.MultiplyAdd(aH1, fHi1, hiDot);

                    loSum = Avx.Add(loSum, Avx.Add(aL0, aL1));
                    hiSum = Avx.Add(hiSum, Avx.Add(aH0, aH1));
                }

                // Reduce once per chunk (4 reductions per chunk, 16 per super-block).
                float loDotS = Vector256.Sum(loDot);
                float loSumS = Vector256.Sum(loSum);
                float hiDotS = Vector256.Sum(hiDot);
                float hiSumS = Vector256.Sum(hiSum);

                total += ss0 * loDotS - sm0 * loSumS;
                total += ss1 * hiDotS - sm1 * hiSumS;
            }
        }

        return total;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe float DotQ4_KScalar(float* aVec, byte* bQ4K, int superBlockCount)
    {
        float total = 0;
        for (int sb = 0; sb < superBlockCount; sb++)
        {
            byte* sbPtr = bQ4K + sb * Q4_KTypeSize;
            float* aSB = aVec + sb * Q4_KSuperBlockSize;

            float d = (float)Unsafe.ReadUnaligned<Half>(sbPtr);
            float dmin = (float)Unsafe.ReadUnaligned<Half>(sbPtr + 2);

            byte* scalesPacked = sbPtr + 4;
            byte* qs = sbPtr + 16;

            for (int chunk = 0; chunk < 4; chunk++)
            {
                int j0 = 2 * chunk, j1 = 2 * chunk + 1;
                int sc0, mn0, sc1, mn1;
                if (j0 < 4)
                {
                    sc0 = scalesPacked[j0] & 63;
                    mn0 = scalesPacked[j0 + 4] & 63;
                }
                else
                {
                    sc0 = (scalesPacked[j0 + 4] & 0x0F) | ((scalesPacked[j0 - 4] >> 6) << 4);
                    mn0 = (scalesPacked[j0 + 4] >> 4)   | ((scalesPacked[j0]     >> 6) << 4);
                }
                if (j1 < 4)
                {
                    sc1 = scalesPacked[j1] & 63;
                    mn1 = scalesPacked[j1 + 4] & 63;
                }
                else
                {
                    sc1 = (scalesPacked[j1 + 4] & 0x0F) | ((scalesPacked[j1 - 4] >> 6) << 4);
                    mn1 = (scalesPacked[j1 + 4] >> 4)   | ((scalesPacked[j1]     >> 6) << 4);
                }
                float ss0 = d * sc0, sm0 = dmin * mn0;
                float ss1 = d * sc1, sm1 = dmin * mn1;

                byte* chunkBytes = qs + chunk * 32;
                float* aLow = aSB + chunk * 64;
                float* aHigh = aSB + chunk * 64 + 32;

                float sDot0 = 0, aSum0 = 0, sDot1 = 0, aSum1 = 0;
                for (int l = 0; l < 32; l++)
                {
                    byte p = chunkBytes[l];
                    int q0 = p & 0x0F;
                    int q1 = p >> 4;
                    sDot0 += aLow[l] * q0;
                    aSum0 += aLow[l];
                    sDot1 += aHigh[l] * q1;
                    aSum1 += aHigh[l];
                }
                total += ss0 * sDot0 - sm0 * aSum0;
                total += ss1 * sDot1 - sm1 * aSum1;
            }
        }
        return total;
    }
}
