using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace Daisi.Llogos.Cpu;

/// <summary>
/// Static dequantization methods for converting quantized GGUF tensor data to FP32.
/// Each method provides a scalar fallback and an AVX2 SIMD fast path.
/// </summary>
public static class Dequantize
{
    private const int Q8_0BlockSize = 32;
    private const int Q8_0TypeSize = 34;

    private const int Q4_0BlockSize = 32;
    private const int Q4_0TypeSize = 18;

    private const int Q4_KSuperBlockSize = 256;
    private const int Q4_KTypeSize = 144;
    private const int Q4_KSubBlocks = 8;
    private const int Q4_KSubBlockSize = 32;

    /// <summary>
    /// Dequantize Q8_0 data to FP32.
    /// Each block: 2-byte FP16 scale + 32 signed bytes = 34 bytes → 32 floats.
    /// </summary>
    public static void DequantizeQ8_0(ReadOnlySpan<byte> source, Span<float> destination)
    {
        int blockCount = source.Length / Q8_0TypeSize;
        if (destination.Length < blockCount * Q8_0BlockSize)
            throw new ArgumentException("Destination too small.");

        if (Avx2.IsSupported)
            DequantizeQ8_0Avx2(source, destination, blockCount);
        else
            DequantizeQ8_0Scalar(source, destination, blockCount);
    }

    /// <summary>
    /// Dequantize Q4_0 data to FP32.
    /// Each block: 2-byte FP16 scale + 16 packed bytes (32 × 4-bit nibbles) = 18 bytes → 32 floats.
    /// Nibbles are unsigned [0..15], re-centered by subtracting 8.
    /// </summary>
    public static void DequantizeQ4_0(ReadOnlySpan<byte> source, Span<float> destination)
    {
        int blockCount = source.Length / Q4_0TypeSize;
        if (destination.Length < blockCount * Q4_0BlockSize)
            throw new ArgumentException("Destination too small.");

        if (Avx2.IsSupported)
            DequantizeQ4_0Avx2(source, destination, blockCount);
        else
            DequantizeQ4_0Scalar(source, destination, blockCount);
    }

    /// <summary>
    /// Dequantize Q4_K data to FP32.
    /// Each super-block: 2-byte d + 2-byte dmin + 12-byte packed scales/mins + 128 packed nibbles = 144 bytes → 256 floats.
    /// </summary>
    public static void DequantizeQ4_K(ReadOnlySpan<byte> source, Span<float> destination)
    {
        int superBlockCount = source.Length / Q4_KTypeSize;
        if (destination.Length < superBlockCount * Q4_KSuperBlockSize)
            throw new ArgumentException("Destination too small.");

        DequantizeQ4_KScalar(source, destination, superBlockCount);
    }

    // ── Q8_0 Scalar ──────────────────────────────────────────────────────────

    internal static void DequantizeQ8_0Scalar(ReadOnlySpan<byte> source, Span<float> destination, int blockCount)
    {
        for (int b = 0; b < blockCount; b++)
        {
            int srcOff = b * Q8_0TypeSize;
            int dstOff = b * Q8_0BlockSize;
            float scale = (float)BitConverter.ToHalf(source.Slice(srcOff, 2));
            for (int i = 0; i < Q8_0BlockSize; i++)
            {
                destination[dstOff + i] = scale * (sbyte)source[srcOff + 2 + i];
            }
        }
    }

    // ── Q8_0 AVX2 ────────────────────────────────────────────────────────────

    internal static void DequantizeQ8_0Avx2(ReadOnlySpan<byte> source, Span<float> destination, int blockCount)
    {
        ref byte srcRef = ref MemoryMarshal.GetReference(source);
        ref float dstRef = ref MemoryMarshal.GetReference(destination);

        for (int b = 0; b < blockCount; b++)
        {
            int srcOff = b * Q8_0TypeSize;
            int dstOff = b * Q8_0BlockSize;
            float scale = (float)Unsafe.ReadUnaligned<Half>(ref Unsafe.Add(ref srcRef, srcOff));
            var vScale = Vector256.Create(scale);

            // Process 32 int8 values in 2 batches of 16
            for (int half = 0; half < 2; half++)
            {
                int byteOffset = srcOff + 2 + half * 16;
                // Load 16 signed bytes, widen to 16 int16s
                var bytes16 = Vector128.LoadUnsafe(ref Unsafe.Add(ref srcRef, byteOffset));
                var shorts = Avx2.ConvertToVector256Int16(bytes16.AsSByte());
                // Low 8 int16s → 8 int32s → 8 floats
                var intsLo = Avx2.ConvertToVector256Int32(shorts.GetLower());
                var intsHi = Avx2.ConvertToVector256Int32(shorts.GetUpper());
                var floatLo = Avx.ConvertToVector256Single(intsLo);
                var floatHi = Avx.ConvertToVector256Single(intsHi);
                Avx.Multiply(floatLo, vScale).StoreUnsafe(ref Unsafe.Add(ref dstRef, dstOff + half * 16));
                Avx.Multiply(floatHi, vScale).StoreUnsafe(ref Unsafe.Add(ref dstRef, dstOff + half * 16 + 8));
            }
        }
    }

    // ── Q4_0 Scalar ──────────────────────────────────────────────────────────

    internal static void DequantizeQ4_0Scalar(ReadOnlySpan<byte> source, Span<float> destination, int blockCount)
    {
        for (int b = 0; b < blockCount; b++)
        {
            int srcOff = b * Q4_0TypeSize;
            int dstOff = b * Q4_0BlockSize;
            float scale = (float)BitConverter.ToHalf(source.Slice(srcOff, 2));
            for (int i = 0; i < 16; i++)
            {
                byte packed = source[srcOff + 2 + i];
                int lo = (packed & 0x0F) - 8;
                int hi = (packed >> 4) - 8;
                destination[dstOff + i] = scale * lo;
                destination[dstOff + 16 + i] = scale * hi;
            }
        }
    }

    // ── Q4_0 AVX2 ────────────────────────────────────────────────────────────

    internal static void DequantizeQ4_0Avx2(ReadOnlySpan<byte> source, Span<float> destination, int blockCount)
    {
        ref byte srcRef = ref MemoryMarshal.GetReference(source);
        ref float dstRef = ref MemoryMarshal.GetReference(destination);

        var maskLow = Vector256.Create((byte)0x0F);
        var offset8 = Vector256.Create((byte)8);

        for (int b = 0; b < blockCount; b++)
        {
            int srcOff = b * Q4_0TypeSize;
            int dstOff = b * Q4_0BlockSize;
            float scale = (float)Unsafe.ReadUnaligned<Half>(ref Unsafe.Add(ref srcRef, srcOff));
            var vScale = Vector256.Create(scale);

            // Load 16 packed bytes
            var packed = Vector128.LoadUnsafe(ref Unsafe.Add(ref srcRef, srcOff + 2));
            // Expand to 256-bit: low nibbles in lower 128, high nibbles in upper 128
            var packed256 = Vector256.Create(packed, packed);

            // Extract low nibbles (elements 0..15) and high nibbles (elements 16..31)
            var loNibbles = Avx2.And(packed256, maskLow);
            var hiNibbles = Avx2.ShiftRightLogical(packed256.AsUInt16(), 4).AsByte();
            hiNibbles = Avx2.And(hiNibbles, maskLow);

            // Subtract 8 to re-center (unsigned subtract, then treat as signed)
            var loSub = Avx2.Subtract(loNibbles, offset8);
            var hiSub = Avx2.Subtract(hiNibbles, offset8);

            // Process low nibbles (first 16 elements → output[0..15])
            // Process in 4 groups of 4 to go byte→int32→float
            UnpackAndStore(ref dstRef, dstOff, loSub.GetLower(), vScale);
            UnpackAndStore(ref dstRef, dstOff + 16, hiSub.GetLower(), vScale);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void UnpackAndStore(ref float dstRef, int dstOff, Vector128<byte> signedBytes, Vector256<float> vScale)
    {
        // Convert 16 signed bytes to 16 floats via int16 → int32 → float
        var shorts = Avx2.ConvertToVector256Int16(signedBytes.AsSByte());
        var lo = Avx2.ConvertToVector256Int32(shorts.GetLower());
        var hi = Avx2.ConvertToVector256Int32(shorts.GetUpper());
        var floatLo = Avx.ConvertToVector256Single(lo);
        var floatHi = Avx.ConvertToVector256Single(hi);
        Avx.Multiply(floatLo, vScale).StoreUnsafe(ref Unsafe.Add(ref dstRef, dstOff));
        Avx.Multiply(floatHi, vScale).StoreUnsafe(ref Unsafe.Add(ref dstRef, dstOff + 8));
    }

    // ── Q5_K ─────────────────────────────────────────────────────────────────

    private const int Q5_KSuperBlockSize = 256;
    private const int Q5_KTypeSize = 176;

    /// <summary>
    /// Dequantize Q5_K data to FP32.
    /// Each super-block: 2b d + 2b dmin + 12b scales/mins + 32b qh + 128b qs = 176 bytes → 256 floats.
    /// qs[j] low nibble = element j (0..127), high nibble = element j+128 (128..255).
    /// qh bit n = high (5th) bit for element n (0..255).
    /// </summary>
    public static void DequantizeQ5_K(ReadOnlySpan<byte> source, Span<float> destination)
    {
        int sbCount = source.Length / Q5_KTypeSize;
        Span<float> scales = stackalloc float[8];
        Span<float> mins = stackalloc float[8];

        for (int sb = 0; sb < sbCount; sb++)
        {
            int srcOff = sb * Q5_KTypeSize;
            int dstOff = sb * Q5_KSuperBlockSize;

            float d = (float)BitConverter.ToHalf(source.Slice(srcOff, 2));
            float dmin = (float)BitConverter.ToHalf(source.Slice(srcOff + 2, 2));

            Unpack6BitScalesMins(source.Slice(srcOff + 4, 12), scales, mins);

            int qhOff = srcOff + 16;  // 32 bytes (256 bits) of high bits
            int qsOff = srcOff + 48;  // 128 bytes of packed low nibbles

            // Dequantize all 256 elements in chunks of 64 (matching ggml layout).
            // Each 32-byte qs chunk serves 64 elements: low nibble → first 32, high nibble → next 32.
            // qh bits: element n's high bit is at qh[n % 32] bit (n / 32).
            int isIdx = 0;
            int qlPtr = qsOff;
            byte u1 = 1, u2 = 2;
            for (int j = 0; j < 4; j++)
            {
                float d1 = d * scales[isIdx];
                float m1 = dmin * mins[isIdx];
                float d2 = d * scales[isIdx + 1];
                float m2 = dmin * mins[isIdx + 1];

                for (int l = 0; l < 32; l++)
                {
                    int lo = source[qlPtr + l] & 0xF;
                    int hbLo = (source[qhOff + l] & u1) != 0 ? 16 : 0;
                    destination[dstOff + j * 64 + l] = d1 * (lo + hbLo) - m1;

                    int hi = source[qlPtr + l] >> 4;
                    int hbHi = (source[qhOff + l] & u2) != 0 ? 16 : 0;
                    destination[dstOff + j * 64 + l + 32] = d2 * (hi + hbHi) - m2;
                }

                qlPtr += 32;
                u1 <<= 2;
                u2 <<= 2;
                isIdx += 2;
            }
        }
    }

    // ── Q6_K ─────────────────────────────────────────────────────────────────

    private const int Q6_KSuperBlockSize = 256;
    private const int Q6_KTypeSize = 210;

    /// <summary>
    /// Dequantize Q6_K data to FP32.
    /// Each super-block: ql[128] + qh[64] + sc[16] + d[2] = 210 bytes → 256 floats.
    /// Layout follows ggml's interleaved scheme: two 128-element halves,
    /// each processed as 4 groups of 32 using interleaved ql nibbles and qh bit pairs.
    /// </summary>
    public static void DequantizeQ6_K(ReadOnlySpan<byte> source, Span<float> destination)
    {
        int sbCount = source.Length / Q6_KTypeSize;

        for (int sb = 0; sb < sbCount; sb++)
        {
            int srcOff = sb * Q6_KTypeSize;
            int dstOff = sb * Q6_KSuperBlockSize;

            float d = (float)BitConverter.ToHalf(source.Slice(srcOff + 208, 2));

            int qlBase = srcOff;
            int qhBase = srcOff + 128;
            int scBase = srcOff + 192;

            // Process two 128-element halves
            for (int half = 0; half < 2; half++)
            {
                int qlOff = qlBase + half * 64;
                int qhOff = qhBase + half * 32;
                int scIdx = half * 8; // sc offset: 0 for first half, 8 for second
                int yOff = dstOff + half * 128;

                for (int l = 0; l < 32; l++)
                {
                    int q1 = ((source[qlOff + l] & 0xF) | (((source[qhOff + l] >> 0) & 3) << 4)) - 32;
                    int q2 = ((source[qlOff + l + 32] & 0xF) | (((source[qhOff + l] >> 2) & 3) << 4)) - 32;
                    int q3 = ((source[qlOff + l] >> 4) | (((source[qhOff + l] >> 4) & 3) << 4)) - 32;
                    int q4 = ((source[qlOff + l + 32] >> 4) | (((source[qhOff + l] >> 6) & 3) << 4)) - 32;

                    // ggml uses is = l/16 to select sub-group scales within each 32-element loop
                    int isc = l / 16;
                    destination[yOff + l] = d * (sbyte)source[scBase + scIdx + isc + 0] * q1;
                    destination[yOff + l + 32] = d * (sbyte)source[scBase + scIdx + isc + 2] * q2;
                    destination[yOff + l + 64] = d * (sbyte)source[scBase + scIdx + isc + 4] * q3;
                    destination[yOff + l + 96] = d * (sbyte)source[scBase + scIdx + isc + 6] * q4;
                }
            }
        }
    }

    // ── Q3_K ─────────────────────────────────────────────────────────────────

    private const int Q3_KSuperBlockSize = 256;
    private const int Q3_KTypeSize = 110; // half d (2) + hmask[32] + qs[64] + scales[12] = 110

    /// <summary>
    /// Dequantize Q3_K data to FP32.
    /// block_q3_K: { half d; uint8_t hmask[32]; uint8_t qs[64]; uint8_t scales[12]; }
    /// Each element is 3 bits: 2 bits from qs + 1 bit from hmask.
    /// 16 scales packed into 12 bytes using a complex bit-field scheme.
    /// </summary>
    public static void DequantizeQ3_K(ReadOnlySpan<byte> source, Span<float> destination)
    {
        int sbCount = source.Length / Q3_KTypeSize;
        Span<int> scales = stackalloc int[16];

        for (int sb = 0; sb < sbCount; sb++)
        {
            int srcOff = sb * Q3_KTypeSize;
            int dstOff = sb * Q3_KSuperBlockSize;

            float d = (float)BitConverter.ToHalf(source.Slice(srcOff, 2));
            int hmaskOff = srcOff + 2;
            int qsOff = srcOff + 2 + 32;
            int scalesOff = srcOff + 2 + 32 + 64;

            // Unpack 16 scales from 12 bytes (matching ggml get_scale_min_k4 pattern for Q3_K).
            // bytes 0..3: low 4 bits of scales 0..3 (low nibble) and 8..11 (high nibble)
            // bytes 4..7: low 4 bits of scales 4..7 (low nibble) and 12..15 (high nibble)
            // bytes 8..11: high 2 bits of scales 0..7 (packed)
            // All scales are 6-bit unsigned, then re-centered by subtracting 32.
            var sc = source.Slice(scalesOff, 12);
            for (int i = 0; i < 4; i++)
            {
                scales[i] = (sc[i] & 0xF) - 8;
                scales[i + 8] = (sc[i] >> 4) - 8;
            }
            for (int i = 0; i < 4; i++)
            {
                scales[i + 4] = (sc[i + 4] & 0xF) - 8;
                scales[i + 12] = (sc[i + 4] >> 4) - 8;
            }
            // High 2 bits from bytes 8..11 (4 bytes = 32 bits total)
            // Each scale gets 2 additional high bits, extending from 4-bit to 6-bit
            uint aux = BitConverter.ToUInt32(source.Slice(scalesOff + 8, 4));
            for (int i = 0; i < 4; i++)
            {
                int hi0 = (int)((aux >> (2 * i)) & 3);
                int hi1 = (int)((aux >> (2 * i + 8)) & 3);
                int hi2 = (int)((aux >> (2 * i + 16)) & 3);
                int hi3 = (int)((aux >> (2 * i + 24)) & 3);
                // Reconstruct full 6-bit value then re-center
                scales[i] = ((scales[i] + 8) | (hi0 << 4)) - 32;
                scales[i + 4] = ((scales[i + 4] + 8) | (hi1 << 4)) - 32;
                scales[i + 8] = ((scales[i + 8] + 8) | (hi2 << 4)) - 32;
                scales[i + 12] = ((scales[i + 12] + 8) | (hi3 << 4)) - 32;
            }

            // Dequantize 256 elements: 3-bit values from qs (2 bits) + hmask (1 bit)
            // hmask uses a rotating bitmask like Q5_K
            byte m = 1;
            int qsPtr = qsOff;
            int yPtr = dstOff;
            for (int n = 0; n < Q3_KSuperBlockSize; n += 128)
            {
                for (int l = 0; l < 32; l++)
                {
                    int qs_val = source[qsPtr + l];

                    int q0 = (qs_val & 3) - ((source[hmaskOff + l] & m) != 0 ? 0 : 4);
                    int q1 = ((qs_val >> 2) & 3) - ((source[hmaskOff + l] & (m << 1)) != 0 ? 0 : 4);
                    int q2 = ((qs_val >> 4) & 3) - ((source[hmaskOff + l] & (m << 2)) != 0 ? 0 : 4);
                    int q3 = ((qs_val >> 6) & 3) - ((source[hmaskOff + l] & (m << 3)) != 0 ? 0 : 4);

                    destination[yPtr + l + 0] = d * scales[(n / 16) + 0] * q0;
                    destination[yPtr + l + 32] = d * scales[(n / 16) + 2] * q1;
                    destination[yPtr + l + 64] = d * scales[(n / 16) + 4] * q2;
                    destination[yPtr + l + 96] = d * scales[(n / 16) + 6] * q3;
                }
                qsPtr += 32;
                yPtr += 128;
                m <<= 4;
            }
        }
    }

    // ── Q2_K ─────────────────────────────────────────────────────────────────

    private const int Q2_KSuperBlockSize = 256;
    private const int Q2_KTypeSize = 96;

    /// <summary>
    /// Dequantize Q2_K data to FP32.
    /// Each super-block: scales[16] + qs[64] + d[2] + dmin[2] + padding = 96 bytes → 256 floats.
    /// </summary>
    public static void DequantizeQ2_K(ReadOnlySpan<byte> source, Span<float> destination)
    {
        int sbCount = source.Length / Q2_KTypeSize;

        for (int sb = 0; sb < sbCount; sb++)
        {
            int srcOff = sb * Q2_KTypeSize;
            int dstOff = sb * Q2_KSuperBlockSize;

            // block_q2_K layout: scales[16] + qs[64] + d[2] + dmin[2]
            int scalesOff = srcOff;
            int qsOff = srcOff + 16;
            float d = (float)BitConverter.ToHalf(source.Slice(srcOff + 80, 2));
            float dmin = (float)BitConverter.ToHalf(source.Slice(srcOff + 82, 2));

            for (int j = 0; j < 16; j++)
            {
                byte sc_byte = source[scalesOff + j];
                float sc = d * (sc_byte & 0xF);
                float m = dmin * (sc_byte >> 4);

                for (int i = 0; i < 16; i++)
                {
                    int n = j * 16 + i;
                    int qs_idx = n / 4;
                    int qs_shift = (n % 4) * 2;
                    int q = (source[qsOff + qs_idx] >> qs_shift) & 3;
                    destination[dstOff + n] = sc * q - m;
                }
            }
        }
    }

    // ── Q4_1 ─────────────────────────────────────────────────────────────────

    private const int Q4_1BlockSize = 32;
    private const int Q4_1TypeSize = 20; // 2b scale(f16) + 2b min(f16) + 16b nibbles

    /// <summary>
    /// Dequantize Q4_1 data to FP32.
    /// Each block: 2-byte FP16 scale + 2-byte FP16 min + 16 packed bytes (32 × 4-bit).
    /// </summary>
    public static void DequantizeQ4_1(ReadOnlySpan<byte> source, Span<float> destination)
    {
        int blockCount = source.Length / Q4_1TypeSize;
        for (int b = 0; b < blockCount; b++)
        {
            int srcOff = b * Q4_1TypeSize;
            int dstOff = b * Q4_1BlockSize;
            float scale = (float)BitConverter.ToHalf(source.Slice(srcOff, 2));
            float min = (float)BitConverter.ToHalf(source.Slice(srcOff + 2, 2));
            for (int i = 0; i < 16; i++)
            {
                byte packed = source[srcOff + 4 + i];
                destination[dstOff + i] = scale * (packed & 0xF) + min;
                destination[dstOff + 16 + i] = scale * (packed >> 4) + min;
            }
        }
    }

    // ── Q5_0 ─────────────────────────────────────────────────────────────────

    private const int Q5_0BlockSize = 32;
    private const int Q5_0TypeSize = 22; // 2b scale(f16) + 4b high bits + 16b nibbles

    /// <summary>
    /// Dequantize Q5_0 data to FP32.
    /// Each block: 2-byte FP16 scale + 4-byte high bit mask + 16 packed bytes.
    /// </summary>
    public static void DequantizeQ5_0(ReadOnlySpan<byte> source, Span<float> destination)
    {
        int blockCount = source.Length / Q5_0TypeSize;
        for (int b = 0; b < blockCount; b++)
        {
            int srcOff = b * Q5_0TypeSize;
            int dstOff = b * Q5_0BlockSize;
            float scale = (float)BitConverter.ToHalf(source.Slice(srcOff, 2));
            uint qh = BitConverter.ToUInt32(source.Slice(srcOff + 2, 4));
            for (int i = 0; i < 16; i++)
            {
                byte packed = source[srcOff + 6 + i];
                int lo = packed & 0xF;
                int hi = packed >> 4;
                int hbLo = (int)((qh >> i) & 1);
                int hbHi = (int)((qh >> (i + 16)) & 1);
                destination[dstOff + i] = scale * ((lo | (hbLo << 4)) - 16);
                destination[dstOff + 16 + i] = scale * ((hi | (hbHi << 4)) - 16);
            }
        }
    }

    // ── Q5_1 ─────────────────────────────────────────────────────────────────

    private const int Q5_1BlockSize = 32;
    private const int Q5_1TypeSize = 24; // 2b scale(f16) + 2b min(f16) + 4b high bits + 16b nibbles

    /// <summary>
    /// Dequantize Q5_1 data to FP32.
    /// Each block: 2-byte FP16 scale + 2-byte FP16 min + 4-byte high bit mask + 16 packed bytes.
    /// </summary>
    public static void DequantizeQ5_1(ReadOnlySpan<byte> source, Span<float> destination)
    {
        int blockCount = source.Length / Q5_1TypeSize;
        for (int b = 0; b < blockCount; b++)
        {
            int srcOff = b * Q5_1TypeSize;
            int dstOff = b * Q5_1BlockSize;
            float scale = (float)BitConverter.ToHalf(source.Slice(srcOff, 2));
            float min = (float)BitConverter.ToHalf(source.Slice(srcOff + 2, 2));
            uint qh = BitConverter.ToUInt32(source.Slice(srcOff + 4, 4));
            for (int i = 0; i < 16; i++)
            {
                byte packed = source[srcOff + 8 + i];
                int lo = packed & 0xF;
                int hi = packed >> 4;
                int hbLo = (int)((qh >> i) & 1);
                int hbHi = (int)((qh >> (i + 16)) & 1);
                destination[dstOff + i] = scale * (lo | (hbLo << 4)) + min;
                destination[dstOff + 16 + i] = scale * (hi | (hbHi << 4)) + min;
            }
        }
    }

    // ── BF16 ─────────────────────────────────────────────────────────────────

    /// <summary>
    /// Dequantize BF16 data to FP32 by inserting 16 zero bits as the low mantissa.
    /// </summary>
    public static void DequantizeBF16(ReadOnlySpan<byte> source, Span<float> destination)
    {
        int count = source.Length / 2;
        for (int i = 0; i < count; i++)
        {
            // BF16 is the upper 16 bits of an IEEE 754 float32
            uint bits = ((uint)source[i * 2 + 1] << 24) | ((uint)source[i * 2] << 16);
            destination[i] = BitConverter.UInt32BitsToSingle(bits);
        }
    }

    // ── Q4_K Scalar ──────────────────────────────────────────────────────────

    internal static void DequantizeQ4_KScalar(ReadOnlySpan<byte> source, Span<float> destination, int superBlockCount)
    {
        Span<float> scales = stackalloc float[Q4_KSubBlocks];
        Span<float> mins = stackalloc float[Q4_KSubBlocks];

        for (int sb = 0; sb < superBlockCount; sb++)
        {
            int srcOff = sb * Q4_KTypeSize;
            int dstOff = sb * Q4_KSuperBlockSize;

            float d = (float)BitConverter.ToHalf(source.Slice(srcOff, 2));
            float dmin = (float)BitConverter.ToHalf(source.Slice(srcOff + 2, 2));

            Unpack6BitScalesMins(source.Slice(srcOff + 4, 12), scales, mins);

            // qs layout (matching ggml): 4 chunks of 32 bytes, lo nibble → first 32 elems, hi → second 32.
            int qsOff = srcOff + 16;
            for (int chunk = 0; chunk < 4; chunk++)
            {
                float ss0 = d * scales[chunk * 2];
                float sm0 = dmin * mins[chunk * 2];
                float ss1 = d * scales[chunk * 2 + 1];
                float sm1 = dmin * mins[chunk * 2 + 1];

                for (int l = 0; l < 32; l++)
                {
                    byte packed = source[qsOff + chunk * 32 + l];
                    destination[dstOff + chunk * 64 + l] = ss0 * (packed & 0xF) - sm0;
                    destination[dstOff + chunk * 64 + l + 32] = ss1 * (packed >> 4) - sm1;
                }
            }
        }
    }

    /// <summary>
    /// Unpack 8 × 6-bit scales and 8 × 6-bit mins from a 12-byte packed section.
    /// Layout: bytes 0..3 hold the low 4 bits of scales[0..7] (one nibble each),
    /// bytes 4..7 hold the low 4 bits of mins[0..7],
    /// bytes 8..11 hold the high 2 bits of both scales and mins.
    /// </summary>
    internal static void Unpack6BitScalesMins(ReadOnlySpan<byte> packed, Span<float> scales, Span<float> mins)
    {
        // Low 4 bits of scales (bytes 0..3, two values per byte)
        Span<int> scalesRaw = stackalloc int[8];
        Span<int> minsRaw = stackalloc int[8];

        for (int i = 0; i < 4; i++)
        {
            scalesRaw[2 * i] = packed[i] & 0x3F;
            scalesRaw[2 * i + 1] = packed[i] >> 4; // This only gets 4 bits; we merge high bits below
        }

        // Actually, the Q4_K scale packing in llama.cpp is:
        // bytes 0..3: scales[0..7] low 4 bits (nibble pairs like Q4_0)
        //   scales[2i]   = bytes[i] & 0x3F  (6 bits)
        //   Wait — let me use the actual llama.cpp layout.
        //
        // In ggml-common.h, block_q4_K:
        //   half d;            // super-block scale (2 bytes)
        //   half dmin;         // super-block min (2 bytes)
        //   uint8_t scales[12]; // 6-bit quantized scales and mins
        //   uint8_t qs[128];   // 4-bit quantized values
        //
        // Scale unpacking from llama.cpp (dequantize_row_q4_K):
        //   For j in 0..7:
        //     if j < 4:
        //       sc = scales[j] & 63
        //       m  = scales[j + 4] & 63
        //     else:
        //       sc = (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4)
        //       m  = (scales[j + 4] >> 4)  | ((scales[j]     >> 6) << 4)

        // Re-do with correct layout
        for (int j = 0; j < 4; j++)
        {
            scalesRaw[j] = packed[j] & 63;
            minsRaw[j] = packed[j + 4] & 63;
        }
        for (int j = 4; j < 8; j++)
        {
            scalesRaw[j] = (packed[j + 4] & 0xF) | ((packed[j - 4] >> 6) << 4);
            minsRaw[j] = (packed[j + 4] >> 4) | ((packed[j] >> 6) << 4);
        }

        for (int j = 0; j < 8; j++)
        {
            scales[j] = scalesRaw[j];
            mins[j] = minsRaw[j];
        }
    }
}
