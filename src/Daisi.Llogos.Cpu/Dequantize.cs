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

            // Unpack 6-bit scales and mins from 12-byte packed section (bytes 4..15)
            Unpack6BitScalesMins(source.Slice(srcOff + 4, 12), scales, mins);

            // Dequantize 8 sub-blocks of 32 elements each
            int nibbleOff = srcOff + 16; // 128 bytes of packed nibbles
            for (int j = 0; j < Q4_KSubBlocks; j++)
            {
                float subScale = d * scales[j];
                float subMin = dmin * mins[j];
                for (int i = 0; i < Q4_KSubBlockSize / 2; i++)
                {
                    byte packed = source[nibbleOff + j * 16 + i];
                    int lo = packed & 0x0F;
                    int hi = packed >> 4;
                    destination[dstOff + j * Q4_KSubBlockSize + i] = subScale * lo - subMin;
                    destination[dstOff + j * Q4_KSubBlockSize + 16 + i] = subScale * hi - subMin;
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
