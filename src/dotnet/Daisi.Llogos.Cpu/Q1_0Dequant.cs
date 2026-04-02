using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace Daisi.Llogos.Cpu;

/// <summary>
/// Dequantization and matrix multiplication for Q1_0 and Q1_0_g128 (PrismML Bonsai) tensors.
/// Binary 1-bit sign quantization: each weight is +scale or -scale.
/// Q1_0: 32 elements/block, 6 bytes (2 FP16 scale + 4 sign bytes).
/// Q1_0_g128: 128 elements/block, 18 bytes (2 FP16 scale + 16 sign bytes).
/// Sign bits packed LSB-first: bit=1 means +scale, bit=0 means -scale.
/// </summary>
internal static class Q1_0Dequant
{
    /// <summary>Dequantize Q1_0 (block size 32) packed data to FP32.</summary>
    public static void Dequantize(ReadOnlySpan<byte> source, Span<float> destination, long elementCount)
    {
        DequantizeCore(source, destination, elementCount, blockSize: 32, signBytes: 4);
    }

    /// <summary>Dequantize Q1_0_g128 (block size 128) packed data to FP32.</summary>
    public static void DequantizeG128(ReadOnlySpan<byte> source, Span<float> destination, long elementCount)
    {
        DequantizeCore(source, destination, elementCount, blockSize: 128, signBytes: 16);
    }

    private static void DequantizeCore(ReadOnlySpan<byte> source, Span<float> destination,
        long elementCount, int blockSize, int signBytes)
    {
        int bytesPerBlock = 2 + signBytes; // FP16 scale + sign bits
        long numBlocks = elementCount / blockSize;
        int srcOff = 0;

        for (long blk = 0; blk < numBlocks; blk++)
        {
            float scale = HalfToFloat(source[srcOff], source[srcOff + 1]);
            float negScale = -scale;

            for (int j = 0; j < blockSize; j++)
            {
                int bit = (source[srcOff + 2 + j / 8] >> (j % 8)) & 1;
                destination[(int)(blk * blockSize + j)] = bit != 0 ? scale : negScale;
            }
            srcOff += bytesPerBlock;
        }
    }

    /// <summary>
    /// FP32 activations x Q1_0 binary weights → FP32 output.
    /// b is [N x K] packed as Q1_0 or Q1_0_g128. Row-major weight layout.
    /// </summary>
    public static unsafe void Multiply(Span<float> output, ReadOnlySpan<float> a, ReadOnlySpan<byte> b,
        int M, int K, int N, int blockSize)
    {
        int signBytes = blockSize / 8;
        int bytesPerBlock = 2 + signBytes;
        int blocksPerRow = K / blockSize;
        int bytesPerRow = blocksPerRow * bytesPerBlock;

        fixed (float* aFixed = a)
        fixed (byte* bFixed = b)
        fixed (float* oFixed = output)
        {
            nint aBase = (nint)aFixed;
            nint bBase = (nint)bFixed;
            nint oBase = (nint)oFixed;
            int bsCapture = blockSize;
            int sbCapture = signBytes;
            int bpbCapture = bytesPerBlock;
            int bprCapture = blocksPerRow;
            int byrCapture = bytesPerRow;

            Parallel.For(0, M * N, mn =>
            {
                int i = mn / N;
                int j = mn % N;

                float* aRow = (float*)aBase + i * K;
                byte* bRow = (byte*)bBase + j * byrCapture;
                float sum = 0;

                for (int blk = 0; blk < bprCapture; blk++)
                {
                    byte* bp = bRow + blk * bpbCapture;
                    float scale = HalfToFloatPtr(bp);
                    int baseIdx = blk * bsCapture;

                    for (int s = 0; s < sbCapture; s++)
                    {
                        byte bits = bp[2 + s];
                        int elemBase = baseIdx + s * 8;
                        sum += aRow[elemBase + 0] * ((bits & 0x01) != 0 ? scale : -scale);
                        sum += aRow[elemBase + 1] * ((bits & 0x02) != 0 ? scale : -scale);
                        sum += aRow[elemBase + 2] * ((bits & 0x04) != 0 ? scale : -scale);
                        sum += aRow[elemBase + 3] * ((bits & 0x08) != 0 ? scale : -scale);
                        sum += aRow[elemBase + 4] * ((bits & 0x10) != 0 ? scale : -scale);
                        sum += aRow[elemBase + 5] * ((bits & 0x20) != 0 ? scale : -scale);
                        sum += aRow[elemBase + 6] * ((bits & 0x40) != 0 ? scale : -scale);
                        sum += aRow[elemBase + 7] * ((bits & 0x80) != 0 ? scale : -scale);
                    }
                }
                ((float*)oBase)[i * N + j] = sum;
            });
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float HalfToFloat(byte lo, byte hi)
    {
        ushort bits = (ushort)(lo | (hi << 8));
        return (float)BitConverter.Int16BitsToHalf((short)bits);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe float HalfToFloatPtr(byte* p)
    {
        ushort bits = (ushort)(p[0] | (p[1] << 8));
        return (float)BitConverter.Int16BitsToHalf((short)bits);
    }
}
