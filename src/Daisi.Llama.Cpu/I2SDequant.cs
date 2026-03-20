using System.Runtime.CompilerServices;

namespace Daisi.Llama.Cpu;

/// <summary>
/// Dequantization and matrix multiplication for I2_S (BitNet ternary) tensors.
/// I2_S packs 4 ternary values per byte (2 bits each) in 128-element interleaved groups.
/// Each byte: bits[7:6]=elem+0*32, bits[5:4]=elem+1*32, bits[3:2]=elem+2*32, bits[1:0]=elem+3*32.
/// Encoding: 0b00=-1, 0b01=0, 0b10=+1 (0b11 treated as 0).
/// Per-tensor float32 scale stored at byte offset nelements/4.
/// </summary>
internal static class I2SDequant
{
    // Lookup: 2-bit code → {-1, 0, +1, 0}
    private static readonly float[] Map = [-1f, 0f, 1f, 0f];

    private const int ParallelThreshold = 32;

    /// <summary>
    /// Dequantize I2_S packed data to FP32.
    /// </summary>
    public static void Dequantize(ReadOnlySpan<byte> source, Span<float> destination, long elementCount)
    {
        long packedBytes = elementCount / 4;
        float scale = ReadScale(source, packedBytes);

        long done = 0;
        int srcIdx = 0;
        while (done < elementCount)
        {
            long remaining = elementCount - done;
            long blkElements = remaining >= 128 ? 128 : remaining;
            long cols0 = blkElements >= 32 ? 32 : blkElements;
            long cols1 = blkElements >= 64 ? 32 : Math.Max(0, blkElements - 32);
            long cols2 = blkElements >= 96 ? 32 : Math.Max(0, blkElements - 64);
            long cols3 = blkElements >= 128 ? 32 : Math.Max(0, blkElements - 96);

            for (int gp = 0; gp < 32; gp++)
            {
                byte b = source[srcIdx + gp];
                if (gp < cols0) destination[(int)(done + 0 * 32 + gp)] = scale * Map[(b >> 6) & 3];
                if (gp < cols1) destination[(int)(done + 1 * 32 + gp)] = scale * Map[(b >> 4) & 3];
                if (gp < cols2) destination[(int)(done + 2 * 32 + gp)] = scale * Map[(b >> 2) & 3];
                if (gp < cols3) destination[(int)(done + 3 * 32 + gp)] = scale * Map[b & 3];
            }
            srcIdx += 32;
            done += blkElements;
        }
    }

    /// <summary>
    /// FP32 activations x I2_S ternary weights. b is [N x K] packed as I2_S.
    /// output[i,j] = dot(a[i,:], dequant(b[j,:]))
    /// </summary>
    public static unsafe void Multiply(Span<float> output, ReadOnlySpan<float> a, ReadOnlySpan<byte> b,
        int M, int K, int N)
    {
        long packedBytesPerRow = (long)K / 4;
        long bytesPerRow = packedBytesPerRow + 32; // packed data + 32-byte trailer per tensor

        // All rows share the same scale (per-tensor, stored once at end of all packed data)
        long totalPackedBytes = (long)K * N / 4;
        float scale = ReadScale(b, totalPackedBytes);

        fixed (float* aPtr = a)
        fixed (byte* bPtr = b)
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
                    long bprCapture = packedBytesPerRow;
                    Parallel.For(0, N, j =>
                    {
                        byte* bRow = (byte*)bCapture + j * bprCapture;
                        oRow[j] = DotI2S(aRow, bRow, K, scale);
                    });
                }
                else
                {
                    for (int j = 0; j < N; j++)
                    {
                        byte* bRow = bPtr + j * packedBytesPerRow;
                        oRow[j] = DotI2S(aRow, bRow, K, scale);
                    }
                }
            }
        }
    }

    /// <summary>
    /// Compute dot product between FP32 activation and I2_S ternary weight row.
    /// Exploits ternary nature: add/subtract/skip instead of multiply.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe float DotI2S(float* aVec, byte* bPacked, int K, float scale)
    {
        float sum = 0;
        int done = 0;
        int srcIdx = 0;

        while (done < K)
        {
            int remaining = K - done;
            int blkElements = remaining >= 128 ? 128 : remaining;
            int cols0 = blkElements >= 32 ? 32 : blkElements;
            int cols1 = blkElements >= 64 ? 32 : Math.Max(0, blkElements - 32);
            int cols2 = blkElements >= 96 ? 32 : Math.Max(0, blkElements - 64);
            int cols3 = blkElements >= 128 ? 32 : Math.Max(0, blkElements - 96);

            for (int gp = 0; gp < 32; gp++)
            {
                byte b = bPacked[srcIdx + gp];

                // Group 0: bits [7:6]
                if (gp < cols0)
                {
                    int code0 = (b >> 6) & 3;
                    if (code0 == 0) sum -= aVec[done + 0 * 32 + gp];       // -1
                    else if (code0 == 2) sum += aVec[done + 0 * 32 + gp];  // +1
                }
                // Group 1: bits [5:4]
                if (gp < cols1)
                {
                    int code1 = (b >> 4) & 3;
                    if (code1 == 0) sum -= aVec[done + 1 * 32 + gp];
                    else if (code1 == 2) sum += aVec[done + 1 * 32 + gp];
                }
                // Group 2: bits [3:2]
                if (gp < cols2)
                {
                    int code2 = (b >> 2) & 3;
                    if (code2 == 0) sum -= aVec[done + 2 * 32 + gp];
                    else if (code2 == 2) sum += aVec[done + 2 * 32 + gp];
                }
                // Group 3: bits [1:0]
                if (gp < cols3)
                {
                    int code3 = b & 3;
                    if (code3 == 0) sum -= aVec[done + 3 * 32 + gp];
                    else if (code3 == 2) sum += aVec[done + 3 * 32 + gp];
                }
            }
            srcIdx += 32;
            done += blkElements;
        }
        return sum * scale;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float ReadScale(ReadOnlySpan<byte> data, long packedBytesTotal)
    {
        return BitConverter.ToSingle(data.Slice((int)packedBytesTotal, 4));
    }
}
