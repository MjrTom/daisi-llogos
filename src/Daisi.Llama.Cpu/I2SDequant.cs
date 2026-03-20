using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

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

        // Per-tensor scale stored once at end of all packed data
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
                    float scaleCapture = scale;
                    int kCapture = K;
                    Parallel.For(0, N, j =>
                    {
                        byte* bRow = (byte*)bCapture + j * bprCapture;
                        oRow[j] = DotI2S(aRow, bRow, kCapture, scaleCapture);
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

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe float DotI2S(float* aVec, byte* bPacked, int K, float scale)
    {
        if (Avx2.IsSupported)
            return DotI2SAvx2(aVec, bPacked, K) * scale;
        return DotI2SScalar(aVec, bPacked, K) * scale;
    }

    /// <summary>
    /// AVX2-optimized ternary dot product. Processes 8 elements at a time per group.
    /// For each 8 bytes: extract 2-bit codes, convert to float {-1,0,+1}, FMA with activations.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe float DotI2SAvx2(float* aVec, byte* bPacked, int K)
    {
        var acc = Vector256<float>.Zero;
        bool hasFma = Fma.IsSupported;

        // Precomputed constants for code→float conversion:
        // code: 0=-1, 1=0, 2=+1, 3=0
        // We convert code to float, then: weight = code - 1 when code != 3, else 0
        // Better: use two comparisons to create +1/-1 masks
        var vOne = Vector256.Create(1f);
        var vNegOne = Vector256.Create(-1f);
        var vZero = Vector256<float>.Zero;
        var iZero = Vector256.Create(0);
        var iTwo = Vector256.Create(2);
        var byteMask = Vector256.Create(3); // 0x03

        int chunks = K / 128; // full 128-element chunks (32 bytes each)

        for (int chunk = 0; chunk < chunks; chunk++)
        {
            byte* bp = bPacked + chunk * 32;
            float* ap = aVec + chunk * 128;

            // Process 32 bytes → 4 groups of 32 elements
            // Unrolled for constant shift values (required for Avx2.ShiftRightLogical)
            ProcessGroup(bp, ap, ref acc, byteMask, iZero, iTwo, vOne, vNegOne, vZero, hasFma, 6, 0);
            ProcessGroup(bp, ap, ref acc, byteMask, iZero, iTwo, vOne, vNegOne, vZero, hasFma, 4, 32);
            ProcessGroup(bp, ap, ref acc, byteMask, iZero, iTwo, vOne, vNegOne, vZero, hasFma, 2, 64);
            ProcessGroup(bp, ap, ref acc, byteMask, iZero, iTwo, vOne, vNegOne, vZero, hasFma, 0, 96);
        }

        float result = Vector256.Sum(acc);

        // Handle tail (K not multiple of 128)
        int done = chunks * 128;
        if (done < K)
            result += DotI2SScalarRange(aVec, bPacked, done, K);

        return result;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void ProcessGroup(byte* bp, float* ap, ref Vector256<float> acc,
        Vector256<int> byteMask, Vector256<int> iZero, Vector256<int> iTwo,
        Vector256<float> vOne, Vector256<float> vNegOne, Vector256<float> vZero,
        bool hasFma, byte shift, int aOffset)
    {
        float* aGroup = ap + aOffset;
        for (int byteOff = 0; byteOff < 32; byteOff += 8)
        {
            var bytes8 = Vector128.Create(
                bp[byteOff], bp[byteOff + 1], bp[byteOff + 2], bp[byteOff + 3],
                bp[byteOff + 4], bp[byteOff + 5], bp[byteOff + 6], bp[byteOff + 7],
                (byte)0, (byte)0, (byte)0, (byte)0, (byte)0, (byte)0, (byte)0, (byte)0);
            var ints = Avx2.ConvertToVector256Int32(bytes8);
            var codes = Avx2.And(Avx2.ShiftRightLogical(ints, shift), byteMask);
            var isNeg = Avx2.CompareEqual(codes, iZero);
            var isPos = Avx2.CompareEqual(codes, iTwo);
            var weights = Avx.BlendVariable(
                Avx.BlendVariable(vZero, vNegOne, isNeg.AsSingle()),
                vOne, isPos.AsSingle());
            var aVals = Avx.LoadVector256(aGroup + byteOff);
            if (hasFma)
                acc = Fma.MultiplyAdd(aVals, weights, acc);
            else
                acc += aVals * weights;
        }
    }

    /// <summary>
    /// Scalar fallback for the full range.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe float DotI2SScalar(float* aVec, byte* bPacked, int K)
    {
        return DotI2SScalarRange(aVec, bPacked, 0, K);
    }

    /// <summary>
    /// Scalar dot product for a range of elements [start, K).
    /// </summary>
    private static unsafe float DotI2SScalarRange(float* aVec, byte* bPacked, int start, int K)
    {
        float sum = 0;
        int chunk = start / 128;
        int done = chunk * 128;
        int srcIdx = chunk * 32;

        while (done < K)
        {
            int remaining = K - done;
            int blkElements = remaining >= 128 ? 128 : remaining;

            for (int gp = 0; gp < 32 && gp < (blkElements + 3) / 4; gp++)
            {
                byte b = bPacked[srcIdx + gp];

                if (done + gp < K)
                {
                    int c0 = (b >> 6) & 3;
                    if (c0 == 0) sum -= aVec[done + gp];
                    else if (c0 == 2) sum += aVec[done + gp];
                }
                if (done + 32 + gp < K && blkElements > 32)
                {
                    int c1 = (b >> 4) & 3;
                    if (c1 == 0) sum -= aVec[done + 32 + gp];
                    else if (c1 == 2) sum += aVec[done + 32 + gp];
                }
                if (done + 64 + gp < K && blkElements > 64)
                {
                    int c2 = (b >> 2) & 3;
                    if (c2 == 0) sum -= aVec[done + 64 + gp];
                    else if (c2 == 2) sum += aVec[done + 64 + gp];
                }
                if (done + 96 + gp < K && blkElements > 96)
                {
                    int c3 = b & 3;
                    if (c3 == 0) sum -= aVec[done + 96 + gp];
                    else if (c3 == 2) sum += aVec[done + 96 + gp];
                }
            }
            srcIdx += 32;
            done += blkElements;
        }
        return sum;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float ReadScale(ReadOnlySpan<byte> data, long packedBytesTotal)
    {
        return BitConverter.ToSingle(data.Slice((int)packedBytesTotal, 4));
    }
}
