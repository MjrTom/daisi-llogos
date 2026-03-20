using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace Daisi.Llama.Cpu;

/// <summary>
/// Matrix multiplication with TQ1_0 (ternary {-1, 0, +1}) weight dequantization.
/// TQ1_0 block: 256 elements packed into 54 bytes.
///   - 52 bytes of base-3 packed trits (5 trits per byte, 3^5=243)
///   - 2 bytes padding
/// The performance advantage: dot product with ternary weights is pure addition/subtraction,
/// no floating-point multiplies needed for the weight side.
/// </summary>
internal static class TernaryMatMul
{
    private const int TQ1_0BlockSize = 256;
    private const int TQ1_0TypeSize = 54;
    private const int TritBytesPerBlock = 52; // ceil(256/5) = 52 bytes

    // Minimum N to justify thread pool overhead
    private const int ParallelThreshold = 32;

    /// <summary>
    /// FP32 activations × TQ1_0 ternary weights. b is [N×K] quantized as TQ1_0.
    /// output[i,j] = dot(a[i,:], dequant(b[j,:]))
    /// </summary>
    public static unsafe void MultiplyTQ1_0(Span<float> output, ReadOnlySpan<float> a, ReadOnlySpan<byte> b, int M, int K, int N)
    {
        int blocksPerRow = K / TQ1_0BlockSize;
        int bytesPerRow = blocksPerRow * TQ1_0TypeSize;

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
                    int blkCapture = blocksPerRow;
                    Parallel.For(0, N, CpuThreading.Options, j =>
                    {
                        byte* bRow = (byte*)bCapture + j * bprCapture;
                        oRow[j] = DotTQ1_0(aRow, bRow, blkCapture);
                    });
                }
                else
                {
                    for (int j = 0; j < N; j++)
                    {
                        byte* bRow = (byte*)bBase + j * bytesPerRow;
                        oRow[j] = DotTQ1_0(aRow, bRow, blocksPerRow);
                    }
                }
            }
        }
    }

    /// <summary>
    /// Compute dot product between FP32 activation vector and TQ1_0 ternary weight row.
    /// Pure add/subtract — no floating-point multiplies for the weight side.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe float DotTQ1_0(float* aVec, byte* bTQ1, int blockCount)
    {
        float sum = 0;
        for (int block = 0; block < blockCount; block++)
        {
            byte* blockPtr = bTQ1 + block * TQ1_0TypeSize;
            float* aPtr = aVec + block * TQ1_0BlockSize;

            int elemIdx = 0;
            for (int byteIdx = 0; byteIdx < TritBytesPerBlock && elemIdx < TQ1_0BlockSize; byteIdx++)
            {
                // Decode 5 trits from one byte using base-3
                int packed = blockPtr[byteIdx];
                for (int t = 0; t < 5 && elemIdx < TQ1_0BlockSize; t++)
                {
                    int trit = packed % 3;
                    packed /= 3;
                    // trit: 0 → -1, 1 → 0, 2 → +1
                    int weight = trit - 1;
                    // Pure add/subtract instead of multiply
                    if (weight == 1)
                        sum += aPtr[elemIdx];
                    else if (weight == -1)
                        sum -= aPtr[elemIdx];
                    // weight == 0: skip (no-op)
                    elemIdx++;
                }
            }
        }
        return sum;
    }

    /// <summary>
    /// Dequantize a TQ1_0 block to FP32 values {-1.0, 0.0, +1.0}.
    /// </summary>
    public static void Dequantize(ReadOnlySpan<byte> source, Span<float> destination)
    {
        int blockCount = source.Length / TQ1_0TypeSize;
        int destIdx = 0;

        for (int block = 0; block < blockCount; block++)
        {
            int srcOff = block * TQ1_0TypeSize;
            for (int byteIdx = 0; byteIdx < TritBytesPerBlock && destIdx < destination.Length; byteIdx++)
            {
                int packed = source[srcOff + byteIdx];
                for (int t = 0; t < 5 && destIdx < destination.Length; t++)
                {
                    int trit = packed % 3;
                    packed /= 3;
                    destination[destIdx++] = trit - 1; // 0→-1, 1→0, 2→+1
                }
            }
        }
    }
}
