namespace Daisi.Llogos.Inference.DaisiTurbo;

/// <summary>
/// MSE-optimal scalar quantizer for WHT-rotated vectors.
/// After random rotation, coordinates follow a concentrated distribution with
/// predictable range, enabling a fixed quantization grid without per-vector calibration.
/// Supports 2-bit (4 levels), 3-bit (8 levels), and 4-bit (16 levels) quantization.
/// </summary>
public sealed class ScalarQuantizer
{
    private readonly float[] _boundaries;
    private readonly float[] _centroids;

    /// <summary>Number of bits per scalar value.</summary>
    public int Bits { get; }

    /// <summary>Number of quantization levels (2^bits).</summary>
    public int Levels { get; }

    private ScalarQuantizer(int bits, float[] boundaries, float[] centroids)
    {
        Bits = bits;
        Levels = 1 << bits;
        _boundaries = boundaries;
        _centroids = centroids;
    }

    /// <summary>
    /// Create a quantizer for the given bit-width.
    /// Grids are optimized for the distribution of WHT-rotated coordinates
    /// (approximately uniform after rotation of typical LLM hidden states).
    /// </summary>
    public static ScalarQuantizer Create(int bits)
    {
        return bits switch
        {
            2 => Create2Bit(),
            3 => Create3Bit(),
            4 => Create4Bit(),
            _ => throw new ArgumentOutOfRangeException(nameof(bits), "Supported: 2, 3, 4")
        };
    }

    /// <summary>
    /// Quantize a single float to its level index [0, Levels).
    /// Uses binary search into boundaries.
    /// </summary>
    public int Quantize(float value)
    {
        // Binary search: boundaries has (Levels-1) entries
        int lo = 0, hi = _boundaries.Length;
        while (lo < hi)
        {
            int mid = (lo + hi) >> 1;
            if (value > _boundaries[mid])
                lo = mid + 1;
            else
                hi = mid;
        }
        return lo;
    }

    /// <summary>
    /// Dequantize a level index back to the centroid value.
    /// </summary>
    public float Dequantize(int level) => _centroids[level];

    /// <summary>
    /// Quantize a vector in-place: values are replaced with their centroid reconstructions.
    /// Returns the packed quantized indices.
    /// </summary>
    public void QuantizeVector(ReadOnlySpan<float> input, Span<byte> packedOutput, Span<float> reconstructed)
    {
        int n = input.Length;

        switch (Bits)
        {
            case 2:
                QuantizeVector2Bit(input, packedOutput, reconstructed);
                break;
            case 3:
                QuantizeVector3Bit(input, packedOutput, reconstructed);
                break;
            case 4:
                QuantizeVector4Bit(input, packedOutput, reconstructed);
                break;
        }
    }

    /// <summary>
    /// Dequantize packed data back into float values.
    /// </summary>
    public void DequantizeVector(ReadOnlySpan<byte> packedInput, Span<float> output)
    {
        switch (Bits)
        {
            case 2:
                DequantizeVector2Bit(packedInput, output);
                break;
            case 3:
                DequantizeVector3Bit(packedInput, output);
                break;
            case 4:
                DequantizeVector4Bit(packedInput, output);
                break;
        }
    }

    /// <summary>Number of bytes needed to store n quantized values.</summary>
    public int PackedBytes(int n)
    {
        return Bits switch
        {
            2 => (n + 3) / 4,       // 4 values per byte
            3 => (n * 3 + 7) / 8,   // 3 bits per value, packed into bytes
            4 => (n + 1) / 2,       // 2 values per byte
            _ => throw new InvalidOperationException()
        };
    }

    // ── 2-bit (4 levels) ────────────────────────────────────────────────────

    private void QuantizeVector2Bit(ReadOnlySpan<float> input, Span<byte> packed, Span<float> reconstructed)
    {
        int n = input.Length;
        int byteIdx = 0;
        byte currentByte = 0;

        for (int i = 0; i < n; i++)
        {
            int level = Quantize(input[i]);
            reconstructed[i] = _centroids[level];

            int shift = (i & 3) * 2;
            currentByte |= (byte)(level << shift);

            if ((i & 3) == 3)
            {
                packed[byteIdx++] = currentByte;
                currentByte = 0;
            }
        }
        if ((n & 3) != 0)
            packed[byteIdx] = currentByte;
    }

    private void DequantizeVector2Bit(ReadOnlySpan<byte> packed, Span<float> output)
    {
        int n = output.Length;
        for (int i = 0; i < n; i++)
        {
            int byteIdx = i >> 2;
            int shift = (i & 3) * 2;
            int level = (packed[byteIdx] >> shift) & 0x3;
            output[i] = _centroids[level];
        }
    }

    // ── 3-bit (8 levels) ────────────────────────────────────────────────────

    private void QuantizeVector3Bit(ReadOnlySpan<float> input, Span<byte> packed, Span<float> reconstructed)
    {
        int n = input.Length;
        int bitPos = 0;

        for (int i = 0; i < n; i++)
        {
            int level = Quantize(input[i]);
            reconstructed[i] = _centroids[level];

            // Pack 3 bits at bitPos
            int byteIdx = bitPos >> 3;
            int bitOff = bitPos & 7;

            packed[byteIdx] |= (byte)(level << bitOff);
            if (bitOff > 5) // Spans byte boundary
                packed[byteIdx + 1] |= (byte)(level >> (8 - bitOff));

            bitPos += 3;
        }
    }

    private void DequantizeVector3Bit(ReadOnlySpan<byte> packed, Span<float> output)
    {
        int n = output.Length;
        int bitPos = 0;

        for (int i = 0; i < n; i++)
        {
            int byteIdx = bitPos >> 3;
            int bitOff = bitPos & 7;

            int level = (packed[byteIdx] >> bitOff) & 0x7;
            if (bitOff > 5)
                level |= (packed[byteIdx + 1] << (8 - bitOff)) & 0x7;

            output[i] = _centroids[level];
            bitPos += 3;
        }
    }

    // ── 4-bit (16 levels) ───────────────────────────────────────────────────

    private void QuantizeVector4Bit(ReadOnlySpan<float> input, Span<byte> packed, Span<float> reconstructed)
    {
        int n = input.Length;
        for (int i = 0; i < n; i += 2)
        {
            int level0 = Quantize(input[i]);
            reconstructed[i] = _centroids[level0];

            int level1 = i + 1 < n ? Quantize(input[i + 1]) : 0;
            if (i + 1 < n)
                reconstructed[i + 1] = _centroids[level1];

            packed[i >> 1] = (byte)(level0 | (level1 << 4));
        }
    }

    private void DequantizeVector4Bit(ReadOnlySpan<byte> packed, Span<float> output)
    {
        int n = output.Length;
        for (int i = 0; i < n; i += 2)
        {
            byte b = packed[i >> 1];
            output[i] = _centroids[b & 0xF];
            if (i + 1 < n)
                output[i + 1] = _centroids[(b >> 4) & 0xF];
        }
    }

    // ── Pre-computed quantization grids ─────────────────────────────────────
    // These are MSE-optimal for approximately uniform distributions on [-range, +range].
    // After WHT rotation of typical LLM hidden states, coordinates concentrate
    // around zero with a range determined by 1/sqrt(dim).
    // We use a normalized grid (assuming unit variance) and scale at runtime
    // via the per-head scale factor stored alongside the quantized data.

    private static ScalarQuantizer Create2Bit()
    {
        // 4 levels, MSE-optimal for uniform: Lloyd-Max on N(0,1)
        float[] boundaries = [-0.9816f, 0.0f, 0.9816f];
        float[] centroids = [-1.51f, -0.4528f, 0.4528f, 1.51f];
        return new ScalarQuantizer(2, boundaries, centroids);
    }

    private static ScalarQuantizer Create3Bit()
    {
        // 8 levels, Lloyd-Max on N(0,1)
        float[] boundaries = [-1.748f, -1.050f, -0.5006f, 0.0f, 0.5006f, 1.050f, 1.748f];
        float[] centroids = [-2.152f, -1.344f, -0.7560f, -0.2451f, 0.2451f, 0.7560f, 1.344f, 2.152f];
        return new ScalarQuantizer(3, boundaries, centroids);
    }

    private static ScalarQuantizer Create4Bit()
    {
        // 16 levels, Lloyd-Max on N(0,1)
        float[] boundaries =
        [
            -2.401f, -1.844f, -1.437f, -1.099f, -0.7975f, -0.5141f, -0.2391f,
            0.0f,
            0.2391f, 0.5141f, 0.7975f, 1.099f, 1.437f, 1.844f, 2.401f
        ];
        float[] centroids =
        [
            -2.733f, -2.069f, -1.618f, -1.256f, -0.9414f, -0.6522f, -0.3747f, -0.1194f,
            0.1194f, 0.3747f, 0.6522f, 0.9414f, 1.256f, 1.618f, 2.069f, 2.733f
        ];
        return new ScalarQuantizer(4, boundaries, centroids);
    }
}
