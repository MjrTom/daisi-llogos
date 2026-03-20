namespace Daisi.Llama.Gguf;

/// <summary>
/// GGML tensor data types. Maps directly to ggml_type enum in llama.cpp.
/// Each type defines a block_size (elements per block) and type_size (bytes per block).
/// </summary>
public enum GgmlType : uint
{
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    // Q4_2 = 4, // removed
    // Q4_3 = 5, // removed
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    IQ2_XXS = 16,
    IQ2_XS = 17,
    IQ3_XXS = 18,
    IQ1_S = 19,
    IQ4_NL = 20,
    IQ3_S = 21,
    IQ2_S = 22,
    IQ4_XS = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    IQ1_M = 29,
    BF16 = 30,
    // 31-33 reserved
    TQ1_0 = 34,
    TQ2_0 = 35,
    I2_S = 36, // BitNet ternary: 2-bit packed, per-tensor scale, 128-element interleaved groups
    // 37-38 reserved
    MXFP4 = 39,
    NVFP4 = 40,
}

/// <summary>
/// Provides block size and byte size per block for each GgmlType.
/// </summary>
public static class GgmlTypeInfo
{
    /// <summary>
    /// Number of elements per quantization block.
    /// </summary>
    public static int BlockSize(GgmlType type) => type switch
    {
        GgmlType.F32 => 1,
        GgmlType.F16 => 1,
        GgmlType.Q4_0 => 32,
        GgmlType.Q4_1 => 32,
        GgmlType.Q5_0 => 32,
        GgmlType.Q5_1 => 32,
        GgmlType.Q8_0 => 32,
        GgmlType.Q8_1 => 32,
        GgmlType.Q2_K => 256,
        GgmlType.Q3_K => 256,
        GgmlType.Q4_K => 256,
        GgmlType.Q5_K => 256,
        GgmlType.Q6_K => 256,
        GgmlType.Q8_K => 256,
        GgmlType.IQ2_XXS => 256,
        GgmlType.IQ2_XS => 256,
        GgmlType.IQ3_XXS => 256,
        GgmlType.IQ1_S => 256,
        GgmlType.IQ4_NL => 32,
        GgmlType.IQ3_S => 256,
        GgmlType.IQ2_S => 256,
        GgmlType.IQ4_XS => 256,
        GgmlType.I8 => 1,
        GgmlType.I16 => 1,
        GgmlType.I32 => 1,
        GgmlType.I64 => 1,
        GgmlType.F64 => 1,
        GgmlType.IQ1_M => 256,
        GgmlType.BF16 => 1,
        GgmlType.TQ1_0 => 256,
        GgmlType.TQ2_0 => 256,
        GgmlType.I2_S => 1,
        GgmlType.MXFP4 => 256,
        GgmlType.NVFP4 => 64,
        _ => throw new NotSupportedException($"Unknown GGML type: {type}")
    };

    /// <summary>
    /// Bytes per quantization block.
    /// </summary>
    public static int TypeSize(GgmlType type) => type switch
    {
        GgmlType.F32 => 4,
        GgmlType.F16 => 2,
        GgmlType.Q4_0 => 18,
        GgmlType.Q4_1 => 20,
        GgmlType.Q5_0 => 22,
        GgmlType.Q5_1 => 24,
        GgmlType.Q8_0 => 34,
        GgmlType.Q8_1 => 36,
        GgmlType.Q2_K => 96,
        GgmlType.Q3_K => 108,
        GgmlType.Q4_K => 144,
        GgmlType.Q5_K => 176,
        GgmlType.Q6_K => 210,
        GgmlType.Q8_K => 292,
        GgmlType.IQ2_XXS => 80,
        GgmlType.IQ2_XS => 112,
        GgmlType.IQ3_XXS => 98,
        GgmlType.IQ1_S => 50,
        GgmlType.IQ4_NL => 18,
        GgmlType.IQ3_S => 128,
        GgmlType.IQ2_S => 132,
        GgmlType.IQ4_XS => 136,
        GgmlType.I8 => 1,
        GgmlType.I16 => 2,
        GgmlType.I32 => 4,
        GgmlType.I64 => 8,
        GgmlType.F64 => 8,
        GgmlType.IQ1_M => 56,
        GgmlType.BF16 => 2,
        GgmlType.TQ1_0 => 54,
        GgmlType.TQ2_0 => 64,
        GgmlType.I2_S => 1,
        GgmlType.MXFP4 => 132,
        GgmlType.NVFP4 => 36,
        _ => throw new NotSupportedException($"Unknown GGML type: {type}")
    };

    /// <summary>
    /// Calculate total byte size for a tensor with the given element count and type.
    /// </summary>
    public static ulong ByteSize(GgmlType type, ulong elementCount)
    {
        // I2_S: 2-bit packed (4 elements/byte) + 32-byte trailer with per-tensor scale
        if (type == GgmlType.I2_S)
            return elementCount / 4 + 32;

        var blockSize = (ulong)BlockSize(type);
        var typeSize = (ulong)TypeSize(type);
        var blockCount = (elementCount + blockSize - 1) / blockSize;
        return blockCount * typeSize;
    }
}
