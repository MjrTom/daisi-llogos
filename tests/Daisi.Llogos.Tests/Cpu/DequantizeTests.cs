using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;
using Daisi.Llogos.Cpu;

namespace Daisi.Llogos.Tests.Cpu;

public class DequantizeQ8_0Tests
{
    [Fact]
    public void SingleBlock_KnownValues()
    {
        // Construct a Q8_0 block: scale=2.0 (FP16) + 32 signed bytes
        var block = new byte[34];
        BitConverter.TryWriteBytes(block, (Half)2.0f);

        // Fill with known values: 0, 1, 2, ..., 31
        for (int i = 0; i < 32; i++)
            block[2 + i] = (byte)(sbyte)i;

        var output = new float[32];
        Dequantize.DequantizeQ8_0(block, output);

        for (int i = 0; i < 32; i++)
            Assert.Equal(2.0f * i, output[i], 0.01f);
    }

    [Fact]
    public void SingleBlock_NegativeValues()
    {
        var block = new byte[34];
        BitConverter.TryWriteBytes(block, (Half)0.5f);

        // Fill with negative values: -1, -2, ..., -32
        for (int i = 0; i < 32; i++)
            block[2 + i] = (byte)(sbyte)(-(i + 1));

        var output = new float[32];
        Dequantize.DequantizeQ8_0(block, output);

        for (int i = 0; i < 32; i++)
            Assert.Equal(0.5f * -(i + 1), output[i], 0.01f);
    }

    [Fact]
    public void MultipleBlocks_CorrectAcrossBoundaries()
    {
        // Two blocks, different scales
        var data = new byte[34 * 2];

        // Block 0: scale=1.0, values all 5
        BitConverter.TryWriteBytes(data, (Half)1.0f);
        for (int i = 0; i < 32; i++) data[2 + i] = (byte)(sbyte)5;

        // Block 1: scale=3.0, values all -2
        BitConverter.TryWriteBytes(data.AsSpan(34), (Half)3.0f);
        for (int i = 0; i < 32; i++) data[34 + 2 + i] = unchecked((byte)(sbyte)(-2));

        var output = new float[64];
        Dequantize.DequantizeQ8_0(data, output);

        for (int i = 0; i < 32; i++)
            Assert.Equal(5.0f, output[i], 0.01f);
        for (int i = 0; i < 32; i++)
            Assert.Equal(-6.0f, output[32 + i], 0.01f);
    }

    [Fact]
    public void ZeroScale_AllZeros()
    {
        var block = new byte[34];
        BitConverter.TryWriteBytes(block, (Half)0.0f);
        for (int i = 0; i < 32; i++) block[2 + i] = (byte)(sbyte)127;

        var output = new float[32];
        Dequantize.DequantizeQ8_0(block, output);

        for (int i = 0; i < 32; i++)
            Assert.Equal(0.0f, output[i]);
    }

    [Fact]
    public void ScalarMatchesAvx2()
    {
        if (!Avx2.IsSupported) return;

        var rng = new Random(42);
        var data = new byte[34 * 100]; // 100 blocks
        rng.NextBytes(data);

        // Fix up scales to be valid FP16 (overwrite every 34 bytes with a reasonable scale)
        for (int b = 0; b < 100; b++)
            BitConverter.TryWriteBytes(data.AsSpan(b * 34), (Half)(rng.NextSingle() * 2.0f - 1.0f));

        var scalarOut = new float[32 * 100];
        var avx2Out = new float[32 * 100];

        Dequantize.DequantizeQ8_0Scalar(data, scalarOut, 100);
        Dequantize.DequantizeQ8_0Avx2(data, avx2Out, 100);

        for (int i = 0; i < scalarOut.Length; i++)
            Assert.Equal(scalarOut[i], avx2Out[i], 0.001f);
    }

    [Fact]
    public void DestinationTooSmall_Throws()
    {
        var block = new byte[34];
        var output = new float[16]; // too small

        Assert.Throws<ArgumentException>(() => Dequantize.DequantizeQ8_0(block, output));
    }
}

public class DequantizeQ4_0Tests
{
    [Fact]
    public void SingleBlock_KnownValues()
    {
        // Q4_0 block: scale=1.0 (FP16) + 16 packed bytes → 32 floats
        var block = new byte[18];
        BitConverter.TryWriteBytes(block, (Half)1.0f);

        // Pack nibbles: low nibble = i & 0xF for byte i
        // Element at position i (low nibble) has value (i & 0xF) - 8
        // Element at position 16+i (high nibble) has value (i >> 4) - 8...
        // Actually let's set all packed bytes to 0x98 = low=8, high=9
        // low nibble: 8, re-centered: 8-8 = 0
        // high nibble: 9, re-centered: 9-8 = 1
        for (int i = 0; i < 16; i++)
            block[2 + i] = 0x98;

        var output = new float[32];
        Dequantize.DequantizeQ4_0(block, output);

        // Elements 0..15 (low nibbles): 1.0 * (8 - 8) = 0.0
        for (int i = 0; i < 16; i++)
            Assert.Equal(0.0f, output[i], 0.01f);
        // Elements 16..31 (high nibbles): 1.0 * (9 - 8) = 1.0
        for (int i = 16; i < 32; i++)
            Assert.Equal(1.0f, output[i], 0.01f);
    }

    [Fact]
    public void SingleBlock_AllNibbleValues()
    {
        // Scale = 0.5, each packed byte = (hi << 4) | lo
        var block = new byte[18];
        BitConverter.TryWriteBytes(block, (Half)0.5f);

        // Pack: byte i has lo=i, hi=15-i
        for (int i = 0; i < 16; i++)
            block[2 + i] = (byte)(((15 - i) << 4) | i);

        var output = new float[32];
        Dequantize.DequantizeQ4_0(block, output);

        for (int i = 0; i < 16; i++)
        {
            Assert.Equal(0.5f * (i - 8), output[i], 0.01f);
            Assert.Equal(0.5f * (15 - i - 8), output[16 + i], 0.01f);
        }
    }

    [Fact]
    public void ScalarMatchesAvx2()
    {
        if (!Avx2.IsSupported) return;

        var rng = new Random(42);
        var data = new byte[18 * 100];
        rng.NextBytes(data);

        for (int b = 0; b < 100; b++)
            BitConverter.TryWriteBytes(data.AsSpan(b * 18), (Half)(rng.NextSingle() * 2.0f - 1.0f));

        var scalarOut = new float[32 * 100];
        var avx2Out = new float[32 * 100];

        Dequantize.DequantizeQ4_0Scalar(data, scalarOut, 100);
        Dequantize.DequantizeQ4_0Avx2(data, avx2Out, 100);

        for (int i = 0; i < scalarOut.Length; i++)
            Assert.Equal(scalarOut[i], avx2Out[i], 0.001f);
    }
}

public class DequantizeQ4_KTests
{
    [Fact]
    public void SingleSuperBlock_UniformNibbles()
    {
        // Build a Q4_K super-block (144 bytes → 256 floats)
        var block = new byte[144];

        // d = 1.0, dmin = 0.0
        BitConverter.TryWriteBytes(block, (Half)1.0f);
        BitConverter.TryWriteBytes(block.AsSpan(2), (Half)0.0f);

        // Scales: all sub-block scales = 2 (6-bit), all mins = 0
        // For j < 4: scales[j] = packed[j] & 63 → set packed[0..3] = 2
        for (int j = 0; j < 4; j++) block[4 + j] = 2;
        // mins: packed[4..7] = 0
        // High bits: packed[8..11] = 0

        // Nibbles: all 5 (so: scale * 5 - min = 2 * 5 - 0 = 10)
        for (int i = 0; i < 128; i++)
            block[16 + i] = 0x55; // low=5, high=5

        var output = new float[256];
        Dequantize.DequantizeQ4_K(block, output);

        // First 4 sub-blocks should all be 10.0
        for (int i = 0; i < 128; i++)
            Assert.Equal(10.0f, output[i], 0.01f);

        // Sub-blocks 4..7: scales[j] comes from high-bit path, which is 0 here
        // so output should be 0.0
        for (int i = 128; i < 256; i++)
            Assert.Equal(0.0f, output[i], 0.01f);
    }

    [Fact]
    public void SingleSuperBlock_WithMin()
    {
        var block = new byte[144];

        // d = 2.0, dmin = 1.0
        BitConverter.TryWriteBytes(block, (Half)2.0f);
        BitConverter.TryWriteBytes(block.AsSpan(2), (Half)1.0f);

        // scales[0] = 3, mins[0] = 5 (only testing first sub-block)
        block[4 + 0] = 3;  // scales[0] & 63 = 3
        block[4 + 4] = 5;  // mins[0] & 63 = 5
        // Rest zero

        // Nibbles for sub-block 0: all 7
        for (int i = 0; i < 16; i++)
            block[16 + i] = 0x77; // low=7, high=7

        var output = new float[256];
        Dequantize.DequantizeQ4_K(block, output);

        // sub_scale = d * scales[0] = 2 * 3 = 6
        // sub_min = dmin * mins[0] = 1 * 5 = 5
        // value = 6 * 7 - 5 = 37
        for (int i = 0; i < 32; i++)
            Assert.Equal(37.0f, output[i], 0.01f);
    }

    [Fact]
    public void ScaleUnpacking_LowSubBlocks()
    {
        // Test the 6-bit scale unpacking directly for sub-blocks 0..3
        var packed = new byte[12];
        packed[0] = 10;  // scales[0] = 10
        packed[1] = 20;  // scales[1] = 20
        packed[2] = 30;  // scales[2] = 30
        packed[3] = 40;  // scales[3] = 40
        packed[4] = 5;   // mins[0] = 5
        packed[5] = 15;  // mins[1] = 15
        packed[6] = 25;  // mins[2] = 25
        packed[7] = 35;  // mins[3] = 35

        var scales = new float[8];
        var mins = new float[8];
        Dequantize.Unpack6BitScalesMins(packed, scales, mins);

        Assert.Equal(10, scales[0]);
        Assert.Equal(20, scales[1]);
        Assert.Equal(30, scales[2]);
        Assert.Equal(40, scales[3]);
        Assert.Equal(5, mins[0]);
        Assert.Equal(15, mins[1]);
        Assert.Equal(25, mins[2]);
        Assert.Equal(35, mins[3]);
    }
}
