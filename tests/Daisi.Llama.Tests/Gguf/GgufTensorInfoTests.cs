using Daisi.Llama.Gguf;

namespace Daisi.Llama.Tests.Gguf;

/// <summary>
/// Tests for GGUF tensor info parsing.
/// </summary>
public class GgufTensorInfoTests
{
    [Fact]
    public void ReadTensorInfo_2D_F32_ParsesCorrectly()
    {
        using var ms = new MemoryStream();
        using var bw = new BinaryWriter(ms);

        // Name: "blk.0.attn_q.weight"
        var name = "blk.0.attn_q.weight"u8;
        bw.Write((uint)name.Length);
        bw.Write(name);
        // n_dimensions: 2
        bw.Write((uint)2);
        // dimensions: [1024, 1024]
        bw.Write((ulong)1024);
        bw.Write((ulong)1024);
        // type: F32 = 0
        bw.Write((uint)GgmlType.F32);
        // offset: 0
        bw.Write((ulong)0);

        ms.Position = 0;
        var reader = new GgufReader(ms);
        var info = reader.ReadTensorInfo();

        Assert.Equal("blk.0.attn_q.weight", info.Name);
        Assert.Equal(2, info.NDimensions);
        Assert.Equal(1024ul, info.Dimensions[0]);
        Assert.Equal(1024ul, info.Dimensions[1]);
        Assert.Equal(GgmlType.F32, info.Type);
        Assert.Equal(0ul, info.Offset);
    }

    [Fact]
    public void ReadTensorInfo_1D_Q8_0_ParsesCorrectly()
    {
        using var ms = new MemoryStream();
        using var bw = new BinaryWriter(ms);

        var name = "output_norm.weight"u8;
        bw.Write((uint)name.Length);
        bw.Write(name);
        bw.Write((uint)1); // 1D
        bw.Write((ulong)1024); // dimension
        bw.Write((uint)GgmlType.Q8_0);
        bw.Write((ulong)4096); // offset

        ms.Position = 0;
        var reader = new GgufReader(ms);
        var info = reader.ReadTensorInfo();

        Assert.Equal("output_norm.weight", info.Name);
        Assert.Equal(1, info.NDimensions);
        Assert.Equal(GgmlType.Q8_0, info.Type);
        Assert.Equal(4096ul, info.Offset);
    }

    [Fact]
    public void TensorInfo_ElementCount_CalculatesCorrectly()
    {
        var info = new GgufTensorInfo
        {
            Name = "test",
            NDimensions = 3,
            Dimensions = [2, 3, 4],
            Type = GgmlType.F32,
            Offset = 0
        };

        Assert.Equal(24ul, info.ElementCount);
    }

    [Fact]
    public void TensorInfo_ByteSize_Q8_0_CalculatesCorrectly()
    {
        // Q8_0: block_size=32, type_size=34 bytes per block
        // 1024 elements = 32 blocks => 32 * 34 = 1088 bytes
        var info = new GgufTensorInfo
        {
            Name = "test",
            NDimensions = 1,
            Dimensions = [1024],
            Type = GgmlType.Q8_0,
            Offset = 0
        };

        Assert.Equal(1088ul, info.ByteSize);
    }

    [Fact]
    public void TensorInfo_ByteSize_F32_CalculatesCorrectly()
    {
        var info = new GgufTensorInfo
        {
            Name = "test",
            NDimensions = 2,
            Dimensions = [1024, 1024],
            Type = GgmlType.F32,
            Offset = 0
        };

        // 1024*1024 * 4 bytes = 4,194,304
        Assert.Equal(4_194_304ul, info.ByteSize);
    }
}
