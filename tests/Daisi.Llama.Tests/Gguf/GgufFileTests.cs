using Daisi.Llama.Gguf;

namespace Daisi.Llama.Tests.Gguf;

/// <summary>
/// Integration tests for reading a complete GGUF file.
/// Requires the Qwen 3.5 0.8B Q8_0 model on disk.
/// </summary>
public class GgufFileTests
{
    [Fact]
    public void ReadFile_Synthetic_ParsesHeaderAndMetadata()
    {
        // Build a minimal valid GGUF file with 1 metadata KV and 1 tensor
        using var ms = new MemoryStream();
        using var bw = new BinaryWriter(ms);

        // Header
        bw.Write((byte)0x47); bw.Write((byte)0x47); bw.Write((byte)0x55); bw.Write((byte)0x46); // magic
        bw.Write((uint)3); // version
        bw.Write((ulong)1); // tensor_count
        bw.Write((ulong)1); // metadata_kv_count

        // Metadata: general.architecture = "test"
        var key = "general.architecture"u8;
        bw.Write((uint)key.Length);
        bw.Write(key);
        bw.Write((uint)GgufMetadataValueType.String);
        var val = "test"u8;
        bw.Write((uint)val.Length);
        bw.Write(val);

        // Tensor info: "weight" (1D, 32 elements, F32)
        var tname = "weight"u8;
        bw.Write((uint)tname.Length);
        bw.Write(tname);
        bw.Write((uint)1); // 1D
        bw.Write((ulong)32); // 32 elements
        bw.Write((uint)GgmlType.F32); // type
        bw.Write((ulong)0); // offset

        // Pad to alignment (32 bytes)
        var pos = ms.Position;
        var alignment = 32;
        var padding = (alignment - (int)(pos % alignment)) % alignment;
        for (var i = 0; i < padding; i++) bw.Write((byte)0);

        // Tensor data: 32 floats = 128 bytes
        for (var i = 0; i < 32; i++) bw.Write((float)i);

        ms.Position = 0;
        var file = GgufFile.Read(ms);

        Assert.Equal(3u, file.Header.Version);
        Assert.Single(file.Metadata);
        Assert.Equal("test", file.GetMetadataString("general.architecture"));
        Assert.Single(file.Tensors);
        Assert.Equal("weight", file.Tensors[0].Name);
        Assert.Equal(32ul, file.Tensors[0].ElementCount);
    }

    [Fact]
    public void ReadFile_Qwen35_08B_Q8_0_ParsesSuccessfully()
    {
        if (!TestConstants.ModelExists)
        {
            Assert.Skip("Test model not found at " + TestConstants.Qwen35_08B_Q8_0);
            return;
        }

        using var stream = File.OpenRead(TestConstants.Qwen35_08B_Q8_0);
        var file = GgufFile.Read(stream);

        // Validate header
        Assert.Equal(GgufMagic.GGUF, file.Header.Magic);
        Assert.True(file.Header.Version >= 2);
        Assert.True(file.Header.TensorCount > 0);
        Assert.True(file.Header.MetadataKvCount > 0);

        // Validate architecture
        var arch = file.GetMetadataString("general.architecture");
        Assert.NotNull(arch);
        Assert.NotEmpty(arch);

        // Validate tensors were parsed
        Assert.Equal((int)file.Header.TensorCount, file.Tensors.Count);
        Assert.All(file.Tensors, t =>
        {
            Assert.NotEmpty(t.Name);
            Assert.True(t.NDimensions > 0);
            Assert.True(t.ElementCount > 0);
        });
    }

    [Fact]
    public void ReadFile_Qwen35_08B_Q8_0_HasExpectedArchitecture()
    {
        if (!TestConstants.ModelExists)
        {
            Assert.Skip("Test model not found");
            return;
        }

        using var stream = File.OpenRead(TestConstants.Qwen35_08B_Q8_0);
        var file = GgufFile.Read(stream);

        // Qwen 3.5 should report its architecture
        var arch = file.GetMetadataString("general.architecture");
        Assert.Contains("qwen", arch, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void ReadFile_Qwen35_08B_Q8_0_TensorDataAccessible()
    {
        if (!TestConstants.ModelExists)
        {
            Assert.Skip("Test model not found");
            return;
        }

        using var stream = File.OpenRead(TestConstants.Qwen35_08B_Q8_0);
        var file = GgufFile.Read(stream);

        // Should be able to read the first tensor's raw data
        var firstTensor = file.Tensors[0];
        var data = file.ReadTensorData(stream, firstTensor);
        Assert.NotNull(data);
        Assert.True(data.Length > 0);
        Assert.Equal((int)firstTensor.ByteSize, data.Length);
    }
}
