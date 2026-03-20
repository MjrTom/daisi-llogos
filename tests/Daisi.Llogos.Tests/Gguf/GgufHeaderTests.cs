using Daisi.Llogos.Gguf;

namespace Daisi.Llogos.Tests.Gguf;

/// <summary>
/// Tests for GGUF header parsing — magic number, version, counts.
/// Uses a synthetic GGUF byte stream for deterministic unit testing.
/// </summary>
public class GgufHeaderTests
{
    [Fact]
    public void ReadHeader_ValidMagic_ParsesCorrectly()
    {
        // GGUF v3 header: magic(4) + version(4) + tensor_count(8) + metadata_kv_count(8) = 24 bytes
        var data = new byte[24];
        // Magic: "GGUF" = 0x47475546
        data[0] = 0x47; data[1] = 0x47; data[2] = 0x55; data[3] = 0x46;
        // Version: 3 (little-endian)
        BitConverter.TryWriteBytes(data.AsSpan(4), (uint)3);
        // Tensor count: 10
        BitConverter.TryWriteBytes(data.AsSpan(8), (ulong)10);
        // Metadata KV count: 5
        BitConverter.TryWriteBytes(data.AsSpan(16), (ulong)5);

        using var stream = new MemoryStream(data);
        var reader = new GgufReader(stream);
        var header = reader.ReadHeader();

        Assert.Equal(GgufMagic.GGUF, header.Magic);
        Assert.Equal(3u, header.Version);
        Assert.Equal(10ul, header.TensorCount);
        Assert.Equal(5ul, header.MetadataKvCount);
    }

    [Fact]
    public void ReadHeader_InvalidMagic_Throws()
    {
        var data = new byte[24];
        data[0] = 0x00; data[1] = 0x00; data[2] = 0x00; data[3] = 0x00;

        using var stream = new MemoryStream(data);
        var reader = new GgufReader(stream);

        Assert.Throws<InvalidDataException>(() => reader.ReadHeader());
    }

    [Fact]
    public void ReadHeader_UnsupportedVersion_Throws()
    {
        var data = new byte[24];
        data[0] = 0x47; data[1] = 0x47; data[2] = 0x55; data[3] = 0x46;
        BitConverter.TryWriteBytes(data.AsSpan(4), (uint)99);

        using var stream = new MemoryStream(data);
        var reader = new GgufReader(stream);

        Assert.Throws<NotSupportedException>(() => reader.ReadHeader());
    }
}
