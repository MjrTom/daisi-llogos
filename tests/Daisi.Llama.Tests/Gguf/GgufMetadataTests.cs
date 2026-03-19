using Daisi.Llama.Gguf;

namespace Daisi.Llama.Tests.Gguf;

/// <summary>
/// Tests for GGUF metadata key-value parsing.
/// Uses synthetic byte streams for deterministic testing.
/// </summary>
public class GgufMetadataTests
{
    [Fact]
    public void ReadMetadataKv_StringValue_ParsesCorrectly()
    {
        // Build a single KV pair: key="general.architecture", type=STRING, value="qwen35"
        using var ms = new MemoryStream();
        using var bw = new BinaryWriter(ms);

        var key = "general.architecture"u8;
        bw.Write((uint)key.Length);
        bw.Write(key);
        bw.Write((uint)GgufMetadataValueType.String); // type
        var val = "qwen35"u8;
        bw.Write((uint)val.Length);
        bw.Write(val);

        ms.Position = 0;
        var reader = new GgufReader(ms);
        var kv = reader.ReadMetadataKv();

        Assert.Equal("general.architecture", kv.Key);
        Assert.Equal(GgufMetadataValueType.String, kv.Type);
        Assert.Equal("qwen35", kv.ValueAs<string>());
    }

    [Fact]
    public void ReadMetadataKv_Uint32Value_ParsesCorrectly()
    {
        using var ms = new MemoryStream();
        using var bw = new BinaryWriter(ms);

        var key = "general.alignment"u8;
        bw.Write((uint)key.Length);
        bw.Write(key);
        bw.Write((uint)GgufMetadataValueType.Uint32);
        bw.Write((uint)32);

        ms.Position = 0;
        var reader = new GgufReader(ms);
        var kv = reader.ReadMetadataKv();

        Assert.Equal("general.alignment", kv.Key);
        Assert.Equal(32u, kv.ValueAs<uint>());
    }

    [Fact]
    public void ReadMetadataKv_Float32Value_ParsesCorrectly()
    {
        using var ms = new MemoryStream();
        using var bw = new BinaryWriter(ms);

        var key = "tokenizer.score"u8;
        bw.Write((uint)key.Length);
        bw.Write(key);
        bw.Write((uint)GgufMetadataValueType.Float32);
        bw.Write(0.5f);

        ms.Position = 0;
        var reader = new GgufReader(ms);
        var kv = reader.ReadMetadataKv();

        Assert.Equal("tokenizer.score", kv.Key);
        Assert.Equal(0.5f, kv.ValueAs<float>());
    }

    [Fact]
    public void ReadMetadataKv_BoolValue_ParsesCorrectly()
    {
        using var ms = new MemoryStream();
        using var bw = new BinaryWriter(ms);

        var key = "general.file_type"u8;
        bw.Write((uint)key.Length);
        bw.Write(key);
        bw.Write((uint)GgufMetadataValueType.Bool);
        bw.Write((byte)1);

        ms.Position = 0;
        var reader = new GgufReader(ms);
        var kv = reader.ReadMetadataKv();

        Assert.True(kv.ValueAs<bool>());
    }

    [Fact]
    public void ReadMetadataKv_ArrayOfStrings_ParsesCorrectly()
    {
        using var ms = new MemoryStream();
        using var bw = new BinaryWriter(ms);

        var key = "tokenizer.tokens"u8;
        bw.Write((uint)key.Length);
        bw.Write(key);
        bw.Write((uint)GgufMetadataValueType.Array);
        // Array: element_type(uint32) + count(uint64) + elements
        bw.Write((uint)GgufMetadataValueType.String);
        bw.Write((ulong)3);
        foreach (var token in new[] { "hello", "world", "test" })
        {
            var bytes = System.Text.Encoding.UTF8.GetBytes(token);
            bw.Write((uint)bytes.Length);
            bw.Write(bytes);
        }

        ms.Position = 0;
        var reader = new GgufReader(ms);
        var kv = reader.ReadMetadataKv();

        Assert.Equal(GgufMetadataValueType.Array, kv.Type);
        var arr = kv.ValueAs<string[]>();
        Assert.Equal(3, arr.Length);
        Assert.Equal("hello", arr[0]);
        Assert.Equal("world", arr[1]);
        Assert.Equal("test", arr[2]);
    }

    [Fact]
    public void ReadMetadataKv_Uint64Value_ParsesCorrectly()
    {
        using var ms = new MemoryStream();
        using var bw = new BinaryWriter(ms);

        var key = "llm.context_length"u8;
        bw.Write((uint)key.Length);
        bw.Write(key);
        bw.Write((uint)GgufMetadataValueType.Uint64);
        bw.Write((ulong)262144);

        ms.Position = 0;
        var reader = new GgufReader(ms);
        var kv = reader.ReadMetadataKv();

        Assert.Equal(262144ul, kv.ValueAs<ulong>());
    }
}
