using System.Text;

namespace Daisi.Llogos.Gguf;

/// <summary>
/// Reads the binary index at the start of a GGUF shard file.
/// Returns tensor name → (offset, size) within the shard's data section.
/// </summary>
public sealed class GgufShardIndex
{
    public GgufShardFormat.ShardType Type { get; init; }
    public int LayerIndex { get; init; }

    /// <summary>Absolute byte offset in the shard file where the data section begins.</summary>
    public long DataSectionOffset { get; init; }

    /// <summary>Tensor entries: name → (offset relative to DataSectionOffset, byteSize).</summary>
    public Dictionary<string, (long offset, long byteSize)> Tensors { get; init; } = new();

    /// <summary>
    /// Parse the shard index from a stream positioned at the start of the shard file.
    /// </summary>
    public static GgufShardIndex Read(Stream stream)
    {
        using var reader = new BinaryReader(stream, Encoding.UTF8, leaveOpen: true);

        uint magic = reader.ReadUInt32();
        if (magic != GgufShardFormat.Magic)
            throw new InvalidDataException($"Invalid shard magic: 0x{magic:X8} (expected 0x{GgufShardFormat.Magic:X8})");

        uint version = reader.ReadUInt32();
        if (version != GgufShardFormat.FormatVersion)
            throw new InvalidDataException($"Unsupported shard format version: {version}");

        var type = (GgufShardFormat.ShardType)reader.ReadUInt32();
        int layerIndex = reader.ReadInt32();
        int tensorCount = reader.ReadInt32();

        var tensors = new Dictionary<string, (long offset, long byteSize)>(tensorCount);
        for (int i = 0; i < tensorCount; i++)
        {
            int nameLen = reader.ReadInt32();
            var nameBytes = reader.ReadBytes(nameLen);
            string name = Encoding.UTF8.GetString(nameBytes);
            long offset = reader.ReadInt64();
            long byteSize = reader.ReadInt64();
            tensors[name] = (offset, byteSize);
        }

        // Data section starts after alignment padding
        long currentPos = stream.Position;
        long remainder = currentPos % GgufShardFormat.Alignment;
        long dataSectionOffset = remainder == 0 ? currentPos : currentPos + (GgufShardFormat.Alignment - remainder);

        return new GgufShardIndex
        {
            Type = type,
            LayerIndex = layerIndex,
            DataSectionOffset = dataSectionOffset,
            Tensors = tensors,
        };
    }

    /// <summary>
    /// Get the absolute byte offset of a tensor's data within the shard file.
    /// </summary>
    public long GetAbsoluteOffset(string tensorName)
    {
        if (!Tensors.TryGetValue(tensorName, out var entry))
            throw new InvalidDataException($"Tensor not found in shard: {tensorName}");
        return DataSectionOffset + entry.offset;
    }
}
