namespace Daisi.Llama.Gguf;

/// <summary>
/// Represents a parsed GGUF file: header, metadata, and tensor info.
/// Does not hold tensor data in memory — use ReadTensorData to access on demand.
/// </summary>
public sealed class GgufFile
{
    public required GgufHeader Header { get; init; }
    public required IReadOnlyList<GgufMetadataKv> Metadata { get; init; }
    public required IReadOnlyList<GgufTensorInfo> Tensors { get; init; }

    /// <summary>
    /// Absolute file offset where tensor data begins.
    /// </summary>
    public required long TensorDataOffset { get; init; }

    /// <summary>
    /// Get a metadata string value by key, or null if not found.
    /// </summary>
    public string? GetMetadataString(string key)
    {
        var kv = Metadata.FirstOrDefault(m => m.Key == key);
        return kv?.ValueAs<string>();
    }

    /// <summary>
    /// Get a metadata value by key, or default if not found.
    /// </summary>
    public T? GetMetadata<T>(string key)
    {
        var kv = Metadata.FirstOrDefault(m => m.Key == key);
        return kv == null ? default : kv.ValueAs<T>();
    }

    /// <summary>
    /// Read raw tensor data bytes from the stream.
    /// </summary>
    public byte[] ReadTensorData(Stream stream, GgufTensorInfo tensor)
    {
        var absoluteOffset = TensorDataOffset + (long)tensor.Offset;
        stream.Seek(absoluteOffset, SeekOrigin.Begin);
        var data = new byte[tensor.ByteSize];
        stream.ReadExactly(data);
        return data;
    }

    /// <summary>
    /// Parse a GGUF file from a stream. Reads header, metadata, and tensor info.
    /// Does not read tensor data into memory.
    /// </summary>
    public static GgufFile Read(Stream stream)
    {
        var reader = new GgufReader(stream);
        var header = reader.ReadHeader();

        // Read metadata
        var metadata = new List<GgufMetadataKv>((int)header.MetadataKvCount);
        for (ulong i = 0; i < header.MetadataKvCount; i++)
            metadata.Add(reader.ReadMetadataKv());

        // Determine alignment (default 32)
        uint alignment = 32;
        var alignmentKv = metadata.FirstOrDefault(m => m.Key == "general.alignment");
        if (alignmentKv != null)
            alignment = alignmentKv.ValueAs<uint>();

        // Read tensor info
        var tensors = new List<GgufTensorInfo>((int)header.TensorCount);
        for (ulong i = 0; i < header.TensorCount; i++)
            tensors.Add(reader.ReadTensorInfo());

        // Calculate tensor data offset (aligned)
        var currentPos = stream.Position;
        var tensorDataOffset = AlignOffset(currentPos, alignment);

        return new GgufFile
        {
            Header = header,
            Metadata = metadata,
            Tensors = tensors,
            TensorDataOffset = tensorDataOffset,
        };
    }

    private static long AlignOffset(long offset, uint alignment)
    {
        var remainder = offset % alignment;
        return remainder == 0 ? offset : offset + (alignment - remainder);
    }
}
