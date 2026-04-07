using System.Text;

namespace Daisi.Llogos.Gguf;

/// <summary>
/// Binary format for GGUF shard files (embed, output, and per-layer shards).
///
/// Layout:
///   [4 bytes: magic "GSHD" = 0x44485347]
///   [4 bytes: format version (1)]
///   [4 bytes: shard type enum]
///   [4 bytes: layer index (-1 for embed/output)]
///   [4 bytes: tensor count]
///   For each tensor:
///     [4 bytes: name length]
///     [N bytes: name (UTF-8)]
///     [8 bytes: data offset within shard data section]
///     [8 bytes: data byte size]
///   [padding to 32-byte alignment]
///   [tensor data: contiguous raw bytes]
/// </summary>
public static class GgufShardFormat
{
    public const uint Magic = 0x44485347; // "GSHD" little-endian
    public const uint FormatVersion = 1;
    public const uint Alignment = 32;

    public enum ShardType : uint
    {
        Embed = 0,
        Output = 1,
        Layer = 2,
    }

    /// <summary>
    /// Write a shard file: index header + raw tensor data.
    /// tensors is a list of (name, dataBytes) pairs. Data is written contiguously
    /// after the index, and offsets are computed relative to the data section start.
    /// </summary>
    public static void WriteShard(Stream output, ShardType type, int layerIndex,
        IReadOnlyList<(string name, byte[] data)> tensors)
    {
        using var writer = new BinaryWriter(output, Encoding.UTF8, leaveOpen: true);

        writer.Write(Magic);
        writer.Write(FormatVersion);
        writer.Write((uint)type);
        writer.Write(layerIndex);
        writer.Write(tensors.Count);

        // Compute data offsets: first pass — measure the index section size
        // to determine where the data section starts (after alignment padding)
        long indexStart = output.Position;

        // We need to write tensor entries, but offsets depend on the total index size.
        // Two-pass: compute offsets first, then write.
        long currentDataOffset = 0;
        var entries = new List<(string name, long offset, long size)>(tensors.Count);
        foreach (var (name, data) in tensors)
        {
            entries.Add((name, currentDataOffset, data.Length));
            currentDataOffset += data.Length;
        }

        // Write tensor index entries
        foreach (var (name, offset, size) in entries)
        {
            var nameBytes = Encoding.UTF8.GetBytes(name);
            writer.Write(nameBytes.Length);
            writer.Write(nameBytes);
            writer.Write(offset);
            writer.Write(size);
        }

        // Pad to alignment boundary
        long currentPos = output.Position;
        long remainder = currentPos % Alignment;
        if (remainder != 0)
        {
            int padding = (int)(Alignment - remainder);
            writer.Write(new byte[padding]);
        }

        // Write tensor data contiguously
        foreach (var (_, data) in tensors)
        {
            writer.Write(data);
        }
    }

    /// <summary>
    /// Write a shard file using spans instead of byte arrays.
    /// Reads tensor data directly from a source stream at specified offsets.
    /// More memory-efficient for large models.
    /// </summary>
    public static void WriteShardFromSource(Stream output, ShardType type, int layerIndex,
        IReadOnlyList<(string name, long sourceOffset, long size)> tensorRefs, Stream source)
    {
        using var writer = new BinaryWriter(output, Encoding.UTF8, leaveOpen: true);

        writer.Write(Magic);
        writer.Write(FormatVersion);
        writer.Write((uint)type);
        writer.Write(layerIndex);
        writer.Write(tensorRefs.Count);

        // Compute contiguous data offsets
        long currentDataOffset = 0;
        var entries = new List<(string name, long offset, long size)>(tensorRefs.Count);
        foreach (var (name, _, size) in tensorRefs)
        {
            entries.Add((name, currentDataOffset, size));
            currentDataOffset += size;
        }

        // Write tensor index entries
        foreach (var (name, offset, size) in entries)
        {
            var nameBytes = Encoding.UTF8.GetBytes(name);
            writer.Write(nameBytes.Length);
            writer.Write(nameBytes);
            writer.Write(offset);
            writer.Write(size);
        }

        // Pad to alignment boundary
        long currentPos = output.Position;
        long remainder = currentPos % Alignment;
        if (remainder != 0)
        {
            int padding = (int)(Alignment - remainder);
            writer.Write(new byte[padding]);
        }

        // Copy tensor data from source stream
        var buffer = new byte[1024 * 1024]; // 1MB copy buffer
        foreach (var (_, sourceOffset, size) in tensorRefs)
        {
            source.Seek(sourceOffset, SeekOrigin.Begin);
            long remaining = size;
            while (remaining > 0)
            {
                int toRead = (int)Math.Min(remaining, buffer.Length);
                int bytesRead = source.Read(buffer, 0, toRead);
                if (bytesRead == 0) throw new EndOfStreamException("Unexpected end of source stream.");
                output.Write(buffer, 0, bytesRead);
                remaining -= bytesRead;
            }
        }
    }
}
