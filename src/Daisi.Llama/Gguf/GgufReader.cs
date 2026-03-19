using System.Text;

namespace Daisi.Llama.Gguf;

/// <summary>
/// Low-level binary reader for GGUF format structures.
/// Reads header, metadata key-value pairs, and tensor info from a stream.
/// </summary>
public sealed class GgufReader
{
    private readonly BinaryReader _br;

    public GgufReader(Stream stream)
    {
        _br = new BinaryReader(stream, Encoding.UTF8, leaveOpen: true);
    }

    /// <summary>
    /// Read the 24-byte GGUF header.
    /// </summary>
    public GgufHeader ReadHeader()
    {
        var magic = _br.ReadUInt32();
        if (magic != GgufMagic.GGUF)
            throw new InvalidDataException($"Invalid GGUF magic: 0x{magic:X8}. Expected 0x{GgufMagic.GGUF:X8}.");

        var version = _br.ReadUInt32();
        if (version < 2 || version > 3)
            throw new NotSupportedException($"Unsupported GGUF version: {version}. Only v2 and v3 are supported.");

        var tensorCount = _br.ReadUInt64();
        var metadataKvCount = _br.ReadUInt64();

        return new GgufHeader(magic, version, tensorCount, metadataKvCount);
    }

    /// <summary>
    /// Read a single metadata key-value pair.
    /// </summary>
    public GgufMetadataKv ReadMetadataKv()
    {
        var key = ReadString();
        var type = (GgufMetadataValueType)_br.ReadUInt32();
        var value = ReadMetadataValue(type);

        return new GgufMetadataKv { Key = key, Type = type, Value = value };
    }

    /// <summary>
    /// Read a single tensor info entry.
    /// </summary>
    public GgufTensorInfo ReadTensorInfo()
    {
        var name = ReadString();
        var nDimensions = (int)_br.ReadUInt32();
        var dimensions = new ulong[nDimensions];
        for (var i = 0; i < nDimensions; i++)
            dimensions[i] = _br.ReadUInt64();
        var type = (GgmlType)_br.ReadUInt32();
        var offset = _br.ReadUInt64();

        return new GgufTensorInfo
        {
            Name = name,
            NDimensions = nDimensions,
            Dimensions = dimensions,
            Type = type,
            Offset = offset,
        };
    }

    private string ReadString()
    {
        // GGUF v2+ uses uint64 for string lengths
        var length = _br.ReadUInt64();
        var bytes = _br.ReadBytes((int)length);
        return Encoding.UTF8.GetString(bytes);
    }

    private object ReadMetadataValue(GgufMetadataValueType type) => type switch
    {
        GgufMetadataValueType.Uint8 => _br.ReadByte(),
        GgufMetadataValueType.Int8 => _br.ReadSByte(),
        GgufMetadataValueType.Uint16 => _br.ReadUInt16(),
        GgufMetadataValueType.Int16 => _br.ReadInt16(),
        GgufMetadataValueType.Uint32 => _br.ReadUInt32(),
        GgufMetadataValueType.Int32 => _br.ReadInt32(),
        GgufMetadataValueType.Float32 => _br.ReadSingle(),
        GgufMetadataValueType.Bool => _br.ReadByte() != 0,
        GgufMetadataValueType.String => ReadString(),
        GgufMetadataValueType.Uint64 => _br.ReadUInt64(),
        GgufMetadataValueType.Int64 => _br.ReadInt64(),
        GgufMetadataValueType.Float64 => _br.ReadDouble(),
        GgufMetadataValueType.Array => ReadArray(),
        _ => throw new NotSupportedException($"Unknown metadata value type: {type}")
    };

    private object ReadArray()
    {
        var elementType = (GgufMetadataValueType)_br.ReadUInt32();
        var count = _br.ReadUInt64();

        return elementType switch
        {
            GgufMetadataValueType.String => ReadTypedArray(count, () => ReadString()),
            GgufMetadataValueType.Uint8 => ReadTypedArray(count, () => _br.ReadByte()),
            GgufMetadataValueType.Int8 => ReadTypedArray(count, () => _br.ReadSByte()),
            GgufMetadataValueType.Uint16 => ReadTypedArray(count, () => _br.ReadUInt16()),
            GgufMetadataValueType.Int16 => ReadTypedArray(count, () => _br.ReadInt16()),
            GgufMetadataValueType.Uint32 => ReadTypedArray(count, () => _br.ReadUInt32()),
            GgufMetadataValueType.Int32 => ReadTypedArray(count, () => _br.ReadInt32()),
            GgufMetadataValueType.Float32 => ReadTypedArray(count, () => _br.ReadSingle()),
            GgufMetadataValueType.Uint64 => ReadTypedArray(count, () => _br.ReadUInt64()),
            GgufMetadataValueType.Int64 => ReadTypedArray(count, () => _br.ReadInt64()),
            GgufMetadataValueType.Float64 => ReadTypedArray(count, () => _br.ReadDouble()),
            GgufMetadataValueType.Bool => ReadTypedArray(count, () => _br.ReadByte() != 0),
            _ => throw new NotSupportedException($"Unsupported array element type: {elementType}")
        };
    }

    private static T[] ReadTypedArray<T>(ulong count, Func<T> readElement)
    {
        var array = new T[count];
        for (ulong i = 0; i < count; i++)
            array[i] = readElement();
        return array;
    }
}
