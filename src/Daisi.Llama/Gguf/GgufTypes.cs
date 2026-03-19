namespace Daisi.Llama.Gguf;

/// <summary>
/// GGUF magic number constant.
/// </summary>
public static class GgufMagic
{
    /// <summary>Magic bytes: "GGUF" = 0x46554747 little-endian.</summary>
    public const uint GGUF = 0x46554747;
}

/// <summary>
/// GGUF file header: magic, version, tensor count, metadata KV count.
/// </summary>
public readonly record struct GgufHeader(uint Magic, uint Version, ulong TensorCount, ulong MetadataKvCount);

/// <summary>
/// GGUF metadata value types.
/// </summary>
public enum GgufMetadataValueType : uint
{
    Uint8 = 0,
    Int8 = 1,
    Uint16 = 2,
    Int16 = 3,
    Uint32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    Uint64 = 10,
    Int64 = 11,
    Float64 = 12,
}

/// <summary>
/// A single GGUF metadata key-value pair with typed value storage.
/// </summary>
public sealed class GgufMetadataKv
{
    public required string Key { get; init; }
    public required GgufMetadataValueType Type { get; init; }
    public required object Value { get; init; }

    /// <summary>
    /// Get the value cast to the specified type.
    /// </summary>
    public T ValueAs<T>() => (T)Value;
}

/// <summary>
/// Tensor metadata from the GGUF tensor info section.
/// </summary>
public sealed class GgufTensorInfo
{
    public required string Name { get; init; }
    public required int NDimensions { get; init; }
    public required ulong[] Dimensions { get; init; }
    public required GgmlType Type { get; init; }
    public required ulong Offset { get; init; }

    /// <summary>
    /// Total number of elements in this tensor (product of all dimensions).
    /// </summary>
    public ulong ElementCount
    {
        get
        {
            if (Dimensions.Length == 0) return 0;
            ulong count = 1;
            for (var i = 0; i < NDimensions; i++)
                count *= Dimensions[i];
            return count;
        }
    }

    /// <summary>
    /// Total byte size of this tensor's data.
    /// </summary>
    public ulong ByteSize => GgmlTypeInfo.ByteSize(Type, ElementCount);
}
