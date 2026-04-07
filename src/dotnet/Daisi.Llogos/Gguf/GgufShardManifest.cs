using System.Text.Json;
using System.Text.Json.Serialization;

namespace Daisi.Llogos.Gguf;

/// <summary>
/// JSON manifest describing a split GGUF model's shard files.
/// Written by GgufSplitter, consumed by hosts to determine which shards to download.
/// </summary>
public sealed class GgufShardManifest
{
    [JsonPropertyName("version")]
    public int Version { get; set; } = 1;

    [JsonPropertyName("modelFileName")]
    public string ModelFileName { get; set; } = "";

    [JsonPropertyName("totalLayers")]
    public int TotalLayers { get; set; }

    [JsonPropertyName("header")]
    public ShardFileInfo Header { get; set; } = new();

    [JsonPropertyName("embed")]
    public ShardFileInfo Embed { get; set; } = new();

    [JsonPropertyName("output")]
    public ShardFileInfo Output { get; set; } = new();

    [JsonPropertyName("layers")]
    public List<LayerShardInfo> Layers { get; set; } = [];

    /// <summary>
    /// If true, quantized tensor data in layer shards is pre-repacked to GPU-aligned layout
    /// (Q4_0: 20-byte blocks, Q8_0: 36-byte blocks). Enables zero-copy mmap → pinned → DMA.
    /// </summary>
    [JsonPropertyName("gpuAligned")]
    public bool GpuAligned { get; set; }

    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        WriteIndented = true,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
    };

    public string ToJson() => JsonSerializer.Serialize(this, JsonOptions);

    public static GgufShardManifest FromJson(string json) =>
        JsonSerializer.Deserialize<GgufShardManifest>(json, JsonOptions)
        ?? throw new InvalidDataException("Failed to parse shard manifest JSON.");

    public static GgufShardManifest FromJsonFile(string path) =>
        FromJson(File.ReadAllText(path));
}

public sealed class ShardFileInfo
{
    [JsonPropertyName("fileName")]
    public string FileName { get; set; } = "";

    [JsonPropertyName("sizeBytes")]
    public long SizeBytes { get; set; }
}

public sealed class LayerShardInfo
{
    [JsonPropertyName("layerIndex")]
    public int LayerIndex { get; set; }

    [JsonPropertyName("fileName")]
    public string FileName { get; set; } = "";

    [JsonPropertyName("sizeBytes")]
    public long SizeBytes { get; set; }
}
