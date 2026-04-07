using Daisi.Llogos.Gguf;

namespace Daisi.Llogos.Tests.Gguf;

/// <summary>
/// Tests for GGUF model splitting into per-layer shards.
/// Uses a synthetic 2-layer GGUF model built in-memory.
/// </summary>
public class GgufSplitterTests
{
    /// <summary>
    /// Build a minimal synthetic GGUF file with embedding, output, and N transformer layers.
    /// Each tensor is filled with its index for verification.
    /// </summary>
    private static string WriteSyntheticGguf(string dir, int numLayers = 2)
    {
        var path = Path.Combine(dir, "test-model.gguf");
        using var fs = File.Create(path);
        using var bw = new BinaryWriter(fs);

        // Tensor definitions: name, element count
        var tensors = new List<(string name, int elements)>
        {
            ("token_embd.weight", 64),
            ("output_norm.weight", 16),
            ("output.weight", 64),
        };
        for (int i = 0; i < numLayers; i++)
        {
            tensors.Add(($"blk.{i}.attn_norm.weight", 16));
            tensors.Add(($"blk.{i}.attn_q.weight", 32));
            tensors.Add(($"blk.{i}.attn_k.weight", 32));
            tensors.Add(($"blk.{i}.attn_v.weight", 32));
            tensors.Add(($"blk.{i}.attn_output.weight", 32));
            tensors.Add(($"blk.{i}.ffn_norm.weight", 16));
            tensors.Add(($"blk.{i}.ffn_gate.weight", 32));
            tensors.Add(($"blk.{i}.ffn_up.weight", 32));
            tensors.Add(($"blk.{i}.ffn_down.weight", 32));
        }

        // Header
        bw.Write((uint)0x46554747); // magic "GGUF"
        bw.Write((uint)3); // version
        bw.Write((ulong)tensors.Count); // tensor_count
        bw.Write((ulong)3); // metadata_kv_count

        // Metadata: architecture = "llama"
        WriteMetadataString(bw, "general.architecture", "llama");
        // Metadata: block_count
        WriteMetadataUint32(bw, "llama.block_count", (uint)numLayers);
        // Metadata: embedding_length (for ModelConfig)
        WriteMetadataUint32(bw, "llama.embedding_length", 16);

        // Tensor info section
        ulong currentOffset = 0;
        foreach (var (name, elements) in tensors)
        {
            var nameBytes = System.Text.Encoding.UTF8.GetBytes(name);
            bw.Write((ulong)nameBytes.Length);
            bw.Write(nameBytes);
            bw.Write((uint)1); // 1D
            bw.Write((ulong)elements);
            bw.Write((uint)GgmlType.F32);
            bw.Write(currentOffset);
            currentOffset += (ulong)(elements * 4);
        }

        // Pad to 32-byte alignment
        var pos = fs.Position;
        var padding = (int)((32 - (pos % 32)) % 32);
        for (int i = 0; i < padding; i++) bw.Write((byte)0);

        // Tensor data: each float = sequential counter for verification
        float counter = 0;
        foreach (var (_, elements) in tensors)
        {
            for (int i = 0; i < elements; i++)
                bw.Write(counter++);
        }

        return path;
    }

    private static void WriteMetadataString(BinaryWriter bw, string key, string value)
    {
        var keyBytes = System.Text.Encoding.UTF8.GetBytes(key);
        bw.Write((ulong)keyBytes.Length);
        bw.Write(keyBytes);
        bw.Write((uint)GgufMetadataValueType.String);
        var valBytes = System.Text.Encoding.UTF8.GetBytes(value);
        bw.Write((ulong)valBytes.Length);
        bw.Write(valBytes);
    }

    private static void WriteMetadataUint32(BinaryWriter bw, string key, uint value)
    {
        var keyBytes = System.Text.Encoding.UTF8.GetBytes(key);
        bw.Write((ulong)keyBytes.Length);
        bw.Write(keyBytes);
        bw.Write((uint)GgufMetadataValueType.Uint32);
        bw.Write(value);
    }

    [Fact]
    public void Split_SyntheticModel_CreatesExpectedShardFiles()
    {
        using var tmpDir = new TempDir();
        var modelPath = WriteSyntheticGguf(tmpDir.Path, numLayers: 2);
        var shardDir = Path.Combine(tmpDir.Path, "shards");

        var manifest = GgufSplitter.Split(modelPath, shardDir);

        Assert.Equal(1, manifest.Version);
        Assert.Equal("test-model.gguf", manifest.ModelFileName);
        Assert.Equal(2, manifest.TotalLayers);

        // Check files exist
        Assert.True(File.Exists(Path.Combine(shardDir, manifest.Header.FileName)));
        Assert.True(File.Exists(Path.Combine(shardDir, manifest.Embed.FileName)));
        Assert.True(File.Exists(Path.Combine(shardDir, manifest.Output.FileName)));
        Assert.Equal(2, manifest.Layers.Count);
        Assert.True(File.Exists(Path.Combine(shardDir, manifest.Layers[0].FileName)));
        Assert.True(File.Exists(Path.Combine(shardDir, manifest.Layers[1].FileName)));
        Assert.True(File.Exists(Path.Combine(shardDir, "test-model.gguf.manifest.json")));
    }

    [Fact]
    public void Split_HeaderShard_IsParsableAsGguf()
    {
        using var tmpDir = new TempDir();
        var modelPath = WriteSyntheticGguf(tmpDir.Path, numLayers: 2);
        var shardDir = Path.Combine(tmpDir.Path, "shards");
        var manifest = GgufSplitter.Split(modelPath, shardDir);

        // The header shard should be parsable as a GGUF (contains all metadata and tensor info)
        var headerPath = Path.Combine(shardDir, manifest.Header.FileName);
        using var headerStream = File.OpenRead(headerPath);
        var gguf = GgufFile.Read(headerStream);

        Assert.Equal("llama", gguf.GetMetadataString("general.architecture"));
        Assert.True(gguf.Tensors.Count > 0);
    }

    [Fact]
    public void Split_LayerShard_ContainsCorrectTensors()
    {
        using var tmpDir = new TempDir();
        var modelPath = WriteSyntheticGguf(tmpDir.Path, numLayers: 2);
        var shardDir = Path.Combine(tmpDir.Path, "shards");
        GgufSplitter.Split(modelPath, shardDir);

        // Read layer 0 shard index
        var layer0Path = Path.Combine(shardDir, "test-model.gguf.layer.0");
        using var layer0Stream = File.OpenRead(layer0Path);
        var index = GgufShardIndex.Read(layer0Stream);

        Assert.Equal(GgufShardFormat.ShardType.Layer, index.Type);
        Assert.Equal(0, index.LayerIndex);
        // Standard attention layer: 9 tensors (attn_norm, q, k, v, o, ffn_norm, gate, up, down)
        Assert.Equal(9, index.Tensors.Count);
        Assert.Contains("blk.0.attn_q.weight", index.Tensors.Keys);
        Assert.Contains("blk.0.ffn_gate.weight", index.Tensors.Keys);
    }

    [Fact]
    public void Split_EmbedShard_ContainsTokenEmbedding()
    {
        using var tmpDir = new TempDir();
        var modelPath = WriteSyntheticGguf(tmpDir.Path, numLayers: 2);
        var shardDir = Path.Combine(tmpDir.Path, "shards");
        GgufSplitter.Split(modelPath, shardDir);

        var embedPath = Path.Combine(shardDir, "test-model.gguf.embed");
        using var embedStream = File.OpenRead(embedPath);
        var index = GgufShardIndex.Read(embedStream);

        Assert.Equal(GgufShardFormat.ShardType.Embed, index.Type);
        Assert.Single(index.Tensors);
        Assert.Contains("token_embd.weight", index.Tensors.Keys);
    }

    [Fact]
    public void Split_TensorData_MatchesOriginal()
    {
        using var tmpDir = new TempDir();
        var modelPath = WriteSyntheticGguf(tmpDir.Path, numLayers: 2);
        var shardDir = Path.Combine(tmpDir.Path, "shards");
        GgufSplitter.Split(modelPath, shardDir);

        // Read a tensor from the original GGUF
        using var originalStream = File.OpenRead(modelPath);
        var gguf = GgufFile.Read(originalStream);
        var tensorInfo = gguf.Tensors.First(t => t.Name == "blk.0.attn_q.weight");
        var originalData = gguf.ReadTensorData(originalStream, tensorInfo);

        // Read the same tensor from the layer shard
        var layer0Path = Path.Combine(shardDir, "test-model.gguf.layer.0");
        using var shardStream = File.OpenRead(layer0Path);
        var index = GgufShardIndex.Read(shardStream);
        var (offset, byteSize) = index.Tensors["blk.0.attn_q.weight"];
        var shardData = new byte[byteSize];
        shardStream.Seek(index.DataSectionOffset + offset, SeekOrigin.Begin);
        shardStream.ReadExactly(shardData);

        Assert.Equal(originalData, shardData);
    }

    [Fact]
    public void Split_ManifestRoundTrip_Preserves()
    {
        using var tmpDir = new TempDir();
        var modelPath = WriteSyntheticGguf(tmpDir.Path, numLayers: 2);
        var shardDir = Path.Combine(tmpDir.Path, "shards");
        GgufSplitter.Split(modelPath, shardDir);

        var manifestPath = Path.Combine(shardDir, "test-model.gguf.manifest.json");
        var manifest = GgufShardManifest.FromJsonFile(manifestPath);

        Assert.Equal(1, manifest.Version);
        Assert.Equal(2, manifest.TotalLayers);
        Assert.Equal(2, manifest.Layers.Count);
    }

    [Fact]
    public void GetShardFilesForStage_ReturnsCorrectFiles()
    {
        var manifest = new GgufShardManifest
        {
            ModelFileName = "model.gguf",
            TotalLayers = 4,
            Header = new ShardFileInfo { FileName = "model.gguf.header" },
            Embed = new ShardFileInfo { FileName = "model.gguf.embed" },
            Output = new ShardFileInfo { FileName = "model.gguf.output" },
            Layers =
            [
                new() { LayerIndex = 0, FileName = "model.gguf.layer.0" },
                new() { LayerIndex = 1, FileName = "model.gguf.layer.1" },
                new() { LayerIndex = 2, FileName = "model.gguf.layer.2" },
                new() { LayerIndex = 3, FileName = "model.gguf.layer.3" },
            ],
        };

        // Stage 0: embedding + layers [0,2)
        var stage0 = GgufSplitter.GetShardFilesForStage(manifest, 0, 2, includeEmbedding: true, includeOutputHead: false);
        Assert.Contains("model.gguf.header", stage0);
        Assert.Contains("model.gguf.embed", stage0);
        Assert.DoesNotContain("model.gguf.output", stage0);
        Assert.Contains("model.gguf.layer.0", stage0);
        Assert.Contains("model.gguf.layer.1", stage0);
        Assert.DoesNotContain("model.gguf.layer.2", stage0);

        // Stage 1: output head + layers [2,4)
        var stage1 = GgufSplitter.GetShardFilesForStage(manifest, 2, 4, includeEmbedding: false, includeOutputHead: true);
        Assert.Contains("model.gguf.header", stage1);
        Assert.DoesNotContain("model.gguf.embed", stage1);
        Assert.Contains("model.gguf.output", stage1);
        Assert.Contains("model.gguf.layer.2", stage1);
        Assert.Contains("model.gguf.layer.3", stage1);
    }

    /// <summary>Helper to create a temp directory that is cleaned up after the test.</summary>
    private sealed class TempDir : IDisposable
    {
        public string Path { get; } = System.IO.Path.Combine(System.IO.Path.GetTempPath(), "llogos-test-" + Guid.NewGuid().ToString("N")[..8]);

        public TempDir() => Directory.CreateDirectory(Path);

        public void Dispose()
        {
            try { Directory.Delete(Path, true); } catch { /* best effort */ }
        }
    }
}
