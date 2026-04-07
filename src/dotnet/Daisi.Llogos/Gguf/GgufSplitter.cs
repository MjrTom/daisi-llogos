namespace Daisi.Llogos.Gguf;

/// <summary>
/// Splits a monolithic GGUF model file into per-layer shard files for DaisiChain
/// partial downloads. Each shard is a small binary index + contiguous tensor data.
///
/// Output files:
///   {name}.gguf.header       — GGUF header + metadata + tensor info (no tensor data)
///   {name}.gguf.embed        — token_embd.weight
///   {name}.gguf.output       — output_norm.weight + output.weight (if present)
///   {name}.gguf.layer.{N}    — all blk.{N}.* tensors
///   {name}.gguf.manifest.json — JSON index of all shard files
/// </summary>
public static class GgufSplitter
{
    /// <summary>
    /// Split a GGUF model file into per-layer shard files.
    /// </summary>
    /// <param name="inputPath">Path to the monolithic GGUF model file.</param>
    /// <param name="outputDir">Directory to write shard files to. Created if it doesn't exist.</param>
    /// <param name="log">Optional log callback for progress reporting.</param>
    /// <returns>The manifest describing all generated shard files.</returns>
    /// <param name="gpuAligned">If true, repack Q4_0/Q8_0 to GPU-aligned layout in layer shards.
    /// Enables zero-copy loading: mmap → pinned → async DMA with no per-token repack.</param>
    public static GgufShardManifest Split(string inputPath, string outputDir,
        Action<string>? log = null, bool gpuAligned = false)
    {
        Directory.CreateDirectory(outputDir);

        var baseName = Path.GetFileName(inputPath);

        // Parse GGUF header and metadata (does not read tensor data)
        log?.Invoke($"Parsing GGUF header: {inputPath}");
        using var inputStream = File.OpenRead(inputPath);
        var gguf = GgufFile.Read(inputStream);

        // Read architecture and layer count directly from GGUF metadata
        // (avoids requiring full ModelConfig which needs tokenizer metadata)
        var architecture = gguf.GetMetadataString("general.architecture") ?? "llama";
        var numLayers = gguf.GetMetadata<uint>($"{architecture}.block_count");
        if (numLayers == 0)
            throw new InvalidDataException($"Could not determine layer count from GGUF metadata key '{architecture}.block_count'");

        log?.Invoke($"Model: {architecture}, {numLayers} layers, {gguf.Tensors.Count} tensors");

        // Build tensor lookup by name
        var tensorMap = new Dictionary<string, GgufTensorInfo>(gguf.Tensors.Count);
        foreach (var t in gguf.Tensors)
            tensorMap[t.Name] = t;

        var manifest = new GgufShardManifest
        {
            ModelFileName = baseName,
            TotalLayers = (int)numLayers,
        };

        // 1. Header shard: verbatim copy of [0, TensorDataOffset)
        var headerFileName = $"{baseName}.header";
        var headerPath = Path.Combine(outputDir, headerFileName);
        log?.Invoke($"Writing header shard: {headerFileName}");
        WriteHeaderShard(inputStream, gguf.TensorDataOffset, headerPath);
        manifest.Header = new ShardFileInfo
        {
            FileName = headerFileName,
            SizeBytes = new FileInfo(headerPath).Length,
        };

        // 2. Embed shard: token_embd.weight
        var embedFileName = $"{baseName}.embed";
        var embedPath = Path.Combine(outputDir, embedFileName);
        log?.Invoke($"Writing embed shard: {embedFileName}");
        var embedTensors = CollectTensors(gguf, tensorMap, ["token_embd.weight"]);
        WriteShardFile(embedPath, GgufShardFormat.ShardType.Embed, -1, embedTensors, inputStream);
        manifest.Embed = new ShardFileInfo
        {
            FileName = embedFileName,
            SizeBytes = new FileInfo(embedPath).Length,
        };

        // 3. Output shard: output_norm.weight + output.weight (if present)
        var outputFileName = $"{baseName}.output";
        var outputPath = Path.Combine(outputDir, outputFileName);
        log?.Invoke($"Writing output shard: {outputFileName}");
        var outputNames = new List<string> { "output_norm.weight" };
        if (tensorMap.ContainsKey("output.weight"))
            outputNames.Add("output.weight");
        var outputTensors = CollectTensors(gguf, tensorMap, outputNames);
        WriteShardFile(outputPath, GgufShardFormat.ShardType.Output, -1, outputTensors, inputStream);
        manifest.Output = new ShardFileInfo
        {
            FileName = outputFileName,
            SizeBytes = new FileInfo(outputPath).Length,
        };

        // 4. Per-layer shards
        for (int i = 0; i < (int)numLayers; i++)
        {
            var layerFileName = $"{baseName}.layer.{i}";
            var layerPath = Path.Combine(outputDir, layerFileName);
            if (i % 8 == 0 || i == (int)numLayers - 1)
                log?.Invoke($"Writing layer shards: {i + 1}/{(int)numLayers}");

            // Find all tensors for this layer: blk.{i}.*
            var prefix = $"blk.{i}.";
            var layerTensorNames = gguf.Tensors
                .Where(t => t.Name.StartsWith(prefix))
                .Select(t => t.Name)
                .OrderBy(n => n)
                .ToList();

            var layerTensors = CollectTensors(gguf, tensorMap, layerTensorNames);
            if (gpuAligned)
                WriteShardFileGpuAligned(layerPath, GgufShardFormat.ShardType.Layer, i, layerTensors, tensorMap, inputStream);
            else
                WriteShardFile(layerPath, GgufShardFormat.ShardType.Layer, i, layerTensors, inputStream);

            manifest.Layers.Add(new LayerShardInfo
            {
                LayerIndex = i,
                FileName = layerFileName,
                SizeBytes = new FileInfo(layerPath).Length,
            });
        }

        manifest.GpuAligned = gpuAligned;

        // 5. Write manifest
        var manifestPath = Path.Combine(outputDir, $"{baseName}.manifest.json");
        log?.Invoke($"Writing manifest: {baseName}.manifest.json");
        File.WriteAllText(manifestPath, manifest.ToJson());

        // Summary
        long totalShardSize = manifest.Header.SizeBytes + manifest.Embed.SizeBytes +
            manifest.Output.SizeBytes + manifest.Layers.Sum(l => l.SizeBytes);
        long originalSize = new FileInfo(inputPath).Length;
        log?.Invoke($"Split complete: {(int)numLayers + 3} shard files, " +
            $"{totalShardSize / (1024.0 * 1024.0):F1} MB total (original: {originalSize / (1024.0 * 1024.0):F1} MB)");

        return manifest;
    }

    /// <summary>
    /// Compute the list of shard file names a pipeline stage needs.
    /// </summary>
    public static List<string> GetShardFilesForStage(
        GgufShardManifest manifest, int startLayer, int endLayer,
        bool includeEmbedding, bool includeOutputHead)
    {
        var files = new List<string> { manifest.Header.FileName };

        if (includeEmbedding)
            files.Add(manifest.Embed.FileName);

        if (includeOutputHead)
            files.Add(manifest.Output.FileName);

        for (int i = startLayer; i < endLayer; i++)
        {
            var layerShard = manifest.Layers.FirstOrDefault(l => l.LayerIndex == i);
            if (layerShard != null)
                files.Add(layerShard.FileName);
        }

        return files;
    }

    private static void WriteHeaderShard(Stream source, long tensorDataOffset, string outputPath)
    {
        source.Seek(0, SeekOrigin.Begin);
        using var output = File.Create(outputPath);
        var buffer = new byte[1024 * 1024]; // 1MB copy buffer
        long remaining = tensorDataOffset;
        while (remaining > 0)
        {
            int toRead = (int)Math.Min(remaining, buffer.Length);
            int bytesRead = source.Read(buffer, 0, toRead);
            if (bytesRead == 0) throw new EndOfStreamException("Unexpected end of GGUF file during header copy.");
            output.Write(buffer, 0, bytesRead);
            remaining -= bytesRead;
        }
    }

    private static List<(string name, long sourceOffset, long size)> CollectTensors(
        GgufFile gguf, Dictionary<string, GgufTensorInfo> tensorMap, IReadOnlyList<string> names)
    {
        var result = new List<(string, long, long)>(names.Count);
        foreach (var name in names)
        {
            if (!tensorMap.TryGetValue(name, out var info))
                throw new InvalidDataException($"Missing tensor in GGUF: {name}");

            long absoluteOffset = gguf.GetTensorDataOffset(info);
            result.Add((name, absoluteOffset, (long)info.ByteSize));
        }
        return result;
    }

    private static void WriteShardFile(string path, GgufShardFormat.ShardType type,
        int layerIndex, List<(string name, long sourceOffset, long size)> tensors, Stream source)
    {
        using var output = File.Create(path);
        GgufShardFormat.WriteShardFromSource(output, type, layerIndex, tensors, source);
    }

    /// <summary>
    /// Write a shard file with GPU-aligned quantized data.
    /// Q4_0: 18-byte blocks → 20-byte (2b scale + 2b pad + 16b nibbles).
    /// Q8_0: 34-byte blocks → 36-byte (2b scale + 2b pad + 32b quants).
    /// Other types: copied verbatim.
    /// The shard index records the REPACKED sizes so the reader sees correct byte counts.
    /// </summary>
    private static void WriteShardFileGpuAligned(string path, GgufShardFormat.ShardType type,
        int layerIndex, List<(string name, long sourceOffset, long size)> tensors,
        Dictionary<string, GgufTensorInfo> tensorMap, Stream source)
    {
        using var output = File.Create(path);

        // Compute repacked sizes for the shard index
        var repackedRefs = new List<(string name, long repackedSize)>(tensors.Count);
        foreach (var (name, _, size) in tensors)
        {
            var info = tensorMap[name];
            long repacked;
            if (info.Type == GgmlType.Q4_0 && info.Dimensions.Length >= 2)
                repacked = (size / 18) * 20;
            else if (info.Type == GgmlType.Q8_0 && info.Dimensions.Length >= 2)
                repacked = (size / 34) * 36;
            else
                repacked = size;
            repackedRefs.Add((name, repacked));
        }

        // Write shard header with repacked sizes
        using var writer = new BinaryWriter(output, System.Text.Encoding.UTF8, leaveOpen: true);
        writer.Write(GgufShardFormat.Magic);
        writer.Write(GgufShardFormat.FormatVersion);
        writer.Write((uint)type);
        writer.Write(layerIndex);
        writer.Write(repackedRefs.Count);

        long currentDataOffset = 0;
        foreach (var (name, repacked) in repackedRefs)
        {
            var nameBytes = System.Text.Encoding.UTF8.GetBytes(name);
            writer.Write(nameBytes.Length);
            writer.Write(nameBytes);
            writer.Write(currentDataOffset);
            writer.Write(repacked);
            currentDataOffset += repacked;
        }

        // Alignment padding
        long pos = output.Position;
        long remainder = pos % GgufShardFormat.Alignment;
        if (remainder != 0)
            writer.Write(new byte[(int)(GgufShardFormat.Alignment - remainder)]);

        // Write tensor data — repack Q4_0/Q8_0 on the fly
        var readBuf = new byte[1024 * 1024];
        foreach (var (name, sourceOffset, size) in tensors)
        {
            var info = tensorMap[name];
            source.Seek(sourceOffset, SeekOrigin.Begin);

            if (info.Type == GgmlType.Q4_0 && info.Dimensions.Length >= 2)
            {
                // Repack Q4_0: 18 → 20 byte blocks
                int blockCount = (int)(size / 18);
                var srcBlock = new byte[18];
                var dstBlock = new byte[20]; // [scale 2b][pad 2b][nibbles 16b]
                for (int b = 0; b < blockCount; b++)
                {
                    source.ReadExactly(srcBlock);
                    dstBlock[0] = srcBlock[0]; dstBlock[1] = srcBlock[1];
                    dstBlock[2] = 0; dstBlock[3] = 0;
                    Buffer.BlockCopy(srcBlock, 2, dstBlock, 4, 16);
                    output.Write(dstBlock);
                }
            }
            else if (info.Type == GgmlType.Q8_0 && info.Dimensions.Length >= 2)
            {
                // Repack Q8_0: 34 → 36 byte blocks
                int blockCount = (int)(size / 34);
                var srcBlock = new byte[34];
                var dstBlock = new byte[36]; // [scale 2b][pad 2b][quants 32b]
                for (int b = 0; b < blockCount; b++)
                {
                    source.ReadExactly(srcBlock);
                    dstBlock[0] = srcBlock[0]; dstBlock[1] = srcBlock[1];
                    dstBlock[2] = 0; dstBlock[3] = 0;
                    Buffer.BlockCopy(srcBlock, 2, dstBlock, 4, 32);
                    output.Write(dstBlock);
                }
            }
            else
            {
                // Verbatim copy
                long rem = size;
                while (rem > 0)
                {
                    int toRead = (int)Math.Min(rem, readBuf.Length);
                    int read = source.Read(readBuf, 0, toRead);
                    if (read == 0) throw new EndOfStreamException();
                    output.Write(readBuf, 0, read);
                    rem -= read;
                }
            }
        }
    }
}
