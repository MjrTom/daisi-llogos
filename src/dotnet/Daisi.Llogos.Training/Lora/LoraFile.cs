using System.Text;

namespace Daisi.Llogos.Training.Lora;

/// <summary>
/// Serialization for LoRA adapter weights.
/// Binary format: header + per-layer A and B matrices.
/// </summary>
public static class LoraFile
{
    private static readonly byte[] Magic = "LLRA"u8.ToArray();
    private const uint Version = 1;

    /// <summary>
    /// Save a LoRA adapter to a binary file.
    /// </summary>
    public static void Save(string path, LoraAdapter adapter)
    {
        using var stream = File.Create(path);
        using var writer = new BinaryWriter(stream);

        // Header
        writer.Write(Magic);
        writer.Write(Version);
        writer.Write(adapter.Config.Rank);
        writer.Write(adapter.Config.Alpha);
        writer.Write((int)adapter.Config.Targets);
        writer.Write(adapter.Layers.Count);

        // Per-layer weights
        foreach (var (name, layer) in adapter.Layers)
        {
            var nameBytes = Encoding.UTF8.GetBytes(name);
            writer.Write(nameBytes.Length);
            writer.Write(nameBytes);
            writer.Write(layer.InFeatures);
            writer.Write(layer.OutFeatures);
            writer.Write(layer.Rank);

            // A matrix: [rank × inFeatures]
            WriteFloatArray(writer, layer.A.Data);

            // B matrix: [outFeatures × rank]
            WriteFloatArray(writer, layer.B.Data);
        }
    }

    /// <summary>
    /// Load a LoRA adapter from a binary file.
    /// </summary>
    public static LoraAdapter Load(string path)
    {
        using var stream = File.OpenRead(path);
        using var reader = new BinaryReader(stream);

        // Header
        var magic = reader.ReadBytes(4);
        if (!magic.AsSpan().SequenceEqual(Magic))
            throw new InvalidDataException("Invalid LoRA file: bad magic");

        uint version = reader.ReadUInt32();
        if (version != Version)
            throw new InvalidDataException($"Unsupported LoRA file version: {version}");

        int rank = reader.ReadInt32();
        float alpha = reader.ReadSingle();
        var targets = (LoraTarget)reader.ReadInt32();
        int layerCount = reader.ReadInt32();

        var config = new LoraConfig { Rank = rank, Alpha = alpha, Targets = targets };
        float scaling = config.Scaling;
        var layers = new Dictionary<string, LoraLayer>();

        for (int i = 0; i < layerCount; i++)
        {
            int nameLen = reader.ReadInt32();
            string name = Encoding.UTF8.GetString(reader.ReadBytes(nameLen));
            int inFeatures = reader.ReadInt32();
            int outFeatures = reader.ReadInt32();
            int layerRank = reader.ReadInt32();

            float[] aData = ReadFloatArray(reader, layerRank * inFeatures);
            float[] bData = ReadFloatArray(reader, outFeatures * layerRank);

            layers[name] = new LoraLayer(name, inFeatures, outFeatures, layerRank, scaling, aData, bData);
        }

        return new LoraAdapter(config, layers);
    }

    private static void WriteFloatArray(BinaryWriter writer, float[] data)
    {
        var bytes = new byte[data.Length * sizeof(float)];
        Buffer.BlockCopy(data, 0, bytes, 0, bytes.Length);
        writer.Write(bytes);
    }

    private static float[] ReadFloatArray(BinaryReader reader, int count)
    {
        var bytes = reader.ReadBytes(count * sizeof(float));
        var data = new float[count];
        Buffer.BlockCopy(bytes, 0, data, 0, bytes.Length);
        return data;
    }
}
