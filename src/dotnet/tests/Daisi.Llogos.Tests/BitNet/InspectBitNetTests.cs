using Daisi.Llogos.Gguf;

namespace Daisi.Llogos.Tests.BitNet;

public class InspectBitNetTests
{
    private const string BitNetPath = @"C:\GGUFS\ggml-model-i2_s.gguf";

    [Fact]
    public void InspectBitNetMetadata()
    {
        if (!File.Exists(BitNetPath)) return;

        using var stream = File.OpenRead(BitNetPath);
        var gguf = GgufFile.Read(stream);

        var lines = new List<string>();
        lines.Add("=== METADATA ===");
        foreach (var kv in gguf.Metadata.OrderBy(x => x.Key))
            lines.Add($"  {kv.Key} = {kv.Value}");

        lines.Add($"\n=== TENSORS ({gguf.Tensors.Count}) ===");
        foreach (var t in gguf.Tensors)
            lines.Add($"  {t.Name} [{t.Type}] dims={string.Join("x", t.Dimensions)}");

        // Write to file so we can read it
        File.WriteAllLines(@"C:\GGUFS\bitnet-inspect.txt", lines);
        Assert.True(gguf.Tensors.Count > 0);
    }
}
