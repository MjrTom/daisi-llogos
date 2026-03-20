using Daisi.Llogos.Gguf;
using Daisi.Llogos.Model;

namespace Daisi.Llogos.Tests.Model;

public class InspectModelTests
{
    [Fact]
    public void InspectTinyLlama()
    {
        var path = @"C:\GGUFS\tinyllama-1.1b-chat-v1.0.Q8_0.gguf";
        if (!File.Exists(path)) return;

        using var stream = File.OpenRead(path);
        var gguf = GgufFile.Read(stream);

        var lines = new List<string>();
        lines.Add("=== METADATA ===");
        foreach (var kv in gguf.Metadata.OrderBy(x => x.Key))
            lines.Add($"  {kv.Key} = {kv.Value}");

        lines.Add($"\n=== TENSORS ({gguf.Tensors.Count}) ===");
        foreach (var t in gguf.Tensors.Take(30))
            lines.Add($"  {t.Name} [{t.Type}] dims={string.Join("x", t.Dimensions)}");

        File.WriteAllLines(@"C:\GGUFS\tinyllama-inspect.txt", lines);

        var config = ModelConfig.FromGguf(gguf);
        Assert.Equal("llama", config.Architecture);
        Assert.True(config.NumLayers > 0);
    }
}
