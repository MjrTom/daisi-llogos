using Daisi.Llogos.Cpu;
using Daisi.Llogos.Gguf;
using Daisi.Llogos.Model;

namespace Daisi.Llogos.Tests.Inference;

public class InspectDeltaNetTests
{
    [Fact]
    public void Inspect_SsmA_Values()
    {
        var models = new[]
        {
            (@"C:\GGUFS\Qwen3.5-0.8B-Q8_0.gguf", "0.8B"),
            (@"C:\GGUFS\Qwen3.5-9B-Q8_0.gguf", "9B"),
        };

        var lines = new List<string>();

        foreach (var (path, label) in models)
        {
            if (!File.Exists(path)) continue;

            using var stream = File.OpenRead(path);
            var gguf = GgufFile.Read(stream);
            var config = ModelConfig.FromGguf(gguf);
            using var backend = new CpuBackend();
            var weights = ModelLoader.Load(gguf, stream, backend, config);

            lines.Add($"=== {label} ===");
            lines.Add($"SsmGroupCount={config.SsmGroupCount}, SsmInnerSize={config.SsmInnerSize}, SsmStateSize={config.SsmStateSize}, SsmHeadDim={config.SsmHeadDim}");

            // Dump first DeltaNet layer's SsmA values
            for (int i = 0; i < config.NumLayers; i++)
            {
                if (!config.IsStandardAttention(i) && weights.Layers[i] is DeltaNetWeights dw)
                {
                    var ssmA = new float[dw.SsmA.ElementCount];
                    dw.SsmA.DequantizeTo(ssmA);
                    lines.Add($"Layer {i} SsmA ({ssmA.Length} values): [{string.Join(", ", ssmA.Select(v => v.ToString("F6")))}]");

                    var dtBias = new float[dw.SsmDtBias.ElementCount];
                    dw.SsmDtBias.DequantizeTo(dtBias);
                    lines.Add($"Layer {i} DtBias ({dtBias.Length} values): [{string.Join(", ", dtBias.Select(v => v.ToString("F6")))}]");

                    lines.Add($"Layer {i} SsmAlpha dims: {string.Join("x", Enumerable.Range(0, dw.SsmAlpha.Dimensions.Length).Select(d => dw.SsmAlpha.Dimensions[d]))}");
                    lines.Add($"Layer {i} AttnQkv dims: {string.Join("x", Enumerable.Range(0, dw.AttnQkv.Dimensions.Length).Select(d => dw.AttnQkv.Dimensions[d]))}");
                    lines.Add($"Layer {i} SsmConv1d dims: {string.Join("x", Enumerable.Range(0, dw.SsmConv1d.Dimensions.Length).Select(d => dw.SsmConv1d.Dimensions[d]))}");

                    // Check: if sA stores -exp(A_log), values should be negative
                    // If sA stores A_log directly, values could be any sign
                    bool allNegative = ssmA.All(v => v < 0);
                    bool looksPreComputed = ssmA.All(v => v < 0 && v > -10);
                    lines.Add($"  All negative: {allNegative}, Looks pre-computed (-exp): {looksPreComputed}");

                    // Compute what decay looks like with both interpretations
                    float testAlpha = 0.0f;
                    float testDt = dtBias[0];
                    float softplus = MathF.Log(1.0f + MathF.Exp(testAlpha + testDt));
                    float decayCurrent = MathF.Exp(ssmA[0] * softplus);
                    float decayIfALog = MathF.Exp(-MathF.Exp(ssmA[0]) * softplus);
                    lines.Add($"  Test decay (current formula): {decayCurrent:F6}");
                    lines.Add($"  Test decay (if A_log): {decayIfALog:F6}");
                    lines.Add($"  Softplus value: {softplus:F6}");

                    break;
                }
            }
            lines.Add("");
            weights.Dispose();
        }

        File.WriteAllLines(@"C:\GGUFS\deltanet-inspect.txt", lines);

        // Output to test console
        foreach (var line in lines)
            Console.WriteLine(line);
    }
}
