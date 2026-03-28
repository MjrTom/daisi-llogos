using Daisi.Llogos.Chat;
using Daisi.Llogos.Gguf;
using Daisi.Llogos.Inference;
using Daisi.Llogos.Model;
using Daisi.Llogos.Tokenizer;
using System.Text;

namespace Daisi.Llogos.Tests.Chat;

/// <summary>
/// Compare token IDs between:
/// 1. ChatTemplateRenderer + Encode (what DaisiLlogosChatSession uses)
/// 2. Raw ChatML string + Encode (what the manual test uses)
/// 3. llama.cpp tokenization (reference)
/// If these differ, the model sees different prompts and produces different output.
/// </summary>
public class TokenIdComparisonTest
{
    [Fact]
    public void CompareTokenizations()
    {
        if (!TestConstants.Model9BExists) return;
        using var stream = File.OpenRead(TestConstants.Qwen35_9B_Q8_0);
        var gguf = GgufFile.Read(stream);
        var tokenizer = TokenizerFactory.FromGguf(gguf);
        var chatTemplate = ChatTemplate.FromGguf(gguf);
        var renderer = new ChatTemplateRenderer(chatTemplate);

        var system = "Return ONLY valid JSON: {\"a\":\"string\"}";
        var user = "Hello";

        // Method 1: ChatTemplateRenderer
        var messages = new List<ChatMessage>
        {
            new("system", system),
            new("user", user),
        };
        var rendered = renderer.Render(messages, addGenerationPrompt: true);
        var renderedIds = tokenizer.Encode(rendered);

        // Method 2: Manual ChatML string
        var manual = $"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n";
        var manualIds = tokenizer.Encode(manual);

        var sb = new StringBuilder();
        sb.AppendLine($"=== Rendered prompt ===");
        sb.AppendLine($"\"{rendered.Replace("\n","\\n")}\"");
        sb.AppendLine($"Tokens ({renderedIds.Length}): [{string.Join(", ", renderedIds)}]");
        sb.AppendLine();
        sb.AppendLine($"=== Manual prompt ===");
        sb.AppendLine($"\"{manual.Replace("\n","\\n")}\"");
        sb.AppendLine($"Tokens ({manualIds.Length}): [{string.Join(", ", manualIds)}]");
        sb.AppendLine();

        // Compare
        bool match = renderedIds.Length == manualIds.Length;
        if (match)
        {
            for (int i = 0; i < renderedIds.Length; i++)
            {
                if (renderedIds[i] != manualIds[i]) { match = false; break; }
            }
        }
        sb.AppendLine(match ? "MATCH!" : "DIFFERENT!");

        if (!match)
        {
            int minLen = Math.Min(renderedIds.Length, manualIds.Length);
            for (int i = 0; i < minLen; i++)
            {
                if (renderedIds[i] != manualIds[i])
                {
                    sb.AppendLine($"First diff at position {i}: rendered={renderedIds[i]}(\"{tokenizer.Decode([renderedIds[i]])}\")" +
                        $" manual={manualIds[i]}(\"{tokenizer.Decode([manualIds[i]])}\")");
                    break;
                }
            }
        }

        File.WriteAllText(@"C:\repos\daisinet-qwen-integration-crm\test-tokenid-compare.txt", sb.ToString());
        Assert.True(match, sb.ToString());
    }
}
