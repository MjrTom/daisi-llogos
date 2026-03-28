using Daisi.Llogos.Chat;
using Daisi.Llogos.Gguf;
using Daisi.Llogos.Model;
using Daisi.Llogos.Tokenizer;

namespace Daisi.Llogos.Tests.Chat;

/// <summary>
/// Compare how the tokenizer handles ChatML special tokens.
/// Check if <|im_start|> is tokenized as one token or multiple.
/// </summary>
public class TokenizerCompareTest
{
    [Fact]
    public void CheckSpecialTokenTokenization()
    {
        if (!TestConstants.Model9BExists) return;
        using var stream = File.OpenRead(TestConstants.Qwen35_9B_Q8_0);
        var gguf = GgufFile.Read(stream);
        var tokenizer = TokenizerFactory.FromGguf(gguf);

        // Check how special tokens are encoded
        var tests = new[]
        {
            "<|im_start|>",
            "<|im_end|>",
            "<|im_start|>system\nHello<|im_end|>\n",
            "<|im_start|>assistant\n",
        };

        var sb = new System.Text.StringBuilder();
        foreach (var text in tests)
        {
            var ids = tokenizer.Encode(text);
            var decoded = string.Join(", ", ids.Select(id => $"{id}=\"{tokenizer.Decode([id])}\""));
            sb.AppendLine($"Input: \"{text.Replace("\n","\\n")}\"");
            sb.AppendLine($"Tokens ({ids.Length}): [{decoded}]");
            sb.AppendLine();
        }

        // Also check what the ChatTemplateRenderer produces
        var chatTemplate = ChatTemplate.FromGguf(gguf);
        var renderer = new ChatTemplateRenderer(chatTemplate);
        var messages = new List<ChatMessage>
        {
            new("system", "Return JSON."),
            new("user", "Hello"),
        };
        var rendered = renderer.Render(messages, addGenerationPrompt: true);
        var renderedIds = tokenizer.Encode(rendered);
        sb.AppendLine($"ChatML rendered: \"{rendered.Replace("\n","\\n")}\"");
        sb.AppendLine($"Tokens ({renderedIds.Length}): [{string.Join(", ", renderedIds.Take(20).Select(id => $"{id}"))}...]");

        File.WriteAllText(@"C:\repos\daisinet-qwen-integration-crm\test-tokenizer.txt", sb.ToString());
        Assert.Fail(sb.ToString());
    }
}
