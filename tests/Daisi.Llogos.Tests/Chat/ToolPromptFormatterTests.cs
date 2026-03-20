using System.Text.Json.Nodes;
using Daisi.Llogos.Chat;

namespace Daisi.Llogos.Tests.Chat;

public class ToolPromptFormatterTests
{
    [Fact]
    public void FormatsToolsBlock()
    {
        var tools = new List<ToolDefinition>
        {
            new("file_read", "Read a file from disk", new JsonObject
            {
                ["type"] = "object",
                ["properties"] = new JsonObject
                {
                    ["path"] = new JsonObject { ["type"] = "string", ["description"] = "File path" }
                },
                ["required"] = new JsonArray("path"),
            }),
        };

        var block = ToolPromptFormatter.FormatToolsBlock(tools);

        Assert.Contains("<tools>", block);
        Assert.Contains("</tools>", block);
        Assert.Contains("file_read", block);
        Assert.Contains("Read a file from disk", block);
        Assert.Contains("<tool_call>", block);
        Assert.Contains("</tool_call>", block);
    }

    [Fact]
    public void ReturnsEmptyForNoTools()
    {
        var block = ToolPromptFormatter.FormatToolsBlock([]);
        Assert.Equal("", block);
    }

    [Fact]
    public void BuildSystemPrompt_AppendsTools()
    {
        var tools = new List<ToolDefinition>
        {
            new("test", "A test tool", new JsonObject { ["type"] = "object" }),
        };

        var prompt = ToolPromptFormatter.BuildSystemPrompt("You are helpful.", tools);

        Assert.StartsWith("You are helpful.", prompt);
        Assert.Contains("<tools>", prompt);
    }

    [Fact]
    public void BuildSystemPrompt_NoTools_ReturnsBase()
    {
        var prompt = ToolPromptFormatter.BuildSystemPrompt("You are helpful.", []);
        Assert.Equal("You are helpful.", prompt);
    }

    [Fact]
    public void GetToolStopSequences_ReturnsExpected()
    {
        var stops = ToolPromptFormatter.GetToolStopSequences();
        Assert.Contains("</tool_call>", stops);
    }
}
