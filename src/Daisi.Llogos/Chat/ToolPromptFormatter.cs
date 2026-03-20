using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace Daisi.Llogos.Chat;

/// <summary>
/// Formats tool definitions for injection into the system prompt.
/// Uses Qwen's expected format with &lt;tools&gt; block.
/// </summary>
public static class ToolPromptFormatter
{
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        WriteIndented = false,
    };

    /// <summary>
    /// Format a list of tool definitions into a tools block for the system prompt.
    /// </summary>
    public static string FormatToolsBlock(IReadOnlyList<ToolDefinition> tools)
    {
        if (tools.Count == 0)
            return "";

        var sb = new StringBuilder();
        sb.AppendLine();
        sb.AppendLine("# Tools");
        sb.AppendLine();
        sb.AppendLine("You are provided with the following tools. To call a tool, respond with a <tool_call> block containing a JSON object with the tool name and arguments:");
        sb.AppendLine();
        sb.AppendLine("<tool_call>");
        sb.AppendLine("{\"name\": \"tool_name\", \"arguments\": {\"arg1\": \"value1\"}}");
        sb.AppendLine("</tool_call>");
        sb.AppendLine();
        sb.AppendLine("You may call multiple tools. Each tool call must be in its own <tool_call> block.");
        sb.AppendLine();
        sb.AppendLine("<tools>");

        var toolArray = new JsonArray();
        foreach (var tool in tools)
        {
            var funcObj = new JsonObject
            {
                ["name"] = tool.Name,
                ["description"] = tool.Description,
                ["parameters"] = JsonNode.Parse(tool.ParametersSchema.ToJsonString(JsonOptions)),
            };
            var toolObj = new JsonObject
            {
                ["type"] = "function",
                ["function"] = funcObj,
            };
            toolArray.Add(toolObj);
        }

        sb.AppendLine(toolArray.ToJsonString(JsonOptions));
        sb.AppendLine("</tools>");

        return sb.ToString();
    }

    /// <summary>
    /// Get the stop sequences needed for tool call detection.
    /// These should be added to the regular stop sequences during generation.
    /// </summary>
    public static string[] GetToolStopSequences() => ["</tool_call>"];

    /// <summary>
    /// Build a complete system prompt by appending the tools block to the base system prompt.
    /// </summary>
    public static string BuildSystemPrompt(string basePrompt, IReadOnlyList<ToolDefinition> tools)
    {
        if (tools.Count == 0)
            return basePrompt;

        return basePrompt + FormatToolsBlock(tools);
    }
}
