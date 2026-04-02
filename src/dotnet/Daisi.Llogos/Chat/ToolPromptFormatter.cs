using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace Daisi.Llogos.Chat;

/// <summary>
/// Formats tool definitions for injection into the system prompt.
/// Adapts the preamble and wrapper text per model family so the tool block
/// matches what the model was trained on.
/// </summary>
public static class ToolPromptFormatter
{
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        WriteIndented = false,
    };

    /// <summary>
    /// Format a list of tool definitions into a tools block for the system prompt.
    /// Uses a generic format suitable for most models.
    /// </summary>
    public static string FormatToolsBlock(IReadOnlyList<ToolDefinition> tools)
        => FormatToolsBlock(tools, ChatTemplateFormat.Generic);

    /// <summary>
    /// Format a list of tool definitions using the model-native preamble for the given format.
    /// </summary>
    public static string FormatToolsBlock(IReadOnlyList<ToolDefinition> tools, ChatTemplateFormat format)
    {
        if (tools.Count == 0)
            return "";

        var toolLines = SerializeToolLines(tools);

        // Qwen (ChatML) uses the exact preamble from the tokenizer.chat_template Jinja2 template.
        // Other models use a generic instruction that works broadly.
        return format switch
        {
            ChatTemplateFormat.ChatML => FormatQwen(toolLines),
            _ => FormatGeneric(toolLines),
        };
    }

    private static string FormatQwen(string toolLines)
    {
        var sb = new StringBuilder();
        sb.AppendLine();
        sb.AppendLine("# Tools");
        sb.AppendLine();
        sb.AppendLine("You may call one or more functions to assist with the user query.");
        sb.AppendLine();
        sb.AppendLine("You are provided with function signatures within <tools></tools> XML tags:");
        sb.AppendLine("<tools>");
        sb.Append(toolLines);
        sb.AppendLine("</tools>");
        sb.AppendLine();
        sb.AppendLine("For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:");
        sb.AppendLine("<tool_call>");
        sb.AppendLine("{\"name\": <function-name>, \"arguments\": <args-json-object>}");
        sb.Append("</tool_call>");
        return sb.ToString();
    }

    private static string FormatGeneric(string toolLines)
    {
        var sb = new StringBuilder();
        sb.AppendLine();
        sb.AppendLine("# Tools");
        sb.AppendLine();
        sb.AppendLine("You are provided with the following tools. To call a tool, respond with a <tool_call> block:");
        sb.AppendLine();
        sb.AppendLine("<tool_call>");
        sb.AppendLine("{\"name\": \"tool_name\", \"arguments\": {\"arg1\": \"value1\"}}");
        sb.AppendLine("</tool_call>");
        sb.AppendLine();
        sb.AppendLine("<tools>");
        sb.Append(toolLines);
        sb.AppendLine("</tools>");
        return sb.ToString();
    }

    /// <summary>Serialize tool definitions as one JSON object per line (GGUF chat_template style).</summary>
    private static string SerializeToolLines(IReadOnlyList<ToolDefinition> tools)
    {
        var sb = new StringBuilder();
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
            sb.AppendLine(toolObj.ToJsonString(JsonOptions));
        }
        return sb.ToString();
    }

    /// <summary>
    /// Get the stop sequences needed for tool call detection.
    /// </summary>
    public static string[] GetToolStopSequences() => ["</tool_call>"];

    /// <summary>
    /// Build a complete system prompt by appending the tools block to the base system prompt.
    /// </summary>
    public static string BuildSystemPrompt(string basePrompt, IReadOnlyList<ToolDefinition> tools)
        => BuildSystemPrompt(basePrompt, tools, ChatTemplateFormat.Generic);

    /// <summary>
    /// Build a complete system prompt with model-native tool formatting.
    /// </summary>
    public static string BuildSystemPrompt(string basePrompt, IReadOnlyList<ToolDefinition> tools, ChatTemplateFormat format)
    {
        if (tools.Count == 0)
            return basePrompt;

        return basePrompt + FormatToolsBlock(tools, format);
    }
}
