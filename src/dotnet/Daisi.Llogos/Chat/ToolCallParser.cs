using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.RegularExpressions;

namespace Daisi.Llogos.Chat;

/// <summary>
/// Extracts tool calls from model output text.
/// Parses &lt;tool_call&gt;...&lt;/tool_call&gt; blocks.
/// </summary>
public static partial class ToolCallParser
{
    /// <summary>
    /// Parse all tool calls from model output text.
    /// Returns an empty list if no tool calls are found.
    /// </summary>
    public static List<ToolCall> Parse(string text)
    {
        var results = new List<ToolCall>();
        var matches = ToolCallRegex().Matches(text);

        foreach (Match match in matches)
        {
            var json = match.Groups[1].Value.Trim();
            var toolCall = TryParseToolCall(json);
            if (toolCall != null)
                results.Add(toolCall);
        }

        return results;
    }

    /// <summary>
    /// Check if the text contains any tool call blocks.
    /// </summary>
    public static bool ContainsToolCalls(string text) =>
        text.Contains("<tool_call>", StringComparison.Ordinal);

    /// <summary>
    /// Extract the text content before the first tool call block.
    /// Returns the full text if no tool calls are present.
    /// </summary>
    public static string GetTextBeforeToolCalls(string text)
    {
        int idx = text.IndexOf("<tool_call>", StringComparison.Ordinal);
        return idx < 0 ? text : text[..idx].TrimEnd();
    }

    private static ToolCall? TryParseToolCall(string json)
    {
        try
        {
            var node = JsonNode.Parse(json);
            if (node is not JsonObject obj)
                return null;

            var name = obj["name"]?.GetValue<string>();
            if (string.IsNullOrEmpty(name))
                return null;

            var args = obj["arguments"] as JsonObject ?? [];

            return new ToolCall(name, args);
        }
        catch (JsonException)
        {
            return null;
        }
    }

    [GeneratedRegex(@"<tool_call>\s*(.*?)\s*</tool_call>", RegexOptions.Singleline)]
    private static partial Regex ToolCallRegex();
}
