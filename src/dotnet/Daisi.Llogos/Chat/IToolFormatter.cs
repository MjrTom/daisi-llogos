using System.Text.Json.Nodes;

namespace Daisi.Llogos.Chat;

/// <summary>
/// Formats tool definitions for the system prompt and parses tool calls from model output.
/// Consumers can implement this to customize tool interaction patterns for different models.
/// </summary>
public interface IToolFormatter
{
    /// <summary>
    /// Format tool definitions into a block that will be appended to the system prompt.
    /// </summary>
    string FormatToolsBlock(IReadOnlyList<ToolDefinition> tools);

    /// <summary>
    /// Check if the model output contains tool calls.
    /// </summary>
    bool ContainsToolCalls(string text);

    /// <summary>
    /// Parse tool calls from model output text.
    /// </summary>
    List<ToolCall> ParseToolCalls(string text);

    /// <summary>
    /// Format a tool result into a ChatMessage for injection back into the conversation.
    /// Different models expect different formatting (e.g. tool role vs user role with XML wrapper).
    /// </summary>
    ChatMessage FormatToolResult(string toolName, string result);

    /// <summary>
    /// Get additional stop sequences needed for tool call detection.
    /// </summary>
    string[] GetToolStopSequences();
}
