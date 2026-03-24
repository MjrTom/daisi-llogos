namespace Daisi.Llogos.Chat;

/// <summary>
/// Default tool formatter using Qwen-style &lt;tool_call&gt; / &lt;tool_response&gt; XML tags.
/// Uses the existing ToolPromptFormatter and ToolCallParser static implementations.
/// </summary>
public sealed class DefaultToolFormatter : IToolFormatter
{
    public static readonly DefaultToolFormatter Instance = new();

    public string FormatToolsBlock(IReadOnlyList<ToolDefinition> tools) =>
        ToolPromptFormatter.FormatToolsBlock(tools);

    public bool ContainsToolCalls(string text) =>
        ToolCallParser.ContainsToolCalls(text);

    public List<ToolCall> ParseToolCalls(string text) =>
        ToolCallParser.Parse(text);

    public ChatMessage FormatToolResult(string toolName, string result) =>
        new("tool", result);

    public string[] GetToolStopSequences() =>
        ToolPromptFormatter.GetToolStopSequences();
}
