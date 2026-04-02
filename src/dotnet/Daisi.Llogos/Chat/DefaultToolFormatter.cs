namespace Daisi.Llogos.Chat;

/// <summary>
/// Default tool formatter using &lt;tool_call&gt; / &lt;tool_response&gt; XML tags.
/// Adapts tool preamble per model family via ChatTemplateFormat.
/// </summary>
public sealed class DefaultToolFormatter : IToolFormatter
{
    public static readonly DefaultToolFormatter Instance = new();

    private readonly ChatTemplateFormat _format;

    public DefaultToolFormatter() : this(ChatTemplateFormat.Generic) { }

    public DefaultToolFormatter(ChatTemplateFormat format) { _format = format; }

    public string FormatToolsBlock(IReadOnlyList<ToolDefinition> tools) =>
        ToolPromptFormatter.FormatToolsBlock(tools, _format);

    public bool ContainsToolCalls(string text) =>
        ToolCallParser.ContainsToolCalls(text);

    public List<ToolCall> ParseToolCalls(string text) =>
        ToolCallParser.Parse(text);

    public ChatMessage FormatToolResult(string toolName, string result) =>
        new("tool", result);

    public string[] GetToolStopSequences() =>
        ToolPromptFormatter.GetToolStopSequences();
}
