using Daisi.Llogos.Chat;

namespace Daisi.Llogos.Tests.Chat;

public class ToolCallParserTests
{
    [Fact]
    public void ParsesSingleToolCall()
    {
        var text = """
            Let me read that file for you.
            <tool_call>
            {"name": "file_read", "arguments": {"path": "/tmp/test.txt"}}
            </tool_call>
            """;

        var calls = ToolCallParser.Parse(text);

        Assert.Single(calls);
        Assert.Equal("file_read", calls[0].Name);
        Assert.Equal("/tmp/test.txt", calls[0].Arguments["path"]!.GetValue<string>());
    }

    [Fact]
    public void ParsesMultipleToolCalls()
    {
        var text = """
            <tool_call>
            {"name": "file_read", "arguments": {"path": "a.txt"}}
            </tool_call>
            <tool_call>
            {"name": "file_read", "arguments": {"path": "b.txt"}}
            </tool_call>
            """;

        var calls = ToolCallParser.Parse(text);

        Assert.Equal(2, calls.Count);
        Assert.Equal("a.txt", calls[0].Arguments["path"]!.GetValue<string>());
        Assert.Equal("b.txt", calls[1].Arguments["path"]!.GetValue<string>());
    }

    [Fact]
    public void ReturnsEmptyForNoToolCalls()
    {
        var calls = ToolCallParser.Parse("Just a regular response with no tools.");
        Assert.Empty(calls);
    }

    [Fact]
    public void HandlesMalformedJson()
    {
        var text = """
            <tool_call>
            {not valid json}
            </tool_call>
            """;

        var calls = ToolCallParser.Parse(text);
        Assert.Empty(calls);
    }

    [Fact]
    public void HandlesMissingName()
    {
        var text = """
            <tool_call>
            {"arguments": {"path": "test.txt"}}
            </tool_call>
            """;

        var calls = ToolCallParser.Parse(text);
        Assert.Empty(calls);
    }

    [Fact]
    public void ContainsToolCalls_True()
    {
        Assert.True(ToolCallParser.ContainsToolCalls("text <tool_call> something"));
    }

    [Fact]
    public void ContainsToolCalls_False()
    {
        Assert.False(ToolCallParser.ContainsToolCalls("no tools here"));
    }

    [Fact]
    public void GetTextBeforeToolCalls_ExtractsPrefix()
    {
        var text = "Here is some text\n<tool_call>\n{\"name\":\"test\",\"arguments\":{}}\n</tool_call>";
        var before = ToolCallParser.GetTextBeforeToolCalls(text);
        Assert.Equal("Here is some text", before);
    }

    [Fact]
    public void GetTextBeforeToolCalls_ReturnsAllIfNoToolCalls()
    {
        var text = "No tool calls here";
        Assert.Equal(text, ToolCallParser.GetTextBeforeToolCalls(text));
    }

    [Fact]
    public void ParsesToolCallWithNoArguments()
    {
        var text = """
            <tool_call>
            {"name": "get_time", "arguments": {}}
            </tool_call>
            """;

        var calls = ToolCallParser.Parse(text);
        Assert.Single(calls);
        Assert.Equal("get_time", calls[0].Name);
        Assert.Empty(calls[0].Arguments);
    }
}
