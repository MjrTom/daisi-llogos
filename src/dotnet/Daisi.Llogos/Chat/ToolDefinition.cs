using System.Text.Json.Nodes;

namespace Daisi.Llogos.Chat;

/// <summary>
/// Defines a tool that the model can call during generation.
/// </summary>
public sealed class ToolDefinition
{
    public string Name { get; set; } = "";
    public string Description { get; set; } = "";
    public JsonObject ParametersSchema { get; set; } = [];

    public ToolDefinition() { }

    public ToolDefinition(string name, string description, JsonObject parametersSchema)
    {
        Name = name;
        Description = description;
        ParametersSchema = parametersSchema;
    }
}

/// <summary>
/// A tool call parsed from model output.
/// </summary>
public sealed class ToolCall
{
    public string Name { get; set; } = "";
    public JsonObject Arguments { get; set; } = [];

    public ToolCall() { }

    public ToolCall(string name, JsonObject arguments)
    {
        Name = name;
        Arguments = arguments;
    }
}
