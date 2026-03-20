namespace Daisi.Llama.Chat;

/// <summary>
/// Renders a list of chat messages into a prompt string using the detected chat template format.
/// </summary>
public sealed class ChatTemplateRenderer
{
    private readonly ChatTemplate _template;

    public ChatTemplateRenderer(ChatTemplate template)
    {
        _template = template;
    }

    /// <summary>
    /// Render messages into a prompt string.
    /// </summary>
    /// <param name="messages">The conversation messages.</param>
    /// <param name="addGenerationPrompt">If true, append the assistant turn prefix so the model continues generating.</param>
    public string Render(IReadOnlyList<ChatMessage> messages, bool addGenerationPrompt = true)
    {
        return _template.Format switch
        {
            ChatTemplateFormat.ChatML => RenderChatML(messages, addGenerationPrompt),
            ChatTemplateFormat.Llama3 => RenderLlama3(messages, addGenerationPrompt),
            ChatTemplateFormat.Gemma => RenderGemma(messages, addGenerationPrompt),
            ChatTemplateFormat.Phi3 => RenderPhi3(messages, addGenerationPrompt),
            _ => RenderChatML(messages, addGenerationPrompt), // default to ChatML
        };
    }

    // <|im_start|>role\ncontent<|im_end|>\n
    private static string RenderChatML(IReadOnlyList<ChatMessage> messages, bool addGenPrompt)
    {
        var sb = new System.Text.StringBuilder();
        foreach (var msg in messages)
        {
            sb.Append("<|im_start|>");
            sb.Append(msg.Role);
            sb.Append('\n');
            sb.Append(msg.Content);
            sb.Append("<|im_end|>\n");
        }
        if (addGenPrompt)
            sb.Append("<|im_start|>assistant\n");
        return sb.ToString();
    }

    // <|start_header_id|>role<|end_header_id|>\n\ncontent<|eot_id|>
    private static string RenderLlama3(IReadOnlyList<ChatMessage> messages, bool addGenPrompt)
    {
        var sb = new System.Text.StringBuilder();
        sb.Append("<|begin_of_text|>");
        foreach (var msg in messages)
        {
            sb.Append("<|start_header_id|>");
            sb.Append(msg.Role);
            sb.Append("<|end_header_id|>\n\n");
            sb.Append(msg.Content);
            sb.Append("<|eot_id|>");
        }
        if (addGenPrompt)
            sb.Append("<|start_header_id|>assistant<|end_header_id|>\n\n");
        return sb.ToString();
    }

    // <start_of_turn>role\ncontent<end_of_turn>\n
    private static string RenderGemma(IReadOnlyList<ChatMessage> messages, bool addGenPrompt)
    {
        var sb = new System.Text.StringBuilder();
        foreach (var msg in messages)
        {
            var role = msg.Role == "assistant" ? "model" : msg.Role;
            sb.Append("<start_of_turn>");
            sb.Append(role);
            sb.Append('\n');
            sb.Append(msg.Content);
            sb.Append("<end_of_turn>\n");
        }
        if (addGenPrompt)
            sb.Append("<start_of_turn>model\n");
        return sb.ToString();
    }

    // <|user|>\ncontent<|end|>\n<|assistant|>\ncontent<|end|>\n
    private static string RenderPhi3(IReadOnlyList<ChatMessage> messages, bool addGenPrompt)
    {
        var sb = new System.Text.StringBuilder();
        foreach (var msg in messages)
        {
            sb.Append("<|");
            sb.Append(msg.Role);
            sb.Append("|>\n");
            sb.Append(msg.Content);
            sb.Append("<|end|>\n");
        }
        if (addGenPrompt)
            sb.Append("<|assistant|>\n");
        return sb.ToString();
    }
}

/// <summary>
/// A chat message with a role and content string.
/// Used internally by daisi-llama's chat infrastructure.
/// </summary>
public sealed class ChatMessage
{
    /// <summary>The role: "system", "user", "assistant", or "tool".</summary>
    public string Role { get; set; } = "";

    /// <summary>The message content.</summary>
    public string Content { get; set; } = "";

    public ChatMessage() { }

    public ChatMessage(string role, string content)
    {
        Role = role;
        Content = content;
    }
}
