using Daisi.Llama.Chat;

namespace Daisi.Llama.Tests.Chat;

public class ChatTemplateTests
{
    [Fact]
    public void RenderChatML_BasicConversation()
    {
        var template = new ChatTemplate_TestHelper(ChatTemplateFormat.ChatML);
        var renderer = new ChatTemplateRenderer(template.Template);

        var messages = new List<ChatMessage>
        {
            new("system", "You are a helpful assistant."),
            new("user", "Hello!"),
        };

        var result = renderer.Render(messages, addGenerationPrompt: true);

        Assert.Contains("<|im_start|>system\nYou are a helpful assistant.<|im_end|>", result);
        Assert.Contains("<|im_start|>user\nHello!<|im_end|>", result);
        Assert.EndsWith("<|im_start|>assistant\n", result);
    }

    [Fact]
    public void RenderChatML_NoGenerationPrompt()
    {
        var template = new ChatTemplate_TestHelper(ChatTemplateFormat.ChatML);
        var renderer = new ChatTemplateRenderer(template.Template);

        var messages = new List<ChatMessage>
        {
            new("user", "Hello!"),
            new("assistant", "Hi there!"),
        };

        var result = renderer.Render(messages, addGenerationPrompt: false);

        Assert.DoesNotContain("<|im_start|>assistant\n", result.Split("Hi there!")[1]);
        Assert.EndsWith("<|im_end|>\n", result);
    }

    [Fact]
    public void RenderChatML_ToolRole()
    {
        var template = new ChatTemplate_TestHelper(ChatTemplateFormat.ChatML);
        var renderer = new ChatTemplateRenderer(template.Template);

        var messages = new List<ChatMessage>
        {
            new("user", "What's the weather?"),
            new("assistant", "Let me check."),
            new("tool", "{\"temperature\": 72}"),
        };

        var result = renderer.Render(messages, addGenerationPrompt: true);

        Assert.Contains("<|im_start|>tool\n{\"temperature\": 72}<|im_end|>", result);
    }

    [Fact]
    public void RenderLlama3_BasicConversation()
    {
        var template = new ChatTemplate_TestHelper(ChatTemplateFormat.Llama3);
        var renderer = new ChatTemplateRenderer(template.Template);

        var messages = new List<ChatMessage>
        {
            new("system", "You are helpful."),
            new("user", "Hi"),
        };

        var result = renderer.Render(messages, addGenerationPrompt: true);

        Assert.StartsWith("<|begin_of_text|>", result);
        Assert.Contains("<|start_header_id|>system<|end_header_id|>\n\nYou are helpful.<|eot_id|>", result);
        Assert.Contains("<|start_header_id|>user<|end_header_id|>\n\nHi<|eot_id|>", result);
        Assert.EndsWith("<|start_header_id|>assistant<|end_header_id|>\n\n", result);
    }

    [Fact]
    public void RenderGemma_MapsAssistantToModel()
    {
        var template = new ChatTemplate_TestHelper(ChatTemplateFormat.Gemma);
        var renderer = new ChatTemplateRenderer(template.Template);

        var messages = new List<ChatMessage>
        {
            new("user", "Hi"),
            new("assistant", "Hello!"),
        };

        var result = renderer.Render(messages, addGenerationPrompt: true);

        Assert.Contains("<start_of_turn>user\nHi<end_of_turn>", result);
        Assert.Contains("<start_of_turn>model\nHello!<end_of_turn>", result);
        Assert.EndsWith("<start_of_turn>model\n", result);
    }

    [Fact]
    public void StopSequences_ChatML()
    {
        var template = new ChatTemplate_TestHelper(ChatTemplateFormat.ChatML);
        var stops = template.Template.GetStopSequences();
        Assert.Contains("<|im_end|>", stops);
    }

    [Fact]
    public void StopSequences_Llama3()
    {
        var template = new ChatTemplate_TestHelper(ChatTemplateFormat.Llama3);
        var stops = template.Template.GetStopSequences();
        Assert.Contains("<|eot_id|>", stops);
    }
}

/// <summary>
/// Helper to create ChatTemplate instances for testing without GGUF files.
/// Uses reflection since the constructor is private.
/// </summary>
internal class ChatTemplate_TestHelper
{
    public ChatTemplate Template { get; }

    public ChatTemplate_TestHelper(ChatTemplateFormat format)
    {
        // Use reflection to access private constructor
        var ctor = typeof(ChatTemplate).GetConstructor(
            System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance,
            [typeof(ChatTemplateFormat), typeof(string)]);
        Template = (ChatTemplate)ctor!.Invoke([format, null]);
    }
}
