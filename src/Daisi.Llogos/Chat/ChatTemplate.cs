using Daisi.Llogos.Gguf;

namespace Daisi.Llogos.Chat;

/// <summary>
/// Known chat template formats that can be pattern-matched from GGUF metadata.
/// </summary>
public enum ChatTemplateFormat
{
    /// <summary>ChatML: &lt;|im_start|&gt;role\ncontent&lt;|im_end|&gt;\n (Qwen, Yi, etc.)</summary>
    ChatML,

    /// <summary>Llama 3: &lt;|start_header_id|&gt;role&lt;|end_header_id|&gt;\n\ncontent&lt;|eot_id|&gt;</summary>
    Llama3,

    /// <summary>Gemma: &lt;start_of_turn&gt;role\ncontent&lt;end_of_turn&gt;\n</summary>
    Gemma,

    /// <summary>Phi-3: &lt;|user|&gt;\ncontent&lt;|end|&gt;\n</summary>
    Phi3,

    /// <summary>Fallback using configurable tokens.</summary>
    Generic,
}

/// <summary>
/// Detects and holds chat template information extracted from GGUF metadata.
/// </summary>
public sealed class ChatTemplate
{
    public ChatTemplateFormat Format { get; }
    public string? RawTemplate { get; }

    private ChatTemplate(ChatTemplateFormat format, string? rawTemplate)
    {
        Format = format;
        RawTemplate = rawTemplate;
    }

    /// <summary>
    /// Extract the chat template format from GGUF metadata.
    /// Reads tokenizer.chat_template and pattern-matches known formats.
    /// </summary>
    public static ChatTemplate FromGguf(GgufFile gguf)
    {
        var raw = gguf.GetMetadataString("tokenizer.chat_template");

        if (string.IsNullOrEmpty(raw))
            return new ChatTemplate(ChatTemplateFormat.Generic, raw);

        var format = DetectFormat(raw);
        return new ChatTemplate(format, raw);
    }

    private static ChatTemplateFormat DetectFormat(string template)
    {
        if (template.Contains("<|im_start|>"))
            return ChatTemplateFormat.ChatML;
        if (template.Contains("<|start_header_id|>"))
            return ChatTemplateFormat.Llama3;
        if (template.Contains("<start_of_turn>"))
            return ChatTemplateFormat.Gemma;
        if (template.Contains("<|user|>") && template.Contains("<|end|>"))
            return ChatTemplateFormat.Phi3;

        return ChatTemplateFormat.Generic;
    }

    /// <summary>
    /// Get the stop sequences for this template format.
    /// These are the strings that signal end-of-turn during generation.
    /// </summary>
    public string[] GetStopSequences() => Format switch
    {
        ChatTemplateFormat.ChatML => ["<|im_end|>"],
        ChatTemplateFormat.Llama3 => ["<|eot_id|>"],
        ChatTemplateFormat.Gemma => ["<end_of_turn>"],
        ChatTemplateFormat.Phi3 => ["<|end|>"],
        ChatTemplateFormat.Generic => ["</s>"],
        _ => ["</s>"],
    };
}
