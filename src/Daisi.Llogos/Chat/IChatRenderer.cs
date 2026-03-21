namespace Daisi.Llogos.Chat;

/// <summary>
/// Renders a list of chat messages into a prompt string for the model.
/// Consumers can implement this to customize chat template rendering,
/// tool call formatting, and generation prompt behavior.
/// </summary>
public interface IChatRenderer
{
    /// <summary>
    /// Render messages into a prompt string.
    /// </summary>
    /// <param name="messages">The conversation messages (system, user, assistant, tool).</param>
    /// <param name="addGenerationPrompt">If true, append the assistant turn prefix so the model continues generating.</param>
    string Render(IReadOnlyList<ChatMessage> messages, bool addGenerationPrompt = true);

    /// <summary>
    /// Get the stop sequences for this renderer.
    /// These are the strings that signal end-of-turn during generation.
    /// </summary>
    string[] GetStopSequences();
}
