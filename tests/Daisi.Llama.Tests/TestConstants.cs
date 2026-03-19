namespace Daisi.Llama.Tests;

/// <summary>
/// Shared constants for test configuration.
/// </summary>
public static class TestConstants
{
    /// <summary>
    /// Path to the Qwen 3.5 0.8B Q8_0 GGUF file used for integration tests.
    /// Downloaded from https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF
    /// </summary>
    public const string Qwen35_08B_Q8_0 = @"C:\GGUFS\Qwen3.5-0.8B-Q8_0.gguf";

    /// <summary>
    /// Returns true if the test model file exists on disk.
    /// </summary>
    public static bool ModelExists => File.Exists(Qwen35_08B_Q8_0);
}
