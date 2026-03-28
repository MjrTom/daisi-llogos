using System.Runtime.InteropServices;

namespace Daisi.Llogos.Tests;

/// <summary>
/// Shared constants for test configuration.
/// </summary>
public static class TestConstants
{
    /// <summary>
    /// Path to the Qwen 3.5 0.8B Q8_0 GGUF file used for integration tests.
    /// Downloaded from https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF
    /// Windows: C:\GGUFS\Qwen3.5-0.8B-Q8_0.gguf
    /// macOS/Linux: ~/GGUFS/Qwen3.5-0.8B-Q8_0.gguf
    /// </summary>
    public static readonly string Qwen35_08B_Q8_0 = RuntimeInformation.IsOSPlatform(OSPlatform.Windows)
        ? @"C:\GGUFS\Qwen3.5-0.8B-Q8_0.gguf"
        : Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "GGUFS", "Qwen3.5-0.8B-Q8_0.gguf");

    /// <summary>
    /// Path to the Qwen 3.5 9B Q8_0 GGUF file used for CRM integration tests.
    /// This is the same model running on the host.
    /// </summary>
    public static readonly string Qwen35_9B_Q8_0 = RuntimeInformation.IsOSPlatform(OSPlatform.Windows)
        ? @"C:\GGUFS\custom\Qwen3.5-9B-Q8_0.gguf"
        : Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "GGUFS", "custom", "Qwen3.5-9B-Q8_0.gguf");

    /// <summary>
    /// Returns true if the test model file exists on disk.
    /// </summary>
    public static bool ModelExists => File.Exists(Qwen35_08B_Q8_0);

    /// <summary>
    /// Returns true if the 9B model file exists on disk.
    /// </summary>
    public static bool Model9BExists => File.Exists(Qwen35_9B_Q8_0);
}
