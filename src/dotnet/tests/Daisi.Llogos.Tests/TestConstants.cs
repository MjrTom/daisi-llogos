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

    /// <summary>
    /// Path to the Qwen 3.5 27B Q4_0 GGUF file used for large-model shard tests.
    /// </summary>
    public static readonly string Qwen35_27B_Q4_0 = RuntimeInformation.IsOSPlatform(OSPlatform.Windows)
        ? @"C:\GGUFS\Qwen3.5-27B-Q4_0.gguf"
        : Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "GGUFS", "Qwen3.5-27B-Q4_0.gguf");

    /// <summary>Pre-split shard directory for the 27B model.</summary>
    public static readonly string Qwen35_27B_Shards = Qwen35_27B_Q4_0 + ".shards";

    /// <summary>Returns true if the 27B model and its shards exist.</summary>
    public static bool Model27BExists => File.Exists(Qwen35_27B_Q4_0);
    public static bool Model27BShardsExist => Directory.Exists(Qwen35_27B_Shards);

    /// <summary>
    /// Path to the BitNet b1.58 2B I2_S GGUF file. Ternary quantization — fast on CPU.
    /// </summary>
    public static readonly string BitNet_I2S = RuntimeInformation.IsOSPlatform(OSPlatform.Windows)
        ? @"C:\GGUFS\ggml-model-i2_s.gguf"
        : Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "GGUFS", "ggml-model-i2_s.gguf");

    /// <summary>
    /// Returns true if the BitNet model file exists on disk.
    /// </summary>
    public static bool BitNetExists => File.Exists(BitNet_I2S);
}
