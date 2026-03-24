namespace Daisi.Llogos.Cpu;

/// <summary>
/// Shared thread count configuration for CPU compute operations.
/// Limits Parallel.For degree of parallelism to avoid saturating all cores.
/// </summary>
public static class CpuThreading
{
    private static int _threadCount = Math.Max(1, Environment.ProcessorCount - 2);
    private static ParallelOptions _options = new() { MaxDegreeOfParallelism = _threadCount };

    /// <summary>
    /// Maximum threads used for parallel matmul. Defaults to (logical cores - 2), minimum 1.
    /// </summary>
    public static int ThreadCount
    {
        get => _threadCount;
        set
        {
            _threadCount = Math.Max(1, value);
            _options = new ParallelOptions { MaxDegreeOfParallelism = _threadCount };
        }
    }

    /// <summary>
    /// Shared ParallelOptions instance with the configured thread limit.
    /// </summary>
    public static ParallelOptions Options => _options;
}
