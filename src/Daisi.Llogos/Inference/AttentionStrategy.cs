namespace Daisi.Llogos.Inference;

/// <summary>
/// Controls how the KV cache manages attention context.
/// Full: unlimited (up to max context). Window: fixed sliding window. Sinks: sliding window with
/// protected initial tokens (attention sinks) that are never evicted.
/// </summary>
public sealed class AttentionStrategy
{
    public static readonly AttentionStrategy Full = new(AttentionMode.Full, 0, 0);

    public AttentionMode Mode { get; }

    /// <summary>Number of initial tokens to retain permanently (attention sinks).</summary>
    public int SinkTokens { get; }

    /// <summary>Size of the sliding window (ring buffer region).</summary>
    public int WindowSize { get; }

    private AttentionStrategy(AttentionMode mode, int sinkTokens, int windowSize)
    {
        Mode = mode;
        SinkTokens = sinkTokens;
        WindowSize = windowSize;
    }

    /// <summary>Fixed sliding window with no sink tokens.</summary>
    public static AttentionStrategy Window(int size)
    {
        ArgumentOutOfRangeException.ThrowIfLessThanOrEqual(size, 0, nameof(size));
        return new AttentionStrategy(AttentionMode.Window, 0, size);
    }

    /// <summary>Sliding window with protected initial tokens (StreamingLLM attention sinks).</summary>
    public static AttentionStrategy Sinks(int sinkTokens, int windowSize)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(sinkTokens, 0, nameof(sinkTokens));
        ArgumentOutOfRangeException.ThrowIfLessThanOrEqual(windowSize, 0, nameof(windowSize));
        return new AttentionStrategy(AttentionMode.Sinks, sinkTokens, windowSize);
    }

    /// <summary>Total number of cache slots needed for this strategy.</summary>
    public int CacheCapacity => Mode == AttentionMode.Full ? 0 : SinkTokens + WindowSize;

    /// <summary>Map a logical token position to a physical cache slot index.</summary>
    public int MapPosition(int position)
    {
        if (Mode == AttentionMode.Full)
            return position;

        int capacity = SinkTokens + WindowSize;

        // Still filling linearly
        if (position < capacity)
            return position;

        // Sink region: positions 0..SinkTokens-1 are never overwritten
        // (they were written when position < capacity, so no mapping needed on write)

        // Ring buffer: overwrite the window region
        return SinkTokens + ((position - SinkTokens) % WindowSize);
    }

    /// <summary>Compute the effective sequence length visible to attention.</summary>
    public int EffectiveSeqLen(int position)
    {
        if (Mode == AttentionMode.Full)
            return position + 1;

        return Math.Min(position + 1, SinkTokens + WindowSize);
    }

    /// <summary>Parse from CLI string: "full", "window:N", "sinks:S,W".</summary>
    public static AttentionStrategy Parse(string value)
    {
        if (string.Equals(value, "full", StringComparison.OrdinalIgnoreCase))
            return Full;

        if (value.StartsWith("window:", StringComparison.OrdinalIgnoreCase))
        {
            int size = int.Parse(value.AsSpan(7));
            return Window(size);
        }

        if (value.StartsWith("sinks:", StringComparison.OrdinalIgnoreCase))
        {
            var parts = value.AsSpan(6);
            int comma = parts.IndexOf(',');
            if (comma < 0)
                throw new FormatException("sinks format: sinks:<sink_tokens>,<window_size>");
            int sink = int.Parse(parts[..comma]);
            int window = int.Parse(parts[(comma + 1)..]);
            return Sinks(sink, window);
        }

        throw new FormatException($"Unknown attention strategy: {value}. Use full, window:<N>, or sinks:<S>,<W>");
    }
}

public enum AttentionMode
{
    /// <summary>Full attention over entire context (default).</summary>
    Full,

    /// <summary>Fixed sliding window — oldest tokens are evicted.</summary>
    Window,

    /// <summary>Sliding window with attention sinks (initial tokens never evicted).</summary>
    Sinks,
}
