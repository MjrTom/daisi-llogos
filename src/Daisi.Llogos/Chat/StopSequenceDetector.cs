namespace Daisi.Llogos.Chat;

/// <summary>
/// Detects multi-token stop sequences in streaming text output.
/// Buffers output to handle partial matches that span token boundaries.
/// </summary>
public sealed class StopSequenceDetector
{
    private readonly string[] _stopSequences;
    private readonly System.Text.StringBuilder _buffer = new();
    private readonly int _maxStopLen;

    public StopSequenceDetector(string[] stopSequences)
    {
        _stopSequences = stopSequences;
        _maxStopLen = 0;
        foreach (var s in stopSequences)
            if (s.Length > _maxStopLen)
                _maxStopLen = s.Length;
    }

    /// <summary>
    /// Process a new token text. Returns the text that can be safely emitted
    /// (not part of a potential stop sequence), and whether generation should stop.
    /// </summary>
    /// <param name="tokenText">The decoded text of the new token.</param>
    /// <param name="emittableText">Text that can be safely output to the user.</param>
    /// <returns>True if a stop sequence was found and generation should stop.</returns>
    public bool Process(string tokenText, out string emittableText)
    {
        _buffer.Append(tokenText);
        var bufStr = _buffer.ToString();

        // Check for complete stop sequence match
        foreach (var stop in _stopSequences)
        {
            int idx = bufStr.IndexOf(stop, StringComparison.Ordinal);
            if (idx >= 0)
            {
                // Found a stop sequence — emit everything before it
                emittableText = bufStr[..idx];
                _buffer.Clear();
                return true;
            }
        }

        // Check for partial match at the end of the buffer
        // (a stop sequence might be starting but not yet complete)
        int safeLen = bufStr.Length;
        foreach (var stop in _stopSequences)
        {
            for (int len = 1; len < stop.Length && len <= bufStr.Length; len++)
            {
                var bufferTail = bufStr.AsSpan(bufStr.Length - len);
                var stopHead = stop.AsSpan(0, len);
                if (bufferTail.SequenceEqual(stopHead))
                {
                    int candidate = bufStr.Length - len;
                    if (candidate < safeLen)
                        safeLen = candidate;
                }
            }
        }

        // Emit the safe prefix and keep the rest buffered
        emittableText = bufStr[..safeLen];
        _buffer.Clear();
        if (safeLen < bufStr.Length)
            _buffer.Append(bufStr[safeLen..]);

        return false;
    }

    /// <summary>
    /// Flush any remaining buffered text. Call when generation ends without a stop sequence.
    /// </summary>
    public string Flush()
    {
        var remaining = _buffer.ToString();
        _buffer.Clear();
        return remaining;
    }

    /// <summary>Reset the detector state for a new generation.</summary>
    public void Reset()
    {
        _buffer.Clear();
    }
}
