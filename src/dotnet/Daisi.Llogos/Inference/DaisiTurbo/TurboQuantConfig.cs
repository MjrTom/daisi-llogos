namespace Daisi.Llogos.Inference.DaisiTurbo;

/// <summary>
/// Configuration for DaisiTurbo KV cache compression.
/// </summary>
public sealed class TurboQuantConfig
{
    /// <summary>Bits per scalar for the MSE quantizer stage (2, 3, or 4).</summary>
    public int QuantBits { get; set; } = 3;

    /// <summary>
    /// QJL projection dimension per head. Controls the variance/memory tradeoff
    /// for the sign-bit residual correction. Set to 0 to disable QJL (biased but smaller).
    /// Default: headDim/2 (adds 0.5 bits per dimension on average).
    /// </summary>
    public int? QjlProjectionDim { get; set; }

    /// <summary>Random seed for WHT sign flips and QJL projection matrix.</summary>
    public int Seed { get; set; } = 42;

    /// <summary>
    /// Whether to apply TurboQuant to Keys, Values, or both.
    /// Default: both.
    /// </summary>
    public TurboQuantTarget Target { get; set; } = TurboQuantTarget.Both;

    /// <summary>
    /// Effective bits per dimension including QJL overhead.
    /// Example: 3-bit quant + 32 sign bits / 64 dims = 3.5 bits/dim.
    /// </summary>
    public float EffectiveBitsPerDim(int headDim)
    {
        int projDim = QjlProjectionDim ?? (headDim / 2);
        float qjlBitsPerDim = projDim > 0 ? (float)projDim / headDim : 0;
        return QuantBits + qjlBitsPerDim;
    }

    /// <summary>
    /// Parse from CLI string: "turbo", "turbo:3", "turbo:4", "turbo:3+qjl32", "turbo:3+noqjl".
    /// </summary>
    public static TurboQuantConfig Parse(string value)
    {
        var config = new TurboQuantConfig();

        // Strip "turbo" prefix
        var rest = value.AsSpan();
        if (rest.StartsWith("turbo", StringComparison.OrdinalIgnoreCase))
            rest = rest[5..];

        if (rest.Length == 0)
            return config;

        if (rest[0] != ':')
            throw new FormatException($"Invalid turbo config: {value}. Use turbo, turbo:3, turbo:4, turbo:3+qjl32, turbo:3+noqjl");

        rest = rest[1..];

        // Split on '+' for options
        int plusIdx = rest.IndexOf('+');
        ReadOnlySpan<char> bitsStr = plusIdx >= 0 ? rest[..plusIdx] : rest;
        ReadOnlySpan<char> optStr = plusIdx >= 0 ? rest[(plusIdx + 1)..] : default;

        if (bitsStr.Length > 0)
            config.QuantBits = int.Parse(bitsStr);

        if (optStr.Length > 0)
        {
            if (optStr.StartsWith("qjl", StringComparison.OrdinalIgnoreCase))
                config.QjlProjectionDim = int.Parse(optStr[3..]);
            else if (optStr.Equals("noqjl", StringComparison.OrdinalIgnoreCase))
                config.QjlProjectionDim = 0;
        }

        return config;
    }
}

public enum TurboQuantTarget
{
    Keys,
    Values,
    Both
}
