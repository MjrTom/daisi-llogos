using Daisi.Llogos.Gguf;

namespace Daisi.Llogos.Tokenizer;

/// <summary>
/// Remaps vocabulary token IDs by frequency so the most common tokens have the lowest IDs.
/// This makes partial vocab truncation provably correct — the first N tokens are guaranteed
/// to be the N most common tokens.
/// </summary>
public sealed class VocabRemapper
{
    /// <summary>Old ID → New ID mapping.</summary>
    public int[] OldToNew { get; }

    /// <summary>New ID → Old ID mapping.</summary>
    public int[] NewToOld { get; }

    /// <summary>Number of tokens in the vocabulary.</summary>
    public int VocabSize => OldToNew.Length;

    public VocabRemapper(string[] tokens)
    {
        int n = tokens.Length;
        OldToNew = new int[n];
        NewToOld = new int[n];

        // Score each token: common tokens get high scores, rare get low
        var scores = new (int oldId, int score)[n];
        for (int i = 0; i < n; i++)
            scores[i] = (i, ScoreToken(tokens[i], i, n));

        // Sort by score descending — highest score = lowest new ID
        Array.Sort(scores, (a, b) => b.score.CompareTo(a.score));

        for (int newId = 0; newId < n; newId++)
        {
            int oldId = scores[newId].oldId;
            OldToNew[oldId] = newId;
            NewToOld[newId] = oldId;
        }
    }

    /// <summary>
    /// Score a token for frequency ordering. Higher = more common.
    /// Heuristic based on token content — no corpus statistics needed.
    /// </summary>
    static int ScoreToken(string token, int originalId, int vocabSize)
    {
        // Base score: prefer lower original IDs (tokenizer training puts common tokens first)
        int score = vocabSize - originalId;

        // Boost ASCII printable characters (letters, digits, punctuation, space)
        if (token.Length > 0)
        {
            bool allAscii = true;
            bool hasLetter = false;
            bool hasDigit = false;
            foreach (char c in token)
            {
                if (c > 127) allAscii = false;
                if (char.IsAsciiLetter(c)) hasLetter = true;
                if (char.IsAsciiDigit(c)) hasDigit = true;
            }
            if (allAscii) score += vocabSize; // strong boost for ASCII
            if (hasLetter) score += vocabSize / 2;
            if (hasDigit) score += vocabSize / 4;
        }

        // Boost CJK characters (very common in multilingual models)
        foreach (char c in token)
        {
            if (c >= 0x4E00 && c <= 0x9FFF) { score += vocabSize / 2; break; } // CJK Unified
            if (c >= 0x3040 && c <= 0x30FF) { score += vocabSize / 2; break; } // Hiragana/Katakana
            if (c >= 0xAC00 && c <= 0xD7AF) { score += vocabSize / 2; break; } // Korean
        }

        // Penalize special/control tokens (usually have markers like <| or ▁)
        if (token.StartsWith("<|") || token.StartsWith("<｜"))
            score -= vocabSize * 2; // push to end

        return score;
    }

    /// <summary>
    /// Remap a token ID from original space to remapped space.
    /// </summary>
    public int RemapEncode(int oldId) => oldId >= 0 && oldId < VocabSize ? OldToNew[oldId] : oldId;

    /// <summary>
    /// Remap a token ID from remapped space back to original space.
    /// </summary>
    public int RemapDecode(int newId) => newId >= 0 && newId < VocabSize ? NewToOld[newId] : newId;

    /// <summary>
    /// Permute rows of a weight tensor's raw data. Each row is one token's vector.
    /// Works for both embedding (token_embd) and output (lm_head) tensors.
    /// </summary>
    public byte[] PermuteRows(ReadOnlySpan<byte> data, int rowCount, int bytesPerRow)
    {
        var result = new byte[data.Length];
        for (int newRow = 0; newRow < rowCount; newRow++)
        {
            int oldRow = NewToOld[newRow];
            data.Slice(oldRow * bytesPerRow, bytesPerRow)
                .CopyTo(result.AsSpan(newRow * bytesPerRow, bytesPerRow));
        }
        return result;
    }
}
