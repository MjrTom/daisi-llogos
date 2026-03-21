using System.Text;
using System.Text.RegularExpressions;

namespace Daisi.Llogos.Tokenizer;

/// <summary>
/// Byte Pair Encoding tokenizer. Encodes text to token IDs and decodes back.
/// Uses a vocabulary and merge table extracted from GGUF metadata.
/// Supports both GPT-2 byte encoding and direct UTF-8 token vocabularies (e.g. Qwen).
/// </summary>
public sealed partial class BpeTokenizer
{
    private readonly Vocabulary _vocab;
    private readonly MergeTable _merges;
    private readonly bool _useByteEncoding;
    private readonly Regex _preTokenizePattern;

    // GPT-2 byte-to-token mapping (only used when _useByteEncoding is true)
    private static readonly string[] ByteToUnicodeTable = BuildByteToUnicode();
    private static readonly Dictionary<char, byte> UnicodeToByte = BuildUnicodeToByte();

    public BpeTokenizer(Vocabulary vocab, MergeTable merges, bool useByteEncoding = false)
    {
        _vocab = vocab;
        _merges = merges;
        _useByteEncoding = useByteEncoding;
        _preTokenizePattern = PreTokenizeRegex();
    }

    /// <summary>The vocabulary used by this tokenizer.</summary>
    public Vocabulary Vocabulary => _vocab;

    /// <summary>
    /// Encode text into a sequence of token IDs.
    /// </summary>
    public int[] Encode(string text)
    {
        if (string.IsNullOrEmpty(text))
            return [];

        var result = new List<int>();
        var matches = _preTokenizePattern.Matches(text);

        foreach (Match match in matches)
        {
            var chunk = match.Value;
            var symbols = _useByteEncoding ? ByteEncodeChunk(chunk) : DirectEncodeChunk(chunk);
            if (symbols.Count == 0) continue;

            ApplyMerges(symbols);

            foreach (var symbol in symbols)
            {
                int id = _vocab.TokenToId(symbol);
                if (id >= 0)
                    result.Add(id);
            }
        }

        return result.ToArray();
    }

    /// <summary>
    /// Decode a sequence of token IDs back to text.
    /// </summary>
    public string Decode(ReadOnlySpan<int> tokenIds)
    {
        var sb = new StringBuilder();
        var byteBuffer = new List<byte>(); // accumulates <0xNN> byte fallback tokens

        foreach (var id in tokenIds)
        {
            if (id == _vocab.BosTokenId || id == _vocab.EosTokenId || id == _vocab.PadTokenId)
                continue;

            var token = _vocab.IdToToken(id);

            // Handle byte fallback tokens like <0x0A>
            // Accumulate into a buffer and decode as UTF-8 when the sequence ends,
            // so multi-byte characters (emoji, CJK, etc.) decode correctly.
            if (token.Length == 6 && token.StartsWith("<0x") && token.EndsWith('>'))
            {
                if (byte.TryParse(token.AsSpan(3, 2), System.Globalization.NumberStyles.HexNumber, null, out var b))
                {
                    byteBuffer.Add(b);
                    continue;
                }
            }

            // Flush any accumulated byte fallback tokens as UTF-8
            if (byteBuffer.Count > 0)
            {
                sb.Append(System.Text.Encoding.UTF8.GetString(byteBuffer.ToArray()));
                byteBuffer.Clear();
            }

            // Normalize fullwidth pipes (U+FF5C ｜) to ASCII pipes (U+007C |)
            // in special tokens. Qwen models store tokens like <｜im_end｜> with
            // fullwidth pipes, but stop sequences and prompts use ASCII pipes.
            if (token.Contains('\uFF5C'))
                token = token.Replace('\uFF5C', '|');

            sb.Append(token);
        }

        // Flush remaining byte fallback tokens
        if (byteBuffer.Count > 0)
            sb.Append(System.Text.Encoding.UTF8.GetString(byteBuffer.ToArray()));

        var text = sb.ToString();
        if (_useByteEncoding)
            return ByteDecodeString(text);
        // SentencePiece uses ▁ (U+2581) as space marker
        return text.Replace('\u2581', ' ');
    }

    // ── Direct encoding (Qwen, LLaMA 3, etc.) ──────────────────────────────

    /// <summary>
    /// For vocabularies that store tokens as direct UTF-8 strings.
    /// Each character becomes a symbol; unknown chars fall back to byte tokens.
    /// </summary>
    private List<string> DirectEncodeChunk(string chunk)
    {
        var symbols = new List<string>();
        var bytes = Encoding.UTF8.GetBytes(chunk);
        int i = 0;
        while (i < bytes.Length)
        {
            // Try to match the longest single character at this position
            int charLen = GetUtf8CharLength(bytes[i]);
            if (i + charLen <= bytes.Length)
            {
                var ch = Encoding.UTF8.GetString(bytes, i, charLen);
                if (_vocab.Contains(ch))
                {
                    symbols.Add(ch);
                    i += charLen;
                    continue;
                }
            }

            // Fall back to byte token <0xHH>
            symbols.Add($"<0x{bytes[i]:X2}>");
            i++;
        }
        return symbols;
    }

    private static int GetUtf8CharLength(byte firstByte)
    {
        if (firstByte < 0x80) return 1;
        if (firstByte < 0xC0) return 1; // continuation byte (shouldn't be first)
        if (firstByte < 0xE0) return 2;
        if (firstByte < 0xF0) return 3;
        return 4;
    }

    // ── GPT-2 byte encoding ─────────────────────────────────────────────────

    /// <summary>
    /// For GPT-2 style vocabularies that use unicode remapping for bytes.
    /// </summary>
    private static List<string> ByteEncodeChunk(string chunk)
    {
        var bytes = Encoding.UTF8.GetBytes(chunk);
        var symbols = new List<string>(bytes.Length);
        foreach (var b in bytes)
            symbols.Add(ByteToUnicodeTable[b]);
        return symbols;
    }

    private static string ByteDecodeString(string bpeText)
    {
        var bytes = new List<byte>(bpeText.Length);
        foreach (var ch in bpeText)
        {
            if (UnicodeToByte.TryGetValue(ch, out var b))
                bytes.Add(b);
            else
                bytes.AddRange(Encoding.UTF8.GetBytes(ch.ToString()));
        }
        return Encoding.UTF8.GetString(bytes.ToArray());
    }

    // ── BPE merge algorithm ─────────────────────────────────────────────────

    /// <summary>
    /// Apply BPE merges to a list of symbols in priority order.
    /// Iteratively finds and applies the highest-priority (lowest rank) merge.
    /// </summary>
    private void ApplyMerges(List<string> symbols)
    {
        while (symbols.Count > 1)
        {
            int bestRank = int.MaxValue;
            int bestIdx = -1;

            for (int i = 0; i < symbols.Count - 1; i++)
            {
                int rank = _merges.GetRank(symbols[i], symbols[i + 1]);
                if (rank >= 0 && rank < bestRank)
                {
                    bestRank = rank;
                    bestIdx = i;
                }
            }

            if (bestIdx < 0) break;

            symbols[bestIdx] = symbols[bestIdx] + symbols[bestIdx + 1];
            symbols.RemoveAt(bestIdx + 1);
        }
    }

    // ── GPT-2 byte table construction ───────────────────────────────────────

    private static string[] BuildByteToUnicode()
    {
        var table = new string[256];
        int n = 256;
        for (int i = 0; i < 256; i++)
        {
            if ((i >= 0x21 && i <= 0x7E) || (i >= 0xA1 && i <= 0xAC) || (i >= 0xAE && i <= 0xFF))
                table[i] = ((char)i).ToString();
            else
            {
                table[i] = ((char)n).ToString();
                n++;
            }
        }
        return table;
    }

    private static Dictionary<char, byte> BuildUnicodeToByte()
    {
        var table = BuildByteToUnicode();
        var reverse = new Dictionary<char, byte>(256);
        for (int i = 0; i < 256; i++)
            reverse[table[i][0]] = (byte)i;
        return reverse;
    }

    // GPT-2 / Qwen pre-tokenization regex
    [GeneratedRegex(@"'(?:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+", RegexOptions.Compiled)]
    private static partial Regex PreTokenizeRegex();
}
