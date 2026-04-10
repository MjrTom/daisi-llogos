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
    private readonly bool _useSentencePiece;
    private readonly Regex _preTokenizePattern;

    /// <summary>SentencePiece "lower one eighth block" — used as the in-vocab space marker.</summary>
    private const char SpmSpaceMarker = '\u2581';

    // GPT-2 byte-to-token mapping (only used when _useByteEncoding is true)
    private static readonly string[] ByteToUnicodeTable = BuildByteToUnicode();
    private static readonly Dictionary<char, byte> UnicodeToByte = BuildUnicodeToByte();

    // Special tokens that should be matched as whole strings (e.g. <|im_start|>, <|endoftext|>)
    private readonly List<(string token, int id)> _specialTokens = new();

    public BpeTokenizer(Vocabulary vocab, MergeTable merges, bool useByteEncoding = false, bool useSentencePiece = false)
    {
        _vocab = vocab;
        _merges = merges;
        _useByteEncoding = useByteEncoding;
        _useSentencePiece = useSentencePiece;
        _preTokenizePattern = PreTokenizeRegex();

        // Build special token list from vocabulary (tokens matching <|...|> or <...> patterns)
        for (int i = 0; i < vocab.Count; i++)
        {
            var token = vocab.IdToToken(i);
            if (token != null && token.StartsWith("<|") && token.EndsWith("|>") && token.Length > 4)
                _specialTokens.Add((token, i));
            // Also handle fullwidth pipe variants (Qwen uses ｜ in GGUF)
            else if (token != null && token.Contains('\uFF5C') && token.StartsWith("<"))
            {
                var normalized = token.Replace('\uFF5C', '|');
                _specialTokens.Add((normalized, i));
            }
        }
        // Sort by length descending so longer tokens match first
        _specialTokens.Sort((a, b) => b.token.Length.CompareTo(a.token.Length));
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

        // Split on special tokens first, then BPE-encode the segments between them
        var segments = SplitOnSpecialTokens(text);
        foreach (var (segment, specialId) in segments)
        {
            if (specialId >= 0)
            {
                result.Add(specialId);
                continue;
            }

            if (_useSentencePiece)
            {
                // SentencePiece (Gemma, Llama): replace ASCII spaces with U+2581 ('▁'),
                // skip the GPT-2 regex pre-tokenization (which would split on spaces and
                // prevent in-word merges), and BPE-encode the entire segment as one chunk.
                var spmText = SentencePiecePreprocess(segment);
                if (spmText.Length == 0) continue;
                var symbols = DirectEncodeChunk(spmText);
                if (symbols.Count == 0) continue;
                ApplyMerges(symbols);
                foreach (var symbol in symbols)
                {
                    int id = _vocab.TokenToId(symbol);
                    if (id >= 0) result.Add(id);
                }
                continue;
            }

            // Regular BPE encoding for non-special text
            var matches = _preTokenizePattern.Matches(segment);
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
        }

        return result.ToArray();
    }

    /// <summary>
    /// SentencePiece preprocessing: replace each ASCII space with U+2581 ('▁').
    /// This is the convention SPM-trained models (Gemma, Llama) use to represent
    /// word boundaries inside the vocabulary.
    /// </summary>
    private static string SentencePiecePreprocess(string text)
    {
        if (text.Length == 0) return text;
        var sb = new StringBuilder(text.Length);
        foreach (var ch in text)
            sb.Append(ch == ' ' ? SpmSpaceMarker : ch);
        return sb.ToString();
    }

    /// <summary>
    /// Split text into segments, identifying special tokens as whole units.
    /// Returns (text_segment, -1) for regular text, (token_text, token_id) for special tokens.
    /// </summary>
    private List<(string text, int id)> SplitOnSpecialTokens(string text)
    {
        if (_specialTokens.Count == 0)
            return [(text, -1)];

        var segments = new List<(string text, int id)>();
        int pos = 0;

        while (pos < text.Length)
        {
            // Try to match a special token at current position
            bool found = false;
            foreach (var (token, id) in _specialTokens)
            {
                if (pos + token.Length <= text.Length &&
                    text.AsSpan(pos, token.Length).SequenceEqual(token.AsSpan()))
                {
                    // Flush any preceding text
                    if (pos > 0 && segments.Count == 0 || (segments.Count > 0 && segments[^1].id < 0))
                    {
                        // Check if there's text before this match not yet added
                    }
                    // Simpler approach: collect start position
                    segments.Add((token, id));
                    pos += token.Length;
                    found = true;
                    break;
                }
            }
            if (!found)
            {
                // Accumulate regular text
                if (segments.Count > 0 && segments[^1].id < 0)
                {
                    segments[^1] = (segments[^1].text + text[pos], -1);
                }
                else
                {
                    segments.Add((text[pos].ToString(), -1));
                }
                pos++;
            }
        }

        return segments;
    }

    /// <summary>
    /// Decode a sequence of token IDs back to text.
    /// </summary>
    public string Decode(ReadOnlySpan<int> tokenIds)
    {
        var sb = new StringBuilder();
        foreach (var id in tokenIds)
        {
            if (id == _vocab.BosTokenId || id == _vocab.EosTokenId || id == _vocab.PadTokenId)
                continue;

            var token = _vocab.IdToToken(id);

            // Handle byte fallback tokens like <0x0A>
            if (token.Length == 6 && token.StartsWith("<0x") && token.EndsWith('>'))
            {
                if (byte.TryParse(token.AsSpan(3, 2), System.Globalization.NumberStyles.HexNumber, null, out var b))
                {
                    sb.Append((char)b);
                    continue;
                }
            }

            // Normalize fullwidth pipes (U+FF5C ｜) to ASCII pipes (U+007C |)
            // in special tokens. Qwen models store tokens like <｜im_end｜> with
            // fullwidth pipes, but stop sequences and prompts use ASCII pipes.
            if (token.Contains('\uFF5C'))
                token = token.Replace('\uFF5C', '|');

            sb.Append(token);
        }

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
