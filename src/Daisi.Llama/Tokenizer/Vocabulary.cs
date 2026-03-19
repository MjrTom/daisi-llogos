namespace Daisi.Llama.Tokenizer;

/// <summary>
/// Token ID ↔ string mapping with special token support.
/// Built from GGUF metadata arrays.
/// </summary>
public sealed class Vocabulary
{
    private readonly string[] _tokens;
    private readonly Dictionary<string, int> _tokenToId;

    public Vocabulary(string[] tokens, int bosTokenId, int eosTokenId, int padTokenId)
    {
        _tokens = tokens;
        BosTokenId = bosTokenId;
        EosTokenId = eosTokenId;
        PadTokenId = padTokenId;

        _tokenToId = new Dictionary<string, int>(tokens.Length);
        for (int i = 0; i < tokens.Length; i++)
        {
            _tokenToId.TryAdd(tokens[i], i);
        }
    }

    /// <summary>Total number of tokens in the vocabulary.</summary>
    public int Count => _tokens.Length;

    /// <summary>Beginning of sequence token ID.</summary>
    public int BosTokenId { get; }

    /// <summary>End of sequence token ID.</summary>
    public int EosTokenId { get; }

    /// <summary>Padding token ID.</summary>
    public int PadTokenId { get; }

    /// <summary>Get the string representation of a token by its ID.</summary>
    public string IdToToken(int id) => _tokens[id];

    /// <summary>Get the token ID for a string, or -1 if not found.</summary>
    public int TokenToId(string token) => _tokenToId.GetValueOrDefault(token, -1);

    /// <summary>Check if a token string exists in the vocabulary.</summary>
    public bool Contains(string token) => _tokenToId.ContainsKey(token);
}
