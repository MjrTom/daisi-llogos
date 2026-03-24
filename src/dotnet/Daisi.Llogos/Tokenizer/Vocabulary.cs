namespace Daisi.Llogos.Tokenizer;

/// <summary>
/// Token ID ↔ string mapping with special token support.
/// Built from GGUF metadata arrays.
/// </summary>
public sealed class Vocabulary
{
    private readonly string[] _tokens;
    private readonly Dictionary<string, int> _tokenToId;
    private VocabRemapper? _remapper;

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

    /// <summary>
    /// Apply a vocabulary remapper. After this, all ID↔token operations use remapped IDs.
    /// The remapper must have been used to permute the embedding/output weight tensors.
    /// </summary>
    public void ApplyRemapper(VocabRemapper remapper)
    {
        _remapper = remapper;

        // Remap special token IDs
        BosTokenId = remapper.RemapEncode(BosTokenId);
        EosTokenId = remapper.RemapEncode(EosTokenId);
        PadTokenId = remapper.RemapEncode(PadTokenId);

        // Rebuild the token→id dictionary with remapped IDs
        _tokenToId.Clear();
        for (int oldId = 0; oldId < _tokens.Length; oldId++)
        {
            int newId = remapper.RemapEncode(oldId);
            _tokenToId.TryAdd(_tokens[oldId], newId);
        }
    }

    /// <summary>Total number of tokens in the vocabulary.</summary>
    public int Count => _tokens.Length;

    /// <summary>Beginning of sequence token ID.</summary>
    public int BosTokenId { get; private set; }

    /// <summary>End of sequence token ID.</summary>
    public int EosTokenId { get; private set; }

    /// <summary>Padding token ID.</summary>
    public int PadTokenId { get; private set; }

    /// <summary>Get the string representation of a token by its (possibly remapped) ID.</summary>
    public string IdToToken(int id) => _tokens[_remapper != null ? _remapper.RemapDecode(id) : id];

    /// <summary>Get the token ID for a string, or -1 if not found.</summary>
    public int TokenToId(string token) => _tokenToId.GetValueOrDefault(token, -1);

    /// <summary>Check if a token string exists in the vocabulary.</summary>
    public bool Contains(string token) => _tokenToId.ContainsKey(token);
}
