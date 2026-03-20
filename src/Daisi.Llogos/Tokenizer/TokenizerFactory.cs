using Daisi.Llogos.Gguf;

namespace Daisi.Llogos.Tokenizer;

/// <summary>
/// Builds a <see cref="BpeTokenizer"/> from GGUF file metadata.
/// Reads vocabulary tokens, merge rules, and special token IDs.
/// </summary>
public static class TokenizerFactory
{
    /// <summary>
    /// Create a BPE tokenizer from GGUF metadata.
    /// </summary>
    public static BpeTokenizer FromGguf(GgufFile gguf)
    {
        // Read vocabulary tokens
        var tokens = gguf.GetMetadata<string[]>("tokenizer.ggml.tokens")
            ?? throw new InvalidDataException("Missing tokenizer.ggml.tokens metadata.");

        // Read merge rules
        var merges = gguf.GetMetadata<string[]>("tokenizer.ggml.merges")
            ?? throw new InvalidDataException("Missing tokenizer.ggml.merges metadata.");

        // Read special token IDs (-1 means not present)
        int bosTokenId = GetInt32OrDefault(gguf, "tokenizer.ggml.bos_token_id", -1);
        int eosTokenId = GetInt32OrDefault(gguf, "tokenizer.ggml.eos_token_id", -1);
        int padTokenId = GetInt32OrDefault(gguf, "tokenizer.ggml.padding_token_id", -1);

        var vocab = new Vocabulary(tokens, bosTokenId, eosTokenId, padTokenId);
        var mergeTable = new MergeTable(merges);

        // Detect encoding mode from the tokenizer model type.
        // GPT-2 models use byte-to-unicode mapping; others (e.g. SentencePiece) use direct UTF-8.
        bool useByteEncoding = false;
        var tokenizerModel = gguf.GetMetadataString("tokenizer.ggml.model");
        if (tokenizerModel == "gpt2")
            useByteEncoding = true;

        return new BpeTokenizer(vocab, mergeTable, useByteEncoding);
    }

    private static int GetInt32OrDefault(GgufFile gguf, string key, int defaultValue)
    {
        var kv = gguf.Metadata.FirstOrDefault(m => m.Key == key);
        if (kv == null) return defaultValue;
        return (int)kv.ValueAs<uint>();
    }
}
