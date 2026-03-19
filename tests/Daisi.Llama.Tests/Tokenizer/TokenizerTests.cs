using Daisi.Llama.Tokenizer;

namespace Daisi.Llama.Tests.Tokenizer;

public class VocabularyTests
{
    [Fact]
    public void IdToToken_RoundTrip()
    {
        var vocab = new Vocabulary(["hello", "world", "<bos>"], 2, 2, 2);
        Assert.Equal("hello", vocab.IdToToken(0));
        Assert.Equal("world", vocab.IdToToken(1));
        Assert.Equal(3, vocab.Count);
    }

    [Fact]
    public void TokenToId_Found()
    {
        var vocab = new Vocabulary(["hello", "world"], 0, 0, 0);
        Assert.Equal(0, vocab.TokenToId("hello"));
        Assert.Equal(1, vocab.TokenToId("world"));
    }

    [Fact]
    public void TokenToId_NotFound()
    {
        var vocab = new Vocabulary(["hello"], 0, 0, 0);
        Assert.Equal(-1, vocab.TokenToId("missing"));
    }

    [Fact]
    public void SpecialTokenIds()
    {
        var vocab = new Vocabulary(["a", "b", "c"], bosTokenId: 10, eosTokenId: 11, padTokenId: 12);
        Assert.Equal(10, vocab.BosTokenId);
        Assert.Equal(11, vocab.EosTokenId);
        Assert.Equal(12, vocab.PadTokenId);
    }
}

public class MergeTableTests
{
    [Fact]
    public void GetRank_Found()
    {
        var merges = new MergeTable(["a b", "c d", "ab cd"]);
        Assert.Equal(0, merges.GetRank("a", "b"));
        Assert.Equal(1, merges.GetRank("c", "d"));
        Assert.Equal(2, merges.GetRank("ab", "cd"));
    }

    [Fact]
    public void GetRank_NotFound()
    {
        var merges = new MergeTable(["a b"]);
        Assert.Equal(-1, merges.GetRank("x", "y"));
    }

    [Fact]
    public void Priority_LowerRankFirst()
    {
        var merges = new MergeTable(["h e", "he l", "hel lo"]);
        Assert.True(merges.GetRank("h", "e") < merges.GetRank("he", "l"));
        Assert.True(merges.GetRank("he", "l") < merges.GetRank("hel", "lo"));
    }
}

public class BpeTokenizerTests
{
    [Fact]
    public void Encode_EmptyString()
    {
        var tokenizer = MakeDirectTokenizer(["a", "b"], []);
        var ids = tokenizer.Encode("");
        Assert.Empty(ids);
    }

    [Fact]
    public void Encode_SimpleWord_DirectMode()
    {
        // Direct mode: "h" and "i" are vocab entries, with merge "h" + "i" → "hi"
        var tokenizer = MakeDirectTokenizer(
            ["h", "i", "hi"],
            ["h i"]);

        var ids = tokenizer.Encode("hi");
        Assert.Single(ids);
        Assert.Equal(2, ids[0]); // "hi" merged token
    }

    [Fact]
    public void Decode_RoundTrip_Ascii()
    {
        // Build a direct-mode tokenizer with individual ASCII characters
        var tokens = new List<string>();
        for (int i = 0; i < 128; i++)
            tokens.Add(((char)i).ToString());

        var tokenizer = MakeDirectTokenizer(tokens.ToArray(), []);

        var text = "Hello";
        var ids = tokenizer.Encode(text);
        var decoded = tokenizer.Decode(ids);
        Assert.Equal(text, decoded);
    }

    [Fact]
    public void Decode_SkipsSpecialTokens()
    {
        var tokens = new string[130];
        for (int i = 0; i < 128; i++) tokens[i] = ((char)i).ToString();
        tokens[128] = "<bos>";
        tokens[129] = "<eos>";

        var vocab = new Vocabulary(tokens, 128, 129, 128);
        var tokenizer = new BpeTokenizer(vocab, new MergeTable([]), useByteEncoding: false);

        var decoded = tokenizer.Decode([128, (int)'H', (int)'i', 129]);
        Assert.Equal("Hi", decoded);
    }

    [Fact]
    public void Decode_ByteFallbackTokens()
    {
        // <0x0A> should decode to newline
        // Use special token IDs that don't conflict with the test data tokens
        var vocab = new Vocabulary(["<0x0A>", "H", "i"], bosTokenId: -1, eosTokenId: -1, padTokenId: -1);
        var tokenizer = new BpeTokenizer(vocab, new MergeTable([]), useByteEncoding: false);

        var decoded = tokenizer.Decode([1, 2, 0]); // "H" + "i" + newline
        Assert.Equal("Hi\n", decoded);
    }

    [Fact]
    public void MergesApplied_InPriorityOrder()
    {
        var tokenizer = MakeDirectTokenizer(
            ["a", "b", "c", "ab", "abc", "bc"],
            ["a b", "ab c"]); // merge a+b first, then ab+c

        var ids = tokenizer.Encode("abc");
        Assert.Single(ids);
        Assert.Equal(4, ids[0]); // "abc" token at index 4
    }

    [Fact]
    public void Encode_WithPunctuation()
    {
        // Ensure punctuation is handled in direct mode
        var tokenizer = MakeDirectTokenizer(
            ["H", "i", "!"],
            []);

        var ids = tokenizer.Encode("Hi!");
        Assert.Equal(3, ids.Length);
    }

    private static BpeTokenizer MakeDirectTokenizer(string[] tokens, string[] merges)
    {
        var vocab = new Vocabulary(tokens, 0, 0, 0);
        var mergeTable = new MergeTable(merges);
        return new BpeTokenizer(vocab, mergeTable, useByteEncoding: false);
    }
}

public class TokenizerIntegrationTests
{
    [Fact]
    public void FromQwen_BuildsTokenizer()
    {
        if (!TestConstants.ModelExists) return;

        using var stream = File.OpenRead(TestConstants.Qwen35_08B_Q8_0);
        var gguf = Daisi.Llama.Gguf.GgufFile.Read(stream);
        var tokenizer = TokenizerFactory.FromGguf(gguf);

        Assert.True(tokenizer.Vocabulary.Count > 100_000);
    }

    [Fact]
    public void FromQwen_EncodeDecodeRoundTrip()
    {
        if (!TestConstants.ModelExists) return;

        using var stream = File.OpenRead(TestConstants.Qwen35_08B_Q8_0);
        var gguf = Daisi.Llama.Gguf.GgufFile.Read(stream);
        var tokenizer = TokenizerFactory.FromGguf(gguf);

        var texts = new[]
        {
            "Hello, world!",
            "The quick brown fox jumps over the lazy dog.",
            "1 + 1 = 2",
            "C# is great for AI inference.",
        };

        foreach (var text in texts)
        {
            var ids = tokenizer.Encode(text);
            Assert.NotEmpty(ids);
            var decoded = tokenizer.Decode(ids);
            Assert.Equal(text, decoded);
        }
    }

    [Fact]
    public void FromQwen_EncodesSpecialCharacters()
    {
        if (!TestConstants.ModelExists) return;

        using var stream = File.OpenRead(TestConstants.Qwen35_08B_Q8_0);
        var gguf = Daisi.Llama.Gguf.GgufFile.Read(stream);
        var tokenizer = TokenizerFactory.FromGguf(gguf);

        var text = "Line 1\nLine 2\ttab";
        var ids = tokenizer.Encode(text);
        Assert.NotEmpty(ids);
        var decoded = tokenizer.Decode(ids);
        Assert.Equal(text, decoded);
    }

    [Fact]
    public void FromQwen_EosTokenId_IsValid()
    {
        if (!TestConstants.ModelExists) return;

        using var stream = File.OpenRead(TestConstants.Qwen35_08B_Q8_0);
        var gguf = Daisi.Llama.Gguf.GgufFile.Read(stream);
        var tokenizer = TokenizerFactory.FromGguf(gguf);

        Assert.True(tokenizer.Vocabulary.EosTokenId > 0);
        Assert.True(tokenizer.Vocabulary.EosTokenId < tokenizer.Vocabulary.Count);
    }
}
