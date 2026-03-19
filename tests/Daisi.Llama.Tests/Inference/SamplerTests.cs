using Daisi.Llama.Inference;

namespace Daisi.Llama.Tests.Inference;

public class SamplerTests
{
    [Fact]
    public void Greedy_SelectsMaxLogit()
    {
        var sampler = new Sampler(seed: 42);
        float[] logits = [1.0f, 5.0f, 3.0f, 2.0f];
        var p = new GenerationParams { Temperature = 0 };

        int token = sampler.Sample(logits, p, []);

        Assert.Equal(1, token);
    }

    [Fact]
    public void Greedy_SelectsMaxLogit_LargeVocab()
    {
        var sampler = new Sampler(seed: 42);
        var logits = new float[1000];
        logits[742] = 10.0f;
        var p = new GenerationParams { Temperature = 0 };

        int token = sampler.Sample(logits, p, []);

        Assert.Equal(742, token);
    }

    [Fact]
    public void Temperature_IncreasesEntropy()
    {
        // With low temperature, distribution should be more peaked
        // With high temperature, distribution should be more uniform
        // We test this by sampling many times and checking spread
        float[] logits = [2.0f, 1.0f, 0.5f, 0.1f];

        var lowTemp = new GenerationParams { Temperature = 0.1f, TopK = 0, TopP = 1.0f, RepetitionPenalty = 1.0f };
        var highTemp = new GenerationParams { Temperature = 2.0f, TopK = 0, TopP = 1.0f, RepetitionPenalty = 1.0f };

        int[] lowCounts = SampleMany(logits, lowTemp, 1000);
        int[] highCounts = SampleMany(logits, highTemp, 1000);

        // Low temp should heavily favor token 0
        Assert.True(lowCounts[0] > 900, $"Low temp token 0 count: {lowCounts[0]} (expected >900)");
        // High temp should spread more evenly
        Assert.True(highCounts[0] < 600, $"High temp token 0 count: {highCounts[0]} (expected <600)");
    }

    [Fact]
    public void TopK_FiltersLowProbability()
    {
        float[] logits = [5.0f, 4.0f, 3.0f, 2.0f, 1.0f, 0.0f, -1.0f, -2.0f];
        var p = new GenerationParams { Temperature = 1.0f, TopK = 3, TopP = 1.0f, RepetitionPenalty = 1.0f };

        int[] counts = SampleMany(logits, p, 1000);

        // Tokens 3-7 should never be selected (filtered by top-k=3)
        for (int i = 3; i < 8; i++)
            Assert.Equal(0, counts[i]);

        // Top 3 tokens should all be sampled
        Assert.True(counts[0] > 0);
        Assert.True(counts[1] > 0);
        Assert.True(counts[2] > 0);
    }

    [Fact]
    public void TopP_CumulativeThreshold()
    {
        // Create logits where one token dominates
        float[] logits = [10.0f, 1.0f, 0.0f, -1.0f];
        var p = new GenerationParams { Temperature = 1.0f, TopK = 0, TopP = 0.5f, RepetitionPenalty = 1.0f };

        int[] counts = SampleMany(logits, p, 500);

        // Token 0 has such high logit that top-p=0.5 should mostly keep only it
        Assert.True(counts[0] > 400, $"TopP token 0 count: {counts[0]}");
    }

    [Fact]
    public void RepetitionPenalty_ReducesSeen()
    {
        float[] logits = [3.0f, 3.0f, 3.0f, 3.0f]; // Equal logits
        var noPenalty = new GenerationParams { Temperature = 0, RepetitionPenalty = 1.0f };
        var withPenalty = new GenerationParams { Temperature = 0, RepetitionPenalty = 2.0f };

        var sampler = new Sampler(seed: 42);

        // Without penalty, any token could win (they're equal)
        int noPenToken = sampler.Sample(logits, noPenalty, [0, 1]);

        // With penalty, tokens 0 and 1 should be penalized, making 2 or 3 win
        int penToken = sampler.Sample(logits, withPenalty, [0, 1]);
        Assert.True(penToken >= 2, $"Penalized token should be 2 or 3, got {penToken}");
    }

    [Fact]
    public void Seed_Deterministic()
    {
        float[] logits = [1.0f, 1.0f, 1.0f, 1.0f, 1.0f];
        var p = new GenerationParams { Temperature = 1.0f, TopK = 0, TopP = 1.0f, RepetitionPenalty = 1.0f };

        var sampler1 = new Sampler(seed: 123);
        var sampler2 = new Sampler(seed: 123);

        var seq1 = new int[20];
        var seq2 = new int[20];
        for (int i = 0; i < 20; i++)
        {
            seq1[i] = sampler1.Sample(logits, p, []);
            seq2[i] = sampler2.Sample(logits, p, []);
        }

        Assert.Equal(seq1, seq2);
    }

    private static int[] SampleMany(float[] logits, GenerationParams p, int count)
    {
        var sampler = new Sampler(seed: 42);
        var counts = new int[logits.Length];
        for (int i = 0; i < count; i++)
            counts[sampler.Sample(logits, p, [])]++;
        return counts;
    }
}
