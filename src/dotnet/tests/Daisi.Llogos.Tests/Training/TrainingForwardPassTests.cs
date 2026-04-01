using Daisi.Llogos.Cpu;
using Daisi.Llogos.Gguf;
using Daisi.Llogos.Model;
using Daisi.Llogos.Training;
using Daisi.Llogos.Training.Lora;
using Xunit;

namespace Daisi.Llogos.Tests.Training;

/// <summary>
/// Tests for the training forward/backward pass, focused on verifying the
/// memory optimization (re-dequantize weights in backward instead of caching).
/// </summary>
public class TrainingForwardPassTests : IDisposable
{
    // Tiny model dimensions — fast tests, exercises all code paths
    private const int NumLayers = 2;
    private const int HiddenDim = 32;
    private const int IntermediateDim = 64;
    private const int VocabSize = 64;
    private const int NumHeads = 4;
    private const int NumKvHeads = 2;
    private const int KeyLength = 8;
    private const int ValueLength = 8;
    private const int SeqLen = 8;
    private const int LoraRank = 4;
    private const float NormEps = 1e-5f;
    private const float RopeTheta = 10000f;

    private readonly CpuBackend _backend = new();
    private readonly ModelConfig _config;
    private readonly ModelWeights _weights;

    public TrainingForwardPassTests()
    {
        _config = new ModelConfig
        {
            Architecture = "test",
            NumLayers = NumLayers,
            HiddenDim = HiddenDim,
            IntermediateDim = IntermediateDim,
            VocabSize = VocabSize,
            MaxContext = 128,
            NormEps = NormEps,
            NumHeads = NumHeads,
            NumKvHeads = NumKvHeads,
            KeyLength = KeyLength,
            ValueLength = ValueLength,
            RopeTheta = RopeTheta,
            RopeDimCount = KeyLength,
            FullAttentionInterval = 0, // all standard attention
            SsmConvKernel = 0,
            SsmStateSize = 0,
            SsmGroupCount = 0,
            SsmInnerSize = 0,
        };

        _weights = BuildSyntheticWeights(seed: 42);
    }

    public void Dispose()
    {
        _weights.Dispose();
        _backend.Dispose();
    }

    // ── Test 1: Dequantization determinism ──────────────────────────────────

    [Fact]
    public void Dequantize_F32_IsDeterministic()
    {
        var tensor = CreateRandomTensor("test_f32", GgmlType.F32, [HiddenDim, IntermediateDim], seed: 99);

        var first = F32Ops.Dequantize(tensor);
        var second = F32Ops.Dequantize(tensor);

        Assert.Equal(first.Length, second.Length);
        for (int i = 0; i < first.Length; i++)
            Assert.Equal(first[i], second[i]);

        tensor.Dispose();
    }

    [Fact]
    public void Dequantize_Q8_0_IsDeterministic()
    {
        // Q8_0 block size is 32 elements, so dimensions must be multiples of 32
        var tensor = CreateRandomTensor("test_q8", GgmlType.Q8_0, [32, 64], seed: 99);

        var first = F32Ops.Dequantize(tensor);
        var second = F32Ops.Dequantize(tensor);

        Assert.Equal(first.Length, second.Length);
        for (int i = 0; i < first.Length; i++)
            Assert.Equal(first[i], second[i]);

        tensor.Dispose();
    }

    // ── Test 2: Forward+backward determinism (proves re-dequant is safe) ───

    [Fact]
    public void ForwardBackward_ProducesIdenticalGradients_OnRepeatedRuns()
    {
        // Run 1: fresh adapter, forward+backward, capture gradients
        var adapter1 = CreateAdapter(seed: 123);
        using var fwd1 = new TrainingForwardPass(_config, _weights, adapter1);

        var input = MakeSequence(SeqLen, seed: 7);
        var targets = MakeSequence(SeqLen, seed: 8);

        fwd1.Forward(input, out float loss1, targets);
        var grads1 = SnapshotGradients(adapter1);

        // Run 2: identical adapter (same seed), same input
        var adapter2 = CreateAdapter(seed: 123);
        using var fwd2 = new TrainingForwardPass(_config, _weights, adapter2);

        fwd2.Forward(input, out float loss2, targets);
        var grads2 = SnapshotGradients(adapter2);

        // Loss must be bit-identical
        Assert.Equal(loss1, loss2);

        // All LoRA gradients must be bit-identical
        Assert.Equal(grads1.Count, grads2.Count);
        foreach (var key in grads1.Keys)
        {
            Assert.True(grads2.ContainsKey(key), $"Missing gradient: {key}");
            var g1 = grads1[key];
            var g2 = grads2[key];
            Assert.Equal(g1.Length, g2.Length);
            for (int i = 0; i < g1.Length; i++)
                Assert.Equal(g1[i], g2[i]);
        }

        adapter1.Dispose();
        adapter2.Dispose();
    }

    [Fact]
    public void ForwardBackward_LossIsFinite()
    {
        var adapter = CreateAdapter(seed: 42);
        using var fwd = new TrainingForwardPass(_config, _weights, adapter);

        var input = MakeSequence(SeqLen, seed: 1);
        var targets = MakeSequence(SeqLen, seed: 2);

        fwd.Forward(input, out float loss, targets);

        Assert.True(float.IsFinite(loss), $"Loss is not finite: {loss}");
        Assert.True(loss > 0, $"Loss should be positive: {loss}");

        adapter.Dispose();
    }

    [Fact]
    public void ForwardBackward_GradientsAreFinite()
    {
        var adapter = CreateAdapter(seed: 42);
        using var fwd = new TrainingForwardPass(_config, _weights, adapter);

        var input = MakeSequence(SeqLen, seed: 1);
        var targets = MakeSequence(SeqLen, seed: 2);

        fwd.Forward(input, out _, targets);

        foreach (var param in adapter.Parameters())
        {
            Assert.NotNull(param.Grad);
            for (int i = 0; i < param.Size; i++)
                Assert.True(float.IsFinite(param.Grad[i]),
                    $"Non-finite gradient at index {i}");
        }

        adapter.Dispose();
    }

    // ── Test 3: Saved activations are cleaned up after backward ────────────

    [Fact]
    public void Backward_CleansUpSavedActivations()
    {
        var adapter = CreateAdapter(seed: 42);
        using var fwd = new TrainingForwardPass(_config, _weights, adapter);

        var input = MakeSequence(SeqLen, seed: 1);
        var targets = MakeSequence(SeqLen, seed: 2);

        // Forward + backward (backward is called inside Forward)
        fwd.Forward(input, out _, targets);

        // The _savedActivations field should have all null entries after backward.
        // We verify this indirectly: a second forward should work without issues
        // (if activations weren't cleaned, stale references could cause problems)
        fwd.Forward(input, out float loss2, targets);
        Assert.True(float.IsFinite(loss2));

        adapter.Dispose();
    }

    // ── Test 4: Multi-step training converges ──────────────────────────────

    [Fact]
    public void MultiStep_LossDecreases()
    {
        var adapter = CreateAdapter(seed: 42);
        using var fwd = new TrainingForwardPass(_config, _weights, adapter);
        var optimizer = new AdamW(lr: 1e-3f, weightDecay: 0);

        var input = MakeSequence(SeqLen, seed: 10);
        var targets = MakeSequence(SeqLen, seed: 11);

        float firstLoss = 0;
        float lastLoss = 0;

        for (int step = 0; step < 20; step++)
        {
            adapter.ZeroGrad();
            fwd.Forward(input, out float loss, targets);

            if (step == 0) firstLoss = loss;
            lastLoss = loss;

            optimizer.Step(adapter.Parameters());
        }

        // Loss should decrease over 20 steps of overfitting on one sequence
        Assert.True(lastLoss < firstLoss,
            $"Loss did not decrease: first={firstLoss:F4}, last={lastLoss:F4}");
    }

    // ── Test 5: Gradient equivalence across multiple steps ─────────────────

    [Fact]
    public void MultiStep_ParameterValues_AreDeterministic()
    {
        // Two independent training runs with same seeds should produce
        // bit-identical parameter values after N steps
        float[] RunTraining(int runSeed)
        {
            var adapter = CreateAdapter(seed: runSeed);
            using var fwd = new TrainingForwardPass(_config, _weights, adapter);
            var optimizer = new AdamW(lr: 1e-3f, weightDecay: 0.01f);
            var input = MakeSequence(SeqLen, seed: 10);
            var targets = MakeSequence(SeqLen, seed: 11);

            for (int step = 0; step < 5; step++)
            {
                adapter.ZeroGrad();
                fwd.Forward(input, out _, targets);
                optimizer.Step(adapter.Parameters());
            }

            // Snapshot all parameter values
            var allParams = new List<float>();
            foreach (var param in adapter.Parameters())
                allParams.AddRange(param.Data.AsSpan(0, param.Size).ToArray());
            adapter.Dispose();
            return allParams.ToArray();
        }

        var params1 = RunTraining(42);
        var params2 = RunTraining(42);

        Assert.Equal(params1.Length, params2.Length);
        for (int i = 0; i < params1.Length; i++)
            Assert.Equal(params1[i], params2[i]);
    }

    // ── Helpers ─────────────────────────────────────────────────────────────

    private LoraAdapter CreateAdapter(int seed)
    {
        var loraConfig = new LoraConfig
        {
            Rank = LoraRank,
            Alpha = LoraRank * 2,
            Targets = LoraTarget.Q | LoraTarget.K | LoraTarget.V | LoraTarget.O,
        };
        return new LoraAdapter(loraConfig, _config, _weights, seed);
    }

    private static Dictionary<string, float[]> SnapshotGradients(LoraAdapter adapter)
    {
        var result = new Dictionary<string, float[]>();
        foreach (var (name, layer) in adapter.Layers)
        {
            if (layer.A.Grad != null)
                result[$"{name}.A"] = layer.A.Grad.ToArray();
            if (layer.B.Grad != null)
                result[$"{name}.B"] = layer.B.Grad.ToArray();
        }
        return result;
    }

    private static int[] MakeSequence(int len, int seed)
    {
        var rng = new Random(seed);
        var seq = new int[len];
        for (int i = 0; i < len; i++)
            seq[i] = rng.Next(VocabSize);
        return seq;
    }

    private ModelWeights BuildSyntheticWeights(int seed)
    {
        var rng = new Random(seed);
        int qOutDim = NumHeads * KeyLength;       // 32 — no gated Q
        int kOutDim = NumKvHeads * KeyLength;     // 16
        int vOutDim = NumKvHeads * ValueLength;   // 16
        int oDim = NumHeads * ValueLength;        // 32

        var layers = new LayerWeights[NumLayers];
        for (int i = 0; i < NumLayers; i++)
        {
            layers[i] = new StandardAttentionWeights
            {
                AttnNorm = CreateRandomTensor($"blk.{i}.attn_norm", GgmlType.F32, [HiddenDim], rng),
                PostAttnNorm = CreateRandomTensor($"blk.{i}.post_attn_norm", GgmlType.F32, [HiddenDim], rng),
                // GGUF layout: [inDim, outDim]
                AttnQ = CreateRandomTensor($"blk.{i}.attn_q", GgmlType.F32, [HiddenDim, qOutDim], rng),
                AttnK = CreateRandomTensor($"blk.{i}.attn_k", GgmlType.F32, [HiddenDim, kOutDim], rng),
                AttnV = CreateRandomTensor($"blk.{i}.attn_v", GgmlType.F32, [HiddenDim, vOutDim], rng),
                AttnO = CreateRandomTensor($"blk.{i}.attn_o", GgmlType.F32, [oDim, HiddenDim], rng),
                FfnGate = CreateRandomTensor($"blk.{i}.ffn_gate", GgmlType.F32, [HiddenDim, IntermediateDim], rng),
                FfnUp = CreateRandomTensor($"blk.{i}.ffn_up", GgmlType.F32, [HiddenDim, IntermediateDim], rng),
                FfnDown = CreateRandomTensor($"blk.{i}.ffn_down", GgmlType.F32, [IntermediateDim, HiddenDim], rng),
            };
        }

        return new ModelWeights
        {
            TokenEmbedding = CreateRandomTensor("token_embd", GgmlType.F32, [HiddenDim, VocabSize], rng),
            OutputNorm = CreateRandomTensor("output_norm", GgmlType.F32, [HiddenDim], rng),
            Output = CreateRandomTensor("output", GgmlType.F32, [HiddenDim, VocabSize], rng),
            Layers = layers,
        };
    }

    private ITensor CreateRandomTensor(string name, GgmlType type, long[] dims, Random rng)
    {
        var tensor = _backend.CreateTensor(name, type, dims);
        // Fill with small random values to avoid numerical issues
        if (type == GgmlType.F32)
        {
            var span = tensor.AsFloatSpan();
            for (int i = 0; i < span.Length; i++)
                span[i] = (float)(rng.NextDouble() * 0.2 - 0.1); // U(-0.1, 0.1)
        }
        else
        {
            // For quantized types, fill raw bytes with random data
            var raw = new byte[tensor.ByteSize];
            rng.NextBytes(raw);
            tensor.CopyFrom(raw);
        }
        return tensor;
    }

    private ITensor CreateRandomTensor(string name, GgmlType type, long[] dims, int seed)
        => CreateRandomTensor(name, type, dims, new Random(seed));
}
