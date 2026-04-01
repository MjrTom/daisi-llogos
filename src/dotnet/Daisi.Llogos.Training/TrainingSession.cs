using System.Diagnostics;
using System.Text.Json;
using Daisi.Llogos.Gguf;
using Daisi.Llogos.Model;
using Daisi.Llogos.Tokenizer;
using Daisi.Llogos.Training.Lora;

namespace Daisi.Llogos.Training;

/// <summary>
/// Orchestrates LoRA training: loads model and data, runs training loop, saves checkpoints.
/// </summary>
public sealed class TrainingSession : IDisposable
{
    private readonly TrainingConfig _config;
    private readonly IComputeBackend _backend;
    private ModelConfig? _modelConfig;
    private ModelWeights? _weights;
    private BpeTokenizer? _tokenizer;
    private LoraAdapter? _adapter;
    private ITrainingForwardPass? _forwardPass;
    private AdamW? _optimizer;
    private GgufFile? _gguf;

    public TrainingSession(TrainingConfig config, IComputeBackend? backend = null)
    {
        _config = config;
        _backend = backend ?? new Daisi.Llogos.Cpu.CpuBackend();
    }

    /// <summary>
    /// Run the full training pipeline.
    /// </summary>
    public void Run()
    {
        Console.Error.WriteLine("=== Llogos LoRA Training ===");
        Console.Error.WriteLine();

        // 1. Load model
        LoadModel();

        // 2. Load and tokenize training data
        var sequences = LoadData();

        // 3. Initialize LoRA adapter
        _adapter = new LoraAdapter(_config.Lora, _modelConfig!, _weights!, _config.Seed);
        Console.Error.WriteLine($"LoRA adapter: {_adapter.Layers.Count} layers, " +
            $"{_adapter.ParameterCount:N0} trainable parameters " +
            $"(rank={_config.Lora.Rank}, alpha={_config.Lora.Alpha})");

        // 4. Initialize training components
        if (_backend is Daisi.Llogos.Cuda.CudaBackend cudaBackend)
        {
            _forwardPass = new GpuTrainingForwardPass(_modelConfig!, _weights!, _adapter, cudaBackend);
            Console.Error.WriteLine($"  Backend: CUDA (GPU training)");
        }
        else
        {
            _forwardPass = new TrainingForwardPass(_modelConfig!, _weights!, _adapter, _backend);
            Console.Error.WriteLine($"  Backend: CPU");
        }
        _optimizer = new AdamW(_config.LearningRate, weightDecay: _config.WeightDecay);

        // 5. Training loop
        Train(sequences);

        // 6. Download final weights from GPU and save adapter
        if (_forwardPass is GpuTrainingForwardPass gpuFwdSave)
            gpuFwdSave.DownloadLoraWeights();
        var outputPath = _config.OutputPath;
        Directory.CreateDirectory(Path.GetDirectoryName(outputPath) ?? outputPath);
        LoraFile.Save(outputPath, _adapter);
        Console.Error.WriteLine($"Saved LoRA adapter to: {outputPath}");
    }

    private void LoadModel()
    {
        Console.Error.Write($"Loading model: {_config.ModelPath}...");
        var sw = Stopwatch.StartNew();

        using var stream = File.OpenRead(_config.ModelPath);
        _gguf = GgufFile.Read(stream);
        _modelConfig = ModelConfig.FromGguf(_gguf);
        _tokenizer = TokenizerFactory.FromGguf(_gguf);

        // Always load on CPU. GPU training uploads weights to GPU lazily via weight cache.
        var cpuBackend = new Daisi.Llogos.Cpu.CpuBackend();
        _weights = ModelLoader.Load(_gguf, stream, cpuBackend, _modelConfig);

        sw.Stop();
        Console.Error.WriteLine($" done ({sw.Elapsed.TotalSeconds:F1}s)");
        Console.Error.WriteLine($"  Architecture: {_modelConfig.Architecture}");
        Console.Error.WriteLine($"  Layers: {_modelConfig.NumLayers} ({CountAttentionLayers()} attention, " +
            $"{_modelConfig.NumLayers - CountAttentionLayers()} DeltaNet)");
        Console.Error.WriteLine($"  Hidden: {_modelConfig.HiddenDim}, Heads: {_modelConfig.NumHeads}/{_modelConfig.NumKvHeads}");
        Console.Error.WriteLine($"  Vocab: {_modelConfig.VocabSize:N0}");
    }

    private int CountAttentionLayers()
    {
        int count = 0;
        for (int i = 0; i < _modelConfig!.NumLayers; i++)
            if (_modelConfig.IsStandardAttention(i)) count++;
        return count;
    }

    /// <summary>Sequence with prompt mask — promptLen tokens have target=-1 (ignored in loss).</summary>
    private record TrainingSequence(int[] Tokens, int PromptLen);

    private List<TrainingSequence> LoadData()
    {
        Console.Error.Write($"Loading training data: {_config.DataPath}...");

        var format = _config.Format;
        if (format == DataFormat.Auto)
        {
            string ext = Path.GetExtension(_config.DataPath).ToLowerInvariant();
            format = ext switch
            {
                ".jsonl" or ".json" => DataFormat.Jsonl,
                _ => DataFormat.PlainText,
            };
        }

        int seqLen = _config.SeqLen;
        var sequences = new List<TrainingSequence>();
        int totalTokens = 0;

        // For JSONL with "text" field: detect chat template and split on assistant response
        // Format: <|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n[RESPONSE]<|im_end|>
        const string assistantMarker = "<|im_start|>assistant\n";

        switch (format)
        {
            case DataFormat.PlainText:
            {
                var allTokens = _tokenizer!.Encode(File.ReadAllText(_config.DataPath));
                totalTokens = allTokens.Length;
                for (int i = 0; i + seqLen < allTokens.Length; i += seqLen)
                {
                    var seq = new int[seqLen + 1];
                    Array.Copy(allTokens, i, seq, 0, seqLen + 1);
                    sequences.Add(new TrainingSequence(seq, 0)); // no masking for plain text
                }
                break;
            }

            case DataFormat.Jsonl:
            {
                foreach (var line in File.ReadLines(_config.DataPath))
                {
                    if (string.IsNullOrWhiteSpace(line)) continue;
                    using var doc = JsonDocument.Parse(line);
                    var text = doc.RootElement.GetProperty("text").GetString();
                    if (string.IsNullOrEmpty(text)) continue;

                    // Detect chat template: find assistant marker to split prompt/completion
                    int promptLen = 0;
                    int markerPos = text.IndexOf(assistantMarker, StringComparison.Ordinal);
                    if (markerPos >= 0)
                    {
                        // Tokenize prompt portion to find token boundary
                        var promptText = text[..(markerPos + assistantMarker.Length)];
                        promptLen = _tokenizer!.Encode(promptText).Length;
                    }

                    var tokens = _tokenizer!.Encode(text);
                    totalTokens += tokens.Length;

                    // Pad or truncate to seqLen+1
                    if (tokens.Length >= seqLen + 1)
                    {
                        var seq = new int[seqLen + 1];
                        Array.Copy(tokens, seq, seqLen + 1);
                        sequences.Add(new TrainingSequence(seq, Math.Min(promptLen, seqLen)));
                    }
                    else if (tokens.Length > 10)
                    {
                        // Pad short sequences
                        var seq = new int[seqLen + 1];
                        Array.Copy(tokens, seq, tokens.Length);
                        sequences.Add(new TrainingSequence(seq, Math.Min(promptLen, tokens.Length)));
                    }
                }
                break;
            }

            case DataFormat.JsonlChat:
            {
                foreach (var line in File.ReadLines(_config.DataPath))
                {
                    if (string.IsNullOrWhiteSpace(line)) continue;
                    using var doc = JsonDocument.Parse(line);
                    var prompt = doc.RootElement.GetProperty("prompt").GetString() ?? "";
                    var completion = doc.RootElement.GetProperty("completion").GetString() ?? "";
                    var promptTokens = _tokenizer!.Encode(prompt);
                    var completionTokens = _tokenizer!.Encode(completion);
                    var allTokens = new int[promptTokens.Length + completionTokens.Length];
                    Array.Copy(promptTokens, allTokens, promptTokens.Length);
                    Array.Copy(completionTokens, 0, allTokens, promptTokens.Length, completionTokens.Length);
                    totalTokens += allTokens.Length;

                    if (allTokens.Length >= seqLen + 1)
                    {
                        var seq = new int[seqLen + 1];
                        Array.Copy(allTokens, seq, seqLen + 1);
                        sequences.Add(new TrainingSequence(seq, Math.Min(promptTokens.Length, seqLen)));
                    }
                    else if (allTokens.Length > 10)
                    {
                        var seq = new int[seqLen + 1];
                        Array.Copy(allTokens, seq, allTokens.Length);
                        sequences.Add(new TrainingSequence(seq, Math.Min(promptTokens.Length, allTokens.Length)));
                    }
                }
                break;
            }
        }

        int maskedSeqs = sequences.Count(s => s.PromptLen > 0);
        Console.Error.WriteLine($" done");
        Console.Error.WriteLine($"  Tokens: {totalTokens:N0}, Sequences: {sequences.Count:N0} (seqLen={seqLen})");
        if (maskedSeqs > 0)
            Console.Error.WriteLine($"  Completion-only loss: {maskedSeqs} sequences with prompt masking");

        if (sequences.Count == 0)
            throw new InvalidOperationException("No training sequences produced — data may be too short.");

        return sequences;
    }

    private void Train(List<TrainingSequence> sequences)
    {
        int totalSteps = sequences.Count * _config.Epochs / _config.GradientAccumulationSteps;
        int warmupSteps = _config.WarmupSteps;
        int step = 0;
        var rng = new Random(_config.Seed);
        var sw = Stopwatch.StartNew();
        float runningLoss = 0;
        int lossCount = 0;

        Console.Error.WriteLine();
        Console.Error.WriteLine($"Training for {_config.Epochs} epochs, {totalSteps} steps");
        Console.Error.WriteLine($"  LR: {_config.LearningRate}, Warmup: {warmupSteps} steps");
        Console.Error.WriteLine($"  Grad accumulation: {_config.GradientAccumulationSteps}");
        Console.Error.WriteLine();

        for (int epoch = 0; epoch < _config.Epochs; epoch++)
        {
            // Shuffle sequences
            for (int i = sequences.Count - 1; i > 0; i--)
            {
                int j = rng.Next(i + 1);
                (sequences[i], sequences[j]) = (sequences[j], sequences[i]);
            }

            float epochLoss = 0;
            int epochSteps = 0;

            for (int seqIdx = 0; seqIdx < sequences.Count; seqIdx++)
            {
                var seq = sequences[seqIdx];
                int seqLen = _config.SeqLen;

                // Input: tokens[0..seqLen-1], Target: tokens[1..seqLen]
                var input = seq.Tokens[..seqLen];
                var targets = seq.Tokens[1..(seqLen + 1)];

                // Mask prompt tokens: set target=-1 for tokens the model shouldn't learn to predict
                // (loss computation skips target=-1). This focuses learning on the completion only.
                if (seq.PromptLen > 0)
                {
                    for (int t = 0; t < Math.Min(seq.PromptLen, seqLen); t++)
                        targets[t] = -1;
                }
                // Also mask padding tokens (sequences shorter than seqLen are zero-padded)
                for (int t = 0; t < seqLen; t++)
                {
                    if (input[t] == 0 && targets[t] == 0)
                        targets[t] = -1;
                }

                // Zero gradients at start of accumulation window
                if (seqIdx % _config.GradientAccumulationSteps == 0 &&
                    _forwardPass is not GpuTrainingForwardPass)
                    _adapter!.ZeroGrad(); // GPU path zeros grads in GpuOptimizerStep

                // Forward + backward
                _forwardPass!.Forward(input, out float loss, targets);

                runningLoss += loss;
                epochLoss += loss;
                lossCount++;
                epochSteps++;

                // Optimizer step at end of accumulation window
                if ((seqIdx + 1) % _config.GradientAccumulationSteps == 0)
                {
                    step++;
                    float lrMult = AdamW.CosineSchedule(step, warmupSteps, totalSteps);

                    if (_forwardPass is GpuTrainingForwardPass gpuFwd)
                    {
                        // Entire optimizer step on GPU — no CPU round-trip
                        gpuFwd.GpuOptimizerStep(_config.LearningRate,
                            0.9f, 0.999f, 1e-8f, _config.WeightDecay,
                            _config.MaxGradNorm, lrMult);
                    }
                    else
                    {
                        ClipGradNorm(_config.MaxGradNorm);
                        _optimizer!.Step(_adapter!.Parameters(), lrMult);
                    }

                    // Logging
                    if (step % _config.LogEverySteps == 0)
                    {
                        float avgLoss = runningLoss / lossCount;
                        float elapsed = (float)sw.Elapsed.TotalSeconds;
                        float seqPerSec = epochSteps / elapsed;
                        Console.Error.WriteLine(
                            $"  epoch {epoch + 1}/{_config.Epochs} | step {step}/{totalSteps} | " +
                            $"loss {avgLoss:F4} | lr {_config.LearningRate * lrMult:E2} | " +
                            $"{seqPerSec:F1} seq/s");
                        runningLoss = 0;
                        lossCount = 0;
                    }

                    // Checkpoint
                    if (_config.SaveEverySteps > 0 && step % _config.SaveEverySteps == 0)
                    {
                        var checkpointPath = Path.Combine(
                            Path.GetDirectoryName(_config.OutputPath) ?? ".",
                            $"checkpoint-step{step}.llra");
                        LoraFile.Save(checkpointPath, _adapter!);
                        Console.Error.WriteLine($"  Checkpoint saved: {checkpointPath}");
                    }
                }
            }

            Console.Error.WriteLine(
                $"  Epoch {epoch + 1} complete — avg loss: {epochLoss / epochSteps:F4}");
        }

        Console.Error.WriteLine();
        Console.Error.WriteLine($"Training complete in {sw.Elapsed.TotalMinutes:F1} minutes ({step} steps)");
    }

    private void ClipGradNorm(float maxNorm)
    {
        // Compute total gradient norm
        float totalNormSq = 0;
        foreach (var param in _adapter!.Parameters())
        {
            if (param.Grad == null) continue;
            for (int i = 0; i < param.Size; i++)
                totalNormSq += param.Grad[i] * param.Grad[i];
        }

        float totalNorm = MathF.Sqrt(totalNormSq);
        if (totalNorm > maxNorm)
        {
            float scale = maxNorm / totalNorm;
            foreach (var param in _adapter.Parameters())
            {
                if (param.Grad == null) continue;
                F32Ops.Scale(param.Grad, scale, param.Size);
            }
        }
    }

    public void Dispose()
    {
        _forwardPass?.Dispose();
        _adapter?.Dispose();
        _weights?.Dispose();
    }
}
