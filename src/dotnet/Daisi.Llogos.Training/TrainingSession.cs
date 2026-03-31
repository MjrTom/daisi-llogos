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

    private List<int[]> LoadData()
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

        var texts = new List<string>();
        switch (format)
        {
            case DataFormat.PlainText:
                texts.Add(File.ReadAllText(_config.DataPath));
                break;

            case DataFormat.Jsonl:
                foreach (var line in File.ReadLines(_config.DataPath))
                {
                    if (string.IsNullOrWhiteSpace(line)) continue;
                    using var doc = JsonDocument.Parse(line);
                    var text = doc.RootElement.GetProperty("text").GetString();
                    if (!string.IsNullOrEmpty(text)) texts.Add(text);
                }
                break;

            case DataFormat.JsonlChat:
                foreach (var line in File.ReadLines(_config.DataPath))
                {
                    if (string.IsNullOrWhiteSpace(line)) continue;
                    using var doc = JsonDocument.Parse(line);
                    var prompt = doc.RootElement.GetProperty("prompt").GetString() ?? "";
                    var completion = doc.RootElement.GetProperty("completion").GetString() ?? "";
                    texts.Add(prompt + completion);
                }
                break;
        }

        // Tokenize and chunk into sequences of SeqLen+1 (input + target)
        int seqLen = _config.SeqLen;
        var allTokens = new List<int>();
        foreach (var text in texts)
        {
            var tokens = _tokenizer!.Encode(text);
            allTokens.AddRange(tokens);
        }

        var sequences = new List<int[]>();
        for (int i = 0; i + seqLen < allTokens.Count; i += seqLen)
        {
            var seq = new int[seqLen + 1]; // +1 for target
            for (int j = 0; j <= seqLen; j++)
                seq[j] = allTokens[i + j];
            sequences.Add(seq);
        }

        Console.Error.WriteLine($" done");
        Console.Error.WriteLine($"  Tokens: {allTokens.Count:N0}, Sequences: {sequences.Count:N0} (seqLen={seqLen})");

        if (sequences.Count == 0)
            throw new InvalidOperationException("No training sequences produced — data may be too short.");

        return sequences;
    }

    private void Train(List<int[]> sequences)
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
                var input = seq[..seqLen];
                var targets = seq[1..(seqLen + 1)];

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
