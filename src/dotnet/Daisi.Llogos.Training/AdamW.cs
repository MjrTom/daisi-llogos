namespace Daisi.Llogos.Training;

/// <summary>
/// AdamW optimizer with decoupled weight decay.
/// Only updates GradTensor parameters that have gradients.
/// </summary>
public sealed class AdamW
{
    private readonly float _lr;
    private readonly float _beta1;
    private readonly float _beta2;
    private readonly float _eps;
    private readonly float _weightDecay;
    private readonly Dictionary<GradTensor, (float[] m, float[] v)> _state = new();
    private int _step;

    public AdamW(float lr = 1e-4f, float beta1 = 0.9f, float beta2 = 0.999f,
        float eps = 1e-8f, float weightDecay = 0.01f)
    {
        _lr = lr;
        _beta1 = beta1;
        _beta2 = beta2;
        _eps = eps;
        _weightDecay = weightDecay;
    }

    /// <summary>Current learning rate (after any warmup/scheduling).</summary>
    public float CurrentLr { get; private set; }

    /// <summary>
    /// Perform one optimization step on all parameters.
    /// </summary>
    public void Step(IEnumerable<GradTensor> parameters, float lrMultiplier = 1.0f)
    {
        _step++;
        float lr = _lr * lrMultiplier;
        CurrentLr = lr;

        // Bias correction
        float bc1 = 1.0f - MathF.Pow(_beta1, _step);
        float bc2 = 1.0f - MathF.Pow(_beta2, _step);

        foreach (var param in parameters)
        {
            if (param.Grad == null) continue;

            // Get or create optimizer state
            if (!_state.TryGetValue(param, out var state))
            {
                state = (new float[param.Size], new float[param.Size]);
                _state[param] = state;
            }

            var (m, v) = state;
            var data = param.Data;
            var grad = param.Grad;

            for (int i = 0; i < param.Size; i++)
            {
                // Decoupled weight decay
                data[i] -= lr * _weightDecay * data[i];

                // Adam update
                m[i] = _beta1 * m[i] + (1.0f - _beta1) * grad[i];
                v[i] = _beta2 * v[i] + (1.0f - _beta2) * grad[i] * grad[i];

                float mHat = m[i] / bc1;
                float vHat = v[i] / bc2;

                data[i] -= lr * mHat / (MathF.Sqrt(vHat) + _eps);
            }
        }
    }

    /// <summary>
    /// Cosine learning rate schedule with linear warmup.
    /// </summary>
    public static float CosineSchedule(int step, int warmupSteps, int totalSteps)
    {
        if (step < warmupSteps)
            return (float)step / warmupSteps; // linear warmup

        float progress = (float)(step - warmupSteps) / (totalSteps - warmupSteps);
        return 0.5f * (1.0f + MathF.Cos(MathF.PI * progress)); // cosine decay
    }
}
