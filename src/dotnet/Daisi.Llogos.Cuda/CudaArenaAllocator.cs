using Daisi.Llogos.Gguf;

namespace Daisi.Llogos.Cuda;

/// <summary>
/// Pre-registered tensor pool for GPU training. All tensor names and sizes are declared
/// upfront, then allocated in one batch via the CudaBackend. This guarantees:
///   - No dynamic allocation during training (CUDA graph compatible)
///   - All tensors can be zeroed in a single pass
///   - Deterministic memory layout for reproducibility
///
/// Uses standard CudaTensor objects (fully compatible with CudaBackend operations).
///
/// Usage:
///   1. Register phase: call Register() for each tensor name + size
///   2. Allocate phase: call Allocate(backend) to create all tensors at once
///   3. Use phase: call Get() to retrieve tensors (same instance every time)
///   4. ZeroAll(backend): zero all tensors via batch memset
/// </summary>
public sealed class CudaArenaAllocator : IDisposable
{
    private readonly Dictionary<string, int> _registry = new();     // name → element count
    private readonly Dictionary<string, ITensor> _tensors = new();  // name → allocated tensor
    private bool _allocated;
    private bool _disposed;

    /// <summary>Number of registered tensors.</summary>
    public int TensorCount => _registry.Count;

    /// <summary>Whether Allocate() has been called.</summary>
    public bool IsAllocated => _allocated;

    /// <summary>
    /// Register a tensor to be allocated. Must be called before Allocate().
    /// Idempotent — registering the same name twice with the same size is fine.
    /// </summary>
    public void Register(string name, int elementCount)
    {
        if (_allocated)
            throw new InvalidOperationException("Cannot register tensors after Allocate().");
        _registry[name] = elementCount;
    }

    /// <summary>
    /// Allocate all registered tensors via the backend. Single batch of cuMemAlloc calls.
    /// </summary>
    public void Allocate(IComputeBackend backend)
    {
        if (_allocated)
            throw new InvalidOperationException("Arena already allocated.");

        long totalBytes = 0;
        foreach (var (name, elementCount) in _registry)
        {
            var tensor = backend.CreateTensor(name, GgmlType.F32, [(long)elementCount]);
            _tensors[name] = tensor;
            totalBytes += elementCount * sizeof(float);
        }

        _allocated = true;
        Console.Error.WriteLine($"  Arena: {_registry.Count} tensors, {totalBytes / 1024.0 / 1024.0:F1} MB");
    }

    /// <summary>
    /// Get a tensor by name. Returns the same ITensor instance every call.
    /// </summary>
    public ITensor Get(string name)
    {
        if (!_allocated)
            throw new InvalidOperationException("Arena not allocated. Call Allocate() first.");
        if (!_tensors.TryGetValue(name, out var tensor))
            throw new KeyNotFoundException($"Tensor '{name}' not registered in arena.");
        return tensor;
    }

    /// <summary>
    /// Try to get a tensor. Returns false if not registered.
    /// </summary>
    public bool TryGet(string name, out ITensor tensor) =>
        _tensors.TryGetValue(name, out tensor!);

    /// <summary>
    /// Zero all tensors. Uses the backend's ZeroTensor for each.
    /// </summary>
    public void ZeroAll(IComputeBackend backend)
    {
        foreach (var tensor in _tensors.Values)
            backend.ZeroTensor(tensor);
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            foreach (var tensor in _tensors.Values)
                tensor.Dispose();
            _tensors.Clear();
            _disposed = true;
        }
    }
}
