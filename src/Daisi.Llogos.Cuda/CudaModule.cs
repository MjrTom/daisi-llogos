using System.Reflection;
using System.Text;

namespace Daisi.Llogos.Cuda;

/// <summary>
/// Wraps a CUDA module loaded from PTX (JIT compiled by the driver).
/// Source .cu files are compiled to PTX at runtime using NVRTC.
/// </summary>
internal sealed class CudaModule : IDisposable
{
    private nint _handle;
    private bool _disposed;
    private readonly Dictionary<string, nint> _functions = new();

    private CudaModule(nint handle)
    {
        _handle = handle;
    }

    /// <summary>
    /// Load a module from PTX source text.
    /// </summary>
    public static CudaModule FromPtx(byte[] ptxBytes)
    {
        CudaApi.Check(
            CudaApi.ModuleLoadDataEx(out nint module, ptxBytes, 0, 0, 0),
            "cuModuleLoadDataEx");
        return new CudaModule(module);
    }

    /// <summary>
    /// Compile CUDA C++ source to PTX using NVRTC, then load as a module.
    /// </summary>
    public static CudaModule FromCudaSource(string cudaSource, string? name = null)
    {
        // Create NVRTC program
        NvrtcApi.Check(
            NvrtcApi.CreateProgram(out nint prog, cudaSource, name, 0, 0, 0),
            "nvrtcCreateProgram");

        try
        {
            // Compile with fast math for performance
            var options = new[] { "--use_fast_math" };
            var result = NvrtcApi.CompileProgram(prog, options.Length, options);
            if (result != NvrtcResult.Success)
            {
                // Get compilation log
                NvrtcApi.GetProgramLogSize(prog, out nuint logSize);
                var logBuf = new byte[(int)logSize];
                NvrtcApi.GetProgramLog(prog, logBuf);
                var log = Encoding.ASCII.GetString(logBuf).TrimEnd('\0');
                throw new CudaException($"NVRTC compilation failed: {log}");
            }

            // Get PTX
            NvrtcApi.Check(NvrtcApi.GetPTXSize(prog, out nuint ptxSize), "nvrtcGetPTXSize");
            var ptxBytes = new byte[(int)ptxSize];
            NvrtcApi.Check(NvrtcApi.GetPTX(prog, ptxBytes), "nvrtcGetPTX");

            // Load PTX into a CUDA module
            return FromPtx(ptxBytes);
        }
        finally
        {
            NvrtcApi.DestroyProgram(ref prog);
        }
    }

    /// <summary>
    /// Load and compile a .cu file embedded as an assembly resource.
    /// </summary>
    public static CudaModule FromEmbeddedResource(string resourceName)
    {
        var assembly = Assembly.GetExecutingAssembly();
        using var stream = assembly.GetManifestResourceStream(resourceName)
            ?? throw new FileNotFoundException($"Embedded resource not found: {resourceName}");
        using var reader = new StreamReader(stream, Encoding.UTF8);
        var source = reader.ReadToEnd();
        return FromCudaSource(source, resourceName);
    }

    /// <summary>
    /// Get a kernel function handle by name (cached).
    /// </summary>
    public nint GetFunction(string name)
    {
        if (_functions.TryGetValue(name, out var cached))
            return cached;

        CudaApi.Check(
            CudaApi.ModuleGetFunction(out nint func, _handle, name),
            $"cuModuleGetFunction({name})");
        _functions[name] = func;
        return func;
    }

    public void Dispose()
    {
        if (!_disposed && _handle != 0)
        {
            CudaApi.ModuleUnload(_handle);
            _handle = 0;
            _disposed = true;
        }
    }
}
