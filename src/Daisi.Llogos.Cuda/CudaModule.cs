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
    public static CudaModule FromCudaSource(string cudaSource, string? name = null, string[]? extraOptions = null)
    {
        // Create NVRTC program
        NvrtcApi.Check(
            NvrtcApi.CreateProgram(out nint prog, cudaSource, name, 0, 0, 0),
            "nvrtcCreateProgram");

        try
        {
            // Compile with fast math + architecture-specific options
            var optionsList = new List<string> { "--use_fast_math" };
            if (extraOptions != null) optionsList.AddRange(extraOptions);
            var options = optionsList.ToArray();
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
    public static CudaModule FromEmbeddedResource(string resourceName, string[]? extraOptions = null)
    {
        var assembly = Assembly.GetExecutingAssembly();
        using var stream = assembly.GetManifestResourceStream(resourceName)
            ?? throw new FileNotFoundException($"Embedded resource not found: {resourceName}");
        using var reader = new StreamReader(stream, Encoding.UTF8);
        var source = reader.ReadToEnd();

        // PTX cache: hash source + options, store in temp directory
        var cacheKey = ComputeCacheKey(source, extraOptions);
        var cacheDir = Path.Combine(Path.GetTempPath(), "daisi-llogos-ptx-cache");
        var cachePath = Path.Combine(cacheDir, $"{cacheKey}.ptx");

        if (File.Exists(cachePath))
        {
            try
            {
                var cachedPtx = File.ReadAllBytes(cachePath);
                return FromPtx(cachedPtx);
            }
            catch { /* cache miss — recompile */ }
        }

        var module = FromCudaSource(source, resourceName, extraOptions);

        // Save compiled PTX to cache (best-effort)
        try
        {
            Directory.CreateDirectory(cacheDir);
            // Re-compile to get PTX bytes for caching
            NvrtcApi.Check(NvrtcApi.CreateProgram(out nint prog, source, resourceName, 0, 0, 0), "nvrtcCreateProgram");
            try
            {
                var optionsList = new List<string> { "--use_fast_math" };
                if (extraOptions != null) optionsList.AddRange(extraOptions);
                var options = optionsList.ToArray();
                if (NvrtcApi.CompileProgram(prog, options.Length, options) == NvrtcResult.Success)
                {
                    NvrtcApi.GetPTXSize(prog, out nuint ptxSize);
                    var ptxBytes = new byte[(int)ptxSize];
                    NvrtcApi.GetPTX(prog, ptxBytes);
                    File.WriteAllBytes(cachePath, ptxBytes);
                }
            }
            finally { NvrtcApi.DestroyProgram(ref prog); }
        }
        catch { /* cache write failure is non-fatal */ }

        return module;
    }

    private static string ComputeCacheKey(string source, string[]? options)
    {
        using var sha = System.Security.Cryptography.SHA256.Create();
        var input = source + "|" + string.Join(",", options ?? []);
        var hash = sha.ComputeHash(Encoding.UTF8.GetBytes(input));
        return Convert.ToHexString(hash)[..32];
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
