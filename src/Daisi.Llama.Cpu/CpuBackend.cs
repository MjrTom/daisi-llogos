using System.Runtime.InteropServices;
using Daisi.Llama.Gguf;

namespace Daisi.Llama.Cpu;

/// <summary>
/// CPU compute backend with SIMD-optimized tensor operations.
/// </summary>
public sealed class CpuBackend : IComputeBackend
{
    /// <inheritdoc />
    public string Name => "CPU";

    /// <inheritdoc />
    public ITensor CreateTensor(string name, GgmlType type, ReadOnlySpan<long> dimensions)
    {
        return new CpuTensor(name, type, dimensions);
    }

    /// <inheritdoc />
    public ITensor LoadTensor(string name, GgmlType type, ReadOnlySpan<long> dimensions, ReadOnlySpan<byte> data)
    {
        return new CpuTensor(name, type, dimensions, data);
    }

    /// <inheritdoc />
    public void MatMul(ITensor output, ITensor a, ITensor b, int M, int K, int N)
    {
        var o = ((CpuTensor)output).AsFloatSpan();
        var aSpan = ((CpuTensor)a).AsFloatSpan();

        if (b.Type == GgmlType.F32)
        {
            var bSpan = ((CpuTensor)b).AsFloatSpan();
            Cpu.MatMul.Multiply(o, aSpan, bSpan, M, K, N);
        }
        else if (b.Type == GgmlType.Q8_0)
        {
            var bRaw = ((CpuTensor)b).RawData;
            Cpu.MatMul.MultiplyQ8_0(o, aSpan, bRaw, M, K, N);
        }
        else
        {
            throw new NotSupportedException($"MatMul not implemented for weight type {b.Type}.");
        }
    }

    /// <inheritdoc />
    public void RmsNorm(ITensor output, ITensor input, ITensor weight, float eps)
    {
        var o = ((CpuTensor)output).AsFloatSpan();
        var i = ((CpuTensor)input).AsFloatSpan();
        var w = ((CpuTensor)weight).AsFloatSpan();
        Cpu.RmsNorm.Apply(o, i, w, eps);
    }

    /// <inheritdoc />
    public void Softmax(ITensor output, ITensor input)
    {
        var o = ((CpuTensor)output).AsFloatSpan();
        var i = ((CpuTensor)input).AsFloatSpan();
        Cpu.Softmax.Apply(o, i);
    }

    /// <inheritdoc />
    public void SiLU(ITensor output, ITensor input)
    {
        var o = ((CpuTensor)output).AsFloatSpan();
        var i = ((CpuTensor)input).AsFloatSpan();
        Cpu.Silu.Apply(o, i);
    }

    /// <inheritdoc />
    public void RoPE(ITensor q, ITensor k, int headDim, int ropeDim, int positionOffset, float ropeTheta)
    {
        var qSpan = ((CpuTensor)q).AsFloatSpan();
        var kSpan = ((CpuTensor)k).AsFloatSpan();
        int effectiveRopeDim = ropeDim > 0 ? ropeDim : headDim;
        Cpu.Rope.Apply(qSpan, kSpan, headDim, effectiveRopeDim, positionOffset, ropeTheta);
    }

    /// <inheritdoc />
    public void ElementMul(ITensor output, ITensor a, ITensor b)
    {
        var o = ((CpuTensor)output).AsFloatSpan();
        var aSpan = ((CpuTensor)a).AsFloatSpan();
        var bSpan = ((CpuTensor)b).AsFloatSpan();
        Cpu.ElementOps.Multiply(o, aSpan, bSpan);
    }

    /// <inheritdoc />
    public void ElementAdd(ITensor output, ITensor a, ITensor b)
    {
        var o = ((CpuTensor)output).AsFloatSpan();
        var aSpan = ((CpuTensor)a).AsFloatSpan();
        var bSpan = ((CpuTensor)b).AsFloatSpan();
        Cpu.ElementOps.Add(o, aSpan, bSpan);
    }

    /// <inheritdoc />
    public void EmbeddingLookup(ITensor output, ITensor table, int tokenId)
    {
        var outSpan = ((CpuTensor)output).AsFloatSpan();
        var raw = ((CpuTensor)table).RawData;
        int hiddenDim = (int)table.Dimensions[0]; // GGUF dim[0] = row length

        switch (table.Type)
        {
            case GgmlType.F32:
            {
                var floats = MemoryMarshal.Cast<byte, float>(raw);
                floats.Slice(tokenId * hiddenDim, hiddenDim).CopyTo(outSpan);
                break;
            }
            case GgmlType.Q8_0:
            {
                int blocksPerRow = hiddenDim / 32;
                int bytesPerRow = blocksPerRow * 34; // Q8_0: 2 bytes scale + 32 bytes quants
                var rowData = raw.Slice(tokenId * bytesPerRow, bytesPerRow);
                Dequantize.DequantizeQ8_0(rowData, outSpan.Slice(0, hiddenDim));
                break;
            }
            case GgmlType.F16:
            {
                int bytesPerRow = hiddenDim * 2;
                var rowBytes = raw.Slice(tokenId * bytesPerRow, bytesPerRow);
                var halfs = MemoryMarshal.Cast<byte, Half>(rowBytes);
                for (int i = 0; i < hiddenDim; i++)
                    outSpan[i] = (float)halfs[i];
                break;
            }
            default:
                throw new NotSupportedException($"EmbeddingLookup not implemented for type {table.Type}.");
        }
    }

    /// <inheritdoc />
    public void Dispose()
    {
        // No unmanaged resources to clean up.
    }
}
