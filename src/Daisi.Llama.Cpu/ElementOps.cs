using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace Daisi.Llama.Cpu;

/// <summary>
/// Element-wise tensor operations: add and multiply.
/// </summary>
internal static class ElementOps
{
    public static void Multiply(Span<float> output, ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        int n = a.Length;
        if (Avx2.IsSupported && n >= 8)
        {
            ref float aRef = ref MemoryMarshal.GetReference(a);
            ref float bRef = ref MemoryMarshal.GetReference(b);
            ref float oRef = ref MemoryMarshal.GetReference(output);
            int i = 0;
            for (; i + 8 <= n; i += 8)
            {
                var va = Vector256.LoadUnsafe(ref Unsafe.Add(ref aRef, i));
                var vb = Vector256.LoadUnsafe(ref Unsafe.Add(ref bRef, i));
                Avx.Multiply(va, vb).StoreUnsafe(ref Unsafe.Add(ref oRef, i));
            }
            for (; i < n; i++)
                output[i] = a[i] * b[i];
        }
        else
        {
            for (int i = 0; i < n; i++)
                output[i] = a[i] * b[i];
        }
    }

    public static void Add(Span<float> output, ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        int n = a.Length;
        if (Avx2.IsSupported && n >= 8)
        {
            ref float aRef = ref MemoryMarshal.GetReference(a);
            ref float bRef = ref MemoryMarshal.GetReference(b);
            ref float oRef = ref MemoryMarshal.GetReference(output);
            int i = 0;
            for (; i + 8 <= n; i += 8)
            {
                var va = Vector256.LoadUnsafe(ref Unsafe.Add(ref aRef, i));
                var vb = Vector256.LoadUnsafe(ref Unsafe.Add(ref bRef, i));
                Avx.Add(va, vb).StoreUnsafe(ref Unsafe.Add(ref oRef, i));
            }
            for (; i < n; i++)
                output[i] = a[i] + b[i];
        }
        else
        {
            for (int i = 0; i < n; i++)
                output[i] = a[i] + b[i];
        }
    }
}
