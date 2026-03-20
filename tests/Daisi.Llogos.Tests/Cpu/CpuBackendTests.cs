using System.Runtime.InteropServices;
using Daisi.Llogos.Cpu;
using Daisi.Llogos.Gguf;

namespace Daisi.Llogos.Tests.Cpu;

public class CpuBackendTests
{
    [Fact]
    public void Name_IsCpu()
    {
        using var backend = new CpuBackend();
        Assert.Equal("CPU", backend.Name);
    }

    [Fact]
    public void CreateTensor_CorrectProperties()
    {
        using var backend = new CpuBackend();
        using var tensor = backend.CreateTensor("test", GgmlType.F32, [2, 3]);

        Assert.Equal("test", tensor.Name);
        Assert.Equal(GgmlType.F32, tensor.Type);
        Assert.Equal(6, tensor.ElementCount);
        Assert.Equal(24, tensor.ByteSize); // 6 * 4 bytes
        var dims = tensor.Dimensions;
        Assert.Equal(2, dims.Length);
        Assert.Equal(2, dims[0]);
        Assert.Equal(3, dims[1]);
    }

    [Fact]
    public void CreateTensor_F32_CopyFrom_RoundTrip()
    {
        using var backend = new CpuBackend();
        using var tensor = backend.CreateTensor("test", GgmlType.F32, [4]);

        float[] values = [1.0f, 2.0f, 3.0f, 4.0f];
        var bytes = MemoryMarshal.AsBytes(values.AsSpan());
        tensor.CopyFrom(bytes);

        var output = new float[4];
        tensor.DequantizeTo(output);

        Assert.Equal(values, output);
    }

    [Fact]
    public void LoadTensor_Q8_0_CorrectProperties()
    {
        using var backend = new CpuBackend();

        // 32 elements = 1 Q8_0 block = 34 bytes
        var data = new byte[34];
        BitConverter.TryWriteBytes(data, (Half)1.0f);
        for (int i = 0; i < 32; i++) data[2 + i] = (byte)(sbyte)i;

        using var tensor = backend.LoadTensor("weight", GgmlType.Q8_0, [32], data);

        Assert.Equal("weight", tensor.Name);
        Assert.Equal(GgmlType.Q8_0, tensor.Type);
        Assert.Equal(32, tensor.ElementCount);
        Assert.Equal(34, tensor.ByteSize);
    }

    [Fact]
    public void LoadTensor_Q8_0_DequantizeRoundTrip()
    {
        using var backend = new CpuBackend();

        var data = new byte[34];
        BitConverter.TryWriteBytes(data, (Half)1.5f);
        for (int i = 0; i < 32; i++) data[2 + i] = (byte)(sbyte)(i - 16);

        using var tensor = backend.LoadTensor("w", GgmlType.Q8_0, [32], data);

        var output = new float[32];
        tensor.DequantizeTo(output);

        for (int i = 0; i < 32; i++)
            Assert.Equal(1.5f * (i - 16), output[i], 0.01f);
    }

    [Fact]
    public void CopyFrom_WrongSize_Throws()
    {
        using var backend = new CpuBackend();
        using var tensor = backend.CreateTensor("test", GgmlType.F32, [4]);

        Assert.Throws<ArgumentException>(() => tensor.CopyFrom(new byte[8]));
    }

    [Fact]
    public void DequantizeTo_DestinationTooSmall_Throws()
    {
        using var backend = new CpuBackend();
        using var tensor = backend.CreateTensor("test", GgmlType.F32, [4]);

        Assert.Throws<ArgumentException>(() => tensor.DequantizeTo(new float[2]));
    }

    [Fact]
    public void DequantizeTo_F16_Works()
    {
        using var backend = new CpuBackend();
        var data = new byte[2];
        BitConverter.TryWriteBytes(data, (Half)1.5f);
        using var tensor = backend.LoadTensor("test", GgmlType.F16, [1], data);

        var result = new float[1];
        tensor.DequantizeTo(result);
        Assert.Equal(1.5f, result[0], 0.01f);
    }
}

public class CpuTensorIntegrationTests
{
    [Fact]
    public void LoadFromGguf_Q8_0_Dequantize()
    {
        if (!TestConstants.ModelExists) return;

        using var stream = File.OpenRead(TestConstants.Qwen35_08B_Q8_0);
        var gguf = Daisi.Llogos.Gguf.GgufFile.Read(stream);

        // Find a Q8_0 tensor
        var tensorInfo = gguf.Tensors.FirstOrDefault(t => t.Type == GgmlType.Q8_0);
        if (tensorInfo == null) return;

        var rawData = gguf.ReadTensorData(stream, tensorInfo);
        using var backend = new CpuBackend();

        var dims = tensorInfo.Dimensions.Select(d => (long)d).ToArray();
        using var tensor = backend.LoadTensor(tensorInfo.Name, tensorInfo.Type, dims, rawData);

        Assert.Equal(tensorInfo.Name, tensor.Name);
        Assert.Equal(GgmlType.Q8_0, tensor.Type);

        // Dequantize and verify non-zero output
        var output = new float[tensor.ElementCount];
        tensor.DequantizeTo(output);

        bool hasNonZero = output.Any(f => f != 0.0f);
        Assert.True(hasNonZero, "Dequantized tensor should have non-zero values.");

        // Verify no NaN/Inf
        bool hasInvalid = output.Any(f => float.IsNaN(f) || float.IsInfinity(f));
        Assert.False(hasInvalid, "Dequantized tensor should not have NaN or Infinity values.");
    }
}
