namespace Daisi.Llogos.Vulkan;

/// <summary>
/// Wraps a Vulkan buffer + device memory allocation. Supports device-local and host-visible memory.
/// </summary>
internal sealed class VulkanBuffer : IDisposable
{
    private readonly VulkanDevice _vkDevice;
    private ulong _buffer;
    private ulong _memory;
    private bool _disposed;

    public ulong ByteSize { get; }
    public ulong Buffer => _buffer;
    public ulong Memory => _memory;
    public bool IsHostVisible { get; }

    public unsafe VulkanBuffer(VulkanDevice vkDevice, ulong byteSize, bool hostVisible, bool transferSrc = false, bool transferDst = true)
    {
        _vkDevice = vkDevice;
        ByteSize = byteSize;
        IsHostVisible = hostVisible;

        uint usage = VkConst.BufferUsageStorageBufferBit;
        if (transferSrc) usage |= VkConst.BufferUsageTransferSrcBit;
        if (transferDst) usage |= VkConst.BufferUsageTransferDstBit;

        // Create buffer
        var bufferInfo = new VkBufferCreateInfo
        {
            sType = 12,
            size = byteSize,
            usage = usage,
            sharingMode = 0, // VK_SHARING_MODE_EXCLUSIVE
        };
        VulkanApi.Check(VulkanApi.CreateBuffer(vkDevice.Device, &bufferInfo, 0, out _buffer), "vkCreateBuffer");

        // Get memory requirements
        VkMemoryRequirements memReqs;
        VulkanApi.GetBufferMemoryRequirements(vkDevice.Device, _buffer, &memReqs);

        // Find memory type
        uint memProps = hostVisible
            ? VkConst.MemoryPropertyHostVisibleBit | VkConst.MemoryPropertyHostCoherentBit
            : VkConst.MemoryPropertyDeviceLocalBit;
        uint memTypeIdx = vkDevice.FindMemoryType(memReqs.memoryTypeBits, memProps);

        // Allocate memory
        var allocInfo = new VkMemoryAllocateInfo
        {
            sType = 5,
            allocationSize = memReqs.size,
            memoryTypeIndex = memTypeIdx,
        };
        VulkanApi.Check(VulkanApi.AllocateMemory(vkDevice.Device, &allocInfo, 0, out _memory), "vkAllocateMemory");

        // Bind memory to buffer
        VulkanApi.Check(VulkanApi.BindBufferMemory(vkDevice.Device, _buffer, _memory, 0), "vkBindBufferMemory");
    }

    /// <summary>
    /// Upload data from host to this buffer. Buffer must be host-visible, or use staging.
    /// </summary>
    public unsafe void Upload(ReadOnlySpan<byte> data)
    {
        if (!IsHostVisible)
            throw new VulkanException("Cannot upload directly to device-local buffer. Use staging.");
        if ((ulong)data.Length > ByteSize)
            throw new ArgumentException($"Data length {data.Length} exceeds buffer size {ByteSize}");

        VulkanApi.Check(VulkanApi.MapMemory(_vkDevice.Device, _memory, 0, (ulong)data.Length, 0, out nint mapped), "vkMapMemory");
        data.CopyTo(new Span<byte>((void*)mapped, data.Length));
        VulkanApi.UnmapMemory(_vkDevice.Device, _memory);
    }

    /// <summary>
    /// Download data from this buffer to host. Buffer must be host-visible.
    /// </summary>
    public unsafe void Download(Span<byte> destination)
    {
        if (!IsHostVisible)
            throw new VulkanException("Cannot download directly from device-local buffer. Use staging.");
        if ((ulong)destination.Length > ByteSize)
            throw new ArgumentException($"Destination length {destination.Length} exceeds buffer size {ByteSize}");

        VulkanApi.Check(VulkanApi.MapMemory(_vkDevice.Device, _memory, 0, (ulong)destination.Length, 0, out nint mapped), "vkMapMemory");
        new Span<byte>((void*)mapped, destination.Length).CopyTo(destination);
        VulkanApi.UnmapMemory(_vkDevice.Device, _memory);
    }

    /// <summary>
    /// Download float data from this buffer.
    /// </summary>
    public unsafe void Download(Span<float> destination)
    {
        int byteLen = destination.Length * sizeof(float);
        if (!IsHostVisible)
            throw new VulkanException("Cannot download directly from device-local buffer.");

        VulkanApi.Check(VulkanApi.MapMemory(_vkDevice.Device, _memory, 0, (ulong)byteLen, 0, out nint mapped), "vkMapMemory");
        new Span<float>((void*)mapped, destination.Length).CopyTo(destination);
        VulkanApi.UnmapMemory(_vkDevice.Device, _memory);
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            if (_buffer != 0) VulkanApi.DestroyBuffer(_vkDevice.Device, _buffer, 0);
            if (_memory != 0) VulkanApi.FreeMemory(_vkDevice.Device, _memory, 0);
            _disposed = true;
        }
    }
}
