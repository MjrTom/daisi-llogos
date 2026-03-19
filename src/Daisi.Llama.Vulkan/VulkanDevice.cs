using System.Runtime.InteropServices;
using System.Text;

#pragma warning disable CS8500 // address of managed type

namespace Daisi.Llama.Vulkan;

/// <summary>
/// Manages Vulkan instance, physical device, logical device, and compute queue.
/// </summary>
internal sealed class VulkanDevice : IDisposable
{
    private nint _instance;
    private nint _device;
    private nint _queue;
    private ulong _commandPool;
    private nint _commandBuffer;
    private ulong _fence;
    private bool _disposed;

    public nint Device => _device;
    public nint Queue => _queue;
    public uint ComputeQueueFamily { get; private set; }
    public string DeviceName { get; private set; } = "";
    public uint MaxPushConstantsSize { get; private set; }
    public VkPhysicalDeviceMemoryProperties MemoryProperties { get; private set; }

    public unsafe VulkanDevice(int deviceOrdinal = 0)
    {
        // Create instance
        var appInfo = new VkApplicationInfo
        {
            sType = 0, // VK_STRUCTURE_TYPE_APPLICATION_INFO
            apiVersion = (1u << 22) | (0u << 12) | 0u, // Vulkan 1.0
        };

        var instanceInfo = new VkInstanceCreateInfo
        {
            sType = 1, // VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO
            pApplicationInfo = (nint)(&appInfo),
        };

        VulkanApi.Check(VulkanApi.CreateInstance(&instanceInfo, 0, out _instance), "vkCreateInstance");

        // Enumerate physical devices
        uint deviceCount = 0;
        VulkanApi.Check(VulkanApi.EnumeratePhysicalDevices(_instance, ref deviceCount, null), "vkEnumeratePhysicalDevices(count)");
        if (deviceCount == 0)
            throw new VulkanException("No Vulkan-capable GPU found.");

        var physDevices = stackalloc nint[(int)deviceCount];
        VulkanApi.Check(VulkanApi.EnumeratePhysicalDevices(_instance, ref deviceCount, physDevices), "vkEnumeratePhysicalDevices");

        if (deviceOrdinal >= (int)deviceCount)
            throw new VulkanException($"Device ordinal {deviceOrdinal} out of range (found {deviceCount} devices).");

        nint physDevice = physDevices[deviceOrdinal];

        // Get device properties — use heap allocation since struct is large (~832 bytes)
        nint propsPtr = (nint)NativeMemory.AllocZeroed(1024);
        try
        {
            VulkanApi.GetPhysicalDeviceProperties(physDevice, propsPtr);
            // apiVersion=4, driverVersion=4, vendorID=4, deviceID=4, deviceType=4 = 20, then deviceName[256]
            DeviceName = Encoding.UTF8.GetString(new ReadOnlySpan<byte>((byte*)propsPtr + 20, 256)).TrimEnd('\0');
            MaxPushConstantsSize = 128;
        }
        finally
        {
            NativeMemory.Free((void*)propsPtr);
        }

        // Get memory properties — also large struct, use heap
        nint memPropsPtr = (nint)NativeMemory.AllocZeroed(1024);
        try
        {
            VulkanApi.GetPhysicalDeviceMemoryProperties(physDevice, memPropsPtr);
            MemoryProperties = Marshal.PtrToStructure<VkPhysicalDeviceMemoryProperties>(memPropsPtr);
        }
        finally
        {
            NativeMemory.Free((void*)memPropsPtr);
        }

        // Find compute queue family
        uint queueFamilyCount = 0;
        VulkanApi.GetPhysicalDeviceQueueFamilyProperties(physDevice, ref queueFamilyCount, null);
        var queueFamilies = stackalloc VkQueueFamilyProperties[(int)queueFamilyCount];
        VulkanApi.GetPhysicalDeviceQueueFamilyProperties(physDevice, ref queueFamilyCount, queueFamilies);

        ComputeQueueFamily = uint.MaxValue;
        for (uint i = 0; i < queueFamilyCount; i++)
        {
            if ((queueFamilies[i].queueFlags & VkConst.QueueComputeBit) != 0)
            {
                ComputeQueueFamily = i;
                break;
            }
        }

        if (ComputeQueueFamily == uint.MaxValue)
            throw new VulkanException("No compute queue family found.");

        // Create logical device with one compute queue
        float queuePriority = 1.0f;
        var queueCreateInfo = new VkDeviceQueueCreateInfo
        {
            sType = 2,
            queueFamilyIndex = ComputeQueueFamily,
            queueCount = 1,
            pQueuePriorities = (nint)(&queuePriority),
        };

        var deviceCreateInfo = new VkDeviceCreateInfo
        {
            sType = 3,
            queueCreateInfoCount = 1,
            pQueueCreateInfos = (nint)(&queueCreateInfo),
        };

        VulkanApi.Check(VulkanApi.CreateDevice(physDevice, &deviceCreateInfo, 0, out _device), "vkCreateDevice");

        // Get queue
        VulkanApi.GetDeviceQueue(_device, ComputeQueueFamily, 0, out _queue);

        // Create command pool (with reset bit so we can reuse command buffers)
        var poolInfo = new VkCommandPoolCreateInfo
        {
            sType = 39,
            flags = VkConst.CommandPoolCreateResetCommandBufferBit,
            queueFamilyIndex = ComputeQueueFamily,
        };
        VulkanApi.Check(VulkanApi.CreateCommandPool(_device, &poolInfo, 0, out _commandPool), "vkCreateCommandPool");

        // Allocate a reusable command buffer
        var allocInfo = new VkCommandBufferAllocateInfo
        {
            sType = 40,
            commandPool = _commandPool,
            level = VkConst.CommandBufferLevelPrimary,
            commandBufferCount = 1,
        };
        nint cmdBuf;
        VulkanApi.Check(VulkanApi.AllocateCommandBuffers(_device, &allocInfo, &cmdBuf), "vkAllocateCommandBuffers");
        _commandBuffer = cmdBuf;

        // Create fence
        var fenceInfo = new VkFenceCreateInfo { sType = 8 };
        VulkanApi.Check(VulkanApi.CreateFence(_device, &fenceInfo, 0, out _fence), "vkCreateFence");
    }

    /// <summary>
    /// Find a memory type index that matches the requirements and desired properties.
    /// </summary>
    public unsafe uint FindMemoryType(uint memoryTypeBits, uint requiredProperties)
    {
        var memProps = MemoryProperties;
        var types = (VkMemoryType*)memProps.memoryTypes;
        for (uint i = 0; i < memProps.memoryTypeCount; i++)
        {
            if ((memoryTypeBits & (1u << (int)i)) != 0 &&
                (types[i].propertyFlags & requiredProperties) == requiredProperties)
                return i;
        }
        throw new VulkanException($"No suitable memory type found (bits={memoryTypeBits:X}, props={requiredProperties:X}).");
    }

    /// <summary>
    /// Record, submit, and wait for a command buffer. Used for compute dispatch and transfers.
    /// </summary>
    public unsafe void SubmitAndWait(Action<nint> recordCommands)
    {
        // Reset command buffer
        VulkanApi.Check(VulkanApi.ResetCommandBuffer(_commandBuffer, 0), "vkResetCommandBuffer");

        // Begin recording
        var beginInfo = new VkCommandBufferBeginInfo
        {
            sType = 42,
            flags = VkConst.CommandBufferUsageOneTimeSubmitBit,
        };
        VulkanApi.Check(VulkanApi.BeginCommandBuffer(_commandBuffer, &beginInfo), "vkBeginCommandBuffer");

        // Record commands
        recordCommands(_commandBuffer);

        // End recording
        VulkanApi.Check(VulkanApi.EndCommandBuffer(_commandBuffer), "vkEndCommandBuffer");

        // Submit
        nint cmdBufLocal = _commandBuffer;
        var submitInfo = new VkSubmitInfo
        {
            sType = 4,
            commandBufferCount = 1,
            pCommandBuffers = (nint)(&cmdBufLocal),
        };

        // Reset fence before submission
        ulong fenceLocal = _fence;
        VulkanApi.Check(VulkanApi.ResetFences(_device, 1, &fenceLocal), "vkResetFences");
        VulkanApi.Check(VulkanApi.QueueSubmit(_queue, 1, &submitInfo, _fence), "vkQueueSubmit");

        // Wait for completion
        VulkanApi.Check(VulkanApi.WaitForFences(_device, 1, &fenceLocal, 1, ulong.MaxValue), "vkWaitForFences");
    }

    public nint CommandBuffer => _commandBuffer;

    public void Dispose()
    {
        if (!_disposed)
        {
            if (_fence != 0) VulkanApi.DestroyFence(_device, _fence, 0);
            if (_commandPool != 0) VulkanApi.DestroyCommandPool(_device, _commandPool, 0);
            if (_device != 0) VulkanApi.DestroyDevice(_device, 0);
            if (_instance != 0) VulkanApi.DestroyInstance(_instance, 0);
            _disposed = true;
        }
    }
}
