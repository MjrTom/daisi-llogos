using System.Runtime.InteropServices;

namespace Daisi.Llogos.Vulkan;

/// <summary>
/// Raw P/Invoke bindings to the Vulkan API (vulkan-1.dll).
/// </summary>
internal static partial class VulkanApi
{
    private const string Lib = "vulkan-1";

    // ── Instance ──────────────────────────────────────────────────────────────

    [LibraryImport(Lib, EntryPoint = "vkCreateInstance")]
    internal static unsafe partial VkResult CreateInstance(
        VkInstanceCreateInfo* pCreateInfo, nint pAllocator, out nint pInstance);

    [LibraryImport(Lib, EntryPoint = "vkDestroyInstance")]
    internal static partial void DestroyInstance(nint instance, nint pAllocator);

    // ── Physical Device ───────────────────────────────────────────────────────

    [LibraryImport(Lib, EntryPoint = "vkEnumeratePhysicalDevices")]
    internal static unsafe partial VkResult EnumeratePhysicalDevices(
        nint instance, ref uint pPhysicalDeviceCount, nint* pPhysicalDevices);

    [LibraryImport(Lib, EntryPoint = "vkGetPhysicalDeviceProperties")]
    internal static partial void GetPhysicalDeviceProperties(nint physicalDevice, nint pProperties);

    [LibraryImport(Lib, EntryPoint = "vkGetPhysicalDeviceMemoryProperties")]
    internal static partial void GetPhysicalDeviceMemoryProperties(nint physicalDevice, nint pMemoryProperties);

    [LibraryImport(Lib, EntryPoint = "vkGetPhysicalDeviceQueueFamilyProperties")]
    internal static unsafe partial void GetPhysicalDeviceQueueFamilyProperties(
        nint physicalDevice, ref uint pQueueFamilyPropertyCount,
        VkQueueFamilyProperties* pQueueFamilyProperties);

    // ── Logical Device ────────────────────────────────────────────────────────

    [LibraryImport(Lib, EntryPoint = "vkCreateDevice")]
    internal static unsafe partial VkResult CreateDevice(
        nint physicalDevice, VkDeviceCreateInfo* pCreateInfo, nint pAllocator, out nint pDevice);

    [LibraryImport(Lib, EntryPoint = "vkDestroyDevice")]
    internal static partial void DestroyDevice(nint device, nint pAllocator);

    [LibraryImport(Lib, EntryPoint = "vkGetDeviceQueue")]
    internal static partial void GetDeviceQueue(nint device, uint queueFamilyIndex, uint queueIndex, out nint pQueue);

    [LibraryImport(Lib, EntryPoint = "vkDeviceWaitIdle")]
    internal static partial VkResult DeviceWaitIdle(nint device);

    // ── Memory ────────────────────────────────────────────────────────────────

    [LibraryImport(Lib, EntryPoint = "vkAllocateMemory")]
    internal static unsafe partial VkResult AllocateMemory(
        nint device, VkMemoryAllocateInfo* pAllocateInfo, nint pAllocator, out ulong pMemory);

    [LibraryImport(Lib, EntryPoint = "vkFreeMemory")]
    internal static partial void FreeMemory(nint device, ulong memory, nint pAllocator);

    [LibraryImport(Lib, EntryPoint = "vkMapMemory")]
    internal static unsafe partial VkResult MapMemory(
        nint device, ulong memory, ulong offset, ulong size, uint flags, out nint ppData);

    [LibraryImport(Lib, EntryPoint = "vkUnmapMemory")]
    internal static partial void UnmapMemory(nint device, ulong memory);

    // ── Buffer ────────────────────────────────────────────────────────────────

    [LibraryImport(Lib, EntryPoint = "vkCreateBuffer")]
    internal static unsafe partial VkResult CreateBuffer(
        nint device, VkBufferCreateInfo* pCreateInfo, nint pAllocator, out ulong pBuffer);

    [LibraryImport(Lib, EntryPoint = "vkDestroyBuffer")]
    internal static partial void DestroyBuffer(nint device, ulong buffer, nint pAllocator);

    [LibraryImport(Lib, EntryPoint = "vkGetBufferMemoryRequirements")]
    internal static unsafe partial void GetBufferMemoryRequirements(
        nint device, ulong buffer, VkMemoryRequirements* pMemoryRequirements);

    [LibraryImport(Lib, EntryPoint = "vkBindBufferMemory")]
    internal static partial VkResult BindBufferMemory(
        nint device, ulong buffer, ulong memory, ulong memoryOffset);

    // ── Descriptor Set ────────────────────────────────────────────────────────

    [LibraryImport(Lib, EntryPoint = "vkCreateDescriptorSetLayout")]
    internal static unsafe partial VkResult CreateDescriptorSetLayout(
        nint device, VkDescriptorSetLayoutCreateInfo* pCreateInfo, nint pAllocator, out ulong pSetLayout);

    [LibraryImport(Lib, EntryPoint = "vkDestroyDescriptorSetLayout")]
    internal static partial void DestroyDescriptorSetLayout(nint device, ulong descriptorSetLayout, nint pAllocator);

    [LibraryImport(Lib, EntryPoint = "vkCreateDescriptorPool")]
    internal static unsafe partial VkResult CreateDescriptorPool(
        nint device, VkDescriptorPoolCreateInfo* pCreateInfo, nint pAllocator, out ulong pDescriptorPool);

    [LibraryImport(Lib, EntryPoint = "vkDestroyDescriptorPool")]
    internal static partial void DestroyDescriptorPool(nint device, ulong descriptorPool, nint pAllocator);

    [LibraryImport(Lib, EntryPoint = "vkResetDescriptorPool")]
    internal static partial VkResult ResetDescriptorPool(nint device, ulong descriptorPool, uint flags);

    [LibraryImport(Lib, EntryPoint = "vkAllocateDescriptorSets")]
    internal static unsafe partial VkResult AllocateDescriptorSets(
        nint device, VkDescriptorSetAllocateInfo* pAllocateInfo, ulong* pDescriptorSets);

    [LibraryImport(Lib, EntryPoint = "vkUpdateDescriptorSets")]
    internal static unsafe partial void UpdateDescriptorSets(
        nint device, uint descriptorWriteCount, VkWriteDescriptorSet* pDescriptorWrites,
        uint descriptorCopyCount, nint pDescriptorCopies);

    // ── Pipeline ──────────────────────────────────────────────────────────────

    [LibraryImport(Lib, EntryPoint = "vkCreateShaderModule")]
    internal static unsafe partial VkResult CreateShaderModule(
        nint device, VkShaderModuleCreateInfo* pCreateInfo, nint pAllocator, out ulong pShaderModule);

    [LibraryImport(Lib, EntryPoint = "vkDestroyShaderModule")]
    internal static partial void DestroyShaderModule(nint device, ulong shaderModule, nint pAllocator);

    [LibraryImport(Lib, EntryPoint = "vkCreatePipelineLayout")]
    internal static unsafe partial VkResult CreatePipelineLayout(
        nint device, VkPipelineLayoutCreateInfo* pCreateInfo, nint pAllocator, out ulong pPipelineLayout);

    [LibraryImport(Lib, EntryPoint = "vkDestroyPipelineLayout")]
    internal static partial void DestroyPipelineLayout(nint device, ulong pipelineLayout, nint pAllocator);

    [LibraryImport(Lib, EntryPoint = "vkCreateComputePipelines")]
    internal static unsafe partial VkResult CreateComputePipelines(
        nint device, ulong pipelineCache, uint createInfoCount,
        VkComputePipelineCreateInfo* pCreateInfos, nint pAllocator, ulong* pPipelines);

    [LibraryImport(Lib, EntryPoint = "vkDestroyPipeline")]
    internal static partial void DestroyPipeline(nint device, ulong pipeline, nint pAllocator);

    // ── Command Buffer ────────────────────────────────────────────────────────

    [LibraryImport(Lib, EntryPoint = "vkCreateCommandPool")]
    internal static unsafe partial VkResult CreateCommandPool(
        nint device, VkCommandPoolCreateInfo* pCreateInfo, nint pAllocator, out ulong pCommandPool);

    [LibraryImport(Lib, EntryPoint = "vkDestroyCommandPool")]
    internal static partial void DestroyCommandPool(nint device, ulong commandPool, nint pAllocator);

    [LibraryImport(Lib, EntryPoint = "vkAllocateCommandBuffers")]
    internal static unsafe partial VkResult AllocateCommandBuffers(
        nint device, VkCommandBufferAllocateInfo* pAllocateInfo, nint* pCommandBuffers);

    [LibraryImport(Lib, EntryPoint = "vkBeginCommandBuffer")]
    internal static unsafe partial VkResult BeginCommandBuffer(
        nint commandBuffer, VkCommandBufferBeginInfo* pBeginInfo);

    [LibraryImport(Lib, EntryPoint = "vkEndCommandBuffer")]
    internal static partial VkResult EndCommandBuffer(nint commandBuffer);

    [LibraryImport(Lib, EntryPoint = "vkResetCommandBuffer")]
    internal static partial VkResult ResetCommandBuffer(nint commandBuffer, uint flags);

    [LibraryImport(Lib, EntryPoint = "vkCmdBindPipeline")]
    internal static partial void CmdBindPipeline(nint commandBuffer, uint pipelineBindPoint, ulong pipeline);

    [LibraryImport(Lib, EntryPoint = "vkCmdBindDescriptorSets")]
    internal static unsafe partial void CmdBindDescriptorSets(
        nint commandBuffer, uint pipelineBindPoint, ulong layout,
        uint firstSet, uint descriptorSetCount, ulong* pDescriptorSets,
        uint dynamicOffsetCount, uint* pDynamicOffsets);

    [LibraryImport(Lib, EntryPoint = "vkCmdDispatch")]
    internal static partial void CmdDispatch(nint commandBuffer, uint groupCountX, uint groupCountY, uint groupCountZ);

    [LibraryImport(Lib, EntryPoint = "vkCmdPushConstants")]
    internal static unsafe partial void CmdPushConstants(
        nint commandBuffer, ulong layout, uint stageFlags, uint offset, uint size, void* pValues);

    [LibraryImport(Lib, EntryPoint = "vkCmdPipelineBarrier")]
    internal static unsafe partial void CmdPipelineBarrier(
        nint commandBuffer, uint srcStageMask, uint dstStageMask, uint dependencyFlags,
        uint memoryBarrierCount, VkMemoryBarrier* pMemoryBarriers,
        uint bufferMemoryBarrierCount, nint pBufferMemoryBarriers,
        uint imageMemoryBarrierCount, nint pImageMemoryBarriers);

    [LibraryImport(Lib, EntryPoint = "vkCmdCopyBuffer")]
    internal static unsafe partial void CmdCopyBuffer(
        nint commandBuffer, ulong srcBuffer, ulong dstBuffer,
        uint regionCount, VkBufferCopy* pRegions);

    [LibraryImport(Lib, EntryPoint = "vkCmdFillBuffer")]
    internal static partial void CmdFillBuffer(
        nint commandBuffer, ulong dstBuffer, ulong dstOffset, ulong size, uint data);

    // ── Synchronization ───────────────────────────────────────────────────────

    [LibraryImport(Lib, EntryPoint = "vkCreateFence")]
    internal static unsafe partial VkResult CreateFence(
        nint device, VkFenceCreateInfo* pCreateInfo, nint pAllocator, out ulong pFence);

    [LibraryImport(Lib, EntryPoint = "vkDestroyFence")]
    internal static partial void DestroyFence(nint device, ulong fence, nint pAllocator);

    [LibraryImport(Lib, EntryPoint = "vkWaitForFences")]
    internal static unsafe partial VkResult WaitForFences(
        nint device, uint fenceCount, ulong* pFences, uint waitAll, ulong timeout);

    [LibraryImport(Lib, EntryPoint = "vkResetFences")]
    internal static unsafe partial VkResult ResetFences(nint device, uint fenceCount, ulong* pFences);

    // ── Queue ─────────────────────────────────────────────────────────────────

    [LibraryImport(Lib, EntryPoint = "vkQueueSubmit")]
    internal static unsafe partial VkResult QueueSubmit(
        nint queue, uint submitCount, VkSubmitInfo* pSubmits, ulong fence);

    [LibraryImport(Lib, EntryPoint = "vkQueueWaitIdle")]
    internal static partial VkResult QueueWaitIdle(nint queue);

    // ── Buffer Device Address ──────────────────────────────────────────────────

    [LibraryImport(Lib, EntryPoint = "vkGetBufferDeviceAddress")]
    internal static unsafe partial ulong GetBufferDeviceAddress(nint device, VkBufferDeviceAddressInfo* pInfo);

    // ── Helpers ───────────────────────────────────────────────────────────────

    internal static void Check(VkResult result, string operation)
    {
        if (result != VkResult.Success)
            throw new VulkanException($"Vulkan {operation} failed: {result} ({(int)result})");
    }
}

internal enum VkResult
{
    Success = 0,
    NotReady = 1,
    Timeout = 2,
    ErrorOutOfHostMemory = -1,
    ErrorOutOfDeviceMemory = -2,
    ErrorInitializationFailed = -3,
    ErrorDeviceLost = -4,
    ErrorMemoryMapFailed = -5,
    ErrorLayerNotPresent = -6,
    ErrorExtensionNotPresent = -7,
    ErrorFeatureNotPresent = -8,
}

public sealed class VulkanException : Exception
{
    public VulkanException(string message) : base(message) { }
}
