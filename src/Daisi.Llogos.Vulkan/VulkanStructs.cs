using System.Runtime.InteropServices;

namespace Daisi.Llogos.Vulkan;

// ── Enums / Constants ───────────────────────────────────────────────────────

internal static class VkConst
{
    public const uint QueueComputeBit = 0x00000002;
    public const uint QueueTransferBit = 0x00000004;
    public const uint BufferUsageStorageBufferBit = 0x00000020;
    public const uint BufferUsageTransferSrcBit = 0x00000001;
    public const uint BufferUsageTransferDstBit = 0x00000002;
    public const uint MemoryPropertyDeviceLocalBit = 0x00000001;
    public const uint MemoryPropertyHostVisibleBit = 0x00000002;
    public const uint MemoryPropertyHostCoherentBit = 0x00000004;
    public const uint ShaderStageComputeBit = 0x00000020;
    public const uint DescriptorTypeStorageBuffer = 7;
    public const uint PipelineBindPointCompute = 1;
    public const uint CommandBufferLevelPrimary = 0;
    public const uint CommandBufferUsageOneTimeSubmitBit = 0x00000001;
    public const uint CommandPoolCreateResetCommandBufferBit = 0x00000002;
    public const uint PipelineStageFlagComputeShaderBit = 0x00000800;
    public const uint PipelineStageFlagTransferBit = 0x00001000;
    public const uint AccessShaderReadBit = 0x00000020;
    public const uint AccessShaderWriteBit = 0x00000040;
    public const uint AccessTransferReadBit = 0x00000800;
    public const uint AccessTransferWriteBit = 0x00001000;
    public const ulong WholeSize = ulong.MaxValue;
}

// ── Vulkan Structures ───────────────────────────────────────────────────────

[StructLayout(LayoutKind.Sequential)]
internal struct VkApplicationInfo
{
    public uint sType; // VK_STRUCTURE_TYPE_APPLICATION_INFO = 0
    public nint pNext;
    public nint pApplicationName;
    public uint applicationVersion;
    public nint pEngineName;
    public uint engineVersion;
    public uint apiVersion;
}

[StructLayout(LayoutKind.Sequential)]
internal struct VkInstanceCreateInfo
{
    public uint sType; // VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO = 1
    public nint pNext;
    public uint flags;
    public nint pApplicationInfo;
    public uint enabledLayerCount;
    public nint ppEnabledLayerNames;
    public uint enabledExtensionCount;
    public nint ppEnabledExtensionNames;
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct VkPhysicalDeviceProperties
{
    public uint apiVersion;
    public uint driverVersion;
    public uint vendorID;
    public uint deviceID;
    public uint deviceType; // VkPhysicalDeviceType
    public fixed byte deviceName[256];
    public fixed byte pipelineCacheUUID[16];
    public VkPhysicalDeviceLimits limits;
    public VkPhysicalDeviceSparseProperties sparseProperties;
}

[StructLayout(LayoutKind.Sequential)]
internal struct VkPhysicalDeviceLimits
{
    // Only the fields we need — pad the rest
    public uint maxImageDimension1D;
    public uint maxImageDimension2D;
    public uint maxImageDimension3D;
    public uint maxImageDimensionCube;
    public uint maxImageArrayLayers;
    public uint maxTexelBufferElements;
    public uint maxUniformBufferRange;
    public uint maxStorageBufferRange;
    public uint maxPushConstantsSize;
    public uint maxMemoryAllocationCount;
    public uint maxSamplerAllocationCount;
    public ulong bufferImageGranularity;
    public ulong sparseAddressSpaceSize;
    public uint maxBoundDescriptorSets;
    public uint maxPerStageDescriptorSamplers;
    public uint maxPerStageDescriptorUniformBuffers;
    public uint maxPerStageDescriptorStorageBuffers;
    public uint maxPerStageDescriptorSampledImages;
    public uint maxPerStageDescriptorStorageImages;
    public uint maxPerStageDescriptorInputAttachments;
    public uint maxPerStageResources;
    public uint maxDescriptorSetSamplers;
    public uint maxDescriptorSetUniformBuffers;
    public uint maxDescriptorSetUniformBuffersDynamic;
    public uint maxDescriptorSetStorageBuffers;
    public uint maxDescriptorSetStorageBuffersDynamic;
    public uint maxDescriptorSetSampledImages;
    public uint maxDescriptorSetStorageImages;
    public uint maxDescriptorSetInputAttachments;
    public uint maxVertexInputAttributes;
    public uint maxVertexInputBindings;
    public uint maxVertexInputAttributeOffset;
    public uint maxVertexInputBindingStride;
    public uint maxVertexOutputComponents;
    public uint maxTessellationGenerationLevel;
    public uint maxTessellationPatchSize;
    public uint maxTessellationControlPerVertexInputComponents;
    public uint maxTessellationControlPerVertexOutputComponents;
    public uint maxTessellationControlPerPatchOutputComponents;
    public uint maxTessellationControlTotalOutputComponents;
    public uint maxTessellationEvaluationInputComponents;
    public uint maxTessellationEvaluationOutputComponents;
    public uint maxGeometryShaderInvocations;
    public uint maxGeometryInputComponents;
    public uint maxGeometryOutputComponents;
    public uint maxGeometryOutputVertices;
    public uint maxGeometryTotalOutputComponents;
    public uint maxFragmentInputComponents;
    public uint maxFragmentOutputAttachments;
    public uint maxFragmentDualSrcAttachments;
    public uint maxFragmentCombinedOutputResources;
    public uint maxComputeSharedMemorySize;
    public uint maxComputeWorkGroupCountX;
    public uint maxComputeWorkGroupCountY;
    public uint maxComputeWorkGroupCountZ;
    public uint maxComputeWorkGroupInvocations;
    public uint maxComputeWorkGroupSizeX;
    public uint maxComputeWorkGroupSizeY;
    public uint maxComputeWorkGroupSizeZ;
    // Remaining fields (many more) — we pad with raw bytes
    // The full VkPhysicalDeviceLimits is 504 bytes on 64-bit
    private unsafe fixed byte _padding[244]; // Fill remaining to reach 504 bytes total
}

[StructLayout(LayoutKind.Sequential)]
internal struct VkPhysicalDeviceSparseProperties
{
    public uint residencyStandard2DBlockShape;
    public uint residencyStandard2DMultisampleBlockShape;
    public uint residencyStandard3DBlockShape;
    public uint residencyAlignedMipSize;
    public uint residencyNonResidentStrict;
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct VkPhysicalDeviceMemoryProperties
{
    public uint memoryTypeCount;
    public fixed byte memoryTypes[32 * 8]; // VK_MAX_MEMORY_TYPES (32) * sizeof(VkMemoryType)
    public uint memoryHeapCount;
    public fixed byte memoryHeaps[16 * 16]; // VK_MAX_MEMORY_HEAPS (16) * sizeof(VkMemoryHeap)
}

[StructLayout(LayoutKind.Sequential)]
internal struct VkMemoryType
{
    public uint propertyFlags;
    public uint heapIndex;
}

[StructLayout(LayoutKind.Sequential)]
internal struct VkMemoryHeap
{
    public ulong size;
    public uint flags;
    private uint _padding;
}

[StructLayout(LayoutKind.Sequential)]
internal struct VkQueueFamilyProperties
{
    public uint queueFlags;
    public uint queueCount;
    public uint timestampValidBits;
    public uint minImageTransferGranularityWidth;
    public uint minImageTransferGranularityHeight;
    public uint minImageTransferGranularityDepth;
}

[StructLayout(LayoutKind.Sequential)]
internal struct VkDeviceQueueCreateInfo
{
    public uint sType; // VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO = 2
    public nint pNext;
    public uint flags;
    public uint queueFamilyIndex;
    public uint queueCount;
    public nint pQueuePriorities;
}

[StructLayout(LayoutKind.Sequential)]
internal struct VkDeviceCreateInfo
{
    public uint sType; // VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO = 3
    public nint pNext;
    public uint flags;
    public uint queueCreateInfoCount;
    public nint pQueueCreateInfos;
    public uint enabledLayerCount;
    public nint ppEnabledLayerNames;
    public uint enabledExtensionCount;
    public nint ppEnabledExtensionNames;
    public nint pEnabledFeatures;
}

[StructLayout(LayoutKind.Sequential)]
internal struct VkMemoryAllocateInfo
{
    public uint sType; // VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO = 5
    public nint pNext;
    public ulong allocationSize;
    public uint memoryTypeIndex;
}

[StructLayout(LayoutKind.Sequential)]
internal struct VkMemoryRequirements
{
    public ulong size;
    public ulong alignment;
    public uint memoryTypeBits;
}

[StructLayout(LayoutKind.Sequential)]
internal struct VkBufferCreateInfo
{
    public uint sType; // VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO = 12
    public nint pNext;
    public uint flags;
    public ulong size;
    public uint usage;
    public uint sharingMode;
    public uint queueFamilyIndexCount;
    public nint pQueueFamilyIndices;
}

[StructLayout(LayoutKind.Sequential)]
internal struct VkShaderModuleCreateInfo
{
    public uint sType; // VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO = 16
    public nint pNext;
    public uint flags;
    public nuint codeSize;
    public nint pCode;
}

[StructLayout(LayoutKind.Sequential)]
internal struct VkComputePipelineCreateInfo
{
    public uint sType; // VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO = 29
    public nint pNext;
    public uint flags;
    public VkPipelineShaderStageCreateInfo stage;
    public ulong layout;
    public ulong basePipelineHandle;
    public int basePipelineIndex;
}

[StructLayout(LayoutKind.Sequential)]
internal struct VkPipelineShaderStageCreateInfo
{
    public uint sType; // VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO = 18
    public nint pNext;
    public uint flags;
    public uint stage; // VK_SHADER_STAGE_COMPUTE_BIT
    public ulong module;
    public nint pName;
    public nint pSpecializationInfo;
}

[StructLayout(LayoutKind.Sequential)]
internal struct VkPipelineLayoutCreateInfo
{
    public uint sType; // VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO = 30
    public nint pNext;
    public uint flags;
    public uint setLayoutCount;
    public nint pSetLayouts;
    public uint pushConstantRangeCount;
    public nint pPushConstantRanges;
}

[StructLayout(LayoutKind.Sequential)]
internal struct VkPushConstantRange
{
    public uint stageFlags;
    public uint offset;
    public uint size;
}

[StructLayout(LayoutKind.Sequential)]
internal struct VkDescriptorSetLayoutBinding
{
    public uint binding;
    public uint descriptorType;
    public uint descriptorCount;
    public uint stageFlags;
    public nint pImmutableSamplers;
}

[StructLayout(LayoutKind.Sequential)]
internal struct VkDescriptorSetLayoutCreateInfo
{
    public uint sType; // VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO = 32
    public nint pNext;
    public uint flags;
    public uint bindingCount;
    public nint pBindings;
}

[StructLayout(LayoutKind.Sequential)]
internal struct VkDescriptorPoolSize
{
    public uint type;
    public uint descriptorCount;
}

[StructLayout(LayoutKind.Sequential)]
internal struct VkDescriptorPoolCreateInfo
{
    public uint sType; // VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO = 33
    public nint pNext;
    public uint flags;
    public uint maxSets;
    public uint poolSizeCount;
    public nint pPoolSizes;
}

[StructLayout(LayoutKind.Sequential)]
internal struct VkDescriptorSetAllocateInfo
{
    public uint sType; // VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO = 34
    public nint pNext;
    public ulong descriptorPool;
    public uint descriptorSetCount;
    public nint pSetLayouts;
}

[StructLayout(LayoutKind.Sequential)]
internal struct VkWriteDescriptorSet
{
    public uint sType; // VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET = 35
    public nint pNext;
    public ulong dstSet;
    public uint dstBinding;
    public uint dstArrayElement;
    public uint descriptorCount;
    public uint descriptorType;
    public nint pImageInfo;
    public nint pBufferInfo;
    public nint pTexelBufferView;
}

[StructLayout(LayoutKind.Sequential)]
internal struct VkDescriptorBufferInfo
{
    public ulong buffer;
    public ulong offset;
    public ulong range;
}

[StructLayout(LayoutKind.Sequential)]
internal struct VkCommandPoolCreateInfo
{
    public uint sType; // VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO = 39
    public nint pNext;
    public uint flags;
    public uint queueFamilyIndex;
}

[StructLayout(LayoutKind.Sequential)]
internal struct VkCommandBufferAllocateInfo
{
    public uint sType; // VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO = 40
    public nint pNext;
    public ulong commandPool;
    public uint level;
    public uint commandBufferCount;
}

[StructLayout(LayoutKind.Sequential)]
internal struct VkCommandBufferBeginInfo
{
    public uint sType; // VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO = 42
    public nint pNext;
    public uint flags;
    public nint pInheritanceInfo;
}

[StructLayout(LayoutKind.Sequential)]
internal struct VkSubmitInfo
{
    public uint sType; // VK_STRUCTURE_TYPE_SUBMIT_INFO = 4
    public nint pNext;
    public uint waitSemaphoreCount;
    public nint pWaitSemaphores;
    public nint pWaitDstStageMask;
    public uint commandBufferCount;
    public nint pCommandBuffers;
    public uint signalSemaphoreCount;
    public nint pSignalSemaphores;
}

[StructLayout(LayoutKind.Sequential)]
internal struct VkFenceCreateInfo
{
    public uint sType; // VK_STRUCTURE_TYPE_FENCE_CREATE_INFO = 8
    public nint pNext;
    public uint flags;
}

[StructLayout(LayoutKind.Sequential)]
internal struct VkMemoryBarrier
{
    public uint sType; // VK_STRUCTURE_TYPE_MEMORY_BARRIER = 46
    public nint pNext;
    public uint srcAccessMask;
    public uint dstAccessMask;
}

[StructLayout(LayoutKind.Sequential)]
internal struct VkBufferCopy
{
    public ulong srcOffset;
    public ulong dstOffset;
    public ulong size;
}
