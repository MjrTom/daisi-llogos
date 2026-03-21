using System.Reflection;
using System.Runtime.InteropServices;

namespace Daisi.Llogos.Vulkan;

/// <summary>
/// Manages a Vulkan compute pipeline: shader module, descriptor set layout, pipeline layout, and pipeline.
/// Each pipeline corresponds to one SPIR-V compute shader with N storage buffer bindings and push constants.
/// </summary>
internal sealed class VulkanPipeline : IDisposable
{
    private readonly VulkanDevice _vkDevice;
    private ulong _shaderModule;
    private ulong _descriptorSetLayout;
    private ulong _pipelineLayout;
    private ulong _pipeline;
    private ulong _descriptorPool;
    private bool _disposed;

    public ulong PipelineLayout => _pipelineLayout;
    public ulong Pipeline => _pipeline;
    public int BindingCount { get; }

    private VulkanPipeline(VulkanDevice vkDevice, int bindingCount)
    {
        _vkDevice = vkDevice;
        BindingCount = bindingCount;
    }

    /// <summary>
    /// Create a compute pipeline using buffer device addresses (no descriptor sets).
    /// Push constants carry buffer addresses + parameters.
    /// </summary>
    public static unsafe VulkanPipeline FromEmbeddedSpirVBda(VulkanDevice vkDevice, string resourceName, uint pushConstantSize)
    {
        var assembly = Assembly.GetExecutingAssembly();
        using var stream = assembly.GetManifestResourceStream(resourceName)
            ?? throw new FileNotFoundException($"Embedded resource not found: {resourceName}");
        var spirvBytes = new byte[stream.Length];
        stream.ReadExactly(spirvBytes);

        var pipeline = new VulkanPipeline(vkDevice, 0);
        nint device = vkDevice.Device;

        fixed (byte* pCode = spirvBytes)
        {
            var moduleInfo = new VkShaderModuleCreateInfo
            {
                sType = 16,
                codeSize = (nuint)spirvBytes.Length,
                pCode = (nint)pCode,
            };
            VulkanApi.Check(VulkanApi.CreateShaderModule(device, &moduleInfo, 0, out pipeline._shaderModule), "vkCreateShaderModule");
        }

        // No descriptor set layout — push constants only
        var pushRange = new VkPushConstantRange
        {
            stageFlags = VkConst.ShaderStageComputeBit,
            offset = 0,
            size = pushConstantSize,
        };

        var pipelineLayoutInfo = new VkPipelineLayoutCreateInfo
        {
            sType = 30,
            setLayoutCount = 0,
            pSetLayouts = 0,
            pushConstantRangeCount = 1,
            pPushConstantRanges = (nint)(&pushRange),
        };
        VulkanApi.Check(VulkanApi.CreatePipelineLayout(device, &pipelineLayoutInfo, 0, out pipeline._pipelineLayout), "vkCreatePipelineLayout");

        var entryPoint = Marshal.StringToHGlobalAnsi("main");
        try
        {
            var stageInfo = new VkPipelineShaderStageCreateInfo
            {
                sType = 18,
                stage = VkConst.ShaderStageComputeBit,
                module = pipeline._shaderModule,
                pName = entryPoint,
            };
            var computeInfo = new VkComputePipelineCreateInfo
            {
                sType = 29,
                stage = stageInfo,
                layout = pipeline._pipelineLayout,
            };
            ulong pipelineHandle;
            VulkanApi.Check(VulkanApi.CreateComputePipelines(device, 0, 1, &computeInfo, 0, &pipelineHandle), "vkCreateComputePipelines");
            pipeline._pipeline = pipelineHandle;
        }
        finally
        {
            Marshal.FreeHGlobal(entryPoint);
        }

        return pipeline;
    }

    /// <summary>
    /// Create a compute pipeline from an embedded SPIR-V resource.
    /// </summary>
    public static unsafe VulkanPipeline FromEmbeddedSpirV(VulkanDevice vkDevice, string resourceName, int bindingCount, uint pushConstantSize = 0)
    {
        var assembly = Assembly.GetExecutingAssembly();
        using var stream = assembly.GetManifestResourceStream(resourceName)
            ?? throw new FileNotFoundException($"Embedded resource not found: {resourceName}");
        var spirvBytes = new byte[stream.Length];
        stream.ReadExactly(spirvBytes);
        return FromSpirV(vkDevice, spirvBytes, bindingCount, pushConstantSize);
    }

    /// <summary>
    /// Create a compute pipeline from raw SPIR-V bytes.
    /// </summary>
    public static unsafe VulkanPipeline FromSpirV(VulkanDevice vkDevice, byte[] spirvBytes, int bindingCount, uint pushConstantSize = 0)
    {
        var pipeline = new VulkanPipeline(vkDevice, bindingCount);
        nint device = vkDevice.Device;

        // Create shader module
        fixed (byte* pCode = spirvBytes)
        {
            var moduleInfo = new VkShaderModuleCreateInfo
            {
                sType = 16,
                codeSize = (nuint)spirvBytes.Length,
                pCode = (nint)pCode,
            };
            VulkanApi.Check(VulkanApi.CreateShaderModule(device, &moduleInfo, 0, out pipeline._shaderModule), "vkCreateShaderModule");
        }

        // Create descriptor set layout with N storage buffer bindings
        var bindings = new VkDescriptorSetLayoutBinding[bindingCount];
        for (int i = 0; i < bindingCount; i++)
        {
            bindings[i] = new VkDescriptorSetLayoutBinding
            {
                binding = (uint)i,
                descriptorType = VkConst.DescriptorTypeStorageBuffer,
                descriptorCount = 1,
                stageFlags = VkConst.ShaderStageComputeBit,
            };
        }

        fixed (VkDescriptorSetLayoutBinding* pBindings = bindings)
        {
            var layoutInfo = new VkDescriptorSetLayoutCreateInfo
            {
                sType = 32,
                bindingCount = (uint)bindingCount,
                pBindings = (nint)pBindings,
            };
            VulkanApi.Check(VulkanApi.CreateDescriptorSetLayout(device, &layoutInfo, 0, out pipeline._descriptorSetLayout), "vkCreateDescriptorSetLayout");
        }

        // Create pipeline layout (with optional push constants)
        ulong dsLayout = pipeline._descriptorSetLayout;
        var pushRange = new VkPushConstantRange
        {
            stageFlags = VkConst.ShaderStageComputeBit,
            offset = 0,
            size = pushConstantSize,
        };

        var pipelineLayoutInfo = new VkPipelineLayoutCreateInfo
        {
            sType = 30,
            setLayoutCount = 1,
            pSetLayouts = (nint)(&dsLayout),
            pushConstantRangeCount = pushConstantSize > 0 ? 1u : 0u,
            pPushConstantRanges = pushConstantSize > 0 ? (nint)(&pushRange) : 0,
        };
        VulkanApi.Check(VulkanApi.CreatePipelineLayout(device, &pipelineLayoutInfo, 0, out pipeline._pipelineLayout), "vkCreatePipelineLayout");

        // Create compute pipeline
        var entryPoint = Marshal.StringToHGlobalAnsi("main");
        try
        {
            var stageInfo = new VkPipelineShaderStageCreateInfo
            {
                sType = 18,
                stage = VkConst.ShaderStageComputeBit,
                module = pipeline._shaderModule,
                pName = entryPoint,
            };

            var computeInfo = new VkComputePipelineCreateInfo
            {
                sType = 29,
                stage = stageInfo,
                layout = pipeline._pipelineLayout,
            };

            ulong pipelineHandle;
            VulkanApi.Check(VulkanApi.CreateComputePipelines(device, 0, 1, &computeInfo, 0, &pipelineHandle), "vkCreateComputePipelines");
            pipeline._pipeline = pipelineHandle;
        }
        finally
        {
            Marshal.FreeHGlobal(entryPoint);
        }

        // Create descriptor pool — enough for batched command buffer recording
        // (multiple dispatches per submit, each needing its own descriptor set)
        const uint maxBatchSets = 4096;
        var poolSize = new VkDescriptorPoolSize
        {
            type = VkConst.DescriptorTypeStorageBuffer,
            descriptorCount = (uint)bindingCount * maxBatchSets,
        };
        var poolInfo = new VkDescriptorPoolCreateInfo
        {
            sType = 33,
            flags = 0,
            maxSets = maxBatchSets,
            poolSizeCount = 1,
            pPoolSizes = (nint)(&poolSize),
        };
        VulkanApi.Check(VulkanApi.CreateDescriptorPool(device, &poolInfo, 0, out pipeline._descriptorPool), "vkCreateDescriptorPool");

        return pipeline;
    }

    /// <summary>
    /// Allocate a descriptor set and bind buffers to it.
    /// </summary>
    /// <summary>Reset the descriptor pool (call at start of each batch).</summary>
    public void ResetDescriptorPool()
    {
        // Only reset if cache is not valid (first run or overflow)
        // With caching, descriptor sets persist across batches since buffer bindings don't change
        if (!_dsCacheValid)
        {
            VulkanApi.ResetDescriptorPool(_vkDevice.Device, _descriptorPool, 0);
            _dsCache.Clear();
        }
    }

    // Cache descriptor sets by buffer handle combination (same buffers = same descriptor set)
    private readonly Dictionary<long, ulong> _dsCache = new();
    private bool _dsCacheValid; // false after pool reset

    public unsafe ulong AllocateDescriptorSet(ReadOnlySpan<VulkanBuffer> buffers)
    {
        if (buffers.Length != BindingCount)
            throw new ArgumentException($"Expected {BindingCount} buffers, got {buffers.Length}.");

        // Build a cache key from buffer handles (XOR-hash for speed)
        long key = BindingCount;
        for (int i = 0; i < buffers.Length; i++)
            key = key * 31 + (long)buffers[i].Buffer;

        if (_dsCacheValid && _dsCache.TryGetValue(key, out var cached))
            return cached;

        // Allocate new descriptor set
        ulong dsLayout = _descriptorSetLayout;
        var allocInfo = new VkDescriptorSetAllocateInfo
        {
            sType = 34,
            descriptorPool = _descriptorPool,
            descriptorSetCount = 1,
            pSetLayouts = (nint)(&dsLayout),
        };

        ulong descriptorSet;
        VulkanApi.Check(VulkanApi.AllocateDescriptorSets(_vkDevice.Device, &allocInfo, &descriptorSet), "vkAllocateDescriptorSets");

        // Bind buffers
        var bufferInfos = new VkDescriptorBufferInfo[BindingCount];
        var writes = new VkWriteDescriptorSet[BindingCount];
        for (int i = 0; i < BindingCount; i++)
        {
            bufferInfos[i] = new VkDescriptorBufferInfo
            {
                buffer = buffers[i].Buffer,
                offset = 0,
                range = VkConst.WholeSize,
            };
        }

        fixed (VkDescriptorBufferInfo* pBufInfos = bufferInfos)
        {
            for (int i = 0; i < BindingCount; i++)
            {
                writes[i] = new VkWriteDescriptorSet
                {
                    sType = 35,
                    dstSet = descriptorSet,
                    dstBinding = (uint)i,
                    descriptorCount = 1,
                    descriptorType = VkConst.DescriptorTypeStorageBuffer,
                    pBufferInfo = (nint)(&pBufInfos[i]),
                };
            }

            fixed (VkWriteDescriptorSet* pWrites = writes)
                VulkanApi.UpdateDescriptorSets(_vkDevice.Device, (uint)BindingCount, pWrites, 0, 0);
        }

        _dsCache[key] = descriptorSet;
        _dsCacheValid = true;
        return descriptorSet;
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            nint dev = _vkDevice.Device;
            if (_descriptorPool != 0) VulkanApi.DestroyDescriptorPool(dev, _descriptorPool, 0);
            if (_pipeline != 0) VulkanApi.DestroyPipeline(dev, _pipeline, 0);
            if (_pipelineLayout != 0) VulkanApi.DestroyPipelineLayout(dev, _pipelineLayout, 0);
            if (_descriptorSetLayout != 0) VulkanApi.DestroyDescriptorSetLayout(dev, _descriptorSetLayout, 0);
            if (_shaderModule != 0) VulkanApi.DestroyShaderModule(dev, _shaderModule, 0);
            _disposed = true;
        }
    }
}
