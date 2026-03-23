/**
 * WebGPU device initialization and capability detection.
 */

export interface GpuCapabilities {
  adapterInfo: GPUAdapterInfo;
  maxBufferSize: number;
  maxStorageBufferBindingSize: number;
  maxComputeWorkgroupSizeX: number;
  maxComputeInvocationsPerWorkgroup: number;
  supportsF16: boolean;
  supportsTimestampQuery: boolean;
}

export interface GpuContext {
  adapter: GPUAdapter;
  device: GPUDevice;
  capabilities: GpuCapabilities;
}

/**
 * Check if WebGPU is available in this browser.
 */
export function isWebGpuAvailable(): boolean {
  return typeof navigator !== 'undefined' && 'gpu' in navigator;
}

/**
 * Initialize WebGPU: request adapter and device with maximum capabilities.
 */
export async function initGpu(): Promise<GpuContext> {
  if (!isWebGpuAvailable()) {
    throw new Error('WebGPU is not available in this browser.');
  }

  const adapter = await navigator.gpu.requestAdapter({
    powerPreference: 'high-performance',
  });

  if (!adapter) {
    throw new Error('No WebGPU adapter found. Your GPU may not support WebGPU.');
  }

  const adapterInfo = adapter.info;
  const limits = adapter.limits;

  const supportsF16 = adapter.features.has('shader-f16');
  const supportsTimestampQuery = adapter.features.has('timestamp-query');

  const requiredFeatures: GPUFeatureName[] = [];
  if (supportsTimestampQuery) requiredFeatures.push('timestamp-query');
  if (supportsF16) requiredFeatures.push('shader-f16');

  const device = await adapter.requestDevice({
    requiredFeatures,
    requiredLimits: {
      maxBufferSize: limits.maxBufferSize,
      maxStorageBufferBindingSize: limits.maxStorageBufferBindingSize,
      maxComputeWorkgroupSizeX: limits.maxComputeWorkgroupSizeX,
      maxComputeInvocationsPerWorkgroup: limits.maxComputeInvocationsPerWorkgroup,
    },
  });

  device.lost.then((info) => {
    console.error(`WebGPU device lost: ${info.message} (reason: ${info.reason})`);
  });

  return {
    adapter,
    device,
    capabilities: {
      adapterInfo,
      maxBufferSize: device.limits.maxBufferSize,
      maxStorageBufferBindingSize: device.limits.maxStorageBufferBindingSize,
      maxComputeWorkgroupSizeX: device.limits.maxComputeWorkgroupSizeX,
      maxComputeInvocationsPerWorkgroup: device.limits.maxComputeInvocationsPerWorkgroup,
      supportsF16,
      supportsTimestampQuery,
    },
  };
}
