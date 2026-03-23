/**
 * Compiled pipeline cache — avoids recompiling WGSL shaders.
 */

export interface PipelineConfig {
  shader: string;
  entryPoint?: string;
  bindGroupLayout: GPUBindGroupLayoutEntry[];
  label?: string;
}

interface CachedPipeline {
  pipeline: GPUComputePipeline;
  bindGroupLayout: GPUBindGroupLayout;
}

export class ShaderCache {
  private device: GPUDevice;
  private moduleCache = new Map<string, GPUShaderModule>();
  private pipelineCache = new Map<string, CachedPipeline>();

  constructor(device: GPUDevice) {
    this.device = device;
  }

  /**
   * Get or compile a shader module from WGSL source.
   */
  getModule(source: string, label?: string): GPUShaderModule {
    let module = this.moduleCache.get(source);
    if (!module) {
      module = this.device.createShaderModule({ code: source, label });
      this.moduleCache.set(source, module);
    }
    return module;
  }

  /**
   * Get or create a compute pipeline from config.
   */
  getPipeline(config: PipelineConfig): CachedPipeline {
    const key = `${config.shader}::${config.entryPoint ?? 'main'}::${JSON.stringify(config.bindGroupLayout)}`;
    let cached = this.pipelineCache.get(key);
    if (!cached) {
      const module = this.getModule(config.shader, config.label);
      const bindGroupLayout = this.device.createBindGroupLayout({
        label: config.label,
        entries: config.bindGroupLayout,
      });
      const pipelineLayout = this.device.createPipelineLayout({
        label: config.label,
        bindGroupLayouts: [bindGroupLayout],
      });
      const pipeline = this.device.createComputePipeline({
        label: config.label,
        layout: pipelineLayout,
        compute: { module, entryPoint: config.entryPoint ?? 'main' },
      });
      cached = { pipeline, bindGroupLayout };
      this.pipelineCache.set(key, cached);
    }
    return cached;
  }

  /**
   * Create a bind group from a cached pipeline's layout.
   */
  createBindGroup(
    cached: CachedPipeline,
    entries: GPUBindGroupEntry[],
    label?: string,
  ): GPUBindGroup {
    return this.device.createBindGroup({
      label,
      layout: cached.bindGroupLayout,
      entries,
    });
  }
}
