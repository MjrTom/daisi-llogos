/**
 * GPU buffer allocation, reuse, and VRAM tracking.
 */

export interface BufferInfo {
  buffer: GPUBuffer;
  size: number;
  label: string;
}

export class BufferPool {
  private device: GPUDevice;
  private buffers = new Map<string, BufferInfo>();
  private totalAllocated = 0;

  constructor(device: GPUDevice) {
    this.device = device;
  }

  /** Total bytes allocated on GPU. */
  get vramUsage(): number { return this.totalAllocated; }

  /**
   * Create or retrieve a named storage buffer.
   */
  createBuffer(label: string, size: number, usage?: GPUBufferUsageFlags): GPUBuffer {
    const existing = this.buffers.get(label);
    if (existing && existing.size >= size) return existing.buffer;

    // Destroy old buffer if it exists
    if (existing) {
      existing.buffer.destroy();
      this.totalAllocated -= existing.size;
    }

    // Align to 4 bytes
    const alignedSize = Math.ceil(size / 4) * 4;
    const buffer = this.device.createBuffer({
      label,
      size: alignedSize,
      usage: usage ?? (GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST),
    });

    this.buffers.set(label, { buffer, size: alignedSize, label });
    this.totalAllocated += alignedSize;
    return buffer;
  }

  /**
   * Create a storage buffer initialized with data.
   */
  createBufferWithData(label: string, data: ArrayBufferLike, usage?: GPUBufferUsageFlags): GPUBuffer {
    const buffer = this.createBuffer(label, data.byteLength, usage);
    this.device.queue.writeBuffer(buffer, 0, data);
    return buffer;
  }

  /**
   * Create a buffer for reading data back to CPU.
   */
  createReadbackBuffer(label: string, size: number): GPUBuffer {
    const alignedSize = Math.ceil(size / 4) * 4;
    return this.device.createBuffer({
      label: `${label}_readback`,
      size: alignedSize,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
  }

  /**
   * Get an existing buffer by label.
   */
  get(label: string): GPUBuffer | undefined {
    return this.buffers.get(label)?.buffer;
  }

  /**
   * Destroy a named buffer and free its VRAM.
   */
  destroy(label: string): void {
    const info = this.buffers.get(label);
    if (info) {
      info.buffer.destroy();
      this.totalAllocated -= info.size;
      this.buffers.delete(label);
    }
  }

  /**
   * Destroy all buffers.
   */
  destroyAll(): void {
    for (const info of this.buffers.values()) {
      info.buffer.destroy();
    }
    this.buffers.clear();
    this.totalAllocated = 0;
  }
}
