/**
 * GPU-resident KV cache for transformer attention.
 * Layout: [numKvHeads x maxSeqLen x headDim] for both K and V.
 */

import { BufferPool } from '../gpu/buffer-pool.js';

export class KvCache {
  private device: GPUDevice;
  readonly numKvHeads: number;
  readonly headDim: number;
  readonly maxSeqLen: number;
  private _seqLen = 0;

  kBuffer: GPUBuffer;
  vBuffer: GPUBuffer;

  constructor(
    device: GPUDevice,
    buffers: BufferPool,
    numKvHeads: number,
    headDim: number,
    maxSeqLen: number,
    label: string,
  ) {
    this.device = device;
    this.numKvHeads = numKvHeads;
    this.headDim = headDim;
    this.maxSeqLen = maxSeqLen;

    const size = numKvHeads * maxSeqLen * headDim * 4; // f32
    this.kBuffer = buffers.createBuffer(`${label}_k`, size);
    this.vBuffer = buffers.createBuffer(`${label}_v`, size);
  }

  get seqLen(): number { return this._seqLen; }

  /**
   * Write K and V vectors for the current position.
   * k, v: [numKvHeads * headDim] float32
   */
  write(k: GPUBuffer, v: GPUBuffer, kSize: number, vSize: number): void {
    const offset = this._seqLen * this.headDim * 4; // byte offset per head
    // For each KV head, write at position seqLen
    // The K/V data is contiguous [numKvHeads * headDim], and the cache layout
    // is [numKvHeads][maxSeqLen][headDim], so we need strided copies.
    const encoder = this.device.createCommandEncoder();
    for (let h = 0; h < this.numKvHeads; h++) {
      const srcOffset = h * this.headDim * 4;
      const dstOffset = (h * this.maxSeqLen * this.headDim + this._seqLen * this.headDim) * 4;
      encoder.copyBufferToBuffer(k, srcOffset, this.kBuffer, dstOffset, this.headDim * 4);
      encoder.copyBufferToBuffer(v, srcOffset, this.vBuffer, dstOffset, this.headDim * 4);
    }
    this.device.queue.submit([encoder.finish()]);
    this._seqLen++;
  }

  /** Reset cache for new sequence. */
  reset(): void {
    this._seqLen = 0;
  }
}
