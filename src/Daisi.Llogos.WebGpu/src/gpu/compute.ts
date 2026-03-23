/**
 * High-level compute dispatch engine. Wraps ShaderCache + BufferPool with
 * convenient dispatch methods for each operation.
 */

import { ShaderCache } from './shader-cache.js';
import { BufferPool } from './buffer-pool.js';
import { GgmlType } from '../gguf/quantization.js';

import embeddingWgsl from './shaders/embedding.wgsl';
import rmsnormWgsl from './shaders/rmsnorm.wgsl';
import ropeWgsl from './shaders/rope.wgsl';
import matmulWgsl from './shaders/matmul.wgsl';
import matmulQ4Wgsl from './shaders/matmul_q4.wgsl';
import matmulQ8Wgsl from './shaders/matmul_q8.wgsl';
import attentionWgsl from './shaders/attention.wgsl';
import softmaxWgsl from './shaders/softmax.wgsl';
import siluWgsl from './shaders/silu.wgsl';
import siluMulWgsl from './shaders/silu_mul.wgsl';
import copyWgsl from './shaders/copy.wgsl';
import copyRmsnormWgsl from './shaders/copy_rmsnorm.wgsl';
import addWgsl from './shaders/add.wgsl';

/** Helpers for creating bind group layout entries. */
function storageReadOnly(binding: number): GPUBindGroupLayoutEntry {
  return { binding, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } };
}
function storageReadWrite(binding: number): GPUBindGroupLayoutEntry {
  return { binding, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } };
}
function uniform(binding: number): GPUBindGroupLayoutEntry {
  return { binding, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } };
}

export class ComputeEngine {
  device: GPUDevice;
  shaders: ShaderCache;
  buffers: BufferPool;

  constructor(device: GPUDevice) {
    this.device = device;
    this.shaders = new ShaderCache(device);
    this.buffers = new BufferPool(device);
  }

  /** Params cache — unique buffer per unique content. Safe because same content = no conflict. */
  private paramsMap = new Map<string, GPUBuffer>();

  private createParams(label: string, data: ArrayBuffer): GPUBuffer {
    // Fast hash: use typed array values directly
    const u32 = new Uint32Array(data);
    let key = label;
    for (let i = 0; i < u32.length; i++) key += ',' + u32[i];

    let buf = this.paramsMap.get(key);
    if (!buf) {
      buf = this.device.createBuffer({
        size: Math.ceil(data.byteLength / 4) * 4,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      this.device.queue.writeBuffer(buf, 0, data);
      this.paramsMap.set(key, buf);
    }
    return buf;
  }

  /** No-op — params are cached permanently (same content = same buffer, always safe). */
  cleanupParams(): void { }

  /** Batched encoder — multiple compute passes, ONE submit. */
  private batchEncoder: GPUCommandEncoder | null = null;

  beginBatch(): void {
    this.batchEncoder = this.device.createCommandEncoder({ label: 'fwd' });
  }

  endBatch(): void {
    if (this.batchEncoder) {
      this.device.queue.submit([this.batchEncoder.finish()]);
      this.batchEncoder = null;
    }
  }

  /** Copy buffer — uses batch encoder if active. */
  copyBuffer(src: GPUBuffer, dst: GPUBuffer, size: number): void {
    const encoder = this.batchEncoder ?? this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(src, 0, dst, 0, size);
    if (!this.batchEncoder) {
      this.device.queue.submit([encoder.finish()]);
    }
  }

  /** Dispatch — each dispatch is its own compute pass (for proper barriers).
   *  Uses batch encoder if active (single submit), otherwise standalone. */
  private dispatch(
    shaderSrc: string,
    label: string,
    layout: GPUBindGroupLayoutEntry[],
    entries: GPUBindGroupEntry[],
    workgroups: [number, number?, number?],
  ): void {
    const cached = this.shaders.getPipeline({ shader: shaderSrc, bindGroupLayout: layout, label });
    const bindGroup = this.shaders.createBindGroup(cached, entries, label);
    const encoder = this.batchEncoder ?? this.device.createCommandEncoder({ label });
    const pass = encoder.beginComputePass({ label });
    pass.setPipeline(cached.pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(workgroups[0], workgroups[1] ?? 1, workgroups[2] ?? 1);
    pass.end();
    if (!this.batchEncoder) {
      this.device.queue.submit([encoder.finish()]);
    }
  }

  // ── Operations ────────────────────────────────────────────────────────

  /** Embedding lookup: output = weights[tokenId * embDim : (tokenId+1) * embDim] */
  embedding(weights: GPUBuffer, output: GPUBuffer, tokenId: number, embDim: number): void {
    const params = this.createParams('embedding_params', new Uint32Array([tokenId, embDim]).buffer);
    this.dispatch(embeddingWgsl, 'embedding', [
      storageReadOnly(0), storageReadWrite(1), uniform(2),
    ], [
      { binding: 0, resource: { buffer: weights } },
      { binding: 1, resource: { buffer: output } },
      { binding: 2, resource: { buffer: params } },
    ], [Math.ceil(embDim / 256)]);
  }

  /** RMS Normalization */
  rmsNorm(input: GPUBuffer, weight: GPUBuffer, output: GPUBuffer, n: number, eps: number): void {
    const buf = new ArrayBuffer(8);
    const view = new DataView(buf);
    view.setUint32(0, n, true);
    view.setFloat32(4, eps, true); // We store as u32 bits in shader, but actually it reads as bitcast
    // Actually let's store eps bits properly
    const paramData = new Uint32Array(2);
    paramData[0] = n;
    const epsView = new Float32Array(1);
    epsView[0] = eps;
    paramData[1] = new Uint32Array(epsView.buffer)[0]; // bitcast float to uint
    const params = this.createParams('rmsnorm_params', paramData.buffer);
    this.dispatch(rmsnormWgsl, 'rmsnorm', [
      storageReadOnly(0), storageReadOnly(1), storageReadWrite(2), uniform(3),
    ], [
      { binding: 0, resource: { buffer: input } },
      { binding: 1, resource: { buffer: weight } },
      { binding: 2, resource: { buffer: output } },
      { binding: 3, resource: { buffer: params } },
    ], [1]);
  }

  /** Fused: copy input→residual AND RMSNorm(input, weight)→output. Saves 1 dispatch. */
  copyAndRmsNorm(
    input: GPUBuffer, weight: GPUBuffer, output: GPUBuffer, residual: GPUBuffer,
    n: number, eps: number,
  ): void {
    const paramData = new Uint32Array(2);
    paramData[0] = n;
    const epsView = new Float32Array(1);
    epsView[0] = eps;
    paramData[1] = new Uint32Array(epsView.buffer)[0];
    const params = this.createParams('copy_rmsnorm_params', paramData.buffer);
    this.dispatch(copyRmsnormWgsl, 'copy_rmsnorm', [
      storageReadOnly(0), storageReadOnly(1), storageReadWrite(2), storageReadWrite(3), uniform(4),
    ], [
      { binding: 0, resource: { buffer: input } },
      { binding: 1, resource: { buffer: weight } },
      { binding: 2, resource: { buffer: output } },
      { binding: 3, resource: { buffer: residual } },
      { binding: 4, resource: { buffer: params } },
    ], [1]);
  }

  /** Matrix-vector multiply for the appropriate quantization type. */
  matmul(
    weights: GPUBuffer, input: GPUBuffer, output: GPUBuffer,
    M: number, K: number, quantType: GgmlType,
  ): void {
    const paramData = new Uint32Array([M, K]);
    const params = this.createParams('matmul_params', paramData.buffer);

    let shader: string;
    switch (quantType) {
      case GgmlType.F32:
        shader = matmulWgsl; break;
      case GgmlType.Q4_0:
        shader = matmulQ4Wgsl; break;
      case GgmlType.Q8_0:
        shader = matmulQ8Wgsl; break;
      default:
        throw new Error(`Unsupported quantization type for matmul: ${quantType}`);
    }

    this.dispatch(shader, 'matmul', [
      storageReadOnly(0), storageReadOnly(1), storageReadWrite(2), uniform(3),
    ], [
      { binding: 0, resource: { buffer: weights } },
      { binding: 1, resource: { buffer: input } },
      { binding: 2, resource: { buffer: output } },
      { binding: 3, resource: { buffer: params } },
    ], [M]); // One workgroup per output row
  }

  /** Pre-built RoPE cos/sin tables indexed by position. */
  private ropeTableCache = new Map<number, { cos: GPUBuffer; sin: GPUBuffer }>();

  /** RoPE: apply rotary position embeddings using cached cos/sin tables. */
  rope(
    data: GPUBuffer, headDim: number, ropeDim: number,
    position: number, theta: number, nElements: number,
  ): void {
    // Get or create cos/sin table for this position
    let table = this.ropeTableCache.get(position);
    if (!table) {
      const halfDim = headDim / 2;
      const cosData = new Float32Array(halfDim);
      const sinData = new Float32Array(halfDim);
      for (let i = 0; i < halfDim; i++) {
        const dimFrac = (i * 2) / headDim;
        const freq = 1.0 / Math.pow(theta, dimFrac);
        const angle = position * freq;
        cosData[i] = Math.fround(Math.cos(angle));
        sinData[i] = Math.fround(Math.sin(angle));
      }
      table = {
        cos: this.buffers.createBufferWithData(`rope_cos_${position}`, cosData.buffer),
        sin: this.buffers.createBufferWithData(`rope_sin_${position}`, sinData.buffer),
      };
      this.ropeTableCache.set(position, table);
    }

    const paramData = new Uint32Array(2);
    paramData[0] = nElements;
    paramData[1] = headDim;
    const params = this.createParams('rope_params', paramData.buffer);

    this.dispatch(ropeWgsl, 'rope', [
      storageReadWrite(0), storageReadOnly(1), storageReadOnly(2), uniform(3),
    ], [
      { binding: 0, resource: { buffer: data } },
      { binding: 1, resource: { buffer: table.cos } },
      { binding: 2, resource: { buffer: table.sin } },
      { binding: 3, resource: { buffer: params } },
    ], [nElements / 2]);
  }

  /** CPU attention — reads GPU buffers, computes on CPU, writes back. */
  async cpuAttention(
    q: GPUBuffer, kCache: GPUBuffer, vCache: GPUBuffer, output: GPUBuffer,
    numHeads: number, numKvHeads: number, headDim: number,
    seqLen: number, maxSeqLen: number,
  ): Promise<void> {
    const scale = 1.0 / Math.sqrt(headDim);
    const headsPerKvGroup = numHeads / numKvHeads;
    const qData = await this.readBuffer(q, numHeads * headDim * 4);
    const kData = await this.readBuffer(kCache, numKvHeads * maxSeqLen * headDim * 4);
    const vData = await this.readBuffer(vCache, numKvHeads * maxSeqLen * headDim * 4);
    const outData = new Float32Array(numHeads * headDim);

    for (let h = 0; h < numHeads; h++) {
      const kvHead = Math.floor(h / headsPerKvGroup);
      const qOff = h * headDim;
      const scores = new Float32Array(seqLen);
      for (let pos = 0; pos < seqLen; pos++) {
        const kOff = kvHead * maxSeqLen * headDim + pos * headDim;
        let dot = 0;
        for (let d = 0; d < headDim; d++) dot += qData[qOff + d] * kData[kOff + d];
        scores[pos] = dot * scale;
      }
      let maxS = -Infinity;
      for (let i = 0; i < seqLen; i++) maxS = Math.max(maxS, scores[i]);
      let sumE = 0;
      for (let i = 0; i < seqLen; i++) { scores[i] = Math.exp(scores[i] - maxS); sumE += scores[i]; }
      for (let i = 0; i < seqLen; i++) scores[i] /= sumE;
      for (let d = 0; d < headDim; d++) {
        let acc = 0;
        for (let pos = 0; pos < seqLen; pos++) {
          const vOff = kvHead * maxSeqLen * headDim + pos * headDim;
          acc += scores[pos] * vData[vOff + d];
        }
        outData[qOff + d] = acc;
      }
    }
    this.device.queue.writeBuffer(output, 0, outData);
  }

  /** GPU attention — single-thread per head. */
  attention(
    q: GPUBuffer, kCache: GPUBuffer, vCache: GPUBuffer, output: GPUBuffer,
    numHeads: number, numKvHeads: number, headDim: number,
    seqLen: number, maxSeqLen: number,
  ): void {
    const scale = 1.0 / Math.sqrt(headDim);
    const paramData = new Uint32Array(6);
    paramData[0] = numHeads;
    paramData[1] = numKvHeads;
    paramData[2] = headDim;
    paramData[3] = seqLen;
    paramData[4] = maxSeqLen;
    const scaleView = new Float32Array(1); scaleView[0] = scale;
    paramData[5] = new Uint32Array(scaleView.buffer)[0];
    const params = this.createParams('attention_params', paramData.buffer);
    this.dispatch(attentionWgsl, 'attention', [
      storageReadOnly(0), storageReadOnly(1), storageReadOnly(2),
      storageReadWrite(3), uniform(4),
    ], [
      { binding: 0, resource: { buffer: q } },
      { binding: 1, resource: { buffer: kCache } },
      { binding: 2, resource: { buffer: vCache } },
      { binding: 3, resource: { buffer: output } },
      { binding: 4, resource: { buffer: params } },
    ], [numHeads]); // One workgroup (1 thread) per head
  }

  /** Softmax over n elements. */
  softmax(input: GPUBuffer, output: GPUBuffer, n: number): void {
    const params = this.createParams('softmax_params', new Uint32Array([n]).buffer);
    this.dispatch(softmaxWgsl, 'softmax', [
      storageReadOnly(0), storageReadWrite(1), uniform(2),
    ], [
      { binding: 0, resource: { buffer: input } },
      { binding: 1, resource: { buffer: output } },
      { binding: 2, resource: { buffer: params } },
    ], [1]);
  }

  /** SiLU activation: output = x * sigmoid(x) */
  silu(input: GPUBuffer, output: GPUBuffer, n: number): void {
    const params = this.createParams('silu_params', new Uint32Array([n]).buffer);
    this.dispatch(siluWgsl, 'silu', [
      storageReadOnly(0), storageReadWrite(1), uniform(2),
    ], [
      { binding: 0, resource: { buffer: input } },
      { binding: 1, resource: { buffer: output } },
      { binding: 2, resource: { buffer: params } },
    ], [Math.ceil(n / 256)]);
  }

  /** Fused SiLU-gate multiply: output = silu(gate) * up */
  siluMul(gate: GPUBuffer, up: GPUBuffer, output: GPUBuffer, n: number): void {
    const params = this.createParams('silu_mul_params', new Uint32Array([n]).buffer);
    this.dispatch(siluMulWgsl, 'silu_mul', [
      storageReadOnly(0), storageReadOnly(1), storageReadWrite(2), uniform(3),
    ], [
      { binding: 0, resource: { buffer: gate } },
      { binding: 1, resource: { buffer: up } },
      { binding: 2, resource: { buffer: output } },
      { binding: 3, resource: { buffer: params } },
    ], [Math.ceil(n / 256)]);
  }

  /** Element-wise add: output = a + b */
  add(a: GPUBuffer, b: GPUBuffer, output: GPUBuffer, n: number): void {
    const params = this.createParams('add_params', new Uint32Array([n]).buffer);
    this.dispatch(addWgsl, 'add', [
      storageReadOnly(0), storageReadOnly(1), storageReadWrite(2), uniform(3),
    ], [
      { binding: 0, resource: { buffer: a } },
      { binding: 1, resource: { buffer: b } },
      { binding: 2, resource: { buffer: output } },
      { binding: 3, resource: { buffer: params } },
    ], [Math.ceil(n / 256)]);
  }

  /** Reusable readback buffer — avoids allocation per readLogits call. */
  private readbackBuf: GPUBuffer | null = null;
  private readbackSize = 0;

  /** Read buffer data back to CPU. */
  async readBuffer(buffer: GPUBuffer, size: number): Promise<Float32Array> {
    // Reuse readback buffer if same size
    if (!this.readbackBuf || this.readbackSize < size) {
      this.readbackBuf?.destroy();
      this.readbackSize = size;
      this.readbackBuf = this.device.createBuffer({
        label: 'readback',
        size,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      });
    }
    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(buffer, 0, this.readbackBuf, 0, size);
    this.device.queue.submit([encoder.finish()]);

    await this.readbackBuf.mapAsync(GPUMapMode.READ);
    const data = new Float32Array(this.readbackBuf.getMappedRange().slice(0));
    this.readbackBuf.unmap();
    return data;
  }
}
