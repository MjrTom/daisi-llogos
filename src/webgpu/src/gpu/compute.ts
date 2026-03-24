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
import addBiasWgsl from './shaders/add_bias.wgsl';
import embeddingQ8Wgsl from './shaders/embedding_q8.wgsl';
import conv1dSiluWgsl from './shaders/conv1d_silu.wgsl';
import siluInplaceWgsl from './shaders/silu_inplace.wgsl';
import l2NormGroupsWgsl from './shaders/l2_norm_groups.wgsl';
import computeDecayBetaWgsl from './shaders/compute_decay_beta.wgsl';
import deinterleaveQWgsl from './shaders/deinterleave_q.wgsl';
import perHeadRmsnormWgsl from './shaders/per_head_rmsnorm.wgsl';
import partialRopeWgsl from './shaders/partial_rope.wgsl';
import gatedAttentionWgsl from './shaders/gated_attention.wgsl';
import repeatTileWgsl from './shaders/repeat_tile.wgsl';
import matmulBatchWgsl from './shaders/matmul_batch.wgsl';
import embeddingBatchWgsl from './shaders/embedding_batch.wgsl';
import rmsnormBatchWgsl from './shaders/rmsnorm_batch.wgsl';
import attentionBatchWgsl from './shaders/attention_batch.wgsl';
import deltanetStepWgsl from './shaders/deltanet_step.wgsl';
import siluGateWgsl from './shaders/silu_gate.wgsl';

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
    this.copyBufferRegion(src, 0, dst, 0, size);
  }

  /** Copy buffer region with offsets — uses batch encoder if active. */
  copyBufferRegion(src: GPUBuffer, srcOffset: number, dst: GPUBuffer, dstOffset: number, size: number): void {
    const encoder = this.batchEncoder ?? this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(src, srcOffset, dst, dstOffset, size);
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

  /** Token embedding lookup for Q8_0 quantized weights. */
  embeddingQ8(weights: GPUBuffer, output: GPUBuffer, tokenId: number, embDim: number): void {
    const params = this.createParams('embedding_q8_params', new Uint32Array([tokenId, embDim]).buffer);
    this.dispatch(embeddingQ8Wgsl, 'embedding_q8', [
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
    ], M <= 65535 ? [M] : [65535, Math.ceil(M / 65535)]); // 2D dispatch for large vocabs
  }

  /** Pre-built RoPE cos/sin tables keyed by "theta,headDim,position". */
  private ropeTableCache = new Map<string, { cos: GPUBuffer; sin: GPUBuffer }>();

  /** Clear cached RoPE tables (call when loading a new model). */
  clearRopeCache(): void {
    this.ropeTableCache.clear();
  }

  /** RoPE: apply rotary position embeddings using cached cos/sin tables. */
  rope(
    data: GPUBuffer, headDim: number, ropeDim: number,
    position: number, theta: number, nElements: number,
  ): void {
    // Get or create cos/sin table for this (theta, headDim, position) tuple
    const cacheKey = `${theta},${headDim},${position}`;
    let table = this.ropeTableCache.get(cacheKey);
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
        cos: this.buffers.createBufferWithData(`rope_cos_${cacheKey}`, cosData.buffer),
        sin: this.buffers.createBufferWithData(`rope_sin_${cacheKey}`, sinData.buffer),
      };
      this.ropeTableCache.set(cacheKey, table);
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

  /** In-place bias add: data[i] += bias[i]. */
  addBias(data: GPUBuffer, bias: GPUBuffer, n: number): void {
    const params = this.createParams('add_bias_params', new Uint32Array([n]).buffer);
    this.dispatch(addBiasWgsl, 'add_bias', [
      storageReadWrite(0), storageReadOnly(1), uniform(2),
    ], [
      { binding: 0, resource: { buffer: data } },
      { binding: 1, resource: { buffer: bias } },
      { binding: 2, resource: { buffer: params } },
    ], [Math.ceil(n / 256)]);
  }

  // ── DeltaNet GPU ops ─────────────────────────────────────────────

  /** Fused causal conv1d + SiLU: convolve data with persistent buffer, apply SiLU. */
  conv1dSilu(data: GPUBuffer, convBuf: GPUBuffer, weights: GPUBuffer, channels: number, kernelSize: number): void {
    const params = this.createParams('conv1d_silu_params', new Uint32Array([channels, kernelSize]).buffer);
    this.dispatch(conv1dSiluWgsl, 'conv1d_silu', [
      storageReadWrite(0), storageReadWrite(1), storageReadOnly(2), uniform(3),
    ], [
      { binding: 0, resource: { buffer: data } },
      { binding: 1, resource: { buffer: convBuf } },
      { binding: 2, resource: { buffer: weights } },
      { binding: 3, resource: { buffer: params } },
    ], [Math.ceil(channels / 256)]);
  }

  /** In-place SiLU activation. */
  siluInplace(data: GPUBuffer, n: number): void {
    const params = this.createParams('silu_ip_params', new Uint32Array([n]).buffer);
    this.dispatch(siluInplaceWgsl, 'silu_inplace', [
      storageReadWrite(0), uniform(1),
    ], [
      { binding: 0, resource: { buffer: data } },
      { binding: 1, resource: { buffer: params } },
    ], [Math.ceil(n / 256)]);
  }

  /** L2 normalize each group independently. */
  l2NormGroups(data: GPUBuffer, numGroups: number, groupDim: number): void {
    const params = this.createParams('l2norm_params', new Uint32Array([numGroups, groupDim]).buffer);
    this.dispatch(l2NormGroupsWgsl, 'l2_norm_groups', [
      storageReadWrite(0), uniform(1),
    ], [
      { binding: 0, resource: { buffer: data } },
      { binding: 1, resource: { buffer: params } },
    ], [numGroups]);
  }

  /** Compute DeltaNet decay and beta values. */
  computeDecayBeta(
    alpha: GPUBuffer, beta: GPUBuffer, ssmA: GPUBuffer, dtBias: GPUBuffer,
    decayOut: GPUBuffer, betaOut: GPUBuffer, n: number,
  ): void {
    const params = this.createParams('decay_beta_params', new Uint32Array([n]).buffer);
    this.dispatch(computeDecayBetaWgsl, 'compute_decay_beta', [
      storageReadOnly(0), storageReadOnly(1), storageReadOnly(2), storageReadOnly(3),
      storageReadWrite(4), storageReadWrite(5), uniform(6),
    ], [
      { binding: 0, resource: { buffer: alpha } },
      { binding: 1, resource: { buffer: beta } },
      { binding: 2, resource: { buffer: ssmA } },
      { binding: 3, resource: { buffer: dtBias } },
      { binding: 4, resource: { buffer: decayOut } },
      { binding: 5, resource: { buffer: betaOut } },
      { binding: 6, resource: { buffer: params } },
    ], [Math.ceil(n / 64)]);
  }

  /** DeltaNet state update + output + per-head RMSNorm. One dispatch for all groups. */
  deltanetStep(
    q: GPUBuffer, k: GPUBuffer, v: GPUBuffer, state: GPUBuffer,
    decay: GPUBuffer, betaVal: GPUBuffer, normWeight: GPUBuffer, output: GPUBuffer,
    numGroups: number, headDim: number, scale: number, normEps: number,
  ): void {
    const paramBuf = new ArrayBuffer(16);
    const u32 = new Uint32Array(paramBuf);
    const f32 = new Float32Array(paramBuf);
    u32[0] = headDim;
    u32[1] = numGroups;
    f32[2] = scale;
    f32[3] = normEps;
    const params = this.createParams('deltanet_step_params', paramBuf);
    this.dispatch(deltanetStepWgsl, 'deltanet_step', [
      storageReadOnly(0), storageReadOnly(1), storageReadOnly(2), storageReadWrite(3),
      storageReadOnly(4), storageReadOnly(5), storageReadOnly(6), storageReadWrite(7), uniform(8),
    ], [
      { binding: 0, resource: { buffer: q } },
      { binding: 1, resource: { buffer: k } },
      { binding: 2, resource: { buffer: v } },
      { binding: 3, resource: { buffer: state } },
      { binding: 4, resource: { buffer: decay } },
      { binding: 5, resource: { buffer: betaVal } },
      { binding: 6, resource: { buffer: normWeight } },
      { binding: 7, resource: { buffer: output } },
      { binding: 8, resource: { buffer: params } },
    ], [numGroups]);
  }

  /** SiLU gate: data[i] *= silu(gate[i]). */
  siluGate(data: GPUBuffer, gate: GPUBuffer, n: number): void {
    const params = this.createParams('silu_gate_params', new Uint32Array([n]).buffer);
    this.dispatch(siluGateWgsl, 'silu_gate', [
      storageReadWrite(0), storageReadOnly(1), uniform(2),
    ], [
      { binding: 0, resource: { buffer: data } },
      { binding: 1, resource: { buffer: gate } },
      { binding: 2, resource: { buffer: params } },
    ], [Math.ceil(n / 256)]);
  }

  // ── Standard Attention GPU ops (Qwen 3.5) ────────────────────────

  /** Deinterleave gated Q: split [attn|gate|attn|gate|...] per head. */
  deinterleaveQ(qFull: GPUBuffer, qAttn: GPUBuffer, qGate: GPUBuffer, numHeads: number, headDim: number): void {
    const params = this.createParams('deinterleave_q_params', new Uint32Array([numHeads, headDim]).buffer);
    this.dispatch(deinterleaveQWgsl, 'deinterleave_q', [
      storageReadOnly(0), storageReadWrite(1), storageReadWrite(2), uniform(3),
    ], [
      { binding: 0, resource: { buffer: qFull } },
      { binding: 1, resource: { buffer: qAttn } },
      { binding: 2, resource: { buffer: qGate } },
      { binding: 3, resource: { buffer: params } },
    ], [Math.ceil(numHeads * headDim / 256)]);
  }

  /** Per-head RMSNorm with shared weight vector. */
  perHeadRmsNorm(data: GPUBuffer, weight: GPUBuffer, numHeads: number, headDim: number, eps: number): void {
    const paramBuf = new Uint32Array(3);
    paramBuf[0] = numHeads;
    paramBuf[1] = headDim;
    paramBuf[2] = new Uint32Array(new Float32Array([eps]).buffer)[0]; // bitcast float to u32
    const params = this.createParams('per_head_rmsnorm_params', paramBuf.buffer);
    this.dispatch(perHeadRmsnormWgsl, 'per_head_rmsnorm', [
      storageReadWrite(0), storageReadOnly(1), uniform(2),
    ], [
      { binding: 0, resource: { buffer: data } },
      { binding: 1, resource: { buffer: weight } },
      { binding: 2, resource: { buffer: params } },
    ], [numHeads]);
  }

  /** Partial RoPE: rotate first ropeDim positions of each head using precomputed tables. */
  partialRope(data: GPUBuffer, numHeads: number, headDim: number, ropeDim: number,
    position: number, theta: number): void {
    const halfDim = ropeDim / 2;
    // Precompute cos/sin tables for this position
    const cacheKey = `partial_rope,${theta},${ropeDim},${position}`;
    let table = this.ropeTableCache.get(cacheKey);
    if (!table) {
      const cosData = new Float32Array(halfDim);
      const sinData = new Float32Array(halfDim);
      for (let i = 0; i < halfDim; i++) {
        const freq = 1.0 / Math.pow(theta, (2 * i) / ropeDim);
        const angle = position * freq;
        cosData[i] = Math.fround(Math.cos(angle));
        sinData[i] = Math.fround(Math.sin(angle));
      }
      table = {
        cos: this.buffers.createBufferWithData(`prope_cos_${cacheKey}`, cosData.buffer),
        sin: this.buffers.createBufferWithData(`prope_sin_${cacheKey}`, sinData.buffer),
      };
      this.ropeTableCache.set(cacheKey, table);
    }
    const params = this.createParams('partial_rope_params', new Uint32Array([numHeads, headDim, ropeDim]).buffer);
    this.dispatch(partialRopeWgsl, 'partial_rope', [
      storageReadWrite(0), storageReadOnly(1), storageReadOnly(2), uniform(3),
    ], [
      { binding: 0, resource: { buffer: data } },
      { binding: 1, resource: { buffer: table.cos } },
      { binding: 2, resource: { buffer: table.sin } },
      { binding: 3, resource: { buffer: params } },
    ], [Math.ceil(numHeads * halfDim / 256)]);
  }

  /** Gated attention: softmax(QK^T/scale) * V * sigmoid(qGate). One workgroup per head. */
  gatedAttention(
    qAttn: GPUBuffer, qGate: GPUBuffer, kCache: GPUBuffer, vCache: GPUBuffer, output: GPUBuffer,
    numHeads: number, numKvHeads: number, keyLen: number, valLen: number,
    maxSeqLen: number, seqLen: number, scale: number,
  ): void {
    const paramBuf = new ArrayBuffer(28);
    const u32 = new Uint32Array(paramBuf);
    u32[0] = numHeads;
    u32[1] = numKvHeads;
    u32[2] = keyLen;
    u32[3] = valLen;
    u32[4] = maxSeqLen;
    u32[5] = seqLen;
    u32[6] = new Uint32Array(new Float32Array([scale]).buffer)[0];
    const params = this.createParams('gated_attn_params', paramBuf);
    this.dispatch(gatedAttentionWgsl, 'gated_attention', [
      storageReadOnly(0), storageReadOnly(1), storageReadOnly(2), storageReadOnly(3),
      storageReadWrite(4), uniform(5),
    ], [
      { binding: 0, resource: { buffer: qAttn } },
      { binding: 1, resource: { buffer: qGate } },
      { binding: 2, resource: { buffer: kCache } },
      { binding: 3, resource: { buffer: vCache } },
      { binding: 4, resource: { buffer: output } },
      { binding: 5, resource: { buffer: params } },
    ], [numHeads]);
  }

  // ── Batched prefill ops ──────────────────────────────────────────

  /** Batched F32 matmul: output[n,m] = sum_k(weights[m,k] * input[n,k]) for N tokens. */
  matmulBatch(weights: GPUBuffer, input: GPUBuffer, output: GPUBuffer, M: number, K: number, N: number): void {
    const paramData = new Uint32Array([M, K, N]);
    const params = this.createParams('matmul_batch_params', paramData.buffer);
    const xGroups = M <= 65535 ? M : 65535;
    const yGroups = M <= 65535 ? 1 : Math.ceil(M / 65535);
    this.dispatch(matmulBatchWgsl, 'matmul_batch', [
      storageReadOnly(0), storageReadOnly(1), storageReadWrite(2), uniform(3),
    ], [
      { binding: 0, resource: { buffer: weights } },
      { binding: 1, resource: { buffer: input } },
      { binding: 2, resource: { buffer: output } },
      { binding: 3, resource: { buffer: params } },
    ], [xGroups, yGroups, N]);
  }

  /** Batched embedding lookup: look up N token IDs at once. weights must be F32. */
  embeddingBatch(weights: GPUBuffer, tokenIds: GPUBuffer, output: GPUBuffer, numTokens: number, embDim: number): void {
    const params = this.createParams('emb_batch_params', new Uint32Array([numTokens, embDim]).buffer);
    this.dispatch(embeddingBatchWgsl, 'embedding_batch', [
      storageReadOnly(0), storageReadOnly(1), storageReadWrite(2), uniform(3),
    ], [
      { binding: 0, resource: { buffer: weights } },
      { binding: 1, resource: { buffer: tokenIds } },
      { binding: 2, resource: { buffer: output } },
      { binding: 3, resource: { buffer: params } },
    ], [Math.ceil(numTokens * embDim / 256)]);
  }

  /** Batched RMSNorm: normalize N rows independently. */
  rmsNormBatch(input: GPUBuffer, weight: GPUBuffer, output: GPUBuffer, dim: number, numRows: number, eps: number): void {
    const paramBuf = new Uint32Array(3);
    paramBuf[0] = dim;
    paramBuf[1] = numRows;
    paramBuf[2] = new Uint32Array(new Float32Array([eps]).buffer)[0];
    const params = this.createParams('rmsnorm_batch_params', paramBuf.buffer);
    this.dispatch(rmsnormBatchWgsl, 'rmsnorm_batch', [
      storageReadOnly(0), storageReadOnly(1), storageReadWrite(2), uniform(3),
    ], [
      { binding: 0, resource: { buffer: input } },
      { binding: 1, resource: { buffer: weight } },
      { binding: 2, resource: { buffer: output } },
      { binding: 3, resource: { buffer: params } },
    ], [numRows]);
  }

  /** Batched causal attention for prefill. Dispatch: [numHeads, numTokens]. */
  attentionBatch(
    q: GPUBuffer, kCache: GPUBuffer, vCache: GPUBuffer, output: GPUBuffer,
    numHeads: number, numKvHeads: number, headDim: number,
    maxSeqLen: number, startPos: number, numTokens: number,
  ): void {
    const scale = 1.0 / Math.sqrt(headDim);
    const paramBuf = new ArrayBuffer(28);
    const u32 = new Uint32Array(paramBuf);
    u32[0] = numHeads; u32[1] = numKvHeads; u32[2] = headDim;
    u32[3] = maxSeqLen; u32[4] = startPos; u32[5] = numTokens;
    u32[6] = new Uint32Array(new Float32Array([scale]).buffer)[0];
    const params = this.createParams('attn_batch_params', paramBuf);
    this.dispatch(attentionBatchWgsl, 'attention_batch', [
      storageReadOnly(0), storageReadOnly(1), storageReadOnly(2), storageReadWrite(3), uniform(4),
    ], [
      { binding: 0, resource: { buffer: q } },
      { binding: 1, resource: { buffer: kCache } },
      { binding: 2, resource: { buffer: vCache } },
      { binding: 3, resource: { buffer: output } },
      { binding: 4, resource: { buffer: params } },
    ], [numHeads, numTokens]);
  }

  /** Repeat-tile: expand numKHeads groups to numVHeads groups by repeating. */
  repeatTile(src: GPUBuffer, dst: GPUBuffer, numKHeads: number, headDim: number, factor: number): void {
    const params = this.createParams('repeat_tile_params', new Uint32Array([numKHeads, headDim, factor]).buffer);
    const total = numKHeads * headDim * factor;
    this.dispatch(repeatTileWgsl, 'repeat_tile', [
      storageReadOnly(0), storageReadWrite(1), uniform(2),
    ], [
      { binding: 0, resource: { buffer: src } },
      { binding: 1, resource: { buffer: dst } },
      { binding: 2, resource: { buffer: params } },
    ], [Math.ceil(total / 256)]);
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
