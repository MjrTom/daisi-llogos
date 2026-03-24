/**
 * Qwen 3.5 hybrid model — DeltaNet (SSM) + standard attention layers.
 *
 * Architecture:
 * - 24 layers, every 4th (3,7,11,15,19,23) is standard attention
 * - Other layers use DeltaNet: fused QKV → conv1d → SiLU → split → L2Norm →
 *   DeltaNet state update → gate → output projection
 * - Both types share: attn_norm → [attention/deltanet] → post_attention_norm → FFN → residual
 */

import { ComputeEngine } from '../gpu/compute.js';
import { GgufModelInfo, GgufTensorInfo } from '../gguf/gguf-parser.js';
import { GgmlType } from '../gguf/quantization.js';
import { KvCache } from './kv-cache.js';

/** CPU-side dequantization (reuse from llama-model) */
function dequantizeToF32(buffer: ArrayBuffer, type: GgmlType, elementCount: number): Float32Array {
  const result = new Float32Array(elementCount);
  const bytes = new Uint8Array(buffer);
  const view = new DataView(buffer);

  if (type === GgmlType.F16) {
    for (let i = 0; i < elementCount; i++) result[i] = f16ToF32(view.getUint16(i * 2, true));
    return result;
  }
  if (type === GgmlType.Q8_0) {
    const blockCount = Math.ceil(elementCount / 32);
    for (let b = 0; b < blockCount; b++) {
      const bo = b * 34;
      const scale = f16ToF32(view.getUint16(bo, true));
      for (let q = 0; q < 32 && b * 32 + q < elementCount; q++) {
        result[b * 32 + q] = scale * view.getInt8(bo + 2 + q);
      }
    }
    return result;
  }
  if (type === GgmlType.F32) {
    return new Float32Array(buffer.slice(0));
  }
  throw new Error(`Unsupported dequant type in Qwen35: ${GgmlType[type]} (${type})`);
}

function f16ToF32(bits: number): number {
  const sign = (bits >> 15) & 1;
  const exp = (bits >> 10) & 0x1F;
  const mant = bits & 0x3FF;
  if (exp === 0) {
    if (mant === 0) return sign ? -0 : 0;
    return (sign ? -1 : 1) * (mant / 1024) * Math.pow(2, -14);
  }
  if (exp === 31) return sign ? -Infinity : Infinity;
  return (sign ? -1 : 1) * (1 + mant / 1024) * Math.pow(2, exp - 15);
}

const GPU_MATMUL_TYPES = new Set([GgmlType.F32, GgmlType.Q4_0, GgmlType.Q8_0]);

interface WeightBuffer { buffer: GPUBuffer; type: GgmlType; }

/** Standard attention layer weights (every 4th layer) */
interface StdAttnWeights {
  kind: 'standard';
  attnNorm: GPUBuffer;
  q: WeightBuffer;
  k: WeightBuffer;
  v: WeightBuffer;
  o: WeightBuffer;
  qNorm?: GPUBuffer;
  kNorm?: GPUBuffer;
  postAttnNorm: GPUBuffer;
  gateProj: WeightBuffer;
  upProj: WeightBuffer;
  downProj: WeightBuffer;
}

/** DeltaNet layer weights (most layers) */
interface DeltaNetWeights {
  kind: 'deltanet';
  attnNorm: GPUBuffer;
  qkv: WeightBuffer;       // fused Q+K+V projection
  gate: Float32Array;        // attention gate [innerSize × E] as F32
  ssmA: Float32Array;       // decay parameters [numVHeads]
  ssmAlpha: Float32Array;    // alpha projection [numVHeads × E] as F32
  ssmBeta: Float32Array;     // beta projection [numVHeads × E] as F32
  ssmConv1d: Float32Array;  // conv kernel [channels × kernelSize]
  ssmDtBias: Float32Array;  // dt bias [numVHeads]
  ssmNorm: Float32Array;    // per-head RMSNorm weight [headDim]
  ssmOut: WeightBuffer;     // output projection
  postAttnNorm: GPUBuffer;
  gateProj: WeightBuffer;
  upProj: WeightBuffer;
  downProj: WeightBuffer;
}

type LayerWeights = StdAttnWeights | DeltaNetWeights;

/** DeltaNet state per layer — [numVHeads × headDim × headDim] */
interface DeltaNetState {
  state: Float32Array;         // [numVHeads * headDim * headDim]
  convBuffer: Float32Array;    // [convChannels * (kernelSize - 1)]
}

export class Qwen35Model {
  private compute: ComputeEngine;
  private info: GgufModelInfo;
  private weights!: { tokenEmbedding: GPUBuffer; outputNorm: GPUBuffer; output: GPUBuffer; layers: LayerWeights[] };
  private kvCaches!: (KvCache | null)[]; // null for DeltaNet layers
  private deltaStates!: (DeltaNetState | null)[]; // null for standard attn layers

  // Working buffers
  private hidden!: GPUBuffer;
  private residual!: GPUBuffer;
  private normed!: GPUBuffer;
  private qBuf!: GPUBuffer;
  private kBuf!: GPUBuffer;
  private vBuf!: GPUBuffer;
  private attnOut!: GPUBuffer;
  private gateBuf!: GPUBuffer;
  private upBuf!: GPUBuffer;
  private ffnOut!: GPUBuffer;
  private temp!: GPUBuffer;
  private logits!: GPUBuffer;
  // DeltaNet-specific
  private qkvBuf!: GPUBuffer;
  private ssmGateBuf!: GPUBuffer;

  // Config
  private fullAttnInterval: number;
  private ssmInnerSize: number;
  private ssmStateSize: number;
  private ssmGroupCount: number;
  private ssmConvKernel: number;
  private ssmNumVHeads: number;
  private ssmHeadDim: number;

  constructor(compute: ComputeEngine, info: GgufModelInfo) {
    this.compute = compute;
    this.info = info;
    const prefix = info.architecture;
    const meta = info.metadata;
    this.fullAttnInterval = (meta.get(`${prefix}.full_attention_interval`) as number) ?? 4;
    this.ssmInnerSize = (meta.get(`${prefix}.ssm.inner_size`) as number) ?? 2048;
    this.ssmStateSize = (meta.get(`${prefix}.ssm.state_size`) as number) ?? 128;
    this.ssmGroupCount = (meta.get(`${prefix}.ssm.group_count`) as number) ?? 16;
    this.ssmConvKernel = (meta.get(`${prefix}.ssm.conv_kernel`) as number) ?? 4;
    // numVHeads derived from alpha weight dims later
    this.ssmNumVHeads = 0;
    this.ssmHeadDim = this.ssmStateSize;
  }

  get embeddingDim(): number { return this.info.embeddingLength; }
  get numLayers(): number { return this.info.blockCount; }
  get numHeads(): number { return this.info.headCount; }
  get numKvHeads(): number { return this.info.headCountKv || this.info.headCount; }
  get headDim(): number {
    const kl = this.info.metadata.get(`${this.info.architecture}.attention.key_length`) as number;
    return kl ? kl / this.numKvHeads : this.embeddingDim / this.numHeads;
  }
  get ffnDim(): number { return this.info.feedForwardLength; }
  get vocabSize(): number { return this.info.vocabSize; }
  get ropeTheta(): number { return this.info.ropeFreqBase; }
  get ropeDim(): number {
    return (this.info.metadata.get(`${this.info.architecture}.rope.dimension_count`) as number) || this.headDim;
  }
  get rmsNormEps(): number { return this.info.rmsNormEps; }
  get position(): number { return this._position; }
  private _position = 0;

  isStandardAttention(layer: number): boolean {
    return (layer + 1) % this.fullAttnInterval === 0;
  }

  async initWeights(tensorMap: Map<string, { buffer: ArrayBuffer; info: GgufTensorInfo }>): Promise<void> {
    const { compute } = this;

    const uploadAsF32 = (name: string): GPUBuffer => {
      const tensor = tensorMap.get(name);
      if (!tensor) throw new Error(`Missing tensor: ${name}`);
      if (tensor.info.type === GgmlType.F32) return compute.buffers.createBufferWithData(name, tensor.buffer);
      const f32 = dequantizeToF32(tensor.buffer, tensor.info.type, tensor.info.elementCount);
      return compute.buffers.createBufferWithData(name, f32.buffer);
    };

    const uploadWeight = (name: string): WeightBuffer => {
      const tensor = tensorMap.get(name);
      if (!tensor) throw new Error(`Missing tensor: ${name}`);
      if (GPU_MATMUL_TYPES.has(tensor.info.type)) {
        return { buffer: compute.buffers.createBufferWithData(name, tensor.buffer), type: tensor.info.type };
      }
      const f32 = dequantizeToF32(tensor.buffer, tensor.info.type, tensor.info.elementCount);
      return { buffer: compute.buffers.createBufferWithData(name, f32.buffer), type: GgmlType.F32 };
    };

    const getF32 = (name: string): Float32Array => {
      const tensor = tensorMap.get(name);
      if (!tensor) throw new Error(`Missing tensor: ${name}`);
      if (tensor.info.type === GgmlType.F32) return new Float32Array(tensor.buffer.slice(0));
      return dequantizeToF32(tensor.buffer, tensor.info.type, tensor.info.elementCount);
    };

    const tryUploadAsF32 = (name: string): GPUBuffer | undefined => {
      return tensorMap.has(name) ? uploadAsF32(name) : undefined;
    };

    // Derive numVHeads from alpha weight
    const alpha0 = tensorMap.get('blk.0.ssm_alpha.weight');
    if (alpha0) {
      this.ssmNumVHeads = alpha0.info.elementCount / this.embeddingDim;
    }

    // Global weights
    const tokenEmbedding = uploadAsF32('token_embd.weight');
    const output = tensorMap.has('output.weight') ? uploadAsF32('output.weight') : tokenEmbedding;
    const outputNorm = uploadAsF32('output_norm.weight');

    // Layer weights
    const layers: LayerWeights[] = [];
    for (let i = 0; i < this.numLayers; i++) {
      const shared = {
        postAttnNorm: uploadAsF32(`blk.${i}.post_attention_norm.weight`),
        gateProj: uploadWeight(`blk.${i}.ffn_gate.weight`),
        upProj: uploadWeight(`blk.${i}.ffn_up.weight`),
        downProj: uploadWeight(`blk.${i}.ffn_down.weight`),
      };

      if (this.isStandardAttention(i)) {
        layers.push({
          kind: 'standard',
          attnNorm: uploadAsF32(`blk.${i}.attn_norm.weight`),
          q: uploadWeight(`blk.${i}.attn_q.weight`),
          k: uploadWeight(`blk.${i}.attn_k.weight`),
          v: uploadWeight(`blk.${i}.attn_v.weight`),
          o: uploadWeight(`blk.${i}.attn_output.weight`),
          qNorm: tryUploadAsF32(`blk.${i}.attn_q_norm.weight`),
          kNorm: tryUploadAsF32(`blk.${i}.attn_k_norm.weight`),
          ...shared,
        });
      } else {
        layers.push({
          kind: 'deltanet',
          attnNorm: uploadAsF32(`blk.${i}.attn_norm.weight`),
          qkv: uploadWeight(`blk.${i}.attn_qkv.weight`),
          gate: getF32(`blk.${i}.attn_gate.weight`),
          ssmA: getF32(`blk.${i}.ssm_a`),
          ssmAlpha: getF32(`blk.${i}.ssm_alpha.weight`),
          ssmBeta: getF32(`blk.${i}.ssm_beta.weight`),
          ssmConv1d: getF32(`blk.${i}.ssm_conv1d.weight`),
          ssmDtBias: getF32(`blk.${i}.ssm_dt.bias`),
          ssmNorm: getF32(`blk.${i}.ssm_norm.weight`),
          ssmOut: uploadWeight(`blk.${i}.ssm_out.weight`),
          ...shared,
        });
      }
    }

    this.weights = { tokenEmbedding, outputNorm, output, layers };

    // Allocate working buffers
    const E = this.embeddingDim;
    const F = this.ffnDim;
    const H = this.numHeads * this.headDim;
    const KV = this.numKvHeads * this.headDim;

    this.hidden = compute.buffers.createBuffer('hidden', E * 4);
    this.residual = compute.buffers.createBuffer('residual', E * 4);
    this.normed = compute.buffers.createBuffer('normed', E * 4);
    this.qBuf = compute.buffers.createBuffer('q_proj', H * 4);
    this.kBuf = compute.buffers.createBuffer('k_proj', KV * 4);
    this.vBuf = compute.buffers.createBuffer('v_proj', KV * 4);
    this.attnOut = compute.buffers.createBuffer('attn_out', E * 4);
    this.gateBuf = compute.buffers.createBuffer('gate', F * 4);
    this.upBuf = compute.buffers.createBuffer('up', F * 4);
    this.ffnOut = compute.buffers.createBuffer('ffn_out', F * 4);
    this.temp = compute.buffers.createBuffer('temp', E * 4);
    this.logits = compute.buffers.createBuffer('logits', this.vocabSize * 4);
    // DeltaNet buffers
    const qkvDim = this.ssmInnerSize * 3; // Q+K+V each = ssmInnerSize
    this.qkvBuf = compute.buffers.createBuffer('qkv', qkvDim * 4);
    this.ssmGateBuf = compute.buffers.createBuffer('ssm_gate', this.ssmInnerSize * 4);

    // KV caches for standard attention layers
    const maxCtx = Math.min(this.info.contextLength, 4096);
    this.kvCaches = [];
    this.deltaStates = [];
    for (let i = 0; i < this.numLayers; i++) {
      if (this.isStandardAttention(i)) {
        this.kvCaches.push(new KvCache(compute.device, compute.buffers, this.numKvHeads, this.headDim, maxCtx, i));
        this.deltaStates.push(null);
      } else {
        this.kvCaches.push(null);
        this.deltaStates.push({
          state: new Float32Array(this.ssmNumVHeads * this.ssmHeadDim * this.ssmHeadDim),
          convBuffer: new Float32Array(qkvDim * (this.ssmConvKernel - 1)),
        });
      }
    }
  }

  resetCache(): void {
    this._position = 0;
    for (const kv of this.kvCaches) kv?.reset();
    for (const ds of this.deltaStates) {
      if (ds) {
        ds.state.fill(0);
        ds.convBuffer.fill(0);
      }
    }
  }

  /**
   * Forward pass for a single token.
   */
  async forward(tokenId: number): Promise<GPUBuffer> {
    const { compute, weights } = this;
    const E = this.embeddingDim;

    // 1. Token embedding
    compute.embedding(weights.tokenEmbedding, this.hidden, tokenId, E);

    // DEBUG: Check embedding output
    if (this._position === 0) {
      const h = new Float32Array(await this.readGpuBuffer(this.hidden, E * 4));
      let hMax = 0; for (let i = 0; i < E; i++) hMax = Math.max(hMax, Math.abs(h[i]));
      console.log(`  [embed] token=${tokenId} max=${hMax.toFixed(4)} h[0..4]=${Array.from(h.slice(0, 5)).map(v => v.toFixed(4))}`);
    }

    // Transformer layers
    for (let layer = 0; layer < this.numLayers; layer++) {
      const lw = weights.layers[layer];

      // Pre-attention RMSNorm + residual copy
      compute.copyAndRmsNorm(this.hidden, lw.attnNorm, this.normed, this.residual, E, this.rmsNormEps);

      if (lw.kind === 'standard') {
        await this.forwardStandardAttention(layer, lw);
      } else {
        await this.forwardDeltaNet(layer, lw);
      }

      // Post-attention RMSNorm + residual copy
      compute.copyAndRmsNorm(this.hidden, lw.postAttnNorm, this.normed, this.residual, E, this.rmsNormEps);

      // FFN: SwiGLU
      compute.matmul(lw.gateProj.buffer, this.normed, this.gateBuf, this.ffnDim, E, lw.gateProj.type);
      compute.matmul(lw.upProj.buffer, this.normed, this.upBuf, this.ffnDim, E, lw.upProj.type);
      compute.siluMul(this.gateBuf, this.upBuf, this.ffnOut, this.ffnDim);
      compute.matmul(lw.downProj.buffer, this.ffnOut, this.temp, E, this.ffnDim, lw.downProj.type);

      // Residual add
      compute.add(this.temp, this.residual, this.hidden, E);
    }

    // Final RMSNorm + LM Head
    if (this._position === 0) {
      const h = new Float32Array(await this.readGpuBuffer(this.hidden, E * 4));
      let hMax = 0, hSum = 0; for (let i = 0; i < E; i++) { hMax = Math.max(hMax, Math.abs(h[i])); hSum += h[i]; }
      console.log(`  [final] hidden max=${hMax.toFixed(4)} sum=${hSum.toFixed(4)} h[0..4]=${Array.from(h.slice(0, 5)).map(v => v.toFixed(4))}`);
    }
    compute.rmsNorm(this.hidden, weights.outputNorm, this.normed, E, this.rmsNormEps);
    compute.matmul(weights.output, this.normed, this.logits, this.vocabSize, E, GgmlType.F32);

    this._position++;
    return this.logits;
  }

  private async forwardStandardAttention(layer: number, lw: StdAttnWeights): Promise<void> {
    const { compute } = this;
    const E = this.embeddingDim;
    const kvCache = this.kvCaches[layer]!;

    // Q, K, V projections
    compute.matmul(lw.q.buffer, this.normed, this.qBuf, this.numHeads * this.headDim, E, lw.q.type);
    compute.matmul(lw.k.buffer, this.normed, this.kBuf, this.numKvHeads * this.headDim, E, lw.k.type);
    compute.matmul(lw.v.buffer, this.normed, this.vBuf, this.numKvHeads * this.headDim, E, lw.v.type);

    // QK norms (per-head RMSNorm) — required for Qwen 3.5 standard attention layers
    if (lw.qNorm) {
      const qData = new Float32Array(await this.readGpuBuffer(this.qBuf, this.numHeads * this.headDim * 4));
      const kData = new Float32Array(await this.readGpuBuffer(this.kBuf, this.numKvHeads * this.headDim * 4));
      const qNormW = new Float32Array(await this.readGpuBuffer(lw.qNorm, this.numKvHeads * this.headDim * 4));
      const kNormW = lw.kNorm ? new Float32Array(await this.readGpuBuffer(lw.kNorm, this.numKvHeads * this.headDim * 4)) : qNormW;

      // Apply per-head RMSNorm to Q (using KV-group-shared weights)
      for (let h = 0; h < this.numHeads; h++) {
        const off = h * this.headDim;
        const kvGroup = Math.floor(h / (this.numHeads / this.numKvHeads));
        const wOff = kvGroup * this.headDim;
        let sumSq = 0;
        for (let i = 0; i < this.headDim; i++) sumSq += qData[off + i] * qData[off + i];
        const rms = Math.sqrt(sumSq / this.headDim + this.rmsNormEps);
        for (let i = 0; i < this.headDim; i++) qData[off + i] = (qData[off + i] / rms) * qNormW[wOff + i];
      }
      // Apply per-head RMSNorm to K
      for (let h = 0; h < this.numKvHeads; h++) {
        const off = h * this.headDim;
        let sumSq = 0;
        for (let i = 0; i < this.headDim; i++) sumSq += kData[off + i] * kData[off + i];
        const rms = Math.sqrt(sumSq / this.headDim + this.rmsNormEps);
        for (let i = 0; i < this.headDim; i++) kData[off + i] = (kData[off + i] / rms) * kNormW[off + i];
      }
      compute.device.queue.writeBuffer(this.qBuf, 0, qData.buffer);
      compute.device.queue.writeBuffer(this.kBuf, 0, kData.buffer);
    }

    // RoPE
    const pos = kvCache.seqLen;
    // Partial RoPE: only first ropeDim positions per head get rotated (CPU for partial support)
    if (this.ropeDim < this.headDim) {
      const qData = new Float32Array(await this.readGpuBuffer(this.qBuf, this.numHeads * this.headDim * 4));
      const kData = new Float32Array(await this.readGpuBuffer(this.kBuf, this.numKvHeads * this.headDim * 4));
      this.applyPartialRoPE(qData, this.numHeads, pos);
      this.applyPartialRoPE(kData, this.numKvHeads, pos);
      compute.device.queue.writeBuffer(this.qBuf, 0, qData.buffer);
      compute.device.queue.writeBuffer(this.kBuf, 0, kData.buffer);
    } else {
      compute.rope(this.qBuf, this.headDim, this.headDim, pos, this.ropeTheta, this.numHeads * this.headDim);
      compute.rope(this.kBuf, this.headDim, this.headDim, pos, this.ropeTheta, this.numKvHeads * this.headDim);
    }

    // KV cache write
    kvCache.write(this.kBuf, this.vBuf, this.numKvHeads * this.headDim * 4, this.numKvHeads * this.headDim * 4);

    // Attention
    compute.attention(this.qBuf, kvCache.kBuffer, kvCache.vBuffer, this.attnOut,
      this.numHeads, this.numKvHeads, this.headDim, kvCache.seqLen, kvCache.maxSeqLen);

    // Output projection + residual
    compute.matmul(lw.o.buffer, this.attnOut, this.temp, E, this.numHeads * this.headDim, lw.o.type);
    compute.add(this.temp, this.residual, this.hidden, E);
  }

  private async forwardDeltaNet(layer: number, lw: DeltaNetWeights): Promise<void> {
    const { compute } = this;
    const E = this.embeddingDim;
    const ds = this.deltaStates[layer]!;
    const innerSize = this.ssmInnerSize;
    const qkvDim = innerSize * 3;
    const numVHeads = this.ssmNumVHeads;
    const numKHeads = this.ssmGroupCount;
    const headDim = this.ssmHeadDim;
    const kernelSize = this.ssmConvKernel;

    // 1. QKV projection (GPU)
    compute.matmul(lw.qkv.buffer, this.normed, this.qkvBuf, qkvDim, E, lw.qkv.type);

    // Read QKV to CPU for sequential DeltaNet operations
    const qkvData = new Float32Array(await this.readGpuBuffer(this.qkvBuf, qkvDim * 4));

    // 2. Causal Conv1d (CPU)
    this.causalConv1d(qkvData, ds.convBuffer, lw.ssmConv1d, qkvDim, kernelSize);

    // DEBUG: check QKV before SiLU
    const dbg = this._position === 0 && layer === 0;
    if (dbg) {
      let qkvMax = 0; for (let i = 0; i < qkvDim; i++) qkvMax = Math.max(qkvMax, Math.abs(qkvData[i]));
      console.log(`  [L0] after conv1d: qkv max=${qkvMax.toFixed(4)} qkv[0..4]=${Array.from(qkvData.slice(0, 5)).map(v => v.toFixed(4))}`);
    }

    // 3. SiLU activation (in-place)
    for (let i = 0; i < qkvDim; i++) {
      qkvData[i] = qkvData[i] / (1 + Math.exp(-qkvData[i]));
    }

    if (dbg) {
      let qkvMax = 0; for (let i = 0; i < qkvDim; i++) qkvMax = Math.max(qkvMax, Math.abs(qkvData[i]));
      console.log(`  [L0] after SiLU: qkv max=${qkvMax.toFixed(4)} qkv[0..4]=${Array.from(qkvData.slice(0, 5)).map(v => v.toFixed(4))}`);
    }

    // 4. Split Q/K/V — equal split
    const q = qkvData.subarray(0, innerSize);
    const k = qkvData.subarray(innerSize, innerSize * 2);
    const v = qkvData.subarray(innerSize * 2, innerSize * 3);

    if (dbg) {
      let qMax = 0, kMax = 0, vMax = 0;
      for (let i = 0; i < innerSize; i++) { qMax = Math.max(qMax, Math.abs(q[i])); kMax = Math.max(kMax, Math.abs(k[i])); vMax = Math.max(vMax, Math.abs(v[i])); }
      console.log(`  [L0] after split: Q max=${qMax.toFixed(4)} K max=${kMax.toFixed(4)} V max=${vMax.toFixed(4)}`);
    }

    // 5. L2 normalize Q and K per group (numKHeads groups)
    const groupDim = innerSize / numKHeads;
    for (let g = 0; g < numKHeads; g++) {
      const off = g * groupDim;
      let normSq = 0;
      for (let i = 0; i < groupDim; i++) normSq += q[off + i] * q[off + i];
      const qNorm = Math.sqrt(normSq) || 1;
      for (let i = 0; i < groupDim; i++) q[off + i] /= qNorm;

      normSq = 0;
      for (let i = 0; i < groupDim; i++) normSq += k[off + i] * k[off + i];
      const kNorm = Math.sqrt(normSq) || 1;
      for (let i = 0; i < groupDim; i++) k[off + i] /= kNorm;
    }

    if (dbg) {
      let qMax = 0, kMax = 0;
      for (let i = 0; i < innerSize; i++) { qMax = Math.max(qMax, Math.abs(q[i])); kMax = Math.max(kMax, Math.abs(k[i])); }
      console.log(`  [L0] after L2 norm: Q max=${qMax.toFixed(4)} K max=${kMax.toFixed(4)}`);
    }

    // 6. Repeat-tile Q and K if numVHeads > numKHeads
    let qExpanded = q;
    let kExpanded = k;
    if (numVHeads > numKHeads) {
      const factor = numVHeads / numKHeads;
      qExpanded = new Float32Array(numVHeads * headDim);
      kExpanded = new Float32Array(numVHeads * headDim);
      for (let rep = 0; rep < factor; rep++) {
        qExpanded.set(q.subarray(0, numKHeads * headDim), rep * numKHeads * headDim);
        kExpanded.set(k.subarray(0, numKHeads * headDim), rep * numKHeads * headDim);
      }
    }

    // 7. Alpha and Beta projections
    const normData = new Float32Array(await this.readGpuBuffer(this.normed, E * 4));
    const alpha = new Float32Array(numVHeads);
    const beta = new Float32Array(numVHeads);

    // CPU matvec for small projections (already F32 on CPU)
    const alphaW = lw.ssmAlpha;
    const betaW = lw.ssmBeta;
    for (let h = 0; h < numVHeads; h++) {
      let aSum = 0, bSum = 0;
      for (let j = 0; j < E; j++) {
        aSum += alphaW[h * E + j] * normData[j];
        bSum += betaW[h * E + j] * normData[j];
      }
      alpha[h] = aSum;
      beta[h] = bSum;
    }

    // 8. Compute decay and beta values
    const decay = new Float32Array(numVHeads);
    const betaVal = new Float32Array(numVHeads);
    for (let g = 0; g < numVHeads; g++) {
      const softplus = Math.log(1 + Math.exp(alpha[g] + lw.ssmDtBias[g]));
      decay[g] = Math.exp(lw.ssmA[g] * softplus);
      betaVal[g] = 1.0 / (1.0 + Math.exp(-beta[g]));
    }

    // 9. DeltaNet state update + output
    const output = new Float32Array(numVHeads * headDim);
    const scale = 1.0 / Math.sqrt(headDim);

    for (let g = 0; g < numVHeads; g++) {
      const baseOff = g * headDim;
      const stateOff = g * headDim * headDim;

      // sk = S^T * k
      const sk = new Float32Array(headDim);
      for (let j = 0; j < headDim; j++) {
        let sum = 0;
        for (let i = 0; i < headDim; i++) {
          sum += ds.state[stateOff + i * headDim + j] * kExpanded[baseOff + i];
        }
        sk[j] = sum;
      }

      // error = (v - decay*sk) * beta
      const error = new Float32Array(headDim);
      for (let j = 0; j < headDim; j++) {
        error[j] = (v[baseOff + j] - decay[g] * sk[j]) * betaVal[g];
      }

      // State update: S = decay*S + k⊗error
      for (let i = 0; i < headDim; i++) {
        for (let j = 0; j < headDim; j++) {
          ds.state[stateOff + i * headDim + j] =
            decay[g] * ds.state[stateOff + i * headDim + j] +
            kExpanded[baseOff + i] * error[j];
        }
      }

      // Output: o = S^T * q * scale
      for (let j = 0; j < headDim; j++) {
        let sum = 0;
        for (let i = 0; i < headDim; i++) {
          sum += ds.state[stateOff + i * headDim + j] * qExpanded[baseOff + i];
        }
        output[baseOff + j] = sum * scale;
      }

      // Per-head RMSNorm
      let sumSq = 0;
      for (let j = 0; j < headDim; j++) sumSq += output[baseOff + j] * output[baseOff + j];
      const rms = Math.sqrt(sumSq / headDim + this.rmsNormEps);
      for (let j = 0; j < headDim; j++) {
        output[baseOff + j] = (output[baseOff + j] / rms) * lw.ssmNorm[j];
      }
    }

    if (dbg) {
      let oMax = 0; for (let i = 0; i < numVHeads * headDim; i++) oMax = Math.max(oMax, Math.abs(output[i]));
      console.log(`  [L0] after DeltaNetStep: output max=${oMax.toFixed(4)} output[0..4]=${Array.from(output.slice(0, 5)).map(v => v.toFixed(4))}`);
    }

    // 10. Gate: output *= silu(gate(normOut))
    const gateData = new Float32Array(innerSize);
    const gateW = lw.gate;
    for (let h = 0; h < innerSize; h++) {
      let sum = 0;
      for (let j = 0; j < E; j++) sum += gateW[h * E + j] * normData[j];
      const silu = sum / (1 + Math.exp(-sum));
      gateData[h] = silu;
    }

    // DEBUG
    if (this._position === 0 && layer === 0) {
      let oMax = 0, gMax = 0;
      for (let i = 0; i < innerSize; i++) { oMax = Math.max(oMax, Math.abs(output[i])); gMax = Math.max(gMax, Math.abs(gateData[i])); }
      console.log(`  [L0 gate] output max BEFORE gate=${oMax.toFixed(4)} gate max=${gMax.toFixed(4)}`);
      console.log(`  [L0 gate] output[0..4]=${Array.from(output.slice(0, 5)).map(v => v.toFixed(4))}`);
      console.log(`  [L0 gate] gate[0..4]=${Array.from(gateData.slice(0, 5)).map(v => v.toFixed(4))}`);
    }

    for (let i = 0; i < innerSize; i++) {
      output[i] *= gateData[i];
    }

    if (this._position === 0 && layer === 0) {
      let oMax = 0;
      for (let i = 0; i < innerSize; i++) oMax = Math.max(oMax, Math.abs(output[i]));
      console.log(`  [L0 gate] output max AFTER gate=${oMax.toFixed(4)}`);
    }

    // 11. Output projection (GPU)
    compute.device.queue.writeBuffer(this.ssmGateBuf, 0, output.buffer, 0, innerSize * 4);
    compute.matmul(lw.ssmOut.buffer, this.ssmGateBuf, this.temp, E, innerSize, lw.ssmOut.type);

    // Residual add
    compute.add(this.temp, this.residual, this.hidden, E);

    // DEBUG: Check hidden state after DeltaNet
    if (layer === 0 && this._position < 3) {
      const stateNorm = Math.sqrt(ds.state.reduce((s, v) => s + v * v, 0));
      console.log(`  [L0 pos=${this._position}] state L2 norm=${stateNorm.toFixed(4)}`);
    }
    if (this._position === 0 && layer === 0) {
      const h = new Float32Array(await this.readGpuBuffer(this.hidden, E * 4));
      const t = new Float32Array(await this.readGpuBuffer(this.temp, E * 4));
      let hMax = 0, tMax = 0;
      for (let i = 0; i < E; i++) { hMax = Math.max(hMax, Math.abs(h[i])); tMax = Math.max(tMax, Math.abs(t[i])); }
      console.log(`  [L0 DN] temp max=${tMax.toFixed(4)} hidden max=${hMax.toFixed(4)} temp[0..4]=${Array.from(t.slice(0, 5)).map(v => v.toFixed(4))}`);
      // Also check ssmGateBuf (output before final projection)
      const g = new Float32Array(await this.readGpuBuffer(this.ssmGateBuf, innerSize * 4));
      let gMax = 0, gNonZero = 0;
      for (let i = 0; i < innerSize; i++) { gMax = Math.max(gMax, Math.abs(g[i])); if (g[i] !== 0) gNonZero++; }
      console.log(`  [L0 DN] ssmGateBuf max=${gMax.toFixed(4)} nonzero=${gNonZero}/${innerSize} first5=${Array.from(g.slice(0, 5)).map(v => v.toFixed(4))}`);
    }
  }

  /** Apply RoPE to first ropeDim dims of each head, leaving rest unchanged. */
  private applyPartialRoPE(data: Float32Array, nHeads: number, position: number): void {
    const headDim = this.headDim;
    const ropeDim = this.ropeDim;
    const halfDim = ropeDim / 2;
    const theta = this.ropeTheta;

    for (let h = 0; h < nHeads; h++) {
      const off = h * headDim;
      for (let i = 0; i < halfDim; i++) {
        const freq = 1.0 / Math.pow(theta, (2 * i) / ropeDim);
        const angle = position * freq;
        const cos = Math.cos(angle);
        const sin = Math.sin(angle);
        const x0 = data[off + 2 * i];
        const x1 = data[off + 2 * i + 1];
        data[off + 2 * i] = x0 * cos - x1 * sin;
        data[off + 2 * i + 1] = x0 * sin + x1 * cos;
      }
      // Dimensions beyond ropeDim are left unchanged
    }
  }

  private causalConv1d(data: Float32Array, convBuf: Float32Array, weights: Float32Array, channels: number, kernelSize: number): void {
    const bufSlots = kernelSize - 1;

    // Compute all channel outputs into separate array (C# reference: CpuBackend.cs:482)
    const result = new Float32Array(channels);
    for (let c = 0; c < channels; c++) {
      let s = 0;
      for (let k = 0; k < bufSlots; k++) {
        s += convBuf[k * channels + c] * weights[c * kernelSize + k];
      }
      s += data[c] * weights[c * kernelSize + bufSlots];
      result[c] = s;
    }

    // Shift buffer: discard oldest row, add current pre-conv values as newest
    for (let k = 0; k < bufSlots - 1; k++) {
      for (let c = 0; c < channels; c++) {
        convBuf[k * channels + c] = convBuf[(k + 1) * channels + c];
      }
    }
    if (bufSlots > 0) {
      for (let c = 0; c < channels; c++) {
        convBuf[(bufSlots - 1) * channels + c] = data[c];
      }
    }

    // Write result back
    data.set(result);
  }

  /** Read a GPU buffer to CPU — creates a fresh readback buffer each time to avoid mapping conflicts. */
  private async readGpuBuffer(buffer: GPUBuffer, size: number): Promise<ArrayBuffer> {
    const readback = this.compute.device.createBuffer({
      size,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    const encoder = this.compute.device.createCommandEncoder();
    encoder.copyBufferToBuffer(buffer, 0, readback, 0, size);
    this.compute.device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const data = readback.getMappedRange().slice(0);
    readback.unmap();
    readback.destroy();
    return data;
  }

  async readLogits(): Promise<Float32Array> {
    return new Float32Array(await this.compute.readBuffer(this.logits, this.vocabSize * 4));
  }
}
