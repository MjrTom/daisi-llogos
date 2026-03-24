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
  qkv: WeightBuffer;         // fused Q+K+V projection
  gate: WeightBuffer;         // attention gate projection
  ssmA: GPUBuffer;            // decay parameters [numVHeads]
  ssmAlpha: WeightBuffer;     // alpha projection [numVHeads × E]
  ssmBeta: WeightBuffer;      // beta projection [numVHeads × E]
  ssmConv1d: GPUBuffer;       // conv kernel [channels × kernelSize] as F32
  ssmDtBias: GPUBuffer;       // dt bias [numVHeads]
  ssmNorm: GPUBuffer;         // per-head RMSNorm weight [headDim]
  ssmOut: WeightBuffer;       // output projection
  postAttnNorm: GPUBuffer;
  gateProj: WeightBuffer;
  upProj: WeightBuffer;
  downProj: WeightBuffer;
}

type LayerWeights = StdAttnWeights | DeltaNetWeights;

/** DeltaNet GPU state per layer */
interface DeltaNetGpuState {
  state: GPUBuffer;     // [numVHeads * headDim * headDim]
  convBuf: GPUBuffer;   // [convChannels * (kernelSize - 1)]
}

export class Qwen35Model {
  private compute: ComputeEngine;
  private info: GgufModelInfo;
  private weights!: { tokenEmbedding: WeightBuffer; outputNorm: GPUBuffer; output: WeightBuffer; layers: LayerWeights[] };
  private qkvDims!: number[]; // per-layer QKV output dimension (from weight)
  private kvCaches!: (KvCache | null)[]; // null for DeltaNet layers
  private deltaStates!: (DeltaNetGpuState | null)[]; // null for standard attn layers

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
  // Standard attention GPU buffers (for gated Q)
  private qAttnBuf!: GPUBuffer;
  private qGateBufAttn!: GPUBuffer;
  // DeltaNet-specific GPU buffers
  private qTiledBuf!: GPUBuffer;  // tiled Q for deltanet [numVHeads * headDim]
  private kTiledBuf!: GPUBuffer;  // tiled K for deltanet [numVHeads * headDim]
  private qkvBuf!: GPUBuffer;
  private ssmGateBuf!: GPUBuffer;
  private ssmOutputBuf!: GPUBuffer;
  private alphaBuf!: GPUBuffer;
  private betaBuf!: GPUBuffer;
  private decayBuf!: GPUBuffer;
  private betaValBuf!: GPUBuffer;

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
    // For DeltaNet: ssmStateSize = 128
    return this.ssmHeadDim;
  }
  /** Per-head key/value dimension for standard attention */
  get keyLength(): number {
    return (this.info.metadata.get(`${this.info.architecture}.attention.key_length`) as number) || (this.embeddingDim / this.numHeads);
  }
  get valueLength(): number {
    return (this.info.metadata.get(`${this.info.architecture}.attention.value_length`) as number) || this.keyLength;
  }
  /** Whether Q projection is gated (Q dim = 2 × keyLength per head) */
  get hasGatedQ(): boolean {
    // Check if Q weight has double the expected rows
    const qTensor = this.info.tensors.find(t => t.name === 'blk.3.attn_q.weight');
    if (!qTensor) return false;
    const qRows = qTensor.elementCount / this.embeddingDim;
    const expectedQ = this.numHeads * this.keyLength;
    return qRows > expectedQ * 1.5; // gated Q = 2× expected
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

    // Global weights — keep embedding in native format (avoids huge F32 allocation)
    const tokenEmbedding = uploadWeight('token_embd.weight');
    const output = tensorMap.has('output.weight') ? uploadWeight('output.weight') : tokenEmbedding;
    const outputNorm = uploadAsF32('output_norm.weight');

    // Layer weights
    this.qkvDims = [];
    const layers: LayerWeights[] = [];
    for (let i = 0; i < this.numLayers; i++) {
      const shared = {
        postAttnNorm: uploadAsF32(`blk.${i}.post_attention_norm.weight`),
        gateProj: uploadWeight(`blk.${i}.ffn_gate.weight`),
        upProj: uploadWeight(`blk.${i}.ffn_up.weight`),
        downProj: uploadWeight(`blk.${i}.ffn_down.weight`),
      };

      if (this.isStandardAttention(i)) {
        this.qkvDims.push(0); // not used for standard attention
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
        const qkvTensor = tensorMap.get(`blk.${i}.attn_qkv.weight`)!;
        this.qkvDims.push(qkvTensor.info.elementCount / this.embeddingDim);
        layers.push({
          kind: 'deltanet',
          attnNorm: uploadAsF32(`blk.${i}.attn_norm.weight`),
          qkv: uploadWeight(`blk.${i}.attn_qkv.weight`),
          gate: uploadWeight(`blk.${i}.attn_gate.weight`),
          ssmA: uploadAsF32(`blk.${i}.ssm_a`),
          ssmAlpha: uploadWeight(`blk.${i}.ssm_alpha.weight`),
          ssmBeta: uploadWeight(`blk.${i}.ssm_beta.weight`),
          ssmConv1d: uploadAsF32(`blk.${i}.ssm_conv1d.weight`),
          ssmDtBias: uploadAsF32(`blk.${i}.ssm_dt.bias`),
          ssmNorm: uploadAsF32(`blk.${i}.ssm_norm.weight`),
          ssmOut: uploadWeight(`blk.${i}.ssm_out.weight`),
          ...shared,
        });
      }
    }

    this.weights = { tokenEmbedding, outputNorm, output, layers };

    // Allocate working buffers
    const E = this.embeddingDim;
    const F = this.ffnDim;
    // Standard attention dims (may differ from DeltaNet headDim)
    const qFullDim = this.hasGatedQ ? this.numHeads * this.keyLength * 2 : this.numHeads * this.keyLength;
    const kDim = this.numKvHeads * this.keyLength;
    const vDim = this.numKvHeads * this.valueLength;
    const H = Math.max(this.numHeads * this.headDim, qFullDim, this.ssmInnerSize); // largest Q dim
    const KV = Math.max(this.numKvHeads * this.headDim, kDim, this.ssmInnerSize);  // largest K/V dim

    this.hidden = compute.buffers.createBuffer('hidden', E * 4);
    this.residual = compute.buffers.createBuffer('residual', E * 4);
    this.normed = compute.buffers.createBuffer('normed', E * 4);
    this.qBuf = compute.buffers.createBuffer('q_proj', H * 4);
    this.kBuf = compute.buffers.createBuffer('k_proj', KV * 4);
    this.vBuf = compute.buffers.createBuffer('v_proj', KV * 4);
    this.attnOut = compute.buffers.createBuffer('attn_out', Math.max(E, this.numHeads * this.valueLength) * 4);
    this.gateBuf = compute.buffers.createBuffer('gate', F * 4);
    this.upBuf = compute.buffers.createBuffer('up', F * 4);
    this.ffnOut = compute.buffers.createBuffer('ffn_out', F * 4);
    this.temp = compute.buffers.createBuffer('temp', E * 4);
    this.logits = compute.buffers.createBuffer('logits', this.vocabSize * 4);
    // Standard attention buffers (for gated Q)
    const qAttnDim = this.numHeads * this.keyLength;
    this.qAttnBuf = compute.buffers.createBuffer('q_attn', qAttnDim * 4);
    this.qGateBufAttn = compute.buffers.createBuffer('q_gate_attn', qAttnDim * 4);
    // DeltaNet buffers
    const maxQkvDim = Math.max(...this.qkvDims, this.ssmInnerSize * 3);
    const numVHeads = this.ssmNumVHeads;
    this.qTiledBuf = compute.buffers.createBuffer('q_tiled', numVHeads * this.ssmHeadDim * 4);
    this.kTiledBuf = compute.buffers.createBuffer('k_tiled', numVHeads * this.ssmHeadDim * 4);
    this.qkvBuf = compute.buffers.createBuffer('qkv', maxQkvDim * 4);
    this.ssmGateBuf = compute.buffers.createBuffer('ssm_gate', this.ssmInnerSize * 4);
    this.ssmOutputBuf = compute.buffers.createBuffer('ssm_output', this.ssmInnerSize * 4);
    this.alphaBuf = compute.buffers.createBuffer('alpha', numVHeads * 4);
    this.betaBuf = compute.buffers.createBuffer('beta', numVHeads * 4);
    this.decayBuf = compute.buffers.createBuffer('decay', numVHeads * 4);
    this.betaValBuf = compute.buffers.createBuffer('betaVal', numVHeads * 4);

    // KV caches for standard attention layers
    const maxCtx = Math.min(this.info.contextLength, 4096);
    this.kvCaches = [];
    this.deltaStates = [];
    for (let i = 0; i < this.numLayers; i++) {
      if (this.isStandardAttention(i)) {
        this.kvCaches.push(new KvCache(compute.device, compute.buffers, this.numKvHeads, this.keyLength, maxCtx, i));
        this.deltaStates.push(null);
      } else {
        this.kvCaches.push(null);
        const stateSize = this.ssmNumVHeads * this.ssmHeadDim * this.ssmHeadDim;
        const convBufSize = this.qkvDims[i] * (this.ssmConvKernel - 1);
        this.deltaStates.push({
          state: compute.buffers.createBuffer(`dn_state_${i}`, stateSize * 4),
          convBuf: compute.buffers.createBuffer(`dn_conv_${i}`, convBufSize * 4),
        });
      }
    }
  }

  resetCache(): void {
    this._position = 0;
    for (const kv of this.kvCaches) kv?.reset();
    for (let i = 0; i < this.deltaStates.length; i++) {
      const ds = this.deltaStates[i];
      if (ds) {
        const stateSize = this.ssmNumVHeads * this.ssmHeadDim * this.ssmHeadDim;
        const convBufSize = this.qkvDims[i] * (this.ssmConvKernel - 1);
        this.compute.device.queue.writeBuffer(ds.state, 0, new Float32Array(stateSize));
        if (convBufSize > 0) this.compute.device.queue.writeBuffer(ds.convBuf, 0, new Float32Array(convBufSize));
      }
    }
  }

  /**
   * Forward pass for a single token.
   */
  forward(tokenId: number): GPUBuffer {
    const { compute, weights } = this;
    const E = this.embeddingDim;

    // 1. Token embedding (use Q8_0 shader if weights are quantized)
    if (weights.tokenEmbedding.type === GgmlType.Q8_0) {
      compute.embeddingQ8(weights.tokenEmbedding.buffer, this.hidden, tokenId, E);
    } else {
      compute.embedding(weights.tokenEmbedding.buffer, this.hidden, tokenId, E);
    }

    if (this._position === 0) {

      // Check normed after copyAndRmsNorm will be called for layer 0
      // Compute expected normed on CPU for comparison
    }

    // Transformer layers
    for (let layer = 0; layer < this.numLayers; layer++) {
      const lw = weights.layers[layer];

      // Pre-attention RMSNorm + residual copy
      compute.copyAndRmsNorm(this.hidden, lw.attnNorm, this.normed, this.residual, E, this.rmsNormEps);


      if (lw.kind === 'standard') {
        this.forwardStandardAttention(layer, lw);
      } else {
        this.forwardDeltaNet(layer, lw);
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
    compute.rmsNorm(this.hidden, weights.outputNorm, this.normed, E, this.rmsNormEps);
    compute.matmul(weights.output.buffer, this.normed, this.logits, this.vocabSize, E, weights.output.type);

    this._position++;
    return this.logits;
  }

  private forwardStandardAttention(layer: number, lw: StdAttnWeights): void {
    const { compute } = this;
    const E = this.embeddingDim;
    const kvCache = this.kvCaches[layer]!;
    const keyLen = this.keyLength;
    const valLen = this.valueLength;
    const nH = this.numHeads;
    const nKV = this.numKvHeads;
    const pos = kvCache.seqLen;
    const scale = 1.0 / Math.sqrt(keyLen);

    // 1. Q, K, V projections (GPU matmul)
    const qFullDim = this.hasGatedQ ? nH * keyLen * 2 : nH * keyLen;
    const kDim = nKV * keyLen;
    const vDim = nKV * valLen;
    compute.matmul(lw.q.buffer, this.normed, this.qBuf, qFullDim, E, lw.q.type);
    compute.matmul(lw.k.buffer, this.normed, this.kBuf, kDim, E, lw.k.type);
    compute.matmul(lw.v.buffer, this.normed, this.vBuf, vDim, E, lw.v.type);

    // 2. Deinterleave gated Q (GPU)
    if (this.hasGatedQ) {
      compute.deinterleaveQ(this.qBuf, this.qAttnBuf, this.qGateBufAttn, nH, keyLen);
    } else {
      compute.copyBuffer(this.qBuf, this.qAttnBuf, nH * keyLen * 4);
    }

    // 3. Per-head QK RMSNorm (GPU)
    if (lw.qNorm) {
      compute.perHeadRmsNorm(this.qAttnBuf, lw.qNorm, nH, keyLen, this.rmsNormEps);
      compute.perHeadRmsNorm(this.kBuf, lw.kNorm || lw.qNorm, nKV, keyLen, this.rmsNormEps);
    }

    // 4. Partial RoPE (GPU)
    compute.partialRope(this.qAttnBuf, nH, keyLen, this.ropeDim, pos, this.ropeTheta);
    compute.partialRope(this.kBuf, nKV, keyLen, this.ropeDim, pos, this.ropeTheta);

    // 5. KV cache write
    kvCache.write(this.kBuf, this.vBuf, kDim * 4, vDim * 4);

    // 6. Gated attention (GPU)
    compute.gatedAttention(
      this.qAttnBuf, this.qGateBufAttn, kvCache.kBuffer, kvCache.vBuffer, this.attnOut,
      nH, nKV, keyLen, valLen, kvCache.maxSeqLen, kvCache.seqLen, scale,
    );

    // 7. Output projection + residual (GPU)
    compute.matmul(lw.o.buffer, this.attnOut, this.temp, E, nH * valLen, lw.o.type);
    compute.add(this.temp, this.residual, this.hidden, E);
  }

  private forwardDeltaNet(layer: number, lw: DeltaNetWeights): void {
    const { compute } = this;
    const E = this.embeddingDim;
    const ds = this.deltaStates[layer]!;
    const innerSize = this.ssmInnerSize;
    const numVHeads = this.ssmNumVHeads;
    const numKHeads = this.ssmGroupCount;
    const headDim = this.ssmHeadDim;

    // Derive actual Q/K/V dimensions from weight size (may be unequal)
    const valueDim = numVHeads * headDim; // V = numVHeads × headDim
    const qkvDim = this.qkvDims[layer]; // actual QKV output size from weight
    const keyDim = (qkvDim - valueDim) / 2; // Q and K are equal
    const groupDim = keyDim / numKHeads;

    // 1. QKV projection (GPU matmul)
    compute.matmul(lw.qkv.buffer, this.normed, this.qkvBuf, qkvDim, E, lw.qkv.type);

    // 2+3. Fused causal conv1d + SiLU (GPU)
    compute.conv1dSilu(this.qkvBuf, ds.convBuf, lw.ssmConv1d, qkvDim, this.ssmConvKernel);

    // 4. Split Q/K/V via GPU copy — layout: [Q: keyDim | K: keyDim | V: valueDim]
    compute.copyBufferRegion(this.qkvBuf, 0, this.qBuf, 0, keyDim * 4);
    compute.copyBufferRegion(this.qkvBuf, keyDim * 4, this.kBuf, 0, keyDim * 4);
    compute.copyBufferRegion(this.qkvBuf, keyDim * 2 * 4, this.vBuf, 0, valueDim * 4);

    // 5. L2 normalize Q and K (GPU) — numKHeads groups of groupDim
    compute.l2NormGroups(this.qBuf, numKHeads, groupDim);
    compute.l2NormGroups(this.kBuf, numKHeads, groupDim);

    // 6. Repeat-tile Q and K if numVHeads > numKHeads
    const repeatFactor = numVHeads / numKHeads;
    let qForDelta = this.qBuf;
    let kForDelta = this.kBuf;
    if (repeatFactor > 1) {
      compute.repeatTile(this.qBuf, this.qTiledBuf, numKHeads, groupDim, repeatFactor);
      compute.repeatTile(this.kBuf, this.kTiledBuf, numKHeads, groupDim, repeatFactor);
      qForDelta = this.qTiledBuf;
      kForDelta = this.kTiledBuf;
    }

    // 7. Alpha and Beta projections (GPU matmul — small: numVHeads outputs from E inputs)
    compute.matmul(lw.ssmAlpha.buffer, this.normed, this.alphaBuf, numVHeads, E, lw.ssmAlpha.type);
    compute.matmul(lw.ssmBeta.buffer, this.normed, this.betaBuf, numVHeads, E, lw.ssmBeta.type);

    // 8. Compute decay and beta values (GPU)
    compute.computeDecayBeta(this.alphaBuf, this.betaBuf, lw.ssmA, lw.ssmDtBias,
      this.decayBuf, this.betaValBuf, numVHeads);

    // 9. DeltaNet state update + output + per-head norm (GPU)
    const scale = 1.0 / Math.sqrt(headDim);
    compute.deltanetStep(qForDelta, kForDelta, this.vBuf, ds.state,
      this.decayBuf, this.betaValBuf, lw.ssmNorm, this.ssmOutputBuf,
      numVHeads, headDim, scale, this.rmsNormEps);

    // 10. Gate projection + SiLU gate (GPU)
    compute.matmul(lw.gate.buffer, this.normed, this.ssmGateBuf, innerSize, E, lw.gate.type);
    compute.siluGate(this.ssmOutputBuf, this.ssmGateBuf, innerSize);

    // 11. Output projection (GPU)
    compute.matmul(lw.ssmOut.buffer, this.ssmOutputBuf, this.temp, E, innerSize, lw.ssmOut.type);

    // Residual add
    compute.add(this.temp, this.residual, this.hidden, E);
  }

  /** Apply RoPE to first ropeDim dims of each head, leaving rest unchanged. */
  async readLogits(): Promise<Float32Array> {
    return new Float32Array(await this.compute.readBuffer(this.logits, this.vocabSize * 4));
  }
}
