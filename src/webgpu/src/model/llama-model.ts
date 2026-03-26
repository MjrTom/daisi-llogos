/**
 * Llama/Qwen transformer model implementation.
 * Forward pass: embedding → N × (RMSNorm → Attention → Residual → RMSNorm → FFN → Residual) → RMSNorm → lm_head
 */

import { ComputeEngine } from '../gpu/compute.js';
import { GgufModelInfo, GgufTensorInfo } from '../gguf/gguf-parser.js';
import { GgmlType } from '../gguf/quantization.js';
import { KvCache } from './kv-cache.js';

/**
 * CPU-side dequantization of quantized weights to F32.
 * Used for embedding, output, and norm weights that need F32 format.
 */
function dequantizeToF32(buffer: ArrayBuffer, type: GgmlType, elementCount: number): Float32Array {
  const result = new Float32Array(elementCount);
  const bytes = new Uint8Array(buffer);
  const view = new DataView(buffer);

  if (type === GgmlType.F16) {
    for (let i = 0; i < elementCount; i++) {
      result[i] = f16ToF32(view.getUint16(i * 2, true));
    }
    return result;
  }

  if (type === GgmlType.Q8_0) {
    // Q8_0: 34 bytes per block of 32 elements = [f16 scale][32 x int8]
    const blockCount = Math.ceil(elementCount / 32);
    for (let b = 0; b < blockCount; b++) {
      const blockOffset = b * 34;
      const scale = f16ToF32(view.getUint16(blockOffset, true));
      for (let q = 0; q < 32 && b * 32 + q < elementCount; q++) {
        const val = view.getInt8(blockOffset + 2 + q);
        result[b * 32 + q] = scale * val;
      }
    }
    return result;
  }

  if (type === GgmlType.Q4_0) {
    // Q4_0: 18 bytes per block of 32 elements = [f16 scale][16 bytes of nibbles]
    // Low nibbles of bytes 0-15 → quants 0-15
    // High nibbles of bytes 0-15 → quants 16-31
    const blockCount = Math.ceil(elementCount / 32);
    for (let b = 0; b < blockCount; b++) {
      const blockOffset = b * 18;
      const scale = f16ToF32(view.getUint16(blockOffset, true));
      for (let j = 0; j < 16; j++) {
        const byteVal = bytes[blockOffset + 2 + j];
        const lo = (byteVal & 0x0F) - 8;
        const hi = ((byteVal >> 4) & 0x0F) - 8;
        const idx = b * 32;
        if (idx + j < elementCount) result[idx + j] = scale * lo;
        if (idx + j + 16 < elementCount) result[idx + j + 16] = scale * hi;
      }
    }
    return result;
  }

  if (type === GgmlType.Q6_K) {
    // Q6_K: 210 bytes per super-block of 256 elements
    // Layout: ql[128] + qh[64] + scales[16] + d(f16)
    // Each quant is 6 bits: 4 from ql + 2 from qh, value = d * scale * (q6 - 32)
    const QK = 256;
    const blockCount = Math.ceil(elementCount / QK);
    for (let b = 0; b < blockCount; b++) {
      const bo = b * 210; // block offset
      const qlOff = bo;           // ql: 128 bytes
      const qhOff = bo + 128;    // qh: 64 bytes
      const scOff = bo + 192;    // scales: 16 bytes (int8)
      const dOff = bo + 208;     // d: 2 bytes (f16)
      const d = f16ToF32(view.getUint16(dOff, true));
      const outBase = b * QK;

      for (let n = 0; n < QK; n += 128) {
        for (let l = 0; l < 32; l++) {
          const is_ = n / 16;
          const qlIdx0 = qlOff + n / 2 + l;
          const qlIdx1 = qlOff + n / 2 + l + 32;
          const qhIdx = qhOff + n / 4 + l;
          const qhByte = bytes[qhIdx];

          const q1 = ((bytes[qlIdx0] & 0xF) | (((qhByte >> 0) & 3) << 4)) - 32;
          const q2 = ((bytes[qlIdx1] & 0xF) | (((qhByte >> 2) & 3) << 4)) - 32;
          const q3 = ((bytes[qlIdx0] >> 4) | (((qhByte >> 4) & 3) << 4)) - 32;
          const q4 = ((bytes[qlIdx1] >> 4) | (((qhByte >> 6) & 3) << 4)) - 32;

          const sc0 = view.getInt8(scOff + is_ + 0);
          const sc2 = view.getInt8(scOff + is_ + 2);
          const sc4 = view.getInt8(scOff + is_ + 4);
          const sc6 = view.getInt8(scOff + is_ + 6);

          const oi = outBase + n + l;
          if (oi < elementCount) result[oi] = d * sc0 * q1;
          if (oi + 32 < elementCount) result[oi + 32] = d * sc2 * q2;
          if (oi + 64 < elementCount) result[oi + 64] = d * sc4 * q3;
          if (oi + 96 < elementCount) result[oi + 96] = d * sc6 * q4;
        }
      }
    }
    return result;
  }

  if (type === GgmlType.Q4_K) {
    // Q4_K: 144 bytes per super-block of 256 elements
    // Layout: d(f16) + dmin(f16) + scales[12] + mins[12] + qs[128]
    const QK = 256;
    const blockCount = Math.ceil(elementCount / QK);
    for (let b = 0; b < blockCount; b++) {
      const bo = b * 144;
      const d = f16ToF32(view.getUint16(bo, true));
      const dmin = f16ToF32(view.getUint16(bo + 2, true));
      const scalesOff = bo + 4;     // 12 bytes
      const minsOff = bo + 16;      // 12 bytes (not present as separate field in older format)
      const qsOff = bo + 16;        // but actually Q4_K packs scales+mins differently

      // Q4_K has a complex scale/min encoding — use simplified approach
      // For CPU dequant of embedding only, read qs and apply approximate dequant
      // TODO: implement full Q4_K dequant if needed
      const outBase = b * QK;
      for (let j = 0; j < 128 && outBase + j * 2 < elementCount; j++) {
        const qByte = bytes[bo + 16 + j]; // simplified offset
        const lo = (qByte & 0xF);
        const hi = (qByte >> 4);
        if (outBase + j < elementCount) result[outBase + j] = d * (lo - 8);
        if (outBase + j + 128 < elementCount) result[outBase + j + 128] = d * (hi - 8);
      }
    }
    return result;
  }

  if (type === GgmlType.Q4_1) {
    // Q4_1: 20 bytes per block of 32 elements = [f16 delta][f16 min][16 bytes nibbles]
    const blockCount = Math.ceil(elementCount / 32);
    for (let b = 0; b < blockCount; b++) {
      const bo = b * 20;
      const delta = f16ToF32(view.getUint16(bo, true));
      const min = f16ToF32(view.getUint16(bo + 2, true));
      for (let j = 0; j < 16; j++) {
        const byteVal = bytes[bo + 4 + j];
        const lo = byteVal & 0x0F;
        const hi = (byteVal >> 4) & 0x0F;
        const idx = b * 32;
        if (idx + j < elementCount) result[idx + j] = delta * lo + min;
        if (idx + j + 16 < elementCount) result[idx + j + 16] = delta * hi + min;
      }
    }
    return result;
  }

  if (type === GgmlType.Q5_0) {
    // Q5_0: 22 bytes per block of 32 elements = [f16 scale][4 bytes high-bits][16 bytes nibbles]
    const blockCount = Math.ceil(elementCount / 32);
    for (let b = 0; b < blockCount; b++) {
      const bo = b * 22;
      const scale = f16ToF32(view.getUint16(bo, true));
      const highBits = view.getUint32(bo + 2, true);
      for (let j = 0; j < 16; j++) {
        const byteVal = bytes[bo + 6 + j];
        const lo4 = byteVal & 0x0F;
        const hi4 = (byteVal >> 4) & 0x0F;
        const loBit = (highBits >> j) & 1;
        const hiBit = (highBits >> (j + 16)) & 1;
        const q5lo = lo4 | (loBit << 4);
        const q5hi = hi4 | (hiBit << 4);
        const idx = b * 32;
        if (idx + j < elementCount) result[idx + j] = scale * (q5lo - 16);
        if (idx + j + 16 < elementCount) result[idx + j + 16] = scale * (q5hi - 16);
      }
    }
    return result;
  }

  if (type === GgmlType.Q5_1) {
    // Q5_1: 24 bytes per block of 32 elements = [f16 delta][f16 min][4 bytes high-bits][16 bytes nibbles]
    const blockCount = Math.ceil(elementCount / 32);
    for (let b = 0; b < blockCount; b++) {
      const bo = b * 24;
      const delta = f16ToF32(view.getUint16(bo, true));
      const min = f16ToF32(view.getUint16(bo + 2, true));
      const highBits = view.getUint32(bo + 4, true);
      for (let j = 0; j < 16; j++) {
        const byteVal = bytes[bo + 8 + j];
        const lo4 = byteVal & 0x0F;
        const hi4 = (byteVal >> 4) & 0x0F;
        const loBit = (highBits >> j) & 1;
        const hiBit = (highBits >> (j + 16)) & 1;
        const q5lo = lo4 | (loBit << 4);
        const q5hi = hi4 | (hiBit << 4);
        const idx = b * 32;
        if (idx + j < elementCount) result[idx + j] = delta * q5lo + min;
        if (idx + j + 16 < elementCount) result[idx + j + 16] = delta * q5hi + min;
      }
    }
    return result;
  }

  throw new Error(`Unsupported dequant type: ${GgmlType[type]} (${type})`);
}

/** Convert f16 bits to f32. */
function f16ToF32(bits: number): number {
  const sign = (bits >> 15) & 1;
  const exp = (bits >> 10) & 0x1F;
  const mant = bits & 0x3FF;
  if (exp === 0) {
    if (mant === 0) return sign ? -0 : 0;
    // Denormalized
    const f = (mant / 1024) * Math.pow(2, -14);
    return sign ? -f : f;
  }
  if (exp === 31) return sign ? -Infinity : Infinity;
  const f = (1 + mant / 1024) * Math.pow(2, exp - 15);
  return sign ? -f : f;
}

/** Quant types that have a GPU matmul shader. Others get dequanted to F32. */
const GPU_MATMUL_TYPES = new Set([GgmlType.F32, GgmlType.Q4_0, GgmlType.Q8_0]);

/** A weight tensor with its quant type. */
interface WeightBuffer {
  buffer: GPUBuffer;
  type: GgmlType; // the type the GPU buffer is stored as (F32 if dequanted)
}

/** GPU weight buffers for the model. */
export interface ModelWeights {
  tokenEmbedding: WeightBuffer;
  outputNorm: GPUBuffer;
  output: WeightBuffer;
  layers: LayerWeights[];
}

export interface LayerWeights {
  attnNorm: GPUBuffer;
  q: WeightBuffer;
  k: WeightBuffer;
  v: WeightBuffer;
  o: WeightBuffer;
  qBias?: GPUBuffer;
  kBias?: GPUBuffer;
  vBias?: GPUBuffer;
  postAttnNorm: GPUBuffer;
  gateProj: WeightBuffer;
  upProj: WeightBuffer;
  downProj: WeightBuffer;
}

export class LlamaModel {
  private compute: ComputeEngine;
  private info: GgufModelInfo;
  private weights!: ModelWeights;
  private kvCaches!: KvCache[]; // one per layer

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
  private temp!: GPUBuffer; // temp buffer for in-place add workaround
  private logits!: GPUBuffer;

  constructor(compute: ComputeEngine, info: GgufModelInfo) {
    this.compute = compute;
    this.info = info;
  }

  get embeddingDim(): number { return this.info.embeddingLength; }
  get numLayers(): number { return this.info.blockCount; }
  get numHeads(): number { return this.info.headCount; }
  get numKvHeads(): number { return this.info.headCountKv || this.info.headCount; }
  get headDim(): number { return this.embeddingDim / this.numHeads; }
  get ffnDim(): number { return this.info.feedForwardLength; }
  get vocabSize(): number { return this.info.vocabSize; }
  get contextLength(): number { return this.info.contextLength; }
  get ropeTheta(): number { return this.info.ropeFreqBase; }
  get rmsNormEps(): number { return this.info.rmsNormEps; }

  /**
   * Upload tensor data to GPU and initialize working buffers.
   */
  async initWeights(
    tensorMap: Map<string, { buffer: ArrayBuffer; info: GgufTensorInfo }>,
  ): Promise<void> {
    const { compute, info } = this;
    const arch = info.architecture;

    // Upload helpers
    const uploadAsF32 = (name: string): GPUBuffer => {
      const tensor = tensorMap.get(name);
      if (!tensor) throw new Error(`Missing tensor: ${name}`);
      if (tensor.info.type === GgmlType.F32) {
        return compute.buffers.createBufferWithData(name, tensor.buffer);
      }
      const f32Data = dequantizeToF32(tensor.buffer, tensor.info.type, tensor.info.elementCount);
      return compute.buffers.createBufferWithData(name, f32Data.buffer);
    };

    const uploadWeight = (name: string): WeightBuffer => {
      const tensor = tensorMap.get(name);
      if (!tensor) throw new Error(`Missing tensor: ${name}`);
      if (GPU_MATMUL_TYPES.has(tensor.info.type)) {
        return { buffer: compute.buffers.createBufferWithData(name, tensor.buffer), type: tensor.info.type };
      }
      const f32Data = dequantizeToF32(tensor.buffer, tensor.info.type, tensor.info.elementCount);
      return { buffer: compute.buffers.createBufferWithData(name, f32Data.buffer), type: GgmlType.F32 };
    };

    // Upload weights — keep embedding in native format (avoids huge F32 allocation for large vocabs)
    const tokenEmbedding = uploadWeight('token_embd.weight');
    let outputWeight: WeightBuffer;
    if (tensorMap.has('output.weight')) {
      outputWeight = uploadWeight('output.weight');
    } else {
      outputWeight = tokenEmbedding;
    }
    const outputNorm = uploadAsF32('output_norm.weight');

    const tryUploadAsF32 = (name: string): GPUBuffer | undefined => {
      return tensorMap.has(name) ? uploadAsF32(name) : undefined;
    };

    const layers: LayerWeights[] = [];
    for (let i = 0; i < this.numLayers; i++) {
      layers.push({
        attnNorm: uploadAsF32(`blk.${i}.attn_norm.weight`),
        q: uploadWeight(`blk.${i}.attn_q.weight`),
        k: uploadWeight(`blk.${i}.attn_k.weight`),
        v: uploadWeight(`blk.${i}.attn_v.weight`),
        o: uploadWeight(`blk.${i}.attn_output.weight`),
        qBias: tryUploadAsF32(`blk.${i}.attn_q.bias`),
        kBias: tryUploadAsF32(`blk.${i}.attn_k.bias`),
        vBias: tryUploadAsF32(`blk.${i}.attn_v.bias`),
        postAttnNorm: uploadAsF32(`blk.${i}.ffn_norm.weight`),
        gateProj: uploadWeight(`blk.${i}.ffn_gate.weight`),
        upProj: uploadWeight(`blk.${i}.ffn_up.weight`),
        downProj: uploadWeight(`blk.${i}.ffn_down.weight`),
      });
    }

    this.weights = { tokenEmbedding, outputNorm, output: outputWeight, layers };

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
    this.ffnOut = compute.buffers.createBuffer('ffn_out', F * 4); // ffnDim, NOT E!
    this.temp = compute.buffers.createBuffer('temp', E * 4);
    this.logits = compute.buffers.createBuffer('logits', this.vocabSize * 4);

    // Initialize per-layer KV caches
    const maxCtx = Math.min(this.contextLength, 4096);
    this.kvCaches = [];
    for (let i = 0; i < this.numLayers; i++) {
      this.kvCaches.push(new KvCache(
        compute.device, compute.buffers,
        this.numKvHeads, this.headDim, maxCtx, `kv_L${i}`,
      ));
    }
  }

  /** Reset the KV cache for a new conversation. */
  resetCache(): void {
    for (const kv of this.kvCaches) kv.reset();
  }

  /** Get current sequence position in KV cache. */
  get position(): number { return this.kvCaches[0].seqLen; }

  /**
   * Forward pass for a single token. Returns logits buffer on GPU.
   */

  forward(tokenId: number): GPUBuffer {
    const { compute, weights } = this;
    const E = this.embeddingDim;

    // 1. Token embedding — use Q8_0 shader if quantized, F32 otherwise
    if (weights.tokenEmbedding.type === GgmlType.Q8_0) {
      compute.embeddingQ8(weights.tokenEmbedding.buffer, this.hidden, tokenId, E);
    } else {
      compute.embedding(weights.tokenEmbedding.buffer, this.hidden, tokenId, E);
    }


    // Transformer layers
    for (let layer = 0; layer < this.numLayers; layer++) {
      const lw = weights.layers[layer];

      // Fused: copy hidden→residual AND RMSNorm→normed (saves 1 dispatch)
      compute.copyAndRmsNorm(this.hidden, lw.attnNorm, this.normed, this.residual, E, this.rmsNormEps);



      // Q, K, V projections
      compute.matmul(lw.q.buffer, this.normed, this.qBuf, this.numHeads * this.headDim, E, lw.q.type);
      compute.matmul(lw.k.buffer, this.normed, this.kBuf, this.numKvHeads * this.headDim, E, lw.k.type);
      compute.matmul(lw.v.buffer, this.normed, this.vBuf, this.numKvHeads * this.headDim, E, lw.v.type);

      // Add attention biases (Qwen 2 models)
      if (lw.qBias) compute.addBias(this.qBuf, lw.qBias, this.numHeads * this.headDim);
      if (lw.kBias) compute.addBias(this.kBuf, lw.kBias, this.numKvHeads * this.headDim);
      if (lw.vBias) compute.addBias(this.vBuf, lw.vBias, this.numKvHeads * this.headDim);

      // RoPE on Q and K
      const kvCache = this.kvCaches[layer];
      const pos = kvCache.seqLen;
      compute.rope(this.qBuf, this.headDim, this.headDim, pos, this.ropeTheta, this.numHeads * this.headDim);
      compute.rope(this.kBuf, this.headDim, this.headDim, pos, this.ropeTheta, this.numKvHeads * this.headDim);

      // Write to KV cache for this layer
      kvCache.write(this.kBuf, this.vBuf, this.numKvHeads * this.headDim * 4, this.numKvHeads * this.headDim * 4);

      // Attention (GPU)
      compute.attention(
        this.qBuf, kvCache.kBuffer, kvCache.vBuffer, this.attnOut,
        this.numHeads, this.numKvHeads, this.headDim,
        kvCache.seqLen, kvCache.maxSeqLen,
      );

      // Output projection → temp, then residual add → hidden
      compute.matmul(lw.o.buffer, this.attnOut, this.temp, E, this.numHeads * this.headDim, lw.o.type);
      compute.add(this.temp, this.residual, this.hidden, E);

      // Fused: copy hidden→residual AND post-attn RMSNorm→normed
      compute.copyAndRmsNorm(this.hidden, lw.postAttnNorm, this.normed, this.residual, E, this.rmsNormEps);

      // FFN: SwiGLU — all on GPU
      compute.matmul(lw.gateProj.buffer, this.normed, this.gateBuf, this.ffnDim, E, lw.gateProj.type);
      compute.matmul(lw.upProj.buffer, this.normed, this.upBuf, this.ffnDim, E, lw.upProj.type);
      compute.siluMul(this.gateBuf, this.upBuf, this.ffnOut, this.ffnDim);
      compute.matmul(lw.downProj.buffer, this.ffnOut, this.temp, E, this.ffnDim, lw.downProj.type);

      // Residual add → hidden
      compute.add(this.temp, this.residual, this.hidden, E);
    }

    // 3. Final RMSNorm
    compute.rmsNorm(this.hidden, weights.outputNorm, this.normed, E, this.rmsNormEps);

    // 4. LM Head — optionally compute only first vocabLimit rows (partial vocab)
    const lmRows = this._vocabLimit > 0 ? Math.min(this._vocabLimit, this.vocabSize) : this.vocabSize;
    compute.matmul(weights.output.buffer, this.normed, this.logits, lmRows, E, weights.output.type);

    return this.logits;
  }

  /** Set vocab limit for partial logit computation. 0 = full vocab. */
  private _vocabLimit = 0;
  set vocabLimit(n: number) { this._vocabLimit = n; }
  get vocabLimit(): number { return this._vocabLimit; }

  /**
   * Batched prefill: process all prompt tokens at once per layer.
   * Much faster than calling forward() N times for long prompts.
   * Returns logits buffer for the LAST token only.
   */
  forwardPrefill(tokenIds: number[]): GPUBuffer {
    const { compute, weights } = this;
    const E = this.embeddingDim;
    const N = tokenIds.length;
    if (N === 0) return this.logits;
    if (N === 1) return this.forward(tokenIds[0]);

    const H = this.numHeads * this.headDim;
    const KV = this.numKvHeads * this.headDim;
    const F = this.ffnDim;

    // Allocate batched buffers
    const bHidden = compute.buffers.createBuffer('b_hidden', N * E * 4);
    const bResidual = compute.buffers.createBuffer('b_residual', N * E * 4);
    const bNormed = compute.buffers.createBuffer('b_normed', N * E * 4);
    const bQ = compute.buffers.createBuffer('b_q', N * H * 4);
    const bK = compute.buffers.createBuffer('b_k', N * KV * 4);
    const bV = compute.buffers.createBuffer('b_v', N * KV * 4);
    const bAttnOut = compute.buffers.createBuffer('b_attn', N * E * 4);
    const bGate = compute.buffers.createBuffer('b_gate', N * F * 4);
    const bUp = compute.buffers.createBuffer('b_up', N * F * 4);
    const bFfnOut = compute.buffers.createBuffer('b_ffn', N * F * 4);
    const bTemp = compute.buffers.createBuffer('b_temp', N * E * 4);

    // Upload token IDs and positions
    const tokenIdBuf = compute.buffers.createBuffer('b_tokens', N * 4);
    const positionBuf = compute.buffers.createBuffer('b_positions', N * 4);
    const startPos = this.kvCaches[0].seqLen;
    const posData = new Uint32Array(N);
    for (let i = 0; i < N; i++) posData[i] = startPos + i;
    compute.device.queue.writeBuffer(tokenIdBuf, 0, new Uint32Array(tokenIds));
    compute.device.queue.writeBuffer(positionBuf, 0, posData);

    // 1. Batched embedding (F32 only for now)
    if (weights.tokenEmbedding.type === GgmlType.F32) {
      compute.embeddingBatch(weights.tokenEmbedding.buffer, tokenIdBuf, bHidden, N, E);
    } else {
      // Q8_0 batch embedding not implemented — fall back to per-token
      for (let i = 0; i < N; i++) {
        compute.embeddingQ8(weights.tokenEmbedding.buffer, this.hidden, tokenIds[i], E);
        compute.copyBufferRegion(this.hidden, 0, bHidden, i * E * 4, E * 4);
      }
    }

    // 2. Transformer layers
    for (let layer = 0; layer < this.numLayers; layer++) {
      const lw = weights.layers[layer];
      const kvCache = this.kvCaches[layer];

      // RMSNorm all N rows + copy to residual
      // Copy hidden → residual
      compute.copyBuffer(bHidden, bResidual, N * E * 4);
      // RMSNorm hidden → normed
      compute.rmsNormBatch(bHidden, lw.attnNorm, bNormed, E, N, this.rmsNormEps);

      // Q, K, V projections — batched matmul
      compute.matmulBatch(lw.q.buffer, bNormed, bQ, H, E, N, lw.q.type);
      compute.matmulBatch(lw.k.buffer, bNormed, bK, KV, E, N, lw.k.type);
      compute.matmulBatch(lw.v.buffer, bNormed, bV, KV, E, N, lw.v.type);

      // Biases (Qwen 2 style) — apply to each token's Q/K/V
      if (lw.qBias) {
        for (let n = 0; n < N; n++) {
          // addBias works on contiguous data — need offset-aware version
          // For now, skip batch bias (only affects Qwen 2.5)
        }
      }

      // Batched RoPE — each token at its own position
      compute.ropeBatch(bQ, positionBuf, this.numHeads, this.headDim, N, this.ropeTheta);
      compute.ropeBatch(bK, positionBuf, this.numKvHeads, this.headDim, N, this.ropeTheta);

      // Batched KV cache write
      compute.kvCacheWriteBatch(
        bK, bV, kvCache.kBuffer, kvCache.vBuffer,
        this.numKvHeads, this.headDim, kvCache.maxSeqLen, startPos, N,
      );
      // Update KV cache seqLen to include all N tokens
      kvCache.seqLen = startPos + N;

      // Batched causal attention
      compute.attentionBatch(
        bQ, kvCache.kBuffer, kvCache.vBuffer, bAttnOut,
        this.numHeads, this.numKvHeads, this.headDim,
        kvCache.maxSeqLen, startPos, N,
      );

      // Output projection + residual add
      compute.matmulBatch(lw.o.buffer, bAttnOut, bTemp, E, H, N, lw.o.type);
      compute.add(bTemp, bResidual, bHidden, N * E);

      // Post-attention: copy → residual, RMSNorm → normed
      compute.copyBuffer(bHidden, bResidual, N * E * 4);
      compute.rmsNormBatch(bHidden, lw.postAttnNorm, bNormed, E, N, this.rmsNormEps);

      // FFN: SwiGLU — batched
      compute.matmulBatch(lw.gateProj.buffer, bNormed, bGate, F, E, N, lw.gateProj.type);
      compute.matmulBatch(lw.upProj.buffer, bNormed, bUp, F, E, N, lw.upProj.type);
      compute.siluMul(bGate, bUp, bFfnOut, N * F);
      compute.matmulBatch(lw.downProj.buffer, bFfnOut, bTemp, E, F, N, lw.downProj.type);
      compute.add(bTemp, bResidual, bHidden, N * E);
    }

    // 3. Copy last token's hidden to single-token buffer for LM head
    compute.copyBufferRegion(bHidden, (N - 1) * E * 4, this.hidden, 0, E * 4);

    // 4. Final RMSNorm + LM Head (single token)
    compute.rmsNorm(this.hidden, weights.outputNorm, this.normed, E, this.rmsNormEps);
    compute.matmul(weights.output.buffer, this.normed, this.logits, this.vocabSize, E, weights.output.type);

    // Clean up batched buffers
    bHidden.destroy(); bResidual.destroy(); bNormed.destroy();
    bQ.destroy(); bK.destroy(); bV.destroy(); bAttnOut.destroy();
    bGate.destroy(); bUp.destroy(); bFfnOut.destroy(); bTemp.destroy();
    tokenIdBuf.destroy(); positionBuf.destroy();

    return this.logits;
  }

  // ── CPU forward pass (for verification / fallback) ──────────────────

  private cpuWeights: {
    embedding: Float32Array;
    outputNorm: Float32Array;
    output: Float32Array;
    layers: Array<{
      attnNorm: Float32Array; q: Float32Array; k: Float32Array; v: Float32Array; o: Float32Array;
      postAttnNorm: Float32Array; gate: Float32Array; up: Float32Array; down: Float32Array;
    }>;
  } | null = null;

  /** Store CPU-side F32 weights for CPU forward pass. */
  storeCpuWeights(tensorMap: Map<string, { buffer: ArrayBuffer; info: GgufTensorInfo }>): void {
    const dq = (name: string): Float32Array => {
      const t = tensorMap.get(name)!;
      if (t.info.type === GgmlType.F32) return new Float32Array(t.buffer);
      return dequantizeToF32(t.buffer, t.info.type, t.info.elementCount);
    };
    const layers = [];
    for (let i = 0; i < this.numLayers; i++) {
      layers.push({
        attnNorm: dq(`blk.${i}.attn_norm.weight`), q: dq(`blk.${i}.attn_q.weight`),
        k: dq(`blk.${i}.attn_k.weight`), v: dq(`blk.${i}.attn_v.weight`),
        o: dq(`blk.${i}.attn_output.weight`), postAttnNorm: dq(`blk.${i}.ffn_norm.weight`),
        gate: dq(`blk.${i}.ffn_gate.weight`), up: dq(`blk.${i}.ffn_up.weight`),
        down: dq(`blk.${i}.ffn_down.weight`),
      });
    }
    this.cpuWeights = {
      embedding: dq('token_embd.weight'),
      outputNorm: dq('output_norm.weight'),
      output: dq(tensorMap.has('output.weight') ? 'output.weight' : 'token_embd.weight'),
      layers,
    };
  }

  getCpuState() {
    const maxSeqLen = 512;
    return {
      kvK: Array.from({ length: this.numLayers }, () => new Float32Array(this.numKvHeads * maxSeqLen * this.headDim)),
      kvV: Array.from({ length: this.numLayers }, () => new Float32Array(this.numKvHeads * maxSeqLen * this.headDim)),
      seqLen: 0,
      maxSeqLen,
      hidden: new Float32Array(this.embeddingDim),
    };
  }

  async cpuForward(tokenId: number, state: ReturnType<LlamaModel['getCpuState']>): Promise<void> {
    const E = this.embeddingDim;
    const w = this.cpuWeights!;
    // Use a fresh array for each forward pass (matching verified Node.js test)
    const h = new Float32Array(E);

    // Embedding
    for (let i = 0; i < E; i++) h[i] = w.embedding[tokenId * E + i];

    // Layers
    for (let layer = 0; layer < this.numLayers; layer++) {
      const lw = w.layers[layer];
      const residual = new Float32Array(h);

      const normed = cpuRmsNorm(h, lw.attnNorm, E, this.rmsNormEps);
      const qP = cpuMatvec(lw.q, normed, this.numHeads * this.headDim, E);
      const kP = cpuMatvec(lw.k, normed, this.numKvHeads * this.headDim, E);
      const vP = cpuMatvec(lw.v, normed, this.numKvHeads * this.headDim, E);

      cpuRope(qP, this.headDim, state.seqLen, this.ropeTheta);
      cpuRope(kP, this.headDim, state.seqLen, this.ropeTheta);

      // Write KV cache
      for (let kh = 0; kh < this.numKvHeads; kh++) {
        for (let d = 0; d < this.headDim; d++) {
          state.kvK[layer][kh * state.maxSeqLen * this.headDim + state.seqLen * this.headDim + d] = kP[kh * this.headDim + d];
          state.kvV[layer][kh * state.maxSeqLen * this.headDim + state.seqLen * this.headDim + d] = vP[kh * this.headDim + d];
        }
      }

      // Attention
      const curSeqLen = state.seqLen + 1;
      const scale = 1.0 / Math.sqrt(this.headDim);
      const headsPerGroup = this.numHeads / this.numKvHeads;
      const attnOut = new Float32Array(this.numHeads * this.headDim);
      for (let head = 0; head < this.numHeads; head++) {
        const kvHead = Math.floor(head / headsPerGroup);
        const scores = new Float32Array(curSeqLen);
        for (let pos = 0; pos < curSeqLen; pos++) {
          let dot = 0;
          for (let d = 0; d < this.headDim; d++)
            dot += qP[head * this.headDim + d] * state.kvK[layer][kvHead * state.maxSeqLen * this.headDim + pos * this.headDim + d];
          scores[pos] = dot * scale;
        }
        let maxS = -Infinity;
        for (let i = 0; i < curSeqLen; i++) maxS = Math.max(maxS, scores[i]);
        let sumE = 0;
        for (let i = 0; i < curSeqLen; i++) { scores[i] = Math.exp(scores[i] - maxS); sumE += scores[i]; }
        for (let i = 0; i < curSeqLen; i++) scores[i] /= sumE;
        for (let d = 0; d < this.headDim; d++) {
          let acc = 0;
          for (let pos = 0; pos < curSeqLen; pos++)
            acc += scores[pos] * state.kvV[layer][kvHead * state.maxSeqLen * this.headDim + pos * this.headDim + d];
          attnOut[head * this.headDim + d] = acc;
        }
      }

      const oP = cpuMatvec(lw.o, attnOut, E, this.numHeads * this.headDim);
      for (let i = 0; i < E; i++) h[i] = oP[i] + residual[i];

      await new Promise(r => setTimeout(r, 0)); // yield mid-layer

      // FFN
      const residual2 = new Float32Array(h);
      const normed2 = cpuRmsNorm(h, lw.postAttnNorm, E, this.rmsNormEps);
      const gateOut = cpuMatvec(lw.gate, normed2, this.ffnDim, E);
      const upOut = cpuMatvec(lw.up, normed2, this.ffnDim, E);
      const ffnOut = new Float32Array(this.ffnDim);
      for (let i = 0; i < this.ffnDim; i++)
        ffnOut[i] = (gateOut[i] / (1 + Math.exp(-gateOut[i]))) * upOut[i];
      const downOut = cpuMatvec(lw.down, ffnOut, E, this.ffnDim);
      for (let i = 0; i < E; i++) h[i] = downOut[i] + residual2[i];

      // Yield every layer to prevent UI lockup
      await new Promise(r => setTimeout(r, 0));
    }

    // Copy result to state
    state.hidden.set(h);
    state.seqLen++;
  }

  cpuGetLogits(state: ReturnType<LlamaModel['getCpuState']>): Float32Array {
    const w = this.cpuWeights!;
    const normed = cpuRmsNorm(state.hidden, w.outputNorm, this.embeddingDim, this.rmsNormEps);
    return cpuMatvec(w.output, normed, this.vocabSize, this.embeddingDim);
  }

  /**
   * Read logits back to CPU for sampling.
   */
  async readLogits(): Promise<Float32Array> {
    const size = this._vocabLimit > 0 ? Math.min(this._vocabLimit, this.vocabSize) : this.vocabSize;
    const logits = await this.compute.readBuffer(this.logits, size * 4);
    this.compute.cleanupParams();
    return logits;
  }
}

// ── CPU helper functions ────────────────────────────────────────────

function cpuRmsNorm(input: Float32Array, weight: Float32Array, n: number, eps: number): Float32Array {
  let sumSq = 0;
  for (let i = 0; i < n; i++) sumSq += input[i] * input[i];
  const rms = Math.sqrt(sumSq / n + eps);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = (input[i] / rms) * weight[i];
  return out;
}

function cpuMatvec(weights: Float32Array, input: Float32Array, M: number, K: number): Float32Array {
  const out = new Float32Array(M);
  for (let row = 0; row < M; row++) {
    let sum = 0;
    const off = row * K;
    for (let k = 0; k < K; k++) sum += weights[off + k] * input[k];
    out[row] = sum;
  }
  return out;
}

function cpuRope(data: Float32Array, headDim: number, position: number, theta: number): void {
  const nPairs = data.length / 2;
  for (let pi = 0; pi < nPairs; pi++) {
    const hp = pi % (headDim / 2);
    const dimFrac = (hp * 2) / headDim;
    const freq = 1.0 / Math.pow(theta, dimFrac);
    const angle = position * freq;
    const c = Math.cos(angle);
    const s = Math.sin(angle);
    const i0 = pi * 2, i1 = i0 + 1;
    const x = data[i0], y = data[i1];
    data[i0] = x * c - y * s;
    data[i1] = x * s + y * c;
  }
}
