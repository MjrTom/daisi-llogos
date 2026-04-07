/**
 * Layer-swap model: trades speed for VRAM by keeping only one layer's weights
 * in GPU memory at a time. For each token, iterates through all layers:
 * upload layer weights → forward one layer → overwrite with next layer's weights.
 *
 * Peak VRAM = embedding + 1 layer + all KV caches + working buffers + output head
 * Instead of: embedding + ALL layers + KV caches + working buffers + output head
 *
 * Requires model to be split into per-layer shard files.
 */

import { ComputeEngine } from '../gpu/compute.js';
import type { GgufModelInfo, GgufTensorInfo } from '../gguf/gguf-parser.js';
import { GgmlType } from '../gguf/quantization.js';
import { KvCache } from './kv-cache.js';
import type { LlamaModel, LayerWeights, WeightBuffer } from './llama-model.js';
import { fetchShard, shardUrl } from '../storage/shard-loader.js';
import { parseShardIndex, extractTensorFromShard } from '../gguf/shard-reader.js';

/** GPU types that have native matmul shaders. */
const GPU_MATMUL_TYPES = new Set([GgmlType.F32, GgmlType.Q4_0, GgmlType.Q8_0]);

export class LayerSwapModel {
  private compute: ComputeEngine;
  private info: GgufModelInfo;
  private model: LlamaModel;
  private baseUrl: string;
  private modelFileName: string;
  private tensorInfoMap: Map<string, GgufTensorInfo>;

  /** Pre-allocated GPU buffers for a single layer's weights (reused each layer). */
  private layerBuffers: LayerWeights | null = null;

  /** Cached shard ArrayBuffers from Cache API (layer index → ArrayBuffer). */
  private shardCache: Map<number, ArrayBuffer> = new Map();

  constructor(
    compute: ComputeEngine,
    info: GgufModelInfo,
    model: LlamaModel,
    baseUrl: string,
    modelFileName: string,
    tensorInfoMap: Map<string, GgufTensorInfo>,
  ) {
    this.compute = compute;
    this.info = info;
    this.model = model;
    this.baseUrl = baseUrl;
    this.modelFileName = modelFileName;
    this.tensorInfoMap = tensorInfoMap;
  }

  get numLayers(): number { return this.info.blockCount; }
  get position(): number { return this.model.position; }

  /**
   * Forward pass for a single token using layer swapping.
   * Uploads each layer's weights one at a time, runs forward, then overwrites.
   */
  async forward(tokenId: number): Promise<GPUBuffer> {
    // 1. Embedding
    this.model.forwardEmbedding(tokenId);

    // 2. Transformer layers — one at a time
    for (let i = 0; i < this.numLayers; i++) {
      await this.uploadLayerWeights(i);
      this.model.forwardLayers(i, i + 1);
    }

    // 3. Output head
    return this.model.forwardOutputHead();
  }

  /** Read logits from GPU. */
  async readLogits(): Promise<Float32Array> {
    return this.model.readLogits();
  }

  /** Reset KV caches. */
  resetCache(): void {
    this.model.resetCache();
  }

  /**
   * Upload weights for a single layer to the reusable GPU buffers.
   * Fetches the shard from cache if available, otherwise from network/Cache API.
   */
  private async uploadLayerWeights(layerIndex: number): Promise<void> {
    // Get or fetch the shard buffer
    let shardBuffer = this.shardCache.get(layerIndex);
    if (!shardBuffer) {
      const url = shardUrl(this.baseUrl, `${this.modelFileName}.layer.${layerIndex}`);
      shardBuffer = await fetchShard(url);
      this.shardCache.set(layerIndex, shardBuffer);
    }

    const index = parseShardIndex(shardBuffer);
    const i = layerIndex;

    const uploadWeight = (name: string): WeightBuffer => {
      const info = this.tensorInfoMap.get(name);
      if (!info) throw new Error(`Missing tensor info: ${name}`);
      const data = extractTensorFromShard(shardBuffer!, index, name);

      if (GPU_MATMUL_TYPES.has(info.type)) {
        // Write directly to existing buffer or create new one
        const buffer = this.compute.buffers.createBufferWithData(`swap_${name}`, data);
        return { buffer, type: info.type };
      }
      // Would need dequantization — for now assume native types in swap mode
      const buffer = this.compute.buffers.createBufferWithData(`swap_${name}`, data);
      return { buffer, type: info.type };
    };

    const uploadAsF32 = (name: string): GPUBuffer => {
      const info = this.tensorInfoMap.get(name);
      if (!info) throw new Error(`Missing tensor info: ${name}`);
      const data = extractTensorFromShard(shardBuffer!, index, name);
      return this.compute.buffers.createBufferWithData(`swap_${name}`, data);
    };

    const tryUploadAsF32 = (name: string): GPUBuffer | undefined => {
      if (!this.tensorInfoMap.has(name) || !index.tensors.has(name)) return undefined;
      return uploadAsF32(name);
    };

    // Destroy previous layer buffers if they exist
    if (this.layerBuffers) {
      this.destroyLayerBuffers(this.layerBuffers);
    }

    this.layerBuffers = {
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
    };

    // Patch the model's weights for this layer index
    (this.model as any).weights.layers[layerIndex] = this.layerBuffers;
  }

  private destroyLayerBuffers(lw: LayerWeights): void {
    lw.attnNorm.destroy();
    lw.q.buffer.destroy();
    lw.k.buffer.destroy();
    lw.v.buffer.destroy();
    lw.o.buffer.destroy();
    lw.qBias?.destroy();
    lw.kBias?.destroy();
    lw.vBias?.destroy();
    lw.postAttnNorm.destroy();
    lw.gateProj.buffer.destroy();
    lw.upProj.buffer.destroy();
    lw.downProj.buffer.destroy();
  }

  /** Pre-download all layer shards into memory/cache for faster inference. */
  async preloadShards(
    onProgress?: (loaded: number, total: number) => void,
  ): Promise<void> {
    for (let i = 0; i < this.numLayers; i++) {
      if (!this.shardCache.has(i)) {
        const url = shardUrl(this.baseUrl, `${this.modelFileName}.layer.${i}`);
        this.shardCache.set(i, await fetchShard(url));
      }
      onProgress?.(i + 1, this.numLayers);
    }
  }
}
