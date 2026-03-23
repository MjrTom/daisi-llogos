/**
 * LlogosEngine — main entry point for browser-based GGUF inference.
 *
 * Usage:
 *   const engine = new LlogosEngine();
 *   await engine.initGpu();
 *   const info = await engine.inspectModel(url);
 *   await engine.loadModel(url, { onProgress });
 *   for await (const token of engine.generate("Hello", opts)) { ... }
 */

import { initGpu, isWebGpuAvailable, GpuCapabilities, GpuContext } from './gpu/device.js';
import { ComputeEngine } from './gpu/compute.js';
import { fetchGgufHeader, parseGguf, GgufModelInfo, GgufTensorInfo } from './gguf/gguf-parser.js';
import { GgmlType } from './gguf/quantization.js';
import { downloadFile, extractTensorData, DownloadProgress } from './storage/download-manager.js';
import { GgmlType } from './gguf/quantization.js';
import { BpeTokenizer, tokenizerFromGguf } from './tokenizer/bpe-tokenizer.js';
import { applyTemplate, ChatMessage } from './tokenizer/chat-template.js';
import { LlamaModel } from './model/llama-model.js';
import { Sampler, SamplerOptions } from './model/sampler.js';

export type EngineStatus = 'uninitialized' | 'ready' | 'loading' | 'loaded' | 'generating' | 'error';

export interface GenerateOptions extends SamplerOptions {
  maxTokens?: number;
  onToken?: (token: string, tokenId: number) => void;
  signal?: AbortSignal;
  raw?: boolean; // if true, skip chat template wrapping
}

export class LlogosEngine {
  private gpu: GpuContext | null = null;
  private compute: ComputeEngine | null = null;
  private model: LlamaModel | null = null;
  private tokenizer: BpeTokenizer | null = null;
  private modelInfo: GgufModelInfo | null = null;
  private _status: EngineStatus = 'uninitialized';

  get status(): EngineStatus { return this._status; }
  get capabilities(): GpuCapabilities | null { return this.gpu?.capabilities ?? null; }
  get info(): GgufModelInfo | null { return this.modelInfo; }

  /** Check if WebGPU is available. */
  static isSupported(): boolean { return isWebGpuAvailable(); }

  /** Initialize the WebGPU device. Must be called first. */
  async initGpu(): Promise<GpuCapabilities> {
    this.gpu = await initGpu();
    this.compute = new ComputeEngine(this.gpu.device);
    this._status = 'ready';
    return this.gpu.capabilities;
  }

  /** Fetch and parse GGUF header to inspect model metadata without downloading weights. */
  async inspectModel(url: string): Promise<GgufModelInfo> {
    return fetchGgufHeader(url);
  }

  /**
   * Download and load a GGUF model into GPU memory.
   */
  async loadModel(
    url: string,
    options?: {
      onProgress?: (progress: DownloadProgress & { phase: string }) => void;
    },
  ): Promise<GgufModelInfo> {
    if (!this.compute) throw new Error('GPU not initialized. Call initGpu() first.');
    this._status = 'loading';

    try {
      // Phase 1: Parse header and estimate VRAM
      options?.onProgress?.({ phase: 'Parsing header', bytesDownloaded: 0, totalBytes: 0 });
      let info = await fetchGgufHeader(url);
      this.modelInfo = info;

      const estimate = this.estimateVram(info);
      const maxBuffer = this.gpu?.capabilities.maxBufferSize ?? 0;
      if (maxBuffer > 0 && estimate.totalBytes > maxBuffer * 4) {
        // Rough check — maxBuffer is per-buffer limit, total can be higher,
        // but if estimate is 4x the per-buffer limit, it probably won't fit
        console.warn(`[llogos] Model may not fit: estimated ${Math.round(estimate.totalBytes / 1024 / 1024)} MB, max buffer ${Math.round(maxBuffer / 1024 / 1024)} MB`);
      }

      // Download entire GGUF file (single fetch, cached)
      let isCached = false;
      options?.onProgress?.({ phase: 'Checking cache...', bytesDownloaded: 0, totalBytes: 0 });
      const fileBuffer = await downloadFile(url, (p) => {
        if (!isCached && p.bytesDownloaded === 0 && p.totalBytes > 0) {
          isCached = true; // cache hit detected (totalBytes known before download starts)
        }
        const phase = isCached ? 'Loading from cache' : 'Downloading';
        options?.onProgress?.({ phase, bytesDownloaded: p.bytesDownloaded, totalBytes: p.totalBytes });
      });

      // Re-parse from the complete file for accurate offsets
      info = parseGguf(fileBuffer);
      this.modelInfo = info;
      this.tokenizer = tokenizerFromGguf(info.metadata);

      // Extract tensors from file buffer
      const tensorMap = new Map<string, { buffer: ArrayBuffer; info: GgufTensorInfo }>();
      for (const tensor of info.tensors) {
        tensorMap.set(tensor.name, {
          buffer: extractTensorData(fileBuffer, info.tensorDataOffset + tensor.offset, tensor.byteSize),
          info: tensor,
        });
      }

      options?.onProgress?.({ phase: 'Uploading to GPU', bytesDownloaded: fileBuffer.byteLength, totalBytes: fileBuffer.byteLength });
      this.model = new LlamaModel(this.compute, info);
      await this.model.initWeights(tensorMap);

      this._status = 'loaded';
      return info;
    } catch (e) {
      this._status = 'error';
      throw e;
    }
  }

  /**
   * Generate tokens from a prompt. Returns an async iterator of token strings.
   */
  async *generate(prompt: string, options?: GenerateOptions): AsyncGenerator<string> {
    if (!this.model || !this.tokenizer || !this.compute) {
      throw new Error('Model not loaded. Call loadModel() first.');
    }

    this._status = 'generating';
    const maxTokens = options?.maxTokens ?? 512;
    const sampler = new Sampler(options);

    try {
      // Apply chat template unless raw mode
      let finalPrompt = prompt;
      if (!options?.raw) {
        finalPrompt = this.applyChatTemplate(prompt);
      }

      // Encode prompt
      const inputTokens: number[] = [];
      if (this.model.position === 0 && this.tokenizer.bosTokenId >= 0 && options?.raw) {
        inputTokens.push(this.tokenizer.bosTokenId);
      }
      inputTokens.push(...this.tokenizer.encode(finalPrompt));
      const allTokens = [...inputTokens];

      // GPU forward pass — prefill
      for (let i = 0; i < inputTokens.length; i++) {
        this.model.forward(inputTokens[i]);
      }
      let logits = await this.model.readLogits();

      // Generate tokens autoregressively
      for (let step = 0; step < maxTokens; step++) {
        if (options?.signal?.aborted) break;

        const nextToken = sampler.sample(logits, allTokens);

        // Check EOS
        if (this.tokenizer.isEos(nextToken)) break;

        allTokens.push(nextToken);
        const text = this.tokenizer.decode([nextToken]);
        options?.onToken?.(text, nextToken);
        yield text;

        this.model.forward(nextToken);
        logits = await this.model.readLogits();
      }
    } finally {
      this._status = 'loaded';
    }
  }

  /** Reset the KV cache and conversation history for a new conversation. */
  resetSession(): void {
    this.model?.resetCache();
    this.conversationHistory = [];
  }

  /** Unload model and free GPU memory. */
  unloadModel(): void {
    this.compute?.buffers.destroyAll();
    this.model = null;
    this.tokenizer = null;
    this.modelInfo = null;
    this._status = this.gpu ? 'ready' : 'uninitialized';
  }

  /** Get current VRAM usage in bytes. */
  get vramUsage(): number {
    return this.compute?.buffers.vramUsage ?? 0;
  }

  // GPU types that stay quantized (have native GPU shaders)
  private static GPU_NATIVE_TYPES = new Set([GgmlType.F32, GgmlType.Q8_0, GgmlType.Q4_0]);

  /**
   * Estimate VRAM needed for a model before downloading.
   * Returns breakdown in bytes: weights, kvCache, working, total.
   */
  estimateVram(info: GgufModelInfo): { weightsBytes: number; kvCacheBytes: number; workingBytes: number; totalBytes: number } {
    // Weights: native types keep their size, others dequant to F32
    let weightsBytes = 0;
    for (const tensor of info.tensors) {
      if (LlogosEngine.GPU_NATIVE_TYPES.has(tensor.type)) {
        weightsBytes += tensor.byteSize;
      } else {
        weightsBytes += tensor.elementCount * 4; // dequant to F32
      }
    }

    // KV cache: 2 (K+V) * layers * kvHeads * maxSeq * headDim * 4 bytes
    const maxSeq = Math.min(info.contextLength, 4096);
    const headDim = info.embeddingLength / info.headCount;
    const kvHeads = info.headCountKv || info.headCount;
    const kvCacheBytes = 2 * info.blockCount * kvHeads * maxSeq * headDim * 4;

    // Working buffers: hidden, residual, normed, q, k, v, attn_out, gate, up, ffn_out, temp, logits
    const E = info.embeddingLength;
    const F = info.feedForwardLength;
    const V = info.vocabSize;
    const workingBytes = (E * 11 + F * 3 + V) * 4;

    return {
      weightsBytes,
      kvCacheBytes,
      workingBytes,
      totalBytes: weightsBytes + kvCacheBytes + workingBytes,
    };
  }

  /**
   * Apply chat template to format the prompt.
   * Uses the Jinja2 template from GGUF metadata if available,
   * falls back to ChatML/Llama2 heuristics.
   */
  private applyChatTemplate(userMessage: string): string {
    const chatTemplate = this.modelInfo?.metadata.get('tokenizer.chat_template') as string | undefined;

    // Build messages array for the template
    const messages: ChatMessage[] = [
      ...this.conversationHistory,
      { role: 'user', content: userMessage },
    ];

    // Llama 3 heuristic — uses <|start_header_id|> tokens (check before Jinja2,
    // because the Llama 3 template uses complex Jinja2 features we can't parse)
    if (this.tokenizer && this.tokenizer.getTokenId('<|start_header_id|>') >= 0) {
      let prompt = '<|begin_of_text|>';
      // System message
      prompt += '<|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|>';
      for (const msg of messages) {
        prompt += `<|start_header_id|>${msg.role}<|end_header_id|>\n\n${msg.content.trim()}<|eot_id|>`;
      }
      prompt += '<|start_header_id|>assistant<|end_header_id|>\n\n';
      return prompt;
    }

    // ChatML heuristic — if model has the special tokens
    if (this.tokenizer && this.tokenizer.getTokenId('<|im_start|>') >= 0) {
      let prompt = '';
      for (const msg of messages) {
        prompt += `<|im_start|>${msg.role}\n${msg.content}<|im_end|>\n`;
      }
      prompt += '<|im_start|>assistant\n';
      return prompt;
    }

    // Llama 2 heuristic
    if (chatTemplate?.includes('[INST]')) {
      return `[INST] ${userMessage} [/INST]`;
    }

    // Try Jinja2 template from GGUF metadata (for models with simple templates)
    if (chatTemplate) {
      try {
        const bosToken = this.tokenizer && this.tokenizer.bosTokenId >= 0
          ? this.tokenizer.decode([this.tokenizer.bosTokenId]) : '';
        const eosToken = this.tokenizer && this.tokenizer.eosTokenId >= 0
          ? this.tokenizer.decode([this.tokenizer.eosTokenId]) : '';
        const result = applyTemplate(chatTemplate, messages, {
          bos_token: bosToken,
          eos_token: eosToken,
          add_generation_prompt: true,
        });
        // Sanity check: if the result is mostly whitespace, the template probably failed
        if (result.trim().length > 0) return result;
      } catch {
        // Template parse error — fall through
      }
    }

    // Fallback: raw message
    return userMessage;
  }

  /** Conversation history for multi-turn chat template support. */
  private conversationHistory: ChatMessage[] = [];
}
