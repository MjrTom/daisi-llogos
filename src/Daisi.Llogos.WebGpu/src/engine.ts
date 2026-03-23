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
import { downloadFile, extractTensorData, DownloadProgress } from './storage/download-manager.js';
import { GgmlType } from './gguf/quantization.js';
import { BpeTokenizer, tokenizerFromGguf } from './tokenizer/bpe-tokenizer.js';
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
      // Phase 1: Parse header
      options?.onProgress?.({ phase: 'Parsing header', bytesDownloaded: 0, totalBytes: 0 });
      let info = await fetchGgufHeader(url);
      this.modelInfo = info;

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
      this.model.storeCpuWeights(tensorMap);

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

      // Encode prompt — only add BOS for first message (no KV cache yet)
      const inputTokens: number[] = [];
      if (this.model.position === 0 && this.tokenizer.bosTokenId >= 0) {
        inputTokens.push(this.tokenizer.bosTokenId);
      }
      // Add a newline separator between turns if continuing conversation
      if (this.model.position > 0) {
        inputTokens.push(...this.tokenizer.encode('\n'));
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

  /** Reset the KV cache for a new conversation. */
  resetSession(): void {
    this.model?.resetCache();
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

  /**
   * Apply chat template based on model architecture.
   * Wraps user message in the appropriate format.
   */
  private applyChatTemplate(userMessage: string): string {
    const arch = this.modelInfo?.architecture ?? '';
    const chatTemplate = this.modelInfo?.metadata.get('tokenizer.chat_template') as string | undefined;

    // ChatML format — only if the model actually has the special tokens
    if (this.tokenizer && this.tokenizer.getTokenId('<|im_start|>') >= 0) {
      return `<|im_start|>user\n${userMessage}<|im_end|>\n<|im_start|>assistant\n`;
    }

    // Llama 2 format
    if (chatTemplate?.includes('[INST]')) {
      return `[INST] ${userMessage} [/INST]`;
    }

    // Fallback: just use the message directly
    return userMessage;
  }
}
