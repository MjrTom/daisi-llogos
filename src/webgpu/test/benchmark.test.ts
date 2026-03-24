/**
 * Performance benchmarks for WebGPU inference.
 * Run with: npx vitest run test/benchmark.test.ts
 *
 * Measures:
 * - Prefill: tokens/sec for processing the prompt
 * - Decode: tokens/sec for autoregressive generation
 * - Total: end-to-end including prefill + decode
 */
import './setup-webgpu.js';
import { describe, it, beforeAll } from 'vitest';
import { readFileSync, existsSync } from 'fs';
import { parseGguf } from '../src/gguf/gguf-parser.js';
import { extractTensorData } from '../src/storage/download-manager.js';
import { tokenizerFromGguf } from '../src/tokenizer/bpe-tokenizer.js';
import { LlamaModel } from '../src/model/llama-model.js';
import { Qwen35Model } from '../src/model/qwen35-model.js';
import { Sampler } from '../src/model/sampler.js';
import { initGpu } from '../src/gpu/device.js';
import { ComputeEngine } from '../src/gpu/compute.js';
import type { GgufModelInfo, GgufTensorInfo } from '../src/gguf/gguf-parser.js';
import type { BpeTokenizer } from '../src/tokenizer/bpe-tokenizer.js';

interface BenchmarkResult {
  model: string;
  promptTokens: number;
  decodeTokens: number;
  prefillMs: number;
  decodeMs: number;
  prefillTokSec: number;
  decodeTokSec: number;
  totalTokSec: number;
  vramMb: number;
}

const DECODE_TOKENS = 32;
const PROMPT = 'The quick brown fox jumps over the lazy dog. What is the meaning of life?';

function loadModel(path: string) {
  const buf = readFileSync(path);
  const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
  const info = parseGguf(ab);
  const tokenizer = tokenizerFromGguf(info.metadata);
  const tensorMap = new Map<string, { buffer: ArrayBuffer; info: GgufTensorInfo }>();
  for (const tensor of info.tensors) {
    tensorMap.set(tensor.name, {
      buffer: extractTensorData(ab, info.tensorDataOffset + tensor.offset, tensor.byteSize),
      info: tensor,
    });
  }
  return { info, tokenizer, tensorMap };
}

async function runBenchmark(
  label: string,
  modelPath: string,
  compute: ComputeEngine,
  isQwen35: boolean,
): Promise<BenchmarkResult> {
  const { info, tokenizer, tensorMap } = loadModel(modelPath);

  // Create model
  const model = isQwen35
    ? new Qwen35Model(compute, info)
    : new LlamaModel(compute, info);
  await model.initWeights(tensorMap);

  // Encode prompt
  const promptTokens = tokenizer.encode(PROMPT);
  const sampler = new Sampler({ temperature: 0, topK: 1 });

  // Warmup: 1 token
  if (isQwen35) {
    await (model as Qwen35Model).forward(promptTokens[0]);
  } else {
    (model as LlamaModel).forward(promptTokens[0]);
  }
  await model.readLogits();
  model.resetCache();

  // Prefill benchmark
  const prefillStart = performance.now();
  for (const t of promptTokens) {
    if (isQwen35) {
      await (model as Qwen35Model).forward(t);
    } else {
      (model as LlamaModel).forward(t);
    }
  }
  let logits = await model.readLogits();
  const prefillMs = performance.now() - prefillStart;

  // Decode benchmark
  const allTokens = [...promptTokens];
  let decodeCount = 0;
  const decodeStart = performance.now();
  for (let i = 0; i < DECODE_TOKENS; i++) {
    const next = sampler.sample(logits, allTokens);
    if (tokenizer.isEos(next)) break;
    allTokens.push(next);
    decodeCount++;
    if (isQwen35) {
      await (model as Qwen35Model).forward(next);
    } else {
      (model as LlamaModel).forward(next);
    }
    logits = await model.readLogits();
  }
  const decodeMs = performance.now() - decodeStart;

  const vramMb = compute.buffers.vramUsage / 1024 / 1024;
  const totalMs = prefillMs + decodeMs;
  const totalTokens = promptTokens.length + decodeCount;

  return {
    model: label,
    promptTokens: promptTokens.length,
    decodeTokens: decodeCount,
    prefillMs,
    decodeMs,
    prefillTokSec: promptTokens.length / (prefillMs / 1000),
    decodeTokSec: decodeCount / (decodeMs / 1000),
    totalTokSec: totalTokens / (totalMs / 1000),
    vramMb: Math.round(vramMb),
  };
}

function printResult(r: BenchmarkResult) {
  console.log(`\n  ╔══════════════════════════════════════════════╗`);
  console.log(`  ║  ${r.model.padEnd(44)}║`);
  console.log(`  ╠══════════════════════════════════════════════╣`);
  console.log(`  ║  Prefill:  ${r.promptTokens} tokens in ${r.prefillMs.toFixed(0)}ms = ${r.prefillTokSec.toFixed(1)} tok/s`.padEnd(49) + '║');
  console.log(`  ║  Decode:   ${r.decodeTokens} tokens in ${r.decodeMs.toFixed(0)}ms = ${r.decodeTokSec.toFixed(1)} tok/s`.padEnd(49) + '║');
  console.log(`  ║  Total:    ${r.promptTokens + r.decodeTokens} tokens = ${r.totalTokSec.toFixed(1)} tok/s`.padEnd(49) + '║');
  console.log(`  ║  VRAM:     ${r.vramMb} MB`.padEnd(49) + '║');
  console.log(`  ╚══════════════════════════════════════════════╝`);
}

// ── Benchmarks ──────────────────────────────────────────────────────

const models: Array<{ label: string; path: string; isQwen35: boolean }> = [
  { label: 'TinyLlama 1.1B Q8_0', path: 'C:/GGUFS/tinyllama-q8.gguf', isQwen35: false },
  { label: 'Llama 3.2 1B Q8_0', path: 'C:/GGUFS/llama32-1b-q8.gguf', isQwen35: false },
  { label: 'Qwen 2.5 0.5B Q8_0', path: 'C:/GGUFS/qwen25-0.5b-q8.gguf', isQwen35: false },
  { label: 'Qwen 3.5 0.8B Q8_0', path: 'C:/GGUFS/Qwen3.5-0.8B-Q8_0.gguf', isQwen35: true },
];

describe('Performance Benchmarks', () => {
  let compute: ComputeEngine;

  beforeAll(async () => {
    const gpu = await initGpu();
    console.log(`\n  GPU: ${gpu.capabilities.adapterInfo.vendor} ${gpu.capabilities.adapterInfo.architecture}`);
    console.log(`  Max buffer: ${Math.round(gpu.device.limits.maxBufferSize / 1024 / 1024)} MB`);
    compute = new ComputeEngine(gpu.device);
  });

  for (const m of models) {
    it(m.label, async () => {
      if (!existsSync(m.path)) {
        console.log(`  SKIP: ${m.path} not found`);
        return;
      }
      const result = await runBenchmark(m.label, m.path, compute, m.isQwen35);
      printResult(result);
    }, 300000);
  }
});
