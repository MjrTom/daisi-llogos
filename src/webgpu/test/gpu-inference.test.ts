/**
 * GPU inference integration tests — runs the full engine in Node.js via Dawn.
 * Requires GGUF model files in test/ directory.
 */
import './setup-webgpu.js';
import { describe, it, expect, beforeAll } from 'vitest';
import { readFileSync } from 'fs';
import { initGpu } from '../src/gpu/device.js';
import { ComputeEngine } from '../src/gpu/compute.js';
import { parseGguf } from '../src/gguf/gguf-parser.js';
import { extractTensorData } from '../src/storage/download-manager.js';
import { tokenizerFromGguf } from '../src/tokenizer/bpe-tokenizer.js';
import { LlamaModel } from '../src/model/llama-model.js';
import { Sampler } from '../src/model/sampler.js';

function loadModel(path: string) {
  const buf = readFileSync(path);
  const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
  const info = parseGguf(ab);
  const tokenizer = tokenizerFromGguf(info.metadata);

  const tensorMap = new Map<string, { buffer: ArrayBuffer; info: any }>();
  for (const tensor of info.tensors) {
    tensorMap.set(tensor.name, {
      buffer: extractTensorData(ab, info.tensorDataOffset + tensor.offset, tensor.byteSize),
      info: tensor,
    });
  }

  return { info, tokenizer, tensorMap };
}

describe('GPU Inference — TinyLlama 1.1B Q8_0', () => {
  let compute: ComputeEngine;
  let model: LlamaModel;
  let tokenizer: ReturnType<typeof tokenizerFromGguf>;

  beforeAll(async () => {
    const gpu = await initGpu();
    compute = new ComputeEngine(gpu.device);

    const { info, tokenizer: tok, tensorMap } = loadModel('C:/GGUFS/tinyllama-q8.gguf');
    tokenizer = tok;
    model = new LlamaModel(compute, info);
    await model.initWeights(tensorMap);
  }, 60000);

  it('initializes model with correct dimensions', () => {
    expect(model.embeddingDim).toBe(2048);
    expect(model.numLayers).toBe(22);
    expect(model.vocabSize).toBe(32000);
  });

  it('forward pass produces logits', async () => {
    const tokenId = tokenizer.encode('Hello')[0];
    model.forward(tokenId);
    const logits = await model.readLogits();
    expect(logits.length).toBe(32000);

    // Logits should not be all zeros
    const nonZero = logits.filter(v => v !== 0).length;
    expect(nonZero).toBeGreaterThan(0);

    // Logits should have a reasonable range (not NaN or Inf)
    let max = -Infinity, min = Infinity;
    for (const v of logits) { if (v > max) max = v; if (v < min) min = v; }
    expect(isFinite(max)).toBe(true);
    expect(isFinite(min)).toBe(true);
    expect(max).toBeGreaterThan(min);
  });

  it('generates coherent tokens', async () => {
    model.resetCache();
    const prompt = 'The capital of France is';
    const inputTokens = tokenizer.encode(prompt);
    const sampler = new Sampler({ temperature: 0, topK: 1 }); // greedy

    // Prefill
    for (const t of inputTokens) {
      model.forward(t);
    }
    let logits = await model.readLogits();

    // Generate 10 tokens
    const allTokens = [...inputTokens];
    const generated: string[] = [];
    for (let i = 0; i < 10; i++) {
      const next = sampler.sample(logits, allTokens);
      if (tokenizer.isEos(next)) break;
      allTokens.push(next);
      generated.push(tokenizer.decode([next]));
      model.forward(next);
      logits = await model.readLogits();
    }

    const text = generated.join('');
    console.log(`  Generated: "${prompt}${text}"`);
    expect(generated.length).toBeGreaterThan(0);
    // With greedy sampling, TinyLlama should produce something relevant
    const lower = text.toLowerCase();
    expect(lower.includes('paris') || lower.includes('france') || lower.includes('capital')).toBe(true);
  });
});

describe('GPU Inference — Llama 3.2 1B Q8_0 (GQA)', () => {
  let compute: ComputeEngine;
  let model: LlamaModel;
  let tokenizer: ReturnType<typeof tokenizerFromGguf>;

  beforeAll(async () => {
    const gpu = await initGpu();
    compute = new ComputeEngine(gpu.device);

    const { info, tokenizer: tok, tensorMap } = loadModel('C:/GGUFS/llama32-1b-q8.gguf');
    tokenizer = tok;
    model = new LlamaModel(compute, info);
    await model.initWeights(tensorMap);
  }, 120000);

  it('has GQA dimensions', () => {
    expect(model.numHeads).toBe(32);
    expect(model.numKvHeads).toBe(8);
  });

  it('forward pass produces logits for 128K vocab', async () => {
    // Encode with Llama 3 special tokens
    const tokens = tokenizer.encode('<|begin_of_text|>Hello');
    expect(tokens[0]).toBe(128000);

    model.forward(tokens[0]);
    model.forward(tokens[1]);
    const logits = await model.readLogits();
    expect(logits.length).toBe(128256);

    const nonZero = logits.filter(v => v !== 0).length;
    expect(nonZero).toBeGreaterThan(0);
    let max = -Infinity; for (const v of logits) if (v > max) max = v;
    expect(isFinite(max)).toBe(true);
  });

  it('generates tokens with GQA', async () => {
    model.resetCache();
    const prompt = '<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nWhat is 2+2?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n';
    const inputTokens = tokenizer.encode(prompt);
    const sampler = new Sampler({ temperature: 0, topK: 1 });

    for (const t of inputTokens) {
      model.forward(t);
    }
    let logits = await model.readLogits();

    const allTokens = [...inputTokens];
    const generated: string[] = [];
    for (let i = 0; i < 20; i++) {
      const next = sampler.sample(logits, allTokens);
      if (tokenizer.isEos(next)) break;
      allTokens.push(next);
      generated.push(tokenizer.decode([next]));
      model.forward(next);
      logits = await model.readLogits();
    }

    const text = generated.join('');
    console.log(`  Generated: "${text}"`);
    expect(generated.length).toBeGreaterThan(0);
    // Should mention 4 somewhere
    expect(text).toContain('4');
  });
});

describe('GPU Inference — Qwen 2.5 0.5B Q8_0 (attention biases)', () => {
  let compute: ComputeEngine;
  let model: LlamaModel;
  let tokenizer: ReturnType<typeof tokenizerFromGguf>;
  let info: ReturnType<typeof parseGguf>;

  beforeAll(async () => {
    const gpu = await initGpu();
    compute = new ComputeEngine(gpu.device);

    const loaded = loadModel('C:/GGUFS/qwen25-0.5b-q8.gguf');
    info = loaded.info;
    tokenizer = loaded.tokenizer;
    model = new LlamaModel(compute, info);
    await model.initWeights(loaded.tensorMap);
  }, 120000);

  it('has qwen2 architecture with biases', () => {
    expect(info.architecture).toBe('qwen2');
    expect(info.tensors.some(t => t.name === 'blk.0.attn_q.bias')).toBe(true);
  });

  it('has correct dimensions', () => {
    expect(model.embeddingDim).toBe(896);
    expect(model.numHeads).toBe(14);
    expect(model.numKvHeads).toBe(2);
    expect(model.numLayers).toBe(24);
    expect(model.vocabSize).toBe(151936);
  });

  it('forward pass produces valid logits', async () => {
    const tokens = tokenizer.encode('Hello');
    model.forward(tokens[0]);
    const logits = await model.readLogits();
    expect(logits.length).toBe(151936);

    let max = -Infinity, nonZero = 0;
    for (const v of logits) { if (v > max) max = v; if (v !== 0) nonZero++; }
    expect(nonZero).toBeGreaterThan(0);
    expect(isFinite(max)).toBe(true);
  });

  it('generates coherent text with ChatML format', async () => {
    model.resetCache();
    // ChatML prompt for Qwen
    const prompt = '<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n';
    const inputTokens = tokenizer.encode(prompt);
    const sampler = new Sampler({ temperature: 0, topK: 1 });

    // Verify special tokens encoded correctly
    const imStartId = tokenizer.getTokenId('<|im_start|>');
    expect(inputTokens[0]).toBe(imStartId);

    for (const t of inputTokens) {
      model.forward(t);
    }
    let logits = await model.readLogits();

    const allTokens = [...inputTokens];
    const generated: string[] = [];
    for (let i = 0; i < 30; i++) {
      const next = sampler.sample(logits, allTokens);
      if (tokenizer.isEos(next)) break;
      allTokens.push(next);
      generated.push(tokenizer.decode([next]));
      model.forward(next);
      logits = await model.readLogits();
    }

    const text = generated.join('');
    console.log(`  Qwen 2.5 generated: "${text}"`);
    expect(generated.length).toBeGreaterThan(0);
    // Should generate coherent English (not gibberish)
    expect(/[a-zA-Z]/.test(text)).toBe(true);
  });
});
