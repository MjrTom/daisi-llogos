/**
 * GPU inference tests for Qwen 3.5 (DeltaNet hybrid architecture).
 * TDD: write failing tests first, then implement until green.
 */
import './setup-webgpu.js';
import { describe, it, expect, beforeAll } from 'vitest';
import { readFileSync } from 'fs';
import { initGpu } from '../src/gpu/device.js';
import { ComputeEngine } from '../src/gpu/compute.js';
import { parseGguf } from '../src/gguf/gguf-parser.js';
import { extractTensorData } from '../src/storage/download-manager.js';
import { tokenizerFromGguf } from '../src/tokenizer/bpe-tokenizer.js';
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

describe('Qwen 3.5 GGUF Parsing', () => {
  const { info } = loadModel('C:/GGUFS/Qwen3.5-0.8B-Q8_0.gguf');

  it('parses qwen35 architecture', () => {
    expect(info.architecture).toBe('qwen35');
  });

  it('has correct dimensions', () => {
    expect(info.blockCount).toBe(24);
    expect(info.embeddingLength).toBe(1024);
    expect(info.headCount).toBe(8);
    expect(info.headCountKv).toBe(2);
    expect(info.feedForwardLength).toBe(3584);
  });

  it('has SSM metadata', () => {
    expect(info.metadata.get('qwen35.ssm.inner_size')).toBe(2048);
    expect(info.metadata.get('qwen35.ssm.state_size')).toBe(128);
    expect(info.metadata.get('qwen35.ssm.group_count')).toBe(16);
    expect(info.metadata.get('qwen35.ssm.conv_kernel')).toBe(4);
    expect(info.metadata.get('qwen35.full_attention_interval')).toBe(4);
  });

  it('has hybrid layer structure', () => {
    // Layers 3,7,11,15,19,23 should have separate Q/K/V (standard attention)
    const layer3 = info.tensors.filter(t => t.name.startsWith('blk.3.'));
    expect(layer3.some(t => t.name === 'blk.3.attn_q.weight')).toBe(true);
    expect(layer3.some(t => t.name === 'blk.3.attn_k.weight')).toBe(true);
    expect(layer3.some(t => t.name === 'blk.3.attn_v.weight')).toBe(true);
    expect(layer3.some(t => t.name === 'blk.3.attn_output.weight')).toBe(true);

    // Layer 0 should have fused QKV + SSM
    const layer0 = info.tensors.filter(t => t.name.startsWith('blk.0.'));
    expect(layer0.some(t => t.name === 'blk.0.attn_qkv.weight')).toBe(true);
    expect(layer0.some(t => t.name === 'blk.0.ssm_a')).toBe(true);
    expect(layer0.some(t => t.name === 'blk.0.attn_gate.weight')).toBe(true);
  });
});

describe('Qwen 3.5 GPU Inference', () => {
  let compute: ComputeEngine;
  let model: any; // Will be Qwen35Model
  let tokenizer: ReturnType<typeof tokenizerFromGguf>;

  beforeAll(async () => {
    const gpu = await initGpu();
    compute = new ComputeEngine(gpu.device);

    const { info, tokenizer: tok, tensorMap } = loadModel('C:/GGUFS/Qwen3.5-0.8B-Q8_0.gguf');
    tokenizer = tok;

    // Import the model class dynamically to avoid breaking if it doesn't exist yet
    const { Qwen35Model } = await import('../src/model/qwen35-model.js');
    model = new Qwen35Model(compute, info);
    await model.initWeights(tensorMap);
  }, 120000);

  it('forward pass produces valid logits', async () => {
    const tokens = tokenizer.encode('Hello');
    await model.forward(tokens[0]);
    const logits = await model.readLogits();
    expect(logits.length).toBe(248320);

    let max = -Infinity, nonZero = 0;
    for (const v of logits) { if (v > max) max = v; if (v !== 0) nonZero++; }
    expect(nonZero).toBeGreaterThan(0);
    expect(isFinite(max)).toBe(true);
  });

  it('generates coherent text', async () => {
    model.resetCache();
    const prompt = '<|im_start|>user\nWhat color is the sky?<|im_end|>\n<|im_start|>assistant\n';
    const inputTokens = tokenizer.encode(prompt);
    const sampler = new Sampler({ temperature: 0, topK: 1 });

    for (const t of inputTokens) {
      await model.forward(t);
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
    console.log(`  Qwen 3.5 generated: "${text}"`);
    expect(generated.length).toBeGreaterThan(0);
    expect(/[a-zA-Z]/.test(text)).toBe(true);
    // Should not be stuck in a repetitive loop
    const hasRepeat = /(.{4,})\1{3,}/.test(text);
    if (hasRepeat) console.log('  WARNING: repetitive output detected');
    expect(hasRepeat).toBe(false);
  });
});
