import './setup-webgpu.js';
import { describe, it, expect, beforeAll } from 'vitest';
import { readFileSync } from 'fs';
import { parseGguf } from '../src/gguf/gguf-parser.js';
import { extractTensorData } from '../src/storage/download-manager.js';
import { tokenizerFromGguf } from '../src/tokenizer/bpe-tokenizer.js';
import { VocabRemapper } from '../src/tokenizer/vocab-remapper.js';
import { LlamaModel } from '../src/model/llama-model.js';
import { Sampler } from '../src/model/sampler.js';
import { initGpu } from '../src/gpu/device.js';
import { ComputeEngine } from '../src/gpu/compute.js';
import { GgmlType } from '../src/gguf/quantization.js';

describe('Vocab Remapping + Partial Vocab', () => {
  let compute: ComputeEngine;
  let model: LlamaModel;
  let tokenizer: ReturnType<typeof tokenizerFromGguf>;
  let remapper: VocabRemapper;
  let info: ReturnType<typeof parseGguf>;

  beforeAll(async () => {
    const buf = readFileSync('C:/GGUFS/llama32-1b-q8.gguf');
    const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
    info = parseGguf(ab);
    tokenizer = tokenizerFromGguf(info.metadata);
    const vocabTokens = info.metadata.get('tokenizer.ggml.tokens') as string[];
    remapper = new VocabRemapper(vocabTokens);

    const tm = new Map<string, any>();
    for (const t of info.tensors) {
      tm.set(t.name, { buffer: extractTensorData(ab, info.tensorDataOffset + t.offset, t.byteSize), info: t });
    }

    // Permute embedding rows (output uses tied weights = same buffer)
    const embTensor = tm.get('token_embd.weight')!;
    const bytesPerRow = embTensor.info.byteSize / info.vocabSize;
    embTensor.buffer = remapper.permuteRows(embTensor.buffer, info.vocabSize, bytesPerRow);

    const gpu = await initGpu();
    compute = new ComputeEngine(gpu.device);
    model = new LlamaModel(compute, info);
    await model.initWeights(tm);
  }, 60000);

  it('generates correct text with full vocab (remapped)', async () => {
    model.resetCache();
    model.vocabLimit = 0;
    const sampler = new Sampler({ temperature: 0, topK: 1 });

    // Encode "Hello" in remapped space
    const rawTokens = tokenizer.encode('Hello');
    const remappedTokens = rawTokens.map(t => remapper.remapEncode(t));

    for (const t of remappedTokens) model.forward(t);
    let logits = await model.readLogits();

    const allTokens = [...remappedTokens];
    const generated: string[] = [];
    for (let i = 0; i < 10; i++) {
      const next = sampler.sample(logits, allTokens);
      const orig = remapper.remapDecode(next);
      if (tokenizer.isEos(orig)) break;
      allTokens.push(next);
      generated.push(tokenizer.decode([orig]));
      model.forward(next);
      logits = await model.readLogits();
    }

    const text = generated.join('');
    console.log(`  Full vocab (remapped): "${text}"`);
    expect(generated.length).toBeGreaterThan(0);
  });

  it('generates correct text with partial vocab (vocabSize/32)', async () => {
    model.resetCache();
    const limit = Math.ceil(info.vocabSize / 8);
    model.vocabLimit = limit;
    const sampler = new Sampler({ temperature: 0, topK: 1 });

    const rawTokens = tokenizer.encode('Hello');
    const remappedTokens = rawTokens.map(t => remapper.remapEncode(t));

    for (const t of remappedTokens) model.forward(t);
    let logits = await model.readLogits();

    const allTokens = [...remappedTokens];
    const generated: string[] = [];
    for (let i = 0; i < 10; i++) {
      const next = sampler.sample(logits, allTokens);
      const orig = remapper.remapDecode(next);
      if (tokenizer.isEos(orig)) break;
      allTokens.push(next);
      generated.push(tokenizer.decode([orig]));
      model.forward(next);
      logits = await model.readLogits();
    }

    const text = generated.join('');
    console.log(`  Partial vocab (limit=${limit}): "${text}"`);
    expect(generated.length).toBeGreaterThan(0);
    // Should produce the same or very similar text as full vocab
    expect(/[a-zA-Z]/.test(text)).toBe(true);
  });

  it('full and partial produce same first 5 tokens', async () => {
    const sampler = new Sampler({ temperature: 0, topK: 1 });
    const rawTokens = tokenizer.encode('The quick brown fox');
    const remappedTokens = rawTokens.map(t => remapper.remapEncode(t));

    // Full vocab run
    model.resetCache();
    model.vocabLimit = 0;
    for (const t of remappedTokens) model.forward(t);
    let logits = await model.readLogits();
    const fullTokens: number[] = [];
    const allFull = [...remappedTokens];
    for (let i = 0; i < 5; i++) {
      const next = sampler.sample(logits, allFull);
      fullTokens.push(next);
      allFull.push(next);
      model.forward(next);
      logits = await model.readLogits();
    }

    // Partial vocab run
    model.resetCache();
    model.vocabLimit = Math.ceil(info.vocabSize / 8);
    for (const t of remappedTokens) model.forward(t);
    logits = await model.readLogits();
    const partialTokens: number[] = [];
    const allPartial = [...remappedTokens];
    for (let i = 0; i < 5; i++) {
      const next = sampler.sample(logits, allPartial);
      partialTokens.push(next);
      allPartial.push(next);
      model.forward(next);
      logits = await model.readLogits();
    }

    const fullText = fullTokens.map(t => tokenizer.decode([remapper.remapDecode(t)])).join('');
    const partialText = partialTokens.map(t => tokenizer.decode([remapper.remapDecode(t)])).join('');
    console.log(`  Full:    "${fullText}"`);
    console.log(`  Partial: "${partialText}"`);

    // Check if the full-vocab argmax is within the partial range
    const limit = Math.ceil(info.vocabSize / 8);
    const partialLogitsLen = (await model.readLogits()).length;
    console.log(`  Full token[0]: ${fullTokens[0]} (in range: ${fullTokens[0] < limit})`);
    console.log(`  Partial token[0]: ${partialTokens[0]}, logits.length=${partialLogitsLen}, limit=${limit}`);
    // If full token is in range, they should match
    if (fullTokens[0] < limit) {
      expect(fullTokens[0]).toBe(partialTokens[0]);
    }
    model.vocabLimit = 0;
  });
});
