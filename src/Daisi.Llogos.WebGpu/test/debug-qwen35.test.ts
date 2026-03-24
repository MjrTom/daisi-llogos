import './setup-webgpu.js';
import { it } from 'vitest';
import { readFileSync } from 'fs';
import { parseGguf } from '../src/gguf/gguf-parser.js';
import { extractTensorData } from '../src/storage/download-manager.js';
import { tokenizerFromGguf } from '../src/tokenizer/bpe-tokenizer.js';
import { Qwen35Model } from '../src/model/qwen35-model.js';
import { Sampler } from '../src/model/sampler.js';
import { initGpu } from '../src/gpu/device.js';
import { ComputeEngine } from '../src/gpu/compute.js';

it('debug qwen35', async () => {
  const buf = readFileSync('C:/GGUFS/Qwen3.5-0.8B-Q8_0.gguf');
  const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
  const info = parseGguf(ab);
  const tokenizer = tokenizerFromGguf(info.metadata);
  const tensorMap = new Map<string, any>();
  for (const tensor of info.tensors) {
    tensorMap.set(tensor.name, { buffer: extractTensorData(ab, info.tensorDataOffset + tensor.offset, tensor.byteSize), info: tensor });
  }

  const gpu = await initGpu();
  const compute = new ComputeEngine(gpu.device);
  const model = new Qwen35Model(compute, info);
  await model.initWeights(tensorMap);

  // Match C# diagnostic: single token "Hello"
  const helloTokens = tokenizer.encode('Hello');
  console.log('Token IDs for Hello:', helloTokens, '(C# ref: 9419)');

  await model.forward(helloTokens[0]);
  let logits = await model.readLogits();
  console.log('Logits length:', logits.length);
  let logitSum = 0, zeroCount = 0, nanCount = 0;
  for (const v of logits) { logitSum += v; if (v === 0) zeroCount++; if (!isFinite(v)) nanCount++; }
  console.log('Logits sum:', logitSum.toFixed(2), '(C# ref: -1013152.75)');
  console.log('Zero logits:', zeroCount, '/', logits.length, 'NaN/Inf:', nanCount);

  // Compare against C# reference:
  // C# top: [11:12.720, 13:11.520, 0:10.845, 198:10.359, 271:9.943]
  console.log('C# ref logits[11]:', 12.720, 'ours:', logits[11]?.toFixed(3));
  console.log('C# ref logits[13]:', 11.520, 'ours:', logits[13]?.toFixed(3));
  console.log('C# ref logits[0]:', 10.845, 'ours:', logits[0]?.toFixed(3));
  console.log('C# ref logits[198]:', 10.359, 'ours:', logits[198]?.toFixed(3));

  // Show top 10
  const indexed = Array.from(logits).map((v, i) => ({v, i}));
  indexed.sort((a, b) => b.v - a.v);
  console.log('Top 10 after prefill:');
  for (const t of indexed.slice(0, 10)) console.log(`  ${t.i} = ${JSON.stringify(tokenizer.decode([t.i]))} (${t.v.toFixed(2)})`);

  // Generate
  const allTokens = [...inputTokens];
  for (let i = 0; i < 30; i++) {
    const next = sampler.sample(logits, allTokens);
    const text = tokenizer.decode([next]);
    console.log(`step ${i}: ${next} = ${JSON.stringify(text)}`);
    if (tokenizer.isEos(next)) break;
    allTokens.push(next);
    await model.forward(next);
    logits = await model.readLogits();
  }
}, 120000);
