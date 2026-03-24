/**
 * Compare Layer 0 DeltaNet output between WebGPU and C# reference.
 * Strategy: run only layer 0, dump hidden state after it, compare.
 */
import './setup-webgpu.js';
import { it } from 'vitest';
import { readFileSync } from 'fs';
import { parseGguf } from '../src/gguf/gguf-parser.js';
import { extractTensorData } from '../src/storage/download-manager.js';
import { tokenizerFromGguf } from '../src/tokenizer/bpe-tokenizer.js';
import { Qwen35Model } from '../src/model/qwen35-model.js';
import { initGpu } from '../src/gpu/device.js';
import { ComputeEngine } from '../src/gpu/compute.js';

it('layer 0 isolation test', async () => {
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
  // Temporarily set numLayers to 1 to test just layer 0
  (model as any).info = { ...info, blockCount: 1 };
  await model.initWeights(tensorMap);
  // Restore for forward pass
  (model as any).info = info;
  // But the weights.layers only has 1 layer now. Let me just run forward with modified model.

  // Actually, let me just run the full model but read hidden after layer 0.
  // The model already has debug logging for layer 0.

  // For a cleaner approach, let me just run the full 24-layer model and
  // focus on specific logit indices to compare with C# reference.

  // C# reference for token 9419 "Hello":
  // argmax=11, logits[11]=12.720, logits[13]=11.520, logits[0]=10.845
  // final hidden: max=6.2413, h[0]=-0.2532

  // The key question is: are the WEIGHTS loaded correctly?
  // Let me verify by computing a simple matmul manually.

  // Read the embedding for token 9419
  const embW = tensorMap.get('token_embd.weight')!;
  const E = info.embeddingLength; // 1024
  // Dequantize embedding row for token 9419
  const embData = new Float32Array(embW.info.elementCount);
  // For Q8_0: each block = 34 bytes (2 byte scale + 32 int8s)
  const embBytes = new Uint8Array(embW.buffer);
  const embView = new DataView(embW.buffer);
  const blockCount = Math.ceil(embW.info.elementCount / 32);
  for (let b = 0; b < blockCount; b++) {
    const bo = b * 34;
    const scale = f16ToF32(embView.getUint16(bo, true));
    for (let q = 0; q < 32 && b * 32 + q < embW.info.elementCount; q++) {
      embData[b * 32 + q] = scale * embView.getInt8(bo + 2 + q);
    }
  }

  // Extract just token 9419's embedding
  const tokenEmb = embData.slice(9419 * E, 9419 * E + E);
  console.log(`Token 9419 embedding: max=${Math.max(...tokenEmb.map(Math.abs)).toFixed(4)} e[0..4]=${Array.from(tokenEmb.slice(0, 5)).map(v => v.toFixed(4))}`);

  // Now compute RMSNorm(embedding) with layer 0 attn_norm
  const normW = tensorMap.get('blk.0.attn_norm.weight')!;
  const normData = new Float32Array(normW.buffer.slice(0)); // F32

  // RMSNorm
  let sumSq = 0;
  for (let i = 0; i < E; i++) sumSq += tokenEmb[i] * tokenEmb[i];
  const rms = Math.sqrt(sumSq / E + info.rmsNormEps);
  const normed = new Float32Array(E);
  for (let i = 0; i < E; i++) normed[i] = (tokenEmb[i] / rms) * normData[i];
  console.log(`Normed: max=${Math.max(...normed.map(Math.abs)).toFixed(4)} n[0..4]=${Array.from(normed.slice(0, 5)).map(v => v.toFixed(4))}`);

  // Now compute QKV = normed × attn_qkv.weight
  const qkvW = tensorMap.get('blk.0.attn_qkv.weight')!;
  const qkvDim = 6144;
  // Dequantize QKV weight (Q8_0)
  const qkvWeightData = new Float32Array(qkvW.info.elementCount);
  const qkvBytes = new Uint8Array(qkvW.buffer);
  const qkvView = new DataView(qkvW.buffer);
  const qkvBlocks = Math.ceil(qkvW.info.elementCount / 32);
  for (let b = 0; b < qkvBlocks; b++) {
    const bo = b * 34;
    const scale = f16ToF32(qkvView.getUint16(bo, true));
    for (let q = 0; q < 32 && b * 32 + q < qkvW.info.elementCount; q++) {
      qkvWeightData[b * 32 + q] = scale * qkvView.getInt8(bo + 2 + q);
    }
  }

  // CPU matmul: qkv[row] = sum(weight[row*K+k] * normed[k])
  const qkvResult = new Float32Array(qkvDim);
  for (let row = 0; row < qkvDim; row++) {
    let dot = 0;
    for (let k = 0; k < E; k++) {
      dot += qkvWeightData[row * E + k] * normed[k];
    }
    qkvResult[row] = dot;
  }
  console.log(`QKV matmul (CPU): max=${Math.max(...qkvResult.slice(0, 100).map(Math.abs)).toFixed(4)} qkv[0..4]=${Array.from(qkvResult.slice(0, 5)).map(v => v.toFixed(4))}`);

  // Now compare with GPU QKV matmul
  const model2 = new Qwen35Model(compute, info);
  await model2.initWeights(tensorMap);
  await model2.forward(9419);
  // The debug output from model2.forward will show the QKV values from GPU
  // We just need to compare.

}, 120000);

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
