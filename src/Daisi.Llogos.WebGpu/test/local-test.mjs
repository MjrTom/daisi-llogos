/**
 * Local test: run CPU forward pass on the GGUF file to verify correctness.
 * Usage: node test/local-test.mjs
 */
import { readFileSync } from 'fs';

const MODEL_PATH = 'C:/GGUFS/tinyllama-1.1b-chat-v1.0.Q4_0.gguf';
console.log('Loading GGUF:', MODEL_PATH);
const fileBuf = readFileSync(MODEL_PATH);
const ab = fileBuf.buffer.slice(fileBuf.byteOffset, fileBuf.byteOffset + fileBuf.byteLength);

// Parse GGUF
import { parseGguf } from '../dist/index.js';
const info = parseGguf(ab);
console.log(`Parsed: ${info.architecture} ${info.blockCount}L ${info.embeddingLength}D vocab=${info.vocabSize}`);
console.log(`tensorDataOffset=${info.tensorDataOffset}`);

// Find key tensors
const findTensor = (name) => {
  const t = info.tensors.find(t => t.name === name);
  if (!t) throw new Error(`Missing tensor: ${name}`);
  return t;
};

// Dequant Q4_0 to F32
function dequantQ4_0(buffer, elementCount) {
  const bytes = new Uint8Array(buffer);
  const view = new DataView(buffer);
  const result = new Float32Array(elementCount);
  const blockCount = Math.ceil(elementCount / 32);
  for (let b = 0; b < blockCount; b++) {
    const bo = b * 18;
    const bits = view.getUint16(bo, true);
    const sign = (bits >> 15) & 1;
    const exp = (bits >> 10) & 0x1F;
    const mant = bits & 0x3FF;
    let scale;
    if (exp === 0 && mant === 0) scale = 0;
    else if (exp === 0) { scale = (mant / 1024) * Math.pow(2, -14); if (sign) scale = -scale; }
    else if (exp === 31) scale = 0;
    else { scale = (1 + mant / 1024) * Math.pow(2, exp - 15); if (sign) scale = -scale; }

    for (let j = 0; j < 16; j++) {
      const bv = bytes[bo + 2 + j];
      result[b * 32 + j] = scale * ((bv & 0x0F) - 8);
      result[b * 32 + j + 16] = scale * (((bv >> 4) & 0x0F) - 8);
    }
  }
  return result;
}

// Dequant Q6_K to F32
function dequantQ6_K(buffer, elementCount) {
  const bytes = new Uint8Array(buffer);
  const view = new DataView(buffer);
  const result = new Float32Array(elementCount);
  const blockCount = Math.ceil(elementCount / 256);
  for (let b = 0; b < blockCount; b++) {
    const bo = b * 210;
    const bits = view.getUint16(bo + 208, true);
    const sign = (bits >> 15) & 1;
    const exp = (bits >> 10) & 0x1F;
    const mant = bits & 0x3FF;
    let d;
    if (exp === 0 && mant === 0) d = 0;
    else if (exp === 0) { d = (mant / 1024) * Math.pow(2, -14); if (sign) d = -d; }
    else if (exp === 31) d = 0;
    else { d = (1 + mant / 1024) * Math.pow(2, exp - 15); if (sign) d = -d; }

    for (let n = 0; n < 256; n += 128) {
      for (let l = 0; l < 32; l++) {
        const is_ = n / 16;
        const qlIdx0 = bo + n / 2 + l;
        const qlIdx1 = bo + n / 2 + l + 32;
        const qhIdx = bo + 128 + n / 4 + l;
        const qhByte = bytes[qhIdx];
        const q1 = ((bytes[qlIdx0] & 0xF) | (((qhByte >> 0) & 3) << 4)) - 32;
        const q2 = ((bytes[qlIdx1] & 0xF) | (((qhByte >> 2) & 3) << 4)) - 32;
        const q3 = ((bytes[qlIdx0] >> 4) | (((qhByte >> 4) & 3) << 4)) - 32;
        const q4 = ((bytes[qlIdx1] >> 4) | (((qhByte >> 6) & 3) << 4)) - 32;
        const sc0 = view.getInt8(bo + 192 + is_ + 0);
        const sc2 = view.getInt8(bo + 192 + is_ + 2);
        const sc4 = view.getInt8(bo + 192 + is_ + 4);
        const sc6 = view.getInt8(bo + 192 + is_ + 6);
        const oi = b * 256 + n + l;
        if (oi < elementCount) result[oi] = d * sc0 * q1;
        if (oi + 32 < elementCount) result[oi + 32] = d * sc2 * q2;
        if (oi + 64 < elementCount) result[oi + 64] = d * sc4 * q3;
        if (oi + 96 < elementCount) result[oi + 96] = d * sc6 * q4;
      }
    }
  }
  return result;
}

function dequant(tensorInfo, buffer) {
  if (tensorInfo.type === 0) return new Float32Array(buffer); // F32
  if (tensorInfo.type === 2) return dequantQ4_0(buffer, tensorInfo.elementCount);
  if (tensorInfo.type === 14) return dequantQ6_K(buffer, tensorInfo.elementCount);
  throw new Error(`Unsupported type: ${tensorInfo.type}`);
}

function extractTensor(name) {
  const t = findTensor(name);
  const buf = ab.slice(info.tensorDataOffset + t.offset, info.tensorDataOffset + t.offset + t.byteSize);
  return { data: dequant(t, buf), info: t };
}

// Load essential weights
console.log('Loading weights...');
const embedding = extractTensor('token_embd.weight');
const outputWeight = extractTensor('output.weight');
const outputNorm = extractTensor('output_norm.weight');

console.log(`Embedding[0..4] for token 0: [${Array.from(embedding.data.slice(0, 5)).map(v => v.toFixed(6))}]`);
console.log(`Embedding[2048..2052] for token 1 (BOS): [${Array.from(embedding.data.slice(2048, 2053)).map(v => v.toFixed(6))}]`);

// Load layer 0 weights
const L0 = {
  attnNorm: extractTensor('blk.0.attn_norm.weight'),
  q: extractTensor('blk.0.attn_q.weight'),
  k: extractTensor('blk.0.attn_k.weight'),
  v: extractTensor('blk.0.attn_v.weight'),
  o: extractTensor('blk.0.attn_output.weight'),
  postAttnNorm: extractTensor('blk.0.ffn_norm.weight'),
  gate: extractTensor('blk.0.ffn_gate.weight'),
  up: extractTensor('blk.0.ffn_up.weight'),
  down: extractTensor('blk.0.ffn_down.weight'),
};

const E = info.embeddingLength;
const numHeads = info.headCount;
const numKvHeads = info.headCountKv;
const headDim = E / numHeads;
const ffnDim = info.feedForwardLength;
const vocabSize = info.vocabSize;
const rmsEps = info.rmsNormEps;
const ropeTheta = info.ropeFreqBase;

console.log(`Config: E=${E} heads=${numHeads} kvHeads=${numKvHeads} headDim=${headDim} ffn=${ffnDim} vocab=${vocabSize} eps=${rmsEps} theta=${ropeTheta}`);

// CPU forward pass for BOS token (token 1) through just layer 0
const tokenId = 1;
const hidden = new Float32Array(E);
for (let i = 0; i < E; i++) hidden[i] = embedding.data[tokenId * E + i];
console.log(`\nBOS hidden[0..4]: [${Array.from(hidden.slice(0, 5)).map(v => v.toFixed(6))}]`);

// RMSNorm
function rmsNorm(input, weight, n, eps) {
  let sumSq = 0;
  for (let i = 0; i < n; i++) sumSq += input[i] * input[i];
  const rms = Math.sqrt(sumSq / n + eps);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = (input[i] / rms) * weight[i];
  return out;
}

// Matmul
function matmul(weights, input, M, K) {
  const out = new Float32Array(M);
  for (let row = 0; row < M; row++) {
    let sum = 0;
    const off = row * K;
    for (let k = 0; k < K; k++) sum += weights[off + k] * input[k];
    out[row] = sum;
  }
  return out;
}

const normed = rmsNorm(hidden, L0.attnNorm.data, E, rmsEps);
console.log(`L0 normed[0..4]: [${Array.from(normed.slice(0, 5)).map(v => v.toFixed(6))}]`);

const qProj = matmul(L0.q.data, normed, numHeads * headDim, E);
console.log(`L0 q[0..4]: [${Array.from(qProj.slice(0, 5)).map(v => v.toFixed(6))}]`);

// Skip full layer — just do embedding → norm → lm_head for a quick test
const finalNormed = rmsNorm(hidden, outputNorm.data, E, rmsEps);
console.log(`\nSkip-layers test: embedding → norm → lm_head`);
console.log(`Final normed[0..4]: [${Array.from(finalNormed.slice(0, 5)).map(v => v.toFixed(6))}]`);

const logits = matmul(outputWeight.data, finalNormed, vocabSize, E);
const indexed = Array.from(logits).map((v, i) => ({ id: i, logit: v }));
indexed.sort((a, b) => b.logit - a.logit);
console.log(`Top-5:`, indexed.slice(0, 5).map(t => `${t.id}(${t.logit.toFixed(2)})`));

// Full forward pass through ALL layers for prompt "The capital of France is"
console.log('\n=== Full forward pass ===');
const promptTokens = [1, 1576, 35, 5030, 2410, 35, 974, 35, 16066, 35, 275]; // BOS + "The capital of France is" (matching llogos)

// Load ALL layer weights
const allLayers = [];
for (let i = 0; i < info.blockCount; i++) {
  allLayers.push({
    attnNorm: extractTensor(`blk.${i}.attn_norm.weight`),
    q: extractTensor(`blk.${i}.attn_q.weight`),
    k: extractTensor(`blk.${i}.attn_k.weight`),
    v: extractTensor(`blk.${i}.attn_v.weight`),
    o: extractTensor(`blk.${i}.attn_output.weight`),
    postAttnNorm: extractTensor(`blk.${i}.ffn_norm.weight`),
    gate: extractTensor(`blk.${i}.ffn_gate.weight`),
    up: extractTensor(`blk.${i}.ffn_up.weight`),
    down: extractTensor(`blk.${i}.ffn_down.weight`),
  });
  if (i % 5 === 0) process.stdout.write(`  layer ${i}..`);
}
console.log(' done');

// KV caches: [layer][numKvHeads][maxSeqLen][headDim]
const maxSeqLen = 64; // small for test
const kvK = allLayers.map(() => new Float32Array(numKvHeads * maxSeqLen * headDim));
const kvV = allLayers.map(() => new Float32Array(numKvHeads * maxSeqLen * headDim));
let seqLen = 0;

function cpuRope(data, hDim, position, theta) {
  const nPairs = data.length / 2;
  for (let pi = 0; pi < nPairs; pi++) {
    const hp = pi % (hDim / 2);
    const dimFrac = (hp * 2) / hDim;
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

for (const tokenId of promptTokens) {
  const h = new Float32Array(E);
  if (seqLen === 0) {
    // First token: initialize hidden from embedding
    for (let i = 0; i < E; i++) h[i] = embedding.data[tokenId * E + i];
  } else {
    // Subsequent tokens need the hidden state from previous layers
    // Actually, each token is processed independently through all layers
    for (let i = 0; i < E; i++) h[i] = embedding.data[tokenId * E + i];
  }

  // Process through all layers
  for (let layer = 0; layer < info.blockCount; layer++) {
    const lw = allLayers[layer];
    const residual = new Float32Array(h);

    const n1 = rmsNorm(h, lw.attnNorm.data, E, rmsEps);
    const qP = matmul(lw.q.data, n1, numHeads * headDim, E);
    const kP = matmul(lw.k.data, n1, numKvHeads * headDim, E);
    const vP = matmul(lw.v.data, n1, numKvHeads * headDim, E);

    cpuRope(qP, headDim, seqLen, ropeTheta);
    cpuRope(kP, headDim, seqLen, ropeTheta);

    // Write K,V to cache
    for (let kh = 0; kh < numKvHeads; kh++) {
      for (let d = 0; d < headDim; d++) {
        kvK[layer][kh * maxSeqLen * headDim + seqLen * headDim + d] = kP[kh * headDim + d];
        kvV[layer][kh * maxSeqLen * headDim + seqLen * headDim + d] = vP[kh * headDim + d];
      }
    }

    // Attention
    const headsPerGroup = numHeads / numKvHeads;
    const attnOut = new Float32Array(numHeads * headDim);
    const curSeqLen = seqLen + 1;
    const scale = 1.0 / Math.sqrt(headDim);

    for (let head = 0; head < numHeads; head++) {
      const kvHead = Math.floor(head / headsPerGroup);

      // QK scores
      const scores = new Float32Array(curSeqLen);
      for (let pos = 0; pos < curSeqLen; pos++) {
        let dot = 0;
        for (let d = 0; d < headDim; d++) {
          dot += qP[head * headDim + d] * kvK[layer][kvHead * maxSeqLen * headDim + pos * headDim + d];
        }
        scores[pos] = dot * scale;
      }

      // Softmax
      let maxS = -Infinity;
      for (let i = 0; i < curSeqLen; i++) maxS = Math.max(maxS, scores[i]);
      let sumE = 0;
      for (let i = 0; i < curSeqLen; i++) { scores[i] = Math.exp(scores[i] - maxS); sumE += scores[i]; }
      for (let i = 0; i < curSeqLen; i++) scores[i] /= sumE;

      // Weighted V
      for (let d = 0; d < headDim; d++) {
        let acc = 0;
        for (let pos = 0; pos < curSeqLen; pos++) {
          acc += scores[pos] * kvV[layer][kvHead * maxSeqLen * headDim + pos * headDim + d];
        }
        attnOut[head * headDim + d] = acc;
      }
    }

    // O projection + residual
    const oP = matmul(lw.o.data, attnOut, E, numHeads * headDim);
    for (let i = 0; i < E; i++) h[i] = oP[i] + residual[i];

    // FFN
    const residual2 = new Float32Array(h);
    const n2 = rmsNorm(h, lw.postAttnNorm.data, E, rmsEps);
    const gateOut = matmul(lw.gate.data, n2, ffnDim, E);
    const upOut = matmul(lw.up.data, n2, ffnDim, E);
    const ffnOut = new Float32Array(ffnDim);
    for (let i = 0; i < ffnDim; i++) {
      ffnOut[i] = (gateOut[i] / (1 + Math.exp(-gateOut[i]))) * upOut[i];
    }
    const downOut = matmul(lw.down.data, ffnOut, E, ffnDim);
    for (let i = 0; i < E; i++) h[i] = downOut[i] + residual2[i];
  }

  seqLen++;
  process.stdout.write(`Token ${tokenId} done (seqLen=${seqLen})\n`);

  // After last token, compute logits
  if (tokenId === promptTokens[promptTokens.length - 1]) {
    const fn = rmsNorm(h, outputNorm.data, E, rmsEps);
    const lgt = matmul(outputWeight.data, fn, vocabSize, E);
    const idx = Array.from(lgt).map((v, i) => ({ id: i, logit: v }));
    idx.sort((a, b) => b.logit - a.logit);
    console.log(`\nFull model top-5:`, idx.slice(0, 10).map(t => `${t.id}(${t.logit.toFixed(2)})`));
  }
}

// Generate 20 tokens
const tokens = info.metadata.get('tokenizer.ggml.tokens');
const eosId = info.metadata.get('tokenizer.ggml.eos_token_id') ?? 2;
console.log('\n=== Generating text ===');
process.stdout.write('> The capital of France is');

// h still has the hidden state from the last prompt token
// Continue generating from the last prompt position
function sampleWithTemp(logits, temperature = 0.7) {
  // Apply temperature
  for (let i = 0; i < logits.length; i++) logits[i] /= temperature;
  // Softmax + top-k
  const topK = 40;
  const indices = Array.from({length: logits.length}, (_, i) => i);
  indices.sort((a, b) => logits[b] - logits[a]);
  const topIndices = indices.slice(0, topK);
  let maxL = logits[topIndices[0]];
  const probs = topIndices.map(i => Math.exp(logits[i] - maxL));
  let sum = probs.reduce((a, b) => a + b, 0);
  for (let i = 0; i < probs.length; i++) probs[i] /= sum;
  // Sample
  let r = Math.random();
  for (let i = 0; i < probs.length; i++) {
    r -= probs[i];
    if (r <= 0) return topIndices[i];
  }
  return topIndices[0];
}

let lastH = new Float32Array(E);
// We need to re-get h from the last forward pass - it's in the closure scope as 'h' but that's block-scoped
// Actually h was declared with `const h = new Float32Array(E)` in the for-of loop, so we lost it.
// Let me just run one more forward pass for the last token to get the hidden state back.

// Actually, let me just generate fresh. Reset state.
seqLen = 0;
for (const kv of kvK) kv.fill(0);
for (const kv of kvV) kv.fill(0);

// Prefill
let genH = new Float32Array(E);
for (const tid of promptTokens) {
  for (let i = 0; i < E; i++) genH[i] = embedding.data[tid * E + i];
  for (let layer = 0; layer < info.blockCount; layer++) {
    const lw = allLayers[layer];
    const residual = new Float32Array(genH);
    const n1 = rmsNorm(genH, lw.attnNorm.data, E, rmsEps);
    const qP = matmul(lw.q.data, n1, numHeads * headDim, E);
    const kP = matmul(lw.k.data, n1, numKvHeads * headDim, E);
    const vP = matmul(lw.v.data, n1, numKvHeads * headDim, E);
    cpuRope(qP, headDim, seqLen, ropeTheta);
    cpuRope(kP, headDim, seqLen, ropeTheta);
    for (let kh = 0; kh < numKvHeads; kh++)
      for (let d = 0; d < headDim; d++) {
        kvK[layer][kh * maxSeqLen * headDim + seqLen * headDim + d] = kP[kh * headDim + d];
        kvV[layer][kh * maxSeqLen * headDim + seqLen * headDim + d] = vP[kh * headDim + d];
      }
    const curSeqLen = seqLen + 1;
    const scale = 1.0 / Math.sqrt(headDim);
    const hpg = numHeads / numKvHeads;
    const attnOut = new Float32Array(numHeads * headDim);
    for (let head = 0; head < numHeads; head++) {
      const kvHead = Math.floor(head / hpg);
      const scores = new Float32Array(curSeqLen);
      for (let pos = 0; pos < curSeqLen; pos++) {
        let dot = 0;
        for (let d = 0; d < headDim; d++)
          dot += qP[head * headDim + d] * kvK[layer][kvHead * maxSeqLen * headDim + pos * headDim + d];
        scores[pos] = dot * scale;
      }
      let maxS = -Infinity;
      for (let i = 0; i < curSeqLen; i++) maxS = Math.max(maxS, scores[i]);
      let sumE = 0;
      for (let i = 0; i < curSeqLen; i++) { scores[i] = Math.exp(scores[i] - maxS); sumE += scores[i]; }
      for (let i = 0; i < curSeqLen; i++) scores[i] /= sumE;
      for (let d = 0; d < headDim; d++) {
        let acc = 0;
        for (let pos = 0; pos < curSeqLen; pos++)
          acc += scores[pos] * kvV[layer][kvHead * maxSeqLen * headDim + pos * headDim + d];
        attnOut[head * headDim + d] = acc;
      }
    }
    const oP = matmul(lw.o.data, attnOut, E, numHeads * headDim);
    for (let i = 0; i < E; i++) genH[i] = oP[i] + residual[i];
    const residual2 = new Float32Array(genH);
    const n2 = rmsNorm(genH, lw.postAttnNorm.data, E, rmsEps);
    const g = matmul(lw.gate.data, n2, ffnDim, E);
    const u = matmul(lw.up.data, n2, ffnDim, E);
    const ff = new Float32Array(ffnDim);
    for (let i = 0; i < ffnDim; i++) ff[i] = (g[i] / (1 + Math.exp(-g[i]))) * u[i];
    const dn = matmul(lw.down.data, ff, E, ffnDim);
    for (let i = 0; i < E; i++) genH[i] = dn[i] + residual2[i];
  }
  seqLen++;
}

// Generate
for (let step = 0; step < 20; step++) {
  const fn = rmsNorm(genH, outputNorm.data, E, rmsEps);
  const lgt = matmul(outputWeight.data, fn, vocabSize, E);
  const nextTok = sampleWithTemp(lgt, 0.7);
  if (nextTok === eosId) { process.stdout.write(' [EOS]'); break; }
  const tokStr = (tokens[nextTok] || '').replace(/▁/g, ' ');
  process.stdout.write(tokStr);

  // Forward pass for next token
  for (let i = 0; i < E; i++) genH[i] = embedding.data[nextTok * E + i];
  for (let layer = 0; layer < info.blockCount; layer++) {
    const lw = allLayers[layer];
    const residual = new Float32Array(genH);
    const n1 = rmsNorm(genH, lw.attnNorm.data, E, rmsEps);
    const qP = matmul(lw.q.data, n1, numHeads * headDim, E);
    const kP = matmul(lw.k.data, n1, numKvHeads * headDim, E);
    const vP = matmul(lw.v.data, n1, numKvHeads * headDim, E);
    cpuRope(qP, headDim, seqLen, ropeTheta);
    cpuRope(kP, headDim, seqLen, ropeTheta);
    for (let kh = 0; kh < numKvHeads; kh++)
      for (let d = 0; d < headDim; d++) {
        kvK[layer][kh * maxSeqLen * headDim + seqLen * headDim + d] = kP[kh * headDim + d];
        kvV[layer][kh * maxSeqLen * headDim + seqLen * headDim + d] = vP[kh * headDim + d];
      }
    const curSeqLen = seqLen + 1;
    const scale = 1.0 / Math.sqrt(headDim);
    const hpg = numHeads / numKvHeads;
    const attnOut = new Float32Array(numHeads * headDim);
    for (let head = 0; head < numHeads; head++) {
      const kvHead = Math.floor(head / hpg);
      const scores = new Float32Array(curSeqLen);
      for (let pos = 0; pos < curSeqLen; pos++) {
        let dot = 0;
        for (let d = 0; d < headDim; d++)
          dot += qP[head * headDim + d] * kvK[layer][kvHead * maxSeqLen * headDim + pos * headDim + d];
        scores[pos] = dot * scale;
      }
      let maxS = -Infinity;
      for (let i = 0; i < curSeqLen; i++) maxS = Math.max(maxS, scores[i]);
      let sumE = 0;
      for (let i = 0; i < curSeqLen; i++) { scores[i] = Math.exp(scores[i] - maxS); sumE += scores[i]; }
      for (let i = 0; i < curSeqLen; i++) scores[i] /= sumE;
      for (let d = 0; d < headDim; d++) {
        let acc = 0;
        for (let pos = 0; pos < curSeqLen; pos++)
          acc += scores[pos] * kvV[layer][kvHead * maxSeqLen * headDim + pos * headDim + d];
        attnOut[head * headDim + d] = acc;
      }
    }
    const oP = matmul(lw.o.data, attnOut, E, numHeads * headDim);
    for (let i = 0; i < E; i++) genH[i] = oP[i] + residual[i];
    const residual2 = new Float32Array(genH);
    const n2 = rmsNorm(genH, lw.postAttnNorm.data, E, rmsEps);
    const g = matmul(lw.gate.data, n2, ffnDim, E);
    const u = matmul(lw.up.data, n2, ffnDim, E);
    const ff = new Float32Array(ffnDim);
    for (let i = 0; i < ffnDim; i++) ff[i] = (g[i] / (1 + Math.exp(-g[i]))) * u[i];
    const dn = matmul(lw.down.data, ff, E, ffnDim);
    for (let i = 0; i < E; i++) genH[i] = dn[i] + residual2[i];
  }
  seqLen++;
}
console.log('\n');
