/**
 * Full CPU forward pass for a single token — no GPU at all.
 * Used to verify the model weights are correct by comparing output with GPU.
 */

export async function cpuForwardOneToken(
  tokenId: number,
  embeddingF32: Float32Array,  // dequanted embedding [vocabSize * embDim]
  outputF32: Float32Array,     // dequanted output weight [vocabSize * embDim]
  outputNormF32: Float32Array, // output norm weight [embDim]
  layerWeights: Array<{
    attnNorm: Float32Array;
    q: Float32Array; k: Float32Array; v: Float32Array; o: Float32Array;
    postAttnNorm: Float32Array;
    gate: Float32Array; up: Float32Array; down: Float32Array;
  }>,
  config: {
    embDim: number; numHeads: number; numKvHeads: number; headDim: number;
    ffnDim: number; vocabSize: number; rmsEps: number; ropeTheta: number;
  },
): Promise<{ topTokens: Array<{ id: number; logit: number }> }> {
  const { embDim: E, numHeads, numKvHeads, headDim, ffnDim, vocabSize, rmsEps, ropeTheta } = config;
  const headsPerKvGroup = numHeads / numKvHeads;

  // 1. Embedding lookup
  const hidden = new Float32Array(E);
  for (let i = 0; i < E; i++) hidden[i] = embeddingF32[tokenId * E + i];
  console.log(`[cpu] embedding[0..4]: [${Array.from(hidden.slice(0, 5)).map(v => v.toFixed(6))}]`);

  // KV caches (only need 1 position for single token)
  const kvCaches = layerWeights.map(() => ({
    k: new Float32Array(numKvHeads * headDim),
    v: new Float32Array(numKvHeads * headDim),
  }));

  // 2. Transformer layers
  for (let layer = 0; layer < layerWeights.length; layer++) {
    const lw = layerWeights[layer];
    const residual = new Float32Array(hidden);

    // RMSNorm
    const normed = cpuRmsNorm(hidden, lw.attnNorm, E, rmsEps);
    if (layer === 0) console.log(`[cpu] L0 normed[0..4]: [${Array.from(normed.slice(0, 5)).map(v => v.toFixed(6))}]`);

    // Q, K, V projections
    const qProj = cpuMatmul(lw.q, normed, numHeads * headDim, E);
    const kProj = cpuMatmul(lw.k, normed, numKvHeads * headDim, E);
    const vProj = cpuMatmul(lw.v, normed, numKvHeads * headDim, E);
    if (layer === 0) console.log(`[cpu] L0 q[0..4]: [${Array.from(qProj.slice(0, 5)).map(v => v.toFixed(6))}]`);

    // RoPE (position 0 for single token)
    cpuRope(qProj, headDim, 0, ropeTheta);
    cpuRope(kProj, headDim, 0, ropeTheta);

    // Store K, V
    kvCaches[layer].k.set(kProj);
    kvCaches[layer].v.set(vProj);

    // Attention (trivial for seq_len=1: output = V)
    const attnOut = new Float32Array(numHeads * headDim);
    for (let h = 0; h < numHeads; h++) {
      const kvH = Math.floor(h / headsPerKvGroup);
      for (let d = 0; d < headDim; d++) {
        attnOut[h * headDim + d] = vProj[kvH * headDim + d];
      }
    }

    // Output projection + residual
    const oProj = cpuMatmul(lw.o, attnOut, E, numHeads * headDim);
    for (let i = 0; i < E; i++) hidden[i] = oProj[i] + residual[i];

    // FFN
    const residual2 = new Float32Array(hidden);
    const normed2 = cpuRmsNorm(hidden, lw.postAttnNorm, E, rmsEps);
    const gateOut = cpuMatmul(lw.gate, normed2, ffnDim, E);
    const upOut = cpuMatmul(lw.up, normed2, ffnDim, E);

    // SiLU(gate) * up
    const ffnOut = new Float32Array(ffnDim);
    for (let i = 0; i < ffnDim; i++) {
      const silu = gateOut[i] / (1 + Math.exp(-gateOut[i]));
      ffnOut[i] = silu * upOut[i];
    }

    const downOut = cpuMatmul(lw.down, ffnOut, E, ffnDim);
    for (let i = 0; i < E; i++) hidden[i] = downOut[i] + residual2[i];

    if (layer === 0) console.log(`[cpu] L0 hidden[0..4]: [${Array.from(hidden.slice(0, 5)).map(v => v.toFixed(6))}]`);
  }

  // 3. Final norm + logits
  const finalNormed = cpuRmsNorm(hidden, outputNormF32, E, rmsEps);
  console.log(`[cpu] final normed[0..4]: [${Array.from(finalNormed.slice(0, 5)).map(v => v.toFixed(6))}]`);

  const logits = cpuMatmul(outputF32, finalNormed, vocabSize, E);

  // Top 5
  const indexed = Array.from(logits).map((v, i) => ({ id: i, logit: v }));
  indexed.sort((a, b) => b.logit - a.logit);
  const top5 = indexed.slice(0, 5);
  console.log(`[cpu] Top-5:`, top5.map(t => `${t.id}(${t.logit.toFixed(2)})`));

  return { topTokens: top5 };
}

function cpuRmsNorm(input: Float32Array, weight: Float32Array, n: number, eps: number): Float32Array {
  let sumSq = 0;
  for (let i = 0; i < n; i++) sumSq += input[i] * input[i];
  const rms = Math.sqrt(sumSq / n + eps);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = (input[i] / rms) * weight[i];
  return out;
}

function cpuMatmul(weights: Float32Array, input: Float32Array, M: number, K: number): Float32Array {
  const out = new Float32Array(M);
  for (let row = 0; row < M; row++) {
    let sum = 0;
    for (let k = 0; k < K; k++) sum += weights[row * K + k] * input[k];
    out[row] = sum;
  }
  return out;
}

function cpuRope(data: Float32Array, headDim: number, position: number, theta: number): void {
  const nPairs = data.length / 2;
  for (let pairIdx = 0; pairIdx < nPairs; pairIdx++) {
    const headPair = pairIdx % (headDim / 2);
    const dimFrac = (headPair * 2) / headDim;
    const freq = 1.0 / Math.pow(theta, dimFrac);
    const angle = position * freq;
    const cos = Math.cos(angle);
    const sin = Math.sin(angle);
    const i0 = pairIdx * 2;
    const i1 = i0 + 1;
    const x = data[i0];
    const y = data[i1];
    data[i0] = x * cos - y * sin;
    data[i1] = x * sin + y * cos;
  }
}
