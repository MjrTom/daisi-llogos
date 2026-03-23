/**
 * Web Worker for CPU inference — runs forward pass off the main thread.
 * Communicates via postMessage.
 */

// Types duplicated here since workers can't import from main bundle easily
interface WorkerState {
  embedding: Float32Array;
  outputNorm: Float32Array;
  output: Float32Array;
  layers: Array<{
    attnNorm: Float32Array; q: Float32Array; k: Float32Array; v: Float32Array; o: Float32Array;
    postAttnNorm: Float32Array; gate: Float32Array; up: Float32Array; down: Float32Array;
  }>;
  config: {
    embDim: number; numHeads: number; numKvHeads: number; headDim: number;
    ffnDim: number; vocabSize: number; rmsNormEps: number; ropeTheta: number;
    numLayers: number; maxSeqLen: number;
  };
  // Runtime state
  kvK: Float32Array[];
  kvV: Float32Array[];
  seqLen: number;
}

let state: WorkerState | null = null;

self.onmessage = (e: MessageEvent) => {
  const { type, data } = e.data;

  switch (type) {
    case 'init':
      state = data as WorkerState;
      // Initialize KV caches
      const c = state.config;
      state.kvK = [];
      state.kvV = [];
      for (let i = 0; i < c.numLayers; i++) {
        state.kvK.push(new Float32Array(c.numKvHeads * c.maxSeqLen * c.headDim));
        state.kvV.push(new Float32Array(c.numKvHeads * c.maxSeqLen * c.headDim));
      }
      state.seqLen = 0;
      self.postMessage({ type: 'ready' });
      break;

    case 'reset':
      if (state) {
        state.seqLen = 0;
        for (const kv of state.kvK) kv.fill(0);
        for (const kv of state.kvV) kv.fill(0);
      }
      self.postMessage({ type: 'reset_done' });
      break;

    case 'forward':
      if (!state) { self.postMessage({ type: 'error', data: 'Not initialized' }); break; }
      forward(data.tokenId);
      self.postMessage({ type: 'forward_done' });
      break;

    case 'get_logits':
      if (!state) { self.postMessage({ type: 'error', data: 'Not initialized' }); break; }
      const logits = getLogits();
      self.postMessage({ type: 'logits', data: logits });
      break;
  }
};

function forward(tokenId: number): void {
  const s = state!;
  const c = s.config;
  const E = c.embDim;
  const h = new Float32Array(E);

  // Embedding
  for (let i = 0; i < E; i++) h[i] = s.embedding[tokenId * E + i];

  // Layers
  for (let layer = 0; layer < c.numLayers; layer++) {
    const lw = s.layers[layer];
    const residual = new Float32Array(h);

    const normed = rmsNorm(h, lw.attnNorm, E, c.rmsNormEps);
    const qP = matvec(lw.q, normed, c.numHeads * c.headDim, E);
    const kP = matvec(lw.k, normed, c.numKvHeads * c.headDim, E);
    const vP = matvec(lw.v, normed, c.numKvHeads * c.headDim, E);

    rope(qP, c.headDim, s.seqLen, c.ropeTheta);
    rope(kP, c.headDim, s.seqLen, c.ropeTheta);

    // Write KV cache
    for (let kh = 0; kh < c.numKvHeads; kh++)
      for (let d = 0; d < c.headDim; d++) {
        s.kvK[layer][kh * c.maxSeqLen * c.headDim + s.seqLen * c.headDim + d] = kP[kh * c.headDim + d];
        s.kvV[layer][kh * c.maxSeqLen * c.headDim + s.seqLen * c.headDim + d] = vP[kh * c.headDim + d];
      }

    // Attention
    const curSeqLen = s.seqLen + 1;
    const scale = 1.0 / Math.sqrt(c.headDim);
    const headsPerGroup = c.numHeads / c.numKvHeads;
    const attnOut = new Float32Array(c.numHeads * c.headDim);
    for (let head = 0; head < c.numHeads; head++) {
      const kvHead = Math.floor(head / headsPerGroup);
      const scores = new Float32Array(curSeqLen);
      for (let pos = 0; pos < curSeqLen; pos++) {
        let dot = 0;
        for (let d = 0; d < c.headDim; d++)
          dot += qP[head * c.headDim + d] * s.kvK[layer][kvHead * c.maxSeqLen * c.headDim + pos * c.headDim + d];
        scores[pos] = dot * scale;
      }
      let maxS = -Infinity;
      for (let i = 0; i < curSeqLen; i++) maxS = Math.max(maxS, scores[i]);
      let sumE = 0;
      for (let i = 0; i < curSeqLen; i++) { scores[i] = Math.exp(scores[i] - maxS); sumE += scores[i]; }
      for (let i = 0; i < curSeqLen; i++) scores[i] /= sumE;
      for (let d = 0; d < c.headDim; d++) {
        let acc = 0;
        for (let pos = 0; pos < curSeqLen; pos++)
          acc += scores[pos] * s.kvV[layer][kvHead * c.maxSeqLen * c.headDim + pos * c.headDim + d];
        attnOut[head * c.headDim + d] = acc;
      }
    }

    const oP = matvec(lw.o, attnOut, E, c.numHeads * c.headDim);
    for (let i = 0; i < E; i++) h[i] = oP[i] + residual[i];

    // FFN
    const residual2 = new Float32Array(h);
    const normed2 = rmsNorm(h, lw.postAttnNorm, E, c.rmsNormEps);
    const gateOut = matvec(lw.gate, normed2, c.ffnDim, E);
    const upOut = matvec(lw.up, normed2, c.ffnDim, E);
    const ffnOut = new Float32Array(c.ffnDim);
    for (let i = 0; i < c.ffnDim; i++)
      ffnOut[i] = (gateOut[i] / (1 + Math.exp(-gateOut[i]))) * upOut[i];
    const downOut = matvec(lw.down, ffnOut, E, c.ffnDim);
    for (let i = 0; i < E; i++) h[i] = downOut[i] + residual2[i];
  }

  // Store result — we keep h as a "last hidden" for getLogits
  (state as any)._lastHidden = h;
  s.seqLen++;
}

function getLogits(): Float32Array {
  const s = state!;
  const h = (s as any)._lastHidden as Float32Array;
  const normed = rmsNorm(h, s.outputNorm, s.config.embDim, s.config.rmsNormEps);
  return matvec(s.output, normed, s.config.vocabSize, s.config.embDim);
}

function rmsNorm(input: Float32Array, weight: Float32Array, n: number, eps: number): Float32Array {
  let sumSq = 0;
  for (let i = 0; i < n; i++) sumSq += input[i] * input[i];
  const rms = Math.sqrt(sumSq / n + eps);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = (input[i] / rms) * weight[i];
  return out;
}

function matvec(weights: Float32Array, input: Float32Array, M: number, K: number): Float32Array {
  const out = new Float32Array(M);
  for (let row = 0; row < M; row++) {
    let sum = 0;
    const off = row * K;
    for (let k = 0; k < K; k++) sum += weights[off + k] * input[k];
    out[row] = sum;
  }
  return out;
}

function rope(data: Float32Array, headDim: number, position: number, theta: number): void {
  const nPairs = data.length / 2;
  for (let pi = 0; pi < nPairs; pi++) {
    const hp = pi % (headDim / 2);
    const dimFrac = (hp * 2) / headDim;
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
