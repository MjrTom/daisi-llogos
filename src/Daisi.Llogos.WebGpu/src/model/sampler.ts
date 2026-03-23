/**
 * Token sampler: temperature, top-k, top-p, repetition penalty.
 * Operates on CPU (logits read back from GPU via mapAsync).
 */

export interface SamplerOptions {
  temperature?: number;
  topK?: number;
  topP?: number;
  repetitionPenalty?: number;
  seed?: number;
}

const DEFAULT_OPTIONS: Required<SamplerOptions> = {
  temperature: 0.7,
  topK: 40,
  topP: 0.9,
  repetitionPenalty: 1.1,
  seed: 0,
};

export class Sampler {
  private options: Required<SamplerOptions>;
  private rng: () => number;

  constructor(options?: SamplerOptions) {
    this.options = { ...DEFAULT_OPTIONS, ...options };
    // Simple seeded RNG (xorshift32)
    if (this.options.seed > 0) {
      let state = this.options.seed;
      this.rng = () => {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        return (state >>> 0) / 0x100000000;
      };
    } else {
      this.rng = Math.random;
    }
  }

  /**
   * Sample a token ID from logits.
   * @param logits - Raw logits array [vocabSize]
   * @param previousTokens - Recent token IDs for repetition penalty
   */
  sample(logits: Float32Array, previousTokens?: number[]): number {
    const { temperature, topK, topP, repetitionPenalty } = this.options;

    // Apply repetition penalty
    if (previousTokens && previousTokens.length > 0 && repetitionPenalty !== 1.0) {
      const seen = new Set(previousTokens.slice(-64)); // Last 64 tokens
      for (const id of seen) {
        if (id < logits.length) {
          if (logits[id] > 0) {
            logits[id] /= repetitionPenalty;
          } else {
            logits[id] *= repetitionPenalty;
          }
        }
      }
    }

    // Greedy (temperature = 0)
    if (temperature === 0 || temperature < 1e-6) {
      return argmax(logits);
    }

    // Apply temperature
    for (let i = 0; i < logits.length; i++) {
      logits[i] /= temperature;
    }

    // Build candidates sorted by logit (descending)
    const indices = new Uint32Array(logits.length);
    for (let i = 0; i < indices.length; i++) indices[i] = i;
    indices.sort((a, b) => logits[b] - logits[a]);

    // Top-K filtering
    let k = Math.min(topK, logits.length);

    // Softmax over top-k candidates
    let maxLogit = logits[indices[0]];
    const probs = new Float32Array(k);
    let sum = 0;
    for (let i = 0; i < k; i++) {
      probs[i] = Math.exp(logits[indices[i]] - maxLogit);
      sum += probs[i];
    }
    for (let i = 0; i < k; i++) probs[i] /= sum;

    // Top-P (nucleus) filtering
    let cumProb = 0;
    let cutoff = k;
    for (let i = 0; i < k; i++) {
      cumProb += probs[i];
      if (cumProb >= topP) {
        cutoff = i + 1;
        break;
      }
    }

    // Renormalize after top-p cutoff
    sum = 0;
    for (let i = 0; i < cutoff; i++) sum += probs[i];
    for (let i = 0; i < cutoff; i++) probs[i] /= sum;

    // Weighted random sampling
    const r = this.rng();
    cumProb = 0;
    for (let i = 0; i < cutoff; i++) {
      cumProb += probs[i];
      if (r <= cumProb) return indices[i];
    }

    return indices[0];
  }
}

function argmax(arr: Float32Array): number {
  let maxIdx = 0;
  let maxVal = arr[0];
  for (let i = 1; i < arr.length; i++) {
    if (arr[i] > maxVal) {
      maxVal = arr[i];
      maxIdx = i;
    }
  }
  return maxIdx;
}
