/**
 * Vocabulary frequency remapper — sorts tokens so common tokens have low IDs.
 * Enables partial vocab truncation: only compute first N logits in LM head.
 * Port of Daisi.Llogos VocabRemapper.cs.
 */

export class VocabRemapper {
  /** Old ID → New ID. */
  readonly oldToNew: Int32Array;
  /** New ID → Old ID. */
  readonly newToOld: Int32Array;
  readonly vocabSize: number;

  constructor(tokens: string[]) {
    const n = tokens.length;
    this.vocabSize = n;
    this.oldToNew = new Int32Array(n);
    this.newToOld = new Int32Array(n);

    // Score each token
    const scores: Array<{ oldId: number; score: number }> = [];
    for (let i = 0; i < n; i++) {
      scores.push({ oldId: i, score: scoreToken(tokens[i], i, n) });
    }

    // Sort descending by score — highest score = lowest new ID
    scores.sort((a, b) => b.score - a.score);

    for (let newId = 0; newId < n; newId++) {
      const oldId = scores[newId].oldId;
      this.oldToNew[oldId] = newId;
      this.newToOld[newId] = oldId;
    }
  }

  /** Remap token ID from original to remapped space. */
  remapEncode(oldId: number): number {
    return oldId >= 0 && oldId < this.vocabSize ? this.oldToNew[oldId] : oldId;
  }

  /** Remap token ID from remapped back to original space. */
  remapDecode(newId: number): number {
    return newId >= 0 && newId < this.vocabSize ? this.newToOld[newId] : newId;
  }

  /**
   * Permute rows of a weight buffer. Each row is one token's vector.
   * Returns new buffer with rows reordered so newRow[i] = oldRow[newToOld[i]].
   */
  permuteRows(data: ArrayBuffer, rowCount: number, bytesPerRow: number): ArrayBuffer {
    const src = new Uint8Array(data);
    const dst = new Uint8Array(data.byteLength);
    for (let newRow = 0; newRow < rowCount; newRow++) {
      const oldRow = this.newToOld[newRow];
      dst.set(
        src.subarray(oldRow * bytesPerRow, oldRow * bytesPerRow + bytesPerRow),
        newRow * bytesPerRow,
      );
    }
    return dst.buffer;
  }
}

/**
 * Score a token for frequency ordering. Higher = more common.
 * Heuristic — no corpus needed.
 */
function scoreToken(token: string, originalId: number, vocabSize: number): number {
  // Primary signal: original ID order. BPE training puts common tokens first.
  // Use a large multiplier so this dominates.
  let score = (vocabSize - originalId) * 4;

  // Small boost for printable content (tiebreaker only)
  if (token.length > 0 && token.length <= 10) {
    let hasPrintable = false;
    for (let i = 0; i < token.length; i++) {
      const c = token.charCodeAt(i);
      if (c >= 32 && c <= 126) hasPrintable = true; // ASCII printable
      if (c >= 0x4E00 && c <= 0x9FFF) hasPrintable = true; // CJK
    }
    if (hasPrintable) score += vocabSize; // mild boost
  }

  // Penalize special/control tokens — push to end
  if (token.startsWith('<|') || token.startsWith('<\uFF5C')) {
    score -= vocabSize * 8;
  }

  return score;
}
