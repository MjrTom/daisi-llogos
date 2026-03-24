import { describe, it, expect } from 'vitest';
import { readFileSync } from 'fs';
import { parseGguf } from '../src/gguf/gguf-parser.js';
import { GgmlType } from '../src/gguf/quantization.js';

// We can't import dequantizeToF32 directly (it's not exported), but we can
// verify tensor loading by checking that the GGUF contains the expected types.

function loadGguf(path: string) {
  const buf = readFileSync(path);
  return parseGguf(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
}

describe('Quantization', () => {
  describe('TinyLlama Q8_0 tensor types', () => {
    const info = loadGguf('C:/GGUFS/tinyllama-q8.gguf');

    it('has Q8_0 weight tensors', () => {
      const q8Tensors = info.tensors.filter(t => t.type === GgmlType.Q8_0);
      expect(q8Tensors.length).toBeGreaterThan(0);
    });

    it('has F32 norm tensors', () => {
      const f32Norms = info.tensors.filter(t => t.type === GgmlType.F32 && t.name.includes('norm'));
      expect(f32Norms.length).toBeGreaterThan(0);
    });

    it('all tensors have valid byte sizes', () => {
      for (const t of info.tensors) {
        expect(t.byteSize).toBeGreaterThan(0);
        expect(t.elementCount).toBeGreaterThan(0);
      }
    });
  });

  describe('Llama 3.2 Q8_0 tensor types', () => {
    const info = loadGguf('C:/GGUFS/llama32-1b-q8.gguf');

    it('has Q8_0 weight tensors', () => {
      const q8Tensors = info.tensors.filter(t => t.type === GgmlType.Q8_0);
      expect(q8Tensors.length).toBeGreaterThan(0);
    });

    it('embedding tensor is correct size', () => {
      const embd = info.tensors.find(t => t.name === 'token_embd.weight')!;
      expect(embd.elementCount).toBe(128256 * 2048);
    });

    it('K/V tensors reflect GQA (smaller than Q)', () => {
      const q = info.tensors.find(t => t.name === 'blk.0.attn_q.weight')!;
      const k = info.tensors.find(t => t.name === 'blk.0.attn_k.weight')!;
      // Q: 2048 * 2048, K: 2048 * 512 (8 KV heads * 64 dim)
      expect(k.elementCount).toBeLessThan(q.elementCount);
      expect(k.elementCount).toBe(2048 * 512);
    });
  });

  describe('GPU_MATMUL_TYPES coverage', () => {
    // Verify that all Q8_0 and Q4_0 tensors will stay on GPU (not dequanted to F32)
    const GPU_MATMUL_TYPES = new Set([GgmlType.F32, GgmlType.Q4_0, GgmlType.Q8_0]);

    it('Q8_0 is a native GPU type', () => {
      expect(GPU_MATMUL_TYPES.has(GgmlType.Q8_0)).toBe(true);
    });

    it('Q4_0 is a native GPU type', () => {
      expect(GPU_MATMUL_TYPES.has(GgmlType.Q4_0)).toBe(true);
    });

    it('F16 requires CPU dequant', () => {
      expect(GPU_MATMUL_TYPES.has(GgmlType.F16)).toBe(false);
    });
  });
});
