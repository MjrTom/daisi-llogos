import { describe, it, expect } from 'vitest';
import { readFileSync } from 'fs';
import { parseGguf } from '../src/gguf/gguf-parser.js';

function loadGguf(path: string) {
  const buf = readFileSync(path);
  return parseGguf(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
}

describe('GGUF Parser', () => {
  describe('TinyLlama 1.1B Q8_0', () => {
    const info = loadGguf('test/tinyllama-q8.gguf');

    it('parses architecture', () => {
      expect(info.architecture).toBe('llama');
    });

    it('parses dimensions', () => {
      expect(info.blockCount).toBe(22);
      expect(info.embeddingLength).toBe(2048);
      expect(info.headCount).toBe(32);
      expect(info.headCountKv).toBe(4);
      expect(info.feedForwardLength).toBe(5632);
    });

    it('parses context and rope', () => {
      expect(info.contextLength).toBe(2048);
      expect(info.ropeFreqBase).toBe(10000);
    });

    it('parses vocab', () => {
      expect(info.vocabSize).toBe(32000);
    });

    it('has expected tensors', () => {
      expect(info.tensors.length).toBeGreaterThan(0);
      expect(info.tensors.find(t => t.name === 'token_embd.weight')).toBeTruthy();
      expect(info.tensors.find(t => t.name === 'blk.0.attn_q.weight')).toBeTruthy();
      expect(info.tensors.find(t => t.name === 'output_norm.weight')).toBeTruthy();
    });

    it('has no attention biases', () => {
      expect(info.tensors.filter(t => t.name.includes('bias')).length).toBe(0);
    });
  });

  describe('Llama 3.2 1B Q8_0', () => {
    const info = loadGguf('test/llama32-1b-q8.gguf');

    it('parses architecture', () => {
      expect(info.architecture).toBe('llama');
    });

    it('parses GQA dimensions', () => {
      expect(info.blockCount).toBe(16);
      expect(info.embeddingLength).toBe(2048);
      expect(info.headCount).toBe(32);
      expect(info.headCountKv).toBe(8);
      expect(info.feedForwardLength).toBe(8192);
    });

    it('has high rope theta for Llama 3', () => {
      expect(info.ropeFreqBase).toBe(500000);
    });

    it('has large vocab', () => {
      expect(info.vocabSize).toBe(128256);
    });

    it('has tied weights (no output.weight)', () => {
      expect(info.tensors.find(t => t.name === 'output.weight')).toBeUndefined();
      expect(info.tensors.find(t => t.name === 'token_embd.weight')).toBeTruthy();
    });

    it('has no attention biases', () => {
      expect(info.tensors.filter(t => t.name.includes('bias')).length).toBe(0);
    });
  });

  describe('Qwen 2.5 0.5B Q8_0 (partial header)', () => {
    const info = loadGguf('test/qwen25-header.bin');

    it('parses qwen2 architecture', () => {
      expect(info.architecture).toBe('qwen2');
    });

    it('parses dimensions', () => {
      expect(info.blockCount).toBe(24);
      expect(info.embeddingLength).toBe(896);
      expect(info.headCount).toBe(14);
      expect(info.headCountKv).toBe(2);
      expect(info.feedForwardLength).toBe(4864);
    });

    it('has very high rope theta', () => {
      expect(info.ropeFreqBase).toBe(1000000);
    });

    it('has large vocab', () => {
      expect(info.vocabSize).toBe(151936);
    });

    it('has attention biases', () => {
      const biases = info.tensors.filter(t => t.name.includes('bias'));
      expect(biases.length).toBeGreaterThan(0);
      expect(info.tensors.find(t => t.name === 'blk.0.attn_q.bias')).toBeTruthy();
      expect(info.tensors.find(t => t.name === 'blk.0.attn_k.bias')).toBeTruthy();
      expect(info.tensors.find(t => t.name === 'blk.0.attn_v.bias')).toBeTruthy();
    });
  });
});
