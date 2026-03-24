import { describe, it, expect } from 'vitest';
import { readFileSync } from 'fs';
import { parseGguf } from '../src/gguf/gguf-parser.js';
import { tokenizerFromGguf } from '../src/tokenizer/bpe-tokenizer.js';

function loadTokenizer(path: string) {
  const buf = readFileSync(path);
  const info = parseGguf(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
  return { tok: tokenizerFromGguf(info.metadata), info };
}

describe('BPE Tokenizer', () => {
  describe('TinyLlama', () => {
    const { tok } = loadTokenizer('test/tinyllama-q8.gguf');

    it('encodes simple text', () => {
      const tokens = tok.encode('hello');
      expect(tokens.length).toBeGreaterThan(0);
    });

    it('round-trips text', () => {
      const text = 'The quick brown fox';
      const tokens = tok.encode(text);
      const decoded = tok.decode(tokens);
      expect(decoded).toBe(text);
    });

    it('has correct vocab size', () => {
      expect(tok.vocabSize).toBe(32000);
    });

    it('has BOS/EOS tokens', () => {
      expect(tok.bosTokenId).toBeGreaterThanOrEqual(0);
      expect(tok.eosTokenId).toBeGreaterThanOrEqual(0);
    });

    it('detects EOS tokens', () => {
      expect(tok.isEos(tok.eosTokenId)).toBe(true);
      expect(tok.isEos(0)).toBe(false);
    });
  });

  describe('Llama 3.2', () => {
    const { tok } = loadTokenizer('test/llama32-1b-q8.gguf');

    it('encodes special tokens correctly', () => {
      const tokens = tok.encode('<|begin_of_text|>hello');
      expect(tokens[0]).toBe(128000); // <|begin_of_text|>
    });

    it('encodes Llama 3 chat format', () => {
      const text = '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are helpful.<|eot_id|>';
      const tokens = tok.encode(text);
      expect(tokens[0]).toBe(128000); // <|begin_of_text|>
      expect(tokens[1]).toBe(128006); // <|start_header_id|>
      // "system" should be encoded after the special tokens
      expect(tokens.length).toBeGreaterThan(5);
    });

    it('detects <|eot_id|> as EOS', () => {
      expect(tok.isEos(128009)).toBe(true); // <|eot_id|>
    });

    it('has 128K vocab', () => {
      expect(tok.vocabSize).toBe(128256);
    });

    it('round-trips regular text', () => {
      const text = 'Hello, how are you?';
      const tokens = tok.encode(text);
      const decoded = tok.decode(tokens);
      expect(decoded).toBe(text);
    });
  });

  describe('Qwen 2.5', () => {
    const { tok } = loadTokenizer('test/qwen25-header.bin');

    it('has ChatML tokens', () => {
      expect(tok.getTokenId('<|im_start|>')).toBeGreaterThanOrEqual(0);
      expect(tok.getTokenId('<|im_end|>')).toBeGreaterThanOrEqual(0);
    });

    it('encodes ChatML format', () => {
      const text = '<|im_start|>user\nhello<|im_end|>';
      const tokens = tok.encode(text);
      expect(tokens[0]).toBe(tok.getTokenId('<|im_start|>'));
    });

    it('detects <|im_end|> as EOS', () => {
      const imEndId = tok.getTokenId('<|im_end|>');
      expect(tok.isEos(imEndId)).toBe(true);
    });

    it('has 151K vocab', () => {
      expect(tok.vocabSize).toBe(151936);
    });
  });
});
