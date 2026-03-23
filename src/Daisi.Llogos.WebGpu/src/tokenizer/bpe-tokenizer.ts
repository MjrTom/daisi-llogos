/**
 * BPE tokenizer — ported from Daisi.Llogos.Tokenizer.BpeTokenizer.cs
 * Supports both GPT-2 byte encoding and direct UTF-8 (Qwen/Llama 3).
 */

// GPT-2 pre-tokenization regex
const PRE_TOKENIZE_RE = /'(?:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+/gu;

export class BpeTokenizer {
  private tokens: string[];
  private tokenToId: Map<string, number>;
  private mergeRank: Map<string, number>;
  private useByteEncoding: boolean;
  private specialTokens: string[]; // sorted longest-first for greedy matching
  readonly bosTokenId: number;
  readonly eosTokenId: number;
  readonly padTokenId: number;

  constructor(
    tokens: string[],
    merges: string[],
    bosTokenId: number,
    eosTokenId: number,
    padTokenId: number,
    useByteEncoding: boolean,
  ) {
    this.tokens = tokens;
    this.bosTokenId = bosTokenId;
    this.eosTokenId = eosTokenId;
    this.padTokenId = padTokenId;
    this.useByteEncoding = useByteEncoding;

    this.tokenToId = new Map();
    for (let i = 0; i < tokens.length; i++) {
      if (!this.tokenToId.has(tokens[i])) {
        this.tokenToId.set(tokens[i], i);
      }
    }

    this.mergeRank = new Map();
    for (let i = 0; i < merges.length; i++) {
      this.mergeRank.set(merges[i], i);
    }

    // Collect special tokens (tokens with < > or similar patterns)
    this.specialTokens = tokens
      .filter(t => t.startsWith('<') && t.endsWith('>') && t.length > 2)
      .sort((a, b) => b.length - a.length); // longest first for greedy match
  }

  get vocabSize(): number { return this.tokens.length; }

  /** Get token ID for a string, or -1 if not found. */
  getTokenId(token: string): number {
    return this.tokenToId.get(token) ?? -1;
  }

  /** Encode text to token IDs, handling special tokens. */
  encode(text: string): number[] {
    if (!text) return [];

    // Split text around special tokens, then BPE-encode the non-special parts
    const segments = this.splitOnSpecialTokens(text);
    const result: number[] = [];

    for (const seg of segments) {
      if (seg.isSpecial) {
        const id = this.tokenToId.get(seg.text);
        if (id !== undefined) result.push(id);
        // Also try with fullwidth pipes (Qwen style)
        else {
          const fwToken = seg.text.replace(/\|/g, '\uFF5C');
          const fwId = this.tokenToId.get(fwToken);
          if (fwId !== undefined) result.push(fwId);
        }
      } else {
        result.push(...this.encodeBpe(seg.text));
      }
    }

    return result;
  }

  /** BPE-encode a text segment (no special tokens). */
  private encodeBpe(text: string): number[] {
    if (!text) return [];
    const result: number[] = [];
    const matches = text.matchAll(PRE_TOKENIZE_RE);

    for (const match of matches) {
      const chunk = match[0];
      const symbols = this.useByteEncoding
        ? byteEncodeChunk(chunk)
        : this.directEncodeChunk(chunk);
      if (symbols.length === 0) continue;

      this.applyMerges(symbols);

      for (const symbol of symbols) {
        const id = this.tokenToId.get(symbol);
        if (id !== undefined) result.push(id);
      }
    }

    return result;
  }

  /** Split text into segments of special tokens and regular text. */
  private splitOnSpecialTokens(text: string): Array<{ text: string; isSpecial: boolean }> {
    const segments: Array<{ text: string; isSpecial: boolean }> = [];
    let remaining = text;

    while (remaining.length > 0) {
      // Find the earliest special token match
      let bestIdx = remaining.length;
      let bestToken = '';

      for (const st of this.specialTokens) {
        const idx = remaining.indexOf(st);
        if (idx >= 0 && idx < bestIdx) {
          bestIdx = idx;
          bestToken = st;
        }
      }

      if (bestToken) {
        // Add text before the special token
        if (bestIdx > 0) {
          segments.push({ text: remaining.slice(0, bestIdx), isSpecial: false });
        }
        // Add the special token
        segments.push({ text: bestToken, isSpecial: true });
        remaining = remaining.slice(bestIdx + bestToken.length);
      } else {
        // No more special tokens — add remaining text
        segments.push({ text: remaining, isSpecial: false });
        break;
      }
    }

    return segments;
  }

  /** Decode token IDs back to text. */
  decode(tokenIds: number[]): string {
    const parts: string[] = [];
    for (const id of tokenIds) {
      if (id === this.bosTokenId || id === this.eosTokenId || id === this.padTokenId) continue;

      let token = this.tokens[id];
      if (!token) continue;

      // Handle byte fallback tokens like <0x0A>
      if (token.length === 6 && token.startsWith('<0x') && token.endsWith('>')) {
        const byte = parseInt(token.slice(3, 5), 16);
        if (!isNaN(byte)) {
          parts.push(String.fromCharCode(byte));
          continue;
        }
      }

      // Normalize fullwidth pipes (Qwen uses ｜ in special tokens)
      if (token.includes('\uFF5C')) {
        token = token.replaceAll('\uFF5C', '|');
      }

      parts.push(token);
    }

    let text = parts.join('');
    if (this.useByteEncoding) {
      return byteDecodeString(text);
    }
    // SentencePiece space marker
    return text.replaceAll('\u2581', ' ');
  }

  /** Check if a token ID is an end-of-sequence token. */
  isEos(tokenId: number): boolean {
    if (tokenId === this.eosTokenId) return true;
    // Some models have multiple EOS tokens
    const token = this.tokens[tokenId];
    return token === '<|endoftext|>' || token === '<|im_end|>'
      || token === '<｜end▁of▁text｜>' || token === '<｜im_end｜>';
  }

  // Direct encoding for SentencePiece/Llama style vocabs
  private directEncodeChunk(chunk: string): string[] {
    const symbols: string[] = [];
    const encoder = new TextEncoder();
    const bytes = encoder.encode(chunk);
    let i = 0;
    while (i < bytes.length) {
      const charLen = utf8CharLength(bytes[i]);
      if (i + charLen <= bytes.length) {
        const ch = new TextDecoder().decode(bytes.slice(i, i + charLen));
        if (this.tokenToId.has(ch)) {
          symbols.push(ch);
          i += charLen;
          continue;
        }
      }
      symbols.push(`<0x${bytes[i].toString(16).toUpperCase().padStart(2, '0')}>`);
      i++;
    }
    return symbols;
  }

  // BPE merge algorithm
  private applyMerges(symbols: string[]): void {
    while (symbols.length > 1) {
      let bestRank = Infinity;
      let bestIdx = -1;

      for (let i = 0; i < symbols.length - 1; i++) {
        const key = `${symbols[i]} ${symbols[i + 1]}`;
        const rank = this.mergeRank.get(key);
        if (rank !== undefined && rank < bestRank) {
          bestRank = rank;
          bestIdx = i;
        }
      }

      if (bestIdx < 0) break;
      symbols[bestIdx] = symbols[bestIdx] + symbols[bestIdx + 1];
      symbols.splice(bestIdx + 1, 1);
    }
  }
}

// ── GPT-2 byte encoding ─────────────────────────────────────────────

const BYTE_TO_UNICODE = buildByteToUnicode();
const UNICODE_TO_BYTE = buildUnicodeToByte();

function buildByteToUnicode(): string[] {
  const table = new Array<string>(256);
  let n = 256;
  for (let i = 0; i < 256; i++) {
    if ((i >= 0x21 && i <= 0x7E) || (i >= 0xA1 && i <= 0xAC) || (i >= 0xAE && i <= 0xFF)) {
      table[i] = String.fromCharCode(i);
    } else {
      table[i] = String.fromCharCode(n);
      n++;
    }
  }
  return table;
}

function buildUnicodeToByte(): Map<string, number> {
  const map = new Map<string, number>();
  for (let i = 0; i < 256; i++) {
    map.set(BYTE_TO_UNICODE[i], i);
  }
  return map;
}

function byteEncodeChunk(chunk: string): string[] {
  const encoder = new TextEncoder();
  const bytes = encoder.encode(chunk);
  return Array.from(bytes, b => BYTE_TO_UNICODE[b]);
}

function byteDecodeString(text: string): string {
  const bytes: number[] = [];
  for (const ch of text) {
    const b = UNICODE_TO_BYTE.get(ch);
    if (b !== undefined) {
      bytes.push(b);
    } else {
      const encoder = new TextEncoder();
      bytes.push(...encoder.encode(ch));
    }
  }
  return new TextDecoder().decode(new Uint8Array(bytes));
}

function utf8CharLength(firstByte: number): number {
  if (firstByte < 0x80) return 1;
  if (firstByte < 0xC0) return 1;
  if (firstByte < 0xE0) return 2;
  if (firstByte < 0xF0) return 3;
  return 4;
}

/**
 * Build a BpeTokenizer from GGUF metadata.
 */
export function tokenizerFromGguf(metadata: Map<string, unknown>): BpeTokenizer {
  const tokens = metadata.get('tokenizer.ggml.tokens') as string[];
  const merges = metadata.get('tokenizer.ggml.merges') as string[];
  if (!tokens || !merges) throw new Error('Missing tokenizer metadata in GGUF');

  const bosTokenId = (metadata.get('tokenizer.ggml.bos_token_id') as number) ?? -1;
  const eosTokenId = (metadata.get('tokenizer.ggml.eos_token_id') as number) ?? -1;
  const padTokenId = (metadata.get('tokenizer.ggml.padding_token_id') as number) ?? -1;
  const model = metadata.get('tokenizer.ggml.model') as string | undefined;
  const useByteEncoding = model === 'gpt2';

  return new BpeTokenizer(tokens, merges, bosTokenId, eosTokenId, padTokenId, useByteEncoding);
}
