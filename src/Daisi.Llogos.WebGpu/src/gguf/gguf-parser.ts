import { GgmlType, tensorByteSize } from './quantization.js';

/**
 * GGUF magic number: "GGUF" in little-endian = 0x46554747
 */
const GGUF_MAGIC = 0x46554747;

/** GGUF metadata value types. */
const enum MetaType {
  Uint8 = 0, Int8 = 1, Uint16 = 2, Int16 = 3,
  Uint32 = 4, Int32 = 5, Float32 = 6, Bool = 7,
  String = 8, Array = 9, Uint64 = 10, Int64 = 11, Float64 = 12,
}

/** Parsed GGUF header. */
export interface GgufHeader {
  magic: number;
  version: number;
  tensorCount: number;
  metadataKvCount: number;
}

/** A single metadata key-value pair. */
export interface GgufMetadataKv {
  key: string;
  type: MetaType;
  value: unknown;
}

/** Tensor metadata from the GGUF tensor info section. */
export interface GgufTensorInfo {
  name: string;
  nDimensions: number;
  dimensions: number[];
  type: GgmlType;
  offset: number;
  elementCount: number;
  byteSize: number;
}

/** High-level model metadata extracted from GGUF. */
export interface GgufModelInfo {
  header: GgufHeader;
  architecture: string;
  blockCount: number;
  embeddingLength: number;
  headCount: number;
  headCountKv: number;
  contextLength: number;
  feedForwardLength: number;
  ropeFreqBase: number;
  rmsNormEps: number;
  vocabSize: number;
  metadata: Map<string, unknown>;
  tensors: GgufTensorInfo[];
  tensorDataOffset: number;
  alignment: number;
}

/**
 * Binary reader wrapping a DataView for GGUF parsing.
 */
class BinaryReader {
  private view: DataView;
  private offset: number;

  constructor(buffer: ArrayBuffer, offset = 0) {
    this.view = new DataView(buffer);
    this.offset = offset;
  }

  get position(): number { return this.offset; }

  readUint8(): number { const v = this.view.getUint8(this.offset); this.offset += 1; return v; }
  readInt8(): number { const v = this.view.getInt8(this.offset); this.offset += 1; return v; }
  readUint16(): number { const v = this.view.getUint16(this.offset, true); this.offset += 2; return v; }
  readInt16(): number { const v = this.view.getInt16(this.offset, true); this.offset += 2; return v; }
  readUint32(): number { const v = this.view.getUint32(this.offset, true); this.offset += 4; return v; }
  readInt32(): number { const v = this.view.getInt32(this.offset, true); this.offset += 4; return v; }
  readFloat32(): number { const v = this.view.getFloat32(this.offset, true); this.offset += 4; return v; }
  readFloat64(): number { const v = this.view.getFloat64(this.offset, true); this.offset += 8; return v; }

  readUint64(): number {
    const lo = this.view.getUint32(this.offset, true);
    const hi = this.view.getUint32(this.offset + 4, true);
    this.offset += 8;
    // Safe for values < 2^53
    return hi * 0x100000000 + lo;
  }

  readInt64(): number {
    const lo = this.view.getUint32(this.offset, true);
    const hi = this.view.getInt32(this.offset + 4, true);
    this.offset += 8;
    return hi * 0x100000000 + lo;
  }

  readString(): string {
    const length = this.readUint64();
    if (this.offset + length > this.view.byteLength) {
      throw new RangeError(
        `String read out of bounds: offset=${this.offset} length=${length} bufferSize=${this.view.byteLength}`
      );
    }
    const bytes = new Uint8Array(this.view.buffer, this.offset, length);
    this.offset += length;
    return new TextDecoder().decode(bytes);
  }

  readBytes(count: number): Uint8Array {
    const bytes = new Uint8Array(this.view.buffer, this.offset, count);
    this.offset += count;
    return bytes;
  }
}

/**
 * Read metadata value by type.
 */
function readMetadataValue(reader: BinaryReader, type: MetaType): unknown {
  switch (type) {
    case MetaType.Uint8: return reader.readUint8();
    case MetaType.Int8: return reader.readInt8();
    case MetaType.Uint16: return reader.readUint16();
    case MetaType.Int16: return reader.readInt16();
    case MetaType.Uint32: return reader.readUint32();
    case MetaType.Int32: return reader.readInt32();
    case MetaType.Float32: return reader.readFloat32();
    case MetaType.Bool: return reader.readUint8() !== 0;
    case MetaType.String: return reader.readString();
    case MetaType.Uint64: return reader.readUint64();
    case MetaType.Int64: return reader.readInt64();
    case MetaType.Float64: return reader.readFloat64();
    case MetaType.Array: return readArray(reader);
    default: throw new Error(`Unknown metadata type: ${type}`);
  }
}

function readArray(reader: BinaryReader): unknown[] {
  const elementType = reader.readUint32() as MetaType;
  const count = reader.readUint64();
  const arr: unknown[] = new Array(count);
  for (let i = 0; i < count; i++) {
    arr[i] = readMetadataValue(reader, elementType);
  }
  return arr;
}

/**
 * Parse a GGUF file header and metadata from an ArrayBuffer.
 * The buffer should contain at least the header + metadata + tensor info sections.
 */
export function parseGguf(buffer: ArrayBuffer): GgufModelInfo {
  const reader = new BinaryReader(buffer);

  // Header
  const magic = reader.readUint32();
  if (magic !== GGUF_MAGIC) {
    throw new Error(`Invalid GGUF magic: 0x${magic.toString(16)}. Expected 0x${GGUF_MAGIC.toString(16)}.`);
  }

  const version = reader.readUint32();
  if (version < 2 || version > 3) {
    throw new Error(`Unsupported GGUF version: ${version}. Only v2 and v3 are supported.`);
  }

  const tensorCount = reader.readUint64();
  const metadataKvCount = reader.readUint64();

  const header: GgufHeader = { magic, version, tensorCount, metadataKvCount };

  // Read metadata KV pairs
  const metadataMap = new Map<string, unknown>();
  for (let i = 0; i < metadataKvCount; i++) {
    const key = reader.readString();
    const type = reader.readUint32() as MetaType;
    const value = readMetadataValue(reader, type);
    metadataMap.set(key, value);
  }

  // Determine alignment
  const alignment = (metadataMap.get('general.alignment') as number) ?? 32;

  // Read tensor info
  const tensors: GgufTensorInfo[] = new Array(tensorCount);
  for (let i = 0; i < tensorCount; i++) {
    const name = reader.readString();
    const nDimensions = reader.readUint32();
    const dimensions: number[] = new Array(nDimensions);
    for (let d = 0; d < nDimensions; d++) {
      dimensions[d] = reader.readUint64();
    }
    const type = reader.readUint32() as GgmlType;
    const offset = reader.readUint64();

    let elementCount = 1;
    for (let d = 0; d < nDimensions; d++) elementCount *= dimensions[d];

    tensors[i] = {
      name, nDimensions, dimensions, type, offset, elementCount,
      byteSize: tensorByteSize(type, elementCount),
    };
  }

  // Calculate tensor data offset (aligned)
  const currentPos = reader.position;
  const remainder = currentPos % alignment;
  const tensorDataOffset = remainder === 0 ? currentPos : currentPos + (alignment - remainder);

  // Extract common model metadata
  const architecture = (metadataMap.get('general.architecture') as string) ?? 'unknown';
  const prefix = architecture;

  return {
    header,
    architecture,
    blockCount: (metadataMap.get(`${prefix}.block_count`) as number) ?? 0,
    embeddingLength: (metadataMap.get(`${prefix}.embedding_length`) as number) ?? 0,
    headCount: (metadataMap.get(`${prefix}.attention.head_count`) as number) ?? 0,
    headCountKv: (metadataMap.get(`${prefix}.attention.head_count_kv`) as number) ?? 0,
    contextLength: (metadataMap.get(`${prefix}.context_length`) as number) ?? 0,
    feedForwardLength: (metadataMap.get(`${prefix}.feed_forward_length`) as number) ?? 0,
    ropeFreqBase: (metadataMap.get(`${prefix}.rope.freq_base`) as number) ?? 10000,
    rmsNormEps: (metadataMap.get(`${prefix}.attention.layer_norm_rms_epsilon`) as number) ?? 1e-5,
    vocabSize: ((metadataMap.get('tokenizer.ggml.tokens') as unknown[]) ?? []).length,
    metadata: metadataMap,
    tensors,
    tensorDataOffset,
    alignment,
  };
}

/**
 * Fetch just enough of a GGUF file (via HTTP range request) to parse the header and metadata.
 * Returns the parsed model info without downloading tensor data.
 */
export async function fetchGgufHeader(url: string, maxBytes = 4 * 1024 * 1024): Promise<GgufModelInfo> {
  // First fetch: get the initial chunk to parse header + metadata
  const response = await fetch(url, {
    headers: { Range: `bytes=0-${maxBytes - 1}` },
  });

  if (!response.ok && response.status !== 206) {
    throw new Error(`Failed to fetch GGUF header: ${response.status} ${response.statusText}`);
  }

  const buffer = await response.arrayBuffer();

  try {
    return parseGguf(buffer);
  } catch (e) {
    // If initial chunk was too small, try a larger one
    if (maxBytes < 64 * 1024 * 1024) {
      return fetchGgufHeader(url, maxBytes * 2);
    }
    throw e;
  }
}
