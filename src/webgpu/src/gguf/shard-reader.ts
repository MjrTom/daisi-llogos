/**
 * Parser for the GGUF shard binary index format.
 * Same format as the C# GgufShardReader — enables TypeScript hosts
 * to read per-layer shard files for partial model loading.
 *
 * Shard binary layout:
 *   [4 bytes: magic "GSHD" = 0x44485347]
 *   [4 bytes: format version (1)]
 *   [4 bytes: shard type enum (0=embed, 1=output, 2=layer)]
 *   [4 bytes: layer index (-1 for embed/output)]
 *   [4 bytes: tensor count]
 *   For each tensor:
 *     [4 bytes: name length]
 *     [N bytes: name (UTF-8)]
 *     [8 bytes: data offset within shard data section]
 *     [8 bytes: data byte size]
 *   [padding to 32-byte alignment]
 *   [tensor data: contiguous raw bytes]
 */

export type ShardType = 'embed' | 'output' | 'layer';

export interface ShardTensorEntry {
  name: string;
  /** Byte offset relative to the data section start. */
  offset: number;
  /** Byte size of this tensor's data. */
  byteSize: number;
}

export interface ShardIndex {
  shardType: ShardType;
  layerIndex: number;
  tensors: Map<string, ShardTensorEntry>;
  /** Absolute byte offset where the data section begins in the shard file. */
  dataOffset: number;
}

const SHARD_MAGIC = 0x44485347; // "GSHD" little-endian
const SHARD_VERSION = 1;
const ALIGNMENT = 32;

const SHARD_TYPE_MAP: Record<number, ShardType> = {
  0: 'embed',
  1: 'output',
  2: 'layer',
};

/**
 * Parse the shard binary index from an ArrayBuffer (the full shard file or its header).
 */
export function parseShardIndex(buffer: ArrayBuffer): ShardIndex {
  const view = new DataView(buffer);
  let pos = 0;

  const magic = view.getUint32(pos, true); pos += 4;
  if (magic !== SHARD_MAGIC) {
    throw new Error(`Invalid shard magic: 0x${magic.toString(16)} (expected 0x${SHARD_MAGIC.toString(16)})`);
  }

  const version = view.getUint32(pos, true); pos += 4;
  if (version !== SHARD_VERSION) {
    throw new Error(`Unsupported shard format version: ${version}`);
  }

  const typeNum = view.getUint32(pos, true); pos += 4;
  const shardType = SHARD_TYPE_MAP[typeNum];
  if (!shardType) throw new Error(`Unknown shard type: ${typeNum}`);

  const layerIndex = view.getInt32(pos, true); pos += 4;
  const tensorCount = view.getInt32(pos, true); pos += 4;

  const decoder = new TextDecoder('utf-8');
  const tensors = new Map<string, ShardTensorEntry>();

  for (let i = 0; i < tensorCount; i++) {
    const nameLen = view.getUint32(pos, true); pos += 4;
    const nameBytes = new Uint8Array(buffer, pos, nameLen);
    const name = decoder.decode(nameBytes); pos += nameLen;

    // Read as two 32-bit values for offset and size (JS doesn't have native 64-bit int)
    const offsetLo = view.getUint32(pos, true);
    const offsetHi = view.getUint32(pos + 4, true);
    pos += 8;
    const sizeLo = view.getUint32(pos, true);
    const sizeHi = view.getUint32(pos + 4, true);
    pos += 8;

    // Combine into Number (safe for files up to ~8 PB)
    const offset = offsetLo + offsetHi * 0x100000000;
    const byteSize = sizeLo + sizeHi * 0x100000000;

    tensors.set(name, { name, offset, byteSize });
  }

  // Data section starts after alignment padding
  const remainder = pos % ALIGNMENT;
  const dataOffset = remainder === 0 ? pos : pos + (ALIGNMENT - remainder);

  return { shardType, layerIndex, tensors, dataOffset };
}

/**
 * JSON manifest describing a split GGUF model's shard files.
 */
export interface ShardManifest {
  version: number;
  modelFileName: string;
  totalLayers: number;
  header: { fileName: string; sizeBytes: number };
  embed: { fileName: string; sizeBytes: number };
  output: { fileName: string; sizeBytes: number };
  layers: Array<{ layerIndex: number; fileName: string; sizeBytes: number }>;
}

/**
 * Extract a tensor's data from a shard ArrayBuffer using the parsed index.
 */
export function extractTensorFromShard(
  shardBuffer: ArrayBuffer,
  index: ShardIndex,
  tensorName: string,
): ArrayBuffer {
  const entry = index.tensors.get(tensorName);
  if (!entry) throw new Error(`Tensor '${tensorName}' not found in shard`);

  const start = index.dataOffset + entry.offset;
  return shardBuffer.slice(start, start + entry.byteSize);
}
