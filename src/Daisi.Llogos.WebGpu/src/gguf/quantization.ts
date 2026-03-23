/**
 * GGML tensor data types — maps directly to ggml_type enum in llama.cpp.
 * Each type defines a blockSize (elements per block) and typeSize (bytes per block).
 */
export enum GgmlType {
  F32 = 0,
  F16 = 1,
  Q4_0 = 2,
  Q4_1 = 3,
  Q5_0 = 6,
  Q5_1 = 7,
  Q8_0 = 8,
  Q8_1 = 9,
  Q2_K = 10,
  Q3_K = 11,
  Q4_K = 12,
  Q5_K = 13,
  Q6_K = 14,
  Q8_K = 15,
  I8 = 24,
  I16 = 25,
  I32 = 26,
  I64 = 27,
  F64 = 28,
  BF16 = 30,
}

/** Number of elements per quantization block. */
export function blockSize(type: GgmlType): number {
  switch (type) {
    case GgmlType.F32: case GgmlType.F16: case GgmlType.BF16:
    case GgmlType.I8: case GgmlType.I16: case GgmlType.I32:
    case GgmlType.I64: case GgmlType.F64:
      return 1;
    case GgmlType.Q4_0: case GgmlType.Q4_1:
    case GgmlType.Q5_0: case GgmlType.Q5_1:
    case GgmlType.Q8_0: case GgmlType.Q8_1:
      return 32;
    case GgmlType.Q2_K: case GgmlType.Q3_K: case GgmlType.Q4_K:
    case GgmlType.Q5_K: case GgmlType.Q6_K: case GgmlType.Q8_K:
      return 256;
    default:
      throw new Error(`Unknown GGML type: ${type}`);
  }
}

/** Bytes per quantization block. */
export function typeSize(type: GgmlType): number {
  switch (type) {
    case GgmlType.F32: case GgmlType.I32: return 4;
    case GgmlType.F16: case GgmlType.BF16: case GgmlType.I16: return 2;
    case GgmlType.I8: return 1;
    case GgmlType.I64: case GgmlType.F64: return 8;
    case GgmlType.Q4_0: return 18;
    case GgmlType.Q4_1: return 20;
    case GgmlType.Q5_0: return 22;
    case GgmlType.Q5_1: return 24;
    case GgmlType.Q8_0: return 34;
    case GgmlType.Q8_1: return 36;
    case GgmlType.Q2_K: return 96;
    case GgmlType.Q3_K: return 110;
    case GgmlType.Q4_K: return 144;
    case GgmlType.Q5_K: return 176;
    case GgmlType.Q6_K: return 210;
    case GgmlType.Q8_K: return 292;
    default:
      throw new Error(`Unknown GGML type: ${type}`);
  }
}

/** Calculate total byte size for a tensor with given element count and type. */
export function tensorByteSize(type: GgmlType, elementCount: number): number {
  const bs = blockSize(type);
  const ts = typeSize(type);
  const blockCount = Math.ceil(elementCount / bs);
  return blockCount * ts;
}
