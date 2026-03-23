// Fused Q4_0 dequantize + matrix-vector multiply
// Q4_0 block layout (18 bytes):
//   [0..1]   f16 scale (delta)
//   [2..17]  16 bytes = 32 x 4-bit quants packed as:
//            low nibbles of bytes 0-15  → quants 0-15
//            high nibbles of bytes 0-15 → quants 16-31
//            (matches llama.cpp dequant order)
//
// weights: [M * ceil(K/32) * 18 bytes], input: [K], output: [M]
// Each workgroup computes one output row.

struct Params {
  M: u32,
  K: u32,
}

@group(0) @binding(0) var<storage, read> weights: array<u32>;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

var<workgroup> shared_sum: array<f32, 256>;

// Read a f16 value from a byte offset within the u32 weight array
fn read_f16(byte_offset: u32) -> f32 {
  let word_idx = byte_offset / 4u;
  let word = weights[word_idx];
  let shift = (byte_offset % 4u) * 8u;
  let bits = (word >> shift) & 0xFFFFu;
  let sign = (bits >> 15u) & 1u;
  let exp = (bits >> 10u) & 0x1Fu;
  let mant = bits & 0x3FFu;
  if (exp == 0u) {
    if (mant == 0u) { return 0.0; }
    let f = f32(mant) / 1024.0 * pow(2.0, -14.0);
    if (sign == 1u) { return -f; }
    return f;
  }
  if (exp == 31u) { return 0.0; }
  let f = (1.0 + f32(mant) / 1024.0) * pow(2.0, f32(exp) - 15.0);
  if (sign == 1u) { return -f; }
  return f;
}

// Read a byte from the u32 weight array at a byte offset
fn read_byte(byte_offset: u32) -> u32 {
  let word_idx = byte_offset / 4u;
  let word = weights[word_idx];
  let shift = (byte_offset % 4u) * 8u;
  return (word >> shift) & 0xFFu;
}

// Read a Q4_0 quant value (0..31) from a block
// quant 0..15  = low nibble of qs[quant_idx]
// quant 16..31 = high nibble of qs[quant_idx - 16]
fn read_q4(block_byte_offset: u32, quant_idx: u32) -> f32 {
  let qs_offset = block_byte_offset + 2u; // skip 2-byte scale
  if (quant_idx < 16u) {
    let byte_val = read_byte(qs_offset + quant_idx);
    return f32(byte_val & 0xFu) - 8.0;
  } else {
    let byte_val = read_byte(qs_offset + quant_idx - 16u);
    return f32((byte_val >> 4u) & 0xFu) - 8.0;
  }
}

@compute @workgroup_size(256)
fn main(
  @builtin(workgroup_id) wg_id: vec3u,
  @builtin(local_invocation_id) lid: vec3u,
) {
  let row = wg_id.x;
  if (row >= params.M) { return; }

  let tid = lid.x;
  let K = params.K;
  let n_blocks = K / 32u;
  let block_bytes = 18u;
  let row_byte_offset = row * n_blocks * block_bytes;

  var sum: f32 = 0.0;
  for (var b = tid; b < n_blocks; b += 256u) {
    let block_offset = row_byte_offset + b * block_bytes;
    let scale = read_f16(block_offset);

    for (var q = 0u; q < 32u; q++) {
      let dequant = scale * read_q4(block_offset, q);
      sum += dequant * input[b * 32u + q];
    }
  }

  shared_sum[tid] = sum;
  workgroupBarrier();

  for (var stride = 128u; stride > 0u; stride >>= 1u) {
    if (tid < stride) {
      shared_sum[tid] += shared_sum[tid + stride];
    }
    workgroupBarrier();
  }

  if (tid == 0u) {
    output[row] = shared_sum[0];
  }
}
