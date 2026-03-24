// Token embedding lookup for Q8_0 quantized weights.
// Q8_0 block: [f16 scale][32 x int8] = 34 bytes per 32 elements.
// output[i] = scale * int8_value for the token's embedding row.

struct Params {
  token_id: u32,
  embedding_dim: u32,
}

@group(0) @binding(0) var<storage, read> weights: array<u32>;  // Q8_0 packed as u32
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

fn read_scale(byte_offset: u32) -> f32 {
  let w = weights[byte_offset / 4u];
  let bits = select(w & 0xFFFFu, (w >> 16u) & 0xFFFFu, (byte_offset % 4u) != 0u);
  return unpack2x16float(bits).x;
}

fn read_i8(byte_offset: u32) -> f32 {
  let v = (weights[byte_offset / 4u] >> ((byte_offset % 4u) * 8u)) & 0xFFu;
  return select(f32(v), f32(v) - 256.0, v >= 128u);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= params.embedding_dim) { return; }

  let dim = params.embedding_dim;
  let nblk = dim / 32u;
  let row_bytes = nblk * 34u;  // Q8_0: 34 bytes per block of 32 elements
  let row_start = params.token_id * row_bytes;

  let block = i / 32u;
  let idx_in_block = i % 32u;
  let block_start = row_start + block * 34u;
  let scale = read_scale(block_start);
  let val = read_i8(block_start + 2u + idx_in_block);

  output[i] = scale * val;
}
