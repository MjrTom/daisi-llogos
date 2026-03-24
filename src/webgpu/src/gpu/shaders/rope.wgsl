// Rotary Position Embedding (RoPE) using precomputed cos/sin table.
// Avoids GPU pow/cos/sin precision issues.

struct Params {
  n_elements: u32,
  head_dim: u32,
}

@group(0) @binding(0) var<storage, read_write> data: array<f32>;
@group(0) @binding(1) var<storage, read> cos_table: array<f32>; // [headDim/2] precomputed cos values
@group(0) @binding(2) var<storage, read> sin_table: array<f32>; // [headDim/2] precomputed sin values
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let pair_idx = gid.x;
  if (pair_idx >= params.n_elements / 2u) { return; }

  let head_pair = pair_idx % (params.head_dim / 2u);
  let cos_a = cos_table[head_pair];
  let sin_a = sin_table[head_pair];

  let idx0 = pair_idx * 2u;
  let idx1 = idx0 + 1u;

  let x = data[idx0];
  let y = data[idx1];

  data[idx0] = x * cos_a - y * sin_a;
  data[idx1] = x * sin_a + y * cos_a;
}
