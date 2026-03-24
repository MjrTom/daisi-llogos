// Repeat-tile: expand data from numKHeads groups to numVHeads groups.
// src: [numKHeads * headDim], dst: [numVHeads * headDim]
// factor = numVHeads / numKHeads
// dst[rep * numKHeads * headDim + g * headDim + d] = src[g * headDim + d]

struct Params {
  num_k_heads: u32,
  head_dim: u32,
  factor: u32,
}

@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let idx = gid.x;
  let total = params.num_k_heads * params.head_dim * params.factor;
  if (idx >= total) { return; }

  let k_total = params.num_k_heads * params.head_dim;
  let src_idx = idx % k_total;
  dst[idx] = src[src_idx];
}
