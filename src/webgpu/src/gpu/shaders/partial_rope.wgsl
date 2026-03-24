// Partial RoPE: apply rotary embeddings to first ropeDim positions of each head.
// Positions ropeDim..headDim-1 are left unchanged.
// data: [numHeads * headDim], cos/sin tables: [ropeDim/2] precomputed for this position.

struct Params {
  num_heads: u32,
  head_dim: u32,
  rope_dim: u32,
}

@group(0) @binding(0) var<storage, read_write> data: array<f32>;
@group(0) @binding(1) var<storage, read> cos_table: array<f32>;
@group(0) @binding(2) var<storage, read> sin_table: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let idx = gid.x;
  let half_rope = params.rope_dim / 2u;
  let total_pairs = params.num_heads * half_rope;
  if (idx >= total_pairs) { return; }

  let head = idx / half_rope;
  let pair = idx % half_rope;

  let base = head * params.head_dim;
  let i0 = base + pair * 2u;
  let i1 = i0 + 1u;

  let cos_a = cos_table[pair];
  let sin_a = sin_table[pair];

  let x = data[i0];
  let y = data[i1];

  data[i0] = x * cos_a - y * sin_a;
  data[i1] = x * sin_a + y * cos_a;
}
