// Batched RoPE: apply rotary embeddings to N tokens at different positions.
// data: [N * numHeads * headDim], each token has its own position.
// positions: [N] — position index for each token.
// Precomputed cos/sin per dimension pair, looked up per position.

struct Params {
  num_heads: u32,
  head_dim: u32,
  num_tokens: u32,
  theta_bits: u32,  // float theta as u32 bits
}

@group(0) @binding(0) var<storage, read_write> data: array<f32>;
@group(0) @binding(1) var<storage, read> positions: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let idx = gid.x;
  let half_dim = params.head_dim / 2u;
  let total_pairs = params.num_tokens * params.num_heads * half_dim;
  if (idx >= total_pairs) { return; }

  let theta = bitcast<f32>(params.theta_bits);
  let pairs_per_token = params.num_heads * half_dim;
  let n = idx / pairs_per_token;
  let rem = idx % pairs_per_token;
  let head = rem / half_dim;
  let pair = rem % half_dim;
  let pos = positions[n];

  let dim_frac = f32(pair * 2u) / f32(params.head_dim);
  let freq = 1.0 / pow(theta, dim_frac);
  let angle = f32(pos) * freq;
  let cos_a = cos(angle);
  let sin_a = sin(angle);

  let base = (n * params.num_heads + head) * params.head_dim;
  let i0 = base + pair * 2u;
  let i1 = i0 + 1u;

  let x = data[i0];
  let y = data[i1];
  data[i0] = x * cos_a - y * sin_a;
  data[i1] = x * sin_a + y * cos_a;
}
