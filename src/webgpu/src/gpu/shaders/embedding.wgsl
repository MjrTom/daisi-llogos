// Token embedding lookup: output[i] = weights[tokenId * embeddingDim + i]

struct Params {
  token_id: u32,
  embedding_dim: u32,
}

@group(0) @binding(0) var<storage, read> weights: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= params.embedding_dim) { return; }
  output[i] = weights[params.token_id * params.embedding_dim + i];
}
