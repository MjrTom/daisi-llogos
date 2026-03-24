// Batched token embedding lookup: look up N tokens at once.
// token_ids: [N], output: [N * embDim]

struct Params {
  num_tokens: u32,
  embedding_dim: u32,
}

@group(0) @binding(0) var<storage, read> weights: array<f32>;
@group(0) @binding(1) var<storage, read> token_ids: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let idx = gid.x;
  let total = params.num_tokens * params.embedding_dim;
  if (idx >= total) { return; }

  let n = idx / params.embedding_dim;
  let d = idx % params.embedding_dim;
  let token_id = token_ids[n];

  output[n * params.embedding_dim + d] = weights[token_id * params.embedding_dim + d];
}
