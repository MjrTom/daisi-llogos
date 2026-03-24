// Batched KV cache write: write N K/V vectors at consecutive positions.
// k_in: [N * nKvHeads * headDim], v_in: [N * nKvHeads * headDim]
// kCache/vCache: [nKvHeads * maxSeqLen * headDim]
// Writes k_in[n] to kCache[kvh, startPos+n, :] for each KV head.

struct Params {
  n_kv_heads: u32,
  head_dim: u32,
  max_seq_len: u32,
  start_pos: u32,
  num_tokens: u32,
}

@group(0) @binding(0) var<storage, read> k_in: array<f32>;
@group(0) @binding(1) var<storage, read> v_in: array<f32>;
@group(0) @binding(2) var<storage, read_write> k_cache: array<f32>;
@group(0) @binding(3) var<storage, read_write> v_cache: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let idx = gid.x;
  let total = params.num_tokens * params.n_kv_heads * params.head_dim;
  if (idx >= total) { return; }

  let kv_size = params.n_kv_heads * params.head_dim;
  let n = idx / kv_size;
  let rem = idx % kv_size;
  let kvh = rem / params.head_dim;
  let d = rem % params.head_dim;

  let cache_idx = kvh * params.max_seq_len * params.head_dim +
                  (params.start_pos + n) * params.head_dim + d;
  let in_idx = n * kv_size + kvh * params.head_dim + d;

  k_cache[cache_idx] = k_in[in_idx];
  v_cache[cache_idx] = v_in[in_idx];
}
