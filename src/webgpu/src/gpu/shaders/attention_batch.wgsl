// Batched causal attention for prefill.
// Processes N query tokens simultaneously. Each query attends to all previous tokens + itself.
// Q: [N * numHeads * headDim], K/V written to cache during this call.
//
// For each token n and head h:
//   scores[t] = Q[n,h] · K[t,kvh] / sqrt(headDim) for t in 0..startPos+n
//   output[n,h] = softmax(scores) · V[kvh]
//
// Dispatch: [numHeads, N] workgroups, 64 threads per workgroup.

struct Params {
  num_heads: u32,
  num_kv_heads: u32,
  head_dim: u32,
  max_seq_len: u32,
  start_pos: u32,   // KV cache position of first token in batch
  num_tokens: u32,   // N
  scale_bits: u32,   // float as u32 bits
}

@group(0) @binding(0) var<storage, read> q: array<f32>;       // [N * numHeads * headDim]
@group(0) @binding(1) var<storage, read> k_cache: array<f32>; // [numKvHeads * maxSeqLen * headDim]
@group(0) @binding(2) var<storage, read> v_cache: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>; // [N * numHeads * headDim]
@group(0) @binding(4) var<uniform> params: Params;

var<workgroup> scores: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let h = wg.x;
  let n = wg.y;  // token index within batch
  if (h >= params.num_heads || n >= params.num_tokens) { return; }

  let tid = lid.x;
  let scale = bitcast<f32>(params.scale_bits);
  let kv_head = h / (params.num_heads / params.num_kv_heads);
  let head_dim = params.head_dim;
  let max_seq = params.max_seq_len;

  // This query attends to tokens 0..startPos+n (inclusive) — causal mask
  let seq_len = params.start_pos + n + 1u;

  let q_off = (n * params.num_heads + h) * head_dim;
  let kv_k_base = kv_head * max_seq * head_dim;
  let kv_v_base = kv_head * max_seq * head_dim;
  let out_off = (n * params.num_heads + h) * head_dim;

  // Initialize output
  for (var d = tid; d < head_dim; d += 64u) {
    output[out_off + d] = 0.0;
  }
  workgroupBarrier();

  var running_max: f32 = -1e30;
  var running_sum: f32 = 0.0;

  // Process in tiles of 64
  for (var tile_start = 0u; tile_start < seq_len; tile_start += 64u) {
    let tile_end = min(tile_start + 64u, seq_len);
    let tile_len = tile_end - tile_start;

    var score: f32 = -1e30;
    let t = tile_start + tid;
    if (t < tile_end) {
      var dot: f32 = 0.0;
      let k_off = kv_k_base + t * head_dim;
      for (var d = 0u; d < head_dim; d++) {
        dot += q[q_off + d] * k_cache[k_off + d];
      }
      score = dot * scale;
    }
    scores[tid] = score;
    workgroupBarrier();

    var tile_max: f32 = -1e30;
    for (var i = 0u; i < tile_len; i++) {
      tile_max = max(tile_max, scores[i]);
    }

    if (tid < tile_len) {
      scores[tid] = exp(scores[tid] - tile_max);
    }
    workgroupBarrier();

    var tile_sum: f32 = 0.0;
    for (var i = 0u; i < tile_len; i++) {
      tile_sum += scores[i];
    }

    let new_max = max(running_max, tile_max);
    let corr_old = exp(running_max - new_max);
    let corr_new = exp(tile_max - new_max);

    for (var d = tid; d < head_dim; d += 64u) {
      var tile_val: f32 = 0.0;
      for (var i = 0u; i < tile_len; i++) {
        tile_val += scores[i] * v_cache[kv_v_base + (tile_start + i) * head_dim + d];
      }
      output[out_off + d] = output[out_off + d] * corr_old + tile_val * corr_new;
    }
    workgroupBarrier();

    running_sum = running_sum * corr_old + tile_sum * corr_new;
    running_max = new_max;
  }

  // Normalize
  let inv_sum = select(0.0, 1.0 / running_sum, running_sum > 0.0);
  for (var d = tid; d < head_dim; d += 64u) {
    output[out_off + d] *= inv_sum;
  }
}
