// Gated attention: softmax(Q*K^T / scale) * V * sigmoid(qGate)
// One workgroup per Q head. Handles GQA (multiple Q heads per KV head).
// Uses online softmax for numerical stability.
//
// qAttn: [numHeads * keyLen], qGate: [numHeads * keyLen]
// kCache: [numKvHeads * maxSeqLen * keyLen], vCache: [numKvHeads * maxSeqLen * valLen]
// output: [numHeads * valLen]

struct Params {
  num_heads: u32,
  num_kv_heads: u32,
  key_len: u32,
  val_len: u32,
  max_seq_len: u32,
  seq_len: u32,
  scale_bits: u32,  // float scale as u32 bits
}

@group(0) @binding(0) var<storage, read> q_attn: array<f32>;
@group(0) @binding(1) var<storage, read> q_gate: array<f32>;
@group(0) @binding(2) var<storage, read> k_cache: array<f32>;
@group(0) @binding(3) var<storage, read> v_cache: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;
@group(0) @binding(5) var<uniform> params: Params;

// Shared memory for attention scores (max 4096 tokens)
var<workgroup> scores: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let h = wg.x;
  if (h >= params.num_heads) { return; }
  let tid = lid.x;
  let scale = bitcast<f32>(params.scale_bits);
  let kv_head = h / (params.num_heads / params.num_kv_heads);
  let q_off = h * params.key_len;
  let kv_k_base = kv_head * params.max_seq_len * params.key_len;
  let kv_v_base = kv_head * params.max_seq_len * params.val_len;
  let out_off = h * params.val_len;
  let seq_len = params.seq_len;

  // Process tokens in tiles of 64
  // Online softmax: track running max and sum
  var running_max: f32 = -1e30;
  var running_sum: f32 = 0.0;

  // Initialize output to zero
  for (var d = tid; d < params.val_len; d += 64u) {
    output[out_off + d] = 0.0;
  }
  workgroupBarrier();

  for (var tile_start = 0u; tile_start < seq_len; tile_start += 64u) {
    let tile_end = min(tile_start + 64u, seq_len);

    // Compute attention score for this thread's token
    var score: f32 = -1e30;
    let t = tile_start + tid;
    if (t < tile_end) {
      var dot: f32 = 0.0;
      let k_off = kv_k_base + t * params.key_len;
      for (var d = 0u; d < params.key_len; d++) {
        dot += q_attn[q_off + d] * k_cache[k_off + d];
      }
      score = dot * scale;
    }
    scores[tid] = score;
    workgroupBarrier();

    // Find tile max
    var tile_max: f32 = -1e30;
    let tile_len = tile_end - tile_start;
    for (var i = 0u; i < tile_len; i++) {
      tile_max = max(tile_max, scores[i]);
    }

    // Compute exp scores
    var tile_sum: f32 = 0.0;
    if (tid < tile_len) {
      scores[tid] = exp(scores[tid] - tile_max);
      tile_sum = scores[tid];
    }
    workgroupBarrier();

    // Sum tile scores (simple serial, fast for ≤64)
    var total_tile_sum: f32 = 0.0;
    for (var i = 0u; i < tile_len; i++) {
      total_tile_sum += scores[i];
    }

    // Online softmax correction
    let new_max = max(running_max, tile_max);
    let correction_old = exp(running_max - new_max);
    let correction_new = exp(tile_max - new_max);

    // Update output: out = out * correction_old + tile_weighted_v * correction_new
    for (var d = tid; d < params.val_len; d += 64u) {
      var tile_val: f32 = 0.0;
      for (var i = 0u; i < tile_len; i++) {
        tile_val += scores[i] * v_cache[kv_v_base + (tile_start + i) * params.val_len + d];
      }
      output[out_off + d] = output[out_off + d] * correction_old + tile_val * correction_new;
    }
    workgroupBarrier();

    running_sum = running_sum * correction_old + total_tile_sum * correction_new;
    running_max = new_max;
  }

  // Normalize and apply sigmoid gate
  let inv_sum = select(0.0, 1.0 / running_sum, running_sum > 0.0);
  for (var d = tid; d < params.val_len; d += 64u) {
    let gate_idx = min(d, params.key_len - 1u);
    let gate_val = 1.0 / (1.0 + exp(-q_gate[h * params.key_len + gate_idx]));
    output[out_off + d] = output[out_off + d] * inv_sum * gate_val;
  }
}
