// Multi-head attention — 1 workgroup per head, 64 threads per workgroup.
// Each thread handles one dimension of head_dim for the weighted V sum.
// Scores computed collaboratively, softmax in shared memory.

struct Params {
  num_heads: u32,
  num_kv_heads: u32,
  head_dim: u32,
  seq_len: u32,
  max_seq_len: u32,
  scale_bits: u32,
}

@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> k_cache: array<f32>;
@group(0) @binding(2) var<storage, read> v_cache: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

var<workgroup> sh_scores: array<f32, 2176>; // max seq_len (2048) + 128 for reduction temps
var<workgroup> sh_max: f32;
var<workgroup> sh_sum: f32;

@compute @workgroup_size(64)
fn main(
  @builtin(workgroup_id) wg: vec3u,
  @builtin(local_invocation_id) lid: vec3u,
) {
  let head = wg.x;
  if (head >= params.num_heads) { return; }
  let tid = lid.x;
  let hd = params.head_dim;
  let sl = params.seq_len;
  let scale = bitcast<f32>(params.scale_bits);
  let kvh = head / (params.num_heads / params.num_kv_heads);
  let qoff = head * hd;
  let kvs = params.max_seq_len * hd;

  // Step 1: Each thread computes scores for a subset of positions
  // Thread tid handles positions tid, tid+64, tid+128, ...
  var local_max: f32 = -1e30;
  for (var pos = tid; pos < sl; pos += 64u) {
    var dot: f32 = 0.0;
    for (var d = 0u; d < hd; d++) {
      dot += q[qoff + d] * k_cache[kvh * kvs + pos * hd + d];
    }
    let score = dot * scale;
    sh_scores[pos] = score;
    local_max = max(local_max, score);
  }

  // Reduce max across threads
  sh_scores[sl + tid] = local_max; // reuse space after scores
  workgroupBarrier();
  // Manual 64-thread reduction for max
  if (tid < 32u) { sh_scores[sl + tid] = max(sh_scores[sl + tid], sh_scores[sl + tid + 32u]); }
  workgroupBarrier();
  if (tid < 16u) { sh_scores[sl + tid] = max(sh_scores[sl + tid], sh_scores[sl + tid + 16u]); }
  workgroupBarrier();
  if (tid < 8u) { sh_scores[sl + tid] = max(sh_scores[sl + tid], sh_scores[sl + tid + 8u]); }
  workgroupBarrier();
  if (tid < 4u) { sh_scores[sl + tid] = max(sh_scores[sl + tid], sh_scores[sl + tid + 4u]); }
  workgroupBarrier();
  if (tid < 2u) { sh_scores[sl + tid] = max(sh_scores[sl + tid], sh_scores[sl + tid + 2u]); }
  workgroupBarrier();
  if (tid == 0u) { sh_max = max(sh_scores[sl], sh_scores[sl + 1u]); }
  workgroupBarrier();

  // Step 2: exp(score - max) and sum
  var local_sum: f32 = 0.0;
  for (var pos = tid; pos < sl; pos += 64u) {
    let e = exp(sh_scores[pos] - sh_max);
    sh_scores[pos] = e;
    local_sum += e;
  }
  sh_scores[sl + tid] = local_sum;
  workgroupBarrier();
  if (tid < 32u) { sh_scores[sl + tid] += sh_scores[sl + tid + 32u]; }
  workgroupBarrier();
  if (tid < 16u) { sh_scores[sl + tid] += sh_scores[sl + tid + 16u]; }
  workgroupBarrier();
  if (tid < 8u) { sh_scores[sl + tid] += sh_scores[sl + tid + 8u]; }
  workgroupBarrier();
  if (tid < 4u) { sh_scores[sl + tid] += sh_scores[sl + tid + 4u]; }
  workgroupBarrier();
  if (tid < 2u) { sh_scores[sl + tid] += sh_scores[sl + tid + 2u]; }
  workgroupBarrier();
  if (tid == 0u) { sh_sum = sh_scores[sl] + sh_scores[sl + 1u]; }
  workgroupBarrier();

  // Normalize scores
  for (var pos = tid; pos < sl; pos += 64u) {
    sh_scores[pos] /= sh_sum;
  }
  workgroupBarrier();

  // Step 3: Weighted V — each thread handles one dimension of output
  // tid 0..63 → dim 0..63 (head_dim is typically 64)
  if (tid < hd) {
    var acc: f32 = 0.0;
    for (var pos = 0u; pos < sl; pos++) {
      acc += sh_scores[pos] * v_cache[kvh * kvs + pos * hd + tid];
    }
    output[qoff + tid] = acc;
  }
}
