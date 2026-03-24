// DeltaNet state update + output computation.
// One workgroup per group (head). 128 threads per group.
//
// For each group g:
//   sk[j] = sum_i(state[g,i,j] * k[g*D+i])
//   error[j] = (v[g*D+j] - decay[g]*sk[j]) * beta[g]
//   state[g,i,j] = decay[g]*state[g,i,j] + k[g*D+i]*error[j]
//   output[g*D+j] = sum_i(state[g,i,j] * q[g*D+i]) * scale
//   Per-head RMSNorm on output[g*D .. g*D+D-1]

struct Params {
  head_dim: u32,    // D = 128
  num_groups: u32,  // 16
  scale: f32,       // 1/sqrt(D)
  norm_eps: f32,    // RMSNorm epsilon
}

@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> k: array<f32>;
@group(0) @binding(2) var<storage, read> v: array<f32>;
@group(0) @binding(3) var<storage, read_write> state: array<f32>;  // [G * D * D]
@group(0) @binding(4) var<storage, read> decay: array<f32>;        // [G]
@group(0) @binding(5) var<storage, read> beta_val: array<f32>;     // [G]
@group(0) @binding(6) var<storage, read> norm_weight: array<f32>;  // [D] shared across groups
@group(0) @binding(7) var<storage, read_write> output: array<f32>; // [G * D]
@group(0) @binding(8) var<uniform> params: Params;

var<workgroup> shared_k: array<f32, 128>;
var<workgroup> shared_error: array<f32, 128>;
var<workgroup> shared_sum: f32;

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let g = wg.x;
  if (g >= params.num_groups) { return; }
  let j = lid.x;  // each thread handles one column j
  let D = params.head_dim;
  let base = g * D;
  let state_base = g * D * D;
  let d = decay[g];
  let b = beta_val[g];

  // Load k into shared memory
  shared_k[j] = k[base + j];
  workgroupBarrier();

  // 1. sk[j] = S^T * k = sum_i(state[i*D+j] * k[i])
  var sk: f32 = 0.0;
  for (var i = 0u; i < D; i++) {
    sk += state[state_base + i * D + j] * shared_k[i];
  }

  // 2. error[j] = (v[j] - decay * sk) * beta
  let err = (v[base + j] - d * sk) * b;
  shared_error[j] = err;
  workgroupBarrier();

  // 3. State update: for each row i that this column j touches
  //    state[i,j] = decay * state[i,j] + k[i] * error[j]
  // Thread j updates column j across all rows
  for (var i = 0u; i < D; i++) {
    let idx = state_base + i * D + j;
    state[idx] = d * state[idx] + shared_k[i] * shared_error[j];
  }
  workgroupBarrier();

  // 4. output[j] = S_new^T * q * scale = sum_i(state[i,j] * q[i]) * scale
  var o: f32 = 0.0;
  for (var i = 0u; i < D; i++) {
    o += state[state_base + i * D + j] * q[base + i];
  }
  output[base + j] = o * params.scale;
  workgroupBarrier();

  // 5. Per-head RMSNorm
  // Compute sum of squares (reduction across threads)
  var my_sq = output[base + j] * output[base + j];

  // Use shared memory for reduction
  // We need to reduce 128 values - use warp-style reduction
  // Store in shared_k (reuse, we're done with k)
  shared_k[j] = my_sq;
  workgroupBarrier();
  for (var s = 64u; s > 0u; s >>= 1u) {
    if (j < s) { shared_k[j] += shared_k[j + s]; }
    workgroupBarrier();
  }

  let rms = sqrt(shared_k[0] / f32(D) + params.norm_eps);
  let inv_rms = 1.0 / rms;
  output[base + j] = output[base + j] * inv_rms * norm_weight[j];
}
