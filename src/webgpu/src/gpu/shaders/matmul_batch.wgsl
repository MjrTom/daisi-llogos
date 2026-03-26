// Batched matrix-vector multiply: process N tokens simultaneously.
// For each token n and output row m:
//   output[n * M + m] = sum_k(weights[m * K + k] * input[n * K + k])
//
// Dispatch: [M, N] workgroups, 256 threads per workgroup for K reduction.

struct Params {
  M: u32,
  K: u32,
  N: u32,  // number of tokens (batch size)
}

@group(0) @binding(0) var<storage, read> weights: array<f32>;
@group(0) @binding(1) var<storage, read> input: array<f32>;    // [N * K]
@group(0) @binding(2) var<storage, read_write> output: array<f32>; // [N * M]
@group(0) @binding(3) var<uniform> params: Params;
var<workgroup> shared_sum: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let m = wg.x + wg.y * 65535u;  // output row (supports large M)
  let n = wg.z;                    // token index
  if (m >= params.M || n >= params.N) { return; }

  let tid = lid.x;
  let K = params.K;

  var sum: f32 = 0.0;
  for (var k = tid; k < K; k += 256u) {
    sum += weights[m * K + k] * input[n * K + k];
  }
  shared_sum[tid] = sum;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) {
    if (tid < s) { shared_sum[tid] += shared_sum[tid + s]; }
    workgroupBarrier();
  }
  if (tid == 0u) {
    output[n * params.M + m] = shared_sum[0];
  }
}
