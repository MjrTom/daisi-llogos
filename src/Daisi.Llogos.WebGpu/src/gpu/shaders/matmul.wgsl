struct Params { M: u32, K: u32, }
@group(0) @binding(0) var<storage, read> weights: array<f32>;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;
var<workgroup> shared_sum: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let row = wg.x;
  if (row >= params.M) { return; }
  let tid = lid.x;
  let K = params.K;
  var sum: f32 = 0.0;
  for (var k = tid; k < K; k += 256u) { sum += weights[row * K + k] * input[k]; }
  shared_sum[tid] = sum;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) { if (tid < s) { shared_sum[tid] += shared_sum[tid + s]; } workgroupBarrier(); }
  if (tid == 0u) { output[row] = shared_sum[0]; }
}
