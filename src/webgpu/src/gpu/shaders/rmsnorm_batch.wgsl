// Batched RMSNorm: normalize N rows independently.
// input: [N * dim], weight: [dim], output: [N * dim]
// One workgroup per row.

struct Params {
  dim: u32,
  num_rows: u32,
  eps_bits: u32,  // float as u32 bits
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;
var<workgroup> shared_sum: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let row = wg.x;
  if (row >= params.num_rows) { return; }
  let tid = lid.x;
  let dim = params.dim;
  let base = row * dim;
  let eps = bitcast<f32>(params.eps_bits);

  // Sum of squares
  var sum_sq: f32 = 0.0;
  for (var i = tid; i < dim; i += 256u) {
    let v = input[base + i];
    sum_sq += v * v;
  }
  shared_sum[tid] = sum_sq;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) {
    if (tid < s) { shared_sum[tid] += shared_sum[tid + s]; }
    workgroupBarrier();
  }

  let rms = sqrt(shared_sum[0] / f32(dim) + eps);
  let inv_rms = 1.0 / rms;

  for (var i = tid; i < dim; i += 256u) {
    output[base + i] = input[base + i] * inv_rms * weight[i];
  }
}
