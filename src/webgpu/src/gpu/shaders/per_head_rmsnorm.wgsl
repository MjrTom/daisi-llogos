// Per-head RMSNorm: normalize each head independently using shared weight.
// data: [numHeads * headDim], weight: [headDim] (shared across all heads)
// One workgroup per head, 256 threads for reduction.

struct Params {
  num_heads: u32,
  head_dim: u32,
  eps_bits: u32,  // float eps stored as u32 bits
}

@group(0) @binding(0) var<storage, read_write> data: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;
var<workgroup> shared_sum: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let h = wg.x;
  if (h >= params.num_heads) { return; }
  let tid = lid.x;
  let dim = params.head_dim;
  let base = h * dim;
  let eps = bitcast<f32>(params.eps_bits);

  // Sum of squares
  var sum_sq: f32 = 0.0;
  for (var i = tid; i < dim; i += 256u) {
    let v = data[base + i];
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

  // Normalize with weight
  for (var i = tid; i < dim; i += 256u) {
    data[base + i] = data[base + i] * inv_rms * weight[i];
  }
}
