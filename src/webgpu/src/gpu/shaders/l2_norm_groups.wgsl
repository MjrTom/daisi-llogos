// L2 normalize each group independently.
// data: [numGroups * groupDim], each group of groupDim elements is normalized.
// One workgroup per group, 256 threads for reduction.

struct Params {
  num_groups: u32,
  group_dim: u32,
}

@group(0) @binding(0) var<storage, read_write> data: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;
var<workgroup> shared_sum: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let g = wg.x;
  if (g >= params.num_groups) { return; }
  let tid = lid.x;
  let base = g * params.group_dim;
  let dim = params.group_dim;

  // Sum of squares
  var sum_sq: f32 = 0.0;
  for (var i = tid; i < dim; i += 256u) {
    let v = data[base + i];
    sum_sq += v * v;
  }
  shared_sum[tid] = sum_sq;
  workgroupBarrier();

  // Reduction
  for (var s = 128u; s > 0u; s >>= 1u) {
    if (tid < s) { shared_sum[tid] += shared_sum[tid + s]; }
    workgroupBarrier();
  }

  let norm = sqrt(shared_sum[0]);
  let inv_norm = select(1.0, 1.0 / norm, norm > 0.0);

  // Normalize
  for (var i = tid; i < dim; i += 256u) {
    data[base + i] *= inv_norm;
  }
}
