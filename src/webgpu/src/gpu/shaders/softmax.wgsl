// Softmax: output[i] = exp(input[i] - max) / sum(exp(input - max))
// Two-pass: 1) find max, 2) exp, sum, normalize
// Used for logits → probabilities in sampling.

struct Params {
  n: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

var<workgroup> shared_data: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3u) {
  let tid = lid.x;
  let n = params.n;

  // Pass 1: find max
  var local_max: f32 = -1e30;
  for (var i = tid; i < n; i += 256u) {
    local_max = max(local_max, input[i]);
  }
  shared_data[tid] = local_max;
  workgroupBarrier();

  for (var stride = 128u; stride > 0u; stride >>= 1u) {
    if (tid < stride) {
      shared_data[tid] = max(shared_data[tid], shared_data[tid + stride]);
    }
    workgroupBarrier();
  }
  let max_val = shared_data[0];
  workgroupBarrier();

  // Pass 2: exp and sum
  var local_sum: f32 = 0.0;
  for (var i = tid; i < n; i += 256u) {
    let v = exp(input[i] - max_val);
    output[i] = v;
    local_sum += v;
  }
  shared_data[tid] = local_sum;
  workgroupBarrier();

  for (var stride = 128u; stride > 0u; stride >>= 1u) {
    if (tid < stride) {
      shared_data[tid] += shared_data[tid + stride];
    }
    workgroupBarrier();
  }
  let total_sum = shared_data[0];
  workgroupBarrier();

  // Pass 3: normalize
  let inv_sum = 1.0 / total_sum;
  for (var i = tid; i < n; i += 256u) {
    output[i] *= inv_sum;
  }
}
