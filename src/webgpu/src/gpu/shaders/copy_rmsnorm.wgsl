// Fused: copy input→residual AND compute RMSNorm(input, weight)→output
// Saves one dispatch per layer half (2 per layer = 44 total)
struct Params { n: u32, eps_bits: u32, }
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<storage, read_write> residual: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;
var<workgroup> shared_sum: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3u) {
  let tid = lid.x;
  let n = params.n;
  let eps = bitcast<f32>(params.eps_bits);
  // Copy input → residual AND accumulate sum of squares
  var sq: f32 = 0.0;
  for (var i = tid; i < n; i += 256u) {
    let v = input[i];
    residual[i] = v;  // copy
    sq += v * v;
  }
  shared_sum[tid] = sq;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) { if (tid < s) { shared_sum[tid] += shared_sum[tid + s]; } workgroupBarrier(); }
  let rms = sqrt(shared_sum[0] / f32(n) + eps);
  for (var i = tid; i < n; i += 256u) { output[i] = (input[i] / rms) * weight[i]; }
}
