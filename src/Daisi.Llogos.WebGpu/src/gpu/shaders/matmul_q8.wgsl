struct Params { M: u32, K: u32, }
@group(0) @binding(0) var<storage, read> weights: array<u32>;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;
var<workgroup> shared_sum: array<f32, 256>;

fn read_scale(bo: u32) -> f32 {
  let w = weights[bo / 4u];
  let bits = select(w & 0xFFFFu, (w >> 16u) & 0xFFFFu, (bo % 4u) != 0u);
  return unpack2x16float(bits).x;
}
fn read_i8(bo: u32) -> f32 {
  let v = (weights[bo / 4u] >> ((bo % 4u) * 8u)) & 0xFFu;
  return select(f32(v), f32(v) - 256.0, v >= 128u);
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let row = wg.x + wg.y * 65535u;
  if (row >= params.M) { return; }
  let tid = lid.x;
  let nblk = params.K / 32u;
  let roff = row * nblk * 34u;
  var sum: f32 = 0.0;
  for (var b = tid; b < nblk; b += 256u) {
    let bo = roff + b * 34u;
    let sc = read_scale(bo);
    let bk = b * 32u;
    for (var q = 0u; q < 32u; q++) { sum += sc * read_i8(bo + 2u + q) * input[bk + q]; }
  }
  shared_sum[tid] = sum;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) { if (tid < s) { shared_sum[tid] += shared_sum[tid + s]; } workgroupBarrier(); }
  if (tid == 0u) { output[row] = shared_sum[0]; }
}
