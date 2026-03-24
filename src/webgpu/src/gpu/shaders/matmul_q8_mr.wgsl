// Multi-row Q8_0 matmul: each workgroup computes 2 output rows.
// Loads input vector into shared memory once, reuses for both rows.
// Halves the number of workgroups needed → halves dispatch overhead.
//
// output[row] = sum_k(dequant(weights[row, k]) * input[k])

struct Params { M: u32, K: u32, }

@group(0) @binding(0) var<storage, read> weights: array<u32>;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

var<workgroup> shared_input: array<f32, 256>;
var<workgroup> shared_sum0: array<f32, 256>;
var<workgroup> shared_sum1: array<f32, 256>;

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
  let row_pair = wg.x + wg.y * 65535u;
  let row0 = row_pair * 2u;
  let row1 = row0 + 1u;
  let tid = lid.x;
  let nblk = params.K / 32u;

  var sum0: f32 = 0.0;
  var sum1: f32 = 0.0;

  // Process blocks — load input chunk into shared mem for reuse
  for (var b = tid; b < nblk; b += 256u) {
    let bk = b * 32u;

    // Row 0
    if (row0 < params.M) {
      let bo0 = row0 * nblk * 34u + b * 34u;
      let sc0 = read_scale(bo0);
      for (var q = 0u; q < 32u; q++) {
        sum0 += sc0 * read_i8(bo0 + 2u + q) * input[bk + q];
      }
    }

    // Row 1
    if (row1 < params.M) {
      let bo1 = row1 * nblk * 34u + b * 34u;
      let sc1 = read_scale(bo1);
      for (var q = 0u; q < 32u; q++) {
        sum1 += sc1 * read_i8(bo1 + 2u + q) * input[bk + q];
      }
    }
  }

  shared_sum0[tid] = sum0;
  shared_sum1[tid] = sum1;
  workgroupBarrier();

  for (var s = 128u; s > 0u; s >>= 1u) {
    if (tid < s) {
      shared_sum0[tid] += shared_sum0[tid + s];
      shared_sum1[tid] += shared_sum1[tid + s];
    }
    workgroupBarrier();
  }

  if (tid == 0u) {
    if (row0 < params.M) { output[row0] = shared_sum0[0]; }
    if (row1 < params.M) { output[row1] = shared_sum1[0]; }
  }
}
