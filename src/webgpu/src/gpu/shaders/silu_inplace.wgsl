// In-place SiLU activation: data[i] = data[i] * sigmoid(data[i])

struct Params { n: u32, }

@group(0) @binding(0) var<storage, read_write> data: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= params.n) { return; }
  let x = data[i];
  data[i] = x / (1.0 + exp(-x));
}
