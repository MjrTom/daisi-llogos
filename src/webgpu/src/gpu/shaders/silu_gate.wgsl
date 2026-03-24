// SiLU gate: output[i] = data[i] * silu(gate[i])
// where silu(x) = x / (1 + exp(-x))

struct Params { n: u32, }

@group(0) @binding(0) var<storage, read_write> data: array<f32>;
@group(0) @binding(1) var<storage, read> gate: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= params.n) { return; }
  let g = gate[i];
  data[i] = data[i] * g / (1.0 + exp(-g));
}
