// SiLU activation: output[i] = input[i] * sigmoid(input[i])
// Also called Swish. Used in FFN gate.

struct Params {
  n: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= params.n) { return; }
  let x = input[i];
  output[i] = x / (1.0 + exp(-x));
}
