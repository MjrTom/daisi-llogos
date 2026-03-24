// Compute decay and beta values for DeltaNet.
// decay[g] = exp(ssm_a[g] * softplus(alpha[g] + dt_bias[g]))
// beta_out[g] = sigmoid(beta[g])

struct Params { n: u32, }

@group(0) @binding(0) var<storage, read> alpha: array<f32>;
@group(0) @binding(1) var<storage, read> beta: array<f32>;
@group(0) @binding(2) var<storage, read> ssm_a: array<f32>;
@group(0) @binding(3) var<storage, read> dt_bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> decay_out: array<f32>;
@group(0) @binding(5) var<storage, read_write> beta_out: array<f32>;
@group(0) @binding(6) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let g = gid.x;
  if (g >= params.n) { return; }

  let softplus = log(1.0 + exp(alpha[g] + dt_bias[g]));
  decay_out[g] = exp(ssm_a[g] * softplus);
  beta_out[g] = 1.0 / (1.0 + exp(-beta[g]));
}
