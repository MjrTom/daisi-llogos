// Deinterleave gated Q: split [h0_attn|h0_gate|h1_attn|h1_gate|...] into separate attn and gate buffers.
// qFull: [numHeads * headDim * 2], qAttn: [numHeads * headDim], qGate: [numHeads * headDim]

struct Params {
  num_heads: u32,
  head_dim: u32,
}

@group(0) @binding(0) var<storage, read> q_full: array<f32>;
@group(0) @binding(1) var<storage, read_write> q_attn: array<f32>;
@group(0) @binding(2) var<storage, read_write> q_gate: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let idx = gid.x;
  let total = params.num_heads * params.head_dim;
  if (idx >= total) { return; }

  let head = idx / params.head_dim;
  let dim = idx % params.head_dim;
  let src_base = head * params.head_dim * 2u;

  q_attn[head * params.head_dim + dim] = q_full[src_base + dim];
  q_gate[head * params.head_dim + dim] = q_full[src_base + params.head_dim + dim];
}
