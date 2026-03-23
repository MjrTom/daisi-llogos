// Buffer copy as compute shader — allows staying in a single compute pass.
struct Params { n: u32, }
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x < params.n) { dst[gid.x] = src[gid.x]; }
}
