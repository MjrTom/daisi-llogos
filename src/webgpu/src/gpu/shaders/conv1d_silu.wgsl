// Fused causal conv1d + SiLU activation.
// Processes all channels in parallel. Uses persistent conv buffer.
//
// For each channel c:
//   result = sum(buf[k] * weight[c*K+k]) + input[c] * weight[c*K+(K-1)]
//   shift buffer: discard oldest, add current pre-conv input
//   output[c] = silu(result)
//
// conv_buf: [(K-1) * channels] — persistent between tokens
// weights: [channels * K] — conv kernel
// data: [channels] — input/output (in-place)

struct Params {
  channels: u32,
  kernel_size: u32,
}

@group(0) @binding(0) var<storage, read_write> data: array<f32>;     // [channels] in/out
@group(0) @binding(1) var<storage, read_write> conv_buf: array<f32>; // [(K-1) * channels]
@group(0) @binding(2) var<storage, read> weights: array<f32>;        // [channels * K]
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let c = gid.x;
  if (c >= params.channels) { return; }

  let K = params.kernel_size;
  let C = params.channels;
  let buf_slots = K - 1u;

  // Save pre-conv input for buffer shift
  let pre_conv = data[c];

  // Compute conv: sum(buf[k*C+c] * w[c*K+k]) + data[c] * w[c*K+(K-1)]
  var s: f32 = 0.0;
  for (var k = 0u; k < buf_slots; k++) {
    s += conv_buf[k * C + c] * weights[c * K + k];
  }
  s += pre_conv * weights[c * K + buf_slots];

  // Shift buffer: slot[k] = slot[k+1], newest = pre_conv
  for (var k = 0u; k < buf_slots - 1u; k++) {
    conv_buf[k * C + c] = conv_buf[(k + 1u) * C + c];
  }
  if (buf_slots > 0u) {
    conv_buf[(buf_slots - 1u) * C + c] = pre_conv;
  }

  // SiLU activation + write output
  data[c] = s / (1.0 + exp(-s));
}
