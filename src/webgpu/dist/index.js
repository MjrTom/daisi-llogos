// src/gpu/device.ts
function isWebGpuAvailable() {
  return typeof navigator !== "undefined" && "gpu" in navigator;
}
async function initGpu() {
  if (!isWebGpuAvailable()) {
    throw new Error("WebGPU is not available in this browser.");
  }
  const adapter = await navigator.gpu.requestAdapter({
    powerPreference: "high-performance"
  });
  if (!adapter) {
    throw new Error("No WebGPU adapter found. Your GPU may not support WebGPU.");
  }
  const adapterInfo = adapter.info;
  const limits = adapter.limits;
  const supportsF16 = adapter.features.has("shader-f16");
  const supportsTimestampQuery = adapter.features.has("timestamp-query");
  const requiredFeatures = [];
  if (supportsTimestampQuery) requiredFeatures.push("timestamp-query");
  if (supportsF16) requiredFeatures.push("shader-f16");
  const device = await adapter.requestDevice({
    requiredFeatures,
    requiredLimits: {
      maxBufferSize: limits.maxBufferSize,
      maxStorageBufferBindingSize: limits.maxStorageBufferBindingSize,
      maxComputeWorkgroupSizeX: limits.maxComputeWorkgroupSizeX,
      maxComputeInvocationsPerWorkgroup: limits.maxComputeInvocationsPerWorkgroup
    }
  });
  device.lost.then((info) => {
    console.error(`WebGPU device lost: ${info.message} (reason: ${info.reason})`);
  });
  return {
    adapter,
    device,
    capabilities: {
      adapterInfo,
      maxBufferSize: device.limits.maxBufferSize,
      maxStorageBufferBindingSize: device.limits.maxStorageBufferBindingSize,
      maxComputeWorkgroupSizeX: device.limits.maxComputeWorkgroupSizeX,
      maxComputeInvocationsPerWorkgroup: device.limits.maxComputeInvocationsPerWorkgroup,
      supportsF16,
      supportsTimestampQuery
    }
  };
}

// src/gpu/shader-cache.ts
var ShaderCache = class {
  device;
  moduleCache = /* @__PURE__ */ new Map();
  pipelineCache = /* @__PURE__ */ new Map();
  constructor(device) {
    this.device = device;
  }
  /**
   * Get or compile a shader module from WGSL source.
   */
  getModule(source, label) {
    let module = this.moduleCache.get(source);
    if (!module) {
      module = this.device.createShaderModule({ code: source, label });
      this.moduleCache.set(source, module);
    }
    return module;
  }
  /**
   * Get or create a compute pipeline from config.
   */
  getPipeline(config) {
    const key = `${config.shader}::${config.entryPoint ?? "main"}::${JSON.stringify(config.bindGroupLayout)}`;
    let cached = this.pipelineCache.get(key);
    if (!cached) {
      const module = this.getModule(config.shader, config.label);
      const bindGroupLayout = this.device.createBindGroupLayout({
        label: config.label,
        entries: config.bindGroupLayout
      });
      const pipelineLayout = this.device.createPipelineLayout({
        label: config.label,
        bindGroupLayouts: [bindGroupLayout]
      });
      const pipeline = this.device.createComputePipeline({
        label: config.label,
        layout: pipelineLayout,
        compute: { module, entryPoint: config.entryPoint ?? "main" }
      });
      cached = { pipeline, bindGroupLayout };
      this.pipelineCache.set(key, cached);
    }
    return cached;
  }
  /**
   * Create a bind group from a cached pipeline's layout.
   */
  createBindGroup(cached, entries, label) {
    return this.device.createBindGroup({
      label,
      layout: cached.bindGroupLayout,
      entries
    });
  }
};

// src/gpu/buffer-pool.ts
var BufferPool = class {
  device;
  buffers = /* @__PURE__ */ new Map();
  totalAllocated = 0;
  constructor(device) {
    this.device = device;
  }
  /** Total bytes allocated on GPU. */
  get vramUsage() {
    return this.totalAllocated;
  }
  /**
   * Create or retrieve a named storage buffer.
   */
  createBuffer(label, size, usage) {
    const existing = this.buffers.get(label);
    if (existing && existing.size >= size) return existing.buffer;
    if (existing) {
      existing.buffer.destroy();
      this.totalAllocated -= existing.size;
    }
    const alignedSize = Math.ceil(size / 4) * 4;
    const buffer = this.device.createBuffer({
      label,
      size: alignedSize,
      usage: usage ?? GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
    });
    this.buffers.set(label, { buffer, size: alignedSize, label });
    this.totalAllocated += alignedSize;
    return buffer;
  }
  /**
   * Create a storage buffer initialized with data.
   */
  createBufferWithData(label, data, usage) {
    const buffer = this.createBuffer(label, data.byteLength, usage);
    this.device.queue.writeBuffer(buffer, 0, data);
    return buffer;
  }
  /**
   * Create a buffer for reading data back to CPU.
   */
  createReadbackBuffer(label, size) {
    const alignedSize = Math.ceil(size / 4) * 4;
    return this.device.createBuffer({
      label: `${label}_readback`,
      size: alignedSize,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
    });
  }
  /**
   * Get an existing buffer by label.
   */
  get(label) {
    return this.buffers.get(label)?.buffer;
  }
  /**
   * Destroy a named buffer and free its VRAM.
   */
  destroy(label) {
    const info = this.buffers.get(label);
    if (info) {
      info.buffer.destroy();
      this.totalAllocated -= info.size;
      this.buffers.delete(label);
    }
  }
  /**
   * Destroy all buffers.
   */
  destroyAll() {
    for (const info of this.buffers.values()) {
      info.buffer.destroy();
    }
    this.buffers.clear();
    this.totalAllocated = 0;
  }
};

// src/gguf/quantization.ts
var GgmlType = /* @__PURE__ */ ((GgmlType3) => {
  GgmlType3[GgmlType3["F32"] = 0] = "F32";
  GgmlType3[GgmlType3["F16"] = 1] = "F16";
  GgmlType3[GgmlType3["Q4_0"] = 2] = "Q4_0";
  GgmlType3[GgmlType3["Q4_1"] = 3] = "Q4_1";
  GgmlType3[GgmlType3["Q5_0"] = 6] = "Q5_0";
  GgmlType3[GgmlType3["Q5_1"] = 7] = "Q5_1";
  GgmlType3[GgmlType3["Q8_0"] = 8] = "Q8_0";
  GgmlType3[GgmlType3["Q8_1"] = 9] = "Q8_1";
  GgmlType3[GgmlType3["Q2_K"] = 10] = "Q2_K";
  GgmlType3[GgmlType3["Q3_K"] = 11] = "Q3_K";
  GgmlType3[GgmlType3["Q4_K"] = 12] = "Q4_K";
  GgmlType3[GgmlType3["Q5_K"] = 13] = "Q5_K";
  GgmlType3[GgmlType3["Q6_K"] = 14] = "Q6_K";
  GgmlType3[GgmlType3["Q8_K"] = 15] = "Q8_K";
  GgmlType3[GgmlType3["I8"] = 24] = "I8";
  GgmlType3[GgmlType3["I16"] = 25] = "I16";
  GgmlType3[GgmlType3["I32"] = 26] = "I32";
  GgmlType3[GgmlType3["I64"] = 27] = "I64";
  GgmlType3[GgmlType3["F64"] = 28] = "F64";
  GgmlType3[GgmlType3["BF16"] = 30] = "BF16";
  return GgmlType3;
})(GgmlType || {});
function blockSize(type) {
  switch (type) {
    case 0 /* F32 */:
    case 1 /* F16 */:
    case 30 /* BF16 */:
    case 24 /* I8 */:
    case 25 /* I16 */:
    case 26 /* I32 */:
    case 27 /* I64 */:
    case 28 /* F64 */:
      return 1;
    case 2 /* Q4_0 */:
    case 3 /* Q4_1 */:
    case 6 /* Q5_0 */:
    case 7 /* Q5_1 */:
    case 8 /* Q8_0 */:
    case 9 /* Q8_1 */:
      return 32;
    case 10 /* Q2_K */:
    case 11 /* Q3_K */:
    case 12 /* Q4_K */:
    case 13 /* Q5_K */:
    case 14 /* Q6_K */:
    case 15 /* Q8_K */:
      return 256;
    default:
      throw new Error(`Unknown GGML type: ${type}`);
  }
}
function typeSize(type) {
  switch (type) {
    case 0 /* F32 */:
    case 26 /* I32 */:
      return 4;
    case 1 /* F16 */:
    case 30 /* BF16 */:
    case 25 /* I16 */:
      return 2;
    case 24 /* I8 */:
      return 1;
    case 27 /* I64 */:
    case 28 /* F64 */:
      return 8;
    case 2 /* Q4_0 */:
      return 18;
    case 3 /* Q4_1 */:
      return 20;
    case 6 /* Q5_0 */:
      return 22;
    case 7 /* Q5_1 */:
      return 24;
    case 8 /* Q8_0 */:
      return 34;
    case 9 /* Q8_1 */:
      return 36;
    case 10 /* Q2_K */:
      return 96;
    case 11 /* Q3_K */:
      return 110;
    case 12 /* Q4_K */:
      return 144;
    case 13 /* Q5_K */:
      return 176;
    case 14 /* Q6_K */:
      return 210;
    case 15 /* Q8_K */:
      return 292;
    default:
      throw new Error(`Unknown GGML type: ${type}`);
  }
}
function tensorByteSize(type, elementCount) {
  const bs = blockSize(type);
  const ts = typeSize(type);
  const blockCount = Math.ceil(elementCount / bs);
  return blockCount * ts;
}

// src/gpu/shaders/embedding.wgsl
var embedding_default = "// Token embedding lookup: output[i] = weights[tokenId * embeddingDim + i]\r\n\r\nstruct Params {\r\n  token_id: u32,\r\n  embedding_dim: u32,\r\n}\r\n\r\n@group(0) @binding(0) var<storage, read> weights: array<f32>;\r\n@group(0) @binding(1) var<storage, read_write> output: array<f32>;\r\n@group(0) @binding(2) var<uniform> params: Params;\r\n\r\n@compute @workgroup_size(256)\r\nfn main(@builtin(global_invocation_id) gid: vec3u) {\r\n  let i = gid.x;\r\n  if (i >= params.embedding_dim) { return; }\r\n  output[i] = weights[params.token_id * params.embedding_dim + i];\r\n}\r\n";

// src/gpu/shaders/rmsnorm.wgsl
var rmsnorm_default = "struct Params { n: u32, eps_bits: u32, }\r\n@group(0) @binding(0) var<storage, read> input: array<f32>;\r\n@group(0) @binding(1) var<storage, read> weight: array<f32>;\r\n@group(0) @binding(2) var<storage, read_write> output: array<f32>;\r\n@group(0) @binding(3) var<uniform> params: Params;\r\nvar<workgroup> shared_sum: array<f32, 256>;\r\n\r\n@compute @workgroup_size(256)\r\nfn main(@builtin(local_invocation_id) lid: vec3u) {\r\n  let tid = lid.x;\r\n  let n = params.n;\r\n  let eps = bitcast<f32>(params.eps_bits);\r\n  var sq: f32 = 0.0;\r\n  for (var i = tid; i < n; i += 256u) { let v = input[i]; sq += v * v; }\r\n  shared_sum[tid] = sq;\r\n  workgroupBarrier();\r\n  for (var s = 128u; s > 0u; s >>= 1u) { if (tid < s) { shared_sum[tid] += shared_sum[tid + s]; } workgroupBarrier(); }\r\n  let rms = sqrt(shared_sum[0] / f32(n) + eps);\r\n  for (var i = tid; i < n; i += 256u) { output[i] = (input[i] / rms) * weight[i]; }\r\n}\r\n";

// src/gpu/shaders/rope.wgsl
var rope_default = "// Rotary Position Embedding (RoPE) using precomputed cos/sin table.\r\n// Avoids GPU pow/cos/sin precision issues.\r\n\r\nstruct Params {\r\n  n_elements: u32,\r\n  head_dim: u32,\r\n}\r\n\r\n@group(0) @binding(0) var<storage, read_write> data: array<f32>;\r\n@group(0) @binding(1) var<storage, read> cos_table: array<f32>; // [headDim/2] precomputed cos values\r\n@group(0) @binding(2) var<storage, read> sin_table: array<f32>; // [headDim/2] precomputed sin values\r\n@group(0) @binding(3) var<uniform> params: Params;\r\n\r\n@compute @workgroup_size(1)\r\nfn main(@builtin(global_invocation_id) gid: vec3u) {\r\n  let pair_idx = gid.x;\r\n  if (pair_idx >= params.n_elements / 2u) { return; }\r\n\r\n  let head_pair = pair_idx % (params.head_dim / 2u);\r\n  let cos_a = cos_table[head_pair];\r\n  let sin_a = sin_table[head_pair];\r\n\r\n  let idx0 = pair_idx * 2u;\r\n  let idx1 = idx0 + 1u;\r\n\r\n  let x = data[idx0];\r\n  let y = data[idx1];\r\n\r\n  data[idx0] = x * cos_a - y * sin_a;\r\n  data[idx1] = x * sin_a + y * cos_a;\r\n}\r\n";

// src/gpu/shaders/matmul.wgsl
var matmul_default = "struct Params { M: u32, K: u32, }\n@group(0) @binding(0) var<storage, read> weights: array<f32>;\n@group(0) @binding(1) var<storage, read> input: array<f32>;\n@group(0) @binding(2) var<storage, read_write> output: array<f32>;\n@group(0) @binding(3) var<uniform> params: Params;\nvar<workgroup> shared_sum: array<f32, 256>;\n\n@compute @workgroup_size(256)\nfn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) lid: vec3u) {\n  let row = wg.x + wg.y * 65535u;\n  if (row >= params.M) { return; }\n  let tid = lid.x;\n  let K = params.K;\n  var sum: f32 = 0.0;\n  for (var k = tid; k < K; k += 256u) { sum += weights[row * K + k] * input[k]; }\n  shared_sum[tid] = sum;\n  workgroupBarrier();\n  for (var s = 128u; s > 0u; s >>= 1u) { if (tid < s) { shared_sum[tid] += shared_sum[tid + s]; } workgroupBarrier(); }\n  if (tid == 0u) { output[row] = shared_sum[0]; }\n}\n";

// src/gpu/shaders/matmul_q4.wgsl
var matmul_q4_default = "// Fused Q4_0 dequantize + matrix-vector multiply\r\n// Q4_0 block layout (18 bytes):\r\n//   [0..1]   f16 scale (delta)\r\n//   [2..17]  16 bytes = 32 x 4-bit quants packed as:\r\n//            low nibbles of bytes 0-15  \u2192 quants 0-15\r\n//            high nibbles of bytes 0-15 \u2192 quants 16-31\r\n//            (matches llama.cpp dequant order)\r\n//\r\n// weights: [M * ceil(K/32) * 18 bytes], input: [K], output: [M]\r\n// Each workgroup computes one output row.\r\n\r\nstruct Params {\r\n  M: u32,\r\n  K: u32,\r\n}\r\n\r\n@group(0) @binding(0) var<storage, read> weights: array<u32>;\r\n@group(0) @binding(1) var<storage, read> input: array<f32>;\r\n@group(0) @binding(2) var<storage, read_write> output: array<f32>;\r\n@group(0) @binding(3) var<uniform> params: Params;\r\n\r\nvar<workgroup> shared_sum: array<f32, 256>;\r\n\r\n// Read a f16 value from a byte offset within the u32 weight array\r\nfn read_f16(byte_offset: u32) -> f32 {\r\n  let word_idx = byte_offset / 4u;\r\n  let word = weights[word_idx];\r\n  let shift = (byte_offset % 4u) * 8u;\r\n  let bits = (word >> shift) & 0xFFFFu;\r\n  let sign = (bits >> 15u) & 1u;\r\n  let exp = (bits >> 10u) & 0x1Fu;\r\n  let mant = bits & 0x3FFu;\r\n  if (exp == 0u) {\r\n    if (mant == 0u) { return 0.0; }\r\n    let f = f32(mant) / 1024.0 * pow(2.0, -14.0);\r\n    if (sign == 1u) { return -f; }\r\n    return f;\r\n  }\r\n  if (exp == 31u) { return 0.0; }\r\n  let f = (1.0 + f32(mant) / 1024.0) * pow(2.0, f32(exp) - 15.0);\r\n  if (sign == 1u) { return -f; }\r\n  return f;\r\n}\r\n\r\n// Read a byte from the u32 weight array at a byte offset\r\nfn read_byte(byte_offset: u32) -> u32 {\r\n  let word_idx = byte_offset / 4u;\r\n  let word = weights[word_idx];\r\n  let shift = (byte_offset % 4u) * 8u;\r\n  return (word >> shift) & 0xFFu;\r\n}\r\n\r\n// Read a Q4_0 quant value (0..31) from a block\r\n// quant 0..15  = low nibble of qs[quant_idx]\r\n// quant 16..31 = high nibble of qs[quant_idx - 16]\r\nfn read_q4(block_byte_offset: u32, quant_idx: u32) -> f32 {\r\n  let qs_offset = block_byte_offset + 2u; // skip 2-byte scale\r\n  if (quant_idx < 16u) {\r\n    let byte_val = read_byte(qs_offset + quant_idx);\r\n    return f32(byte_val & 0xFu) - 8.0;\r\n  } else {\r\n    let byte_val = read_byte(qs_offset + quant_idx - 16u);\r\n    return f32((byte_val >> 4u) & 0xFu) - 8.0;\r\n  }\r\n}\r\n\r\n@compute @workgroup_size(256)\r\nfn main(\r\n  @builtin(workgroup_id) wg_id: vec3u,\r\n  @builtin(local_invocation_id) lid: vec3u,\r\n) {\r\n  let row = wg_id.x + wg_id.y * 65535u;\r\n  if (row >= params.M) { return; }\r\n\r\n  let tid = lid.x;\r\n  let K = params.K;\r\n  let n_blocks = K / 32u;\r\n  let block_bytes = 18u;\r\n  let row_byte_offset = row * n_blocks * block_bytes;\r\n\r\n  var sum: f32 = 0.0;\r\n  for (var b = tid; b < n_blocks; b += 256u) {\r\n    let block_offset = row_byte_offset + b * block_bytes;\r\n    let scale = read_f16(block_offset);\r\n\r\n    for (var q = 0u; q < 32u; q++) {\r\n      let dequant = scale * read_q4(block_offset, q);\r\n      sum += dequant * input[b * 32u + q];\r\n    }\r\n  }\r\n\r\n  shared_sum[tid] = sum;\r\n  workgroupBarrier();\r\n\r\n  for (var stride = 128u; stride > 0u; stride >>= 1u) {\r\n    if (tid < stride) {\r\n      shared_sum[tid] += shared_sum[tid + stride];\r\n    }\r\n    workgroupBarrier();\r\n  }\r\n\r\n  if (tid == 0u) {\r\n    output[row] = shared_sum[0];\r\n  }\r\n}\r\n";

// src/gpu/shaders/matmul_q8_mr.wgsl
var matmul_q8_mr_default = "// Multi-row Q8_0 matmul: each workgroup computes 2 output rows.\n// Loads input vector into shared memory once, reuses for both rows.\n// Halves the number of workgroups needed \u2192 halves dispatch overhead.\n//\n// output[row] = sum_k(dequant(weights[row, k]) * input[k])\n\nstruct Params { M: u32, K: u32, }\n\n@group(0) @binding(0) var<storage, read> weights: array<u32>;\n@group(0) @binding(1) var<storage, read> input: array<f32>;\n@group(0) @binding(2) var<storage, read_write> output: array<f32>;\n@group(0) @binding(3) var<uniform> params: Params;\n\nvar<workgroup> shared_input: array<f32, 256>;\nvar<workgroup> shared_sum0: array<f32, 256>;\nvar<workgroup> shared_sum1: array<f32, 256>;\n\nfn read_scale(bo: u32) -> f32 {\n  let w = weights[bo / 4u];\n  let bits = select(w & 0xFFFFu, (w >> 16u) & 0xFFFFu, (bo % 4u) != 0u);\n  return unpack2x16float(bits).x;\n}\nfn read_i8(bo: u32) -> f32 {\n  let v = (weights[bo / 4u] >> ((bo % 4u) * 8u)) & 0xFFu;\n  return select(f32(v), f32(v) - 256.0, v >= 128u);\n}\n\n@compute @workgroup_size(256)\nfn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) lid: vec3u) {\n  let row_pair = wg.x + wg.y * 65535u;\n  let row0 = row_pair * 2u;\n  let row1 = row0 + 1u;\n  let tid = lid.x;\n  let nblk = params.K / 32u;\n\n  var sum0: f32 = 0.0;\n  var sum1: f32 = 0.0;\n\n  // Process blocks \u2014 load input chunk into shared mem for reuse\n  for (var b = tid; b < nblk; b += 256u) {\n    let bk = b * 32u;\n\n    // Row 0\n    if (row0 < params.M) {\n      let bo0 = row0 * nblk * 34u + b * 34u;\n      let sc0 = read_scale(bo0);\n      for (var q = 0u; q < 32u; q++) {\n        sum0 += sc0 * read_i8(bo0 + 2u + q) * input[bk + q];\n      }\n    }\n\n    // Row 1\n    if (row1 < params.M) {\n      let bo1 = row1 * nblk * 34u + b * 34u;\n      let sc1 = read_scale(bo1);\n      for (var q = 0u; q < 32u; q++) {\n        sum1 += sc1 * read_i8(bo1 + 2u + q) * input[bk + q];\n      }\n    }\n  }\n\n  shared_sum0[tid] = sum0;\n  shared_sum1[tid] = sum1;\n  workgroupBarrier();\n\n  for (var s = 128u; s > 0u; s >>= 1u) {\n    if (tid < s) {\n      shared_sum0[tid] += shared_sum0[tid + s];\n      shared_sum1[tid] += shared_sum1[tid + s];\n    }\n    workgroupBarrier();\n  }\n\n  if (tid == 0u) {\n    if (row0 < params.M) { output[row0] = shared_sum0[0]; }\n    if (row1 < params.M) { output[row1] = shared_sum1[0]; }\n  }\n}\n";

// src/gpu/shaders/attention.wgsl
var attention_default = "// Multi-head attention \u2014 1 workgroup per head, 64 threads per workgroup.\r\n// Each thread handles one dimension of head_dim for the weighted V sum.\r\n// Scores computed collaboratively, softmax in shared memory.\r\n\r\nstruct Params {\r\n  num_heads: u32,\r\n  num_kv_heads: u32,\r\n  head_dim: u32,\r\n  seq_len: u32,\r\n  max_seq_len: u32,\r\n  scale_bits: u32,\r\n}\r\n\r\n@group(0) @binding(0) var<storage, read> q: array<f32>;\r\n@group(0) @binding(1) var<storage, read> k_cache: array<f32>;\r\n@group(0) @binding(2) var<storage, read> v_cache: array<f32>;\r\n@group(0) @binding(3) var<storage, read_write> output: array<f32>;\r\n@group(0) @binding(4) var<uniform> params: Params;\r\n\r\nvar<workgroup> sh_scores: array<f32, 2176>; // max seq_len (2048) + 128 for reduction temps\r\nvar<workgroup> sh_max: f32;\r\nvar<workgroup> sh_sum: f32;\r\n\r\n@compute @workgroup_size(64)\r\nfn main(\r\n  @builtin(workgroup_id) wg: vec3u,\r\n  @builtin(local_invocation_id) lid: vec3u,\r\n) {\r\n  let head = wg.x;\r\n  if (head >= params.num_heads) { return; }\r\n  let tid = lid.x;\r\n  let hd = params.head_dim;\r\n  let sl = params.seq_len;\r\n  let scale = bitcast<f32>(params.scale_bits);\r\n  let kvh = head / (params.num_heads / params.num_kv_heads);\r\n  let qoff = head * hd;\r\n  let kvs = params.max_seq_len * hd;\r\n\r\n  // Step 1: Each thread computes scores for a subset of positions\r\n  // Thread tid handles positions tid, tid+64, tid+128, ...\r\n  var local_max: f32 = -1e30;\r\n  for (var pos = tid; pos < sl; pos += 64u) {\r\n    var dot: f32 = 0.0;\r\n    for (var d = 0u; d < hd; d++) {\r\n      dot += q[qoff + d] * k_cache[kvh * kvs + pos * hd + d];\r\n    }\r\n    let score = dot * scale;\r\n    sh_scores[pos] = score;\r\n    local_max = max(local_max, score);\r\n  }\r\n\r\n  // Reduce max across threads\r\n  sh_scores[sl + tid] = local_max; // reuse space after scores\r\n  workgroupBarrier();\r\n  // Manual 64-thread reduction for max\r\n  if (tid < 32u) { sh_scores[sl + tid] = max(sh_scores[sl + tid], sh_scores[sl + tid + 32u]); }\r\n  workgroupBarrier();\r\n  if (tid < 16u) { sh_scores[sl + tid] = max(sh_scores[sl + tid], sh_scores[sl + tid + 16u]); }\r\n  workgroupBarrier();\r\n  if (tid < 8u) { sh_scores[sl + tid] = max(sh_scores[sl + tid], sh_scores[sl + tid + 8u]); }\r\n  workgroupBarrier();\r\n  if (tid < 4u) { sh_scores[sl + tid] = max(sh_scores[sl + tid], sh_scores[sl + tid + 4u]); }\r\n  workgroupBarrier();\r\n  if (tid < 2u) { sh_scores[sl + tid] = max(sh_scores[sl + tid], sh_scores[sl + tid + 2u]); }\r\n  workgroupBarrier();\r\n  if (tid == 0u) { sh_max = max(sh_scores[sl], sh_scores[sl + 1u]); }\r\n  workgroupBarrier();\r\n\r\n  // Step 2: exp(score - max) and sum\r\n  var local_sum: f32 = 0.0;\r\n  for (var pos = tid; pos < sl; pos += 64u) {\r\n    let e = exp(sh_scores[pos] - sh_max);\r\n    sh_scores[pos] = e;\r\n    local_sum += e;\r\n  }\r\n  sh_scores[sl + tid] = local_sum;\r\n  workgroupBarrier();\r\n  if (tid < 32u) { sh_scores[sl + tid] += sh_scores[sl + tid + 32u]; }\r\n  workgroupBarrier();\r\n  if (tid < 16u) { sh_scores[sl + tid] += sh_scores[sl + tid + 16u]; }\r\n  workgroupBarrier();\r\n  if (tid < 8u) { sh_scores[sl + tid] += sh_scores[sl + tid + 8u]; }\r\n  workgroupBarrier();\r\n  if (tid < 4u) { sh_scores[sl + tid] += sh_scores[sl + tid + 4u]; }\r\n  workgroupBarrier();\r\n  if (tid < 2u) { sh_scores[sl + tid] += sh_scores[sl + tid + 2u]; }\r\n  workgroupBarrier();\r\n  if (tid == 0u) { sh_sum = sh_scores[sl] + sh_scores[sl + 1u]; }\r\n  workgroupBarrier();\r\n\r\n  // Normalize scores\r\n  for (var pos = tid; pos < sl; pos += 64u) {\r\n    sh_scores[pos] /= sh_sum;\r\n  }\r\n  workgroupBarrier();\r\n\r\n  // Step 3: Weighted V \u2014 each thread handles one dimension of output\r\n  // tid 0..63 \u2192 dim 0..63 (head_dim is typically 64)\r\n  if (tid < hd) {\r\n    var acc: f32 = 0.0;\r\n    for (var pos = 0u; pos < sl; pos++) {\r\n      acc += sh_scores[pos] * v_cache[kvh * kvs + pos * hd + tid];\r\n    }\r\n    output[qoff + tid] = acc;\r\n  }\r\n}\r\n";

// src/gpu/shaders/softmax.wgsl
var softmax_default = "// Softmax: output[i] = exp(input[i] - max) / sum(exp(input - max))\r\n// Two-pass: 1) find max, 2) exp, sum, normalize\r\n// Used for logits \u2192 probabilities in sampling.\r\n\r\nstruct Params {\r\n  n: u32,\r\n}\r\n\r\n@group(0) @binding(0) var<storage, read> input: array<f32>;\r\n@group(0) @binding(1) var<storage, read_write> output: array<f32>;\r\n@group(0) @binding(2) var<uniform> params: Params;\r\n\r\nvar<workgroup> shared_data: array<f32, 256>;\r\n\r\n@compute @workgroup_size(256)\r\nfn main(@builtin(local_invocation_id) lid: vec3u) {\r\n  let tid = lid.x;\r\n  let n = params.n;\r\n\r\n  // Pass 1: find max\r\n  var local_max: f32 = -1e30;\r\n  for (var i = tid; i < n; i += 256u) {\r\n    local_max = max(local_max, input[i]);\r\n  }\r\n  shared_data[tid] = local_max;\r\n  workgroupBarrier();\r\n\r\n  for (var stride = 128u; stride > 0u; stride >>= 1u) {\r\n    if (tid < stride) {\r\n      shared_data[tid] = max(shared_data[tid], shared_data[tid + stride]);\r\n    }\r\n    workgroupBarrier();\r\n  }\r\n  let max_val = shared_data[0];\r\n  workgroupBarrier();\r\n\r\n  // Pass 2: exp and sum\r\n  var local_sum: f32 = 0.0;\r\n  for (var i = tid; i < n; i += 256u) {\r\n    let v = exp(input[i] - max_val);\r\n    output[i] = v;\r\n    local_sum += v;\r\n  }\r\n  shared_data[tid] = local_sum;\r\n  workgroupBarrier();\r\n\r\n  for (var stride = 128u; stride > 0u; stride >>= 1u) {\r\n    if (tid < stride) {\r\n      shared_data[tid] += shared_data[tid + stride];\r\n    }\r\n    workgroupBarrier();\r\n  }\r\n  let total_sum = shared_data[0];\r\n  workgroupBarrier();\r\n\r\n  // Pass 3: normalize\r\n  let inv_sum = 1.0 / total_sum;\r\n  for (var i = tid; i < n; i += 256u) {\r\n    output[i] *= inv_sum;\r\n  }\r\n}\r\n";

// src/gpu/shaders/silu.wgsl
var silu_default = "// SiLU activation: output[i] = input[i] * sigmoid(input[i])\r\n// Also called Swish. Used in FFN gate.\r\n\r\nstruct Params {\r\n  n: u32,\r\n}\r\n\r\n@group(0) @binding(0) var<storage, read> input: array<f32>;\r\n@group(0) @binding(1) var<storage, read_write> output: array<f32>;\r\n@group(0) @binding(2) var<uniform> params: Params;\r\n\r\n@compute @workgroup_size(256)\r\nfn main(@builtin(global_invocation_id) gid: vec3u) {\r\n  let i = gid.x;\r\n  if (i >= params.n) { return; }\r\n  let x = input[i];\r\n  output[i] = x / (1.0 + exp(-x));\r\n}\r\n";

// src/gpu/shaders/silu_mul.wgsl
var silu_mul_default = "// Fused SiLU-gate multiply: output[i] = silu(gate[i]) * up[i]\r\n// Combines two FFN operations into one kernel dispatch.\r\n\r\nstruct Params {\r\n  n: u32,\r\n}\r\n\r\n@group(0) @binding(0) var<storage, read> gate: array<f32>;\r\n@group(0) @binding(1) var<storage, read> up: array<f32>;\r\n@group(0) @binding(2) var<storage, read_write> output: array<f32>;\r\n@group(0) @binding(3) var<uniform> params: Params;\r\n\r\n@compute @workgroup_size(256)\r\nfn main(@builtin(global_invocation_id) gid: vec3u) {\r\n  let i = gid.x;\r\n  if (i >= params.n) { return; }\r\n  let x = gate[i];\r\n  let silu_x = x / (1.0 + exp(-x));\r\n  output[i] = silu_x * up[i];\r\n}\r\n";

// src/gpu/shaders/copy_rmsnorm.wgsl
var copy_rmsnorm_default = "// Fused: copy input\u2192residual AND compute RMSNorm(input, weight)\u2192output\r\n// Saves one dispatch per layer half (2 per layer = 44 total)\r\nstruct Params { n: u32, eps_bits: u32, }\r\n@group(0) @binding(0) var<storage, read> input: array<f32>;\r\n@group(0) @binding(1) var<storage, read> weight: array<f32>;\r\n@group(0) @binding(2) var<storage, read_write> output: array<f32>;\r\n@group(0) @binding(3) var<storage, read_write> residual: array<f32>;\r\n@group(0) @binding(4) var<uniform> params: Params;\r\nvar<workgroup> shared_sum: array<f32, 256>;\r\n\r\n@compute @workgroup_size(256)\r\nfn main(@builtin(local_invocation_id) lid: vec3u) {\r\n  let tid = lid.x;\r\n  let n = params.n;\r\n  let eps = bitcast<f32>(params.eps_bits);\r\n  // Copy input \u2192 residual AND accumulate sum of squares\r\n  var sq: f32 = 0.0;\r\n  for (var i = tid; i < n; i += 256u) {\r\n    let v = input[i];\r\n    residual[i] = v;  // copy\r\n    sq += v * v;\r\n  }\r\n  shared_sum[tid] = sq;\r\n  workgroupBarrier();\r\n  for (var s = 128u; s > 0u; s >>= 1u) { if (tid < s) { shared_sum[tid] += shared_sum[tid + s]; } workgroupBarrier(); }\r\n  let rms = sqrt(shared_sum[0] / f32(n) + eps);\r\n  for (var i = tid; i < n; i += 256u) { output[i] = (input[i] / rms) * weight[i]; }\r\n}\r\n";

// src/gpu/shaders/add.wgsl
var add_default = "// Element-wise add: output[i] = a[i] + b[i]\r\n// Used for residual connections.\r\n\r\nstruct Params {\r\n  n: u32,\r\n}\r\n\r\n@group(0) @binding(0) var<storage, read> a: array<f32>;\r\n@group(0) @binding(1) var<storage, read> b: array<f32>;\r\n@group(0) @binding(2) var<storage, read_write> output: array<f32>;\r\n@group(0) @binding(3) var<uniform> params: Params;\r\n\r\n@compute @workgroup_size(256)\r\nfn main(@builtin(global_invocation_id) gid: vec3u) {\r\n  let i = gid.x;\r\n  if (i >= params.n) { return; }\r\n  output[i] = a[i] + b[i];\r\n}\r\n";

// src/gpu/shaders/add_bias.wgsl
var add_bias_default = "// In-place bias add: data[i] += bias[i]\n\nstruct Params {\n  n: u32,\n}\n\n@group(0) @binding(0) var<storage, read_write> data: array<f32>;\n@group(0) @binding(1) var<storage, read> bias: array<f32>;\n@group(0) @binding(2) var<uniform> params: Params;\n\n@compute @workgroup_size(256)\nfn main(@builtin(global_invocation_id) gid: vec3u) {\n  let i = gid.x;\n  if (i >= params.n) { return; }\n  data[i] += bias[i];\n}\n";

// src/gpu/shaders/embedding_q8.wgsl
var embedding_q8_default = "// Token embedding lookup for Q8_0 quantized weights.\n// Q8_0 block: [f16 scale][32 x int8] = 34 bytes per 32 elements.\n// output[i] = scale * int8_value for the token's embedding row.\n\nstruct Params {\n  token_id: u32,\n  embedding_dim: u32,\n}\n\n@group(0) @binding(0) var<storage, read> weights: array<u32>;  // Q8_0 packed as u32\n@group(0) @binding(1) var<storage, read_write> output: array<f32>;\n@group(0) @binding(2) var<uniform> params: Params;\n\nfn read_scale(byte_offset: u32) -> f32 {\n  let w = weights[byte_offset / 4u];\n  let bits = select(w & 0xFFFFu, (w >> 16u) & 0xFFFFu, (byte_offset % 4u) != 0u);\n  return unpack2x16float(bits).x;\n}\n\nfn read_i8(byte_offset: u32) -> f32 {\n  let v = (weights[byte_offset / 4u] >> ((byte_offset % 4u) * 8u)) & 0xFFu;\n  return select(f32(v), f32(v) - 256.0, v >= 128u);\n}\n\n@compute @workgroup_size(256)\nfn main(@builtin(global_invocation_id) gid: vec3u) {\n  let i = gid.x;\n  if (i >= params.embedding_dim) { return; }\n\n  let dim = params.embedding_dim;\n  let nblk = dim / 32u;\n  let row_bytes = nblk * 34u;  // Q8_0: 34 bytes per block of 32 elements\n  let row_start = params.token_id * row_bytes;\n\n  let block = i / 32u;\n  let idx_in_block = i % 32u;\n  let block_start = row_start + block * 34u;\n  let scale = read_scale(block_start);\n  let val = read_i8(block_start + 2u + idx_in_block);\n\n  output[i] = scale * val;\n}\n";

// src/gpu/shaders/conv1d_silu.wgsl
var conv1d_silu_default = "// Fused causal conv1d + SiLU activation.\n// Processes all channels in parallel. Uses persistent conv buffer.\n//\n// For each channel c:\n//   result = sum(buf[k] * weight[c*K+k]) + input[c] * weight[c*K+(K-1)]\n//   shift buffer: discard oldest, add current pre-conv input\n//   output[c] = silu(result)\n//\n// conv_buf: [(K-1) * channels] \u2014 persistent between tokens\n// weights: [channels * K] \u2014 conv kernel\n// data: [channels] \u2014 input/output (in-place)\n\nstruct Params {\n  channels: u32,\n  kernel_size: u32,\n}\n\n@group(0) @binding(0) var<storage, read_write> data: array<f32>;     // [channels] in/out\n@group(0) @binding(1) var<storage, read_write> conv_buf: array<f32>; // [(K-1) * channels]\n@group(0) @binding(2) var<storage, read> weights: array<f32>;        // [channels * K]\n@group(0) @binding(3) var<uniform> params: Params;\n\n@compute @workgroup_size(256)\nfn main(@builtin(global_invocation_id) gid: vec3u) {\n  let c = gid.x;\n  if (c >= params.channels) { return; }\n\n  let K = params.kernel_size;\n  let C = params.channels;\n  let buf_slots = K - 1u;\n\n  // Save pre-conv input for buffer shift\n  let pre_conv = data[c];\n\n  // Compute conv: sum(buf[k*C+c] * w[c*K+k]) + data[c] * w[c*K+(K-1)]\n  var s: f32 = 0.0;\n  for (var k = 0u; k < buf_slots; k++) {\n    s += conv_buf[k * C + c] * weights[c * K + k];\n  }\n  s += pre_conv * weights[c * K + buf_slots];\n\n  // Shift buffer: slot[k] = slot[k+1], newest = pre_conv\n  for (var k = 0u; k < buf_slots - 1u; k++) {\n    conv_buf[k * C + c] = conv_buf[(k + 1u) * C + c];\n  }\n  if (buf_slots > 0u) {\n    conv_buf[(buf_slots - 1u) * C + c] = pre_conv;\n  }\n\n  // SiLU activation + write output\n  data[c] = s / (1.0 + exp(-s));\n}\n";

// src/gpu/shaders/silu_inplace.wgsl
var silu_inplace_default = "// In-place SiLU activation: data[i] = data[i] * sigmoid(data[i])\n\nstruct Params { n: u32, }\n\n@group(0) @binding(0) var<storage, read_write> data: array<f32>;\n@group(0) @binding(1) var<uniform> params: Params;\n\n@compute @workgroup_size(256)\nfn main(@builtin(global_invocation_id) gid: vec3u) {\n  let i = gid.x;\n  if (i >= params.n) { return; }\n  let x = data[i];\n  data[i] = x / (1.0 + exp(-x));\n}\n";

// src/gpu/shaders/l2_norm_groups.wgsl
var l2_norm_groups_default = "// L2 normalize each group independently.\n// data: [numGroups * groupDim], each group of groupDim elements is normalized.\n// One workgroup per group, 256 threads for reduction.\n\nstruct Params {\n  num_groups: u32,\n  group_dim: u32,\n}\n\n@group(0) @binding(0) var<storage, read_write> data: array<f32>;\n@group(0) @binding(1) var<uniform> params: Params;\nvar<workgroup> shared_sum: array<f32, 256>;\n\n@compute @workgroup_size(256)\nfn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) lid: vec3u) {\n  let g = wg.x;\n  if (g >= params.num_groups) { return; }\n  let tid = lid.x;\n  let base = g * params.group_dim;\n  let dim = params.group_dim;\n\n  // Sum of squares\n  var sum_sq: f32 = 0.0;\n  for (var i = tid; i < dim; i += 256u) {\n    let v = data[base + i];\n    sum_sq += v * v;\n  }\n  shared_sum[tid] = sum_sq;\n  workgroupBarrier();\n\n  // Reduction\n  for (var s = 128u; s > 0u; s >>= 1u) {\n    if (tid < s) { shared_sum[tid] += shared_sum[tid + s]; }\n    workgroupBarrier();\n  }\n\n  let norm = sqrt(shared_sum[0]);\n  let inv_norm = select(1.0, 1.0 / norm, norm > 0.0);\n\n  // Normalize\n  for (var i = tid; i < dim; i += 256u) {\n    data[base + i] *= inv_norm;\n  }\n}\n";

// src/gpu/shaders/compute_decay_beta.wgsl
var compute_decay_beta_default = "// Compute decay and beta values for DeltaNet.\n// decay[g] = exp(ssm_a[g] * softplus(alpha[g] + dt_bias[g]))\n// beta_out[g] = sigmoid(beta[g])\n\nstruct Params { n: u32, }\n\n@group(0) @binding(0) var<storage, read> alpha: array<f32>;\n@group(0) @binding(1) var<storage, read> beta: array<f32>;\n@group(0) @binding(2) var<storage, read> ssm_a: array<f32>;\n@group(0) @binding(3) var<storage, read> dt_bias: array<f32>;\n@group(0) @binding(4) var<storage, read_write> decay_out: array<f32>;\n@group(0) @binding(5) var<storage, read_write> beta_out: array<f32>;\n@group(0) @binding(6) var<uniform> params: Params;\n\n@compute @workgroup_size(64)\nfn main(@builtin(global_invocation_id) gid: vec3u) {\n  let g = gid.x;\n  if (g >= params.n) { return; }\n\n  let softplus = log(1.0 + exp(alpha[g] + dt_bias[g]));\n  decay_out[g] = exp(ssm_a[g] * softplus);\n  beta_out[g] = 1.0 / (1.0 + exp(-beta[g]));\n}\n";

// src/gpu/shaders/deinterleave_q.wgsl
var deinterleave_q_default = "// Deinterleave gated Q: split [h0_attn|h0_gate|h1_attn|h1_gate|...] into separate attn and gate buffers.\n// qFull: [numHeads * headDim * 2], qAttn: [numHeads * headDim], qGate: [numHeads * headDim]\n\nstruct Params {\n  num_heads: u32,\n  head_dim: u32,\n}\n\n@group(0) @binding(0) var<storage, read> q_full: array<f32>;\n@group(0) @binding(1) var<storage, read_write> q_attn: array<f32>;\n@group(0) @binding(2) var<storage, read_write> q_gate: array<f32>;\n@group(0) @binding(3) var<uniform> params: Params;\n\n@compute @workgroup_size(256)\nfn main(@builtin(global_invocation_id) gid: vec3u) {\n  let idx = gid.x;\n  let total = params.num_heads * params.head_dim;\n  if (idx >= total) { return; }\n\n  let head = idx / params.head_dim;\n  let dim = idx % params.head_dim;\n  let src_base = head * params.head_dim * 2u;\n\n  q_attn[head * params.head_dim + dim] = q_full[src_base + dim];\n  q_gate[head * params.head_dim + dim] = q_full[src_base + params.head_dim + dim];\n}\n";

// src/gpu/shaders/per_head_rmsnorm.wgsl
var per_head_rmsnorm_default = "// Per-head RMSNorm: normalize each head independently using shared weight.\n// data: [numHeads * headDim], weight: [headDim] (shared across all heads)\n// One workgroup per head, 256 threads for reduction.\n\nstruct Params {\n  num_heads: u32,\n  head_dim: u32,\n  eps_bits: u32,  // float eps stored as u32 bits\n}\n\n@group(0) @binding(0) var<storage, read_write> data: array<f32>;\n@group(0) @binding(1) var<storage, read> weight: array<f32>;\n@group(0) @binding(2) var<uniform> params: Params;\nvar<workgroup> shared_sum: array<f32, 256>;\n\n@compute @workgroup_size(256)\nfn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) lid: vec3u) {\n  let h = wg.x;\n  if (h >= params.num_heads) { return; }\n  let tid = lid.x;\n  let dim = params.head_dim;\n  let base = h * dim;\n  let eps = bitcast<f32>(params.eps_bits);\n\n  // Sum of squares\n  var sum_sq: f32 = 0.0;\n  for (var i = tid; i < dim; i += 256u) {\n    let v = data[base + i];\n    sum_sq += v * v;\n  }\n  shared_sum[tid] = sum_sq;\n  workgroupBarrier();\n\n  for (var s = 128u; s > 0u; s >>= 1u) {\n    if (tid < s) { shared_sum[tid] += shared_sum[tid + s]; }\n    workgroupBarrier();\n  }\n\n  let rms = sqrt(shared_sum[0] / f32(dim) + eps);\n  let inv_rms = 1.0 / rms;\n\n  // Normalize with weight\n  for (var i = tid; i < dim; i += 256u) {\n    data[base + i] = data[base + i] * inv_rms * weight[i];\n  }\n}\n";

// src/gpu/shaders/partial_rope.wgsl
var partial_rope_default = "// Partial RoPE: apply rotary embeddings to first ropeDim positions of each head.\n// Positions ropeDim..headDim-1 are left unchanged.\n// data: [numHeads * headDim], cos/sin tables: [ropeDim/2] precomputed for this position.\n\nstruct Params {\n  num_heads: u32,\n  head_dim: u32,\n  rope_dim: u32,\n}\n\n@group(0) @binding(0) var<storage, read_write> data: array<f32>;\n@group(0) @binding(1) var<storage, read> cos_table: array<f32>;\n@group(0) @binding(2) var<storage, read> sin_table: array<f32>;\n@group(0) @binding(3) var<uniform> params: Params;\n\n@compute @workgroup_size(256)\nfn main(@builtin(global_invocation_id) gid: vec3u) {\n  let idx = gid.x;\n  let half_rope = params.rope_dim / 2u;\n  let total_pairs = params.num_heads * half_rope;\n  if (idx >= total_pairs) { return; }\n\n  let head = idx / half_rope;\n  let pair = idx % half_rope;\n\n  let base = head * params.head_dim;\n  let i0 = base + pair * 2u;\n  let i1 = i0 + 1u;\n\n  let cos_a = cos_table[pair];\n  let sin_a = sin_table[pair];\n\n  let x = data[i0];\n  let y = data[i1];\n\n  data[i0] = x * cos_a - y * sin_a;\n  data[i1] = x * sin_a + y * cos_a;\n}\n";

// src/gpu/shaders/gated_attention.wgsl
var gated_attention_default = "// Gated attention: softmax(Q*K^T / scale) * V * sigmoid(qGate)\n// One workgroup per Q head. Handles GQA (multiple Q heads per KV head).\n// Uses online softmax for numerical stability.\n//\n// qAttn: [numHeads * keyLen], qGate: [numHeads * keyLen]\n// kCache: [numKvHeads * maxSeqLen * keyLen], vCache: [numKvHeads * maxSeqLen * valLen]\n// output: [numHeads * valLen]\n\nstruct Params {\n  num_heads: u32,\n  num_kv_heads: u32,\n  key_len: u32,\n  val_len: u32,\n  max_seq_len: u32,\n  seq_len: u32,\n  scale_bits: u32,  // float scale as u32 bits\n}\n\n@group(0) @binding(0) var<storage, read> q_attn: array<f32>;\n@group(0) @binding(1) var<storage, read> q_gate: array<f32>;\n@group(0) @binding(2) var<storage, read> k_cache: array<f32>;\n@group(0) @binding(3) var<storage, read> v_cache: array<f32>;\n@group(0) @binding(4) var<storage, read_write> output: array<f32>;\n@group(0) @binding(5) var<uniform> params: Params;\n\n// Shared memory for attention scores (max 4096 tokens)\nvar<workgroup> scores: array<f32, 64>;\n\n@compute @workgroup_size(64)\nfn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) lid: vec3u) {\n  let h = wg.x;\n  if (h >= params.num_heads) { return; }\n  let tid = lid.x;\n  let scale = bitcast<f32>(params.scale_bits);\n  let kv_head = h / (params.num_heads / params.num_kv_heads);\n  let q_off = h * params.key_len;\n  let kv_k_base = kv_head * params.max_seq_len * params.key_len;\n  let kv_v_base = kv_head * params.max_seq_len * params.val_len;\n  let out_off = h * params.val_len;\n  let seq_len = params.seq_len;\n\n  // Process tokens in tiles of 64\n  // Online softmax: track running max and sum\n  var running_max: f32 = -1e30;\n  var running_sum: f32 = 0.0;\n\n  // Initialize output to zero\n  for (var d = tid; d < params.val_len; d += 64u) {\n    output[out_off + d] = 0.0;\n  }\n  workgroupBarrier();\n\n  for (var tile_start = 0u; tile_start < seq_len; tile_start += 64u) {\n    let tile_end = min(tile_start + 64u, seq_len);\n\n    // Compute attention score for this thread's token\n    var score: f32 = -1e30;\n    let t = tile_start + tid;\n    if (t < tile_end) {\n      var dot: f32 = 0.0;\n      let k_off = kv_k_base + t * params.key_len;\n      for (var d = 0u; d < params.key_len; d++) {\n        dot += q_attn[q_off + d] * k_cache[k_off + d];\n      }\n      score = dot * scale;\n    }\n    scores[tid] = score;\n    workgroupBarrier();\n\n    // Find tile max\n    var tile_max: f32 = -1e30;\n    let tile_len = tile_end - tile_start;\n    for (var i = 0u; i < tile_len; i++) {\n      tile_max = max(tile_max, scores[i]);\n    }\n\n    // Compute exp scores\n    var tile_sum: f32 = 0.0;\n    if (tid < tile_len) {\n      scores[tid] = exp(scores[tid] - tile_max);\n      tile_sum = scores[tid];\n    }\n    workgroupBarrier();\n\n    // Sum tile scores (simple serial, fast for \u226464)\n    var total_tile_sum: f32 = 0.0;\n    for (var i = 0u; i < tile_len; i++) {\n      total_tile_sum += scores[i];\n    }\n\n    // Online softmax correction\n    let new_max = max(running_max, tile_max);\n    let correction_old = exp(running_max - new_max);\n    let correction_new = exp(tile_max - new_max);\n\n    // Update output: out = out * correction_old + tile_weighted_v * correction_new\n    for (var d = tid; d < params.val_len; d += 64u) {\n      var tile_val: f32 = 0.0;\n      for (var i = 0u; i < tile_len; i++) {\n        tile_val += scores[i] * v_cache[kv_v_base + (tile_start + i) * params.val_len + d];\n      }\n      output[out_off + d] = output[out_off + d] * correction_old + tile_val * correction_new;\n    }\n    workgroupBarrier();\n\n    running_sum = running_sum * correction_old + total_tile_sum * correction_new;\n    running_max = new_max;\n  }\n\n  // Normalize and apply sigmoid gate\n  let inv_sum = select(0.0, 1.0 / running_sum, running_sum > 0.0);\n  for (var d = tid; d < params.val_len; d += 64u) {\n    let gate_idx = min(d, params.key_len - 1u);\n    let gate_val = 1.0 / (1.0 + exp(-q_gate[h * params.key_len + gate_idx]));\n    output[out_off + d] = output[out_off + d] * inv_sum * gate_val;\n  }\n}\n";

// src/gpu/shaders/repeat_tile.wgsl
var repeat_tile_default = "// Repeat-tile: expand data from numKHeads groups to numVHeads groups.\n// src: [numKHeads * headDim], dst: [numVHeads * headDim]\n// factor = numVHeads / numKHeads\n// dst[rep * numKHeads * headDim + g * headDim + d] = src[g * headDim + d]\n\nstruct Params {\n  num_k_heads: u32,\n  head_dim: u32,\n  factor: u32,\n}\n\n@group(0) @binding(0) var<storage, read> src: array<f32>;\n@group(0) @binding(1) var<storage, read_write> dst: array<f32>;\n@group(0) @binding(2) var<uniform> params: Params;\n\n@compute @workgroup_size(256)\nfn main(@builtin(global_invocation_id) gid: vec3u) {\n  let idx = gid.x;\n  let total = params.num_k_heads * params.head_dim * params.factor;\n  if (idx >= total) { return; }\n\n  let k_total = params.num_k_heads * params.head_dim;\n  let src_idx = idx % k_total;\n  dst[idx] = src[src_idx];\n}\n";

// src/gpu/shaders/matmul_batch.wgsl
var matmul_batch_default = "// Batched matrix-vector multiply: process N tokens simultaneously.\n// For each token n and output row m:\n//   output[n * M + m] = sum_k(weights[m * K + k] * input[n * K + k])\n//\n// Dispatch: [M, N] workgroups, 256 threads per workgroup for K reduction.\n\nstruct Params {\n  M: u32,\n  K: u32,\n  N: u32,  // number of tokens (batch size)\n}\n\n@group(0) @binding(0) var<storage, read> weights: array<f32>;\n@group(0) @binding(1) var<storage, read> input: array<f32>;    // [N * K]\n@group(0) @binding(2) var<storage, read_write> output: array<f32>; // [N * M]\n@group(0) @binding(3) var<uniform> params: Params;\nvar<workgroup> shared_sum: array<f32, 256>;\n\n@compute @workgroup_size(256)\nfn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) lid: vec3u) {\n  let m = wg.x + wg.y * 65535u;  // output row (supports large M)\n  let n = wg.z;                    // token index\n  if (m >= params.M || n >= params.N) { return; }\n\n  let tid = lid.x;\n  let K = params.K;\n\n  var sum: f32 = 0.0;\n  for (var k = tid; k < K; k += 256u) {\n    sum += weights[m * K + k] * input[n * K + k];\n  }\n  shared_sum[tid] = sum;\n  workgroupBarrier();\n  for (var s = 128u; s > 0u; s >>= 1u) {\n    if (tid < s) { shared_sum[tid] += shared_sum[tid + s]; }\n    workgroupBarrier();\n  }\n  if (tid == 0u) {\n    output[n * params.M + m] = shared_sum[0];\n  }\n}\n";

// src/gpu/shaders/embedding_batch.wgsl
var embedding_batch_default = "// Batched token embedding lookup: look up N tokens at once.\n// token_ids: [N], output: [N * embDim]\n\nstruct Params {\n  num_tokens: u32,\n  embedding_dim: u32,\n}\n\n@group(0) @binding(0) var<storage, read> weights: array<f32>;\n@group(0) @binding(1) var<storage, read> token_ids: array<u32>;\n@group(0) @binding(2) var<storage, read_write> output: array<f32>;\n@group(0) @binding(3) var<uniform> params: Params;\n\n@compute @workgroup_size(256)\nfn main(@builtin(global_invocation_id) gid: vec3u) {\n  let idx = gid.x;\n  let total = params.num_tokens * params.embedding_dim;\n  if (idx >= total) { return; }\n\n  let n = idx / params.embedding_dim;\n  let d = idx % params.embedding_dim;\n  let token_id = token_ids[n];\n\n  output[n * params.embedding_dim + d] = weights[token_id * params.embedding_dim + d];\n}\n";

// src/gpu/shaders/rmsnorm_batch.wgsl
var rmsnorm_batch_default = "// Batched RMSNorm: normalize N rows independently.\n// input: [N * dim], weight: [dim], output: [N * dim]\n// One workgroup per row.\n\nstruct Params {\n  dim: u32,\n  num_rows: u32,\n  eps_bits: u32,  // float as u32 bits\n}\n\n@group(0) @binding(0) var<storage, read> input: array<f32>;\n@group(0) @binding(1) var<storage, read> weight: array<f32>;\n@group(0) @binding(2) var<storage, read_write> output: array<f32>;\n@group(0) @binding(3) var<uniform> params: Params;\nvar<workgroup> shared_sum: array<f32, 256>;\n\n@compute @workgroup_size(256)\nfn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) lid: vec3u) {\n  let row = wg.x;\n  if (row >= params.num_rows) { return; }\n  let tid = lid.x;\n  let dim = params.dim;\n  let base = row * dim;\n  let eps = bitcast<f32>(params.eps_bits);\n\n  // Sum of squares\n  var sum_sq: f32 = 0.0;\n  for (var i = tid; i < dim; i += 256u) {\n    let v = input[base + i];\n    sum_sq += v * v;\n  }\n  shared_sum[tid] = sum_sq;\n  workgroupBarrier();\n  for (var s = 128u; s > 0u; s >>= 1u) {\n    if (tid < s) { shared_sum[tid] += shared_sum[tid + s]; }\n    workgroupBarrier();\n  }\n\n  let rms = sqrt(shared_sum[0] / f32(dim) + eps);\n  let inv_rms = 1.0 / rms;\n\n  for (var i = tid; i < dim; i += 256u) {\n    output[base + i] = input[base + i] * inv_rms * weight[i];\n  }\n}\n";

// src/gpu/shaders/attention_batch.wgsl
var attention_batch_default = "// Batched causal attention for prefill.\n// Processes N query tokens simultaneously. Each query attends to all previous tokens + itself.\n// Q: [N * numHeads * headDim], K/V written to cache during this call.\n//\n// For each token n and head h:\n//   scores[t] = Q[n,h] \xB7 K[t,kvh] / sqrt(headDim) for t in 0..startPos+n\n//   output[n,h] = softmax(scores) \xB7 V[kvh]\n//\n// Dispatch: [numHeads, N] workgroups, 64 threads per workgroup.\n\nstruct Params {\n  num_heads: u32,\n  num_kv_heads: u32,\n  head_dim: u32,\n  max_seq_len: u32,\n  start_pos: u32,   // KV cache position of first token in batch\n  num_tokens: u32,   // N\n  scale_bits: u32,   // float as u32 bits\n}\n\n@group(0) @binding(0) var<storage, read> q: array<f32>;       // [N * numHeads * headDim]\n@group(0) @binding(1) var<storage, read> k_cache: array<f32>; // [numKvHeads * maxSeqLen * headDim]\n@group(0) @binding(2) var<storage, read> v_cache: array<f32>;\n@group(0) @binding(3) var<storage, read_write> output: array<f32>; // [N * numHeads * headDim]\n@group(0) @binding(4) var<uniform> params: Params;\n\nvar<workgroup> scores: array<f32, 64>;\n\n@compute @workgroup_size(64)\nfn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) lid: vec3u) {\n  let h = wg.x;\n  let n = wg.y;  // token index within batch\n  if (h >= params.num_heads || n >= params.num_tokens) { return; }\n\n  let tid = lid.x;\n  let scale = bitcast<f32>(params.scale_bits);\n  let kv_head = h / (params.num_heads / params.num_kv_heads);\n  let head_dim = params.head_dim;\n  let max_seq = params.max_seq_len;\n\n  // This query attends to tokens 0..startPos+n (inclusive) \u2014 causal mask\n  let seq_len = params.start_pos + n + 1u;\n\n  let q_off = (n * params.num_heads + h) * head_dim;\n  let kv_k_base = kv_head * max_seq * head_dim;\n  let kv_v_base = kv_head * max_seq * head_dim;\n  let out_off = (n * params.num_heads + h) * head_dim;\n\n  // Initialize output\n  for (var d = tid; d < head_dim; d += 64u) {\n    output[out_off + d] = 0.0;\n  }\n  workgroupBarrier();\n\n  var running_max: f32 = -1e30;\n  var running_sum: f32 = 0.0;\n\n  // Process in tiles of 64\n  for (var tile_start = 0u; tile_start < seq_len; tile_start += 64u) {\n    let tile_end = min(tile_start + 64u, seq_len);\n    let tile_len = tile_end - tile_start;\n\n    var score: f32 = -1e30;\n    let t = tile_start + tid;\n    if (t < tile_end) {\n      var dot: f32 = 0.0;\n      let k_off = kv_k_base + t * head_dim;\n      for (var d = 0u; d < head_dim; d++) {\n        dot += q[q_off + d] * k_cache[k_off + d];\n      }\n      score = dot * scale;\n    }\n    scores[tid] = score;\n    workgroupBarrier();\n\n    var tile_max: f32 = -1e30;\n    for (var i = 0u; i < tile_len; i++) {\n      tile_max = max(tile_max, scores[i]);\n    }\n\n    if (tid < tile_len) {\n      scores[tid] = exp(scores[tid] - tile_max);\n    }\n    workgroupBarrier();\n\n    var tile_sum: f32 = 0.0;\n    for (var i = 0u; i < tile_len; i++) {\n      tile_sum += scores[i];\n    }\n\n    let new_max = max(running_max, tile_max);\n    let corr_old = exp(running_max - new_max);\n    let corr_new = exp(tile_max - new_max);\n\n    for (var d = tid; d < head_dim; d += 64u) {\n      var tile_val: f32 = 0.0;\n      for (var i = 0u; i < tile_len; i++) {\n        tile_val += scores[i] * v_cache[kv_v_base + (tile_start + i) * head_dim + d];\n      }\n      output[out_off + d] = output[out_off + d] * corr_old + tile_val * corr_new;\n    }\n    workgroupBarrier();\n\n    running_sum = running_sum * corr_old + tile_sum * corr_new;\n    running_max = new_max;\n  }\n\n  // Normalize\n  let inv_sum = select(0.0, 1.0 / running_sum, running_sum > 0.0);\n  for (var d = tid; d < head_dim; d += 64u) {\n    output[out_off + d] *= inv_sum;\n  }\n}\n";

// src/gpu/shaders/rope_batch.wgsl
var rope_batch_default = "// Batched RoPE: apply rotary embeddings to N tokens at different positions.\n// data: [N * numHeads * headDim], each token has its own position.\n// positions: [N] \u2014 position index for each token.\n// Precomputed cos/sin per dimension pair, looked up per position.\n\nstruct Params {\n  num_heads: u32,\n  head_dim: u32,\n  num_tokens: u32,\n  theta_bits: u32,  // float theta as u32 bits\n}\n\n@group(0) @binding(0) var<storage, read_write> data: array<f32>;\n@group(0) @binding(1) var<storage, read> positions: array<u32>;\n@group(0) @binding(2) var<uniform> params: Params;\n\n@compute @workgroup_size(256)\nfn main(@builtin(global_invocation_id) gid: vec3u) {\n  let idx = gid.x;\n  let half_dim = params.head_dim / 2u;\n  let total_pairs = params.num_tokens * params.num_heads * half_dim;\n  if (idx >= total_pairs) { return; }\n\n  let theta = bitcast<f32>(params.theta_bits);\n  let pairs_per_token = params.num_heads * half_dim;\n  let n = idx / pairs_per_token;\n  let rem = idx % pairs_per_token;\n  let head = rem / half_dim;\n  let pair = rem % half_dim;\n  let pos = positions[n];\n\n  let dim_frac = f32(pair * 2u) / f32(params.head_dim);\n  let freq = 1.0 / pow(theta, dim_frac);\n  let angle = f32(pos) * freq;\n  let cos_a = cos(angle);\n  let sin_a = sin(angle);\n\n  let base = (n * params.num_heads + head) * params.head_dim;\n  let i0 = base + pair * 2u;\n  let i1 = i0 + 1u;\n\n  let x = data[i0];\n  let y = data[i1];\n  data[i0] = x * cos_a - y * sin_a;\n  data[i1] = x * sin_a + y * cos_a;\n}\n";

// src/gpu/shaders/kv_cache_write_batch.wgsl
var kv_cache_write_batch_default = "// Batched KV cache write: write N K/V vectors at consecutive positions.\n// k_in: [N * nKvHeads * headDim], v_in: [N * nKvHeads * headDim]\n// kCache/vCache: [nKvHeads * maxSeqLen * headDim]\n// Writes k_in[n] to kCache[kvh, startPos+n, :] for each KV head.\n\nstruct Params {\n  n_kv_heads: u32,\n  head_dim: u32,\n  max_seq_len: u32,\n  start_pos: u32,\n  num_tokens: u32,\n}\n\n@group(0) @binding(0) var<storage, read> k_in: array<f32>;\n@group(0) @binding(1) var<storage, read> v_in: array<f32>;\n@group(0) @binding(2) var<storage, read_write> k_cache: array<f32>;\n@group(0) @binding(3) var<storage, read_write> v_cache: array<f32>;\n@group(0) @binding(4) var<uniform> params: Params;\n\n@compute @workgroup_size(256)\nfn main(@builtin(global_invocation_id) gid: vec3u) {\n  let idx = gid.x;\n  let total = params.num_tokens * params.n_kv_heads * params.head_dim;\n  if (idx >= total) { return; }\n\n  let kv_size = params.n_kv_heads * params.head_dim;\n  let n = idx / kv_size;\n  let rem = idx % kv_size;\n  let kvh = rem / params.head_dim;\n  let d = rem % params.head_dim;\n\n  let cache_idx = kvh * params.max_seq_len * params.head_dim +\n                  (params.start_pos + n) * params.head_dim + d;\n  let in_idx = n * kv_size + kvh * params.head_dim + d;\n\n  k_cache[cache_idx] = k_in[in_idx];\n  v_cache[cache_idx] = v_in[in_idx];\n}\n";

// src/gpu/shaders/matmul_q8_batch.wgsl
var matmul_q8_batch_default = "// Batched Q8_0 matrix-vector multiply: process N tokens simultaneously.\n// Same as matmul_q8 but with batch dimension in z workgroup.\n// output[n * M + m] = sum_k(dequant(weights[m, k]) * input[n * K + k])\n\nstruct Params {\n  M: u32,\n  K: u32,\n  N: u32,\n}\n\n@group(0) @binding(0) var<storage, read> weights: array<u32>;\n@group(0) @binding(1) var<storage, read> input: array<f32>;\n@group(0) @binding(2) var<storage, read_write> output: array<f32>;\n@group(0) @binding(3) var<uniform> params: Params;\nvar<workgroup> shared_sum: array<f32, 256>;\n\nfn read_scale(bo: u32) -> f32 {\n  let w = weights[bo / 4u];\n  let bits = select(w & 0xFFFFu, (w >> 16u) & 0xFFFFu, (bo % 4u) != 0u);\n  return unpack2x16float(bits).x;\n}\nfn read_i8(bo: u32) -> f32 {\n  let v = (weights[bo / 4u] >> ((bo % 4u) * 8u)) & 0xFFu;\n  return select(f32(v), f32(v) - 256.0, v >= 128u);\n}\n\n@compute @workgroup_size(256)\nfn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) lid: vec3u) {\n  let row = wg.x + wg.y * 65535u;\n  let n = wg.z;\n  if (row >= params.M || n >= params.N) { return; }\n\n  let tid = lid.x;\n  let nblk = params.K / 32u;\n  let roff = row * nblk * 34u;\n  var sum: f32 = 0.0;\n  for (var b = tid; b < nblk; b += 256u) {\n    let bo = roff + b * 34u;\n    let sc = read_scale(bo);\n    let bk = b * 32u;\n    for (var q = 0u; q < 32u; q++) {\n      sum += sc * read_i8(bo + 2u + q) * input[n * params.K + bk + q];\n    }\n  }\n  shared_sum[tid] = sum;\n  workgroupBarrier();\n  for (var s = 128u; s > 0u; s >>= 1u) {\n    if (tid < s) { shared_sum[tid] += shared_sum[tid + s]; }\n    workgroupBarrier();\n  }\n  if (tid == 0u) { output[n * params.M + row] = shared_sum[0]; }\n}\n";

// src/gpu/shaders/deltanet_step.wgsl
var deltanet_step_default = "// DeltaNet state update + output computation.\n// One workgroup per group (head). 128 threads per group.\n//\n// For each group g:\n//   sk[j] = sum_i(state[g,i,j] * k[g*D+i])\n//   error[j] = (v[g*D+j] - decay[g]*sk[j]) * beta[g]\n//   state[g,i,j] = decay[g]*state[g,i,j] + k[g*D+i]*error[j]\n//   output[g*D+j] = sum_i(state[g,i,j] * q[g*D+i]) * scale\n//   Per-head RMSNorm on output[g*D .. g*D+D-1]\n\nstruct Params {\n  head_dim: u32,    // D = 128\n  num_groups: u32,  // 16\n  scale: f32,       // 1/sqrt(D)\n  norm_eps: f32,    // RMSNorm epsilon\n}\n\n@group(0) @binding(0) var<storage, read> q: array<f32>;\n@group(0) @binding(1) var<storage, read> k: array<f32>;\n@group(0) @binding(2) var<storage, read> v: array<f32>;\n@group(0) @binding(3) var<storage, read_write> state: array<f32>;  // [G * D * D]\n@group(0) @binding(4) var<storage, read> decay: array<f32>;        // [G]\n@group(0) @binding(5) var<storage, read> beta_val: array<f32>;     // [G]\n@group(0) @binding(6) var<storage, read> norm_weight: array<f32>;  // [D] shared across groups\n@group(0) @binding(7) var<storage, read_write> output: array<f32>; // [G * D]\n@group(0) @binding(8) var<uniform> params: Params;\n\nvar<workgroup> shared_k: array<f32, 128>;\nvar<workgroup> shared_error: array<f32, 128>;\nvar<workgroup> shared_sum: f32;\n\n@compute @workgroup_size(128)\nfn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) lid: vec3u) {\n  let g = wg.x;\n  if (g >= params.num_groups) { return; }\n  let j = lid.x;  // each thread handles one column j\n  let D = params.head_dim;\n  let base = g * D;\n  let state_base = g * D * D;\n  let d = decay[g];\n  let b = beta_val[g];\n\n  // Load k into shared memory\n  shared_k[j] = k[base + j];\n  workgroupBarrier();\n\n  // 1. sk[j] = S^T * k = sum_i(state[i*D+j] * k[i])\n  var sk: f32 = 0.0;\n  for (var i = 0u; i < D; i++) {\n    sk += state[state_base + i * D + j] * shared_k[i];\n  }\n\n  // 2. error[j] = (v[j] - decay * sk) * beta\n  let err = (v[base + j] - d * sk) * b;\n  shared_error[j] = err;\n  workgroupBarrier();\n\n  // 3. State update: for each row i that this column j touches\n  //    state[i,j] = decay * state[i,j] + k[i] * error[j]\n  // Thread j updates column j across all rows\n  for (var i = 0u; i < D; i++) {\n    let idx = state_base + i * D + j;\n    state[idx] = d * state[idx] + shared_k[i] * shared_error[j];\n  }\n  workgroupBarrier();\n\n  // 4. output[j] = S_new^T * q * scale = sum_i(state[i,j] * q[i]) * scale\n  var o: f32 = 0.0;\n  for (var i = 0u; i < D; i++) {\n    o += state[state_base + i * D + j] * q[base + i];\n  }\n  output[base + j] = o * params.scale;\n  workgroupBarrier();\n\n  // 5. Per-head RMSNorm\n  // Compute sum of squares (reduction across threads)\n  var my_sq = output[base + j] * output[base + j];\n\n  // Use shared memory for reduction\n  // We need to reduce 128 values - use warp-style reduction\n  // Store in shared_k (reuse, we're done with k)\n  shared_k[j] = my_sq;\n  workgroupBarrier();\n  for (var s = 64u; s > 0u; s >>= 1u) {\n    if (j < s) { shared_k[j] += shared_k[j + s]; }\n    workgroupBarrier();\n  }\n\n  let rms = sqrt(shared_k[0] / f32(D) + params.norm_eps);\n  let inv_rms = 1.0 / rms;\n  output[base + j] = output[base + j] * inv_rms * norm_weight[j];\n}\n";

// src/gpu/shaders/silu_gate.wgsl
var silu_gate_default = "// SiLU gate: output[i] = data[i] * silu(gate[i])\n// where silu(x) = x / (1 + exp(-x))\n\nstruct Params { n: u32, }\n\n@group(0) @binding(0) var<storage, read_write> data: array<f32>;\n@group(0) @binding(1) var<storage, read> gate: array<f32>;\n@group(0) @binding(2) var<uniform> params: Params;\n\n@compute @workgroup_size(256)\nfn main(@builtin(global_invocation_id) gid: vec3u) {\n  let i = gid.x;\n  if (i >= params.n) { return; }\n  let g = gate[i];\n  data[i] = data[i] * g / (1.0 + exp(-g));\n}\n";

// src/gpu/compute.ts
function storageReadOnly(binding) {
  return { binding, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } };
}
function storageReadWrite(binding) {
  return { binding, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } };
}
function uniform(binding) {
  return { binding, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } };
}
var ComputeEngine = class {
  device;
  shaders;
  buffers;
  constructor(device) {
    this.device = device;
    this.shaders = new ShaderCache(device);
    this.buffers = new BufferPool(device);
  }
  /** Params cache — unique buffer per unique content. Safe because same content = no conflict. */
  paramsMap = /* @__PURE__ */ new Map();
  createParams(label, data) {
    const u32 = new Uint32Array(data);
    let key = label;
    for (let i = 0; i < u32.length; i++) key += "," + u32[i];
    let buf = this.paramsMap.get(key);
    if (!buf) {
      buf = this.device.createBuffer({
        size: Math.ceil(data.byteLength / 4) * 4,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
      });
      this.device.queue.writeBuffer(buf, 0, data);
      this.paramsMap.set(key, buf);
    }
    return buf;
  }
  /** No-op — params are cached permanently (same content = same buffer, always safe). */
  cleanupParams() {
  }
  /** Batched encoder — multiple compute passes, ONE submit. */
  batchEncoder = null;
  beginBatch() {
    this.batchEncoder = this.device.createCommandEncoder({ label: "fwd" });
  }
  endBatch() {
    if (this.batchEncoder) {
      this.device.queue.submit([this.batchEncoder.finish()]);
      this.batchEncoder = null;
    }
  }
  /** Copy buffer — uses batch encoder if active. */
  copyBuffer(src, dst, size) {
    this.copyBufferRegion(src, 0, dst, 0, size);
  }
  /** Copy buffer region with offsets — uses batch encoder if active. */
  copyBufferRegion(src, srcOffset, dst, dstOffset, size) {
    const encoder = this.batchEncoder ?? this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(src, srcOffset, dst, dstOffset, size);
    if (!this.batchEncoder) {
      this.device.queue.submit([encoder.finish()]);
    }
  }
  /** Dispatch — each dispatch is its own compute pass (for proper barriers).
   *  Uses batch encoder if active (single submit), otherwise standalone. */
  dispatch(shaderSrc, label, layout, entries, workgroups) {
    const cached = this.shaders.getPipeline({ shader: shaderSrc, bindGroupLayout: layout, label });
    const bindGroup = this.shaders.createBindGroup(cached, entries, label);
    const encoder = this.batchEncoder ?? this.device.createCommandEncoder({ label });
    const pass = encoder.beginComputePass({ label });
    pass.setPipeline(cached.pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(workgroups[0], workgroups[1] ?? 1, workgroups[2] ?? 1);
    pass.end();
    if (!this.batchEncoder) {
      this.device.queue.submit([encoder.finish()]);
    }
  }
  // ── Operations ────────────────────────────────────────────────────────
  /** Embedding lookup: output = weights[tokenId * embDim : (tokenId+1) * embDim] */
  embedding(weights, output, tokenId, embDim) {
    const params = this.createParams("embedding_params", new Uint32Array([tokenId, embDim]).buffer);
    this.dispatch(embedding_default, "embedding", [
      storageReadOnly(0),
      storageReadWrite(1),
      uniform(2)
    ], [
      { binding: 0, resource: { buffer: weights } },
      { binding: 1, resource: { buffer: output } },
      { binding: 2, resource: { buffer: params } }
    ], [Math.ceil(embDim / 256)]);
  }
  /** Token embedding lookup for Q8_0 quantized weights. */
  embeddingQ8(weights, output, tokenId, embDim) {
    const params = this.createParams("embedding_q8_params", new Uint32Array([tokenId, embDim]).buffer);
    this.dispatch(embedding_q8_default, "embedding_q8", [
      storageReadOnly(0),
      storageReadWrite(1),
      uniform(2)
    ], [
      { binding: 0, resource: { buffer: weights } },
      { binding: 1, resource: { buffer: output } },
      { binding: 2, resource: { buffer: params } }
    ], [Math.ceil(embDim / 256)]);
  }
  /** RMS Normalization */
  rmsNorm(input, weight, output, n, eps) {
    const buf = new ArrayBuffer(8);
    const view = new DataView(buf);
    view.setUint32(0, n, true);
    view.setFloat32(4, eps, true);
    const paramData = new Uint32Array(2);
    paramData[0] = n;
    const epsView = new Float32Array(1);
    epsView[0] = eps;
    paramData[1] = new Uint32Array(epsView.buffer)[0];
    const params = this.createParams("rmsnorm_params", paramData.buffer);
    this.dispatch(rmsnorm_default, "rmsnorm", [
      storageReadOnly(0),
      storageReadOnly(1),
      storageReadWrite(2),
      uniform(3)
    ], [
      { binding: 0, resource: { buffer: input } },
      { binding: 1, resource: { buffer: weight } },
      { binding: 2, resource: { buffer: output } },
      { binding: 3, resource: { buffer: params } }
    ], [1]);
  }
  /** Fused: copy input→residual AND RMSNorm(input, weight)→output. Saves 1 dispatch. */
  copyAndRmsNorm(input, weight, output, residual, n, eps) {
    const paramData = new Uint32Array(2);
    paramData[0] = n;
    const epsView = new Float32Array(1);
    epsView[0] = eps;
    paramData[1] = new Uint32Array(epsView.buffer)[0];
    const params = this.createParams("copy_rmsnorm_params", paramData.buffer);
    this.dispatch(copy_rmsnorm_default, "copy_rmsnorm", [
      storageReadOnly(0),
      storageReadOnly(1),
      storageReadWrite(2),
      storageReadWrite(3),
      uniform(4)
    ], [
      { binding: 0, resource: { buffer: input } },
      { binding: 1, resource: { buffer: weight } },
      { binding: 2, resource: { buffer: output } },
      { binding: 3, resource: { buffer: residual } },
      { binding: 4, resource: { buffer: params } }
    ], [1]);
  }
  /** Matrix-vector multiply for the appropriate quantization type. */
  matmul(weights, input, output, M, K, quantType) {
    const paramData = new Uint32Array([M, K]);
    const params = this.createParams("matmul_params", paramData.buffer);
    let shader;
    let workgroups;
    switch (quantType) {
      case 0 /* F32 */:
        shader = matmul_default;
        workgroups = M <= 65535 ? [M] : [65535, Math.ceil(M / 65535)];
        break;
      case 2 /* Q4_0 */:
        shader = matmul_q4_default;
        workgroups = M <= 65535 ? [M] : [65535, Math.ceil(M / 65535)];
        break;
      case 8 /* Q8_0 */:
        shader = matmul_q8_mr_default;
        const halfM = Math.ceil(M / 2);
        workgroups = halfM <= 65535 ? [halfM] : [65535, Math.ceil(halfM / 65535)];
        break;
      default:
        throw new Error(`Unsupported quantization type for matmul: ${quantType}`);
    }
    this.dispatch(shader, "matmul", [
      storageReadOnly(0),
      storageReadOnly(1),
      storageReadWrite(2),
      uniform(3)
    ], [
      { binding: 0, resource: { buffer: weights } },
      { binding: 1, resource: { buffer: input } },
      { binding: 2, resource: { buffer: output } },
      { binding: 3, resource: { buffer: params } }
    ], workgroups);
  }
  /** Pre-built RoPE cos/sin tables keyed by "theta,headDim,position". */
  ropeTableCache = /* @__PURE__ */ new Map();
  /** Clear cached RoPE tables (call when loading a new model). */
  clearRopeCache() {
    this.ropeTableCache.clear();
  }
  /** RoPE: apply rotary position embeddings using cached cos/sin tables. */
  rope(data, headDim, ropeDim, position, theta, nElements) {
    const cacheKey = `${theta},${headDim},${position}`;
    let table = this.ropeTableCache.get(cacheKey);
    if (!table) {
      const halfDim = headDim / 2;
      const cosData = new Float32Array(halfDim);
      const sinData = new Float32Array(halfDim);
      for (let i = 0; i < halfDim; i++) {
        const dimFrac = i * 2 / headDim;
        const freq = 1 / Math.pow(theta, dimFrac);
        const angle = position * freq;
        cosData[i] = Math.fround(Math.cos(angle));
        sinData[i] = Math.fround(Math.sin(angle));
      }
      table = {
        cos: this.buffers.createBufferWithData(`rope_cos_${cacheKey}`, cosData.buffer),
        sin: this.buffers.createBufferWithData(`rope_sin_${cacheKey}`, sinData.buffer)
      };
      this.ropeTableCache.set(cacheKey, table);
    }
    const paramData = new Uint32Array(2);
    paramData[0] = nElements;
    paramData[1] = headDim;
    const params = this.createParams("rope_params", paramData.buffer);
    this.dispatch(rope_default, "rope", [
      storageReadWrite(0),
      storageReadOnly(1),
      storageReadOnly(2),
      uniform(3)
    ], [
      { binding: 0, resource: { buffer: data } },
      { binding: 1, resource: { buffer: table.cos } },
      { binding: 2, resource: { buffer: table.sin } },
      { binding: 3, resource: { buffer: params } }
    ], [nElements / 2]);
  }
  /** CPU attention — reads GPU buffers, computes on CPU, writes back. */
  async cpuAttention(q, kCache, vCache, output, numHeads, numKvHeads, headDim, seqLen, maxSeqLen) {
    const scale = 1 / Math.sqrt(headDim);
    const headsPerKvGroup = numHeads / numKvHeads;
    const qData = await this.readBuffer(q, numHeads * headDim * 4);
    const kData = await this.readBuffer(kCache, numKvHeads * maxSeqLen * headDim * 4);
    const vData = await this.readBuffer(vCache, numKvHeads * maxSeqLen * headDim * 4);
    const outData = new Float32Array(numHeads * headDim);
    for (let h = 0; h < numHeads; h++) {
      const kvHead = Math.floor(h / headsPerKvGroup);
      const qOff = h * headDim;
      const scores = new Float32Array(seqLen);
      for (let pos = 0; pos < seqLen; pos++) {
        const kOff = kvHead * maxSeqLen * headDim + pos * headDim;
        let dot = 0;
        for (let d = 0; d < headDim; d++) dot += qData[qOff + d] * kData[kOff + d];
        scores[pos] = dot * scale;
      }
      let maxS = -Infinity;
      for (let i = 0; i < seqLen; i++) maxS = Math.max(maxS, scores[i]);
      let sumE = 0;
      for (let i = 0; i < seqLen; i++) {
        scores[i] = Math.exp(scores[i] - maxS);
        sumE += scores[i];
      }
      for (let i = 0; i < seqLen; i++) scores[i] /= sumE;
      for (let d = 0; d < headDim; d++) {
        let acc = 0;
        for (let pos = 0; pos < seqLen; pos++) {
          const vOff = kvHead * maxSeqLen * headDim + pos * headDim;
          acc += scores[pos] * vData[vOff + d];
        }
        outData[qOff + d] = acc;
      }
    }
    this.device.queue.writeBuffer(output, 0, outData);
  }
  /** GPU attention — single-thread per head. */
  attention(q, kCache, vCache, output, numHeads, numKvHeads, headDim, seqLen, maxSeqLen) {
    const scale = 1 / Math.sqrt(headDim);
    const paramData = new Uint32Array(6);
    paramData[0] = numHeads;
    paramData[1] = numKvHeads;
    paramData[2] = headDim;
    paramData[3] = seqLen;
    paramData[4] = maxSeqLen;
    const scaleView = new Float32Array(1);
    scaleView[0] = scale;
    paramData[5] = new Uint32Array(scaleView.buffer)[0];
    const params = this.createParams("attention_params", paramData.buffer);
    this.dispatch(attention_default, "attention", [
      storageReadOnly(0),
      storageReadOnly(1),
      storageReadOnly(2),
      storageReadWrite(3),
      uniform(4)
    ], [
      { binding: 0, resource: { buffer: q } },
      { binding: 1, resource: { buffer: kCache } },
      { binding: 2, resource: { buffer: vCache } },
      { binding: 3, resource: { buffer: output } },
      { binding: 4, resource: { buffer: params } }
    ], [numHeads]);
  }
  /** Softmax over n elements. */
  softmax(input, output, n) {
    const params = this.createParams("softmax_params", new Uint32Array([n]).buffer);
    this.dispatch(softmax_default, "softmax", [
      storageReadOnly(0),
      storageReadWrite(1),
      uniform(2)
    ], [
      { binding: 0, resource: { buffer: input } },
      { binding: 1, resource: { buffer: output } },
      { binding: 2, resource: { buffer: params } }
    ], [1]);
  }
  /** SiLU activation: output = x * sigmoid(x) */
  silu(input, output, n) {
    const params = this.createParams("silu_params", new Uint32Array([n]).buffer);
    this.dispatch(silu_default, "silu", [
      storageReadOnly(0),
      storageReadWrite(1),
      uniform(2)
    ], [
      { binding: 0, resource: { buffer: input } },
      { binding: 1, resource: { buffer: output } },
      { binding: 2, resource: { buffer: params } }
    ], [Math.ceil(n / 256)]);
  }
  /** Fused SiLU-gate multiply: output = silu(gate) * up */
  siluMul(gate, up, output, n) {
    const params = this.createParams("silu_mul_params", new Uint32Array([n]).buffer);
    this.dispatch(silu_mul_default, "silu_mul", [
      storageReadOnly(0),
      storageReadOnly(1),
      storageReadWrite(2),
      uniform(3)
    ], [
      { binding: 0, resource: { buffer: gate } },
      { binding: 1, resource: { buffer: up } },
      { binding: 2, resource: { buffer: output } },
      { binding: 3, resource: { buffer: params } }
    ], [Math.ceil(n / 256)]);
  }
  /** Element-wise add: output = a + b */
  add(a, b, output, n) {
    const params = this.createParams("add_params", new Uint32Array([n]).buffer);
    this.dispatch(add_default, "add", [
      storageReadOnly(0),
      storageReadOnly(1),
      storageReadWrite(2),
      uniform(3)
    ], [
      { binding: 0, resource: { buffer: a } },
      { binding: 1, resource: { buffer: b } },
      { binding: 2, resource: { buffer: output } },
      { binding: 3, resource: { buffer: params } }
    ], [Math.ceil(n / 256)]);
  }
  /** In-place bias add: data[i] += bias[i]. */
  addBias(data, bias, n) {
    const params = this.createParams("add_bias_params", new Uint32Array([n]).buffer);
    this.dispatch(add_bias_default, "add_bias", [
      storageReadWrite(0),
      storageReadOnly(1),
      uniform(2)
    ], [
      { binding: 0, resource: { buffer: data } },
      { binding: 1, resource: { buffer: bias } },
      { binding: 2, resource: { buffer: params } }
    ], [Math.ceil(n / 256)]);
  }
  // ── DeltaNet GPU ops ─────────────────────────────────────────────
  /** Fused causal conv1d + SiLU: convolve data with persistent buffer, apply SiLU. */
  conv1dSilu(data, convBuf, weights, channels, kernelSize) {
    const params = this.createParams("conv1d_silu_params", new Uint32Array([channels, kernelSize]).buffer);
    this.dispatch(conv1d_silu_default, "conv1d_silu", [
      storageReadWrite(0),
      storageReadWrite(1),
      storageReadOnly(2),
      uniform(3)
    ], [
      { binding: 0, resource: { buffer: data } },
      { binding: 1, resource: { buffer: convBuf } },
      { binding: 2, resource: { buffer: weights } },
      { binding: 3, resource: { buffer: params } }
    ], [Math.ceil(channels / 256)]);
  }
  /** In-place SiLU activation. */
  siluInplace(data, n) {
    const params = this.createParams("silu_ip_params", new Uint32Array([n]).buffer);
    this.dispatch(silu_inplace_default, "silu_inplace", [
      storageReadWrite(0),
      uniform(1)
    ], [
      { binding: 0, resource: { buffer: data } },
      { binding: 1, resource: { buffer: params } }
    ], [Math.ceil(n / 256)]);
  }
  /** L2 normalize each group independently. */
  l2NormGroups(data, numGroups, groupDim) {
    const params = this.createParams("l2norm_params", new Uint32Array([numGroups, groupDim]).buffer);
    this.dispatch(l2_norm_groups_default, "l2_norm_groups", [
      storageReadWrite(0),
      uniform(1)
    ], [
      { binding: 0, resource: { buffer: data } },
      { binding: 1, resource: { buffer: params } }
    ], [numGroups]);
  }
  /** Compute DeltaNet decay and beta values. */
  computeDecayBeta(alpha, beta, ssmA, dtBias, decayOut, betaOut, n) {
    const params = this.createParams("decay_beta_params", new Uint32Array([n]).buffer);
    this.dispatch(compute_decay_beta_default, "compute_decay_beta", [
      storageReadOnly(0),
      storageReadOnly(1),
      storageReadOnly(2),
      storageReadOnly(3),
      storageReadWrite(4),
      storageReadWrite(5),
      uniform(6)
    ], [
      { binding: 0, resource: { buffer: alpha } },
      { binding: 1, resource: { buffer: beta } },
      { binding: 2, resource: { buffer: ssmA } },
      { binding: 3, resource: { buffer: dtBias } },
      { binding: 4, resource: { buffer: decayOut } },
      { binding: 5, resource: { buffer: betaOut } },
      { binding: 6, resource: { buffer: params } }
    ], [Math.ceil(n / 64)]);
  }
  /** DeltaNet state update + output + per-head RMSNorm. One dispatch for all groups. */
  deltanetStep(q, k, v, state, decay, betaVal, normWeight, output, numGroups, headDim, scale, normEps) {
    const paramBuf = new ArrayBuffer(16);
    const u32 = new Uint32Array(paramBuf);
    const f32 = new Float32Array(paramBuf);
    u32[0] = headDim;
    u32[1] = numGroups;
    f32[2] = scale;
    f32[3] = normEps;
    const params = this.createParams("deltanet_step_params", paramBuf);
    this.dispatch(deltanet_step_default, "deltanet_step", [
      storageReadOnly(0),
      storageReadOnly(1),
      storageReadOnly(2),
      storageReadWrite(3),
      storageReadOnly(4),
      storageReadOnly(5),
      storageReadOnly(6),
      storageReadWrite(7),
      uniform(8)
    ], [
      { binding: 0, resource: { buffer: q } },
      { binding: 1, resource: { buffer: k } },
      { binding: 2, resource: { buffer: v } },
      { binding: 3, resource: { buffer: state } },
      { binding: 4, resource: { buffer: decay } },
      { binding: 5, resource: { buffer: betaVal } },
      { binding: 6, resource: { buffer: normWeight } },
      { binding: 7, resource: { buffer: output } },
      { binding: 8, resource: { buffer: params } }
    ], [numGroups]);
  }
  /** SiLU gate: data[i] *= silu(gate[i]). */
  siluGate(data, gate, n) {
    const params = this.createParams("silu_gate_params", new Uint32Array([n]).buffer);
    this.dispatch(silu_gate_default, "silu_gate", [
      storageReadWrite(0),
      storageReadOnly(1),
      uniform(2)
    ], [
      { binding: 0, resource: { buffer: data } },
      { binding: 1, resource: { buffer: gate } },
      { binding: 2, resource: { buffer: params } }
    ], [Math.ceil(n / 256)]);
  }
  // ── Standard Attention GPU ops (Qwen 3.5) ────────────────────────
  /** Deinterleave gated Q: split [attn|gate|attn|gate|...] per head. */
  deinterleaveQ(qFull, qAttn, qGate, numHeads, headDim) {
    const params = this.createParams("deinterleave_q_params", new Uint32Array([numHeads, headDim]).buffer);
    this.dispatch(deinterleave_q_default, "deinterleave_q", [
      storageReadOnly(0),
      storageReadWrite(1),
      storageReadWrite(2),
      uniform(3)
    ], [
      { binding: 0, resource: { buffer: qFull } },
      { binding: 1, resource: { buffer: qAttn } },
      { binding: 2, resource: { buffer: qGate } },
      { binding: 3, resource: { buffer: params } }
    ], [Math.ceil(numHeads * headDim / 256)]);
  }
  /** Per-head RMSNorm with shared weight vector. */
  perHeadRmsNorm(data, weight, numHeads, headDim, eps) {
    const paramBuf = new Uint32Array(3);
    paramBuf[0] = numHeads;
    paramBuf[1] = headDim;
    paramBuf[2] = new Uint32Array(new Float32Array([eps]).buffer)[0];
    const params = this.createParams("per_head_rmsnorm_params", paramBuf.buffer);
    this.dispatch(per_head_rmsnorm_default, "per_head_rmsnorm", [
      storageReadWrite(0),
      storageReadOnly(1),
      uniform(2)
    ], [
      { binding: 0, resource: { buffer: data } },
      { binding: 1, resource: { buffer: weight } },
      { binding: 2, resource: { buffer: params } }
    ], [numHeads]);
  }
  /** Partial RoPE: rotate first ropeDim positions of each head using precomputed tables. */
  partialRope(data, numHeads, headDim, ropeDim, position, theta) {
    const halfDim = ropeDim / 2;
    const cacheKey = `partial_rope,${theta},${ropeDim},${position}`;
    let table = this.ropeTableCache.get(cacheKey);
    if (!table) {
      const cosData = new Float32Array(halfDim);
      const sinData = new Float32Array(halfDim);
      for (let i = 0; i < halfDim; i++) {
        const freq = 1 / Math.pow(theta, 2 * i / ropeDim);
        const angle = position * freq;
        cosData[i] = Math.fround(Math.cos(angle));
        sinData[i] = Math.fround(Math.sin(angle));
      }
      table = {
        cos: this.buffers.createBufferWithData(`prope_cos_${cacheKey}`, cosData.buffer),
        sin: this.buffers.createBufferWithData(`prope_sin_${cacheKey}`, sinData.buffer)
      };
      this.ropeTableCache.set(cacheKey, table);
    }
    const params = this.createParams("partial_rope_params", new Uint32Array([numHeads, headDim, ropeDim]).buffer);
    this.dispatch(partial_rope_default, "partial_rope", [
      storageReadWrite(0),
      storageReadOnly(1),
      storageReadOnly(2),
      uniform(3)
    ], [
      { binding: 0, resource: { buffer: data } },
      { binding: 1, resource: { buffer: table.cos } },
      { binding: 2, resource: { buffer: table.sin } },
      { binding: 3, resource: { buffer: params } }
    ], [Math.ceil(numHeads * halfDim / 256)]);
  }
  /** Gated attention: softmax(QK^T/scale) * V * sigmoid(qGate). One workgroup per head. */
  gatedAttention(qAttn, qGate, kCache, vCache, output, numHeads, numKvHeads, keyLen, valLen, maxSeqLen, seqLen, scale) {
    const paramBuf = new ArrayBuffer(28);
    const u32 = new Uint32Array(paramBuf);
    u32[0] = numHeads;
    u32[1] = numKvHeads;
    u32[2] = keyLen;
    u32[3] = valLen;
    u32[4] = maxSeqLen;
    u32[5] = seqLen;
    u32[6] = new Uint32Array(new Float32Array([scale]).buffer)[0];
    const params = this.createParams("gated_attn_params", paramBuf);
    this.dispatch(gated_attention_default, "gated_attention", [
      storageReadOnly(0),
      storageReadOnly(1),
      storageReadOnly(2),
      storageReadOnly(3),
      storageReadWrite(4),
      uniform(5)
    ], [
      { binding: 0, resource: { buffer: qAttn } },
      { binding: 1, resource: { buffer: qGate } },
      { binding: 2, resource: { buffer: kCache } },
      { binding: 3, resource: { buffer: vCache } },
      { binding: 4, resource: { buffer: output } },
      { binding: 5, resource: { buffer: params } }
    ], [numHeads]);
  }
  // ── Batched prefill ops ──────────────────────────────────────────
  /** Batched matmul: output[n,m] = sum_k(weights[m,k] * input[n,k]) for N tokens. */
  matmulBatch(weights, input, output, M, K, N, quantType = 0 /* F32 */) {
    const paramData = new Uint32Array([M, K, N]);
    const params = this.createParams("matmul_batch_params", paramData.buffer);
    const xGroups = M <= 65535 ? M : 65535;
    const yGroups = M <= 65535 ? 1 : Math.ceil(M / 65535);
    let shader;
    switch (quantType) {
      case 0 /* F32 */:
        shader = matmul_batch_default;
        break;
      case 8 /* Q8_0 */:
        shader = matmul_q8_batch_default;
        break;
      default:
        shader = matmul_batch_default;
        break;
    }
    this.dispatch(shader, "matmul_batch", [
      storageReadOnly(0),
      storageReadOnly(1),
      storageReadWrite(2),
      uniform(3)
    ], [
      { binding: 0, resource: { buffer: weights } },
      { binding: 1, resource: { buffer: input } },
      { binding: 2, resource: { buffer: output } },
      { binding: 3, resource: { buffer: params } }
    ], [xGroups, yGroups, N]);
  }
  /** Batched embedding lookup: look up N token IDs at once. weights must be F32. */
  embeddingBatch(weights, tokenIds, output, numTokens, embDim) {
    const params = this.createParams("emb_batch_params", new Uint32Array([numTokens, embDim]).buffer);
    this.dispatch(embedding_batch_default, "embedding_batch", [
      storageReadOnly(0),
      storageReadOnly(1),
      storageReadWrite(2),
      uniform(3)
    ], [
      { binding: 0, resource: { buffer: weights } },
      { binding: 1, resource: { buffer: tokenIds } },
      { binding: 2, resource: { buffer: output } },
      { binding: 3, resource: { buffer: params } }
    ], [Math.ceil(numTokens * embDim / 256)]);
  }
  /** Batched RMSNorm: normalize N rows independently. */
  rmsNormBatch(input, weight, output, dim, numRows, eps) {
    const paramBuf = new Uint32Array(3);
    paramBuf[0] = dim;
    paramBuf[1] = numRows;
    paramBuf[2] = new Uint32Array(new Float32Array([eps]).buffer)[0];
    const params = this.createParams("rmsnorm_batch_params", paramBuf.buffer);
    this.dispatch(rmsnorm_batch_default, "rmsnorm_batch", [
      storageReadOnly(0),
      storageReadOnly(1),
      storageReadWrite(2),
      uniform(3)
    ], [
      { binding: 0, resource: { buffer: input } },
      { binding: 1, resource: { buffer: weight } },
      { binding: 2, resource: { buffer: output } },
      { binding: 3, resource: { buffer: params } }
    ], [numRows]);
  }
  /** Batched causal attention for prefill. Dispatch: [numHeads, numTokens]. */
  attentionBatch(q, kCache, vCache, output, numHeads, numKvHeads, headDim, maxSeqLen, startPos, numTokens) {
    const scale = 1 / Math.sqrt(headDim);
    const paramBuf = new ArrayBuffer(28);
    const u32 = new Uint32Array(paramBuf);
    u32[0] = numHeads;
    u32[1] = numKvHeads;
    u32[2] = headDim;
    u32[3] = maxSeqLen;
    u32[4] = startPos;
    u32[5] = numTokens;
    u32[6] = new Uint32Array(new Float32Array([scale]).buffer)[0];
    const params = this.createParams("attn_batch_params", paramBuf);
    this.dispatch(attention_batch_default, "attention_batch", [
      storageReadOnly(0),
      storageReadOnly(1),
      storageReadOnly(2),
      storageReadWrite(3),
      uniform(4)
    ], [
      { binding: 0, resource: { buffer: q } },
      { binding: 1, resource: { buffer: kCache } },
      { binding: 2, resource: { buffer: vCache } },
      { binding: 3, resource: { buffer: output } },
      { binding: 4, resource: { buffer: params } }
    ], [numHeads, numTokens]);
  }
  /** Batched RoPE: apply rotary embeddings to N tokens at different positions. */
  ropeBatch(data, positions, numHeads, headDim, numTokens, theta) {
    const paramBuf = new Uint32Array(4);
    paramBuf[0] = numHeads;
    paramBuf[1] = headDim;
    paramBuf[2] = numTokens;
    paramBuf[3] = new Uint32Array(new Float32Array([theta]).buffer)[0];
    const params = this.createParams("rope_batch_params", paramBuf.buffer);
    const totalPairs = numTokens * numHeads * (headDim / 2);
    this.dispatch(rope_batch_default, "rope_batch", [
      storageReadWrite(0),
      storageReadOnly(1),
      uniform(2)
    ], [
      { binding: 0, resource: { buffer: data } },
      { binding: 1, resource: { buffer: positions } },
      { binding: 2, resource: { buffer: params } }
    ], [Math.ceil(totalPairs / 256)]);
  }
  /** Batched KV cache write: write N K/V vectors at consecutive positions. */
  kvCacheWriteBatch(kIn, vIn, kCache, vCache, nKvHeads, headDim, maxSeqLen, startPos, numTokens) {
    const paramBuf = new Uint32Array([nKvHeads, headDim, maxSeqLen, startPos, numTokens]);
    const params = this.createParams("kv_write_batch_params", paramBuf.buffer);
    const total = numTokens * nKvHeads * headDim;
    this.dispatch(kv_cache_write_batch_default, "kv_cache_write_batch", [
      storageReadOnly(0),
      storageReadOnly(1),
      storageReadWrite(2),
      storageReadWrite(3),
      uniform(4)
    ], [
      { binding: 0, resource: { buffer: kIn } },
      { binding: 1, resource: { buffer: vIn } },
      { binding: 2, resource: { buffer: kCache } },
      { binding: 3, resource: { buffer: vCache } },
      { binding: 4, resource: { buffer: params } }
    ], [Math.ceil(total / 256)]);
  }
  /** Repeat-tile: expand numKHeads groups to numVHeads groups by repeating. */
  repeatTile(src, dst, numKHeads, headDim, factor) {
    const params = this.createParams("repeat_tile_params", new Uint32Array([numKHeads, headDim, factor]).buffer);
    const total = numKHeads * headDim * factor;
    this.dispatch(repeat_tile_default, "repeat_tile", [
      storageReadOnly(0),
      storageReadWrite(1),
      uniform(2)
    ], [
      { binding: 0, resource: { buffer: src } },
      { binding: 1, resource: { buffer: dst } },
      { binding: 2, resource: { buffer: params } }
    ], [Math.ceil(total / 256)]);
  }
  /** Reusable readback buffer — avoids allocation per readLogits call. */
  readbackBuf = null;
  readbackSize = 0;
  /** Read buffer data back to CPU. */
  async readBuffer(buffer, size) {
    if (!this.readbackBuf || this.readbackSize < size) {
      this.readbackBuf?.destroy();
      this.readbackSize = size;
      this.readbackBuf = this.device.createBuffer({
        label: "readback",
        size,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
      });
    }
    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(buffer, 0, this.readbackBuf, 0, size);
    this.device.queue.submit([encoder.finish()]);
    await this.readbackBuf.mapAsync(GPUMapMode.READ);
    const data = new Float32Array(this.readbackBuf.getMappedRange().slice(0));
    this.readbackBuf.unmap();
    return data;
  }
};

// src/gguf/gguf-parser.ts
var GGUF_MAGIC = 1179993927;
var BinaryReader = class {
  view;
  offset;
  constructor(buffer, offset = 0) {
    this.view = new DataView(buffer);
    this.offset = offset;
  }
  get position() {
    return this.offset;
  }
  readUint8() {
    const v = this.view.getUint8(this.offset);
    this.offset += 1;
    return v;
  }
  readInt8() {
    const v = this.view.getInt8(this.offset);
    this.offset += 1;
    return v;
  }
  readUint16() {
    const v = this.view.getUint16(this.offset, true);
    this.offset += 2;
    return v;
  }
  readInt16() {
    const v = this.view.getInt16(this.offset, true);
    this.offset += 2;
    return v;
  }
  readUint32() {
    const v = this.view.getUint32(this.offset, true);
    this.offset += 4;
    return v;
  }
  readInt32() {
    const v = this.view.getInt32(this.offset, true);
    this.offset += 4;
    return v;
  }
  readFloat32() {
    const v = this.view.getFloat32(this.offset, true);
    this.offset += 4;
    return v;
  }
  readFloat64() {
    const v = this.view.getFloat64(this.offset, true);
    this.offset += 8;
    return v;
  }
  readUint64() {
    const lo = this.view.getUint32(this.offset, true);
    const hi = this.view.getUint32(this.offset + 4, true);
    this.offset += 8;
    return hi * 4294967296 + lo;
  }
  readInt64() {
    const lo = this.view.getUint32(this.offset, true);
    const hi = this.view.getInt32(this.offset + 4, true);
    this.offset += 8;
    return hi * 4294967296 + lo;
  }
  readString() {
    const length = this.readUint64();
    if (this.offset + length > this.view.byteLength) {
      throw new RangeError(
        `String read out of bounds: offset=${this.offset} length=${length} bufferSize=${this.view.byteLength}`
      );
    }
    const bytes = new Uint8Array(this.view.buffer, this.offset, length);
    this.offset += length;
    return new TextDecoder().decode(bytes);
  }
  readBytes(count) {
    const bytes = new Uint8Array(this.view.buffer, this.offset, count);
    this.offset += count;
    return bytes;
  }
};
function readMetadataValue(reader, type) {
  switch (type) {
    case 0 /* Uint8 */:
      return reader.readUint8();
    case 1 /* Int8 */:
      return reader.readInt8();
    case 2 /* Uint16 */:
      return reader.readUint16();
    case 3 /* Int16 */:
      return reader.readInt16();
    case 4 /* Uint32 */:
      return reader.readUint32();
    case 5 /* Int32 */:
      return reader.readInt32();
    case 6 /* Float32 */:
      return reader.readFloat32();
    case 7 /* Bool */:
      return reader.readUint8() !== 0;
    case 8 /* String */:
      return reader.readString();
    case 10 /* Uint64 */:
      return reader.readUint64();
    case 11 /* Int64 */:
      return reader.readInt64();
    case 12 /* Float64 */:
      return reader.readFloat64();
    case 9 /* Array */:
      return readArray(reader);
    default:
      throw new Error(`Unknown metadata type: ${type}`);
  }
}
function readArray(reader) {
  const elementType = reader.readUint32();
  const count = reader.readUint64();
  const arr = new Array(count);
  for (let i = 0; i < count; i++) {
    arr[i] = readMetadataValue(reader, elementType);
  }
  return arr;
}
function parseGguf(buffer) {
  const reader = new BinaryReader(buffer);
  const magic = reader.readUint32();
  if (magic !== GGUF_MAGIC) {
    throw new Error(`Invalid GGUF magic: 0x${magic.toString(16)}. Expected 0x${GGUF_MAGIC.toString(16)}.`);
  }
  const version = reader.readUint32();
  if (version < 2 || version > 3) {
    throw new Error(`Unsupported GGUF version: ${version}. Only v2 and v3 are supported.`);
  }
  const tensorCount = reader.readUint64();
  const metadataKvCount = reader.readUint64();
  const header = { magic, version, tensorCount, metadataKvCount };
  const metadataMap = /* @__PURE__ */ new Map();
  for (let i = 0; i < metadataKvCount; i++) {
    const key = reader.readString();
    const type = reader.readUint32();
    const value = readMetadataValue(reader, type);
    metadataMap.set(key, value);
  }
  const alignment = metadataMap.get("general.alignment") ?? 32;
  const tensors = new Array(tensorCount);
  for (let i = 0; i < tensorCount; i++) {
    const name = reader.readString();
    const nDimensions = reader.readUint32();
    const dimensions = new Array(nDimensions);
    for (let d = 0; d < nDimensions; d++) {
      dimensions[d] = reader.readUint64();
    }
    const type = reader.readUint32();
    const offset = reader.readUint64();
    let elementCount = 1;
    for (let d = 0; d < nDimensions; d++) elementCount *= dimensions[d];
    tensors[i] = {
      name,
      nDimensions,
      dimensions,
      type,
      offset,
      elementCount,
      byteSize: tensorByteSize(type, elementCount)
    };
  }
  const currentPos = reader.position;
  const remainder = currentPos % alignment;
  const tensorDataOffset = remainder === 0 ? currentPos : currentPos + (alignment - remainder);
  const architecture = metadataMap.get("general.architecture") ?? "unknown";
  const prefix = architecture;
  return {
    header,
    architecture,
    blockCount: metadataMap.get(`${prefix}.block_count`) ?? 0,
    embeddingLength: metadataMap.get(`${prefix}.embedding_length`) ?? 0,
    headCount: metadataMap.get(`${prefix}.attention.head_count`) ?? 0,
    headCountKv: metadataMap.get(`${prefix}.attention.head_count_kv`) ?? 0,
    contextLength: metadataMap.get(`${prefix}.context_length`) ?? 0,
    feedForwardLength: metadataMap.get(`${prefix}.feed_forward_length`) ?? 0,
    ropeFreqBase: metadataMap.get(`${prefix}.rope.freq_base`) ?? 1e4,
    rmsNormEps: metadataMap.get(`${prefix}.attention.layer_norm_rms_epsilon`) ?? 1e-5,
    vocabSize: (metadataMap.get("tokenizer.ggml.tokens") ?? []).length,
    metadata: metadataMap,
    tensors,
    tensorDataOffset,
    alignment
  };
}
async function fetchGgufHeader(url, maxBytes = 4 * 1024 * 1024) {
  const response = await fetch(url, {
    headers: { Range: `bytes=0-${maxBytes - 1}` }
  });
  if (!response.ok && response.status !== 206) {
    throw new Error(`Failed to fetch GGUF header: ${response.status} ${response.statusText}`);
  }
  const buffer = await response.arrayBuffer();
  try {
    return parseGguf(buffer);
  } catch (e) {
    if (maxBytes < 64 * 1024 * 1024) {
      return fetchGgufHeader(url, maxBytes * 2);
    }
    throw e;
  }
}

// src/storage/streaming-loader.ts
var CACHE_NAME = "llogos-webgpu-models";
async function streamTensors(url, info, onTensor, onProgress) {
  const tensorDataStart = info.tensorDataOffset;
  const totalTensors = info.tensors.length;
  const sortedTensors = [...info.tensors].sort((a, b) => a.offset - b.offset);
  const extractions = sortedTensors.map((t) => ({
    tensor: t,
    start: tensorDataStart + t.offset,
    end: tensorDataStart + t.offset + t.byteSize
  }));
  const response = await getResponse(url, onProgress);
  const contentLength = parseInt(response.headers.get("content-length") ?? "0", 10);
  const reader = response.body?.getReader();
  if (!reader) {
    throw new Error("Response body is not readable as a stream");
  }
  let filePos = 0;
  let extractIdx = 0;
  let tensorsLoaded = 0;
  let pendingBuffer = null;
  let pendingStart = 0;
  let pendingNeeded = 0;
  while (extractIdx < extractions.length) {
    const { done, value } = await reader.read();
    if (done) break;
    const chunk = value;
    const chunkStart = filePos;
    const chunkEnd = filePos + chunk.byteLength;
    filePos = chunkEnd;
    onProgress?.({
      phase: "Loading tensors",
      bytesDownloaded: filePos,
      totalBytes: contentLength || filePos,
      tensorsLoaded,
      totalTensors
    });
    if (pendingBuffer) {
      const needed = pendingNeeded - (pendingBuffer.byteLength - pendingStart);
      const available = Math.min(chunk.byteLength, needed);
      const newBuf = new Uint8Array(pendingBuffer.byteLength - pendingStart + available);
      newBuf.set(new Uint8Array(pendingBuffer.buffer, pendingBuffer.byteOffset + pendingStart, pendingBuffer.byteLength - pendingStart));
      newBuf.set(chunk.subarray(0, available), pendingBuffer.byteLength - pendingStart);
      if (newBuf.byteLength >= pendingNeeded) {
        const ext = extractions[extractIdx];
        onTensor(ext.tensor.name, newBuf.buffer.slice(newBuf.byteOffset, newBuf.byteOffset + ext.tensor.byteSize), ext.tensor);
        tensorsLoaded++;
        extractIdx++;
        pendingBuffer = null;
        pendingStart = 0;
        pendingNeeded = 0;
      } else {
        pendingBuffer = newBuf;
        pendingStart = 0;
        pendingNeeded = pendingNeeded;
        continue;
      }
    }
    while (extractIdx < extractions.length) {
      const ext = extractions[extractIdx];
      if (ext.start >= chunkEnd) break;
      if (ext.start >= chunkStart && ext.end <= chunkEnd) {
        const localStart = ext.start - chunkStart;
        const tensorData = chunk.buffer.slice(
          chunk.byteOffset + localStart,
          chunk.byteOffset + localStart + ext.tensor.byteSize
        );
        onTensor(ext.tensor.name, tensorData, ext.tensor);
        tensorsLoaded++;
        extractIdx++;
      } else if (ext.start >= chunkStart && ext.start < chunkEnd) {
        const localStart = ext.start - chunkStart;
        pendingBuffer = chunk.subarray(localStart);
        pendingStart = 0;
        pendingNeeded = ext.tensor.byteSize;
        if (pendingBuffer.byteLength >= pendingNeeded) {
          onTensor(ext.tensor.name, pendingBuffer.buffer.slice(
            pendingBuffer.byteOffset,
            pendingBuffer.byteOffset + ext.tensor.byteSize
          ), ext.tensor);
          tensorsLoaded++;
          extractIdx++;
          pendingBuffer = null;
          pendingNeeded = 0;
        }
        break;
      } else {
        extractIdx++;
      }
    }
  }
  while (true) {
    const { done } = await reader.read();
    if (done) break;
  }
  onProgress?.({
    phase: "Complete",
    bytesDownloaded: filePos,
    totalBytes: contentLength || filePos,
    tensorsLoaded,
    totalTensors
  });
}
async function getResponse(url, onProgress) {
  let cache = null;
  try {
    if (typeof caches !== "undefined") {
      cache = await caches.open(CACHE_NAME);
      const cached = await cache.match(url);
      if (cached) {
        onProgress?.({ phase: "Loading from cache", bytesDownloaded: 0, totalBytes: 0, tensorsLoaded: 0, totalTensors: 0 });
        return cached;
      }
    }
  } catch {
  }
  onProgress?.({ phase: "Downloading", bytesDownloaded: 0, totalBytes: 0, tensorsLoaded: 0, totalTensors: 0 });
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Download failed: ${response.status}`);
  if (cache) {
    try {
      const [stream1, stream2] = response.body.tee();
      const cachedResponse = new Response(stream2, {
        headers: { "content-length": response.headers.get("content-length") ?? "0" }
      });
      cache.put(url, cachedResponse).catch(() => {
      });
      return new Response(stream1, { headers: response.headers });
    } catch {
    }
  }
  return response;
}

// src/tokenizer/bpe-tokenizer.ts
var PRE_TOKENIZE_RE = /'(?:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+/gu;
var BpeTokenizer = class {
  tokens;
  tokenToId;
  mergeRank;
  useByteEncoding;
  specialTokens;
  // sorted longest-first for greedy matching
  bosTokenId;
  eosTokenId;
  padTokenId;
  constructor(tokens, merges, bosTokenId, eosTokenId, padTokenId, useByteEncoding) {
    this.tokens = tokens;
    this.bosTokenId = bosTokenId;
    this.eosTokenId = eosTokenId;
    this.padTokenId = padTokenId;
    this.useByteEncoding = useByteEncoding;
    this.tokenToId = /* @__PURE__ */ new Map();
    for (let i = 0; i < tokens.length; i++) {
      if (!this.tokenToId.has(tokens[i])) {
        this.tokenToId.set(tokens[i], i);
      }
    }
    this.mergeRank = /* @__PURE__ */ new Map();
    for (let i = 0; i < merges.length; i++) {
      this.mergeRank.set(merges[i], i);
    }
    this.specialTokens = tokens.filter((t) => t.startsWith("<") && t.endsWith(">") && t.length > 2).sort((a, b) => b.length - a.length);
  }
  get vocabSize() {
    return this.tokens.length;
  }
  /** Get token ID for a string, or -1 if not found. */
  getTokenId(token) {
    return this.tokenToId.get(token) ?? -1;
  }
  /** Encode text to token IDs, handling special tokens. */
  encode(text) {
    if (!text) return [];
    const segments = this.splitOnSpecialTokens(text);
    const result = [];
    for (const seg of segments) {
      if (seg.isSpecial) {
        const id = this.tokenToId.get(seg.text);
        if (id !== void 0) result.push(id);
        else {
          const fwToken = seg.text.replace(/\|/g, "\uFF5C");
          const fwId = this.tokenToId.get(fwToken);
          if (fwId !== void 0) result.push(fwId);
        }
      } else {
        result.push(...this.encodeBpe(seg.text));
      }
    }
    return result;
  }
  /** BPE-encode a text segment (no special tokens). */
  encodeBpe(text) {
    if (!text) return [];
    const result = [];
    const matches = text.matchAll(PRE_TOKENIZE_RE);
    for (const match of matches) {
      const chunk = match[0];
      const symbols = this.useByteEncoding ? byteEncodeChunk(chunk) : this.directEncodeChunk(chunk);
      if (symbols.length === 0) continue;
      this.applyMerges(symbols);
      for (const symbol of symbols) {
        const id = this.tokenToId.get(symbol);
        if (id !== void 0) result.push(id);
      }
    }
    return result;
  }
  /** Split text into segments of special tokens and regular text. */
  splitOnSpecialTokens(text) {
    const segments = [];
    let remaining = text;
    while (remaining.length > 0) {
      let bestIdx = remaining.length;
      let bestToken = "";
      for (const st of this.specialTokens) {
        const idx = remaining.indexOf(st);
        if (idx >= 0 && idx < bestIdx) {
          bestIdx = idx;
          bestToken = st;
        }
      }
      if (bestToken) {
        if (bestIdx > 0) {
          segments.push({ text: remaining.slice(0, bestIdx), isSpecial: false });
        }
        segments.push({ text: bestToken, isSpecial: true });
        remaining = remaining.slice(bestIdx + bestToken.length);
      } else {
        segments.push({ text: remaining, isSpecial: false });
        break;
      }
    }
    return segments;
  }
  /** Decode token IDs back to text. */
  decode(tokenIds) {
    const parts = [];
    for (const id of tokenIds) {
      if (id === this.bosTokenId || id === this.eosTokenId || id === this.padTokenId) continue;
      let token = this.tokens[id];
      if (!token) continue;
      if (token.length === 6 && token.startsWith("<0x") && token.endsWith(">")) {
        const byte = parseInt(token.slice(3, 5), 16);
        if (!isNaN(byte)) {
          parts.push(String.fromCharCode(byte));
          continue;
        }
      }
      if (token.includes("\uFF5C")) {
        token = token.replaceAll("\uFF5C", "|");
      }
      parts.push(token);
    }
    let text = parts.join("");
    if (this.useByteEncoding) {
      return byteDecodeString(text);
    }
    return text.replaceAll("\u2581", " ");
  }
  /** Check if a token ID is an end-of-sequence token. */
  isEos(tokenId) {
    if (tokenId === this.eosTokenId) return true;
    const token = this.tokens[tokenId];
    return token === "<|endoftext|>" || token === "<|im_end|>" || token === "<\uFF5Cend\u2581of\u2581text\uFF5C>" || token === "<\uFF5Cim_end\uFF5C>" || token === "<|eot_id|>" || token === "<|end_of_text|>";
  }
  // Direct encoding for SentencePiece/Llama style vocabs
  directEncodeChunk(chunk) {
    const symbols = [];
    const encoder = new TextEncoder();
    const bytes = encoder.encode(chunk);
    let i = 0;
    while (i < bytes.length) {
      const charLen = utf8CharLength(bytes[i]);
      if (i + charLen <= bytes.length) {
        const ch = new TextDecoder().decode(bytes.slice(i, i + charLen));
        if (this.tokenToId.has(ch)) {
          symbols.push(ch);
          i += charLen;
          continue;
        }
      }
      symbols.push(`<0x${bytes[i].toString(16).toUpperCase().padStart(2, "0")}>`);
      i++;
    }
    return symbols;
  }
  // BPE merge algorithm
  applyMerges(symbols) {
    while (symbols.length > 1) {
      let bestRank = Infinity;
      let bestIdx = -1;
      for (let i = 0; i < symbols.length - 1; i++) {
        const key = `${symbols[i]} ${symbols[i + 1]}`;
        const rank = this.mergeRank.get(key);
        if (rank !== void 0 && rank < bestRank) {
          bestRank = rank;
          bestIdx = i;
        }
      }
      if (bestIdx < 0) break;
      symbols[bestIdx] = symbols[bestIdx] + symbols[bestIdx + 1];
      symbols.splice(bestIdx + 1, 1);
    }
  }
};
var BYTE_TO_UNICODE = buildByteToUnicode();
var UNICODE_TO_BYTE = buildUnicodeToByte();
function buildByteToUnicode() {
  const table = new Array(256);
  let n = 256;
  for (let i = 0; i < 256; i++) {
    if (i >= 33 && i <= 126 || i >= 161 && i <= 172 || i >= 174 && i <= 255) {
      table[i] = String.fromCharCode(i);
    } else {
      table[i] = String.fromCharCode(n);
      n++;
    }
  }
  return table;
}
function buildUnicodeToByte() {
  const map = /* @__PURE__ */ new Map();
  for (let i = 0; i < 256; i++) {
    map.set(BYTE_TO_UNICODE[i], i);
  }
  return map;
}
function byteEncodeChunk(chunk) {
  const encoder = new TextEncoder();
  const bytes = encoder.encode(chunk);
  return Array.from(bytes, (b) => BYTE_TO_UNICODE[b]);
}
function byteDecodeString(text) {
  const bytes = [];
  for (const ch of text) {
    const b = UNICODE_TO_BYTE.get(ch);
    if (b !== void 0) {
      bytes.push(b);
    } else {
      const encoder = new TextEncoder();
      bytes.push(...encoder.encode(ch));
    }
  }
  return new TextDecoder().decode(new Uint8Array(bytes));
}
function utf8CharLength(firstByte) {
  if (firstByte < 128) return 1;
  if (firstByte < 192) return 1;
  if (firstByte < 224) return 2;
  if (firstByte < 240) return 3;
  return 4;
}
function tokenizerFromGguf(metadata) {
  const tokens = metadata.get("tokenizer.ggml.tokens");
  const merges = metadata.get("tokenizer.ggml.merges");
  if (!tokens || !merges) throw new Error("Missing tokenizer metadata in GGUF");
  const bosTokenId = metadata.get("tokenizer.ggml.bos_token_id") ?? -1;
  const eosTokenId = metadata.get("tokenizer.ggml.eos_token_id") ?? -1;
  const padTokenId = metadata.get("tokenizer.ggml.padding_token_id") ?? -1;
  const model = metadata.get("tokenizer.ggml.model");
  const useByteEncoding = model === "gpt2";
  return new BpeTokenizer(tokens, merges, bosTokenId, eosTokenId, padTokenId, useByteEncoding);
}

// src/tokenizer/chat-template.ts
function tokenize(template) {
  const tokens = [];
  let i = 0;
  while (i < template.length) {
    if (template.startsWith("{%", i)) {
      const end = template.indexOf("%}", i + 2);
      if (end < 0) break;
      tokens.push({ type: "tag", value: template.substring(i + 2, end).trim() });
      i = end + 2;
    } else if (template.startsWith("{{", i)) {
      const end = template.indexOf("}}", i + 2);
      if (end < 0) break;
      tokens.push({ type: "expr", value: template.substring(i + 2, end).trim() });
      i = end + 2;
    } else if (template.startsWith("{#", i)) {
      const end = template.indexOf("#}", i + 2);
      i = end < 0 ? template.length : end + 2;
    } else {
      let next = template.length;
      for (const marker of ["{%", "{{", "{#"]) {
        const pos = template.indexOf(marker, i);
        if (pos >= 0 && pos < next) next = pos;
      }
      tokens.push({ type: "text", value: template.substring(i, next) });
      i = next;
    }
  }
  return tokens;
}
function resolve(expr, ctx) {
  expr = expr.trim();
  if (expr.startsWith("'") && expr.endsWith("'") || expr.startsWith('"') && expr.endsWith('"')) {
    return expr.slice(1, -1);
  }
  if (expr === "true" || expr === "True") return true;
  if (expr === "false" || expr === "False") return false;
  if (expr === "none" || expr === "None") return void 0;
  if (/^\d+$/.test(expr)) return parseInt(expr, 10);
  const parts = expr.split(".");
  let val = ctx;
  for (const part of parts) {
    if (val == null || typeof val !== "object") return void 0;
    val = val[part];
  }
  return val;
}
function evaluate(expr, ctx) {
  expr = expr.trim();
  if (expr.startsWith("not ")) {
    return !truthy(evaluate(expr.substring(4), ctx));
  }
  for (const op of [" or ", " and "]) {
    const idx = expr.lastIndexOf(op);
    if (idx > 0) {
      const left = evaluate(expr.substring(0, idx), ctx);
      const right = evaluate(expr.substring(idx + op.length), ctx);
      return op === " or " ? truthy(left) ? left : right : truthy(left) ? right : left;
    }
  }
  if (expr.endsWith(" is defined")) {
    const varName = expr.slice(0, -" is defined".length).trim();
    return resolve(varName, ctx) !== void 0;
  }
  if (expr.endsWith(" is not defined")) {
    const varName = expr.slice(0, -" is not defined".length).trim();
    return resolve(varName, ctx) === void 0;
  }
  for (const op of ["!=", "=="]) {
    const idx = expr.indexOf(op);
    if (idx > 0) {
      const left = evaluate(expr.substring(0, idx), ctx);
      const right = evaluate(expr.substring(idx + op.length), ctx);
      return op === "==" ? left === right : left !== right;
    }
  }
  if (expr.includes("|")) {
    const [base, ...filters] = expr.split("|");
    let val = evaluate(base, ctx);
    for (const f of filters) {
      const filter = f.trim();
      if (filter === "trim" && typeof val === "string") val = val.trim();
    }
    return val;
  }
  const bracketMatch = expr.match(/^(.+)\[(\d+)\]$/);
  if (bracketMatch) {
    const arr = resolve(bracketMatch[1], ctx);
    if (Array.isArray(arr)) return arr[parseInt(bracketMatch[2], 10)];
    return void 0;
  }
  if (expr.endsWith(".length") || expr.endsWith("|length")) {
    const base = expr.replace(/[.|]length$/, "");
    const val = resolve(base, ctx);
    if (Array.isArray(val)) return val.length;
    if (typeof val === "string") return val.length;
    return 0;
  }
  return resolve(expr, ctx);
}
function truthy(val) {
  if (val === void 0 || val === null || val === false || val === 0 || val === "") return false;
  if (Array.isArray(val) && val.length === 0) return false;
  return true;
}
function toString(val) {
  if (val === void 0 || val === null) return "";
  return String(val);
}
function execute(tokens, ctx) {
  let output = "";
  let i = 0;
  while (i < tokens.length) {
    const tok = tokens[i];
    if (tok.type === "text") {
      output += tok.value;
      i++;
    } else if (tok.type === "expr") {
      output += toString(evaluate(tok.value, ctx));
      i++;
    } else if (tok.type === "tag") {
      const tag = tok.value;
      if (tag.startsWith("for ")) {
        const forMatch = tag.match(/^for\s+(\w+)\s+in\s+(.+)$/);
        if (!forMatch) {
          i++;
          continue;
        }
        const [, varName, iterExpr] = forMatch;
        const iterable = evaluate(iterExpr, ctx);
        const items = Array.isArray(iterable) ? iterable : [];
        const body = [];
        let depth = 1;
        i++;
        while (i < tokens.length && depth > 0) {
          if (tokens[i].type === "tag") {
            if (tokens[i].value.startsWith("for ")) depth++;
            else if (tokens[i].value === "endfor") {
              depth--;
              if (depth === 0) {
                i++;
                break;
              }
            }
          }
          body.push(tokens[i]);
          i++;
        }
        for (let idx = 0; idx < items.length; idx++) {
          const loopCtx = {
            ...ctx,
            [varName]: items[idx],
            loop: {
              index0: idx,
              index: idx + 1,
              first: idx === 0,
              last: idx === items.length - 1,
              length: items.length
            }
          };
          output += execute(body, loopCtx);
        }
      } else if (tag.startsWith("if ")) {
        const branches = [];
        let currentCond = tag.substring(3).trim();
        let currentBody = [];
        let depth = 1;
        i++;
        while (i < tokens.length && depth > 0) {
          if (tokens[i].type === "tag") {
            const inner = tokens[i].value;
            if (inner.startsWith("if ")) {
              depth++;
              currentBody.push(tokens[i]);
            } else if (inner === "endif") {
              depth--;
              if (depth === 0) {
                branches.push({ cond: currentCond, body: currentBody });
                i++;
                break;
              }
              currentBody.push(tokens[i]);
            } else if (depth === 1 && inner.startsWith("elif ")) {
              branches.push({ cond: currentCond, body: currentBody });
              currentCond = inner.substring(5).trim();
              currentBody = [];
            } else if (depth === 1 && inner === "else") {
              branches.push({ cond: currentCond, body: currentBody });
              currentCond = null;
              currentBody = [];
            } else {
              currentBody.push(tokens[i]);
            }
          } else {
            currentBody.push(tokens[i]);
          }
          i++;
        }
        for (const branch of branches) {
          if (branch.cond === null || truthy(evaluate(branch.cond, ctx))) {
            output += execute(branch.body, ctx);
            break;
          }
        }
      } else if (tag.startsWith("set ")) {
        const setMatch = tag.match(/^set\s+(\w+)\s*=\s*(.+)$/);
        if (setMatch) {
          ctx[setMatch[1]] = evaluate(setMatch[2], ctx);
        }
        i++;
      } else {
        i++;
      }
    } else {
      i++;
    }
  }
  return output;
}
function applyTemplate(template, messages, options) {
  const ctx = {
    messages,
    bos_token: options?.bos_token ?? "",
    eos_token: options?.eos_token ?? "",
    add_generation_prompt: options?.add_generation_prompt ?? true
  };
  const tokens = tokenize(template);
  return execute(tokens, ctx);
}

// src/model/kv-cache.ts
var KvCache = class {
  device;
  numKvHeads;
  headDim;
  maxSeqLen;
  _seqLen = 0;
  kBuffer;
  vBuffer;
  constructor(device, buffers, numKvHeads, headDim, maxSeqLen, label) {
    this.device = device;
    this.numKvHeads = numKvHeads;
    this.headDim = headDim;
    this.maxSeqLen = maxSeqLen;
    const size = numKvHeads * maxSeqLen * headDim * 4;
    this.kBuffer = buffers.createBuffer(`${label}_k`, size);
    this.vBuffer = buffers.createBuffer(`${label}_v`, size);
  }
  get seqLen() {
    return this._seqLen;
  }
  set seqLen(val) {
    this._seqLen = val;
  }
  /**
   * Write K and V vectors for the current position.
   * k, v: [numKvHeads * headDim] float32
   */
  write(k, v, kSize, vSize) {
    const offset = this._seqLen * this.headDim * 4;
    const encoder = this.device.createCommandEncoder();
    for (let h = 0; h < this.numKvHeads; h++) {
      const srcOffset = h * this.headDim * 4;
      const dstOffset = (h * this.maxSeqLen * this.headDim + this._seqLen * this.headDim) * 4;
      encoder.copyBufferToBuffer(k, srcOffset, this.kBuffer, dstOffset, this.headDim * 4);
      encoder.copyBufferToBuffer(v, srcOffset, this.vBuffer, dstOffset, this.headDim * 4);
    }
    this.device.queue.submit([encoder.finish()]);
    this._seqLen++;
  }
  /** Reset cache for new sequence. */
  reset() {
    this._seqLen = 0;
  }
};

// src/model/llama-model.ts
function dequantizeToF32(buffer, type, elementCount) {
  const result = new Float32Array(elementCount);
  const bytes = new Uint8Array(buffer);
  const view = new DataView(buffer);
  if (type === 1 /* F16 */) {
    for (let i = 0; i < elementCount; i++) {
      result[i] = f16ToF32(view.getUint16(i * 2, true));
    }
    return result;
  }
  if (type === 8 /* Q8_0 */) {
    const blockCount = Math.ceil(elementCount / 32);
    for (let b = 0; b < blockCount; b++) {
      const blockOffset = b * 34;
      const scale = f16ToF32(view.getUint16(blockOffset, true));
      for (let q = 0; q < 32 && b * 32 + q < elementCount; q++) {
        const val = view.getInt8(blockOffset + 2 + q);
        result[b * 32 + q] = scale * val;
      }
    }
    return result;
  }
  if (type === 2 /* Q4_0 */) {
    const blockCount = Math.ceil(elementCount / 32);
    for (let b = 0; b < blockCount; b++) {
      const blockOffset = b * 18;
      const scale = f16ToF32(view.getUint16(blockOffset, true));
      for (let j = 0; j < 16; j++) {
        const byteVal = bytes[blockOffset + 2 + j];
        const lo = (byteVal & 15) - 8;
        const hi = (byteVal >> 4 & 15) - 8;
        const idx = b * 32;
        if (idx + j < elementCount) result[idx + j] = scale * lo;
        if (idx + j + 16 < elementCount) result[idx + j + 16] = scale * hi;
      }
    }
    return result;
  }
  if (type === 14 /* Q6_K */) {
    const QK = 256;
    const blockCount = Math.ceil(elementCount / QK);
    for (let b = 0; b < blockCount; b++) {
      const bo = b * 210;
      const qlOff = bo;
      const qhOff = bo + 128;
      const scOff = bo + 192;
      const dOff = bo + 208;
      const d = f16ToF32(view.getUint16(dOff, true));
      const outBase = b * QK;
      for (let n = 0; n < QK; n += 128) {
        for (let l = 0; l < 32; l++) {
          const is_ = n / 16;
          const qlIdx0 = qlOff + n / 2 + l;
          const qlIdx1 = qlOff + n / 2 + l + 32;
          const qhIdx = qhOff + n / 4 + l;
          const qhByte = bytes[qhIdx];
          const q1 = (bytes[qlIdx0] & 15 | (qhByte >> 0 & 3) << 4) - 32;
          const q2 = (bytes[qlIdx1] & 15 | (qhByte >> 2 & 3) << 4) - 32;
          const q3 = (bytes[qlIdx0] >> 4 | (qhByte >> 4 & 3) << 4) - 32;
          const q4 = (bytes[qlIdx1] >> 4 | (qhByte >> 6 & 3) << 4) - 32;
          const sc0 = view.getInt8(scOff + is_ + 0);
          const sc2 = view.getInt8(scOff + is_ + 2);
          const sc4 = view.getInt8(scOff + is_ + 4);
          const sc6 = view.getInt8(scOff + is_ + 6);
          const oi = outBase + n + l;
          if (oi < elementCount) result[oi] = d * sc0 * q1;
          if (oi + 32 < elementCount) result[oi + 32] = d * sc2 * q2;
          if (oi + 64 < elementCount) result[oi + 64] = d * sc4 * q3;
          if (oi + 96 < elementCount) result[oi + 96] = d * sc6 * q4;
        }
      }
    }
    return result;
  }
  if (type === 12 /* Q4_K */) {
    const QK = 256;
    const blockCount = Math.ceil(elementCount / QK);
    for (let b = 0; b < blockCount; b++) {
      const bo = b * 144;
      const d = f16ToF32(view.getUint16(bo, true));
      const dmin = f16ToF32(view.getUint16(bo + 2, true));
      const scalesOff = bo + 4;
      const minsOff = bo + 16;
      const qsOff = bo + 16;
      const outBase = b * QK;
      for (let j = 0; j < 128 && outBase + j * 2 < elementCount; j++) {
        const qByte = bytes[bo + 16 + j];
        const lo = qByte & 15;
        const hi = qByte >> 4;
        if (outBase + j < elementCount) result[outBase + j] = d * (lo - 8);
        if (outBase + j + 128 < elementCount) result[outBase + j + 128] = d * (hi - 8);
      }
    }
    return result;
  }
  if (type === 3 /* Q4_1 */) {
    const blockCount = Math.ceil(elementCount / 32);
    for (let b = 0; b < blockCount; b++) {
      const bo = b * 20;
      const delta = f16ToF32(view.getUint16(bo, true));
      const min = f16ToF32(view.getUint16(bo + 2, true));
      for (let j = 0; j < 16; j++) {
        const byteVal = bytes[bo + 4 + j];
        const lo = byteVal & 15;
        const hi = byteVal >> 4 & 15;
        const idx = b * 32;
        if (idx + j < elementCount) result[idx + j] = delta * lo + min;
        if (idx + j + 16 < elementCount) result[idx + j + 16] = delta * hi + min;
      }
    }
    return result;
  }
  if (type === 6 /* Q5_0 */) {
    const blockCount = Math.ceil(elementCount / 32);
    for (let b = 0; b < blockCount; b++) {
      const bo = b * 22;
      const scale = f16ToF32(view.getUint16(bo, true));
      const highBits = view.getUint32(bo + 2, true);
      for (let j = 0; j < 16; j++) {
        const byteVal = bytes[bo + 6 + j];
        const lo4 = byteVal & 15;
        const hi4 = byteVal >> 4 & 15;
        const loBit = highBits >> j & 1;
        const hiBit = highBits >> j + 16 & 1;
        const q5lo = lo4 | loBit << 4;
        const q5hi = hi4 | hiBit << 4;
        const idx = b * 32;
        if (idx + j < elementCount) result[idx + j] = scale * (q5lo - 16);
        if (idx + j + 16 < elementCount) result[idx + j + 16] = scale * (q5hi - 16);
      }
    }
    return result;
  }
  if (type === 7 /* Q5_1 */) {
    const blockCount = Math.ceil(elementCount / 32);
    for (let b = 0; b < blockCount; b++) {
      const bo = b * 24;
      const delta = f16ToF32(view.getUint16(bo, true));
      const min = f16ToF32(view.getUint16(bo + 2, true));
      const highBits = view.getUint32(bo + 4, true);
      for (let j = 0; j < 16; j++) {
        const byteVal = bytes[bo + 8 + j];
        const lo4 = byteVal & 15;
        const hi4 = byteVal >> 4 & 15;
        const loBit = highBits >> j & 1;
        const hiBit = highBits >> j + 16 & 1;
        const q5lo = lo4 | loBit << 4;
        const q5hi = hi4 | hiBit << 4;
        const idx = b * 32;
        if (idx + j < elementCount) result[idx + j] = delta * q5lo + min;
        if (idx + j + 16 < elementCount) result[idx + j + 16] = delta * q5hi + min;
      }
    }
    return result;
  }
  throw new Error(`Unsupported dequant type: ${GgmlType[type]} (${type})`);
}
function f16ToF32(bits) {
  const sign = bits >> 15 & 1;
  const exp = bits >> 10 & 31;
  const mant = bits & 1023;
  if (exp === 0) {
    if (mant === 0) return sign ? -0 : 0;
    const f2 = mant / 1024 * Math.pow(2, -14);
    return sign ? -f2 : f2;
  }
  if (exp === 31) return sign ? -Infinity : Infinity;
  const f = (1 + mant / 1024) * Math.pow(2, exp - 15);
  return sign ? -f : f;
}
var GPU_MATMUL_TYPES = /* @__PURE__ */ new Set([0 /* F32 */, 2 /* Q4_0 */, 8 /* Q8_0 */]);
var LlamaModel = class {
  compute;
  info;
  weights;
  kvCaches;
  // one per layer
  // Working buffers
  hidden;
  residual;
  normed;
  qBuf;
  kBuf;
  vBuf;
  attnOut;
  gateBuf;
  upBuf;
  ffnOut;
  temp;
  // temp buffer for in-place add workaround
  logits;
  constructor(compute, info) {
    this.compute = compute;
    this.info = info;
  }
  get embeddingDim() {
    return this.info.embeddingLength;
  }
  get numLayers() {
    return this.info.blockCount;
  }
  get numHeads() {
    return this.info.headCount;
  }
  get numKvHeads() {
    return this.info.headCountKv || this.info.headCount;
  }
  get headDim() {
    return this.embeddingDim / this.numHeads;
  }
  get ffnDim() {
    return this.info.feedForwardLength;
  }
  get vocabSize() {
    return this.info.vocabSize;
  }
  get contextLength() {
    return this.info.contextLength;
  }
  get ropeTheta() {
    return this.info.ropeFreqBase;
  }
  get rmsNormEps() {
    return this.info.rmsNormEps;
  }
  /**
   * Upload tensor data to GPU and initialize working buffers.
   */
  async initWeights(tensorMap) {
    const { compute, info } = this;
    const arch = info.architecture;
    const uploadAsF32 = (name) => {
      const tensor = tensorMap.get(name);
      if (!tensor) throw new Error(`Missing tensor: ${name}`);
      if (tensor.info.type === 0 /* F32 */) {
        return compute.buffers.createBufferWithData(name, tensor.buffer);
      }
      const f32Data = dequantizeToF32(tensor.buffer, tensor.info.type, tensor.info.elementCount);
      return compute.buffers.createBufferWithData(name, f32Data.buffer);
    };
    const uploadWeight = (name) => {
      const tensor = tensorMap.get(name);
      if (!tensor) throw new Error(`Missing tensor: ${name}`);
      if (GPU_MATMUL_TYPES.has(tensor.info.type)) {
        return { buffer: compute.buffers.createBufferWithData(name, tensor.buffer), type: tensor.info.type };
      }
      const f32Data = dequantizeToF32(tensor.buffer, tensor.info.type, tensor.info.elementCount);
      return { buffer: compute.buffers.createBufferWithData(name, f32Data.buffer), type: 0 /* F32 */ };
    };
    const tokenEmbedding = uploadWeight("token_embd.weight");
    let outputWeight;
    if (tensorMap.has("output.weight")) {
      outputWeight = uploadWeight("output.weight");
    } else {
      outputWeight = tokenEmbedding;
    }
    const outputNorm = uploadAsF32("output_norm.weight");
    const tryUploadAsF32 = (name) => {
      return tensorMap.has(name) ? uploadAsF32(name) : void 0;
    };
    const layers = [];
    for (let i = 0; i < this.numLayers; i++) {
      layers.push({
        attnNorm: uploadAsF32(`blk.${i}.attn_norm.weight`),
        q: uploadWeight(`blk.${i}.attn_q.weight`),
        k: uploadWeight(`blk.${i}.attn_k.weight`),
        v: uploadWeight(`blk.${i}.attn_v.weight`),
        o: uploadWeight(`blk.${i}.attn_output.weight`),
        qBias: tryUploadAsF32(`blk.${i}.attn_q.bias`),
        kBias: tryUploadAsF32(`blk.${i}.attn_k.bias`),
        vBias: tryUploadAsF32(`blk.${i}.attn_v.bias`),
        postAttnNorm: uploadAsF32(`blk.${i}.ffn_norm.weight`),
        gateProj: uploadWeight(`blk.${i}.ffn_gate.weight`),
        upProj: uploadWeight(`blk.${i}.ffn_up.weight`),
        downProj: uploadWeight(`blk.${i}.ffn_down.weight`)
      });
    }
    this.weights = { tokenEmbedding, outputNorm, output: outputWeight, layers };
    const E = this.embeddingDim;
    const F = this.ffnDim;
    const H = this.numHeads * this.headDim;
    const KV = this.numKvHeads * this.headDim;
    this.hidden = compute.buffers.createBuffer("hidden", E * 4);
    this.residual = compute.buffers.createBuffer("residual", E * 4);
    this.normed = compute.buffers.createBuffer("normed", E * 4);
    this.qBuf = compute.buffers.createBuffer("q_proj", H * 4);
    this.kBuf = compute.buffers.createBuffer("k_proj", KV * 4);
    this.vBuf = compute.buffers.createBuffer("v_proj", KV * 4);
    this.attnOut = compute.buffers.createBuffer("attn_out", E * 4);
    this.gateBuf = compute.buffers.createBuffer("gate", F * 4);
    this.upBuf = compute.buffers.createBuffer("up", F * 4);
    this.ffnOut = compute.buffers.createBuffer("ffn_out", F * 4);
    this.temp = compute.buffers.createBuffer("temp", E * 4);
    this.logits = compute.buffers.createBuffer("logits", this.vocabSize * 4);
    const maxCtx = Math.min(this.contextLength, 4096);
    this.kvCaches = [];
    for (let i = 0; i < this.numLayers; i++) {
      this.kvCaches.push(new KvCache(
        compute.device,
        compute.buffers,
        this.numKvHeads,
        this.headDim,
        maxCtx,
        `kv_L${i}`
      ));
    }
  }
  /** Reset the KV cache for a new conversation. */
  resetCache() {
    for (const kv of this.kvCaches) kv.reset();
  }
  /** Get current sequence position in KV cache. */
  get position() {
    return this.kvCaches[0].seqLen;
  }
  /**
   * Forward pass for a single token. Returns logits buffer on GPU.
   */
  forward(tokenId) {
    const { compute, weights } = this;
    const E = this.embeddingDim;
    if (weights.tokenEmbedding.type === 8 /* Q8_0 */) {
      compute.embeddingQ8(weights.tokenEmbedding.buffer, this.hidden, tokenId, E);
    } else {
      compute.embedding(weights.tokenEmbedding.buffer, this.hidden, tokenId, E);
    }
    for (let layer = 0; layer < this.numLayers; layer++) {
      const lw = weights.layers[layer];
      compute.copyAndRmsNorm(this.hidden, lw.attnNorm, this.normed, this.residual, E, this.rmsNormEps);
      compute.matmul(lw.q.buffer, this.normed, this.qBuf, this.numHeads * this.headDim, E, lw.q.type);
      compute.matmul(lw.k.buffer, this.normed, this.kBuf, this.numKvHeads * this.headDim, E, lw.k.type);
      compute.matmul(lw.v.buffer, this.normed, this.vBuf, this.numKvHeads * this.headDim, E, lw.v.type);
      if (lw.qBias) compute.addBias(this.qBuf, lw.qBias, this.numHeads * this.headDim);
      if (lw.kBias) compute.addBias(this.kBuf, lw.kBias, this.numKvHeads * this.headDim);
      if (lw.vBias) compute.addBias(this.vBuf, lw.vBias, this.numKvHeads * this.headDim);
      const kvCache = this.kvCaches[layer];
      const pos = kvCache.seqLen;
      compute.rope(this.qBuf, this.headDim, this.headDim, pos, this.ropeTheta, this.numHeads * this.headDim);
      compute.rope(this.kBuf, this.headDim, this.headDim, pos, this.ropeTheta, this.numKvHeads * this.headDim);
      kvCache.write(this.kBuf, this.vBuf, this.numKvHeads * this.headDim * 4, this.numKvHeads * this.headDim * 4);
      compute.attention(
        this.qBuf,
        kvCache.kBuffer,
        kvCache.vBuffer,
        this.attnOut,
        this.numHeads,
        this.numKvHeads,
        this.headDim,
        kvCache.seqLen,
        kvCache.maxSeqLen
      );
      compute.matmul(lw.o.buffer, this.attnOut, this.temp, E, this.numHeads * this.headDim, lw.o.type);
      compute.add(this.temp, this.residual, this.hidden, E);
      compute.copyAndRmsNorm(this.hidden, lw.postAttnNorm, this.normed, this.residual, E, this.rmsNormEps);
      compute.matmul(lw.gateProj.buffer, this.normed, this.gateBuf, this.ffnDim, E, lw.gateProj.type);
      compute.matmul(lw.upProj.buffer, this.normed, this.upBuf, this.ffnDim, E, lw.upProj.type);
      compute.siluMul(this.gateBuf, this.upBuf, this.ffnOut, this.ffnDim);
      compute.matmul(lw.downProj.buffer, this.ffnOut, this.temp, E, this.ffnDim, lw.downProj.type);
      compute.add(this.temp, this.residual, this.hidden, E);
    }
    compute.rmsNorm(this.hidden, weights.outputNorm, this.normed, E, this.rmsNormEps);
    compute.matmul(weights.output.buffer, this.normed, this.logits, this.vocabSize, E, weights.output.type);
    return this.logits;
  }
  /**
   * Batched prefill: process all prompt tokens at once per layer.
   * Much faster than calling forward() N times for long prompts.
   * Returns logits buffer for the LAST token only.
   */
  forwardPrefill(tokenIds) {
    const { compute, weights } = this;
    const E = this.embeddingDim;
    const N = tokenIds.length;
    if (N === 0) return this.logits;
    if (N === 1) return this.forward(tokenIds[0]);
    const H = this.numHeads * this.headDim;
    const KV = this.numKvHeads * this.headDim;
    const F = this.ffnDim;
    const bHidden = compute.buffers.createBuffer("b_hidden", N * E * 4);
    const bResidual = compute.buffers.createBuffer("b_residual", N * E * 4);
    const bNormed = compute.buffers.createBuffer("b_normed", N * E * 4);
    const bQ = compute.buffers.createBuffer("b_q", N * H * 4);
    const bK = compute.buffers.createBuffer("b_k", N * KV * 4);
    const bV = compute.buffers.createBuffer("b_v", N * KV * 4);
    const bAttnOut = compute.buffers.createBuffer("b_attn", N * E * 4);
    const bGate = compute.buffers.createBuffer("b_gate", N * F * 4);
    const bUp = compute.buffers.createBuffer("b_up", N * F * 4);
    const bFfnOut = compute.buffers.createBuffer("b_ffn", N * F * 4);
    const bTemp = compute.buffers.createBuffer("b_temp", N * E * 4);
    const tokenIdBuf = compute.buffers.createBuffer("b_tokens", N * 4);
    const positionBuf = compute.buffers.createBuffer("b_positions", N * 4);
    const startPos = this.kvCaches[0].seqLen;
    const posData = new Uint32Array(N);
    for (let i = 0; i < N; i++) posData[i] = startPos + i;
    compute.device.queue.writeBuffer(tokenIdBuf, 0, new Uint32Array(tokenIds));
    compute.device.queue.writeBuffer(positionBuf, 0, posData);
    if (weights.tokenEmbedding.type === 0 /* F32 */) {
      compute.embeddingBatch(weights.tokenEmbedding.buffer, tokenIdBuf, bHidden, N, E);
    } else {
      for (let i = 0; i < N; i++) {
        compute.embeddingQ8(weights.tokenEmbedding.buffer, this.hidden, tokenIds[i], E);
        compute.copyBufferRegion(this.hidden, 0, bHidden, i * E * 4, E * 4);
      }
    }
    for (let layer = 0; layer < this.numLayers; layer++) {
      const lw = weights.layers[layer];
      const kvCache = this.kvCaches[layer];
      compute.copyBuffer(bHidden, bResidual, N * E * 4);
      compute.rmsNormBatch(bHidden, lw.attnNorm, bNormed, E, N, this.rmsNormEps);
      compute.matmulBatch(lw.q.buffer, bNormed, bQ, H, E, N, lw.q.type);
      compute.matmulBatch(lw.k.buffer, bNormed, bK, KV, E, N, lw.k.type);
      compute.matmulBatch(lw.v.buffer, bNormed, bV, KV, E, N, lw.v.type);
      if (lw.qBias) {
        for (let n = 0; n < N; n++) {
        }
      }
      compute.ropeBatch(bQ, positionBuf, this.numHeads, this.headDim, N, this.ropeTheta);
      compute.ropeBatch(bK, positionBuf, this.numKvHeads, this.headDim, N, this.ropeTheta);
      compute.kvCacheWriteBatch(
        bK,
        bV,
        kvCache.kBuffer,
        kvCache.vBuffer,
        this.numKvHeads,
        this.headDim,
        kvCache.maxSeqLen,
        startPos,
        N
      );
      kvCache.seqLen = startPos + N;
      compute.attentionBatch(
        bQ,
        kvCache.kBuffer,
        kvCache.vBuffer,
        bAttnOut,
        this.numHeads,
        this.numKvHeads,
        this.headDim,
        kvCache.maxSeqLen,
        startPos,
        N
      );
      compute.matmulBatch(lw.o.buffer, bAttnOut, bTemp, E, H, N, lw.o.type);
      compute.add(bTemp, bResidual, bHidden, N * E);
      compute.copyBuffer(bHidden, bResidual, N * E * 4);
      compute.rmsNormBatch(bHidden, lw.postAttnNorm, bNormed, E, N, this.rmsNormEps);
      compute.matmulBatch(lw.gateProj.buffer, bNormed, bGate, F, E, N, lw.gateProj.type);
      compute.matmulBatch(lw.upProj.buffer, bNormed, bUp, F, E, N, lw.upProj.type);
      compute.siluMul(bGate, bUp, bFfnOut, N * F);
      compute.matmulBatch(lw.downProj.buffer, bFfnOut, bTemp, E, F, N, lw.downProj.type);
      compute.add(bTemp, bResidual, bHidden, N * E);
    }
    compute.copyBufferRegion(bHidden, (N - 1) * E * 4, this.hidden, 0, E * 4);
    compute.rmsNorm(this.hidden, weights.outputNorm, this.normed, E, this.rmsNormEps);
    compute.matmul(weights.output.buffer, this.normed, this.logits, this.vocabSize, E, weights.output.type);
    bHidden.destroy();
    bResidual.destroy();
    bNormed.destroy();
    bQ.destroy();
    bK.destroy();
    bV.destroy();
    bAttnOut.destroy();
    bGate.destroy();
    bUp.destroy();
    bFfnOut.destroy();
    bTemp.destroy();
    tokenIdBuf.destroy();
    positionBuf.destroy();
    return this.logits;
  }
  // ── CPU forward pass (for verification / fallback) ──────────────────
  cpuWeights = null;
  /** Store CPU-side F32 weights for CPU forward pass. */
  storeCpuWeights(tensorMap) {
    const dq = (name) => {
      const t = tensorMap.get(name);
      if (t.info.type === 0 /* F32 */) return new Float32Array(t.buffer);
      return dequantizeToF32(t.buffer, t.info.type, t.info.elementCount);
    };
    const layers = [];
    for (let i = 0; i < this.numLayers; i++) {
      layers.push({
        attnNorm: dq(`blk.${i}.attn_norm.weight`),
        q: dq(`blk.${i}.attn_q.weight`),
        k: dq(`blk.${i}.attn_k.weight`),
        v: dq(`blk.${i}.attn_v.weight`),
        o: dq(`blk.${i}.attn_output.weight`),
        postAttnNorm: dq(`blk.${i}.ffn_norm.weight`),
        gate: dq(`blk.${i}.ffn_gate.weight`),
        up: dq(`blk.${i}.ffn_up.weight`),
        down: dq(`blk.${i}.ffn_down.weight`)
      });
    }
    this.cpuWeights = {
      embedding: dq("token_embd.weight"),
      outputNorm: dq("output_norm.weight"),
      output: dq(tensorMap.has("output.weight") ? "output.weight" : "token_embd.weight"),
      layers
    };
  }
  getCpuState() {
    const maxSeqLen = 512;
    return {
      kvK: Array.from({ length: this.numLayers }, () => new Float32Array(this.numKvHeads * maxSeqLen * this.headDim)),
      kvV: Array.from({ length: this.numLayers }, () => new Float32Array(this.numKvHeads * maxSeqLen * this.headDim)),
      seqLen: 0,
      maxSeqLen,
      hidden: new Float32Array(this.embeddingDim)
    };
  }
  async cpuForward(tokenId, state) {
    const E = this.embeddingDim;
    const w = this.cpuWeights;
    const h = new Float32Array(E);
    for (let i = 0; i < E; i++) h[i] = w.embedding[tokenId * E + i];
    for (let layer = 0; layer < this.numLayers; layer++) {
      const lw = w.layers[layer];
      const residual = new Float32Array(h);
      const normed = cpuRmsNorm(h, lw.attnNorm, E, this.rmsNormEps);
      const qP = cpuMatvec(lw.q, normed, this.numHeads * this.headDim, E);
      const kP = cpuMatvec(lw.k, normed, this.numKvHeads * this.headDim, E);
      const vP = cpuMatvec(lw.v, normed, this.numKvHeads * this.headDim, E);
      cpuRope(qP, this.headDim, state.seqLen, this.ropeTheta);
      cpuRope(kP, this.headDim, state.seqLen, this.ropeTheta);
      for (let kh = 0; kh < this.numKvHeads; kh++) {
        for (let d = 0; d < this.headDim; d++) {
          state.kvK[layer][kh * state.maxSeqLen * this.headDim + state.seqLen * this.headDim + d] = kP[kh * this.headDim + d];
          state.kvV[layer][kh * state.maxSeqLen * this.headDim + state.seqLen * this.headDim + d] = vP[kh * this.headDim + d];
        }
      }
      const curSeqLen = state.seqLen + 1;
      const scale = 1 / Math.sqrt(this.headDim);
      const headsPerGroup = this.numHeads / this.numKvHeads;
      const attnOut = new Float32Array(this.numHeads * this.headDim);
      for (let head = 0; head < this.numHeads; head++) {
        const kvHead = Math.floor(head / headsPerGroup);
        const scores = new Float32Array(curSeqLen);
        for (let pos = 0; pos < curSeqLen; pos++) {
          let dot = 0;
          for (let d = 0; d < this.headDim; d++)
            dot += qP[head * this.headDim + d] * state.kvK[layer][kvHead * state.maxSeqLen * this.headDim + pos * this.headDim + d];
          scores[pos] = dot * scale;
        }
        let maxS = -Infinity;
        for (let i = 0; i < curSeqLen; i++) maxS = Math.max(maxS, scores[i]);
        let sumE = 0;
        for (let i = 0; i < curSeqLen; i++) {
          scores[i] = Math.exp(scores[i] - maxS);
          sumE += scores[i];
        }
        for (let i = 0; i < curSeqLen; i++) scores[i] /= sumE;
        for (let d = 0; d < this.headDim; d++) {
          let acc = 0;
          for (let pos = 0; pos < curSeqLen; pos++)
            acc += scores[pos] * state.kvV[layer][kvHead * state.maxSeqLen * this.headDim + pos * this.headDim + d];
          attnOut[head * this.headDim + d] = acc;
        }
      }
      const oP = cpuMatvec(lw.o, attnOut, E, this.numHeads * this.headDim);
      for (let i = 0; i < E; i++) h[i] = oP[i] + residual[i];
      await new Promise((r) => setTimeout(r, 0));
      const residual2 = new Float32Array(h);
      const normed2 = cpuRmsNorm(h, lw.postAttnNorm, E, this.rmsNormEps);
      const gateOut = cpuMatvec(lw.gate, normed2, this.ffnDim, E);
      const upOut = cpuMatvec(lw.up, normed2, this.ffnDim, E);
      const ffnOut = new Float32Array(this.ffnDim);
      for (let i = 0; i < this.ffnDim; i++)
        ffnOut[i] = gateOut[i] / (1 + Math.exp(-gateOut[i])) * upOut[i];
      const downOut = cpuMatvec(lw.down, ffnOut, E, this.ffnDim);
      for (let i = 0; i < E; i++) h[i] = downOut[i] + residual2[i];
      await new Promise((r) => setTimeout(r, 0));
    }
    state.hidden.set(h);
    state.seqLen++;
  }
  cpuGetLogits(state) {
    const w = this.cpuWeights;
    const normed = cpuRmsNorm(state.hidden, w.outputNorm, this.embeddingDim, this.rmsNormEps);
    return cpuMatvec(w.output, normed, this.vocabSize, this.embeddingDim);
  }
  /**
   * Read logits back to CPU for sampling.
   */
  async readLogits() {
    const logits = await this.compute.readBuffer(this.logits, this.vocabSize * 4);
    this.compute.cleanupParams();
    return logits;
  }
};
function cpuRmsNorm(input, weight, n, eps) {
  let sumSq = 0;
  for (let i = 0; i < n; i++) sumSq += input[i] * input[i];
  const rms = Math.sqrt(sumSq / n + eps);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = input[i] / rms * weight[i];
  return out;
}
function cpuMatvec(weights, input, M, K) {
  const out = new Float32Array(M);
  for (let row = 0; row < M; row++) {
    let sum = 0;
    const off = row * K;
    for (let k = 0; k < K; k++) sum += weights[off + k] * input[k];
    out[row] = sum;
  }
  return out;
}
function cpuRope(data, headDim, position, theta) {
  const nPairs = data.length / 2;
  for (let pi = 0; pi < nPairs; pi++) {
    const hp = pi % (headDim / 2);
    const dimFrac = hp * 2 / headDim;
    const freq = 1 / Math.pow(theta, dimFrac);
    const angle = position * freq;
    const c = Math.cos(angle);
    const s = Math.sin(angle);
    const i0 = pi * 2, i1 = i0 + 1;
    const x = data[i0], y = data[i1];
    data[i0] = x * c - y * s;
    data[i1] = x * s + y * c;
  }
}

// src/model/qwen35-model.ts
function dequantizeToF322(buffer, type, elementCount) {
  const result = new Float32Array(elementCount);
  const bytes = new Uint8Array(buffer);
  const view = new DataView(buffer);
  if (type === 1 /* F16 */) {
    for (let i = 0; i < elementCount; i++) result[i] = f16ToF322(view.getUint16(i * 2, true));
    return result;
  }
  if (type === 8 /* Q8_0 */) {
    const blockCount = Math.ceil(elementCount / 32);
    for (let b = 0; b < blockCount; b++) {
      const bo = b * 34;
      const scale = f16ToF322(view.getUint16(bo, true));
      for (let q = 0; q < 32 && b * 32 + q < elementCount; q++) {
        result[b * 32 + q] = scale * view.getInt8(bo + 2 + q);
      }
    }
    return result;
  }
  if (type === 0 /* F32 */) {
    return new Float32Array(buffer.slice(0));
  }
  throw new Error(`Unsupported dequant type in Qwen35: ${GgmlType[type]} (${type})`);
}
function f16ToF322(bits) {
  const sign = bits >> 15 & 1;
  const exp = bits >> 10 & 31;
  const mant = bits & 1023;
  if (exp === 0) {
    if (mant === 0) return sign ? -0 : 0;
    return (sign ? -1 : 1) * (mant / 1024) * Math.pow(2, -14);
  }
  if (exp === 31) return sign ? -Infinity : Infinity;
  return (sign ? -1 : 1) * (1 + mant / 1024) * Math.pow(2, exp - 15);
}
var GPU_MATMUL_TYPES2 = /* @__PURE__ */ new Set([0 /* F32 */, 2 /* Q4_0 */, 8 /* Q8_0 */]);
var Qwen35Model = class {
  compute;
  info;
  weights;
  qkvDims;
  // per-layer QKV output dimension (from weight)
  kvCaches;
  // null for DeltaNet layers
  deltaStates;
  // null for standard attn layers
  // Working buffers
  hidden;
  residual;
  normed;
  qBuf;
  kBuf;
  vBuf;
  attnOut;
  gateBuf;
  upBuf;
  ffnOut;
  temp;
  logits;
  // Standard attention GPU buffers (for gated Q)
  qAttnBuf;
  qGateBufAttn;
  // DeltaNet-specific GPU buffers
  qTiledBuf;
  // tiled Q for deltanet [numVHeads * headDim]
  kTiledBuf;
  // tiled K for deltanet [numVHeads * headDim]
  qkvBuf;
  ssmGateBuf;
  ssmOutputBuf;
  alphaBuf;
  betaBuf;
  decayBuf;
  betaValBuf;
  // Config
  fullAttnInterval;
  ssmInnerSize;
  ssmStateSize;
  ssmGroupCount;
  ssmConvKernel;
  ssmNumVHeads;
  ssmHeadDim;
  constructor(compute, info) {
    this.compute = compute;
    this.info = info;
    const prefix = info.architecture;
    const meta = info.metadata;
    this.fullAttnInterval = meta.get(`${prefix}.full_attention_interval`) ?? 4;
    this.ssmInnerSize = meta.get(`${prefix}.ssm.inner_size`) ?? 2048;
    this.ssmStateSize = meta.get(`${prefix}.ssm.state_size`) ?? 128;
    this.ssmGroupCount = meta.get(`${prefix}.ssm.group_count`) ?? 16;
    this.ssmConvKernel = meta.get(`${prefix}.ssm.conv_kernel`) ?? 4;
    this.ssmNumVHeads = 0;
    this.ssmHeadDim = this.ssmStateSize;
  }
  get embeddingDim() {
    return this.info.embeddingLength;
  }
  get numLayers() {
    return this.info.blockCount;
  }
  get numHeads() {
    return this.info.headCount;
  }
  get numKvHeads() {
    return this.info.headCountKv || this.info.headCount;
  }
  get headDim() {
    return this.ssmHeadDim;
  }
  /** Per-head key/value dimension for standard attention */
  get keyLength() {
    return this.info.metadata.get(`${this.info.architecture}.attention.key_length`) || this.embeddingDim / this.numHeads;
  }
  get valueLength() {
    return this.info.metadata.get(`${this.info.architecture}.attention.value_length`) || this.keyLength;
  }
  /** Whether Q projection is gated (Q dim = 2 × keyLength per head) */
  get hasGatedQ() {
    const qTensor = this.info.tensors.find((t) => t.name === "blk.3.attn_q.weight");
    if (!qTensor) return false;
    const qRows = qTensor.elementCount / this.embeddingDim;
    const expectedQ = this.numHeads * this.keyLength;
    return qRows > expectedQ * 1.5;
  }
  get ffnDim() {
    return this.info.feedForwardLength;
  }
  get vocabSize() {
    return this.info.vocabSize;
  }
  get ropeTheta() {
    return this.info.ropeFreqBase;
  }
  get ropeDim() {
    return this.info.metadata.get(`${this.info.architecture}.rope.dimension_count`) || this.headDim;
  }
  get rmsNormEps() {
    return this.info.rmsNormEps;
  }
  get position() {
    return this._position;
  }
  _position = 0;
  isStandardAttention(layer) {
    return (layer + 1) % this.fullAttnInterval === 0;
  }
  async initWeights(tensorMap) {
    const { compute } = this;
    const uploadAsF32 = (name) => {
      const tensor = tensorMap.get(name);
      if (!tensor) throw new Error(`Missing tensor: ${name}`);
      if (tensor.info.type === 0 /* F32 */) return compute.buffers.createBufferWithData(name, tensor.buffer);
      const f32 = dequantizeToF322(tensor.buffer, tensor.info.type, tensor.info.elementCount);
      return compute.buffers.createBufferWithData(name, f32.buffer);
    };
    const uploadWeight = (name) => {
      const tensor = tensorMap.get(name);
      if (!tensor) throw new Error(`Missing tensor: ${name}`);
      if (GPU_MATMUL_TYPES2.has(tensor.info.type)) {
        return { buffer: compute.buffers.createBufferWithData(name, tensor.buffer), type: tensor.info.type };
      }
      const f32 = dequantizeToF322(tensor.buffer, tensor.info.type, tensor.info.elementCount);
      return { buffer: compute.buffers.createBufferWithData(name, f32.buffer), type: 0 /* F32 */ };
    };
    const getF32 = (name) => {
      const tensor = tensorMap.get(name);
      if (!tensor) throw new Error(`Missing tensor: ${name}`);
      if (tensor.info.type === 0 /* F32 */) return new Float32Array(tensor.buffer.slice(0));
      return dequantizeToF322(tensor.buffer, tensor.info.type, tensor.info.elementCount);
    };
    const tryUploadAsF32 = (name) => {
      return tensorMap.has(name) ? uploadAsF32(name) : void 0;
    };
    const alpha0 = tensorMap.get("blk.0.ssm_alpha.weight");
    if (alpha0) {
      this.ssmNumVHeads = alpha0.info.elementCount / this.embeddingDim;
    }
    const tokenEmbedding = uploadWeight("token_embd.weight");
    const output = tensorMap.has("output.weight") ? uploadWeight("output.weight") : tokenEmbedding;
    const outputNorm = uploadAsF32("output_norm.weight");
    this.qkvDims = [];
    const layers = [];
    for (let i = 0; i < this.numLayers; i++) {
      const shared = {
        postAttnNorm: uploadAsF32(`blk.${i}.post_attention_norm.weight`),
        gateProj: uploadWeight(`blk.${i}.ffn_gate.weight`),
        upProj: uploadWeight(`blk.${i}.ffn_up.weight`),
        downProj: uploadWeight(`blk.${i}.ffn_down.weight`)
      };
      if (this.isStandardAttention(i)) {
        this.qkvDims.push(0);
        layers.push({
          kind: "standard",
          attnNorm: uploadAsF32(`blk.${i}.attn_norm.weight`),
          q: uploadWeight(`blk.${i}.attn_q.weight`),
          k: uploadWeight(`blk.${i}.attn_k.weight`),
          v: uploadWeight(`blk.${i}.attn_v.weight`),
          o: uploadWeight(`blk.${i}.attn_output.weight`),
          qNorm: tryUploadAsF32(`blk.${i}.attn_q_norm.weight`),
          kNorm: tryUploadAsF32(`blk.${i}.attn_k_norm.weight`),
          ...shared
        });
      } else {
        const qkvTensor = tensorMap.get(`blk.${i}.attn_qkv.weight`);
        this.qkvDims.push(qkvTensor.info.elementCount / this.embeddingDim);
        layers.push({
          kind: "deltanet",
          attnNorm: uploadAsF32(`blk.${i}.attn_norm.weight`),
          qkv: uploadWeight(`blk.${i}.attn_qkv.weight`),
          gate: uploadWeight(`blk.${i}.attn_gate.weight`),
          ssmA: uploadAsF32(`blk.${i}.ssm_a`),
          ssmAlpha: uploadWeight(`blk.${i}.ssm_alpha.weight`),
          ssmBeta: uploadWeight(`blk.${i}.ssm_beta.weight`),
          ssmConv1d: uploadAsF32(`blk.${i}.ssm_conv1d.weight`),
          ssmDtBias: uploadAsF32(`blk.${i}.ssm_dt.bias`),
          ssmNorm: uploadAsF32(`blk.${i}.ssm_norm.weight`),
          ssmOut: uploadWeight(`blk.${i}.ssm_out.weight`),
          ...shared
        });
      }
    }
    this.weights = { tokenEmbedding, outputNorm, output, layers };
    const E = this.embeddingDim;
    const F = this.ffnDim;
    const qFullDim = this.hasGatedQ ? this.numHeads * this.keyLength * 2 : this.numHeads * this.keyLength;
    const kDim = this.numKvHeads * this.keyLength;
    const vDim = this.numKvHeads * this.valueLength;
    const H = Math.max(this.numHeads * this.headDim, qFullDim, this.ssmInnerSize);
    const KV = Math.max(this.numKvHeads * this.headDim, kDim, this.ssmInnerSize);
    this.hidden = compute.buffers.createBuffer("hidden", E * 4);
    this.residual = compute.buffers.createBuffer("residual", E * 4);
    this.normed = compute.buffers.createBuffer("normed", E * 4);
    this.qBuf = compute.buffers.createBuffer("q_proj", H * 4);
    this.kBuf = compute.buffers.createBuffer("k_proj", KV * 4);
    this.vBuf = compute.buffers.createBuffer("v_proj", KV * 4);
    this.attnOut = compute.buffers.createBuffer("attn_out", Math.max(E, this.numHeads * this.valueLength) * 4);
    this.gateBuf = compute.buffers.createBuffer("gate", F * 4);
    this.upBuf = compute.buffers.createBuffer("up", F * 4);
    this.ffnOut = compute.buffers.createBuffer("ffn_out", F * 4);
    this.temp = compute.buffers.createBuffer("temp", E * 4);
    this.logits = compute.buffers.createBuffer("logits", this.vocabSize * 4);
    const qAttnDim = this.numHeads * this.keyLength;
    this.qAttnBuf = compute.buffers.createBuffer("q_attn", qAttnDim * 4);
    this.qGateBufAttn = compute.buffers.createBuffer("q_gate_attn", qAttnDim * 4);
    const maxQkvDim = Math.max(...this.qkvDims, this.ssmInnerSize * 3);
    const numVHeads = this.ssmNumVHeads;
    this.qTiledBuf = compute.buffers.createBuffer("q_tiled", numVHeads * this.ssmHeadDim * 4);
    this.kTiledBuf = compute.buffers.createBuffer("k_tiled", numVHeads * this.ssmHeadDim * 4);
    this.qkvBuf = compute.buffers.createBuffer("qkv", maxQkvDim * 4);
    this.ssmGateBuf = compute.buffers.createBuffer("ssm_gate", this.ssmInnerSize * 4);
    this.ssmOutputBuf = compute.buffers.createBuffer("ssm_output", this.ssmInnerSize * 4);
    this.alphaBuf = compute.buffers.createBuffer("alpha", numVHeads * 4);
    this.betaBuf = compute.buffers.createBuffer("beta", numVHeads * 4);
    this.decayBuf = compute.buffers.createBuffer("decay", numVHeads * 4);
    this.betaValBuf = compute.buffers.createBuffer("betaVal", numVHeads * 4);
    const maxCtx = Math.min(this.info.contextLength, 4096);
    this.kvCaches = [];
    this.deltaStates = [];
    for (let i = 0; i < this.numLayers; i++) {
      if (this.isStandardAttention(i)) {
        this.kvCaches.push(new KvCache(compute.device, compute.buffers, this.numKvHeads, this.keyLength, maxCtx, i));
        this.deltaStates.push(null);
      } else {
        this.kvCaches.push(null);
        const stateSize = this.ssmNumVHeads * this.ssmHeadDim * this.ssmHeadDim;
        const convBufSize = this.qkvDims[i] * (this.ssmConvKernel - 1);
        this.deltaStates.push({
          state: compute.buffers.createBuffer(`dn_state_${i}`, stateSize * 4),
          convBuf: compute.buffers.createBuffer(`dn_conv_${i}`, convBufSize * 4)
        });
      }
    }
  }
  resetCache() {
    this._position = 0;
    for (const kv of this.kvCaches) kv?.reset();
    for (let i = 0; i < this.deltaStates.length; i++) {
      const ds = this.deltaStates[i];
      if (ds) {
        const stateSize = this.ssmNumVHeads * this.ssmHeadDim * this.ssmHeadDim;
        const convBufSize = this.qkvDims[i] * (this.ssmConvKernel - 1);
        this.compute.device.queue.writeBuffer(ds.state, 0, new Float32Array(stateSize));
        if (convBufSize > 0) this.compute.device.queue.writeBuffer(ds.convBuf, 0, new Float32Array(convBufSize));
      }
    }
  }
  /**
   * Forward pass for a single token.
   */
  forward(tokenId) {
    const { compute, weights } = this;
    const E = this.embeddingDim;
    if (weights.tokenEmbedding.type === 8 /* Q8_0 */) {
      compute.embeddingQ8(weights.tokenEmbedding.buffer, this.hidden, tokenId, E);
    } else {
      compute.embedding(weights.tokenEmbedding.buffer, this.hidden, tokenId, E);
    }
    if (this._position === 0) {
    }
    for (let layer = 0; layer < this.numLayers; layer++) {
      const lw = weights.layers[layer];
      compute.copyAndRmsNorm(this.hidden, lw.attnNorm, this.normed, this.residual, E, this.rmsNormEps);
      if (lw.kind === "standard") {
        this.forwardStandardAttention(layer, lw);
      } else {
        this.forwardDeltaNet(layer, lw);
      }
      compute.copyAndRmsNorm(this.hidden, lw.postAttnNorm, this.normed, this.residual, E, this.rmsNormEps);
      compute.matmul(lw.gateProj.buffer, this.normed, this.gateBuf, this.ffnDim, E, lw.gateProj.type);
      compute.matmul(lw.upProj.buffer, this.normed, this.upBuf, this.ffnDim, E, lw.upProj.type);
      compute.siluMul(this.gateBuf, this.upBuf, this.ffnOut, this.ffnDim);
      compute.matmul(lw.downProj.buffer, this.ffnOut, this.temp, E, this.ffnDim, lw.downProj.type);
      compute.add(this.temp, this.residual, this.hidden, E);
    }
    compute.rmsNorm(this.hidden, weights.outputNorm, this.normed, E, this.rmsNormEps);
    compute.matmul(weights.output.buffer, this.normed, this.logits, this.vocabSize, E, weights.output.type);
    this._position++;
    return this.logits;
  }
  forwardStandardAttention(layer, lw) {
    const { compute } = this;
    const E = this.embeddingDim;
    const kvCache = this.kvCaches[layer];
    const keyLen = this.keyLength;
    const valLen = this.valueLength;
    const nH = this.numHeads;
    const nKV = this.numKvHeads;
    const pos = kvCache.seqLen;
    const scale = 1 / Math.sqrt(keyLen);
    const qFullDim = this.hasGatedQ ? nH * keyLen * 2 : nH * keyLen;
    const kDim = nKV * keyLen;
    const vDim = nKV * valLen;
    compute.matmul(lw.q.buffer, this.normed, this.qBuf, qFullDim, E, lw.q.type);
    compute.matmul(lw.k.buffer, this.normed, this.kBuf, kDim, E, lw.k.type);
    compute.matmul(lw.v.buffer, this.normed, this.vBuf, vDim, E, lw.v.type);
    if (this.hasGatedQ) {
      compute.deinterleaveQ(this.qBuf, this.qAttnBuf, this.qGateBufAttn, nH, keyLen);
    } else {
      compute.copyBuffer(this.qBuf, this.qAttnBuf, nH * keyLen * 4);
    }
    if (lw.qNorm) {
      compute.perHeadRmsNorm(this.qAttnBuf, lw.qNorm, nH, keyLen, this.rmsNormEps);
      compute.perHeadRmsNorm(this.kBuf, lw.kNorm || lw.qNorm, nKV, keyLen, this.rmsNormEps);
    }
    compute.partialRope(this.qAttnBuf, nH, keyLen, this.ropeDim, pos, this.ropeTheta);
    compute.partialRope(this.kBuf, nKV, keyLen, this.ropeDim, pos, this.ropeTheta);
    kvCache.write(this.kBuf, this.vBuf, kDim * 4, vDim * 4);
    compute.gatedAttention(
      this.qAttnBuf,
      this.qGateBufAttn,
      kvCache.kBuffer,
      kvCache.vBuffer,
      this.attnOut,
      nH,
      nKV,
      keyLen,
      valLen,
      kvCache.maxSeqLen,
      kvCache.seqLen,
      scale
    );
    compute.matmul(lw.o.buffer, this.attnOut, this.temp, E, nH * valLen, lw.o.type);
    compute.add(this.temp, this.residual, this.hidden, E);
  }
  forwardDeltaNet(layer, lw) {
    const { compute } = this;
    const E = this.embeddingDim;
    const ds = this.deltaStates[layer];
    const innerSize = this.ssmInnerSize;
    const numVHeads = this.ssmNumVHeads;
    const numKHeads = this.ssmGroupCount;
    const headDim = this.ssmHeadDim;
    const valueDim = numVHeads * headDim;
    const qkvDim = this.qkvDims[layer];
    const keyDim = (qkvDim - valueDim) / 2;
    const groupDim = keyDim / numKHeads;
    compute.matmul(lw.qkv.buffer, this.normed, this.qkvBuf, qkvDim, E, lw.qkv.type);
    compute.conv1dSilu(this.qkvBuf, ds.convBuf, lw.ssmConv1d, qkvDim, this.ssmConvKernel);
    compute.copyBufferRegion(this.qkvBuf, 0, this.qBuf, 0, keyDim * 4);
    compute.copyBufferRegion(this.qkvBuf, keyDim * 4, this.kBuf, 0, keyDim * 4);
    compute.copyBufferRegion(this.qkvBuf, keyDim * 2 * 4, this.vBuf, 0, valueDim * 4);
    compute.l2NormGroups(this.qBuf, numKHeads, groupDim);
    compute.l2NormGroups(this.kBuf, numKHeads, groupDim);
    const repeatFactor = numVHeads / numKHeads;
    let qForDelta = this.qBuf;
    let kForDelta = this.kBuf;
    if (repeatFactor > 1) {
      compute.repeatTile(this.qBuf, this.qTiledBuf, numKHeads, groupDim, repeatFactor);
      compute.repeatTile(this.kBuf, this.kTiledBuf, numKHeads, groupDim, repeatFactor);
      qForDelta = this.qTiledBuf;
      kForDelta = this.kTiledBuf;
    }
    compute.matmul(lw.ssmAlpha.buffer, this.normed, this.alphaBuf, numVHeads, E, lw.ssmAlpha.type);
    compute.matmul(lw.ssmBeta.buffer, this.normed, this.betaBuf, numVHeads, E, lw.ssmBeta.type);
    compute.computeDecayBeta(
      this.alphaBuf,
      this.betaBuf,
      lw.ssmA,
      lw.ssmDtBias,
      this.decayBuf,
      this.betaValBuf,
      numVHeads
    );
    const scale = 1 / Math.sqrt(headDim);
    compute.deltanetStep(
      qForDelta,
      kForDelta,
      this.vBuf,
      ds.state,
      this.decayBuf,
      this.betaValBuf,
      lw.ssmNorm,
      this.ssmOutputBuf,
      numVHeads,
      headDim,
      scale,
      this.rmsNormEps
    );
    compute.matmul(lw.gate.buffer, this.normed, this.ssmGateBuf, innerSize, E, lw.gate.type);
    compute.siluGate(this.ssmOutputBuf, this.ssmGateBuf, innerSize);
    compute.matmul(lw.ssmOut.buffer, this.ssmOutputBuf, this.temp, E, innerSize, lw.ssmOut.type);
    compute.add(this.temp, this.residual, this.hidden, E);
  }
  /** Apply RoPE to first ropeDim dims of each head, leaving rest unchanged. */
  async readLogits() {
    return new Float32Array(await this.compute.readBuffer(this.logits, this.vocabSize * 4));
  }
};

// src/model/sampler.ts
var DEFAULT_OPTIONS = {
  temperature: 0.7,
  topK: 40,
  topP: 0.9,
  repetitionPenalty: 1.1,
  seed: 0
};
var Sampler = class {
  options;
  rng;
  constructor(options) {
    this.options = { ...DEFAULT_OPTIONS, ...options };
    if (this.options.seed > 0) {
      let state = this.options.seed;
      this.rng = () => {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        return (state >>> 0) / 4294967296;
      };
    } else {
      this.rng = Math.random;
    }
  }
  /**
   * Sample a token ID from logits.
   * @param logits - Raw logits array [vocabSize]
   * @param previousTokens - Recent token IDs for repetition penalty
   */
  sample(logits, previousTokens) {
    const { temperature, topK, topP, repetitionPenalty } = this.options;
    if (previousTokens && previousTokens.length > 0 && repetitionPenalty !== 1) {
      const seen = new Set(previousTokens.slice(-64));
      for (const id of seen) {
        if (id < logits.length) {
          if (logits[id] > 0) {
            logits[id] /= repetitionPenalty;
          } else {
            logits[id] *= repetitionPenalty;
          }
        }
      }
    }
    if (temperature === 0 || temperature < 1e-6) {
      return argmax(logits);
    }
    for (let i = 0; i < logits.length; i++) {
      logits[i] /= temperature;
    }
    const indices = new Uint32Array(logits.length);
    for (let i = 0; i < indices.length; i++) indices[i] = i;
    indices.sort((a, b) => logits[b] - logits[a]);
    let k = Math.min(topK, logits.length);
    let maxLogit = logits[indices[0]];
    const probs = new Float32Array(k);
    let sum = 0;
    for (let i = 0; i < k; i++) {
      probs[i] = Math.exp(logits[indices[i]] - maxLogit);
      sum += probs[i];
    }
    for (let i = 0; i < k; i++) probs[i] /= sum;
    let cumProb = 0;
    let cutoff = k;
    for (let i = 0; i < k; i++) {
      cumProb += probs[i];
      if (cumProb >= topP) {
        cutoff = i + 1;
        break;
      }
    }
    sum = 0;
    for (let i = 0; i < cutoff; i++) sum += probs[i];
    for (let i = 0; i < cutoff; i++) probs[i] /= sum;
    const r = this.rng();
    cumProb = 0;
    for (let i = 0; i < cutoff; i++) {
      cumProb += probs[i];
      if (r <= cumProb) return indices[i];
    }
    return indices[0];
  }
};
function argmax(arr) {
  let maxIdx = 0;
  let maxVal = arr[0];
  for (let i = 1; i < arr.length; i++) {
    if (arr[i] > maxVal) {
      maxVal = arr[i];
      maxIdx = i;
    }
  }
  return maxIdx;
}

// src/engine.ts
var LlogosEngine = class _LlogosEngine {
  gpu = null;
  compute = null;
  model = null;
  tokenizer = null;
  modelInfo = null;
  _status = "uninitialized";
  get status() {
    return this._status;
  }
  get capabilities() {
    return this.gpu?.capabilities ?? null;
  }
  get info() {
    return this.modelInfo;
  }
  /** Check if WebGPU is available. */
  static isSupported() {
    return isWebGpuAvailable();
  }
  /** Initialize the WebGPU device. Must be called first. */
  async initGpu() {
    this.gpu = await initGpu();
    this.compute = new ComputeEngine(this.gpu.device);
    this._status = "ready";
    return this.gpu.capabilities;
  }
  /** Fetch and parse GGUF header to inspect model metadata without downloading weights. */
  async inspectModel(url) {
    return fetchGgufHeader(url);
  }
  /**
   * Download and load a GGUF model into GPU memory.
   */
  async loadModel(url, options) {
    if (!this.compute) throw new Error("GPU not initialized. Call initGpu() first.");
    this._status = "loading";
    try {
      options?.onProgress?.({ phase: "Parsing header", bytesDownloaded: 0, totalBytes: 0 });
      let info = await fetchGgufHeader(url);
      this.modelInfo = info;
      const estimate = this.estimateVram(info);
      const maxBuffer = this.gpu?.capabilities.maxBufferSize ?? 0;
      if (maxBuffer > 0 && estimate.totalBytes > maxBuffer * 4) {
        console.warn(`[llogos] Model may not fit: estimated ${Math.round(estimate.totalBytes / 1024 / 1024)} MB, max buffer ${Math.round(maxBuffer / 1024 / 1024)} MB`);
      }
      this.tokenizer = tokenizerFromGguf(info.metadata);
      const tensorMap = /* @__PURE__ */ new Map();
      let tensorCount = 0;
      await streamTensors(
        url,
        info,
        (name, data, tensorInfo) => {
          tensorMap.set(name, { buffer: data, info: tensorInfo });
          tensorCount++;
        },
        (p) => {
          options?.onProgress?.({
            phase: `${p.phase} (${p.tensorsLoaded}/${p.totalTensors} tensors)`,
            bytesDownloaded: p.bytesDownloaded,
            totalBytes: p.totalBytes
          });
        }
      );
      options?.onProgress?.({ phase: "Uploading to GPU", bytesDownloaded: 0, totalBytes: 0 });
      if (info.architecture === "qwen35") {
        this.model = new Qwen35Model(this.compute, info);
      } else {
        this.model = new LlamaModel(this.compute, info);
      }
      console.log("[llogos] Uploading weights to GPU...");
      await this.model.initWeights(tensorMap);
      console.log("[llogos] Weights uploaded, clearing CPU data...");
      tensorMap.clear();
      console.log("[llogos] Model loaded successfully");
      this._status = "loaded";
      return info;
    } catch (e) {
      this._status = "error";
      throw e;
    }
  }
  /**
   * Generate tokens from a prompt. Returns an async iterator of token strings.
   */
  async *generate(prompt, options) {
    if (!this.model || !this.tokenizer || !this.compute) {
      throw new Error("Model not loaded. Call loadModel() first.");
    }
    this._status = "generating";
    const maxTokens = options?.maxTokens ?? 512;
    const sampler = new Sampler(options);
    try {
      let finalPrompt = prompt;
      if (!options?.raw) {
        finalPrompt = this.applyChatTemplate(prompt);
      }
      const inputTokens = [];
      if (this.model.position === 0 && this.tokenizer.bosTokenId >= 0 && options?.raw) {
        inputTokens.push(this.tokenizer.bosTokenId);
      }
      inputTokens.push(...this.tokenizer.encode(finalPrompt));
      const allTokens = [...inputTokens];
      if (inputTokens.length > 1 && "forwardPrefill" in this.model) {
        this.model.forwardPrefill(inputTokens);
      } else {
        for (let i = 0; i < inputTokens.length; i++) {
          await this.model.forward(inputTokens[i]);
        }
      }
      let logits = await this.model.readLogits();
      const thinkStartId = this.tokenizer.getTokenId("<think>");
      const thinkEndId = this.tokenizer.getTokenId("</think>");
      let inThinking = false;
      for (let step = 0; step < maxTokens; step++) {
        if (options?.signal?.aborted) break;
        const nextToken = sampler.sample(logits, allTokens);
        if (this.tokenizer.isEos(nextToken)) break;
        allTokens.push(nextToken);
        if (nextToken === thinkStartId) {
          inThinking = true;
        } else if (nextToken === thinkEndId) {
          inThinking = false;
        } else if (!inThinking) {
          const text = this.tokenizer.decode([nextToken]);
          options?.onToken?.(text, nextToken);
          yield text;
        }
        await this.model.forward(nextToken);
        logits = await this.model.readLogits();
      }
    } finally {
      this._status = "loaded";
    }
  }
  /** Reset the KV cache and conversation history for a new conversation. */
  resetSession() {
    this.model?.resetCache();
    this.conversationHistory = [];
  }
  /** Unload model and free GPU memory. */
  unloadModel() {
    this.compute?.buffers.destroyAll();
    this.model = null;
    this.tokenizer = null;
    this.modelInfo = null;
    this._status = this.gpu ? "ready" : "uninitialized";
  }
  /** Get current VRAM usage in bytes. */
  get vramUsage() {
    return this.compute?.buffers.vramUsage ?? 0;
  }
  // GPU types that stay quantized (have native GPU shaders)
  static GPU_NATIVE_TYPES = /* @__PURE__ */ new Set([0 /* F32 */, 8 /* Q8_0 */, 2 /* Q4_0 */]);
  /**
   * Estimate VRAM needed for a model before downloading.
   * Returns breakdown in bytes: weights, kvCache, working, total.
   */
  estimateVram(info) {
    let weightsBytes = 0;
    for (const tensor of info.tensors) {
      if (_LlogosEngine.GPU_NATIVE_TYPES.has(tensor.type)) {
        weightsBytes += tensor.byteSize;
      } else {
        weightsBytes += tensor.elementCount * 4;
      }
    }
    const maxSeq = Math.min(info.contextLength, 4096);
    const headDim = info.embeddingLength / info.headCount;
    const kvHeads = info.headCountKv || info.headCount;
    const kvCacheBytes = 2 * info.blockCount * kvHeads * maxSeq * headDim * 4;
    const E = info.embeddingLength;
    const F = info.feedForwardLength;
    const V = info.vocabSize;
    const workingBytes = (E * 11 + F * 3 + V) * 4;
    return {
      weightsBytes,
      kvCacheBytes,
      workingBytes,
      totalBytes: weightsBytes + kvCacheBytes + workingBytes
    };
  }
  /**
   * Apply chat template to format the prompt.
   * Uses the Jinja2 template from GGUF metadata if available,
   * falls back to ChatML/Llama2 heuristics.
   */
  applyChatTemplate(userMessage) {
    const chatTemplate = this.modelInfo?.metadata.get("tokenizer.chat_template");
    const messages = [
      ...this.conversationHistory,
      { role: "user", content: userMessage }
    ];
    if (this.tokenizer && this.tokenizer.getTokenId("<|start_header_id|>") >= 0) {
      let prompt = "<|begin_of_text|>";
      prompt += "<|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|>";
      for (const msg of messages) {
        prompt += `<|start_header_id|>${msg.role}<|end_header_id|>

${msg.content.trim()}<|eot_id|>`;
      }
      prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n";
      return prompt;
    }
    if (this.tokenizer && this.tokenizer.getTokenId("<|im_start|>") >= 0) {
      let prompt = "";
      for (const msg of messages) {
        prompt += `<|im_start|>${msg.role}
${msg.content}<|im_end|>
`;
      }
      prompt += "<|im_start|>assistant\n";
      return prompt;
    }
    if (chatTemplate?.includes("[INST]")) {
      return `[INST] ${userMessage} [/INST]`;
    }
    if (chatTemplate) {
      try {
        const bosToken = this.tokenizer && this.tokenizer.bosTokenId >= 0 ? this.tokenizer.decode([this.tokenizer.bosTokenId]) : "";
        const eosToken = this.tokenizer && this.tokenizer.eosTokenId >= 0 ? this.tokenizer.decode([this.tokenizer.eosTokenId]) : "";
        const result = applyTemplate(chatTemplate, messages, {
          bos_token: bosToken,
          eos_token: eosToken,
          add_generation_prompt: true
        });
        if (result.trim().length > 0) return result;
      } catch {
      }
    }
    return userMessage;
  }
  /** Conversation history for multi-turn chat template support. */
  conversationHistory = [];
};
export {
  BpeTokenizer,
  GgmlType,
  LlogosEngine,
  parseGguf
};
//# sourceMappingURL=index.js.map
