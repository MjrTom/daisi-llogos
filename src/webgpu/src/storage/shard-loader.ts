/**
 * Shard-based model loader for WebGPU.
 * Fetches individual shard files (header, embed, output, per-layer) and extracts tensors.
 * Uses Cache API for persistence, same pattern as streaming-loader.ts.
 */

import type { GgufModelInfo, GgufTensorInfo } from '../gguf/gguf-parser.js';
import { parseShardIndex, extractTensorFromShard, type ShardIndex, type ShardManifest } from '../gguf/shard-reader.js';
import type { StreamProgress } from './streaming-loader.js';

const CACHE_NAME = 'llogos-webgpu-shards';

export interface ShardProgress {
  phase: string;
  shardsDownloaded: number;
  totalShards: number;
  bytesDownloaded: number;
  totalBytes: number;
}

/**
 * Fetch the shard manifest JSON for a model.
 * @param baseUrl The base URL of the model (e.g., "https://cdn.example.com/models/model.gguf")
 */
export async function fetchManifest(baseUrl: string): Promise<ShardManifest> {
  const url = `${baseUrl}.manifest.json`;
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Failed to fetch manifest: ${response.status} from ${url}`);
  return response.json();
}

/**
 * Fetch a shard file (from cache or network) and return its ArrayBuffer.
 * The result is cached for future loads.
 */
export async function fetchShard(
  shardUrl: string,
  onProgress?: (downloaded: number, total: number) => void,
): Promise<ArrayBuffer> {
  // Try cache first
  let cache: Cache | null = null;
  try {
    if (typeof caches !== 'undefined') {
      cache = await caches.open(CACHE_NAME);
      const cached = await cache.match(shardUrl);
      if (cached) {
        return cached.arrayBuffer();
      }
    }
  } catch { /* cache unavailable */ }

  // Fetch from network
  const response = await fetch(shardUrl);
  if (!response.ok) throw new Error(`Shard download failed: ${response.status} from ${shardUrl}`);

  const contentLength = parseInt(response.headers.get('content-length') ?? '0', 10);
  const reader = response.body?.getReader();

  if (!reader) {
    // No streaming — just get the full buffer
    const buffer = await response.arrayBuffer();
    if (cache) {
      try { await cache.put(shardUrl, new Response(buffer.slice(0))); } catch { /* */ }
    }
    return buffer;
  }

  // Stream into buffer
  const chunks: Uint8Array[] = [];
  let downloaded = 0;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    downloaded += value.byteLength;
    onProgress?.(downloaded, contentLength || downloaded);
  }

  // Combine chunks
  const result = new Uint8Array(downloaded);
  let offset = 0;
  for (const chunk of chunks) {
    result.set(chunk, offset);
    offset += chunk.byteLength;
  }

  // Cache the result
  if (cache) {
    try { await cache.put(shardUrl, new Response(result.buffer.slice(0))); } catch { /* */ }
  }

  return result.buffer;
}

/**
 * Compute the shard URL from a base URL and shard filename.
 * Given baseUrl "https://cdn.example.com/models/model.gguf" and shard "model.gguf.layer.0",
 * returns "https://cdn.example.com/models/model.gguf.layer.0"
 */
export function shardUrl(baseUrl: string, shardFileName: string): string {
  // The shard filename is "{basename}.gguf.{suffix}", and baseUrl is "{prefix}/{basename}.gguf"
  // So the URL is "{prefix}/{shardFileName}"
  const lastSlash = baseUrl.lastIndexOf('/');
  const prefix = lastSlash >= 0 ? baseUrl.substring(0, lastSlash + 1) : '';
  return prefix + shardFileName;
}

/**
 * Fetch a shard file and extract all tensors from it.
 * Returns a map of tensor name → { buffer, info } using the header GGUF's tensor metadata.
 */
export async function fetchShardTensors(
  url: string,
  tensorInfoMap: Map<string, GgufTensorInfo>,
  onProgress?: (downloaded: number, total: number) => void,
): Promise<Map<string, { buffer: ArrayBuffer; info: GgufTensorInfo }>> {
  const shardBuffer = await fetchShard(url, onProgress);
  const index = parseShardIndex(shardBuffer);

  const result = new Map<string, { buffer: ArrayBuffer; info: GgufTensorInfo }>();

  for (const [name] of index.tensors) {
    const info = tensorInfoMap.get(name);
    if (!info) continue; // Skip tensors not in the header metadata

    const data = extractTensorFromShard(shardBuffer, index, name);
    result.set(name, { buffer: data, info });
  }

  return result;
}
