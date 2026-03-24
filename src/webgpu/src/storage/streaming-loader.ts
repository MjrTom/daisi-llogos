/**
 * Streaming tensor loader — loads GGUF tensors without holding the entire file in memory.
 *
 * Strategy:
 * 1. Parse header via range request (already done by fetchGgufHeader)
 * 2. Stream the full file from network/cache
 * 3. As bytes arrive, extract tensors at their known offsets
 * 4. Upload each tensor to GPU immediately, discard CPU copy
 *
 * Peak memory = header + largest single tensor + working buffer
 * NOT the entire file (which can be > 2 GB).
 */

import type { GgufModelInfo, GgufTensorInfo } from '../gguf/gguf-parser.js';

export interface StreamProgress {
  phase: string;
  bytesDownloaded: number;
  totalBytes: number;
  tensorsLoaded: number;
  totalTensors: number;
}

const CACHE_NAME = 'llogos-webgpu-models';

/**
 * Stream a GGUF file and call onTensor for each tensor as it becomes available.
 * The file is cached in the Cache API for future loads.
 *
 * @param url - URL to the GGUF file
 * @param info - Pre-parsed GGUF model info (from fetchGgufHeader)
 * @param onTensor - Called with tensor name, data buffer, and tensor info
 * @param onProgress - Progress callback
 */
export async function streamTensors(
  url: string,
  info: GgufModelInfo,
  onTensor: (name: string, data: ArrayBuffer, tensorInfo: GgufTensorInfo) => void,
  onProgress?: (progress: StreamProgress) => void,
): Promise<void> {
  const tensorDataStart = info.tensorDataOffset;
  const totalTensors = info.tensors.length;

  // Sort tensors by offset for sequential streaming
  const sortedTensors = [...info.tensors].sort((a, b) => a.offset - b.offset);

  // Build a map of absolute byte ranges we need to extract
  // Each tensor: [tensorDataStart + tensor.offset, tensorDataStart + tensor.offset + tensor.byteSize)
  const extractions = sortedTensors.map(t => ({
    tensor: t,
    start: tensorDataStart + t.offset,
    end: tensorDataStart + t.offset + t.byteSize,
  }));

  // Get the response (from cache or network)
  const response = await getResponse(url, onProgress);
  const contentLength = parseInt(response.headers.get('content-length') ?? '0', 10);
  const reader = response.body?.getReader();

  if (!reader) {
    throw new Error('Response body is not readable as a stream');
  }

  // Stream through the file, extracting tensors as we pass their offsets
  let filePos = 0;
  let extractIdx = 0;
  let tensorsLoaded = 0;

  // Buffer for accumulating bytes when a tensor spans multiple chunks
  let pendingBuffer: Uint8Array | null = null;
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
      phase: 'Loading tensors',
      bytesDownloaded: filePos,
      totalBytes: contentLength || filePos,
      tensorsLoaded,
      totalTensors,
    });

    // Process any pending tensor that needs more data
    if (pendingBuffer) {
      const needed = pendingNeeded - (pendingBuffer.byteLength - pendingStart);
      const available = Math.min(chunk.byteLength, needed);
      const newBuf = new Uint8Array(pendingBuffer.byteLength - pendingStart + available);
      newBuf.set(new Uint8Array(pendingBuffer.buffer, pendingBuffer.byteOffset + pendingStart, pendingBuffer.byteLength - pendingStart));
      newBuf.set(chunk.subarray(0, available), pendingBuffer.byteLength - pendingStart);

      if (newBuf.byteLength >= pendingNeeded) {
        // Tensor complete
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

    // Check if any tensors fall within this chunk
    while (extractIdx < extractions.length) {
      const ext = extractions[extractIdx];

      if (ext.start >= chunkEnd) break; // tensor starts after this chunk

      if (ext.start >= chunkStart && ext.end <= chunkEnd) {
        // Tensor fully within this chunk — extract directly
        const localStart = ext.start - chunkStart;
        const tensorData = chunk.buffer.slice(
          chunk.byteOffset + localStart,
          chunk.byteOffset + localStart + ext.tensor.byteSize,
        );
        onTensor(ext.tensor.name, tensorData, ext.tensor);
        tensorsLoaded++;
        extractIdx++;
      } else if (ext.start >= chunkStart && ext.start < chunkEnd) {
        // Tensor starts in this chunk but extends beyond — start buffering
        const localStart = ext.start - chunkStart;
        pendingBuffer = chunk.subarray(localStart);
        pendingStart = 0;
        pendingNeeded = ext.tensor.byteSize;

        if (pendingBuffer.byteLength >= pendingNeeded) {
          // Actually fits (edge case)
          onTensor(ext.tensor.name, pendingBuffer.buffer.slice(
            pendingBuffer.byteOffset,
            pendingBuffer.byteOffset + ext.tensor.byteSize,
          ), ext.tensor);
          tensorsLoaded++;
          extractIdx++;
          pendingBuffer = null;
          pendingNeeded = 0;
        }
        break; // need more chunks for this tensor
      } else {
        // Tensor starts before this chunk — skip (shouldn't happen with sorted tensors)
        extractIdx++;
      }
    }
  }

  // Read remaining stream to ensure caching completes
  while (true) {
    const { done } = await reader.read();
    if (done) break;
  }

  onProgress?.({
    phase: 'Complete',
    bytesDownloaded: filePos,
    totalBytes: contentLength || filePos,
    tensorsLoaded,
    totalTensors,
  });
}

/**
 * Get a response for the URL — from cache if available, otherwise fetch and cache.
 */
async function getResponse(
  url: string,
  onProgress?: (progress: StreamProgress) => void,
): Promise<Response> {
  // Try cache first
  let cache: Cache | null = null;
  try {
    if (typeof caches !== 'undefined') {
      cache = await caches.open(CACHE_NAME);
      const cached = await cache.match(url);
      if (cached) {
        onProgress?.({ phase: 'Loading from cache', bytesDownloaded: 0, totalBytes: 0, tensorsLoaded: 0, totalTensors: 0 });
        return cached;
      }
    }
  } catch {}

  // Fetch from network
  onProgress?.({ phase: 'Downloading', bytesDownloaded: 0, totalBytes: 0, tensorsLoaded: 0, totalTensors: 0 });
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Download failed: ${response.status}`);

  // Clone and cache (we consume the original body for streaming)
  if (cache) {
    try {
      const [stream1, stream2] = response.body!.tee();
      const cachedResponse = new Response(stream2, {
        headers: { 'content-length': response.headers.get('content-length') ?? '0' },
      });
      cache.put(url, cachedResponse).catch(() => {});
      return new Response(stream1, { headers: response.headers });
    } catch {
      // tee() might fail — fall back to non-cached streaming
    }
  }

  return response;
}
