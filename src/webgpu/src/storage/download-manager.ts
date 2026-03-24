/**
 * Model downloader with Cache API persistence.
 * Downloads the entire GGUF file as a single streaming fetch.
 */

export interface DownloadProgress {
  bytesDownloaded: number;
  totalBytes: number;
  tensorName?: string;
}

export type ProgressCallback = (progress: DownloadProgress) => void;

const CACHE_NAME = 'llogos-webgpu-models';

/**
 * Download (or retrieve from cache) an entire GGUF file.
 * Returns the complete file as an ArrayBuffer.
 */
export async function downloadFile(
  url: string,
  onProgress?: ProgressCallback,
): Promise<ArrayBuffer> {
  // Check cache first
  let cache: Cache | null = null;
  try {
    if (typeof caches !== 'undefined') {
      cache = await caches.open(CACHE_NAME);
      const cached = await cache.match(url);
      if (cached) {
        // Stream from cache with progress updates
        const total = parseInt(cached.headers.get('content-length') ?? '0', 10);
        const reader = cached.body?.getReader();
        if (reader && total > 0) {
          onProgress?.({ bytesDownloaded: 0, totalBytes: total });
          const chunks: Uint8Array[] = [];
          let loaded = 0;
          while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            chunks.push(value);
            loaded += value.byteLength;
            onProgress?.({ bytesDownloaded: loaded, totalBytes: total });
          }
          const result = new Uint8Array(loaded);
          let off = 0;
          for (const c of chunks) { result.set(c, off); off += c.byteLength; }
          return result.buffer;
        }
        // Fallback: no streaming from cache
        const buf = await cached.arrayBuffer();
        onProgress?.({ bytesDownloaded: buf.byteLength, totalBytes: buf.byteLength });
        return buf;
      }
    }
  } catch {
    // Cache API not available
  }

  // Download with streaming progress
  const response = await fetch(url);

  if (!response.ok) {
    throw new Error(`Download failed: ${response.status} ${response.statusText}`);
  }

  const contentLength = parseInt(response.headers.get('content-length') ?? '0', 10);
  const reader = response.body?.getReader();

  if (!reader) {
    // Fallback: no streaming, just await the whole thing
    const buf = await response.arrayBuffer();
    if (cache) {
      try { await cache.put(url, new Response(buf.slice(0))); } catch {}
    }
    return buf;
  }

  // Stream into a single buffer
  const chunks: Uint8Array[] = [];
  let downloaded = 0;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    downloaded += value.byteLength;
    onProgress?.({ bytesDownloaded: downloaded, totalBytes: contentLength || downloaded });
  }

  // Assemble chunks into a single ArrayBuffer
  const result = new Uint8Array(downloaded);
  let offset = 0;
  for (const chunk of chunks) {
    result.set(chunk, offset);
    offset += chunk.byteLength;
  }


  // Cache the complete file
  if (cache) {
    try {
      await cache.put(url, new Response(result.buffer.slice(0), {
        headers: { 'content-length': String(downloaded) },
      }));
    } catch {
    }
  }

  return result.buffer;
}

/**
 * Extract tensor data from a complete GGUF file buffer.
 */
export function extractTensorData(
  fileBuffer: ArrayBuffer,
  absoluteOffset: number,
  byteSize: number,
): ArrayBuffer {
  return fileBuffer.slice(absoluteOffset, absoluteOffset + byteSize);
}

/**
 * Clear all cached model data.
 */
export async function clearModelCache(): Promise<void> {
  if (typeof caches !== 'undefined') {
    await caches.delete(CACHE_NAME);
  }
}
