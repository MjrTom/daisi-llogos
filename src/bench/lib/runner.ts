// LLogos CLI benchmark runner — spawns dotnet process and parses output

import { spawn } from "child_process";
import { LLOGOS_PROJECT } from "./config";
import type { BenchConfig, Backend, ContextPreset } from "./config";

export interface BenchResult {
  configId: string;
  modelPath: string;
  backend: Backend;
  prefillTokens: number;
  prefillTokPerSec: number;
  prefillMs: number;
  decodeTokens: number;
  decodeTokPerSec: number;
  decodeMs: number;
  compressionRatio: number | null;
  compressedKB: number | null;
  uncompressedKB: number | null;
  effectiveBits: number | null;
  error: string | null;
}

export async function runBenchmark(
  modelPath: string,
  config: BenchConfig,
  backend: Backend,
  context: ContextPreset,
  signal?: AbortSignal,
): Promise<BenchResult> {
  const args = [
    "run", "--project", LLOGOS_PROJECT, "-c", "Release", "--",
    "--model", modelPath,
    "--bench",
    "--prompt", context.prompt,
    "--max-tokens", String(context.maxTokens),
    "--max-context", String(context.maxContext),
    "--backend", backend,
    ...config.args,
  ];

  return new Promise((resolve) => {
    const proc = spawn("dotnet", args, {
      stdio: ["ignore", "pipe", "pipe"],
      timeout: 600_000,
    });

    // Kill process if abort signal fires
    const onAbort = () => { proc.kill("SIGTERM"); };
    signal?.addEventListener("abort", onAbort, { once: true });

    let stderr = "";
    proc.stderr.on("data", (data: Buffer) => { stderr += data.toString(); });
    proc.stdout.on("data", (data: Buffer) => { stderr += data.toString(); });

    proc.on("close", (code) => {
      signal?.removeEventListener("abort", onAbort);
      const result: BenchResult = {
        configId: config.id,
        modelPath,
        backend,
        prefillTokens: 0,
        prefillTokPerSec: 0,
        prefillMs: 0,
        decodeTokens: 0,
        decodeTokPerSec: 0,
        decodeMs: 0,
        compressionRatio: null,
        compressedKB: null,
        uncompressedKB: null,
        effectiveBits: null,
        error: code !== 0 ? `Exit code ${code}` : null,
      };

      // Parse: "Prefill:  26 tokens in  993.3 ms  (  26.2 tok/s)"
      const prefillMatch = stderr.match(/Prefill:\s+(\d+)\s+tokens\s+in\s+([\d.]+)\s+ms\s+\(\s*([\d.]+)\s+tok\/s\)/);
      if (prefillMatch) {
        result.prefillTokens = parseInt(prefillMatch[1]);
        result.prefillMs = parseFloat(prefillMatch[2]);
        result.prefillTokPerSec = parseFloat(prefillMatch[3]);
      }

      // Parse: "Decode:  128 tokens in  3381.4 ms  (  37.9 tok/s)"
      const decodeMatch = stderr.match(/Decode:\s+(\d+)\s+tokens\s+in\s+([\d.]+)\s+ms\s+\(\s*([\d.]+)\s+tok\/s\)/);
      if (decodeMatch) {
        result.decodeTokens = parseInt(decodeMatch[1]);
        result.decodeMs = parseFloat(decodeMatch[2]);
        result.decodeTokPerSec = parseFloat(decodeMatch[3]);
      }

      // Parse: "Ratio:  8.8x (3.5 bits/dim)"
      const ratioMatch = stderr.match(/Ratio:\s+([\d.]+)x\s+\(([\d.]+)\s+bits\/dim\)/);
      if (ratioMatch) {
        result.compressionRatio = parseFloat(ratioMatch[1]);
        result.effectiveBits = parseFloat(ratioMatch[2]);
      }

      // Parse: "Compressed:  418.7 KB"
      const compMatch = stderr.match(/Compressed:\s+([\d.]+)\s+KB/);
      if (compMatch) result.compressedKB = parseFloat(compMatch[1]);

      const uncompMatch = stderr.match(/Uncompressed:\s+([\d.]+)\s+KB/);
      if (uncompMatch) result.uncompressedKB = parseFloat(uncompMatch[1]);

      if (!prefillMatch && !decodeMatch && !result.error) {
        result.error = "No benchmark output parsed";
      }

      resolve(result);
    });

    proc.on("error", (err) => {
      resolve({
        configId: config.id, modelPath, backend,
        prefillTokens: 0, prefillTokPerSec: 0, prefillMs: 0,
        decodeTokens: 0, decodeTokPerSec: 0, decodeMs: 0,
        compressionRatio: null, compressedKB: null, uncompressedKB: null,
        effectiveBits: null, error: err.message,
      });
    });
  });
}
