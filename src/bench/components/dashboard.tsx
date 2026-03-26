"use client";

import { useState, useEffect, useCallback } from "react";
import { CONFIGS, BACKENDS } from "@/lib/config";
import type { BenchConfig, Backend } from "@/lib/config";
import type { BenchResult } from "@/lib/runner";

interface ModelEntry {
  path: string;
  name: string;
  shortName: string;
  filename: string;
}

interface SystemInfo {
  cpu: string;
  gpu: string;
  vram: string;
  ram: string;
  os: string;
  dotnet: string;
  gpuDriver?: string;
}

type CellState = "idle" | "running" | "done" | "error";

interface CellData {
  state: CellState;
  result?: BenchResult;
}

// Key for the results map: "modelPath|configId"
function cellKey(modelPath: string, configId: string) {
  return `${modelPath}|${configId}`;
}

export default function Dashboard() {
  const [models, setModels] = useState<ModelEntry[]>([]);
  const [selectedModels, setSelectedModels] = useState<Set<string>>(new Set());
  const [selectedConfigs, setSelectedConfigs] = useState<Set<string>>(
    new Set(CONFIGS.map((c) => c.id))
  );
  const [backend, setBackend] = useState<Backend>("cuda");
  const [cells, setCells] = useState<Map<string, CellData>>(new Map());
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState(0);
  const [system, setSystem] = useState<SystemInfo | null>(null);

  // Load models + system info on mount
  useEffect(() => {
    fetch("/api/models")
      .then((r) => r.json())
      .then((data) => {
        setModels(data.models || []);
        // Auto-select first 3 models
        const auto = (data.models || []).slice(0, 3).map((m: ModelEntry) => m.path);
        setSelectedModels(new Set(auto));
      });
    fetch("/api/system")
      .then((r) => r.json())
      .then((data) => setSystem(data));
  }, []);

  const toggleModel = (path: string) => {
    setSelectedModels((prev) => {
      const next = new Set(prev);
      next.has(path) ? next.delete(path) : next.add(path);
      return next;
    });
  };

  const toggleConfig = (id: string) => {
    setSelectedConfigs((prev) => {
      const next = new Set(prev);
      next.has(id) ? next.delete(id) : next.add(id);
      return next;
    });
  };

  const runBenchmark = useCallback(async () => {
    if (running || selectedModels.size === 0 || selectedConfigs.size === 0) return;
    setRunning(true);
    setProgress(0);
    setCells(new Map());

    const response = await fetch("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        models: Array.from(selectedModels),
        configs: Array.from(selectedConfigs),
        backend,
      }),
    });

    const reader = response.body?.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (reader) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n\n");
      buffer = lines.pop() || "";

      for (const line of lines) {
        const match = line.match(/^data: (.+)$/);
        if (!match) continue;
        try {
          const data = JSON.parse(match[1]);

          if (data.type === "running") {
            setCells((prev) => {
              const next = new Map(prev);
              next.set(cellKey(data.modelPath, data.configId), { state: "running" });
              return next;
            });
            setProgress(data.progress);
          }

          if (data.type === "result") {
            const result = data as BenchResult;
            setCells((prev) => {
              const next = new Map(prev);
              next.set(cellKey(result.modelPath, result.configId), {
                state: result.error ? "error" : "done",
                result,
              });
              return next;
            });
            setProgress(data.progress);
          }

          if (data.type === "done") {
            setProgress(1);
          }
        } catch {}
      }
    }

    setRunning(false);
  }, [running, selectedModels, selectedConfigs, backend]);

  // Get baseline decode tok/s for a model (for relative comparison)
  const getBaseline = (modelPath: string): number | null => {
    const cell = cells.get(cellKey(modelPath, "baseline"));
    return cell?.result?.decodeTokPerSec || null;
  };

  const activeConfigs = CONFIGS.filter((c) => selectedConfigs.has(c.id));
  const activeModels = models.filter((m) => selectedModels.has(m.path));

  return (
    <div className="p-6 max-w-[1600px] mx-auto">
      {/* Header */}
      <div className="mb-6">
        <h1 className="text-3xl font-bold text-white mb-1">LLogos Bench</h1>
        <p className="text-zinc-400 text-sm">LLogos Inference Engine Benchmark Dashboard</p>
      </div>

      {/* System Info */}
      {system && (
        <div className="mb-6 bg-zinc-900 border border-zinc-800 rounded-lg p-4">
          <h2 className="text-xs font-semibold text-zinc-500 uppercase tracking-wider mb-2">System</h2>
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3 text-sm">
            <div><span className="text-zinc-500">CPU</span><br/><span className="text-zinc-200">{system.cpu}</span></div>
            <div><span className="text-zinc-500">GPU</span><br/><span className="text-zinc-200">{system.gpu}</span></div>
            <div><span className="text-zinc-500">VRAM</span><br/><span className="text-zinc-200">{system.vram}</span></div>
            <div><span className="text-zinc-500">RAM</span><br/><span className="text-zinc-200">{system.ram}</span></div>
            <div><span className="text-zinc-500">OS</span><br/><span className="text-zinc-200">{system.os}</span></div>
            <div><span className="text-zinc-500">.NET</span><br/><span className="text-zinc-200">{system.dotnet}</span></div>
          </div>
        </div>
      )}

      {/* Controls */}
      <div className="mb-6 grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Models */}
        <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-4">
          <h2 className="text-xs font-semibold text-zinc-500 uppercase tracking-wider mb-2">Models</h2>
          <div className="space-y-1 max-h-48 overflow-y-auto">
            {models.map((m) => (
              <label key={m.path} className="flex items-center gap-2 text-sm cursor-pointer hover:bg-zinc-800 px-2 py-1 rounded">
                <input
                  type="checkbox"
                  checked={selectedModels.has(m.path)}
                  onChange={() => toggleModel(m.path)}
                  className="accent-emerald-500"
                />
                <span className="text-zinc-300 truncate">{m.shortName}</span>
              </label>
            ))}
          </div>
        </div>

        {/* Configurations */}
        <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-4">
          <h2 className="text-xs font-semibold text-zinc-500 uppercase tracking-wider mb-2">Configurations</h2>
          <div className="space-y-1">
            {CONFIGS.map((c) => (
              <label key={c.id} className="flex items-center gap-2 text-sm cursor-pointer hover:bg-zinc-800 px-2 py-1 rounded">
                <input
                  type="checkbox"
                  checked={selectedConfigs.has(c.id)}
                  onChange={() => toggleConfig(c.id)}
                  className="accent-emerald-500"
                />
                <span className={`w-2 h-2 rounded-full ${c.color}`} />
                <span className="text-zinc-300">{c.label}</span>
              </label>
            ))}
          </div>
        </div>

        {/* Backend + Run */}
        <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-4 flex flex-col justify-between">
          <div>
            <h2 className="text-xs font-semibold text-zinc-500 uppercase tracking-wider mb-2">Backend</h2>
            <div className="flex gap-2 mb-4">
              {BACKENDS.map((b) => (
                <button
                  key={b}
                  onClick={() => setBackend(b)}
                  className={`px-4 py-2 rounded text-sm font-medium transition-colors ${
                    backend === b
                      ? "bg-emerald-600 text-white"
                      : "bg-zinc-800 text-zinc-400 hover:bg-zinc-700"
                  }`}
                >
                  {b.toUpperCase()}
                </button>
              ))}
            </div>
          </div>

          <button
            onClick={runBenchmark}
            disabled={running || selectedModels.size === 0}
            className={`w-full py-3 rounded-lg font-semibold text-lg transition-colors ${
              running
                ? "bg-amber-600 text-amber-100 cursor-wait"
                : "bg-emerald-600 hover:bg-emerald-500 text-white cursor-pointer"
            } disabled:opacity-50 disabled:cursor-not-allowed`}
          >
            {running ? `Running... ${Math.round(progress * 100)}%` : "Run Benchmark"}
          </button>
        </div>
      </div>

      {/* Progress Bar */}
      {running && (
        <div className="mb-4 h-1 bg-zinc-800 rounded-full overflow-hidden">
          <div
            className="h-full bg-emerald-500 transition-all duration-300"
            style={{ width: `${progress * 100}%` }}
          />
        </div>
      )}

      {/* Results Matrix */}
      {activeModels.length > 0 && activeConfigs.length > 0 && (
        <div className="bg-zinc-900 border border-zinc-800 rounded-lg overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr>
                <th className="text-left p-3 text-zinc-500 font-medium sticky left-0 bg-zinc-900 z-10">Model</th>
                {activeConfigs.map((c) => (
                  <th key={c.id} className="p-3 text-center min-w-[120px]">
                    <div className={`${c.color} text-white text-xs font-semibold px-2 py-1 rounded`}>
                      {c.shortLabel}
                    </div>
                    <div className="text-zinc-500 text-[10px] mt-1">{c.label}</div>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {activeModels.map((model) => {
                const baseline = getBaseline(model.path);
                return (
                  <tr key={model.path} className="border-t border-zinc-800 hover:bg-zinc-800/50">
                    <td className="p-3 text-zinc-300 font-medium sticky left-0 bg-zinc-900 z-10 max-w-[200px] truncate">
                      {model.shortName}
                    </td>
                    {activeConfigs.map((config) => {
                      const cell = cells.get(cellKey(model.path, config.id));
                      return (
                        <td key={config.id} className="p-2 text-center">
                          <ResultCell cell={cell} baseline={baseline} />
                        </td>
                      );
                    })}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      {/* Empty State */}
      {cells.size === 0 && !running && (
        <div className="text-center py-16 text-zinc-600">
          Select models and configurations, then click Run Benchmark
        </div>
      )}
    </div>
  );
}

function ResultCell({ cell, baseline }: { cell?: CellData; baseline: number | null }) {
  if (!cell) return <div className="text-zinc-700">-</div>;

  if (cell.state === "running") {
    return (
      <div className="animate-pulse">
        <div className="w-12 h-4 bg-amber-900/50 rounded mx-auto" />
      </div>
    );
  }

  if (cell.state === "error") {
    return (
      <div className="text-red-400 text-xs" title={cell.result?.error || "Error"}>
        ERR
      </div>
    );
  }

  const r = cell.result;
  if (!r) return <div className="text-zinc-700">-</div>;

  const decode = r.decodeTokPerSec;
  const pct = baseline && baseline > 0 ? (decode / baseline) * 100 : null;

  // Color based on performance relative to baseline
  let bg = "bg-zinc-800";
  let textColor = "text-zinc-300";
  if (pct !== null) {
    if (pct >= 98) { bg = "bg-emerald-900/60"; textColor = "text-emerald-300"; }
    else if (pct >= 90) { bg = "bg-emerald-900/30"; textColor = "text-emerald-400"; }
    else if (pct >= 75) { bg = "bg-amber-900/30"; textColor = "text-amber-400"; }
    else if (pct >= 50) { bg = "bg-orange-900/30"; textColor = "text-orange-400"; }
    else { bg = "bg-red-900/30"; textColor = "text-red-400"; }
  }

  return (
    <div className={`${bg} rounded-lg p-2 space-y-0.5`}>
      <div className={`text-lg font-bold ${textColor}`}>
        {decode.toFixed(1)}
      </div>
      <div className="text-zinc-500 text-[10px]">tok/s decode</div>
      {pct !== null && (
        <div className={`text-[10px] font-medium ${textColor}`}>
          {pct.toFixed(1)}% baseline
        </div>
      )}
      <div className="text-zinc-600 text-[10px]">
        P: {r.prefillTokPerSec.toFixed(0)} tok/s
      </div>
      {r.compressionRatio && (
        <div className="text-teal-500 text-[10px]">
          {r.compressionRatio.toFixed(1)}x compress
        </div>
      )}
    </div>
  );
}
