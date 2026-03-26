// LLogos Bench — benchmark configurations

export const GGUF_DIR = process.env.GGUF_DIR || "C:\\GGUFS";
export const LLOGOS_PROJECT = process.env.LLOGOS_PROJECT ||
  "C:\\repos\\daisinet-turbo-quant\\daisi-llogos\\src\\dotnet\\Daisi.Llogos.Cli";

export interface BenchConfig {
  id: string;
  label: string;
  shortLabel: string;
  args: string[];  // extra CLI args beyond --model, --bench, --prompt
  color: string;   // tailwind color class for the column header
}

export const CONFIGS: BenchConfig[] = [
  {
    id: "baseline",
    label: "Baseline (Partial Vocab)",
    shortLabel: "F16 PV",
    args: ["--vocab-limit", "32"],
    color: "bg-slate-700",
  },
  {
    id: "baseline-fullvocab",
    label: "Baseline (Full Vocab)",
    shortLabel: "F16 Full",
    args: ["--vocab-limit", "1"],
    color: "bg-slate-600",
  },
  {
    id: "turbo4",
    label: "Turbo 4-bit (Partial Vocab)",
    shortLabel: "T4 PV",
    args: ["--kv-quant", "turbo:4+noqjl", "--vocab-limit", "32"],
    color: "bg-emerald-700",
  },
  {
    id: "turbo3",
    label: "Turbo 3-bit",
    shortLabel: "T3",
    args: ["--kv-quant", "turbo:3+noqjl", "--vocab-limit", "32"],
    color: "bg-emerald-600",
  },
  {
    id: "turbo2",
    label: "Turbo 2-bit",
    shortLabel: "T2",
    args: ["--kv-quant", "turbo:2+noqjl", "--vocab-limit", "32"],
    color: "bg-emerald-500",
  },
  {
    id: "turbo3-qjl",
    label: "Turbo 3-bit + QJL",
    shortLabel: "T3+Q",
    args: ["--kv-quant", "turbo:3", "--vocab-limit", "32"],
    color: "bg-teal-600",
  },
  {
    id: "turbo4-fullvocab",
    label: "Turbo 4-bit (Full Vocab)",
    shortLabel: "T4 Full",
    args: ["--kv-quant", "turbo:4+noqjl", "--vocab-limit", "1"],
    color: "bg-teal-500",
  },
];

export interface ModelEntry {
  path: string;
  name: string;        // display name
  shortName: string;   // for compact grid
}

export const BACKENDS = ["cpu", "cuda"] as const;
export type Backend = typeof BACKENDS[number];

export interface ContextPreset {
  id: string;
  label: string;
  prompt: string;
  maxTokens: number;
  maxContext: number;
}

export const CONTEXT_PRESETS: ContextPreset[] = [
  {
    id: "short",
    label: "Short (~150 tokens)",
    prompt: "Explain the theory of relativity in simple terms. Start from the basics and build up to the key insights that Einstein discovered about space, time, and gravity.",
    maxTokens: 128,
    maxContext: 2048,
  },
  {
    id: "long",
    label: "Long (~4K tokens)",
    prompt: Array(400).fill("The history of science spans many centuries of human endeavor and discovery with remarkable breakthroughs. ").join(""),
    maxTokens: 64,
    maxContext: 8192,
  },
];
