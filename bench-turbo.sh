#!/bin/bash
# DaisiTurbo Benchmark Panel
# Compares baseline vs TurboQuant at various bit-widths and vocab-limit settings

MODEL="C:/GGUFS/Qwen3.5-0.8B-Q8_0.gguf"
PROJECT="src/dotnet/Daisi.Llogos.Cli"
PROMPT="Explain the theory of relativity in simple terms. Start from the basics and build up to the key insights that Einstein discovered."
TOKENS=128
RESULTS_FILE="bench-turbo-results.txt"

echo "============================================================" | tee "$RESULTS_FILE"
echo "  DaisiTurbo Benchmark Panel" | tee -a "$RESULTS_FILE"
echo "  Model: $MODEL" | tee -a "$RESULTS_FILE"
echo "  Tokens: $TOKENS" | tee -a "$RESULTS_FILE"
echo "  Date: $(date)" | tee -a "$RESULTS_FILE"
echo "============================================================" | tee -a "$RESULTS_FILE"
echo "" | tee -a "$RESULTS_FILE"

run_bench() {
    local label="$1"
    shift
    echo "── $label ──" | tee -a "$RESULTS_FILE"
    dotnet run --project "$PROJECT" -c Release -- \
        --model "$MODEL" \
        --bench \
        --prompt "$PROMPT" \
        --max-tokens "$TOKENS" \
        "$@" 2>&1 | tee -a "$RESULTS_FILE"
    echo "" | tee -a "$RESULTS_FILE"
}

# Build once in Release mode
echo "Building Release..." | tee -a "$RESULTS_FILE"
dotnet build "$PROJECT" -c Release -v minimal 2>&1 | tail -3 | tee -a "$RESULTS_FILE"
echo "" | tee -a "$RESULTS_FILE"

# ── Baseline (no TurboQuant) ──────────────────────────────────────────
run_bench "Baseline (F16 KV, vocab-limit 32)" --vocab-limit 32
run_bench "Baseline (F16 KV, vocab-limit 1)"  --vocab-limit 1

# ── TurboQuant 4-bit ──────────────────────────────────────────────────
run_bench "Turbo 4-bit (vocab-limit 32)" --kv-quant turbo:4 --vocab-limit 32
run_bench "Turbo 4-bit (vocab-limit 1)"  --kv-quant turbo:4 --vocab-limit 1

# ── TurboQuant 3-bit (default) ────────────────────────────────────────
run_bench "Turbo 3-bit (vocab-limit 32)" --kv-quant turbo:3 --vocab-limit 32
run_bench "Turbo 3-bit (vocab-limit 1)"  --kv-quant turbo:3 --vocab-limit 1

# ── TurboQuant 2-bit (extreme) ────────────────────────────────────────
run_bench "Turbo 2-bit (vocab-limit 32)" --kv-quant turbo:2 --vocab-limit 32
run_bench "Turbo 2-bit (vocab-limit 1)"  --kv-quant turbo:2 --vocab-limit 1

# ── TurboQuant 3-bit + QJL variants ──────────────────────────────────
run_bench "Turbo 3-bit+qjl32 (vocab-limit 32)" --kv-quant turbo:3+qjl32 --vocab-limit 32
run_bench "Turbo 3-bit+noqjl (vocab-limit 32)"  --kv-quant turbo:3+noqjl --vocab-limit 32

echo "============================================================" | tee -a "$RESULTS_FILE"
echo "  Benchmark complete. Results saved to $RESULTS_FILE" | tee -a "$RESULTS_FILE"
echo "============================================================" | tee -a "$RESULTS_FILE"
