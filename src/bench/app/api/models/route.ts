// List available GGUF models in the configured directory

import { readdirSync } from "fs";
import { GGUF_DIR } from "@/lib/config";

export async function GET() {
  try {
    const files = readdirSync(GGUF_DIR)
      .filter((f) => f.endsWith(".gguf"))
      .map((f) => {
        const name = f.replace(".gguf", "").replace(/-/g, " ").replace(/_/g, "_");
        const shortName = f.replace(".gguf", "");
        // Extract parameter size: "0.8B" → 0.8, "27B" → 27
        const sizeMatch = f.match(/(\d+\.?\d*)[Bb]/);
        const paramSize = sizeMatch ? parseFloat(sizeMatch[1]) : 0;
        // Extract quant bits for secondary sort: Q8 → 8, Q4 → 4, BF16 → 16
        const quantMatch = f.match(/[Qq](\d+)|BF(\d+)|F(\d+)/);
        const quantBits = quantMatch
          ? parseInt(quantMatch[1] || quantMatch[2] || quantMatch[3])
          : 0;
        return {
          path: `${GGUF_DIR}\\${f}`,
          name,
          shortName,
          filename: f,
          paramSize,
          quantBits,
        };
      })
      .sort((a, b) => a.paramSize - b.paramSize || b.quantBits - a.quantBits || a.name.localeCompare(b.name));

    return Response.json({ models: files, ggufDir: GGUF_DIR });
  } catch (e) {
    return Response.json(
      { error: `Cannot read ${GGUF_DIR}: ${e}`, models: [] },
      { status: 500 },
    );
  }
}
