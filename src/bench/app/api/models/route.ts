// List available GGUF models in the configured directory

import { readdirSync } from "fs";
import { GGUF_DIR } from "@/lib/config";

export async function GET() {
  try {
    const files = readdirSync(GGUF_DIR)
      .filter((f) => f.endsWith(".gguf"))
      .map((f) => {
        // Parse model name from filename: "Qwen3.5-0.8B-Q8_0.gguf" → "Qwen3.5 0.8B Q8_0"
        const name = f.replace(".gguf", "").replace(/-/g, " ").replace(/_/g, "_");
        const shortName = f.replace(".gguf", "");
        return {
          path: `${GGUF_DIR}\\${f}`,
          name,
          shortName,
          filename: f,
        };
      })
      .sort((a, b) => a.name.localeCompare(b.name));

    return Response.json({ models: files, ggufDir: GGUF_DIR });
  } catch (e) {
    return Response.json(
      { error: `Cannot read ${GGUF_DIR}: ${e}`, models: [] },
      { status: 500 },
    );
  }
}
