// SSE endpoint — runs benchmark configurations and streams results
// Supports cancellation: client aborts fetch → server kills running process

import { CONFIGS } from "@/lib/config";
import type { Backend } from "@/lib/config";
import { runBenchmark } from "@/lib/runner";

export const dynamic = "force-dynamic";

export async function POST(request: Request) {
  const body = await request.json() as {
    models: string[];
    configs: string[];
    backend: Backend;
  };

  const selectedConfigs = CONFIGS.filter((c) => body.configs.includes(c.id));
  const encoder = new TextEncoder();

  // Create an AbortController that fires when the client disconnects
  const abortController = new AbortController();
  request.signal.addEventListener("abort", () => abortController.abort(), { once: true });
  const signal = abortController.signal;

  const stream = new ReadableStream({
    async start(controller) {
      const send = (data: Record<string, unknown>) => {
        try {
          controller.enqueue(encoder.encode(`data: ${JSON.stringify(data)}\n\n`));
        } catch {
          // Stream closed — abort
          abortController.abort();
        }
      };

      const total = body.models.length * selectedConfigs.length;
      let completed = 0;

      send({ type: "start", total, models: body.models, configs: body.configs, backend: body.backend });

      for (const modelPath of body.models) {
        for (const config of selectedConfigs) {
          if (signal.aborted) {
            send({ type: "cancelled", completed });
            controller.close();
            return;
          }

          send({
            type: "running",
            modelPath,
            configId: config.id,
            progress: completed / total,
          });

          const result = await runBenchmark(modelPath, config, body.backend, signal);
          completed++;

          if (signal.aborted) {
            send({ type: "cancelled", completed });
            controller.close();
            return;
          }

          send({
            type: "result",
            ...result,
            progress: completed / total,
          });
        }
      }

      send({ type: "done", total: completed });
      controller.close();
    },
  });

  return new Response(stream, {
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      Connection: "keep-alive",
    },
  });
}
