import { defineConfig } from 'vitest/config';
import { readFileSync } from 'fs';

export default defineConfig({
  plugins: [
    {
      name: 'wgsl-loader',
      transform(code, id) {
        if (id.endsWith('.wgsl')) {
          const contents = readFileSync(id, 'utf8');
          return { code: `export default ${JSON.stringify(contents)};` };
        }
      },
    },
  ],
  test: {
    include: ['test/**/*.test.ts'],
    testTimeout: 120000,
  },
});
