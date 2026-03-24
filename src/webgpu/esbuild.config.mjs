import { build, context } from 'esbuild';
import { readFileSync, readdirSync } from 'fs';
import { join, resolve } from 'path';

// Inline .wgsl files as strings
const wgslPlugin = {
  name: 'wgsl-loader',
  setup(build) {
    build.onLoad({ filter: /\.wgsl$/ }, (args) => {
      const contents = readFileSync(args.path, 'utf8');
      return { contents: `export default ${JSON.stringify(contents)};`, loader: 'js' };
    });
  },
};

const isWatch = process.argv.includes('--watch');

const buildOptions = {
  entryPoints: ['src/index.ts'],
  bundle: true,
  format: 'esm',
  outfile: 'dist/index.js',
  sourcemap: true,
  target: 'es2022',
  platform: 'browser',
  plugins: [wgslPlugin],
};

if (isWatch) {
  const ctx = await context(buildOptions);
  await ctx.watch();
  console.log('Watching for changes...');
} else {
  await build(buildOptions);
  console.log('Build complete.');
}
