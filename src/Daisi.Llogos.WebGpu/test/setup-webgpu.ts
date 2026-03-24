/**
 * Setup Dawn WebGPU for Node.js tests.
 * Import this before any code that uses navigator.gpu.
 */
import { create, globals } from 'webgpu';

// Polyfill browser globals
Object.assign(globalThis, globals);
Object.defineProperty(globalThis, 'navigator', {
  value: { gpu: create([]) },
  writable: true,
  configurable: true,
});
