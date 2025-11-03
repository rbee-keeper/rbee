// TEAM-391: TypeScript entry point for SD Worker SDK
// Pattern: Same as llm-worker-sdk index.ts

// Re-export everything from the generated WASM bindings
export * from '../pkg/bundler/sd_worker_sdk';

// Export types for TypeScript consumers
export type { TextToImageRequest } from '../pkg/bundler/sd_worker_sdk';
