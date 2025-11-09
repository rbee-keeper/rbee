// TEAM-391: TypeScript entry point for SD Worker SDK
// Pattern: Same as llm-worker-sdk index.ts

// Export types for TypeScript consumers
export type { TextToImageRequest } from '../pkg/bundler/sd_worker_sdk'
// Re-export everything from the generated WASM bindings
export * from '../pkg/bundler/sd_worker_sdk'
