/**
 * Test environment type declarations
 * Provides proper types for test files without requiring @types/node in production builds
 */

declare global {
  // Node.js global object for test environment
  var global: typeof globalThis
}

export {}
