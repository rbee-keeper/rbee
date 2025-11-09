// TEAM-404: Worker Catalog Types
// CANONICAL SOURCE: bin/97_contracts/artifacts-contract/src/worker.rs
// Generated TypeScript types: bin/99_shared_crates/marketplace-sdk/pkg/bundler/marketplace_sdk.d.ts
//
// These types are AUTO-GENERATED from Rust via WASM/tsify
// DO NOT modify these manually - they come from artifacts-contract via marketplace-sdk

/**
 * Worker type (backend acceleration)
 * CANONICAL SOURCE: artifacts-contract::WorkerType
 * Generated via: marketplace-sdk WASM build
 * TEAM-461: Added ROCm support for AMD GPUs
 */
export type WorkerType = 'cpu' | 'cuda' | 'metal' | 'rocm'

/**
 * Platform (operating system)
 * CANONICAL SOURCE: artifacts-contract::Platform
 * Generated via: marketplace-sdk WASM build
 */
export type Platform = 'linux' | 'macos' | 'windows'

/**
 * Architecture (CPU instruction set)
 */
export type Architecture = 'x86_64' | 'aarch64'

/**
 * Worker implementation type
 * TEAM-421: Must match Rust enum WorkerImplementation in artifacts-contract
 */
export type WorkerImplementation = 'rust' | 'python' | 'cpp'

/**
 * Build system type
 */
export type BuildSystem = 'cargo' | 'cmake' | 'pip' | 'npm'

/**
 * Worker catalog entry
 * Provides all information needed to download, build, and install a worker
 */
export interface WorkerCatalogEntry {
  // ━━━ Identity ━━━
  /** Unique worker ID (e.g., "llm-worker-rbee-cpu") */
  id: string

  /** Worker implementation type */
  implementation: WorkerImplementation

  /** Worker type (backend) - MUST be camelCase to match Rust */
  workerType: WorkerType

  /** Version (semver) */
  version: string

  // ━━━ Platform Support ━━━
  /** Supported platforms */
  platforms: Platform[]

  /** Supported architectures */
  architectures: Architecture[]

  // ━━━ Metadata ━━━
  /** Human-readable name */
  name: string

  /** Short description */
  description: string

  /** License (SPDX identifier) */
  license: string

  // ━━━ Build Instructions ━━━
  /** URL to PKGBUILD file - MUST be camelCase */
  pkgbuildUrl: string

  /** Build system - MUST be camelCase */
  buildSystem: BuildSystem

  /** Source repository */
  source: {
    /** Source type - Rust uses #[serde(rename = "type")] which overrides camelCase */
    type: 'git' | 'tarball'
    url: string
    branch?: string
    tag?: string
    path?: string // Path within repo (e.g., "bin/30_llm_worker_rbee")
  }

  /** Build configuration */
  build: {
    /** Cargo features (for Rust) */
    features?: string[]
    /** Build profile (release, debug) */
    profile?: string
    /** Additional build flags */
    flags?: string[]
  }

  // ━━━ Dependencies ━━━
  /** Runtime dependencies */
  depends: string[]

  /** Build dependencies */
  makedepends: string[]

  // ━━━ Binary Info ━━━
  /** Binary name (output) - MUST be camelCase */
  binaryName: string

  /** Installation path - MUST be camelCase */
  installPath: string

  // ━━━ Capabilities ━━━
  /** Supported model formats - MUST be camelCase */
  supportedFormats: string[]

  /** Maximum context length - MUST be camelCase */
  maxContextLength?: number

  /** Supports streaming - MUST be camelCase */
  supportsStreaming: boolean

  /** Supports batching - MUST be camelCase */
  supportsBatching: boolean
}
