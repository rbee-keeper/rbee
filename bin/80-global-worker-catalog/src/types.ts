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
 * Build variant for a worker
 * Represents different backend acceleration options (CPU, CUDA, Metal, ROCm)
 * TEAM-481: Consolidated from separate worker entries per backend
 */
export interface BuildVariant {
  /** Backend type (cpu, cuda, metal, rocm) */
  backend: WorkerType

  /** Supported platforms for this variant */
  platforms: Platform[]

  /** Supported architectures for this variant */
  architectures: Architecture[]

  /** URL to PKGBUILD file for this variant */
  pkgbuildUrl: string

  /** Build configuration */
  build: {
    /** Cargo features (for Rust) */
    features?: string[]
    /** Build profile (release, debug) */
    profile?: string
    /** Additional build flags */
    flags?: string[]
  }

  /** Runtime dependencies specific to this variant */
  depends: string[]

  /** Build dependencies specific to this variant */
  makedepends: string[]

  /** Binary name for this variant (e.g., "llm-worker-rbee-cuda") */
  binaryName: string

  /** Installation path for this variant */
  installPath: string
}

/**
 * Worker catalog entry
 * Provides all information needed to download, build, and install a worker
 * TEAM-481: Now represents a single worker type (e.g., "LLM Worker") with multiple build variants
 */
export interface WorkerCatalogEntry {
  // ━━━ Identity ━━━
  /** Unique worker ID (e.g., "llm-worker-rbee") */
  id: string

  /** Worker implementation type */
  implementation: WorkerImplementation

  /** Version (semver) */
  version: string

  // ━━━ Metadata ━━━
  /** Human-readable name (e.g., "LLM Worker") */
  name: string

  /** Short description */
  description: string

  /** License (SPDX identifier) */
  license: string

  // ━━━ Build System ━━━
  /** Build system - MUST be camelCase */
  buildSystem: BuildSystem

  /** Source repository (shared across all variants) */
  source: {
    /** Source type - Rust uses #[serde(rename = "type")] which overrides camelCase */
    type: 'git' | 'tarball'
    url: string
    branch?: string
    tag?: string
    path?: string // Path within repo (e.g., "bin/30_llm_worker_rbee")
  }

  // ━━━ Build Variants ━━━
  /** Available build variants (CPU, CUDA, Metal, ROCm) */
  variants: BuildVariant[]

  // ━━━ Capabilities (shared across all variants) ━━━
  /** Supported model formats - MUST be camelCase */
  supportedFormats: string[]

  /** Maximum context length - MUST be camelCase */
  maxContextLength?: number

  /** Supports streaming - MUST be camelCase */
  supportsStreaming: boolean

  /** Supports batching - MUST be camelCase */
  supportsBatching: boolean
}
