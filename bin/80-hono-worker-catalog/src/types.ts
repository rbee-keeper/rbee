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
 */
export type WorkerType = "cpu" | "cuda" | "metal";

/**
 * Platform (operating system)
 * CANONICAL SOURCE: artifacts-contract::Platform
 * Generated via: marketplace-sdk WASM build
 */
export type Platform = "linux" | "macos" | "windows";

/**
 * Architecture (CPU instruction set)
 */
export type Architecture = "x86_64" | "aarch64";

/**
 * Worker implementation type
 * - "llm-worker-rbee": Bespoke Candle-based worker (our implementation)
 * - "llama-cpp-adapter": llama.cpp wrapper (future)
 * - "vllm-adapter": vLLM wrapper (future)
 * - "ollama-adapter": Ollama wrapper (future)
 * - "comfyui-adapter": ComfyUI wrapper (future)
 */
export type WorkerImplementation = 
  | "llm-worker-rbee"
  | "llama-cpp-adapter"
  | "vllm-adapter"
  | "ollama-adapter"
  | "comfyui-adapter";

/**
 * Build system type
 */
export type BuildSystem = "cargo" | "cmake" | "pip" | "npm";

/**
 * Worker catalog entry
 * Provides all information needed to download, build, and install a worker
 */
export interface WorkerCatalogEntry {
  // ━━━ Identity ━━━
  /** Unique worker ID (e.g., "llm-worker-rbee-cpu") */
  id: string;
  
  /** Worker implementation type */
  implementation: WorkerImplementation;
  
  /** Worker type (backend) */
  worker_type: WorkerType;
  
  /** Version (semver) */
  version: string;
  
  // ━━━ Platform Support ━━━
  /** Supported platforms */
  platforms: Platform[];
  
  /** Supported architectures */
  architectures: Architecture[];
  
  // ━━━ Metadata ━━━
  /** Human-readable name */
  name: string;
  
  /** Short description */
  description: string;
  
  /** License (SPDX identifier) */
  license: string;
  
  // ━━━ Build Instructions ━━━
  /** URL to PKGBUILD file */
  pkgbuild_url: string;
  
  /** Build system */
  build_system: BuildSystem;
  
  /** Source repository */
  source: {
    type: "git" | "tarball";
    url: string;
    branch?: string;
    tag?: string;
    path?: string;  // Path within repo (e.g., "bin/30_llm_worker_rbee")
  };
  
  /** Build configuration */
  build: {
    /** Cargo features (for Rust) */
    features?: string[];
    /** Build profile (release, debug) */
    profile?: string;
    /** Additional build flags */
    flags?: string[];
  };
  
  // ━━━ Dependencies ━━━
  /** Runtime dependencies */
  depends: string[];
  
  /** Build dependencies */
  makedepends: string[];
  
  // ━━━ Binary Info ━━━
  /** Binary name (output) */
  binary_name: string;
  
  /** Installation path */
  install_path: string;
  
  // ━━━ Capabilities ━━━
  /** Supported model formats */
  supported_formats: string[];
  
  /** Maximum context length */
  max_context_length?: number;
  
  /** Supports streaming */
  supports_streaming: boolean;
  
  /** Supports batching */
  supports_batching: boolean;
}
