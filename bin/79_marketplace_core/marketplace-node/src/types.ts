// TEAM-415: Shared types for marketplace-node
// TEAM-410: Added compatibility types
// TEAM-463: Import ModelFile from WASM SDK (source of truth: artifacts-contract)

// TEAM-463: Re-export ModelFile from WASM SDK instead of duplicating
// Source of truth: bin/97_contracts/artifacts-contract/src/model/mod.rs
import type { ModelFile as WasmModelFile } from '../wasm/marketplace_sdk'
export type ModelFile = WasmModelFile

export interface Model {
  id: string
  name: string
  author?: string
  description: string
  downloads: number
  likes: number
  size: string
  tags: string[]
  source: 'huggingface' | 'civitai'
  imageUrl?: string
  createdAt?: string
  lastModified?: string
  config?: any
  siblings?: ModelFile[]
}

export interface SearchOptions {
  query?: string
  limit?: number
  offset?: number  // TEAM-462: For pagination
  page?: number    // TEAM-462: For pagination (CivitAI uses page numbers)
  sort?: string
  filter?: string
}

export interface Worker {
  id: string
  name: string
  type: 'cpu' | 'cuda' | 'metal'
  platform: 'linux' | 'macos' | 'windows'
  version: string
  downloadUrl: string
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// TEAM-410: COMPATIBILITY TYPES
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/**
 * Model metadata for compatibility checking
 */
export interface ModelMetadata {
  architecture: string
  format: string
  quantization: string | null
  parameters: string
  sizeBytes: number
  maxContextLength: number
}

/**
 * Compatibility confidence level
 */
export type CompatibilityConfidence = 'high' | 'medium' | 'low' | 'none'

/**
 * Compatibility check result
 */
export interface CompatibilityResult {
  compatible: boolean
  confidence: CompatibilityConfidence
  reasons: string[]
  warnings: string[]
  recommendations: string[]
}
