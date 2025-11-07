// TEAM-415: Shared types for marketplace-node
// TEAM-410: Added compatibility types

export interface ModelFile {
  rfilename: string
  size: number
}

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
  limit?: number
  sort?: 'popular' | 'recent' | 'trending'
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
