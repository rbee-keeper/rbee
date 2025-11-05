// TEAM-410: Shared compatibility types
// These match the types from @rbee/marketplace-node

export interface CompatibilityResult {
  compatible: boolean
  confidence: 'high' | 'medium' | 'low' | 'none'
  reasons: string[]
  warnings: string[]
  recommendations: string[]
}

export interface Worker {
  id: string
  name: string
  worker_type: 'cpu' | 'cuda' | 'metal'
  platform: 'linux' | 'macos' | 'windows'
  version?: string
  downloadUrl?: string
}
