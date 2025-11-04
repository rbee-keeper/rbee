// TEAM-405: React hooks for marketplace-sdk (HuggingFace, CivitAI, Worker Catalog)
// Provides React Query hooks for marketplace operations via Tauri commands

// Re-export types from marketplace-sdk for convenience
export type { 
  Model,
  ModelSource,
} from '@rbee/marketplace-sdk'

// Export hooks
export { useMarketplaceModels } from './hooks/useMarketplaceModels'
export type { UseMarketplaceModelsResult } from './hooks/useMarketplaceModels'
