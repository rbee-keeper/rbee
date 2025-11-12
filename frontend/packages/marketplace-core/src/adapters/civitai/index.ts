// TEAM-476: CivitAI adapter - Unified exports

import type { MarketplaceAdapter } from '../adapter'
import type { MarketplaceModel, PaginatedResponse } from '../common'
import type { CivitAIListModelsParams } from './types'

// Re-export details functions
export { fetchCivitAIModel } from './details'
// Re-export list functions
export { convertCivitAIModel, fetchCivitAIModels } from './list'

// Re-export types
export type * from './types'

import { fetchCivitAIModel } from './details'
// Import functions for adapter implementation
import { fetchCivitAIModels } from './list'

/**
 * CivitAI Adapter - implements MarketplaceAdapter with CivitAI-specific filters
 */
export const civitaiAdapter: MarketplaceAdapter<CivitAIListModelsParams> = {
  name: 'civitai',

  async fetchModels(params: CivitAIListModelsParams = {}): Promise<PaginatedResponse<MarketplaceModel>> {
    // Pass params directly - no mapping needed!
    return fetchCivitAIModels(params)
  },

  async fetchModel(id: string | number): Promise<MarketplaceModel> {
    const modelId = typeof id === 'string' ? Number.parseInt(id, 10) : id
    return fetchCivitAIModel(modelId)
  },
}
