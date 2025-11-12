// TEAM-476: HuggingFace adapter - Unified exports

import type { MarketplaceAdapter } from '../adapter'
import type { MarketplaceModel, PaginatedResponse } from '../common'
import type { HuggingFaceListModelsParams } from './types'

// Re-export details functions
export { fetchHuggingFaceModel, fetchHuggingFaceModelReadme } from './details'
// Re-export list functions
export { convertHFModel, fetchHuggingFaceModels } from './list'

// Re-export types
export type * from './types'

import { fetchHuggingFaceModel } from './details'
// Import functions for adapter implementation
import { fetchHuggingFaceModels } from './list'

/**
 * HuggingFace Adapter - implements MarketplaceAdapter with HuggingFace-specific filters
 */
export const huggingfaceAdapter: MarketplaceAdapter<HuggingFaceListModelsParams> = {
  name: 'huggingface',

  async fetchModels(params: HuggingFaceListModelsParams = {}): Promise<PaginatedResponse<MarketplaceModel>> {
    // Pass params directly - no mapping needed!
    return fetchHuggingFaceModels(params)
  },

  async fetchModel(id: string | number): Promise<MarketplaceModel> {
    const modelId = String(id)
    return fetchHuggingFaceModel(modelId)
  },
}
