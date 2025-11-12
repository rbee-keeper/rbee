// TEAM-482: Global Worker Catalog (GWC) adapter implementation

import type { MarketplaceAdapter } from '../adapter'
import type { MarketplaceModel, PaginatedResponse } from '../common'
import { fetchGWCWorker } from './details'
import { fetchGWCWorkers } from './list'
import type { GWCListWorkersParams } from './types'

/**
 * GWC Marketplace Adapter
 * Implements the unified MarketplaceAdapter interface for Global Worker Catalog
 */
export const gwcAdapter: MarketplaceAdapter<GWCListWorkersParams> = {
  name: 'gwc',

  async fetchModels(params?: GWCListWorkersParams): Promise<PaginatedResponse<MarketplaceModel>> {
    return fetchGWCWorkers(params)
  },

  async fetchModel(id: string | number): Promise<MarketplaceModel> {
    return fetchGWCWorker(String(id))
  },
}
