// TEAM-482: Global Worker Catalog (GWC) adapter implementation

import type { MarketplaceAdapter } from '../adapter'
import type { MarketplaceModel, PaginatedResponse } from '../common'
import { fetchGWCWorker } from './details'
import { convertGWCWorker, fetchGWCWorkers } from './list'
import type { GWCListWorkersParams } from './types'

/**
 * GWC Marketplace Adapter
 * Implements the unified MarketplaceAdapter interface for Global Worker Catalog
 */
export const gwcAdapter: MarketplaceAdapter<GWCListWorkersParams> = {
  name: 'gwc',

  async fetchModels(params?: GWCListWorkersParams): Promise<PaginatedResponse<MarketplaceModel>> {
    const workers = await fetchGWCWorkers(params)
    const items: MarketplaceModel[] = workers.map(convertGWCWorker)
    const limit = params?.limit ?? items.length

    return {
      items: items.slice(0, limit),
      meta: {
        page: 1,
        limit,
        total: items.length,
        hasNext: false,
      },
    }
  },

  async fetchModel(id: string | number): Promise<MarketplaceModel> {
    return fetchGWCWorker(String(id))
  },
}
