// TEAM-476: Server-side data fetching for marketplace models
import type { MarketplaceModel, PaginatedResponse, VendorName } from '@rbee/marketplace-core'
import { getAdapter } from '@rbee/marketplace-core'
import { cache } from 'react'

/**
 * Fetch models from a vendor (server-side, cached)
 *
 * This is a React Server Component function that can be called directly
 * in server components or used in Server Actions.
 *
 * @param vendor - Vendor name (civitai, huggingface)
 * @param filters - Vendor-specific filter parameters
 * @returns Paginated response with models
 */
export const fetchModels = cache(
  async (vendor: VendorName, filters?: unknown): Promise<PaginatedResponse<MarketplaceModel>> => {
    try {
      const adapter = getAdapter(vendor)
      // Type assertion is safe here because getAdapter returns the correct adapter type
      // and each adapter's fetchModels accepts its own filter type
      const response = await adapter.fetchModels(filters)
      return response
    } catch (error) {
      console.error(`[fetchModels] Error fetching ${vendor} models:`, error)
      throw error
    }
  },
)
