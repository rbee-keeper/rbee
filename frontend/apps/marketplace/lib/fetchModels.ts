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
export const fetchModels = cache(async <TFilters = unknown>(
  vendor: VendorName,
  filters?: TFilters
): Promise<PaginatedResponse<MarketplaceModel>> => {
  try {
    const adapter = getAdapter(vendor)
    const response = await adapter.fetchModels(filters as any)
    return response
  } catch (error) {
    console.error(`[fetchModels] Error fetching ${vendor} models:`, error)
    throw error
  }
})
