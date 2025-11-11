// TEAM-476: Model page container - DATA + CONTROL layer (filters, sorting, fetching)
import type { MarketplaceModel, VendorName } from '@rbee/marketplace-core'
import { FeatureHeader } from '@rbee/ui/molecules/FeatureHeader'
import type { ReactNode } from 'react'
import { fetchModels } from '@/lib/fetchModels'

export interface ModelPageContainerProps<TFilters = unknown> {
  /** Vendor name */
  vendor: VendorName

  /** Page title */
  title: string

  /** Page subtitle (optional) */
  subtitle?: string

  /** Vendor-specific filters */
  filters?: TFilters

  /** Filter bar component (LEFT side) */
  filterBar?: ReactNode

  /** Render function for model display (PRESENTATION layer) */
  children: (props: ModelPageRenderProps) => ReactNode
}

export interface ModelPageRenderProps {
  models: MarketplaceModel[]
  pagination: {
    page: number
    limit: number
    total?: number
    hasNext: boolean
  }
}

/**
 * Model page container - handles DATA + CONTROL layer
 *
 * - Fetches models from vendor (SSR)
 * - Renders page header (title + subtitle)
 * - Renders filter bar (if provided)
 * - Delegates model display to children (PRESENTATION layer)
 */
export async function ModelPageContainer<TFilters = unknown>({
  vendor,
  title,
  subtitle,
  filters,
  filterBar,
  children,
}: ModelPageContainerProps<TFilters>) {
  // Fetch models on server (SSR!)
  const response = await fetchModels(vendor, filters)

  return (
    <div className="container mx-auto py-8">
      <div className="space-y-4 mb-8">
        <FeatureHeader title={title} subtitle={subtitle || ''} />
        {filterBar}
      </div>

      {children({
        models: response.items,
        pagination: {
          page: response.meta.page,
          limit: response.meta.limit,
          ...(response.meta.total !== undefined && { total: response.meta.total }),
          hasNext: response.meta.hasNext,
        },
      })}
    </div>
  )
}
