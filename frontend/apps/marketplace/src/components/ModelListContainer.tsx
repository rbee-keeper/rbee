// TEAM-476: Model List Container - handles data layer for ALL vendors (vendor-specific filters)
'use client'

import type { MarketplaceModel, PaginatedResponse, VendorName } from '@rbee/marketplace-core'
import { getAdapter } from '@rbee/marketplace-core'
import { type ReactNode, useEffect, useState } from 'react'

export interface ModelListContainerProps<TFilters = unknown> {
  vendor: VendorName
  filters?: TFilters
  children: (props: ModelListRenderProps) => ReactNode
}

export interface ModelListRenderProps {
  models: MarketplaceModel[]
  loading: boolean
  error: Error | null
  pagination: {
    page: number
    limit: number
    total?: number
    hasNext: boolean
  }
  refetch: () => void
}

export function ModelListContainer<TFilters = unknown>({
  vendor,
  filters,
  children,
}: ModelListContainerProps<TFilters>) {
  const [models, setModels] = useState<MarketplaceModel[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<Error | null>(null)
  const [pagination, setPagination] = useState({
    page: 1,
    limit: 50,
    total: undefined as number | undefined,
    hasNext: false,
  })

  const fetchModels = async () => {
    setLoading(true)
    setError(null)

    try {
      // Get the adapter for this vendor (unified interface!)
      const adapter = getAdapter(vendor)

      // Fetch models using vendor-specific filters
      const response = await adapter.fetchModels(filters as any)

      setModels(response.items)
      setPagination((prev) => ({
        ...prev,
        total: response.meta.total,
        hasNext: response.meta.hasNext,
      }))
    } catch (err) {
      console.error(`[ModelListContainer] Error fetching ${vendor} models:`, err)
      setError(err instanceof Error ? err : new Error('Failed to fetch models'))
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchModels()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [vendor, pagination.page])

  return (
    <>
      {children({
        models,
        loading,
        error,
        pagination,
        refetch: fetchModels,
      })}
    </>
  )
}
