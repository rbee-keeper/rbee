// TEAM-477: CivitAI models list page for Tauri
// Uses @rbee/marketplace-core adapters (client-side fetching)
// Uses UniversalFilterBar + ModelCardVertical (reusable components)

import type { CivitAIListModelsParams } from '@rbee/marketplace-core'
import { fetchCivitAIModels } from '@rbee/marketplace-core'
import { CIVITAI_FILTER_GROUPS, CIVITAI_SORT_GROUP, ModelCardVertical, UniversalFilterBar } from '@rbee/ui/marketplace'
import { useQuery } from '@tanstack/react-query'
import { useState } from 'react'
import { useNavigate } from 'react-router-dom'

// UI filter state
interface CivitAIUIFilters {
  sort?: CivitAIListModelsParams['sort']
  types?: CivitAIListModelsParams['types']
}

export function CAIListPage() {
  const navigate = useNavigate()

  // Filter state
  const [filters, setFilters] = useState<CivitAIUIFilters>({
    sort: 'Highest Rated',
    types: undefined,
  })

  // Fetch models using marketplace-core adapter
  const { data, isLoading, error } = useQuery({
    queryKey: ['civitai-models', filters],
    queryFn: async () => {
      const params: CivitAIListModelsParams = {
        sort: filters.sort,
        types: filters.types,
        limit: 50,
      }
      const response = await fetchCivitAIModels(params)
      return response.items
    },
    staleTime: 5 * 60 * 1000, // 5 minutes
  })

  const models = data || []

  return (
    <div className="container mx-auto px-4 py-12 max-w-7xl">
      {/* Header Section */}
      <div className="mb-8 space-y-4">
        <div className="space-y-2">
          <h1 className="text-4xl md:text-5xl font-bold tracking-tight">Civitai Models</h1>
          <p className="text-muted-foreground text-lg md:text-xl max-w-3xl">
            Discover and download Stable Diffusion models from Civitai's community
          </p>
        </div>

        {/* Stats */}
        <div className="flex items-center gap-6 text-sm text-muted-foreground">
          <div className="flex items-center gap-2">
            <div className="size-2 rounded-full bg-primary" />
            <span>{isLoading ? 'Loading...' : `${models.length.toLocaleString()} compatible models`}</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="size-2 rounded-full bg-green-500" />
            <span>Checkpoints & LORAs</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="size-2 rounded-full bg-blue-500" />
            <span>Safe for work</span>
          </div>
        </div>
      </div>

      {/* Filter Bar */}
      <UniversalFilterBar
        groups={CIVITAI_FILTER_GROUPS}
        sortGroup={CIVITAI_SORT_GROUP}
        currentFilters={filters}
        onFiltersChange={(newFilters) => {
          setFilters({ ...filters, ...newFilters })
        }}
      />

      {/* Loading/Error States */}
      {isLoading && <div className="text-center py-12 text-muted-foreground">Loading models...</div>}
      {error && <div className="text-center py-12 text-destructive">Error: {String(error)}</div>}

      {/* Vertical Card Grid - Portrait aspect ratio for CivitAI images */}
      {!isLoading && !error && (
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-4">
          {models.map((model) => (
            <div
              key={model.id}
              onClick={() => navigate(`/marketplace/civitai/${encodeURIComponent(model.id)}`)}
              className="cursor-pointer"
            >
              <ModelCardVertical
                model={{
                  id: model.id,
                  name: model.name,
                  description: model.description ?? '',
                  ...(model.author ? { author: model.author } : {}),
                  ...(model.imageUrl ? { imageUrl: model.imageUrl } : {}),
                  tags: model.tags,
                  downloads: model.downloads,
                  likes: model.likes,
                  size: model.sizeBytes ? String(model.sizeBytes) : '0',
                }}
              />
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
