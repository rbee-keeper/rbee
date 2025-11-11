// TEAM-477: HuggingFace models list page for Tauri
// Uses @rbee/marketplace-core adapters (client-side fetching)
// Uses UniversalFilterBar + ModelTable (reusable components)

import type { HuggingFaceListModelsParams } from '@rbee/marketplace-core'
import { fetchHuggingFaceModels } from '@rbee/marketplace-core'
import { HUGGINGFACE_FILTER_GROUPS, HUGGINGFACE_SORT_GROUP, ModelTable, UniversalFilterBar } from '@rbee/ui/marketplace'
import { useQuery } from '@tanstack/react-query'
import { useState } from 'react'
import { useNavigate } from 'react-router-dom'

// UI filter state
interface HuggingFaceUIFilters {
  sort?: HuggingFaceListModelsParams['sort']
  library?: HuggingFaceListModelsParams['library']
}

export function HFListPage() {
  const navigate = useNavigate()

  // Filter state
  const [filters, setFilters] = useState<HuggingFaceUIFilters>({
    sort: 'downloads',
    library: undefined,
  })

  // Fetch models using marketplace-core adapter
  const { data, isLoading, error } = useQuery({
    queryKey: ['huggingface-models', filters],
    queryFn: async () => {
      const params: HuggingFaceListModelsParams = {
        sort: filters.sort,
        library: filters.library,
        limit: 50,
      }
      const response = await fetchHuggingFaceModels(params)
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
          <h1 className="text-4xl md:text-5xl font-bold tracking-tight">HuggingFace LLM Models</h1>
          <p className="text-muted-foreground text-lg md:text-xl max-w-3xl">
            Discover and download state-of-the-art language models
          </p>
        </div>

        {/* Stats */}
        <div className="flex items-center gap-6 text-sm text-muted-foreground">
          <div className="flex items-center gap-2">
            <div className="size-2 rounded-full bg-primary" />
            <span>{isLoading ? 'Loading...' : `${models.length.toLocaleString()} models`}</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="size-2 rounded-full bg-orange-500" />
            <span>HuggingFace Hub</span>
          </div>
        </div>
      </div>

      {/* Filter Bar */}
      <UniversalFilterBar
        groups={HUGGINGFACE_FILTER_GROUPS}
        sortGroup={HUGGINGFACE_SORT_GROUP}
        currentFilters={filters}
        onFiltersChange={(newFilters) => {
          setFilters({ ...filters, ...newFilters })
        }}
      />

      {/* Loading/Error States */}
      {isLoading && <div className="text-center py-12 text-muted-foreground">Loading models...</div>}
      {error && <div className="text-center py-12 text-destructive">Error: {String(error)}</div>}

      {/* Table */}
      {!isLoading && !error && (
        <div className="rounded-lg border border-border bg-card p-6">
          <ModelTable
            models={models.map((m) => ({
              id: m.id,
              name: m.name,
              description: m.description,
              author: m.author,
              downloads: m.downloads,
              likes: m.likes,
              tags: m.tags,
              size: m.sizeBytes,
              imageUrl: m.imageUrl,
            }))}
            onModelClick={(modelId: string) => navigate(`/marketplace/huggingface/${encodeURIComponent(modelId)}`)}
          />
        </div>
      )}
    </div>
  )
}
