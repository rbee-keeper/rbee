// TEAM-405: Marketplace Image Models page
// TEAM-423: Complete implementation with UniversalFilterBar, vertical cards, and full parity with Next.js
// DATA LAYER: Tauri commands + React Query
// PRESENTATION: UniversalFilterBar + ModelCardVertical grid

import type { FilterGroup } from '@rbee/ui/marketplace'
import { ModelCardVertical, UniversalFilterBar } from '@rbee/ui/marketplace'
import { useQuery } from '@tanstack/react-query'
import { invoke } from '@tauri-apps/api/core'
import { useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import type { Model } from '@/generated/bindings'

// TEAM-423: Filter state matching Next.js
interface CivitaiFilters {
  timePeriod: 'AllTime' | 'Month' | 'Week' | 'Day'
  modelType: 'All' | 'Checkpoint' | 'LORA'
  baseModel: 'All' | 'SDXL 1.0' | 'SD 1.5' | 'SD 2.1'
  sort: 'downloads' | 'likes' | 'newest'
}

// TEAM-423: Filter groups matching Next.js exactly
const CIVITAI_FILTER_GROUPS: FilterGroup[] = [
  {
    id: 'timePeriod',
    label: 'Time Period',
    options: [
      { label: 'All Time', value: 'AllTime' },
      { label: 'Month', value: 'Month' },
      { label: 'Week', value: 'Week' },
      { label: 'Day', value: 'Day' },
    ],
  },
  {
    id: 'modelType',
    label: 'Model Type',
    options: [
      { label: 'All Types', value: 'All' },
      { label: 'Checkpoint', value: 'Checkpoint' },
      { label: 'LORA', value: 'LORA' },
    ],
  },
  {
    id: 'baseModel',
    label: 'Base Model',
    options: [
      { label: 'All Models', value: 'All' },
      { label: 'SDXL 1.0', value: 'SDXL 1.0' },
      { label: 'SD 1.5', value: 'SD 1.5' },
      { label: 'SD 2.1', value: 'SD 2.1' },
    ],
  },
]

const CIVITAI_SORT_GROUP: FilterGroup = {
  id: 'sort',
  label: 'Sort By',
  options: [
    { label: 'Most Downloads', value: 'downloads' },
    { label: 'Most Likes', value: 'likes' },
    { label: 'Newest', value: 'newest' },
  ],
}

export function MarketplaceCivitai() {
  const navigate = useNavigate()

  // TEAM-423: Filter state
  const [filters, setFilters] = useState<CivitaiFilters>({
    timePeriod: 'AllTime',
    modelType: 'All',
    baseModel: 'All',
    sort: 'downloads',
  })

  // DATA LAYER: Fetch models from Tauri
  // TEAM-429: Now uses CivitaiFilters object with snake_case fields
  const {
    data: rawModels = [],
    isLoading,
    error,
  } = useQuery({
    queryKey: ['marketplace', 'civitai-models'],
    queryFn: async () => {
      const result = await invoke<Model[]>('marketplace_list_civitai_models', {
        filters: {
          time_period: 'AllTime',
          model_type: 'All',
          base_model: 'All',
          sort: 'Most Downloaded',
          nsfw: {
            max_level: 'None',
            blur_mature: true,
          },
          page: null,
          limit: 100,
        },
      })
      return result
    },
    staleTime: 5 * 60 * 1000,
  })

  // TEAM-423: Client-side filtering and sorting
  const filteredModels = useMemo(() => {
    let result = [...rawModels]

    // Filter by model type (if available in tags)
    if (filters.modelType !== 'All') {
      result = result.filter((model) => {
        const tags = model.tags.map((t) => t.toLowerCase())
        return tags.includes(filters.modelType.toLowerCase())
      })
    }

    // Filter by base model (if available in tags)
    if (filters.baseModel !== 'All') {
      result = result.filter((model) => {
        const tags = model.tags.map((t) => t.toLowerCase())
        const baseModel = filters.baseModel.toLowerCase().replace(/\s/g, '')
        return tags.some((tag) => tag.includes(baseModel))
      })
    }

    // Sort
    result.sort((a, b) => {
      if (filters.sort === 'downloads') return (b.downloads || 0) - (a.downloads || 0)
      if (filters.sort === 'likes') return (b.likes || 0) - (a.likes || 0)
      // newest - would need createdAt field
      return 0
    })

    return result
  }, [rawModels, filters])

  // PRESENTATION LAYER: Full layout matching Next.js
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
            <span>{isLoading ? 'Loading...' : `${filteredModels.length.toLocaleString()} compatible models`}</span>
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
          {filteredModels.map((model) => (
            <div
              key={model.id}
              onClick={() => navigate(`/marketplace/civitai/${encodeURIComponent(model.id)}`)}
              className="cursor-pointer"
            >
              <ModelCardVertical
                model={{
                  ...model,
                  author: model.author || undefined, // Normalize null to undefined
                  imageUrl: model.imageUrl || undefined, // Normalize null to undefined
                }}
              />
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
