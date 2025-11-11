// TEAM-405: Marketplace Image Models page
// TEAM-423: Complete implementation with UniversalFilterBar, vertical cards, and full parity with Next.js
// DATA LAYER: Tauri commands + React Query
// PRESENTATION: UniversalFilterBar + ModelCardVertical grid

// TEAM-XXX RULE ZERO: Import constants and utilities from @rbee/marketplace-node (source of truth)
import { applyCivitAIFilters, FILTER_DEFAULTS } from '@rbee/marketplace-node'
import type { BaseModel, CivitaiModelType, CivitaiSort, FilterableModel, TimePeriod } from '@rbee/marketplace-node'
import {
  CIVITAI_FILTER_GROUPS,
  CIVITAI_SORT_GROUP,
  ModelCardVertical,
  UniversalFilterBar,
} from '@rbee/ui/marketplace'
import { useQuery } from '@tanstack/react-query'
import { invoke } from '@tauri-apps/api/core'
import { useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import type { Model } from '@/generated/bindings'

// TEAM-467: UI filter state (subset of CivitaiFilters API type)
interface CivitaiUIFilters {
  timePeriod: TimePeriod
  modelType: CivitaiModelType
  baseModel: BaseModel
  sort: CivitaiSort
}

export function MarketplaceCivitai() {
  const navigate = useNavigate()

  // TEAM-467: Filter state using API enum values
  const [filters, setFilters] = useState<CivitaiUIFilters>({
    timePeriod: FILTER_DEFAULTS.CIVITAI_TIME_PERIOD as TimePeriod,
    modelType: FILTER_DEFAULTS.CIVITAI_MODEL_TYPE as CivitaiModelType,
    baseModel: FILTER_DEFAULTS.CIVITAI_BASE_MODEL as BaseModel,
    sort: FILTER_DEFAULTS.CIVITAI_SORT as CivitaiSort,
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
          time_period: FILTER_DEFAULTS.CIVITAI_TIME_PERIOD,
          model_type: FILTER_DEFAULTS.CIVITAI_MODEL_TYPE,
          base_model: FILTER_DEFAULTS.CIVITAI_BASE_MODEL,
          sort: FILTER_DEFAULTS.CIVITAI_SORT,
          nsfw: {
            max_level: FILTER_DEFAULTS.CIVITAI_NSFW_LEVEL,
            blur_mature: FILTER_DEFAULTS.CIVITAI_BLUR_MATURE,
          },
          page: null,
          limit: FILTER_DEFAULTS.CIVITAI_LIMIT,
        },
      })
      return result
    },
    staleTime: 5 * 60 * 1000,
  })

  // TEAM-XXX: Use shared filter utilities from marketplace-node
  const filteredModels = useMemo(() => {
    return applyCivitAIFilters(rawModels as FilterableModel[], {
      modelType: filters.modelType,
      baseModel: filters.baseModel,
      sort: filters.sort,
    })
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
