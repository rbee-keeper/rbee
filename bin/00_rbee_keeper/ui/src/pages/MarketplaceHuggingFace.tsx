// TEAM-405: Marketplace LLM Models page - Using reusable components
// TEAM-413: Fixed to use ModelTable instead of non-existent ModelListTableTemplate
// TEAM-423: Complete rewrite with UniversalFilterBar, stats, and full parity with Next.js
// DATA LAYER: Tauri commands + React Query
// PRESENTATION: UniversalFilterBar + ModelTable + environment-aware actions

// TEAM-476: TODO - Replace with client-side fetcher
import {
  applyHuggingFaceFilters,
  buildHuggingFaceFilterDescription,
  HF_DEFAULTS,
  HF_LICENSES,
  HF_SIZES,
} from '@/lib/marketplace-stubs'
import type { FilterableModel, HuggingFaceSort } from '@/lib/marketplace-stubs'
import {
  HUGGINGFACE_FILTER_GROUPS,
  HUGGINGFACE_SORT_GROUP,
  ModelTable,
  UniversalFilterBar,
} from '@rbee/ui/marketplace'
import { useQuery } from '@tanstack/react-query'
import { invoke } from '@tauri-apps/api/core'
import { useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import type { Model } from '@/generated/bindings'

// TEAM-467: UI filter state (subset of HuggingFaceFilters API type)
interface HuggingFaceUIFilters {
  sort: HuggingFaceSort
  size: typeof HF_SIZES[number]
  license: typeof HF_LICENSES[number]
}


export function MarketplaceHuggingFace() {
  const navigate = useNavigate()

  // TEAM-467: Filter state using API enum values
  const [filters, setFilters] = useState<HuggingFaceUIFilters>({
    sort: HF_DEFAULTS.SORT as HuggingFaceSort,
    size: HF_DEFAULTS.SIZE,
    license: HF_DEFAULTS.LICENSE,
  })

  // DATA LAYER: Fetch models from Tauri
  const {
    data: rawModels = [],
    isLoading,
    error,
  } = useQuery({
    queryKey: ['marketplace', 'huggingface-models'],
    queryFn: async () => {
      const result = await invoke<Model[]>('marketplace_list_models', {
        query: null,
        sort: HF_DEFAULTS.SORT,
        filterTags: null,
        limit: HF_DEFAULTS.LIMIT,
      })
      return result
    },
    staleTime: 5 * 60 * 1000,
  })

  // TEAM-XXX: Use shared filter utilities from marketplace-node
  const filteredModels = useMemo(() => {
    const filtered = applyHuggingFaceFilters(rawModels as FilterableModel[], {
      size: filters.size,
      license: filters.license,
      sort: filters.sort,
    })
    // Map back to Model type for ModelTable compatibility
    return filtered as Model[]
  }, [rawModels, filters])

  const filterDescription = buildHuggingFaceFilterDescription({
    size: filters.size,
    license: filters.license,
    sort: filters.sort,
  })

  // PRESENTATION LAYER: Full layout matching Next.js
  return (
    <div className="container mx-auto px-4 py-12 max-w-7xl">
      {/* Header Section */}
      <div className="mb-8 space-y-4">
        <div className="space-y-2">
          <h1 className="text-4xl md:text-5xl font-bold tracking-tight">HuggingFace LLM Models</h1>
          <p className="text-muted-foreground text-lg md:text-xl max-w-3xl">
            {filterDescription} · Discover and download state-of-the-art language models
          </p>
        </div>

        {/* Stats */}
        <div className="flex items-center gap-6 text-sm text-muted-foreground">
          <div className="flex items-center gap-2">
            <div className="size-2 rounded-full bg-primary" />
            <span>{isLoading ? 'Loading...' : `${filteredModels.length.toLocaleString()} models`}</span>
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
            models={filteredModels}
            onModelClick={(modelId: string) => navigate(`/marketplace/huggingface/${encodeURIComponent(modelId)}`)}
          />
        </div>
      )}
    </div>
  )
}
