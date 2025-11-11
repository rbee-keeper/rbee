// TEAM-405: Marketplace LLM Models page - Using reusable components
// TEAM-413: Fixed to use ModelTable instead of non-existent ModelListTableTemplate
// TEAM-423: Complete rewrite with UniversalFilterBar, stats, and full parity with Next.js
// DATA LAYER: Tauri commands + React Query
// PRESENTATION: UniversalFilterBar + ModelTable + environment-aware actions

// TEAM-467 RULE ZERO: Import shared constants and API types from @rbee/ui package
import {
  DISPLAY_LABELS,
  FILTER_DEFAULTS,
  HF_LICENSES,
  HF_SIZES,
  HUGGINGFACE_FILTER_GROUPS,
  HUGGINGFACE_SORT_GROUP,
  LICENSE_PATTERNS,
  MODEL_SIZE_PATTERNS,
  ModelTable,
  UniversalFilterBar,
} from '@rbee/ui/marketplace'
import type { HuggingFaceSort } from '@rbee/ui/marketplace'
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

// TEAM-467: Build filter description using API enum values
function buildFilterDescription(filters: HuggingFaceUIFilters): string {
  const parts: string[] = []

  if (filters.sort === 'Likes') parts.push(DISPLAY_LABELS.MOST_LIKED)
  else parts.push(DISPLAY_LABELS.MOST_DOWNLOADED)

  if (filters.size !== FILTER_DEFAULTS.HF_SIZE) {
    if (filters.size === 'Small') parts.push(DISPLAY_LABELS.SMALL_MODELS)
    else if (filters.size === 'Medium') parts.push(DISPLAY_LABELS.MEDIUM_MODELS)
    else parts.push(DISPLAY_LABELS.LARGE_MODELS)
  }

  if (filters.license !== FILTER_DEFAULTS.HF_LICENSE) {
    parts.push(
      filters.license === 'Apache'
        ? DISPLAY_LABELS.APACHE_2_0
        : filters.license === 'MIT'
          ? DISPLAY_LABELS.MIT_LICENSE
          : DISPLAY_LABELS.OTHER_LICENSE,
    )
  }

  return parts.length > 0 ? parts.join(' · ') : DISPLAY_LABELS.ALL_MODELS
}

export function MarketplaceHuggingFace() {
  const navigate = useNavigate()

  // TEAM-467: Filter state using API enum values
  const [filters, setFilters] = useState<HuggingFaceUIFilters>({
    sort: FILTER_DEFAULTS.HF_SORT as HuggingFaceSort,
    size: FILTER_DEFAULTS.HF_SIZE,
    license: FILTER_DEFAULTS.HF_LICENSE,
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
        sort: FILTER_DEFAULTS.HF_SORT,
        filterTags: null,
        limit: FILTER_DEFAULTS.HF_LIMIT,
      })
      return result
    },
    staleTime: 5 * 60 * 1000,
  })

  // TEAM-423: Client-side filtering and sorting
  const filteredModels = useMemo(() => {
    let result = [...rawModels]

    // Filter by size (based on model name heuristics)
    if (filters.size !== FILTER_DEFAULTS.HF_SIZE) {
      result = result.filter((model) => {
        const name = model.name.toLowerCase()
        if (filters.size === 'Small') {
          return MODEL_SIZE_PATTERNS.SMALL.some((pattern) => name.includes(pattern))
        } else if (filters.size === 'Medium') {
          return MODEL_SIZE_PATTERNS.MEDIUM.some((pattern) => name.includes(pattern))
        } else {
          // Large
          return MODEL_SIZE_PATTERNS.LARGE.some((pattern) => name.includes(pattern))
        }
      })
    }

    // Filter by license (if available in model data)
    if (filters.license !== FILTER_DEFAULTS.HF_LICENSE && 'license' in rawModels[0]) {
      result = result.filter((model) => {
        const license = (model as any).license?.toLowerCase() || ''
        if (filters.license === 'Apache') return license.includes(LICENSE_PATTERNS.APACHE)
        if (filters.license === 'MIT') return license.includes(LICENSE_PATTERNS.MIT)
        return !license.includes(LICENSE_PATTERNS.APACHE) && !license.includes(LICENSE_PATTERNS.MIT)
      })
    }

    // Sort (using API enum values)
    result.sort((a, b) => {
      if (filters.sort === FILTER_DEFAULTS.HF_SORT) return (b.downloads || 0) - (a.downloads || 0)
      if (filters.sort === 'Likes') return (b.likes || 0) - (a.likes || 0)
      // Recent - would need updatedAt field
      return 0
    })

    return result
  }, [rawModels, filters])

  const filterDescription = buildFilterDescription(filters)

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
