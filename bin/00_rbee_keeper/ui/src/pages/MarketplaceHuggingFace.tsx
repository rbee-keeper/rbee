// TEAM-405: Marketplace LLM Models page - Using reusable components
// TEAM-413: Fixed to use ModelTable instead of non-existent ModelListTableTemplate
// TEAM-423: Complete rewrite with UniversalFilterBar, stats, and full parity with Next.js
// DATA LAYER: Tauri commands + React Query
// PRESENTATION: UniversalFilterBar + ModelTable + environment-aware actions

import type { FilterGroup } from '@rbee/ui/marketplace'
import { ModelTable, UniversalFilterBar } from '@rbee/ui/marketplace'
import { useQuery } from '@tanstack/react-query'
import { invoke } from '@tauri-apps/api/core'
import { useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import type { Model } from '@/generated/bindings'

// TEAM-423: Filter state matching Next.js
interface HuggingFaceFilters {
  sort: 'downloads' | 'likes' | 'recent'
  size: 'all' | 'small' | 'medium' | 'large'
  license: 'all' | 'apache' | 'mit' | 'other'
}

// TEAM-423: Filter groups matching Next.js exactly
const HUGGINGFACE_FILTER_GROUPS: FilterGroup[] = [
  {
    id: 'size',
    label: 'Model Size',
    options: [
      { label: 'All Sizes', value: 'all' },
      { label: 'Small (<7B)', value: 'small' },
      { label: 'Medium (7B-13B)', value: 'medium' },
      { label: 'Large (>13B)', value: 'large' },
    ],
  },
  {
    id: 'license',
    label: 'License',
    options: [
      { label: 'All Licenses', value: 'all' },
      { label: 'Apache 2.0', value: 'apache' },
      { label: 'MIT', value: 'mit' },
      { label: 'Other', value: 'other' },
    ],
  },
]

const HUGGINGFACE_SORT_GROUP: FilterGroup = {
  id: 'sort',
  label: 'Sort By',
  options: [
    { label: 'Most Downloads', value: 'downloads' },
    { label: 'Most Likes', value: 'likes' },
    { label: 'Recently Updated', value: 'recent' },
  ],
}

// TEAM-423: Build filter description matching Next.js
function buildFilterDescription(filters: HuggingFaceFilters): string {
  const parts: string[] = []

  if (filters.sort === 'likes') parts.push('Most Liked')
  else if (filters.sort === 'recent') parts.push('Recently Updated')
  else parts.push('Most Downloaded')

  if (filters.size !== 'all') {
    if (filters.size === 'small') parts.push('Small Models')
    else if (filters.size === 'medium') parts.push('Medium Models')
    else parts.push('Large Models')
  }

  if (filters.license !== 'all') {
    parts.push(filters.license === 'apache' ? 'Apache 2.0' : filters.license === 'mit' ? 'MIT' : 'Other License')
  }

  return parts.length > 0 ? parts.join(' · ') : 'All Models'
}

export function MarketplaceHuggingFace() {
  const navigate = useNavigate()

  // TEAM-423: Filter state
  const [filters, setFilters] = useState<HuggingFaceFilters>({
    sort: 'downloads',
    size: 'all',
    license: 'all',
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
        sort: 'downloads',
        filterTags: null,
        limit: 100,
      })
      return result
    },
    staleTime: 5 * 60 * 1000,
  })

  // TEAM-423: Client-side filtering and sorting
  const filteredModels = useMemo(() => {
    let result = [...rawModels]

    // Filter by size (based on model name heuristics)
    if (filters.size !== 'all') {
      result = result.filter((model) => {
        const name = model.name.toLowerCase()
        if (filters.size === 'small') {
          return name.includes('7b') || name.includes('3b') || name.includes('1b')
        } else if (filters.size === 'medium') {
          return name.includes('13b') || name.includes('8b')
        } else {
          // large
          return name.includes('70b') || name.includes('34b') || name.includes('30b')
        }
      })
    }

    // Filter by license (if available in model data)
    if (filters.license !== 'all' && 'license' in rawModels[0]) {
      result = result.filter((model) => {
        const license = (model as any).license?.toLowerCase() || ''
        if (filters.license === 'apache') return license.includes('apache')
        if (filters.license === 'mit') return license.includes('mit')
        return !license.includes('apache') && !license.includes('mit')
      })
    }

    // Sort
    result.sort((a, b) => {
      if (filters.sort === 'downloads') return (b.downloads || 0) - (a.downloads || 0)
      if (filters.sort === 'likes') return (b.likes || 0) - (a.likes || 0)
      // recent - would need updatedAt field
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
