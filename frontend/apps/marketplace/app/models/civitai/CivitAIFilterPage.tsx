// TEAM-464: Client-side filter page with SSR initial data
// TEAM-475: Uses SSR data only, no client-side manifest loading
// TEAM-475: Added client-side filtering and missing imports
'use client'

import {
  type BaseModel,
  CIVITAI_BASE_MODELS,
  CIVITAI_MODEL_TYPES,
  CIVITAI_NSFW_LEVELS,
  CIVITAI_SORTS,
  CIVITAI_TIME_PERIODS,
  type CivitaiModelType,
  type CivitaiSort,
  type NsfwLevel,
  type TimePeriod,
} from '@rbee/marketplace-node'
import { ModelCardVertical } from '@rbee/ui/marketplace/organisms/ModelCardVertical'
import Link from 'next/link'
import { usePathname, useRouter, useSearchParams } from 'next/navigation'
import { useCallback, useMemo, useState } from 'react'
import { ModelsFilterBar } from '../ModelsFilterBar'
import { CIVITAI_FILTER_GROUPS, CIVITAI_SORT_GROUP, type CivitaiFilters } from './filters'

// Helper to convert model ID to slug
function modelIdToSlug(id: string): string {
  return id.replace('civitai-', '')
}

// TEAM-476: Added 'type' and 'nsfw' fields for proper filtering
interface Model {
  id: string
  name: string
  description: string
  author: string
  downloads: number
  likes: number
  size: string
  tags: string[]
  imageUrl?: string
  type: string // Model type from CivitAI (Checkpoint, LORA, etc.)
  nsfw: boolean // NSFW flag from CivitAI for content rating filtering
}

interface Props {
  initialModels: Model[]
}

export function CivitAIFilterPage({ initialModels }: Props) {
  const searchParams = useSearchParams()
  const router = useRouter()
  const pathname = usePathname()
  // TEAM-475: Client-side filtering - filter and sort the SSR data
  const [loading] = useState(false)

  // TEAM-476: FAIL FAST - Validate URL params and log warnings for invalid values
  const periodParam = searchParams.get('period')
  const typeParam = searchParams.get('type')
  const baseParam = searchParams.get('base')
  const sortParam = searchParams.get('sort')
  const nsfwParam = searchParams.get('nsfw')

  // TEAM-476: FAIL FAST - Log invalid filter values
  if (periodParam && !(CIVITAI_TIME_PERIODS as readonly string[]).includes(periodParam)) {
    console.error(`❌ INVALID FILTER: period="${periodParam}" not in`, CIVITAI_TIME_PERIODS)
  }
  if (typeParam && !(CIVITAI_MODEL_TYPES as readonly string[]).includes(typeParam)) {
    console.error(`❌ INVALID FILTER: type="${typeParam}" not in`, CIVITAI_MODEL_TYPES)
  }
  if (baseParam && !(CIVITAI_BASE_MODELS as readonly string[]).includes(baseParam)) {
    console.error(`❌ INVALID FILTER: base="${baseParam}" not in`, CIVITAI_BASE_MODELS)
  }
  if (sortParam && !(CIVITAI_SORTS as readonly string[]).includes(sortParam)) {
    console.error(`❌ INVALID FILTER: sort="${sortParam}" not in`, CIVITAI_SORTS)
  }
  if (nsfwParam && !(CIVITAI_NSFW_LEVELS as readonly string[]).includes(nsfwParam)) {
    console.error(`❌ INVALID FILTER: nsfw="${nsfwParam}" not in`, CIVITAI_NSFW_LEVELS)
  }

  const currentFilter: CivitaiFilters = {
    timePeriod:
      periodParam && (CIVITAI_TIME_PERIODS as readonly string[]).includes(periodParam)
        ? (periodParam as TimePeriod)
        : CIVITAI_TIME_PERIODS[2],
    modelType:
      typeParam && (CIVITAI_MODEL_TYPES as readonly string[]).includes(typeParam)
        ? (typeParam as CivitaiModelType)
        : CIVITAI_MODEL_TYPES[0],
    baseModel:
      baseParam && (CIVITAI_BASE_MODELS as readonly string[]).includes(baseParam)
        ? (baseParam as BaseModel)
        : CIVITAI_BASE_MODELS[0],
    sort:
      sortParam && (CIVITAI_SORTS as readonly string[]).includes(sortParam)
        ? (sortParam as CivitaiSort)
        : CIVITAI_SORTS[0],
    nsfwLevel:
      nsfwParam && (CIVITAI_NSFW_LEVELS as readonly string[]).includes(nsfwParam)
        ? (nsfwParam as NsfwLevel)
        : CIVITAI_NSFW_LEVELS[4], // TEAM-476: Default to 'XXX' (all levels) to match API default
  }

  // TEAM-476: FAIL FAST - Log the actual filter being applied
  console.log('[Filter] URL params:', { periodParam, typeParam, baseParam, sortParam, nsfwParam })
  console.log('[Filter] Applied filter:', currentFilter)
  console.log('[Filter] Initial models count:', initialModels.length)

  // TEAM-475: Client-side filtering - filter and sort models based on URL params
  const filteredModels = useMemo(() => {
    let result = [...initialModels]
    console.log('[Filter] Starting with', result.length, 'models')

    // TEAM-476: Filter by actual model type (not tags!)
    // BUG FIX: Was searching tags, but model type is stored in the 'type' field
    if (currentFilter.modelType !== 'All') {
      const beforeCount = result.length
      result = result.filter((model) => {
        return model.type === currentFilter.modelType
      })
      console.log(`[Filter] Model type filter: "${currentFilter.modelType}" → ${beforeCount} to ${result.length} models`)
      if (result.length === 0) {
        console.error(`❌ NO MODELS after type filter! Looking for type="${currentFilter.modelType}"`)
        console.error('Available types:', [...new Set(initialModels.map(m => m.type))])
      }
    }

    // TEAM-476: Filter by NSFW level using the actual nsfw field
    // BUG FIX: Was only filtering when nsfwLevel === 'None', now handles all levels
    if (currentFilter.nsfwLevel === 'None') {
      const beforeCount = result.length
      // PG (Safe for work) - exclude NSFW models
      result = result.filter((model) => !model.nsfw)
      console.log(`[Filter] NSFW filter: "None" (PG only) → ${beforeCount} to ${result.length} models`)
      const nsfwCount = initialModels.filter(m => m.nsfw).length
      const safeCount = initialModels.filter(m => !m.nsfw).length
      console.log(`[Filter] NSFW breakdown: ${nsfwCount} NSFW, ${safeCount} safe`)
    } else {
      console.log(`[Filter] NSFW filter: "${currentFilter.nsfwLevel}" (showing all)`)
    }
    // For other NSFW levels (Soft, Mature, X, XXX), we show all models
    // The API defaults to XXX (all levels) so all models are available for filtering

    // Sort models
    result.sort((a, b) => {
      switch (currentFilter.sort) {
        case 'Highest Rated':
          return (b.likes || 0) - (a.likes || 0)
        case 'Newest':
          return 0 // Keep original order (already sorted by newest from API)
        default:
          return (b.downloads || 0) - (a.downloads || 0)
      }
    })
    console.log(`[Filter] After sorting by "${currentFilter.sort}": ${result.length} models`)

    // TEAM-476: Show actual model names to verify they're different
    const sampleModels = [0, 1, 2, 4, 7].map(i => result[i]).filter(Boolean)
    console.log('[Filter] FINAL RESULT:', result.length, 'models')
    console.log('[Filter] Sample models (1st, 2nd, 3rd, 5th, 8th):', sampleModels.map(m => ({
      name: m.name,
      type: m.type,
      nsfw: m.nsfw,
      downloads: m.downloads,
      likes: m.likes
    })))
    return result
  }, [initialModels, currentFilter.modelType, currentFilter.nsfwLevel, currentFilter.sort])

  // Handle filter changes - update URL without page reload (SPA experience)
  const handleFilterChange = useCallback(
    (newFilters: Partial<Record<string, string>>) => {
      const params = new URLSearchParams(searchParams.toString())

      // Map filter keys to URL param names
      if (newFilters.timePeriod) params.set('period', newFilters.timePeriod)
      if (newFilters.modelType) params.set('type', newFilters.modelType)
      if (newFilters.baseModel) params.set('base', newFilters.baseModel)
      if (newFilters.sort) params.set('sort', newFilters.sort)
      if (newFilters.nsfwLevel) params.set('nsfw', newFilters.nsfwLevel)

      const newUrl = params.toString() ? `${pathname}?${params.toString()}` : pathname

      // TEAM-475: Update URL without reload - client-side filtering handles the rest
      router.replace(newUrl, { scroll: false })
    },
    [searchParams, pathname, router],
  )

  // Build filter description
  const filterParts: string[] = []
  if (currentFilter.nsfwLevel) {
    const ratingLabels = {
      None: 'PG (Safe for work)',
      Soft: 'PG-13 (Suggestive)',
      Mature: 'R (Mature)',
      X: 'X (Explicit)',
      XXX: 'XXX (Pornographic)',
    }
    filterParts.push(ratingLabels[currentFilter.nsfwLevel] || currentFilter.nsfwLevel)
  }
  if (currentFilter.timePeriod !== 'AllTime') filterParts.push(currentFilter.timePeriod)
  if (currentFilter.modelType !== 'All') filterParts.push(currentFilter.modelType)
  if (currentFilter.baseModel !== 'All') filterParts.push(currentFilter.baseModel)

  const filterDescription = filterParts.length > 0 ? filterParts.join(' · ') : 'All Models'

  return (
    <div className="container mx-auto px-4 py-12 max-w-7xl">
      {/* Header Section */}
      <div className="mb-8 space-y-4">
        <div className="space-y-2">
          <h1 className="text-4xl md:text-5xl font-bold tracking-tight">CivitAI Models</h1>
          <p className="text-muted-foreground text-lg md:text-xl max-w-3xl">
            {filterDescription} · Discover and download Stable Diffusion models
          </p>
        </div>

        {/* Stats */}
        <div className="flex items-center gap-6 text-sm text-muted-foreground">
          <div className="flex items-center gap-2">
            <div className="size-2 rounded-full bg-primary" />
            <span>{filteredModels.length.toLocaleString()} models</span>
          </div>
          {currentFilter.nsfwLevel && (
            <div className="flex items-center gap-2">
              <div
                className={`size-2 rounded-full ${
                  currentFilter.nsfwLevel === 'None'
                    ? 'bg-green-500'
                    : currentFilter.nsfwLevel === 'Soft'
                      ? 'bg-yellow-500'
                      : currentFilter.nsfwLevel === 'Mature'
                        ? 'bg-orange-500'
                        : 'bg-red-500'
                }`}
              />
              <span>
                {currentFilter.nsfwLevel === 'None'
                  ? 'Safe for work'
                  : currentFilter.nsfwLevel === 'Soft'
                    ? 'Suggestive content'
                    : currentFilter.nsfwLevel === 'Mature'
                      ? 'Mature content'
                      : 'Explicit content'}
              </span>
            </div>
          )}
          {loading && (
            <div className="flex items-center gap-2">
              <div className="size-2 rounded-full bg-blue-500 animate-pulse" />
              <span>Loading...</span>
            </div>
          )}
        </div>
      </div>

      {/* Filter Bar */}
      <ModelsFilterBar
        groups={CIVITAI_FILTER_GROUPS}
        sortGroup={CIVITAI_SORT_GROUP}
        currentFilters={currentFilter}
        onChange={handleFilterChange}
      />

      {/* Model Grid */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-4">
        {filteredModels.map((model) => (
          <Link key={model.id} href={`/models/civitai/${modelIdToSlug(model.id)}`} className="block">
            <ModelCardVertical model={model} />
          </Link>
        ))}
      </div>
    </div>
  )
}
