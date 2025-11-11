// TEAM-464: Client-side filter page with SSG initial data
// Starts with server-rendered models, then allows client-side filtering via manifests
'use client'

import { ModelCardVertical } from '@rbee/ui/marketplace'
import Link from 'next/link'
import { useRouter, useSearchParams } from 'next/navigation'
import { useEffect, useState } from 'react'
import { loadFilterManifestClient } from '@/lib/manifests-client'
import { modelIdToSlug } from '@/lib/slugify'
import { ModelsFilterBar } from '../ModelsFilterBar'
import { CIVITAI_FILTER_GROUPS, CIVITAI_SORT_GROUP, type CivitaiFilters, PREGENERATED_FILTERS } from './filters'

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
}

interface Props {
  initialModels: Model[]
  initialFilter: CivitaiFilters
}

export function CivitAIFilterPage({ initialModels, initialFilter }: Props) {
  const searchParams = useSearchParams()
  const router = useRouter()
  const [models, setModels] = useState<Model[]>(initialModels)
  const [loading, setLoading] = useState(false)

  // Build current filter from URL search params
  const currentFilter: CivitaiFilters = {
    timePeriod: (searchParams.get('period') as any) || initialFilter.timePeriod,
    modelType: (searchParams.get('type') as any) || initialFilter.modelType,
    baseModel: (searchParams.get('base') as any) || initialFilter.baseModel,
    sort: (searchParams.get('sort') as any) || initialFilter.sort,
    nsfwLevel: (searchParams.get('nsfw') as any) || initialFilter.nsfwLevel,
  }

  // Handle filter changes - update URL instead of state
  const handleFilterChange = (newFilters: Partial<Record<string, string>>) => {
    const params = new URLSearchParams(searchParams.toString())

    // Map filter keys to URL param names
    if (newFilters.timePeriod) params.set('period', newFilters.timePeriod)
    if (newFilters.modelType) params.set('type', newFilters.modelType)
    if (newFilters.baseModel) params.set('base', newFilters.baseModel)
    if (newFilters.sort) params.set('sort', newFilters.sort)
    if (newFilters.nsfwLevel) params.set('nsfw', newFilters.nsfwLevel)

    // Update URL (this will trigger useEffect)
    router.push(`?${params.toString()}`)
  }

  // Load manifest when URL params change
  useEffect(() => {
    async function loadManifest() {
      // Get current filter values from URL
      const period = searchParams.get('period') || initialFilter.timePeriod
      const type = searchParams.get('type') || initialFilter.modelType
      const base = searchParams.get('base') || initialFilter.baseModel
      const nsfw = searchParams.get('nsfw') || initialFilter.nsfwLevel

      // Find the filter path for current filter config
      const filterConfig = PREGENERATED_FILTERS.find(
        (f) =>
          f.filters.timePeriod === period &&
          f.filters.modelType === type &&
          f.filters.baseModel === base &&
          f.filters.nsfwLevel === nsfw,
      )

      if (!filterConfig || filterConfig.path === '') {
        // Default filter, use initial models
        setModels(initialModels)
        return
      }

      setLoading(true)
      try {
        const manifest = await loadFilterManifestClient('civitai', filterConfig.path)

        if (manifest) {
          // Convert manifest models to full model objects
          const manifestModels = manifest.models.map((m) => ({
            id: m.id,
            name: m.name,
            description: '',
            author: '',
            downloads: 0,
            likes: 0,
            size: '',
            tags: [],
            imageUrl: undefined,
          }))
          setModels(manifestModels)
        } else {
          // Fallback to initial models if manifest not found
          setModels(initialModels)
        }
      } catch (error) {
        console.error('Failed to load manifest:', error)
        setModels(initialModels)
      } finally {
        setLoading(false)
      }
    }

    loadManifest()
  }, [searchParams, initialModels, initialFilter])

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
            <span>{models.length.toLocaleString()} models</span>
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

      {/* Vertical Card Grid */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-4">
        {models.map((model) => (
          <Link key={model.id} href={`/models/civitai/${modelIdToSlug(model.id)}`} className="block">
            <ModelCardVertical model={model} />
          </Link>
        ))}
      </div>
    </div>
  )
}
