// TEAM-464: Client-side filter page with SSG initial data
// Starts with server-rendered models, then allows client-side filtering via manifests
'use client'

import { useState, useEffect } from 'react'
import Link from 'next/link'
import { ModelCardVertical } from '@rbee/ui/marketplace'
import { ModelsFilterBar } from '../ModelsFilterBar'
import { modelIdToSlug } from '@/lib/slugify'
import { loadFilterManifestClient } from '@/lib/manifests-client'
import {
  buildFilterParams,
  CIVITAI_FILTER_GROUPS,
  CIVITAI_SORT_GROUP,
  getFilterFromPath,
  PREGENERATED_FILTERS,
  type CivitaiFilters,
} from './filters'

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
  const [models, setModels] = useState<Model[]>(initialModels)
  const [currentFilter, setCurrentFilter] = useState<CivitaiFilters>(initialFilter)
  const [loading, setLoading] = useState(false)

  // Load manifest when filter changes
  useEffect(() => {
    async function loadManifest() {
      // Find the filter path for current filter config
      const filterConfig = PREGENERATED_FILTERS.find(
        (f) =>
          f.filters.timePeriod === currentFilter.timePeriod &&
          f.filters.modelType === currentFilter.modelType &&
          f.filters.baseModel === currentFilter.baseModel &&
          f.filters.nsfwLevel === currentFilter.nsfwLevel
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
          // In a real implementation, you might want to fetch full details
          // For now, we'll use the manifest data as-is
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
  }, [currentFilter, initialModels])

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
        buildUrlFn="/models/civitai"
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
