// TEAM-464: Client-side filter page with SSG initial data
// Starts with server-rendered models, then allows client-side filtering via manifests
'use client'

import { useState, useEffect } from 'react'
import { ModelTableWithRouting } from '@/components/ModelTableWithRouting'
import { ModelsFilterBar } from '../ModelsFilterBar'
import { loadFilterManifestClient } from '@/lib/manifests-client'
import {
  buildHFFilterDescription,
  HUGGINGFACE_FILTER_GROUPS,
  HUGGINGFACE_SORT_GROUP,
  PREGENERATED_HF_FILTERS,
  type HuggingFaceFilters,
} from './filters'

interface Model {
  id: string
  name: string
  description: string
  author?: string
  downloads: number
  likes: number
  tags: string[]
}

interface Props {
  initialModels: Model[]
  initialFilter: HuggingFaceFilters
}

export function HFFilterPage({ initialModels, initialFilter }: Props) {
  const [models, setModels] = useState<Model[]>(initialModels)
  const [currentFilter, setCurrentFilter] = useState<HuggingFaceFilters>(initialFilter)
  const [loading, setLoading] = useState(false)

  const filterDescription = buildHFFilterDescription(currentFilter)

  // Load manifest when filter changes
  useEffect(() => {
    async function loadManifest() {
      // Find the filter path for current filter config
      const filterConfig = PREGENERATED_HF_FILTERS.find(
        (f) =>
          f.filters.sort === currentFilter.sort &&
          f.filters.size === currentFilter.size &&
          f.filters.license === currentFilter.license
      )

      if (!filterConfig || filterConfig.path === '') {
        // Default filter, use initial models
        setModels(initialModels)
        return
      }

      setLoading(true)
      try {
        const manifest = await loadFilterManifestClient('huggingface', filterConfig.path)
        
        if (manifest) {
          // Convert manifest models to full model objects
          const manifestModels = manifest.models.map((m) => ({
            id: m.id,
            name: m.name,
            description: '',
            author: undefined,
            downloads: 0,
            likes: 0,
            tags: [],
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

  return (
    <div className="container mx-auto px-4 py-12 max-w-7xl">
      {/* Header Section */}
      <div className="mb-8 space-y-4">
        <div className="space-y-2">
          <h1 className="text-4xl md:text-5xl font-bold tracking-tight">HuggingFace Models</h1>
          <p className="text-muted-foreground text-lg md:text-xl max-w-3xl">
            {filterDescription} · Browse language models from HuggingFace
          </p>
        </div>

        {/* Stats */}
        <div className="flex items-center gap-6 text-sm text-muted-foreground">
          <div className="flex items-center gap-2">
            <div className="size-2 rounded-full bg-primary" />
            <span>{models.length.toLocaleString()} models</span>
          </div>
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
        groups={HUGGINGFACE_FILTER_GROUPS}
        sortGroup={HUGGINGFACE_SORT_GROUP}
        currentFilters={currentFilter}
        buildUrlFn="/models/huggingface"
      />

      {/* Model Table */}
      <ModelTableWithRouting models={models} />
    </div>
  )
}
