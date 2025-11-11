// TEAM-464: Client-side filter page with SSG initial data
// Starts with server-rendered models, then allows client-side filtering via manifests
'use client'

import { useState, useEffect, useCallback } from 'react'
import { useSearchParams, useRouter, usePathname } from 'next/navigation'
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
  const searchParams = useSearchParams()
  const router = useRouter()
  const pathname = usePathname()
  const [models, setModels] = useState<Model[]>(initialModels)
  const [loading, setLoading] = useState(false)
  
  // Build current filter from URL search params
  const currentFilter: HuggingFaceFilters = {
    sort: (searchParams.get('sort') as any) || initialFilter.sort,
    size: (searchParams.get('size') as any) || initialFilter.size,
    license: (searchParams.get('license') as any) || initialFilter.license,
  }

  // TEAM-464: Handle filter changes - following Next.js official pattern
  // Using useCallback with searchParams dependency for stable reference
  const handleFilterChange = useCallback((newFilters: Partial<Record<string, string>>) => {
    // Build new filter state by merging current + new
    const currentSort = searchParams.get('sort') || initialFilter.sort
    const currentSize = searchParams.get('size') || initialFilter.size
    const currentLicense = searchParams.get('license') || initialFilter.license
    
    const merged = {
      sort: (newFilters.sort as any) || currentSort,
      size: (newFilters.size as any) || currentSize,
      license: (newFilters.license as any) || currentLicense,
    }
    
    // Build URL params, excluding defaults ('all', 'downloads')
    const params = new URLSearchParams(searchParams.toString())
    
    // Update or delete each filter param
    if (merged.sort && merged.sort !== 'downloads') {
      params.set('sort', merged.sort)
    } else {
      params.delete('sort')
    }
    
    if (merged.size && merged.size !== 'all') {
      params.set('size', merged.size)
    } else {
      params.delete('size')
    }
    
    if (merged.license && merged.license !== 'all') {
      params.set('license', merged.license)
    } else {
      params.delete('license')
    }
    
    // TEAM-464: Build full URL with pathname (fixes infinite loop)
    // Don't use relative URLs like "?query" - Next.js treats them as server navigations
    const queryString = params.toString()
    const newUrl = queryString ? `${pathname}?${queryString}` : pathname
    
    router.push(newUrl, { scroll: false })
  }, [searchParams, pathname, router, initialFilter])

  const filterDescription = buildHFFilterDescription(currentFilter)

  // Load manifest when URL params change
  useEffect(() => {
    // TEAM-464: Capture initial values to avoid dependency issues
    // biome-ignore lint/correctness/useExhaustiveDependencies: initialFilter and initialModels are stable SSG props
    const defaultSort = initialFilter.sort
    const defaultSize = initialFilter.size
    const defaultLicense = initialFilter.license
    const defaultModels = initialModels
    
    async function loadManifest() {
      // Get current filter values from URL
      const sort = searchParams.get('sort') || defaultSort
      const size = searchParams.get('size') || defaultSize
      const license = searchParams.get('license') || defaultLicense
      
      // Find the filter path for current filter config
      const filterConfig = PREGENERATED_HF_FILTERS.find(
        (f) =>
          f.filters.sort === sort &&
          f.filters.size === size &&
          f.filters.license === license
      )

      if (!filterConfig || filterConfig.path === '') {
        // Default filter, use initial models
        setModels(defaultModels)
        return
      }

      setLoading(true)
      try {
        const manifest = await loadFilterManifestClient('huggingface', filterConfig.path)
        
        if (manifest) {
          // TEAM-464: Try to enrich manifest data with SSG data
          // Manifests only have {id, slug, name}, but SSG has full metadata
          const manifestModels = manifest.models.map((m) => {
            // Find matching model in initial SSG data
            const ssgModel = defaultModels.find(model => model.id === m.id)
            
            if (ssgModel) {
              // Use full SSG data if available
              return ssgModel
            } else {
              // Fallback to minimal manifest data
              return {
                id: m.id,
                name: m.name,
                description: '',
                author: undefined,
                downloads: 0,
                likes: 0,
                tags: [],
              }
            }
          })
          setModels(manifestModels)
        } else {
          // Fallback to initial models if manifest not found
          setModels(defaultModels)
        }
      } catch (error) {
        console.error('Failed to load manifest:', error)
        setModels(defaultModels)
      } finally {
        setLoading(false)
      }
    }

    loadManifest()
    // TEAM-464: Only depend on searchParams to avoid infinite loops
    // initialModels and initialFilter are captured as constants above
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [searchParams])

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
        onChange={handleFilterChange}
      />

      {/* Model Table */}
      <ModelTableWithRouting models={models} />
    </div>
  )
}
