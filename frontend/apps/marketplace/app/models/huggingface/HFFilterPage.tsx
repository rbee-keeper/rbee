// TEAM-464: Client-side filter page with SSG initial data
// Starts with server-rendered models, then allows client-side filtering via manifests
'use client'

import { HF_LICENSES, HF_SIZES, HF_SORTS } from '@rbee/marketplace-node'
import { usePathname, useRouter, useSearchParams } from 'next/navigation'
import { useCallback, useEffect, useState } from 'react'
import { ModelTableWithRouting } from '@/components/ModelTableWithRouting'
import { loadFilterManifestClient } from '@/lib/manifests-client'
import { ModelsFilterBar } from '../ModelsFilterBar'
import {
  buildHFFilterDescription,
  HUGGINGFACE_FILTER_GROUPS,
  HUGGINGFACE_SORT_GROUP,
  type HuggingFaceFilters,
  PREGENERATED_HF_FILTERS,
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

  // TEAM-467: Validate URL params - FAIL FAST on invalid values
  // Uses SHARED constants from filter-constants.ts
  const validSorts = HF_SORTS
  const validSizes = HF_SIZES
  const validLicenses = HF_LICENSES

  const sortParam = searchParams.get('sort')
  const sizeParam = searchParams.get('size')
  const licenseParam = searchParams.get('license')

  // Build current filter from URL search params
  const currentFilter: HuggingFaceFilters = {
    sort: sortParam && validSorts.includes(sortParam as any) ? (sortParam as any) : initialFilter.sort,
    size: sizeParam && validSizes.includes(sizeParam as any) ? (sizeParam as any) : initialFilter.size,
    license:
      licenseParam && validLicenses.includes(licenseParam as any) ? (licenseParam as any) : initialFilter.license,
  }

  // TEAM-467: Track if we have invalid params
  const hasInvalidParams =
    (sortParam && !validSorts.includes(sortParam as any)) ||
    (sizeParam && !validSizes.includes(sizeParam as any)) ||
    (licenseParam && !validLicenses.includes(licenseParam as any))

  // TEAM-464: Handle filter changes - following Next.js official pattern
  // Using useCallback with searchParams dependency for stable reference
  const handleFilterChange = useCallback(
    (newFilters: Partial<Record<string, string>>) => {
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
    },
    [searchParams, pathname, router, initialFilter],
  )

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
        (f) => f.filters.sort === sort && f.filters.size === size && f.filters.license === license,
      )

      // TEAM-467: FAIL FAST - Don't silently fallback
      if (!filterConfig) {
        console.error('[HFFilterPage] No filter config found for:', { sort, size, license })
        setModels([])
        setLoading(false)
        return
      }

      if (filterConfig.path === '') {
        // Default filter, use initial models
        setModels(defaultModels)
        return
      }

      setLoading(true)
      try {
        const manifest = await loadFilterManifestClient('huggingface', filterConfig.path)

        if (manifest) {
          // TEAM-467: Map ModelMetadata to Model (ensure required fields)
          const mappedModels = manifest.models.map((m) => ({
            id: m.id,
            name: m.name,
            description: m.description || '',
            author: m.author,
            downloads: m.downloads || 0,
            likes: m.likes || 0,
            tags: m.tags || [],
          }))
          setModels(mappedModels)
        } else {
          // TEAM-467: FAIL FAST - Show error instead of fallback
          console.error('[HFFilterPage] Manifest not found for filter:', filterConfig.path)
          setModels([])
        }
      } catch (error) {
        console.error('[HFFilterPage] Failed to load manifest:', error)
        setModels([])
      } finally {
        setLoading(false)
      }
    }

    loadManifest()
    // TEAM-464: Only depend on searchParams to avoid infinite loops
    // initialModels and initialFilter are captured as constants above
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [searchParams, initialFilter.license, initialFilter.size, initialFilter.sort, initialModels])

  return (
    <div className="container mx-auto px-4 py-12 max-w-7xl">
      {/* TEAM-467: Show error for invalid filter params */}
      {hasInvalidParams && (
        <div className="mb-6 p-4 border border-destructive/50 bg-destructive/10 rounded-lg">
          <h3 className="font-semibold text-destructive mb-2">❌ Invalid Filter Parameters</h3>
          <p className="text-sm text-destructive/90">One or more filter parameters are invalid. Valid values:</p>
          <ul className="text-sm text-destructive/90 mt-2 ml-4 list-disc">
            <li>sort: downloads, likes</li>
            <li>size: all, small, medium, large</li>
            <li>license: all, apache, mit, other</li>
          </ul>
          <p className="text-sm text-destructive/90 mt-2">
            Current URL: <code className="bg-destructive/20 px-1 rounded">{window.location.search}</code>
          </p>
        </div>
      )}

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
