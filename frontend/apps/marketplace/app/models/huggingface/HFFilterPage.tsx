// TEAM-464: Client-side filter page with SSR initial data
// TEAM-475: Uses SSR data only, no client-side manifest loading
'use client'

import { HF_LICENSES, HF_SIZES, HF_SORTS } from '@rbee/marketplace-node'
import { usePathname, useRouter, useSearchParams } from 'next/navigation'
import { useCallback, useMemo, useState } from 'react'
import { ModelTableWithRouting } from '@/components/ModelTableWithRouting'
import { ModelsFilterBar } from '../ModelsFilterBar'
import {
  buildHFFilterDescription,
  HUGGINGFACE_FILTER_GROUPS,
  HUGGINGFACE_SORT_GROUP,
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
}

export function HFFilterPage({ initialModels }: Props) {
  const searchParams = useSearchParams()
  const router = useRouter()
  const pathname = usePathname()
  // TEAM-475: Client-side filtering - filter and sort the SSR data
  const [loading] = useState(false)

  // TEAM-475: Build current filter from URL search params (with defaults)
  // Properly validate URL params against allowed values
  const sortParam = searchParams.get('sort')
  const sizeParam = searchParams.get('size')
  const licenseParam = searchParams.get('license')

  const currentFilter: HuggingFaceFilters = {
    sort: sortParam && (HF_SORTS as readonly string[]).includes(sortParam) ? sortParam as typeof HF_SORTS[number] : HF_SORTS[0],
    size: sizeParam && (HF_SIZES as readonly string[]).includes(sizeParam) ? sizeParam as typeof HF_SIZES[number] : HF_SIZES[0],
    license: licenseParam && (HF_LICENSES as readonly string[]).includes(licenseParam) ? licenseParam as typeof HF_LICENSES[number] : HF_LICENSES[0],
  }

  // Handle filter changes - update URL without page reload (SPA experience)
  const handleFilterChange = useCallback(
    (newFilters: Partial<Record<string, string>>) => {
      const params = new URLSearchParams(searchParams.toString())

      // Map filter keys to URL param names
      if (newFilters.sort) params.set('sort', newFilters.sort)
      if (newFilters.size) params.set('size', newFilters.size)
      if (newFilters.license) params.set('license', newFilters.license)

      const newUrl = params.toString() ? `${pathname}?${params.toString()}` : pathname

      // TEAM-475: Update URL without reload - client-side filtering handles the rest
      router.replace(newUrl, { scroll: false })
    },
    [searchParams, pathname, router],
  )

  const filterDescription = buildHFFilterDescription(currentFilter)

  // TEAM-475: Client-side filtering - filter and sort models based on URL params
  const filteredModels = useMemo(() => {
    let result = [...initialModels]

    // Filter by size (based on model name/tags)
    if (currentFilter.size !== HF_SIZES[0]) {
      result = result.filter((model) => {
        const modelText = `${model.name} ${model.tags.join(' ')}`.toLowerCase()
        switch (currentFilter.size) {
          case HF_SIZES[1]: // Small
            return modelText.includes('small') || modelText.includes('mini') || modelText.includes('tiny')
          case HF_SIZES[2]: // Medium
            return modelText.includes('medium') || modelText.includes('base')
          case HF_SIZES[3]: // Large
            return modelText.includes('large') || modelText.includes('xl') || modelText.includes('giant')
          default:
            return true
        }
      })
    }

    // Filter by license (based on tags)
    if (currentFilter.license !== HF_LICENSES[0]) {
      result = result.filter((model) => {
        const tags = model.tags.map(t => t.toLowerCase())
        switch (currentFilter.license) {
          case HF_LICENSES[1]: // Apache
            return tags.some(t => t.includes('apache'))
          case HF_LICENSES[2]: // MIT
            return tags.some(t => t.includes('mit'))
          case HF_LICENSES[3]: // Other
            return !tags.some(t => t.includes('apache') || t.includes('mit'))
          default:
            return true
        }
      })
    }

    // Sort models
    result.sort((a, b) => {
      switch (currentFilter.sort) {
        case HF_SORTS[1]: // Likes
          return (b.likes || 0) - (a.likes || 0)
        case HF_SORTS[0]: // Downloads
        default:
          return (b.downloads || 0) - (a.downloads || 0)
      }
    })

    return result
  }, [initialModels, currentFilter.sort, currentFilter.size, currentFilter.license])

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
            <span>{filteredModels.length.toLocaleString()} models</span>
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
      <ModelTableWithRouting models={filteredModels} />
    </div>
  )
}
