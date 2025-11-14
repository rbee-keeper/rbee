"use client"
// TEAM-505: Sidebar filters for HuggingFace models page (inspired by HF official site)
// TEAM-502: Upgraded to worker-driven HFFilterSidebar implementation

import { useEffect, useMemo, useState } from 'react'
import type { GWCWorker, MarketplaceModel } from '@rbee/marketplace-core'
import { fetchHuggingFaceModels } from '@rbee/marketplace-core'
import { HFModelListCard } from '@rbee/ui/marketplace'
import { DevelopmentBanner } from '@rbee/ui/molecules'
import { HFFilterSidebar, type HFFilterOptions, type HFFilterState } from '@rbee/ui/marketplace/organisms/HFFilterSidebar'
import { buildHuggingFaceParamsFromFilters } from '@/lib/buildHuggingFaceParams'

interface HuggingFaceSidebarFiltersProps {
  workers: GWCWorker[]
}

function createInitialFilters(): HFFilterState {
  return {
    workers: [],
    tasks: [],
    libraries: [],
    formats: [],
    languages: [],
    licenses: [],
    minParameters: undefined,
    maxParameters: undefined,
    sort: 'downloads',
    direction: -1,
  }
}

function buildFilterOptions(workers: GWCWorker[]): HFFilterOptions {
  const tasks = new Set<string>()
  const libraries = new Set<string>()
  const formats = new Set<string>()
  const languages = new Set<string>()
  const licenses = new Set<string>()

  workers.forEach((worker) => {
    const compat = worker.marketplaceCompatibility.huggingface
    if (!compat) return

    compat.tasks.forEach((t) => tasks.add(t))
    compat.libraries.forEach((l) => libraries.add(l))
    compat.formats.forEach((f) => formats.add(f))
    compat.languages?.forEach((lang) => languages.add(lang))
    compat.licenses?.forEach((lic) => licenses.add(lic))
  })

  return {
    availableWorkers: workers as unknown as HFFilterOptions['availableWorkers'],
    availableTasks: Array.from(tasks),
    availableLibraries: Array.from(libraries),
    availableFormats: Array.from(formats),
    availableLanguages: Array.from(languages),
    availableLicenses: Array.from(licenses),
  }
}

function applyParameterFilter(models: MarketplaceModel[], filters: HFFilterState): MarketplaceModel[] {
  if (filters.minParameters === undefined && filters.maxParameters === undefined) {
    return models
  }

  return models.filter((model) => {
    const match = model.id.match(/(\d+(?:\.\d+)?)b/i)
    if (!match) return true

    const captured = match[1]
    if (!captured) return true

    const paramsB = parseFloat(captured)

    if (filters.minParameters !== undefined && paramsB < filters.minParameters) {
      return false
    }

    if (filters.maxParameters !== undefined && paramsB > filters.maxParameters) {
      return false
    }

    return true
  })
}

export function HuggingFaceSidebarFilters({ workers }: HuggingFaceSidebarFiltersProps) {
  const [filters, setFilters] = useState<HFFilterState>(createInitialFilters)
  const [searchQuery, setSearchQuery] = useState('')
  const [collapsed, setCollapsed] = useState(false)
  const [models, setModels] = useState<MarketplaceModel[]>([])
  const [loading, setLoading] = useState(false)

  const options = useMemo(() => buildFilterOptions(workers), [workers])

  useEffect(() => {
    let cancelled = false

    const run = async () => {
      setLoading(true)
      try {
        const params = buildHuggingFaceParamsFromFilters(filters, searchQuery, workers, 50)
        const response = await fetchHuggingFaceModels(params)
        const items = applyParameterFilter(response.items, filters)
        if (!cancelled) {
          setModels(items)
        }
      } catch (error) {
        console.error('[HuggingFaceSidebarFilters] Error fetching models:', error)
      } finally {
        if (!cancelled) {
          setLoading(false)
        }
      }
    }

    run()

    return () => {
      cancelled = true
    }
  }, [filters, searchQuery, workers])

  return (
    <div className="container mx-auto py-8">
      <DevelopmentBanner
        variant="mvp"
        message="Marketplace: Worker-driven HuggingFace filters"
        details="Filters are driven by worker compatibility and forwarded directly to HuggingFace."
      />

      <div className="flex gap-8 mt-4">
        <aside className="shrink-0">
          <HFFilterSidebar
            filters={filters}
            options={options}
            searchQuery={searchQuery}
            onFiltersChange={setFilters}
            onSearchChange={setSearchQuery}
            collapsed={collapsed}
            onToggleCollapse={() => setCollapsed((prev) => !prev)}
          />
        </aside>

        <main className="flex-1 min-w-0">
          <div className="space-y-6">
            <div className="flex items-center justify-between">
              <p className="text-sm text-muted-foreground">
                {loading ? 'Loading models…' : `${models.length} models`}
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-3">
              {models.map((model) => (
                <HFModelListCard
                  key={model.id}
                  href={`/models/huggingface/${encodeURIComponent(model.id)}`}
                  model={{
                    id: model.id,
                    name: model.name,
                    author: model.author,
                    type: model.type,
                    downloads: model.downloads,
                    likes: model.likes,
                  }}
                />
              ))}
            </div>

            {!loading && models.length === 0 && (
              <div className="text-center text-sm text-muted-foreground py-12">
                No models found. Try adjusting your filters.
              </div>
            )}
          </div>
        </main>
      </div>
    </div>
  )
}
