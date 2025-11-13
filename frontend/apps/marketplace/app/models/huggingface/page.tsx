// TEAM-476: HuggingFace models page - CARD presentation
// TEAM-477: Added MVP compatibility banner
// TEAM-478: Redesigned to card layout (2 columns)
// TEAM-478: Added clickable cards linking to detail pages
// TEAM-481: Refactored to use reusable HFModelListCard component
// TEAM-505: Redesigned with sidebar filters (inspired by HuggingFace official site)

import type { HuggingFaceListModelsParams } from '@rbee/marketplace-core'
import { HFModelListCard } from '@rbee/ui/marketplace'
import { DevelopmentBanner } from '@rbee/ui/molecules'
import { HuggingFaceSidebarFilters } from '../../../components/HuggingFaceSidebarFilters'
import { fetchModels } from '@/lib/fetchModels'

export default async function HuggingFaceModelsPage({
  searchParams,
}: {
  searchParams: Promise<{ search?: string; sort?: string; library?: string }>
}) {
  // Next.js 15: searchParams is now a Promise
  const params = await searchParams
  
  // Build vendor-specific filters from URL params
  const filters: HuggingFaceListModelsParams = {
    ...(params.search && { search: params.search }),
    ...(params.sort && { sort: params.sort as HuggingFaceListModelsParams['sort'] }),
    ...(params.library && { library: params.library as HuggingFaceListModelsParams['library'] }),
    limit: 50,
  }

  // Fetch models on server (SSR)
  const response = await fetchModels('huggingface', filters)

  return (
    <>
      {/* MVP Compatibility Notice */}
      <DevelopmentBanner
        variant="mvp"
        message="🔨 Marketplace MVP: Currently showing text-generation models compatible with llm-worker-rbee."
        details="More workers (Audio, Video, Multi-modal) are actively in development. Model compatibility will expand as new workers are released."
      />

      {/* TEAM-505: Sidebar layout inspired by HuggingFace official site */}
      <div className="container mx-auto py-8">
        <div className="flex gap-8">
          {/* LEFT SIDEBAR: Filters */}
          <aside className="w-64 shrink-0">
            <div className="sticky top-8">
              <h2 className="text-2xl font-bold mb-6">Models</h2>
              <HuggingFaceSidebarFilters
                searchValue={params.search || ''}
                libraryValue={params.library}
                sortValue={params.sort || 'downloads'}
              />
            </div>
          </aside>

          {/* RIGHT CONTENT: Model grid */}
          <main className="flex-1 min-w-0">
            <div className="space-y-6">
              {/* Header with count */}
              <div className="flex items-center justify-between">
                <p className="text-sm text-muted-foreground">
                  {response.meta.total ? (
                    <>
                      <span className="font-semibold text-foreground">{response.meta.total.toLocaleString()}</span> models
                    </>
                  ) : (
                    `${response.items.length} models`
                  )}
                </p>
              </div>

              {/* Model grid (3 columns) */}
              <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-3">
                {response.items.map((model) => (
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

              {/* Pagination info */}
              {response.items.length === 0 && (
                <div className="text-center text-sm text-muted-foreground py-12">
                  No models found. Try adjusting your filters.
                </div>
              )}
            </div>
          </main>
        </div>
      </div>
    </>
  )
}
