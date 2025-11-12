// TEAM-476: HuggingFace models page - CARD presentation
// TEAM-477: Added MVP compatibility banner
// TEAM-478: Redesigned to card layout (2 columns)
// TEAM-478: Added clickable cards linking to detail pages
// TEAM-481: Refactored to use reusable HFModelListCard component

import type { HuggingFaceListModelsParams } from '@rbee/marketplace-core'
import { DevelopmentBanner } from '@rbee/ui/molecules'
import { HFModelListCard } from '@rbee/ui/marketplace'
import { HuggingFaceFilterBar } from '../../../components/HuggingFaceFilterBar'
import { ModelPageContainer } from '../../../components/ModelPageContainer'

export default async function HuggingFaceModelsPage({
  searchParams,
}: {
  searchParams: { search?: string; sort?: string; library?: string }
}) {
  // Build vendor-specific filters from URL params
  const filters: HuggingFaceListModelsParams = {
    ...(searchParams.search && { search: searchParams.search }),
    ...(searchParams.sort && { sort: searchParams.sort as HuggingFaceListModelsParams['sort'] }),
    ...(searchParams.library && { library: searchParams.library as HuggingFaceListModelsParams['library'] }),
    limit: 50,
  }

  return (
    <>
      {/* MVP Compatibility Notice */}
      <DevelopmentBanner
        variant="mvp"
        message="🔨 Marketplace MVP: Currently showing text-generation models compatible with llm-worker-rbee."
        details="More workers (Audio, Video, Multi-modal) are actively in development. Model compatibility will expand as new workers are released."
      />

      <ModelPageContainer
        vendor="huggingface"
        title="HuggingFace Models"
        subtitle="Browse language models from HuggingFace Hub"
        filters={filters}
        filterBar={
          <HuggingFaceFilterBar
            searchValue={searchParams.search || ''}
            libraryValue={searchParams.library}
            sortValue={searchParams.sort || 'downloads'}
          />
        }
      >
        {({ models, pagination }) => (
          <div className="space-y-4">
            {/* CARD GRID presentation for HuggingFace (3 columns) */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-3">
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

            {/* Pagination info */}
            <div className="text-center text-sm text-muted-foreground">
              Showing {models.length} models
              {pagination.total && ` of ${pagination.total.toLocaleString()}`}
            </div>
          </div>
        )}
      </ModelPageContainer>
    </>
  )
}
