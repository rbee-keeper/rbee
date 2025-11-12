// TEAM-476: CivitAI models page - IMAGE CARD presentation
// TEAM-477: Added MVP compatibility banner
// TEAM-479: Added clickable cards linking to detail pages

import type { CivitAIListModelsParams } from '@rbee/marketplace-core'
import { DevelopmentBanner } from '@rbee/ui/molecules'
import { ModelCardVertical } from '@rbee/ui/marketplace'
import { ModelPageContainer } from '../../../components/ModelPageContainer'
import { CivitAIFilterBar } from '../../../components/CivitAIFilterBar'

export default async function CivitAIModelsPage({
  searchParams,
}: {
  searchParams: { query?: string; sort?: string; types?: string; baseModels?: string }
}) {
  // Build vendor-specific filters from URL params
  const filters: CivitAIListModelsParams = {
    ...(searchParams.query && { query: searchParams.query }),
    ...(searchParams.sort && { sort: searchParams.sort as CivitAIListModelsParams['sort'] }),
    ...(searchParams.types && { types: searchParams.types.split(',') as CivitAIListModelsParams['types'] }),
    ...(searchParams.baseModels && { baseModels: searchParams.baseModels.split(',') as CivitAIListModelsParams['baseModels'] }),
    limit: 50,
  }

  return (
    <>
      {/* MVP Compatibility Notice */}
      <DevelopmentBanner
        variant="mvp"
        message="🔨 Marketplace MVP: Currently showing Stable Diffusion models compatible with sd-worker-rbee."
        details="More workers (LLM, Audio, Video) are actively in development. Model compatibility will expand as new workers are released."
      />

      <ModelPageContainer
        vendor="civitai"
        title="CivitAI Models"
        subtitle="Browse image generation models from CivitAI"
        filters={filters}
        filterBar={
          <CivitAIFilterBar
            searchValue={searchParams.query || ''}
            typeValue={searchParams.types}
            sortValue={searchParams.sort || 'Most Downloaded'}
          />
        }
      >
      {({ models, pagination }) => (
        <div className="space-y-4">
          {/* IMAGE CARD GRID presentation for CivitAI */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
            {models.map((model) => (
              <ModelCardVertical
                key={model.id}
                href={`/models/civitai/${model.id}`}
                model={{
                  id: model.id,
                  name: model.name,
                  description: model.description || '',
                  ...(model.author ? { author: model.author } : {}),
                  ...(model.imageUrl ? { imageUrl: model.imageUrl } : {}),
                  tags: model.tags.slice(0, 3), // Show top 3 tags
                  downloads: model.downloads,
                  likes: model.likes,
                  size: model.sizeBytes ? `${(model.sizeBytes / (1024 * 1024 * 1024)).toFixed(2)} GB` : model.type,
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
