// TEAM-476: CivitAI models page - IMAGE CARD presentation

import type { CivitAIListModelsParams } from '@rbee/marketplace-core'
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
  )
}
