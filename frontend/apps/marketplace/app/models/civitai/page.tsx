// TEAM-476: CivitAI models page - IMAGE CARD presentation

import type { CivitAIListModelsParams } from '@rbee/marketplace-core'
import { FilterBar, FilterMultiSelect, FilterSearch, ModelCardVertical } from '@rbee/ui/marketplace'
import { ModelPageContainer } from '@/components/ModelPageContainer'

export default async function CivitAIModelsPage({
  searchParams,
}: {
  searchParams: { query?: string; sort?: string; types?: string; baseModels?: string }
}) {
  // Build vendor-specific filters from URL params
  const filters: CivitAIListModelsParams = {
    ...(searchParams.query && { query: searchParams.query }),
    ...(searchParams.sort && { sort: searchParams.sort as any }),
    ...(searchParams.types && { types: searchParams.types.split(',') as any }),
    ...(searchParams.baseModels && { baseModels: searchParams.baseModels.split(',') as any }),
    limit: 50,
  }

  return (
    <ModelPageContainer
      vendor="civitai"
      title="CivitAI Models"
      subtitle="Browse image generation models from CivitAI"
      filters={filters}
      filterBar={
        <FilterBar
          filters={
            <>
              <FilterSearch
                label="Search"
                value={searchParams.query || ''}
                onChange={() => {}} // TODO: Client-side filtering
                placeholder="Search models..."
              />
              <FilterMultiSelect
                label="Model Types"
                values={searchParams.types?.split(',') || []}
                onChange={() => {}} // TODO: Client-side filtering
                options={[
                  { value: 'Checkpoint', label: 'Checkpoint' },
                  { value: 'LORA', label: 'LORA' },
                  { value: 'ControlNet', label: 'ControlNet' },
                  { value: 'TextualInversion', label: 'Textual Inversion' },
                ]}
              />
              <FilterMultiSelect
                label="Base Models"
                values={searchParams.baseModels?.split(',') || []}
                onChange={() => {}} // TODO: Client-side filtering
                options={[
                  { value: 'SD 1.5', label: 'SD 1.5' },
                  { value: 'SDXL 1.0', label: 'SDXL 1.0' },
                  { value: 'Flux.1 D', label: 'Flux.1 D' },
                ]}
              />
            </>
          }
          sort={searchParams.sort || 'Most Downloaded'}
          onSortChange={() => {}} // TODO: Client-side sorting
          sortOptions={[
            { value: 'Most Downloaded', label: 'Most Downloaded' },
            { value: 'Highest Rated', label: 'Highest Rated' },
            { value: 'Newest', label: 'Newest' },
          ]}
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
                  author: model.author,
                  imageUrl: model.imageUrl,
                  tags: model.tags.slice(0, 3), // Show top 3 tags
                  downloads: model.downloads,
                  likes: model.likes,
                  size: model.sizeBytes
                    ? `${(model.sizeBytes / (1024 * 1024 * 1024)).toFixed(2)} GB`
                    : model.type,
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
