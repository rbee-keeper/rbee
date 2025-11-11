// TEAM-476: HuggingFace models page - TABLE presentation

import type { HuggingFaceListModelsParams } from '@rbee/marketplace-core'
import { FilterBar, FilterDropdown, FilterSearch } from '@rbee/ui/marketplace'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@rbee/ui/atoms'
import { ModelPageContainer } from '@/components/ModelPageContainer'

export default async function HuggingFaceModelsPage({
  searchParams,
}: {
  searchParams: { search?: string; sort?: string; library?: string }
}) {
  // Build vendor-specific filters from URL params
  const filters: HuggingFaceListModelsParams = {
    ...(searchParams.search && { search: searchParams.search }),
    ...(searchParams.sort && { sort: searchParams.sort as any }),
    ...(searchParams.library && { library: searchParams.library as any }),
    limit: 50,
  }

  return (
    <ModelPageContainer
      vendor="huggingface"
      title="HuggingFace Models"
      subtitle="Browse language models from HuggingFace Hub"
      filters={filters}
      filterBar={
        <FilterBar
          filters={
            <>
              <FilterSearch
                label="Search"
                value={searchParams.search || ''}
                onChange={() => {}} // TODO: Client-side filtering
                placeholder="Search models..."
              />
              <FilterDropdown
                label="Library"
                value={searchParams.library}
                onChange={() => {}} // TODO: Client-side filtering
                options={[
                  { value: 'transformers', label: 'Transformers' },
                  { value: 'diffusers', label: 'Diffusers' },
                  { value: 'pytorch', label: 'PyTorch' },
                ]}
              />
            </>
          }
          sort={searchParams.sort || 'downloads'}
          onSortChange={() => {}} // TODO: Client-side sorting
          sortOptions={[
            { value: 'downloads', label: 'Most Downloaded' },
            { value: 'likes', label: 'Most Liked' },
            { value: 'trending', label: 'Trending' },
            { value: 'updated', label: 'Recently Updated' },
          ]}
        />
      }
    >
      {({ models, pagination }) => (
        <div className="space-y-4">
          {/* TABLE presentation for HuggingFace */}
          <div className="border rounded-lg overflow-hidden">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead className="w-[40%]">Model</TableHead>
                  <TableHead>Author</TableHead>
                  <TableHead className="text-right">Downloads</TableHead>
                  <TableHead className="text-right">Likes</TableHead>
                  <TableHead>Type</TableHead>
                  <TableHead className="text-right">Size</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {models.map((model) => (
                  <TableRow key={model.id}>
                    <TableCell>
                      <div>
                        <div className="font-medium">{model.name}</div>
                        {model.description && (
                          <div className="text-sm text-muted-foreground line-clamp-1">{model.description}</div>
                        )}
                      </div>
                    </TableCell>
                    <TableCell>{model.author}</TableCell>
                    <TableCell className="text-right">{model.downloads.toLocaleString()}</TableCell>
                    <TableCell className="text-right">{model.likes.toLocaleString()}</TableCell>
                    <TableCell>
                      <code className="text-xs bg-muted px-2 py-1 rounded">{model.type}</code>
                    </TableCell>
                    <TableCell className="text-right text-sm text-muted-foreground">
                      {model.sizeBytes ? `${(model.sizeBytes / (1024 * 1024 * 1024)).toFixed(2)} GB` : '—'}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
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
