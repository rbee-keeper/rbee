// TEAM-482: Workers list page - CARD presentation using GWC adapter
// Uses ModelPageContainer pattern like HuggingFace and CivitAI pages

import type { GWCListWorkersParams } from '@rbee/marketplace-core'
import { WorkerListCard } from '@rbee/ui/marketplace'
import { DevelopmentBanner } from '@rbee/ui/molecules'
import { ModelPageContainer } from '../../components/ModelPageContainer'

export default async function WorkersPage({
  searchParams,
}: {
  searchParams: Promise<{ backend?: string; platform?: string }>
}) {
  // Next.js 15: searchParams is now a Promise
  const params = await searchParams

  // Build GWC-specific filters from URL params
  const filters: GWCListWorkersParams = {
    limit: 50,
  }

  if (params.backend) {
    const backend = params.backend as 'cpu' | 'cuda' | 'metal' | 'rocm'
    filters.backend = backend
  }
  if (params.platform) {
    const platform = params.platform as 'linux' | 'macos' | 'windows'
    filters.platform = platform
  }

  return (
    <>
      {/* MVP Notice */}
      <DevelopmentBanner
        variant="mvp"
        message="🔨 Marketplace MVP: Worker catalog is under active development."
        details="Currently showing available workers from Global Worker Catalog. Installation and management features coming soon."
      />

      <ModelPageContainer
        vendor="gwc"
        title="AI Workers"
        subtitle="Browse and install AI workers for your rbee cluster"
        filters={filters}
      >
        {({ models, pagination }) => {
          // Group workers by ID to show one card per worker (not per variant)
          // Each worker has multiple backends in tags, show the primary one
          const uniqueWorkers = models.filter((model, index, self) =>
            index === self.findIndex((m) => m.id === model.id)
          )

          return (
            <div className="space-y-4">
              {/* CARD GRID presentation for Workers (3 columns) */}
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {uniqueWorkers.map((model) => {
                  // Extract primary backend from tags (tags[1] is usually the first backend)
                  const workerType = (model.tags[1] as 'cpu' | 'cuda' | 'metal' | 'rocm') || 'cpu'

                  return (
                    <WorkerListCard
                      key={model.id}
                      href={`/workers/${encodeURIComponent(model.id)}`}
                      worker={{
                        id: model.id,
                        name: model.name,
                        description: model.description || '',
                        version: (model.metadata?.version as string) || '0.1.0',
                        workerType,
                        ...(model.imageUrl ? { imageUrl: model.imageUrl } : {}),
                      }}
                    />
                  )
                })}
              </div>

              {/* Pagination info */}
              <div className="text-center text-sm text-muted-foreground">
                Showing {uniqueWorkers.length} workers
                {pagination.total && ` of ${pagination.total.toLocaleString()}`}
              </div>
            </div>
          )
        }}
      </ModelPageContainer>
    </>
  )
}
