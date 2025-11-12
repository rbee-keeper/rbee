// TEAM-476: HuggingFace models page - CARD presentation
// TEAM-477: Added MVP compatibility banner
// TEAM-478: Redesigned to card layout (2 columns)
// TEAM-478: Added clickable cards linking to detail pages

import type { HuggingFaceListModelsParams } from '@rbee/marketplace-core'
import { DevelopmentBanner } from '@rbee/ui/molecules'
import { HuggingFaceFilterBar } from '../../../components/HuggingFaceFilterBar'
import { ModelPageContainer } from '../../../components/ModelPageContainer'
import Link from 'next/link'

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
            {/* CARD GRID presentation for HuggingFace (2 columns) */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-3">
              {models.map((model) => (
                <Link
                  key={model.id}
                  href={`/models/huggingface/${encodeURIComponent(model.id)}`}
                  className="block border border-border rounded-lg p-4 hover:border-primary/50 transition-colors bg-card cursor-pointer"
                >
                  {/* First row: author/model name */}
                  <div className="flex items-start justify-between mb-3">
                    <div className="flex-1 min-w-0">
                      <h3 className="font-semibold text-base truncate">
                        {model.author}/{model.name.split('/').pop() || model.name}
                      </h3>
                    </div>
                  </div>

                  {/* Second row: task, updated, downloads, likes */}
                  <div className="flex items-center gap-4 text-sm text-muted-foreground flex-wrap">
                    <div className="flex items-center gap-1.5">
                      <span className="text-xs bg-muted px-2 py-0.5 rounded font-mono">{model.type}</span>
                    </div>
                    <div className="flex items-center gap-1">
                      <svg
                        className="w-3.5 h-3.5"
                        fill="none"
                        stroke="currentColor"
                        viewBox="0 0 24 24"
                        aria-hidden="true"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={2}
                          d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
                        />
                      </svg>
                      <span>{model.downloads.toLocaleString()}</span>
                    </div>
                    <div className="flex items-center gap-1">
                      <svg className="w-3.5 h-3.5" fill="currentColor" viewBox="0 0 20 20" aria-hidden="true">
                        <path
                          fillRule="evenodd"
                          d="M3.172 5.172a4 4 0 015.656 0L10 6.343l1.172-1.171a4 4 0 115.656 5.656L10 17.657l-6.828-6.829a4 4 0 010-5.656z"
                          clipRule="evenodd"
                        />
                      </svg>
                      <span>{model.likes.toLocaleString()}</span>
                    </div>
                  </div>
                </Link>
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
