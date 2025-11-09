// TEAM-460: HuggingFace models marketplace page (migrated from /models)
// TEAM-415: Pure SSG page for maximum SEO
// TEAM-421: Show top 100 popular models (WASM filtering doesn't work in SSG)
// TEAM-461: Added CategoryFilterBar for filtering
import { listHuggingFaceModels } from '@rbee/marketplace-node'
import type { Metadata } from 'next'
import { ModelTableWithRouting } from '@/components/ModelTableWithRouting'
import { ModelsFilterBar } from '../ModelsFilterBar'
import {
  buildHFFilterDescription,
  buildHFFilterUrl,
  HUGGINGFACE_FILTER_GROUPS,
  HUGGINGFACE_SORT_GROUP,
  PREGENERATED_HF_FILTERS,
} from './filters'

export const metadata: Metadata = {
  title: 'HuggingFace LLM Models | rbee Marketplace',
  description: 'Browse compatible language models from HuggingFace. Pre-rendered for instant loading and maximum SEO.',
}

export default async function HuggingFaceModelsPage() {
  // Default filter (downloads, all sizes, all licenses)
  const currentFilter = PREGENERATED_HF_FILTERS[0].filters
  const filterDescription = buildHFFilterDescription(currentFilter)

  // TEAM-421: WASM doesn't work in Next.js SSG - show top 100 popular models
  const FETCH_LIMIT = 100

  console.log(`[SSG] Fetching top ${FETCH_LIMIT} most popular HuggingFace models`)

  const hfModels = await listHuggingFaceModels({ limit: FETCH_LIMIT })

  console.log(`[SSG] Showing ${hfModels.length} HuggingFace models`)

  const models = hfModels.map((model) => {
    const m = model as unknown as Record<string, unknown>
    return {
      id: m.id as string,
      name: (m.name as string) || (m.id as string),
      description: (m.description as string) || `${(m.author as string) || 'Community'} model`,
      author: m.author as string | undefined,
      downloads: (m.downloads as number) ?? 0,
      likes: (m.likes as number) ?? 0,
      tags: (m.tags as string[] | undefined)?.slice(0, 10) ?? [],
    }
  })

  return (
    <div className="container mx-auto px-4 py-12 max-w-7xl">
      {/* Header Section */}
      <div className="mb-8 space-y-4">
        <div className="space-y-2">
          <h1 className="text-4xl md:text-5xl font-bold tracking-tight">HuggingFace LLM Models</h1>
          <p className="text-muted-foreground text-lg md:text-xl max-w-3xl">
            {filterDescription} · Discover and download state-of-the-art language models
          </p>
        </div>

        {/* Stats */}
        <div className="flex items-center gap-6 text-sm text-muted-foreground">
          <div className="flex items-center gap-2">
            <div className="size-2 rounded-full bg-primary" />
            <span>{models.length.toLocaleString()} LLM models</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="size-2 rounded-full bg-orange-500" />
            <span>HuggingFace Hub</span>
          </div>
        </div>
      </div>

      {/* Filter Bar */}
      <ModelsFilterBar
        groups={HUGGINGFACE_FILTER_GROUPS}
        sortGroup={HUGGINGFACE_SORT_GROUP}
        currentFilters={currentFilter}
        buildUrlFn="/models/huggingface"
      />

      {/* Table with client-side routing (minimal JS for navigation only) */}
      <div className="rounded-lg border border-border bg-card p-6">
        <ModelTableWithRouting models={models} />
      </div>
    </div>
  )
}
