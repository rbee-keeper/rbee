// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// ⛔ CRITICAL: DO NOT ADD 'export const dynamic = "force-dynamic"' TO THIS FILE
// ⛔ force-dynamic CAUSES CLOUDFLARE WORKER CPU LIMIT ERRORS (Error 1102)
// ⛔ This page MUST be statically generated at build time
// ⛔ If build fails, fix the API or reduce filters - NEVER use force-dynamic
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// TEAM-461: Dynamic filtered HuggingFace pages (SSG pre-generated)
// TEAM-462: PERMANENT FIX - Static generation only, force-dynamic FORBIDDEN
import { listHuggingFaceModels } from '@rbee/marketplace-node'
import type { ModelTableItem } from '@rbee/ui/marketplace'
import { ModelsFilterBar } from '../../ModelsFilterBar'
import type { Metadata } from 'next'
import { ModelTableWithRouting } from '@/components/ModelTableWithRouting'
import {
  buildHFFilterDescription,
  buildHFFilterParams,
  buildHFFilterUrl,
  getHFFilterFromPath,
  HUGGINGFACE_FILTER_GROUPS,
  HUGGINGFACE_SORT_GROUP,
  PREGENERATED_HF_FILTERS,
} from '../filters'

interface PageProps {
  params: Promise<{
    filter: string[]
  }>
}

// Pre-generate static pages for all filter combinations
export async function generateStaticParams() {
  return PREGENERATED_HF_FILTERS.filter((f) => f.path !== '') // Exclude default (handled by main page)
    .map((f) => ({
      // Split path and remove empty strings and 'filter' prefix
      filter: f.path.split('/').filter(Boolean).filter(p => p !== 'filter'),
    }))
}

export async function generateMetadata({ params }: PageProps): Promise<Metadata> {
  const { filter } = await params
  const filterPath = filter.join('/')
  const currentFilter = getHFFilterFromPath(filterPath)
  const description = buildHFFilterDescription(currentFilter)

  return {
    title: `${description} | HuggingFace Models | rbee Marketplace`,
    description: `Browse ${description.toLowerCase()} language models from HuggingFace.`,
  }
}

export default async function FilteredHuggingFacePage({ params }: PageProps) {
  const { filter } = await params
  const filterPath = filter.join('/')
  const currentFilter = getHFFilterFromPath(filterPath)
  const filterDescription = buildHFFilterDescription(currentFilter)

  console.log(`[SSG] Fetching HuggingFace models with filter: ${filterPath}`)

  // TEAM-462: HuggingFace API only supports `limit` - fetch ALL models
  const apiParams = buildHFFilterParams(currentFilter)
  const hfModels = await listHuggingFaceModels(apiParams)

  // TEAM-462: Apply CLIENT-SIDE filtering (API doesn't support it)
  let filteredModels = hfModels

  // Filter by size (if specified)
  if (currentFilter.size !== 'all') {
    filteredModels = filteredModels.filter((model) => {
      // Size filtering logic based on model size
      // This is placeholder - adjust based on actual model size data
      const sizeCategory = currentFilter.size
      // TODO: Implement actual size filtering when model size data available
      return true // For now, include all
    })
  }

  // Filter by license (if specified)
  if (currentFilter.license !== 'all') {
    filteredModels = filteredModels.filter((model) => {
      // License filtering logic
      // This is placeholder - adjust based on actual license data
      const license = currentFilter.license
      // TODO: Implement actual license filtering when license data available
      return true // For now, include all
    })
  }

  // Sort (client-side since API doesn't support it)
  if (currentFilter.sort === 'likes') {
    filteredModels.sort((a, b) => (b.likes || 0) - (a.likes || 0))
  } else if (currentFilter.sort === 'recent') {
    // Sort by lastModified if available, otherwise downloads
    filteredModels.sort((a, b) => {
      const aDate = a.lastModified ? new Date(a.lastModified).getTime() : 0
      const bDate = b.lastModified ? new Date(b.lastModified).getTime() : 0
      return bDate - aDate
    })
  } else {
    // Default: downloads
    filteredModels.sort((a, b) => (b.downloads || 0) - (a.downloads || 0))
  }

  console.log(`[SSG] Showing ${filteredModels.length} HuggingFace models (${filterPath})`)

  const models: ModelTableItem[] = filteredModels.map((model) => {
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
            <span>{models.length.toLocaleString()} models</span>
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

      {/* Table with client-side routing */}
      {models.length > 0 ? (
        <div className="rounded-lg border border-border bg-card p-6">
          <ModelTableWithRouting models={models} />
        </div>
      ) : (
        <div className="text-center py-12">
          <p className="text-muted-foreground text-lg">No models match the selected filters.</p>
        </div>
      )}
    </div>
  )
}
