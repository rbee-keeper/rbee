// TEAM-461: Dynamic filtered HuggingFace pages (SSG pre-generated)
import { listHuggingFaceModels } from '@rbee/marketplace-node'
import type { ModelTableItem } from '@rbee/ui/marketplace'
import { CategoryFilterBar } from '@rbee/ui/marketplace'
import { ModelTableWithRouting } from '@/components/ModelTableWithRouting'
import type { Metadata } from 'next'
import { 
  HUGGINGFACE_FILTER_GROUPS,
  HUGGINGFACE_SORT_GROUP,
  PREGENERATED_HF_FILTERS, 
  getHFFilterFromPath,
  buildHFFilterUrl,
  buildHFFilterDescription,
  buildHFFilterParams
} from '../filters'

interface PageProps {
  params: Promise<{
    filter: string[]
  }>
}

// Pre-generate static pages for all filter combinations
export async function generateStaticParams() {
  return PREGENERATED_HF_FILTERS
    .filter(f => f.path !== '') // Exclude default (handled by main page)
    .map(f => ({
      filter: f.path.split('/').filter(Boolean), // Remove empty strings
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
  
  // Build API parameters from filter
  const apiParams = buildHFFilterParams(currentFilter)
  const hfModels = await listHuggingFaceModels(apiParams)
  
  console.log(`[SSG] Showing ${hfModels.length} HuggingFace models (${filterPath})`)
  
  const models: ModelTableItem[] = hfModels.map((model) => {
    const m = model as unknown as Record<string, unknown>
    return {
      id: m.id as string,
      name: (m.name as string) || (m.id as string),
      description: (m.description as string) || `${(m.author as string) || 'Community'} model`,
      author: m.author as string | undefined,
      downloads: (m.downloads as number) ?? 0,
      likes: (m.likes as number) ?? 0,
      tags: (m.tags as string[] | undefined)?.slice(0, 10) ?? []
    }
  })

  return (
    <div className="container mx-auto px-4 py-12 max-w-7xl">
      {/* Header Section */}
      <div className="mb-8 space-y-4">
        <div className="space-y-2">
          <h1 className="text-4xl md:text-5xl font-bold tracking-tight">
            HuggingFace LLM Models
          </h1>
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
      <CategoryFilterBar
        groups={HUGGINGFACE_FILTER_GROUPS}
        sortGroup={HUGGINGFACE_SORT_GROUP}
        currentFilters={currentFilter}
        buildUrl={(filters) => buildHFFilterUrl({ ...currentFilter, ...filters })}
      />

      {/* Table with client-side routing */}
      {models.length > 0 ? (
        <div className="rounded-lg border border-border bg-card p-6">
          <ModelTableWithRouting models={models} />
        </div>
      ) : (
        <div className="text-center py-12">
          <p className="text-muted-foreground text-lg">
            No models match the selected filters.
          </p>
        </div>
      )}
    </div>
  )
}
