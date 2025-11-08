// TEAM-422: Dynamic filtered CivitAI pages (SSG pre-generated)
// TEAM-461: Using ModelsFilterBar directly (Rule Zero - no wrapper shims)
import { getCompatibleCivitaiModels } from '@rbee/marketplace-node'
import { ModelCardVertical } from '@rbee/ui/marketplace'
import { ModelsFilterBar } from '../../ModelsFilterBar'
import type { Metadata } from 'next'
import Link from 'next/link'
import { modelIdToSlug } from '@/lib/slugify'
import { PREGENERATED_FILTERS, CIVITAI_FILTER_GROUPS, CIVITAI_SORT_GROUP, getFilterFromPath, buildFilterParams, buildFilterUrl } from '../filters'

interface PageProps {
  params: Promise<{
    filter: string[]
  }>
}

// Pre-generate static pages for all filter combinations
export async function generateStaticParams() {
  return PREGENERATED_FILTERS
    .filter(f => f.path !== '') // Exclude default (handled by main page)
    .map(f => ({
      filter: f.path.split('/'),
    }))
}

export async function generateMetadata({ params }: PageProps): Promise<Metadata> {
  const { filter } = await params
  const filterPath = filter.join('/')
  const config = getFilterFromPath(filterPath)
  
  let title = 'CivitAI Models'
  const parts: string[] = []
  
  if (config.timePeriod !== 'AllTime') parts.push(config.timePeriod)
  if (config.modelType !== 'All') parts.push(config.modelType)
  if (config.baseModel !== 'All') parts.push(config.baseModel)
  
  if (parts.length > 0) {
    title = `${parts.join(' ')} - ${title}`
  }
  
  return {
    title: `${title} | rbee Marketplace`,
    description: `Browse ${parts.join(' ').toLowerCase()} Stable Diffusion models from CivitAI.`,
  }
}

export default async function FilteredCivitaiPage({ params }: PageProps) {
  const { filter } = await params
  const filterPath = filter.join('/')
  const currentFilter = getFilterFromPath(filterPath)
  
  console.log(`[SSG] Fetching CivitAI models with filter: ${filterPath}`)
  
  // TEAM-461: Use buildFilterParams helper
  const apiParams = buildFilterParams(currentFilter)
  const civitaiModels = await getCompatibleCivitaiModels(apiParams)
  
  console.log(`[SSG] Showing ${civitaiModels.length} Civitai models (${filterPath})`)
  
  const models = civitaiModels.map((model) => ({
    id: model.id,
    name: model.name,
    description: model.description.substring(0, 200),
    author: model.author || 'Unknown',
    downloads: model.downloads,
    likes: model.likes,
    size: model.size,
    tags: model.tags.slice(0, 10),
    imageUrl: model.imageUrl,
  }))

  // Build filter description
  const filterParts: string[] = []
  if (currentFilter.timePeriod !== 'AllTime') filterParts.push(currentFilter.timePeriod)
  if (currentFilter.modelType !== 'All') filterParts.push(currentFilter.modelType)
  if (currentFilter.baseModel !== 'All') filterParts.push(currentFilter.baseModel)
  
  const filterDescription = filterParts.length > 0 
    ? filterParts.join(' · ')
    : 'All Models'

  return (
    <div className="container mx-auto px-4 py-12 max-w-7xl">
      {/* Header Section */}
      <div className="mb-8 space-y-4">
        <div className="space-y-2">
          <h1 className="text-4xl md:text-5xl font-bold tracking-tight">
            CivitAI Models
          </h1>
          <p className="text-muted-foreground text-lg md:text-xl max-w-3xl">
            {filterDescription} · Discover and download Stable Diffusion models
          </p>
        </div>
        
        {/* Stats */}
        <div className="flex items-center gap-6 text-sm text-muted-foreground">
          <div className="flex items-center gap-2">
            <div className="size-2 rounded-full bg-primary" />
            <span>{models.length.toLocaleString()} models</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="size-2 rounded-full bg-blue-500" />
            <span>Safe for work</span>
          </div>
        </div>
      </div>

      {/* Filter Bar */}
      <ModelsFilterBar
        groups={CIVITAI_FILTER_GROUPS}
        sortGroup={CIVITAI_SORT_GROUP}
        currentFilters={currentFilter}
        buildUrlFn="/models/civitai"
      />

      {/* Vertical Card Grid */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-4">
        {models.map((model) => (
          <Link 
            key={model.id} 
            href={`/models/civitai/${modelIdToSlug(model.id)}`}
            className="block"
          >
            <ModelCardVertical model={model} />
          </Link>
        ))}
      </div>
    </div>
  )
}
