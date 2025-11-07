// TEAM-460: Civitai models marketplace page
// TEAM-422: Changed to vertical card grid layout for CivitAI's portrait images
// TEAM-422: Added SSG-based filtering with pre-generated pages
// TEAM-461: Using CategoryFilterBar directly (Rule Zero - no wrapper shims)
import { getCompatibleCivitaiModels } from '@rbee/marketplace-node'
import { ModelCardVertical, CategoryFilterBar } from '@rbee/ui/marketplace'
import type { Metadata } from 'next'
import Link from 'next/link'
import { modelIdToSlug } from '@/lib/slugify'
import { PREGENERATED_FILTERS, CIVITAI_FILTER_GROUPS, buildFilterUrl } from './filters'

export const metadata: Metadata = {
  title: 'Civitai Models | rbee Marketplace',
  description: 'Browse compatible Stable Diffusion models from Civitai. Pre-rendered for instant loading and maximum SEO.',
}

export default async function CivitaiModelsPage() {
  // Default filter (All Time, All Types, All Models)
  const currentFilter = PREGENERATED_FILTERS[0].filters
  
  console.log('[SSG] Fetching top 100 compatible Civitai models')
  
  const civitaiModels = await getCompatibleCivitaiModels()
  
  console.log(`[SSG] Showing ${civitaiModels.length} Civitai models`)
  
  // TEAM-422: getCompatibleCivitaiModels() returns Model[], not CivitAIModel[]
  // The data is already normalized and converted
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

  return (
    <div className="container mx-auto px-4 py-12 max-w-7xl">
      {/* Header Section */}
      <div className="mb-8 space-y-4">
        <div className="space-y-2">
          <h1 className="text-4xl md:text-5xl font-bold tracking-tight">
            Civitai Models
          </h1>
          <p className="text-muted-foreground text-lg md:text-xl max-w-3xl">
            Discover and download Stable Diffusion models from Civitai&apos;s community
          </p>
        </div>
        
        {/* Stats */}
        <div className="flex items-center gap-6 text-sm text-muted-foreground">
          <div className="flex items-center gap-2">
            <div className="size-2 rounded-full bg-primary" />
            <span>{models.length.toLocaleString()} compatible models</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="size-2 rounded-full bg-green-500" />
            <span>Checkpoints & LORAs</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="size-2 rounded-full bg-blue-500" />
            <span>Safe for work</span>
          </div>
        </div>
      </div>

      {/* Filter Bar */}
      <CategoryFilterBar
        groups={CIVITAI_FILTER_GROUPS}
        currentFilters={currentFilter}
        buildUrl={(filters) => buildFilterUrl({ ...currentFilter, ...filters })}
      />

      {/* Vertical Card Grid - Portrait aspect ratio for CivitAI images */}
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
