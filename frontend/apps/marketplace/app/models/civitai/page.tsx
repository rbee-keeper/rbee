// TEAM-460: Civitai models marketplace page
// TEAM-464: Hybrid SSG + client-side filtering (Phase 3)
// Server renders with default filter for SEO, then client-side loads manifests
import { getCompatibleCivitaiModels } from '@rbee/marketplace-node'
import type { Metadata } from 'next'
import { PREGENERATED_FILTERS } from './filters'
import { CivitAIFilterPage } from './CivitAIFilterPage'

export const metadata: Metadata = {
  title: 'Civitai Models | rbee Marketplace',
  description:
    'Browse compatible Stable Diffusion models from Civitai. Pre-rendered for instant loading and maximum SEO.',
}

// TEAM-464: SSG with default filter data, then client-side filtering
export default async function CivitAIModelsPage() {
  // Default filter (Most Downloaded, all types, all periods, PG only)
  const currentFilter = PREGENERATED_FILTERS[0].filters

  // Fetch top 100 compatible models for SSG/SEO
  const FETCH_LIMIT = 100

  console.log(`[SSG] Fetching top ${FETCH_LIMIT} compatible Civitai models for initial render`)

  const civitaiModels = await getCompatibleCivitaiModels({ limit: FETCH_LIMIT })

  console.log(`[SSG] Pre-rendering ${civitaiModels.length} Civitai models`)

  // Normalize models for client component
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

  // Pass SSG data to client component
  return <CivitAIFilterPage initialModels={models} initialFilter={currentFilter} />
}
