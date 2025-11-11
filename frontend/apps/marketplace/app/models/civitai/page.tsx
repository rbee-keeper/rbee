// TEAM-460: Civitai models marketplace page
// TEAM-475: SSR initial load, then client-side filtering (SPA experience)
// TEAM-475: Fetches ALL models once on initial load, client filters them

import { getCompatibleCivitaiModels } from '@rbee/marketplace-node'
import type { Metadata } from 'next'
import { CivitAIFilterPage } from './CivitAIFilterPage'

export const metadata: Metadata = {
  title: 'Civitai Models | rbee Marketplace',
  description:
    'Browse compatible Stable Diffusion models from Civitai. Real-time data, updated every 5 minutes.',
}

// TEAM-XXX: Force dynamic rendering to ensure client-side filtering works correctly
// Without this, Next.js may treat this route as static and cause SSR on filter changes
export const dynamic = 'force-dynamic'

// TEAM-475: SSR - fetch ALL models once on initial load
// Client-side filtering handles filter changes (no re-fetch)
export default async function CivitAIModelsPage() {
  const FETCH_LIMIT = 100

  console.log(`[SSR] Fetching top ${FETCH_LIMIT} compatible Civitai models`)

  // TEAM-476: API defaults to NSFW_LEVEL: XXX (all levels) for client-side filtering
  const civitaiModels = await getCompatibleCivitaiModels({ limit: FETCH_LIMIT })

  console.log(`[SSR] Rendering ${civitaiModels.length} Civitai models`)
  console.log(`[SSR] NSFW breakdown:`, {
    total: civitaiModels.length,
    nsfw: civitaiModels.filter(m => (m as any).nsfw === true).length,
    safe: civitaiModels.filter(m => (m as any).nsfw === false).length,
  })

  // Normalize models for client component
  // TEAM-476: Added 'type' and 'nsfw' fields for proper filtering
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
    type: model.type || 'Unknown', // Model type from CivitAI (Checkpoint, LORA, etc.)
    nsfw: model.nsfw || false, // NSFW flag from CivitAI for content rating filtering
  }))

  // TEAM-475: Pass SSR data to client component
  // Client component handles all filtering without re-fetching
  return <CivitAIFilterPage initialModels={models} />
}
