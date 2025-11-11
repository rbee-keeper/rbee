// TEAM-460: HuggingFace models marketplace page (migrated from /models)
// TEAM-464: Hybrid SSG + client-side filtering (Phase 3)
// Server renders with default filter for SEO, then client-side loads manifests

import { listHuggingFaceModels } from '@rbee/marketplace-node'
import type { Metadata } from 'next'
import { Suspense } from 'react'
import { PREGENERATED_HF_FILTERS } from './filters'
import { HFFilterPage } from './HFFilterPage'

export const metadata: Metadata = {
  title: 'HuggingFace LLM Models | rbee Marketplace',
  description: 'Browse compatible language models from HuggingFace. Pre-rendered for instant loading and maximum SEO.',
}

// TEAM-464: SSG with default filter data, then client-side filtering
export default async function HuggingFaceModelsPage() {
  // Default filter (downloads, all sizes, all licenses)
  const currentFilter = PREGENERATED_HF_FILTERS[0].filters

  // Fetch top 100 models for SSG/SEO
  const FETCH_LIMIT = 100

  console.log(`[SSG] Fetching top ${FETCH_LIMIT} HuggingFace models for initial render`)

  const hfModels = await listHuggingFaceModels({ limit: FETCH_LIMIT })

  console.log(`[SSG] Pre-rendering ${hfModels.length} HuggingFace models`)

  // Normalize models for client component
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

  // Pass SSG data to client component (wrapped in Suspense for useSearchParams)
  return (
    <Suspense fallback={<div>Loading...</div>}>
      <HFFilterPage initialModels={models} initialFilter={currentFilter} />
    </Suspense>
  )
}
